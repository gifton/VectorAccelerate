# 6.1 AcceleratedVectorIndex Deep Dive

> **The complete picture—how VectorAccelerate's flagship API brings everything together.**

---

## The Concept

AcceleratedVectorIndex is VectorAccelerate's **GPU-first vector index**. It combines:

- Direct GPU buffer ownership (vectors live on GPU)
- Opaque handle-based identification
- Lazy deletion with compaction
- Automatic CPU/GPU routing
- Support for Flat and IVF index types

```swift
// 📍 See: Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift

public actor AcceleratedVectorIndex {
    // Configuration
    public let configuration: IndexConfiguration

    // GPU infrastructure
    private let context: Metal4Context

    // Storage
    private var storage: GPUVectorStorage
    private var handleAllocator: HandleAllocator
    private var deletionMask: DeletionMask
    private var metadataStore: MetadataStore

    // Kernels
    private var fusedL2TopKKernel: FusedL2TopKKernel?

    // IVF (optional)
    private var ivfStructure: IVFStructure?
    private var ivfSearchPipeline: IVFSearchPipeline?
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                  AcceleratedVectorIndex Architecture                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PUBLIC API                                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  insert()  search()  remove()  compact()  statistics()      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│  ┌────────────────────────┼────────────────────────────────────┐    │
│  │ CORE COMPONENTS        │                                     │    │
│  │                        ▼                                     │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │HandleAllocator│  │DeletionMask │  │MetadataStore │       │    │
│  │  │              │  │              │  │              │       │    │
│  │  │ Index↔Handle │  │ Valid/Deleted│  │ Key-Value    │       │    │
│  │  │ Generation   │  │ Tracking     │  │ Per Vector   │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  │                                                              │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │               GPUVectorStorage                        │   │    │
│  │  │                                                       │   │    │
│  │  │  MTLBuffer: [v₀][v₁][v₂]...[vₙ]  (D floats each)     │   │    │
│  │  │  Unified memory, direct GPU access                    │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│  ┌────────────────────────┼────────────────────────────────────┐    │
│  │ GPU KERNELS            │                                     │    │
│  │                        ▼                                     │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │FusedL2TopK   │  │IVFSearch     │  │Compaction    │       │    │
│  │  │Kernel        │  │Pipeline      │  │Kernels       │       │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│  ┌────────────────────────┼────────────────────────────────────┐    │
│  │ METAL INFRASTRUCTURE   │                                     │    │
│  │                        ▼                                     │    │
│  │  ┌──────────────────────────────────────────────────────┐   │    │
│  │  │                 Metal4Context                         │   │    │
│  │  │                                                       │   │    │
│  │  │  Device, CommandQueue, BufferPool, ResidencyManager   │   │    │
│  │  └──────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. GPU-First Storage

Vectors are stored directly in GPU buffers, not CPU arrays:

```swift
// 📍 See: Sources/VectorAccelerate/Index/Internal/GPUVectorStorage.swift:148-177

final class GPUVectorStorage {
    private(set) var buffer: (any MTLBuffer)?
    let dimension: Int

    var bytesPerSlot: Int {
        dimension * MemoryLayout<Float>.size
    }

    func writeVector(_ vector: [Float], at slotIndex: Int) throws {
        guard let buffer = buffer else {
            throw IndexError.gpuNotInitialized(operation: "writeVector")
        }
        let offset = slotIndex * bytesPerSlot
        let ptr = buffer.contents().advanced(by: offset)
        _ = vector.withUnsafeBytes { src in
            memcpy(ptr, src.baseAddress!, bytesPerSlot)
        }
    }
}
```

**Why**: Avoids CPU↔GPU transfer on every search. Search reads directly from GPU memory.

### 2. Handle-Based Identification

Users get opaque handles, not raw indices:

```swift
// 📍 See: Sources/VectorAccelerate/Index/Types/VectorHandle.swift

public struct VectorHandle: Hashable, Sendable {
    public let index: UInt32
    public let generation: UInt16
}
```

**Why**: Handles remain valid across compaction. Generation detects stale references.

### 3. Lazy Deletion

Remove marks vectors as deleted; compaction reclaims space later:

```swift
// 📍 See: Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift:530-550

public func remove(_ handle: VectorHandle) throws {
    guard handleAllocator.validate(handle) else {
        throw IndexError.invalidInput(message: "Invalid handle")
    }

    handleAllocator.markDeleted(handle)
    deletionMask.markDeleted(Int(handle.index))
    metadataStore.remove(handle.index)
    ivfStructure?.removeVector(slotIndex: handle.index, generation: handle.generation)
}
```

**Why**: O(1) deletion. Compaction is O(N) but can be deferred.

### 4. Actor Isolation

The index is an actor for thread safety:

```swift
public actor AcceleratedVectorIndex {
    // All methods are actor-isolated
    // Safe concurrent access from multiple tasks
}
```

**Why**: GPU operations must be serialized. Actor model enforces this.

---

## Data Flow: Insert

```
insert([0.1, 0.2, ...], metadata: ["key": "value"])
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Validate dimension                                              │
│     vector.count == configuration.dimension ?                       │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Ensure capacity                                                 │
│     storage.ensureCapacity(allocatedSlots + 1)                     │
│     May grow GPU buffer if needed                                   │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. Allocate handle                                                 │
│     handle = handleAllocator.allocate()                            │
│     Returns: VectorHandle(index: 42, generation: 1)                │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Write to GPU buffer                                             │
│     storage.writeVector(vector, at: handle.index)                  │
│     Direct memcpy to MTLBuffer.contents()                          │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Store metadata (if provided)                                    │
│     metadataStore[handle.index] = metadata                         │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. Update IVF structure (if IVF index)                            │
│     If trained: Assign to cluster                                   │
│     If not trained: Add to staging, maybe auto-train               │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
           Return handle
```

---

## Data Flow: Search

```
search(query: [0.5, 0.6, ...], k: 10)
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Validate query dimension                                        │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. Route to implementation                                         │
│     IVF trained? → searchIVF()                                     │
│     Filter provided? → searchFiltered()                            │
│     Otherwise → searchUnfiltered()                                 │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. Create query buffer                                             │
│     queryBuffer = device.makeBuffer(bytes: query, ...)             │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. Execute fused kernel                                            │
│     fusedL2TopKKernel.execute(                                     │
│         queries: queryBuffer,                                       │
│         dataset: storage.buffer,                                    │
│         parameters: FusedL2TopKParameters(...)                     │
│     )                                                               │
│                                                                      │
│     GPU computes all distances AND selects top-K in one pass       │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. Filter deleted vectors                                          │
│     for (rawIndex, distance) in gpuResults:                        │
│         if deletionMask.isDeleted(rawIndex): continue              │
│         handle = handleAllocator.handle(for: rawIndex)             │
│         results.append(IndexSearchResult(handle, distance))        │
└─────────────────────────────────────────────────────────────────────┘
                │
                ▼
           Return results: [IndexSearchResult]
```

---

## Index Types

### Flat Index

Best for < 100K vectors, 100% recall:

```swift
let index = try await AcceleratedVectorIndex(
    configuration: .flat(dimension: 768, capacity: 100_000)
)
```

Uses `FusedL2TopKKernel` for single-pass search.

### IVF Index

Best for > 100K vectors, configurable recall:

```swift
let index = try await AcceleratedVectorIndex(
    configuration: .ivf(
        dimension: 768,
        nlist: 256,     // Number of clusters
        nprobe: 16,     // Clusters to search
        capacity: 1_000_000
    )
)
```

Uses `IVFSearchPipeline` with centroid search + list scanning.

---

## Handle Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HANDLE LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ALLOCATE                                                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  handle = handleAllocator.allocate()                        │    │
│  │  Returns: VectorHandle(index: 5, generation: 1)             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  USE                                                                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  // Handle is valid                                         │    │
│  │  let vector = try index.vector(for: handle)  // Works!     │    │
│  │  let results = try await index.search(...)   // Finds it   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  DELETE                                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  try index.remove(handle)                                   │    │
│  │  // Handle marked deleted, generation unchanged             │    │
│  │  // Vector still in buffer, but filtered from search        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  COMPACT                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  let mapping = try await index.compact()                    │    │
│  │  // Old handles become stale                                │    │
│  │  // mapping[oldHandle] = newHandle                          │    │
│  │  // New handle has incremented generation                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  STALE HANDLE                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  // Using old handle after compaction                       │    │
│  │  let vector = try index.vector(for: oldHandle)  // nil!    │    │
│  │  // Generation mismatch detected                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete Example

```swift
import VectorAccelerate

// MARK: - Setup

// Create index
let index = try await AcceleratedVectorIndex(
    configuration: .flat(dimension: 768, capacity: 100_000)
)

// MARK: - Insert Vectors

// Single insert
let embedding: [Float] = /* your 768D embedding */
let handle = try await index.insert(embedding, metadata: ["doc_id": "123"])

// Batch insert
let embeddings: [[Float]] = /* batch of embeddings */
let handles = try await index.insert(embeddings)

// MARK: - Search

// Single query
let query: [Float] = /* query embedding */
let results = try await index.search(query: query, k: 10)

for result in results {
    print("Handle: \(result.handle), Distance: \(result.distance)")

    // Get metadata
    if let meta = await index.metadata(for: result.handle) {
        print("  Doc ID: \(meta["doc_id"] ?? "unknown")")
    }

    // Get original vector
    if let vector = try await index.vector(for: result.handle) {
        print("  Vector: \(vector.prefix(5))...")
    }
}

// Batch search (more efficient)
let queries: [[Float]] = /* multiple queries */
let batchResults = try await index.search(queries: queries, k: 10)

// Filtered search
let filteredResults = try await index.search(query: query, k: 10) { handle, metadata in
    metadata?["category"] == "important"
}

// MARK: - Update and Delete

// Update metadata
try await index.setMetadata(["doc_id": "123", "updated": "true"], for: handle)

// Remove vector
try index.remove(handle)

// MARK: - Maintenance

// Check statistics
let stats = await index.statistics()
print("Vectors: \(stats.vectorCount)")
print("Deleted: \(stats.deletedSlots)")
print("GPU Memory: \(stats.gpuVectorMemoryBytes / 1_000_000) MB")

// Compact if fragmented
if stats.shouldCompact {
    let mapping = try await index.compact()
    print("Compacted, updated \(mapping.count) handles")
}

// MARK: - Cleanup

await index.releaseResources()
```

---

## 🔗 VectorCore Connection

AcceleratedVectorIndex builds on VectorCore types:

```swift
// VectorCore types used internally
import VectorCore

// Vector protocol for flexible input
func insert<V: VectorProtocol>(_ vector: V) where V.Scalar == Float

// Distance metrics
let metric: SupportedDistanceMetric = .euclidean
```

---

## 🔗 VectorIndex Connection

AcceleratedVectorIndex mirrors VectorIndex's API patterns:

```swift
// VectorIndex pattern
let cpuIndex = FlatIndex<D768>()
let handle = try await cpuIndex.insert(vector)
let results = try await cpuIndex.search(query: query, k: k)

// VectorAccelerate pattern (same API, GPU backend)
let gpuIndex = try await AcceleratedVectorIndex(...)
let handle = try await gpuIndex.insert(vector)
let results = try await gpuIndex.search(query: query, k: k)
```

Migration is straightforward—same conceptual model, different backend.

---

## Performance Summary

| Operation | 10K Vectors | 100K Vectors | 1M Vectors |
|-----------|-------------|--------------|------------|
| Insert (single) | ~0.1 ms | ~0.1 ms | ~0.1 ms |
| Insert (batch 1K) | ~2 ms | ~2 ms | ~2 ms |
| Search (single query) | ~0.3 ms | ~1.2 ms | ~12 ms |
| Search (100 queries) | ~0.5 ms | ~2 ms | ~15 ms |
| Compact | ~5 ms | ~50 ms | ~500 ms |

*Measured on M2 Max, 768D embeddings, K=10*

---

## Key Takeaways

1. **GPU-first design**: Vectors stay on GPU for zero-transfer search

2. **Handle abstraction**: Stable references across mutations

3. **Lazy deletion**: O(1) remove, deferred compaction

4. **Unified API**: Flat and IVF share the same interface

5. **Actor safety**: Thread-safe by design

---

## Congratulations!

You've completed the VectorAccelerate Learning Guide.

You now understand:
- ✅ GPU fundamentals for vector search
- ✅ Distance and selection kernel implementation
- ✅ Memory management on Apple Silicon
- ✅ Pipeline optimization techniques
- ✅ Production-ready accelerated vector index

**Go build something fast!**

---

## Further Resources

- [VectorAccelerate API Documentation](../../Sources/VectorAccelerate/VectorAccelerate.swift)
- [Metal Programming Guide](https://developer.apple.com/documentation/metal)
- [Apple Silicon GPU Architecture](https://developer.apple.com/documentation/metal/gpu_features/understanding_gpu_family_4)

---

*Guide 6.1 of 6.1 • Chapter 6: Capstone • VectorAccelerate Learning Guide*
