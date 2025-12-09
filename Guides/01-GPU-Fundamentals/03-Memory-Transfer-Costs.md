# 1.3 Memory and Transfer Costs

> **Understanding unified memory and how to minimize data movement overhead.**

---

## The Concept

Traditional GPUs have separate memory from the CPU, requiring explicit copies:

```
Traditional GPU (Discrete):
┌─────────────────┐         PCIe Bus          ┌─────────────────┐
│      CPU        │ ←───────────────────────→ │      GPU        │
│                 │    ~16 GB/s (PCIe 4.0)    │                 │
│  [System RAM]   │                           │  [VRAM]         │
│    64 GB        │                           │    8-24 GB      │
└─────────────────┘                           └─────────────────┘

Data must be COPIED across the PCIe bus!
```

Apple Silicon uses **unified memory**—CPU and GPU share the same physical RAM:

```
Apple Silicon (Unified Memory):
┌─────────────────────────────────────────────────────────────────────┐
│                      Unified Memory (8-192 GB)                       │
│                                                                      │
│    ┌──────────────────┐              ┌──────────────────┐           │
│    │       CPU        │              │       GPU        │           │
│    │                  │              │                  │           │
│    │   Can access     │◄────────────►│   Can access     │           │
│    │   same memory    │   ~200 GB/s  │   same memory    │           │
│    └──────────────────┘              └──────────────────┘           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

No copy needed! Just pointer sharing (with caveats).
```

---

## Why It Matters

Transfer costs can dominate GPU compute time if you're not careful:

```
Scenario: Search 100 queries × 1M database × 768D

Data sizes:
  Queries:   100 × 768 × 4 bytes = 307 KB
  Database:  1M × 768 × 4 bytes = 3.07 GB
  Results:   100 × 1M × 4 bytes = 400 MB

On discrete GPU (PCIe transfer):
  Upload database: 3.07 GB ÷ 16 GB/s = 192 ms
  Upload queries:  0.3 MB ÷ 16 GB/s = 0.02 ms
  GPU compute:     ~50 ms
  Download results: 400 MB ÷ 16 GB/s = 25 ms
  ─────────────────────────────────────────
  Total: 267 ms (81% transfer, 19% compute!)

On Apple Silicon (unified memory):
  No transfer needed for persistent data!
  GPU compute:     ~50 ms
  ─────────────────────────────────────────
  Total: 50 ms (100% compute)
```

---

## The Technique: Buffer Management

### Buffer Storage Modes

Metal buffers have different storage modes:

```swift
// 📍 See: Sources/VectorAccelerate/Core/MetalBufferFactory.swift

// Shared: CPU and GPU can both access (Apple Silicon optimal)
let sharedBuffer = device.makeBuffer(
    length: size,
    options: .storageModeShared
)

// Private: GPU-only, fastest for GPU-exclusive data
let privateBuffer = device.makeBuffer(
    length: size,
    options: .storageModePrivate
)

// Managed: Requires explicit synchronization (macOS only)
let managedBuffer = device.makeBuffer(
    length: size,
    options: .storageModeManaged
)
```

### Storage Mode Decision Tree

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STORAGE MODE SELECTION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Does CPU need to read the buffer contents?                         │
│       │                                                              │
│       ├── YES → Does GPU need to write it?                          │
│       │           │                                                  │
│       │           ├── YES → Use .storageModeShared                  │
│       │           │         (Results, intermediate data)             │
│       │           │                                                  │
│       │           └── NO → Use .storageModeShared                   │
│       │                    (Input data from CPU)                     │
│       │                                                              │
│       └── NO → Is data written once, read many times?               │
│                 │                                                    │
│                 ├── YES → Consider .storageModePrivate              │
│                 │         (Index data, constant embeddings)          │
│                 │         Use blit to upload                         │
│                 │                                                    │
│                 └── NO → Use .storageModeShared                     │
│                          (Temporary/scratch buffers)                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Unified Memory: Zero-Copy Pattern

```swift
// 📍 See: Sources/VectorAccelerate/Core/MetalBufferFactory.swift

/// Create buffer from existing array (zero-copy when possible)
func makeBuffer<T>(from array: [T], label: String? = nil) throws -> any MTLBuffer {
    let size = array.count * MemoryLayout<T>.stride

    // On Apple Silicon with unified memory:
    // - .storageModeShared means CPU and GPU see same physical memory
    // - No actual copy if data is already aligned

    guard let buffer = device.makeBuffer(
        bytes: array,
        length: size,
        options: .storageModeShared
    ) else {
        throw VectorError.bufferAllocationFailed(size: size)
    }

    buffer.label = label
    return buffer
}
```

---

## Buffer Pools: Avoiding Allocation Overhead

Creating buffers is expensive. VectorAccelerate uses buffer pooling:

```swift
// 📍 See: Sources/VectorAccelerate/Core/BufferPool.swift

/// Buffer pool for efficient reuse
public actor BufferPool {
    private var buckets: [Int: [BufferToken]] = [:]
    private let device: MetalDevice

    /// Get a buffer of at least the specified size
    public func getBuffer(size: Int) async throws -> BufferToken {
        // Round up to nearest bucket size (power of 2)
        let bucketSize = nextPowerOf2(size)

        // Try to reuse existing buffer
        if var bucket = buckets[bucketSize], !bucket.isEmpty {
            let token = bucket.removeLast()
            buckets[bucketSize] = bucket
            return token
        }

        // Allocate new buffer
        return try await allocateBuffer(size: bucketSize)
    }

    /// Return buffer to pool for reuse
    public func returnBuffer(_ token: BufferToken) {
        let bucketSize = nextPowerOf2(token.buffer.length)
        buckets[bucketSize, default: []].append(token)
    }
}
```

### Using the Buffer Pool

```swift
// ✅ Good: Reuse buffers from pool
let token = try await context.getBuffer(size: outputSize)
defer { await context.bufferPool.returnBuffer(token) }

// ❌ Bad: Allocate new buffer each time
let buffer = device.makeBuffer(length: outputSize, options: .storageModeShared)
```

---

## Residency Management

Metal 4 introduces explicit residency management:

```swift
// 📍 See: Sources/VectorAccelerate/Core/ResidencyManager.swift

/// Residency manager for explicit GPU memory tracking
public actor ResidencyManager {
    private var residencySet: MTLResidencySet?

    public enum ResidencyMode {
        case persistent  // Keep in GPU memory
        case ephemeral   // Can be evicted when not in use
    }

    /// Register a buffer for GPU access
    public func registerBuffer(_ buffer: any MTLBuffer, mode: ResidencyMode) async throws {
        guard let residencySet else { return }

        switch mode {
        case .persistent:
            residencySet.addAllocation(buffer)
        case .ephemeral:
            // Tracked but can be evicted
            residencySet.addAllocation(buffer)
        }
    }

    /// Commit residency changes before command execution
    public func commit() async throws {
        guard let residencySet else { return }
        residencySet.commit()
    }
}
```

### Residency Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RESIDENCY STRATEGY                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Index data (vectors):                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Mode: PERSISTENT                                           │    │
│  │  Reason: Accessed every search, must be resident            │    │
│  │  Lifecycle: Entire application lifetime                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Query buffers:                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Mode: EPHEMERAL                                            │    │
│  │  Reason: Used briefly per search batch                      │    │
│  │  Lifecycle: Single search operation                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Result buffers:                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Mode: EPHEMERAL                                            │    │
│  │  Reason: Written once, read by CPU, then discarded          │    │
│  │  Lifecycle: Single search operation                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Batching Strategy

Even with unified memory, there's overhead per GPU dispatch. Batching amortizes this:

```
Single query overhead:
  - Command buffer creation: ~10 μs
  - Pipeline state binding: ~5 μs
  - Dispatch encoding: ~5 μs
  - GPU execution: ~50 μs
  - Command buffer completion: ~10 μs
  ────────────────────────────────────
  Total: ~80 μs per query

Batch of 100 queries:
  - Same overhead: ~30 μs (fixed cost)
  - GPU execution: ~500 μs (10× more work, ~10× time)
  ────────────────────────────────────
  Total: ~530 μs for 100 queries = 5.3 μs per query

Batching gives 15× better per-query latency!
```

### Implementing Batching

```swift
// 📍 See: Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift:664-700

/// Batch search for multiple queries
public func search(
    queries: [[Float]],
    k: Int
) async throws -> [[IndexSearchResult]] {
    // Single GPU dispatch for all queries
    let parameters = FusedL2TopKParameters(
        numQueries: queries.count,        // All queries at once
        numDataset: storage.allocatedSlots,
        dimension: configuration.dimension,
        k: k
    )

    let gpuResult = try await kernel.execute(
        queries: queryBuffer,
        dataset: datasetBuffer,
        parameters: parameters
    )

    // Unpack results for each query
    return (0..<queries.count).map { queryIdx in
        gpuResult.results(for: queryIdx)
    }
}
```

---

## Memory Coalescing

How threads access memory matters enormously:

```metal
// ✅ GOOD: Coalesced access (adjacent threads read adjacent memory)
kernel void coalesced(device float* data, uint tid [[thread_position_in_grid]]) {
    float value = data[tid];  // Thread 0→data[0], Thread 1→data[1], ...
    // GPU can batch these into a single memory transaction
}

// ❌ BAD: Strided access (threads read far-apart memory)
kernel void strided(device float* data, uint tid [[thread_position_in_grid]]) {
    float value = data[tid * 1024];  // Thread 0→data[0], Thread 1→data[1024], ...
    // Each thread requires separate memory transaction
}
```

### Distance Kernel Memory Pattern

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal:50-52

// Each thread reads one query and one database vector
device const float* query = queryVectors + (queryIdx * params.strideQuery);
device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

// Within a SIMD group:
// - Thread 0 reads query[0], database[0]
// - Thread 1 reads query[0], database[1]  (different DB vector, same query)
// - Thread 2 reads query[0], database[2]
// ...

// Database reads are coalesced (adjacent threads, adjacent memory)
// Query is broadcast (same query for multiple threads)
```

---

## 🔗 VectorCore Connection

VectorCore taught you about memory alignment for SIMD:

```swift
// VectorCore: 16-byte alignment for SIMD4
let aligned = UnsafeMutableRawPointer.allocate(
    byteCount: size,
    alignment: 16
)
```

GPU buffers have similar but stricter requirements:

```swift
// VectorAccelerate: Buffer alignment
// Metal buffers are typically page-aligned (4KB or 16KB)
// float4 vectors should be 16-byte aligned for optimal access

// 📍 See: Sources/VectorAccelerate/Index/Internal/GPUVectorStorage.swift
let bytesPerVector = dimension * MemoryLayout<Float>.stride
let alignedBytes = (bytesPerVector + 15) & ~15  // Round up to 16
```

---

## 🔗 VectorIndex Connection

VectorIndex's memory-mapped indices translate well to GPU:

```swift
// VectorIndex: Memory-mapped file for large index
let mappedData = try Data(contentsOf: indexURL, options: .mappedIfSafe)

// VectorAccelerate: GPU buffer from mapped data
// On Apple Silicon, can create GPU buffer from mapped memory!
let buffer = device.makeBuffer(
    bytesNoCopy: mappedData.baseAddress!,
    length: mappedData.count,
    options: .storageModeShared,
    deallocator: nil
)
```

---

## In VectorAccelerate

The Metal4Context handles memory management:

📍 See: `Sources/VectorAccelerate/Core/Metal4Context.swift:373-398`

```swift
// MARK: - Buffer Management

/// Get a buffer from the pool with automatic residency registration
public func getBuffer(size: Int) async throws -> BufferToken {
    let token = try await bufferPool.getBuffer(size: size)

    // Register with residency manager
    try await residencyManager.registerBuffer(token.buffer, mode: .ephemeral)

    return token
}

/// Get a buffer for typed data with automatic residency
public func getBuffer<T: Sendable>(for data: [T]) async throws -> BufferToken {
    let token = try await bufferPool.getBuffer(for: data)

    // Register with residency manager
    try await residencyManager.registerBuffer(token.buffer, mode: .ephemeral)

    return token
}

/// Commit any pending residency changes
public func commitResidency() async throws {
    try await residencyManager.commit()
}
```

---

## Key Takeaways

1. **Unified memory is a game-changer**: Apple Silicon eliminates most transfer overhead

2. **Storage modes matter**: Use `.storageModeShared` for CPU↔GPU data, `.storageModePrivate` for GPU-only

3. **Pool your buffers**: Allocation is expensive; reuse buffers when possible

4. **Batch operations**: Amortize per-dispatch overhead across many operations

5. **Coalesce memory access**: Adjacent threads should read adjacent memory locations

6. **Manage residency**: Keep frequently-accessed data resident in GPU memory

---

## Chapter Summary

You now understand:
- ✅ Why GPUs accelerate vector search (massive parallelism)
- ✅ How Metal organizes threads into SIMD groups and threadgroups
- ✅ How unified memory and buffer management work

Next, we'll apply these concepts to the hot path: **distance computation**.

**[→ Chapter 2: Distance Kernels](../02-Distance-Kernels/README.md)**

---

*Guide 1.3 of 1.3 • Chapter 1: GPU Fundamentals*
