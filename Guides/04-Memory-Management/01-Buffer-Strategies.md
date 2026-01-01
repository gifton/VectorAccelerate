# 4.1 Buffer Strategies

> **Choosing the right buffer types and maximizing reuse.**

---

## The Concept

Metal buffers are containers for GPU-accessible data. Choosing the right buffer type and managing their lifecycle is crucial for performance.

```
Buffer Storage Modes:

┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│  .storageModeShared │  │ .storageModePrivate│  │.storageModeManaged │
│                    │  │                    │  │                    │
│  CPU ←──────────→ GPU │  │  GPU only          │  │  CPU sync required │
│                    │  │                    │  │                    │
│  ✓ Read/Write both│  │  ✓ Fastest GPU     │  │  ✓ Discrete GPU    │
│  ✓ No copy needed │  │  ✗ CPU cannot read │  │  ✗ Not Apple Silicon│
│  ✓ Apple Silicon  │  │  ✓ Private cache   │  │                    │
└────────────────────┘  └────────────────────┘  └────────────────────┘
```

---

## Why It Matters

Wrong buffer choices cause:
- **Unnecessary copies**: Moving data between CPU/GPU
- **Cache thrashing**: GPU can't use its private caches effectively
- **Allocation overhead**: Creating buffers is expensive (~10-100μs)

---

## The Technique: Storage Mode Selection

### .storageModeShared (Default for Apple Silicon)

```swift
// 📍 See: Sources/VectorAccelerate/Core/MetalBufferFactory.swift

/// Create a shared buffer - optimal for Apple Silicon
public func makeSharedBuffer(size: Int, label: String? = nil) throws -> any MTLBuffer {
    guard let buffer = device.makeBuffer(
        length: size,
        options: .storageModeShared
    ) else {
        throw VectorError.bufferAllocationFailed(size: size)
    }
    buffer.label = label
    return buffer
}
```

**Use for:**
- Input data (queries, database vectors)
- Output data (distances, results)
- Anything CPU needs to read/write

### .storageModePrivate (GPU-only)

```swift
/// Create a private buffer - fastest for GPU-only data
public func makePrivateBuffer(size: Int, label: String? = nil) throws -> any MTLBuffer {
    guard let buffer = device.makeBuffer(
        length: size,
        options: .storageModePrivate
    ) else {
        throw VectorError.bufferAllocationFailed(size: size)
    }
    buffer.label = label
    return buffer
}
```

**Use for:**
- Intermediate results (scratch space)
- Data that GPU writes and reads without CPU involvement
- Index structures that stay on GPU

---

## Buffer Pooling

Creating buffers is expensive. VectorAccelerate uses pooling:

```swift
// 📍 See: Sources/VectorAccelerate/Core/BufferPool.swift:155-309

public actor BufferPool {
    private var buckets: [Int: BufferBucket] = [:]
    private let factory: MetalBufferFactory

    // Standard bucket sizes for efficient reuse
    private let bucketSizes: [Int] = MetalBufferFactory.standardBucketSizes

    /// Get a buffer from pool or create new
    public func getBuffer(size: Int) async throws -> BufferToken {
        // Select appropriate bucket size (rounds to standard sizes)
        let bucketSize = MetalBufferFactory.selectBucketSize(for: size)

        // Try reuse from bucket
        if var bucket = buckets[bucketSize], !bucket.available.isEmpty {
            let buffer = bucket.available.removeLast()
            bucket.inUse.insert(ObjectIdentifier(buffer))
            buckets[bucketSize] = bucket
            hitCount += 1
            return BufferToken(buffer: buffer, size: bucketSize, pool: self)
        }

        // Allocate new buffer via factory (synchronous)
        missCount += 1
        guard let buffer = factory.createBuffer(length: bucketSize) else {
            throw VectorError.bufferAllocationFailed(size: bucketSize)
        }
        return BufferToken(buffer: buffer, size: bucketSize, pool: self)
    }

    /// Return buffer to pool (called automatically by BufferToken deinit)
    func returnBuffer(_ buffer: any MTLBuffer, size: Int) {
        guard var bucket = buckets[size] else { return }
        bucket.inUse.remove(ObjectIdentifier(buffer))
        bucket.available.append(buffer)
        buckets[size] = bucket
    }
}
```

### Using BufferToken

```swift
// RAII-style buffer management
func computeDistances() async throws {
    // Get buffer from pool
    let outputToken = try await context.getBuffer(size: outputSize)

    // Use buffer
    try await kernel.execute(output: outputToken.buffer)

    // Return to pool when done (or let token deinitialize)
    await context.bufferPool.returnBuffer(outputToken)
}
```

---

## Power-of-2 Bucketing

Pool uses power-of-2 sizes to maximize reuse:

```
Request: 3000 bytes → Bucket: 4096 bytes
Request: 5000 bytes → Bucket: 8192 bytes
Request: 10000 bytes → Bucket: 16384 bytes

Benefits:
1. High reuse rate (many requests fit same bucket)
2. Simple allocation/deallocation
3. Minimal fragmentation
```

---

## Buffer Lifecycle Management

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BUFFER LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  PERSISTENT BUFFERS (Index Data)                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Created: Once at index initialization                        │   │
│  │  Lifetime: Entire index lifetime                              │   │
│  │  Example: Vector storage buffer, centroid buffer              │   │
│  │  Strategy: Direct allocation, no pooling                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  EPHEMERAL BUFFERS (Per-Query)                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Created: Each search operation                               │   │
│  │  Lifetime: Single search                                      │   │
│  │  Example: Query buffer, distance buffer, result buffer        │   │
│  │  Strategy: Pool-based allocation                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  SCRATCH BUFFERS (Intermediate)                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Created: During kernel execution                             │   │
│  │  Lifetime: Within single command buffer                       │   │
│  │  Example: Threadgroup memory, reduction scratch               │   │
│  │  Strategy: Allocated per-threadgroup in kernel                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 VectorCore Connection

VectorCore uses Swift's memory management:

```swift
// VectorCore: Array-based storage
let vectors: [[Float]]  // Managed by Swift ARC

// VectorAccelerate: GPU buffer storage
let buffer: MTLBuffer   // Managed by Metal
```

The key difference: GPU buffers have explicit storage modes.

---

## 🔗 VectorIndex Connection

VectorIndex stores data in Swift arrays:

```swift
// VectorIndex: FlatIndex storage
public actor FlatIndex<D: StaticDimension> {
    private var vectors: [Vector<D>]  // Swift-managed
}
```

VectorAccelerate stores directly in GPU buffers:

```swift
// VectorAccelerate: GPU storage
public actor AcceleratedVectorIndex {
    private var storage: GPUVectorStorage  // GPU buffer
}
```

---

## In VectorAccelerate

GPU vector storage implementation:

📍 See: `Sources/VectorAccelerate/Index/Internal/GPUVectorStorage.swift`

```swift
/// GPU buffer storage for vectors
internal final class GPUVectorStorage: @unchecked Sendable {
    private(set) var buffer: (any MTLBuffer)?
    private let device: any MTLDevice
    let dimension: Int
    private(set) var capacity: Int
    private(set) var allocatedSlots: Int = 0

    init(device: any MTLDevice, dimension: Int, capacity: Int) throws {
        self.device = device
        self.dimension = dimension
        self.capacity = capacity

        let bufferSize = capacity * dimension * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(
            length: bufferSize,
            options: .storageModeShared
        ) else {
            throw IndexError.bufferError(
                operation: "init",
                reason: "Failed to allocate vector buffer"
            )
        }
        buffer.label = "GPUVectorStorage.vectors"
        self.buffer = buffer
    }

    func writeVector(_ vector: [Float], at slotIndex: Int) throws {
        guard let buffer = buffer else {
            throw IndexError.bufferError(operation: "write", reason: "No buffer")
        }

        let offset = slotIndex * dimension * MemoryLayout<Float>.stride
        let destination = buffer.contents().advanced(by: offset)
        vector.withUnsafeBytes { src in
            destination.copyMemory(from: src.baseAddress!, byteCount: src.count)
        }

        allocatedSlots = max(allocatedSlots, slotIndex + 1)
    }
}
```

---

## Performance Impact

Buffer allocation vs. pooling:

```
1000 search operations:

Without pooling:
  Buffer allocations: 3000 (query, distance, result per search)
  Time in allocation: ~150 ms
  Total time: 650 ms

With pooling:
  Buffer allocations: 3 (initial, then reused)
  Time in allocation: ~0.15 ms
  Total time: 500 ms

23% faster with pooling!
```

---

## Vector Quantization for Memory Savings (v0.3.2+)

For large IVF indexes, vector quantization can dramatically reduce memory usage:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QUANTIZATION MEMORY SAVINGS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  VectorQuantization Options:                                        │
│                                                                      │
│  .none (default)                                                    │
│  └── 4 bytes/element (float32)                                     │
│  └── 768D vector = 3,072 bytes                                     │
│                                                                      │
│  .sq8 (symmetric INT8)                                              │
│  └── 1 byte/element                                                 │
│  └── 768D vector = 768 bytes (4× smaller)                          │
│                                                                      │
│  .sq8Asymmetric (asymmetric INT8)                                   │
│  └── 1 byte/element + zero-point offset                            │
│  └── Better for skewed distributions                                │
│                                                                      │
│  .sq4 (packed INT4)                                                 │
│  └── 0.5 bytes/element                                              │
│  └── 768D vector = 384 bytes (8× smaller)                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Memory Comparison by Scale

| Vectors | Dimension | No Quant | SQ8 | SQ4 |
|---------|-----------|----------|-----|-----|
| 100K | 768 | 307 MB | 77 MB | 38 MB |
| 1M | 768 | 3.1 GB | 0.8 GB | 0.4 GB |
| 10M | 768 | 30.7 GB | 7.7 GB | 3.8 GB |

### Using Quantization

```swift
// IVF index with scalar quantization
let index = try await AcceleratedVectorIndex(
    configuration: .ivf(
        dimension: 768,
        nlist: 256,
        nprobe: 16,
        capacity: 1_000_000,
        quantization: .sq8  // 4× memory savings
    )
)

// Check if quantization is active
if index.configuration.isQuantized {
    let bytesPerVector = index.configuration.bytesPerVector
    print("Using \(bytesPerVector) bytes per vector")
}
```

> ⚠️ **Note**: Quantization is only available for IVF indexes. Flat indexes use full float32 precision. There is a small recall trade-off with quantization (~1-3% for SQ8, ~5-10% for SQ4).

---

## Key Takeaways

1. **Use .storageModeShared on Apple Silicon**: Zero-copy CPU↔GPU

2. **Pool ephemeral buffers**: Avoid allocation overhead

3. **Power-of-2 bucketing**: Maximize pool reuse

4. **Persistent vs. ephemeral**: Different strategies for different lifetimes

5. **Label your buffers**: Helps with debugging in GPU profiler

---

## Next Up

Understanding Apple Silicon's unified memory:

**[→ 4.2 Unified Memory Patterns](./02-Unified-Memory-Patterns.md)**

---

*Guide 4.1 of 4.3 • Chapter 4: Memory Management*
