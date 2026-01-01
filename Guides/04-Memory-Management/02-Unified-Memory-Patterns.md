# 4.2 Unified Memory Patterns

> **Leveraging Apple Silicon's unified memory architecture for zero-copy operations.**

---

## The Concept

Apple Silicon has a **unified memory architecture**—CPU and GPU share the same physical RAM:

```
Traditional (Discrete GPU):
┌─────────────┐    PCIe Bus    ┌─────────────┐
│    CPU      │ ←────────────→ │    GPU      │
│             │   16 GB/s      │             │
│ [System RAM]│                │   [VRAM]    │
│   64 GB     │                │   8-24 GB   │
└─────────────┘                └─────────────┘
Data MUST be copied between memories!

Apple Silicon (Unified Memory):
┌─────────────────────────────────────────────┐
│            Unified Memory (8-192 GB)        │
│                                             │
│    ┌─────────┐         ┌─────────┐         │
│    │   CPU   │ ←─────→ │   GPU   │         │
│    └─────────┘  Same   └─────────┘         │
│                memory!                      │
└─────────────────────────────────────────────┘
NO copy needed - just share pointers!
```

---

## Why It Matters

Unified memory transforms GPU programming:

```
Traditional approach (discrete GPU):
  1. Allocate CPU memory, fill with data
  2. Allocate GPU memory
  3. Copy CPU → GPU (expensive!)
  4. GPU compute
  5. Copy GPU → CPU (expensive!)
  6. Read results

Apple Silicon approach:
  1. Allocate shared buffer
  2. Fill with data (CPU can write directly)
  3. GPU compute (reads same memory)
  4. Read results (CPU reads same memory)

No copies = lower latency, less memory used!
```

---

## The Technique: Zero-Copy Patterns

### Pattern 1: Direct Buffer Access

```swift
// 📍 See: Sources/VectorAccelerate/Core/MetalBufferFactory.swift

/// Create buffer from existing data - zero-copy on Apple Silicon
public func makeBuffer<T>(from array: [T], label: String? = nil) throws -> any MTLBuffer {
    let size = array.count * MemoryLayout<T>.stride

    // .storageModeShared = same physical memory for CPU and GPU
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

### Pattern 2: In-Place Modification

```swift
// CPU writes, GPU reads - no copy
func updateVectors(buffer: MTLBuffer, newData: [Float], offset: Int) {
    let destination = buffer.contents().advanced(by: offset)
    newData.withUnsafeBytes { src in
        destination.copyMemory(from: src.baseAddress!, byteCount: src.count)
    }
    // GPU will see updates automatically (with proper synchronization)
}
```

### Pattern 3: Reading GPU Results

```swift
// GPU writes, CPU reads - no copy
func readResults(buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}
```

---

## Synchronization Requirements

Even with unified memory, synchronization is needed:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   SYNCHRONIZATION POINTS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CPU writes → GPU reads:                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  CPU writes complete BEFORE command buffer is committed      │    │
│  │  No explicit sync needed if using executeAndWait()          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  GPU writes → CPU reads:                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Wait for command buffer completion before reading          │    │
│  │  Use: await commandBuffer.waitUntilCompleted()              │    │
│  │  Or: executeAndWait() handles this automatically            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  GPU writes → GPU reads (different dispatch):                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Insert memory barrier between dispatches                   │    │
│  │  encoder.memoryBarrier(scope: .buffers)                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Using Memory Barriers

```swift
// 📍 See: Sources/VectorAccelerate/Core/Metal4Context.swift

try await context.executeAndWait { commandBuffer, encoder in
    // First kernel: compute distances
    distanceKernel.encode(into: encoder, ...)

    // Memory barrier: ensure distances are visible
    encoder.memoryBarrier(scope: .buffers)

    // Second kernel: select top-K from distances
    topKKernel.encode(into: encoder, ...)
}
```

---

## Residency Management

Metal 4 introduces explicit residency management:

```swift
// 📍 See: Sources/VectorAccelerate/Core/ResidencyManager.swift:60-237

/// Residency modes for different buffer lifecycles
public enum ResidencyMode: String, Sendable {
    case `static`    // Long-lived, rarely changes (model weights)
    case ephemeral   // Transient, frequently allocated/freed
    case hot         // Frequently accessed, should stay resident
}

public actor ResidencyManager {
    private let residencySet: any MTLResidencySet

    /// Register buffer for GPU access (batched - call commit() after)
    public func registerBuffer(_ buffer: any MTLBuffer, mode: ResidencyMode = .ephemeral) throws {
        // Adds to pending batch
        residencySet.addAllocation(buffer)
        // Track for statistics and management
    }

    /// Commit pending changes to make resources resident
    public func commit() throws {
        try residencySet.commit()
    }

    /// Evict ephemeral allocations under memory pressure
    public func evictEphemeral() throws -> Int {
        // Returns bytes freed
    }
}
```

### Why Residency Matters

```
Without residency management:
  - GPU may evict buffers under memory pressure
  - Re-loading evicted buffers causes stalls
  - Unpredictable performance

With residency management:
  - Explicitly mark important buffers as resident
  - GPU keeps them in fast memory
  - Predictable, consistent performance
```

---

## Memory Mapping Large Files

For very large indices, memory-map from disk:

```swift
/// Load index from memory-mapped file
func loadLargeIndex(from url: URL) throws -> MTLBuffer {
    // Memory-map the file
    let data = try Data(contentsOf: url, options: .mappedIfSafe)

    // Create buffer from mapped memory
    // On Apple Silicon, this can share the same pages!
    let buffer = device.makeBuffer(
        bytesNoCopy: UnsafeMutableRawPointer(mutating: (data as NSData).bytes),
        length: data.count,
        options: .storageModeShared,
        deallocator: nil  // Data object manages memory
    )

    return buffer!
}
```

---

## 🔗 VectorCore Connection

VectorCore works with Swift arrays in CPU memory:

```swift
// VectorCore: CPU-only
let distances = queries.map { query in
    database.map { db in l2Distance(query, db) }
}
```

VectorAccelerate uses unified memory for seamless CPU↔GPU:

```swift
// VectorAccelerate: Unified memory
let distances = try await kernel.compute(queries: queries, database: database)
// CPU writes queries, GPU computes, CPU reads results - no copies!
```

---

## 🔗 VectorIndex Connection

VectorIndex stores data in CPU memory:

```swift
// VectorIndex: CPU arrays
private var vectors: [[Float]]
```

VectorAccelerate stores in unified memory accessible to both:

```swift
// VectorAccelerate: Unified memory buffer
private var buffer: MTLBuffer  // .storageModeShared
```

---

## In VectorAccelerate

The Metal4Context integrates residency:

📍 See: `Sources/VectorAccelerate/Core/Metal4Context.swift:373-398`

```swift
/// Get a buffer with automatic residency registration
public func getBuffer(size: Int) async throws -> BufferToken {
    let token = try await bufferPool.getBuffer(size: size)

    // Register with residency manager
    try await residencyManager.registerBuffer(token.buffer, mode: .ephemeral)

    return token
}
```

---

## Performance Benefits

Unified memory impact:

```
Transfer 3 GB index data:

Discrete GPU (PCIe 4.0):
  Transfer time: 3 GB ÷ 16 GB/s = 188 ms
  Must happen before EVERY cold start

Apple Silicon (Unified):
  Transfer time: 0 ms (no transfer needed)
  Data is immediately accessible

For frequent index updates:
  Discrete: Each update requires transfer
  Unified: Updates are instant
```

---

## Key Takeaways

1. **Unified memory eliminates copies**: CPU and GPU share physical RAM

2. **Use .storageModeShared**: Enables zero-copy on Apple Silicon

3. **Synchronization still required**: Wait for GPU completion before CPU reads

4. **Residency management**: Keep important buffers in fast memory

5. **Memory mapping**: Large files can share memory with GPU

---

## Next Up

Handling indices larger than available memory:

**[→ 4.3 Streaming Large Indices](./03-Streaming-Large-Indices.md)**

---

*Guide 4.2 of 4.3 • Chapter 4: Memory Management*
