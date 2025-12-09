# 4.3 Streaming Large Indices

> **Handling datasets larger than GPU memory through chunked processing.**

---

## The Concept

Sometimes your index is larger than available memory:

```
Example scenario:
  Index: 100M vectors × 768D × 4 bytes = 307 GB
  System memory: 64 GB

  The entire index doesn't fit!
```

The solution: **streaming**—process the index in chunks, maintaining running results.

---

## Why It Matters

Without streaming, you're limited to indices that fit in memory. With streaming:

- Search indices of any size
- Trade memory for time (more chunks = more passes, but works)
- Enable on-device search for large models

---

## The Technique: Streaming Top-K

The key insight: top-K can be computed incrementally.

```
Streaming Top-K Algorithm:

Chunk 1: [d₀, d₁, ..., d₉₉₉₉]
  → Local top-K: [(idx, dist), ...]

Chunk 2: [d₁₀₀₀₀, d₁₀₀₀₁, ..., d₁₉₉₉₉]
  → Local top-K: [(idx, dist), ...]
  → Merge with running top-K

...continue for all chunks...

Final: Global top-K from all chunks
```

### Metal Implementation

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/AdvancedTopK.metal:387-437

/// Initialize streaming buffers
kernel void streaming_topk_init(
    device float* running_distances [[buffer(0)]],
    device uint* running_indices [[buffer(1)]],
    constant uint& K [[buffer(2)]],
    constant uint& num_queries [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const ulong total_elements = (ulong)num_queries * K;
    if ((ulong)tid >= total_elements) return;

    // Initialize with worst possible values
    running_distances[tid] = INFINITY;
    running_indices[tid] = 0xFFFFFFFF;
}

/// Process a chunk of data
kernel void streaming_topk_process_chunk(
    device const float* chunk_distances [[buffer(0)]],
    device float* running_distances [[buffer(2)]],
    device uint* running_indices [[buffer(3)]],
    constant StreamConfig& config [[buffer(5)]],
    uint q_idx [[thread_position_in_grid]]
) {
    if (q_idx >= config.Q) return;

    const uint K = config.k_value;
    const uint CHUNK_SIZE = config.chunk_size;
    const ulong BASE_INDEX = config.chunk_base_index;

    // Load current best K into private heap
    PrivateMaxHeap heap;
    heap.load(running_distances + q_idx * K,
              running_indices + q_idx * K, K);

    // Process chunk distances
    for (uint i = 0; i < CHUNK_SIZE; ++i) {
        float distance = chunk_distances[q_idx * CHUNK_SIZE + i];
        ulong global_index = BASE_INDEX + i;
        heap.insert(distance, (uint)global_index, K);
    }

    // Store updated heap
    heap.store(running_distances + q_idx * K,
               running_indices + q_idx * K, K);
}
```

---

## Streaming Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STREAMING SEARCH PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: Initialize running top-K buffers                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  running_distances[Q × K] = INFINITY                        │    │
│  │  running_indices[Q × K] = INVALID                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Step 2: Process each chunk                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  for chunk in database.chunks(chunkSize):                   │    │
│  │    chunk_distances = computeDistances(queries, chunk)       │    │
│  │    updateRunningTopK(chunk_distances, running_topK)         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Step 3: Finalize (sort running heap)                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Sort each query's K results by distance                    │    │
│  │  Return sorted top-K per query                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Swift Implementation

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/StreamingTopKKernel.swift

public final class StreamingTopKKernel: @unchecked Sendable {

    /// Stream through dataset in chunks
    public func streamSearch(
        queries: [[Float]],
        datasetIterator: AsyncSequence<[[Float]]>,
        k: Int
    ) async throws -> Metal4TopKResult {

        let numQueries = queries.count

        // Initialize running state
        let runningDistances = try await context.getBuffer(
            size: numQueries * k * MemoryLayout<Float>.size
        )
        let runningIndices = try await context.getBuffer(
            size: numQueries * k * MemoryLayout<UInt32>.size
        )

        // Initialize to infinity
        try await initializePipeline.execute(
            distances: runningDistances.buffer,
            indices: runningIndices.buffer,
            k: k,
            numQueries: numQueries
        )

        // Process chunks
        var globalOffset: UInt64 = 0

        for try await chunk in datasetIterator {
            // Compute distances for this chunk
            let chunkDistances = try await distanceKernel.compute(
                queries: queries,
                database: chunk
            )

            // Update running top-K
            try await updatePipeline.execute(
                chunkDistances: chunkDistances,
                runningDistances: runningDistances.buffer,
                runningIndices: runningIndices.buffer,
                config: StreamConfig(
                    Q: UInt32(numQueries),
                    chunk_size: UInt32(chunk.count),
                    k_value: UInt32(k),
                    chunk_base_index: globalOffset
                )
            )

            globalOffset += UInt64(chunk.count)
        }

        // Finalize: sort the heaps
        try await finalizePipeline.execute(
            distances: runningDistances.buffer,
            indices: runningIndices.buffer,
            k: k,
            numQueries: numQueries
        )

        return Metal4TopKResult(
            indices: runningIndices.buffer,
            distances: runningDistances.buffer,
            k: k,
            numQueries: numQueries
        )
    }
}
```

---

## Chunk Size Selection

Choosing the right chunk size balances:

```
Small chunks (e.g., 10K vectors):
  ✓ Lower memory usage
  ✓ Fits any system
  ✗ More kernel launches (overhead)
  ✗ Less GPU parallelism

Large chunks (e.g., 1M vectors):
  ✓ Better GPU utilization
  ✓ Fewer kernel launches
  ✗ More memory required
  ✗ May not fit on small systems

Recommended: Choose chunk size to use ~25% of available memory
  8 GB system: ~500K vectors of 768D
  16 GB system: ~1M vectors of 768D
  64 GB system: ~4M vectors of 768D
```

---

## Memory-Mapped Streaming

For disk-based indices, combine streaming with memory mapping:

```swift
/// Stream from disk without loading entire index
func streamFromDisk(
    indexPath: URL,
    queries: [[Float]],
    k: Int,
    chunkSize: Int
) async throws -> Metal4TopKResult {

    let fileHandle = try FileHandle(forReadingFrom: indexPath)
    defer { try? fileHandle.close() }

    // Read header to get dimensions
    let header = try readHeader(from: fileHandle)

    // Create async chunk iterator
    let chunkIterator = AsyncStream<[[Float]]> { continuation in
        Task {
            var offset: UInt64 = header.dataOffset

            while offset < header.fileSize {
                // Memory-map just this chunk
                let chunkData = try fileHandle.read(
                    upToCount: chunkSize * header.dimension * 4,
                    from: offset
                )

                let chunk = parseVectors(chunkData, dimension: header.dimension)
                continuation.yield(chunk)

                offset += UInt64(chunkData.count)
            }

            continuation.finish()
        }
    }

    return try await streamingKernel.streamSearch(
        queries: queries,
        datasetIterator: chunkIterator,
        k: k
    )
}
```

---

## 🔗 VectorCore Connection

VectorCore doesn't have streaming—it's designed for in-memory data:

```swift
// VectorCore: Assumes all data in memory
let distances = vectors.map { l2Distance(query, $0) }
```

VectorAccelerate extends this to streaming scenarios.

---

## 🔗 VectorIndex Connection

VectorIndex's IVF provides an alternative to streaming:

```swift
// VectorIndex: IVF avoids full scan
let results = ivfIndex.search(query, k: k, nprobe: 16)
// Only searches 16/256 = 6.25% of data
```

Streaming is for when you need 100% recall from huge datasets.

---

## Performance Characteristics

Streaming 100M vectors × 768D, K=10, 100 queries:

| Chunk Size | Chunks | Time per Chunk | Total Time |
|------------|--------|----------------|------------|
| 100K | 1000 | ~1.2 ms | ~1.2 s |
| 1M | 100 | ~12 ms | ~1.2 s |
| 10M | 10 | ~120 ms | ~1.2 s |

Total time is similar (dominated by compute), but larger chunks have less overhead.

---

## Key Takeaways

1. **Streaming enables any-size indices**: Not limited by memory

2. **Incremental top-K**: Maintain running best K across chunks

3. **Chunk size tradeoff**: Balance memory usage vs. overhead

4. **Memory mapping**: Stream from disk efficiently

5. **Consider IVF first**: If approximate recall is acceptable, IVF may be faster

---

## Chapter Summary

You've learned GPU memory management:

- ✅ Buffer storage modes and pooling
- ✅ Unified memory zero-copy patterns
- ✅ Streaming for large indices

Next, we'll optimize the full pipeline for production.

**[→ Chapter 5: Pipeline Optimization](../05-Pipeline-Optimization/README.md)**

---

*Guide 4.3 of 4.3 • Chapter 4: Memory Management*
