# 2.3 Batch Distance Matrix

> **Computing all Q×N distances in a single GPU dispatch.**

---

## The Concept

When searching, we need distances from each query to every database vector:

```
Queries (Q × D):        Database (N × D):       Distances (Q × N):
┌───────────────┐       ┌───────────────┐       ┌─────────────────────┐
│ q₀: [........]│       │ d₀: [........]│       │ dist(q₀,d₀) ... (q₀,dₙ)│
│ q₁: [........]│   ×   │ d₁: [........]│   =   │ dist(q₁,d₀) ... (q₁,dₙ)│
│ q₂: [........]│       │ ...           │       │ ...                    │
│ ...           │       │ dₙ: [........]│       │ dist(qₘ,d₀) ... (qₘ,dₙ)│
└───────────────┘       └───────────────┘       └─────────────────────┘
    Q vectors              N vectors              Q × N distances
```

This is a **batch operation**—and batch operations are where GPUs shine.

---

## Why It Matters

The alternative is computing distances one at a time:

```swift
// Naive: Q × N separate operations
for query in queries {
    for dbVector in database {
        distances.append(l2Distance(query, dbVector))
    }
}
// Q × N GPU dispatches = massive overhead!

// Better: Single dispatch computes all Q × N
let allDistances = try await l2Kernel.compute(
    queries: queries,
    database: database
)
// 1 GPU dispatch = minimal overhead
```

With Q=100 and N=10,000, you'd have 1,000,000 separate dispatches vs. 1 dispatch.

---

## The Technique: 2D Grid Dispatch

Map the Q×N distance matrix to a 2D grid of threads:

```
Distance Matrix [Q × N]:
        ┌───────────────────────────────────────┐
        │ d₀    d₁    d₂    d₃    ...    dₙ    │
        │  ↓     ↓     ↓     ↓           ↓     │
     q₀ │ t₀,₀  t₀,₁  t₀,₂  t₀,₃  ...  t₀,ₙ  │
     q₁ │ t₁,₀  t₁,₁  t₁,₂  t₁,₃  ...  t₁,ₙ  │
     q₂ │ t₂,₀  t₂,₁  t₂,₂  t₂,₃  ...  t₂,ₙ  │
    ... │  ...   ...   ...   ...  ...   ...   │
     qₘ │ tₘ,₀  tₘ,₁  tₘ,₂  tₘ,₃  ...  tₘ,ₙ  │
        └───────────────────────────────────────┘

Each thread t[i,j] computes distance(query[i], database[j])
```

### Thread Position Mapping

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal:34-41

kernel void l2_distance_kernel(
    // ... buffers ...
    uint3 tid [[thread_position_in_grid]]  // 2D position
) {
    const uint queryIdx = tid.x;   // Which query (0 to Q-1)
    const uint dbIdx = tid.y;      // Which database vector (0 to N-1)

    // Bounds check
    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    // This thread computes ONE distance value
    // ...
}
```

### Dispatch Configuration

```swift
// 📍 See: Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift:270-287

/// Thread configuration for distance kernels
public static func forDistanceKernel(
    numQueries: Int,
    numDatabase: Int,
    pipeline: any MTLComputePipelineState
) -> Metal4ThreadConfiguration {
    let maxThreads = pipeline.maxTotalThreadsPerThreadgroup

    // 2D threadgroup: balance between query and database dimensions
    let tgWidth = min(16, numQueries)
    let tgHeight = min(maxThreads / tgWidth, numDatabase)

    // Number of threadgroups needed
    let numTgX = (numQueries + tgWidth - 1) / tgWidth
    let numTgY = (numDatabase + tgHeight - 1) / tgHeight

    return Metal4ThreadConfiguration(
        threadgroups: MTLSize(width: numTgX, height: numTgY, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tgWidth, height: tgHeight, depth: 1)
    )
}

// Example: 100 queries × 10,000 database
// tgWidth = 16, tgHeight = 16 (256 threads per group)
// numTgX = 7, numTgY = 625
// Total: 7 × 625 = 4,375 threadgroups
// Total threads: 4,375 × 256 = 1,120,000 (covers 100 × 10,000 = 1,000,000)
```

---

## Memory Layout: Row-Major Storage

Both input and output use row-major (C-style) layout:

```
Queries [Q × D]:
Memory: [q₀[0], q₀[1], ..., q₀[D-1], q₁[0], q₁[1], ..., qQ-1[D-1]]
        └─────── query 0 ──────┘    └─ query 1 ─┘    └─ query Q-1 ─┘

Database [N × D]:
Memory: [d₀[0], d₀[1], ..., d₀[D-1], d₁[0], d₁[1], ..., dN-1[D-1]]

Distances [Q × N]:
Memory: [dist(q₀,d₀), dist(q₀,d₁), ..., dist(q₀,dN-1), dist(q₁,d₀), ...]
        └────────────── row 0 ──────────────────┘    └── row 1 ──...
```

### Addressing in Kernel

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/L2Distance.metal:50-52, 89-90

// Input addressing (with strides for flexibility)
device const float* query = queryVectors + (queryIdx * params.strideQuery);
device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

// Output addressing
const uint outputIdx = queryIdx * params.strideOutput + dbIdx;
distances[outputIdx] = distance;
```

---

## Memory Access Patterns

### Coalesced Database Reads

Adjacent threads (in tid.y) read adjacent database vectors:

```
SIMD Group (32 threads), same queryIdx, different dbIdx:
  Thread 0: reads database[0 × D : 1 × D]
  Thread 1: reads database[1 × D : 2 × D]
  Thread 2: reads database[2 × D : 3 × D]
  ...

For float4 reads within each thread:
  All threads read database4[0], then database4[1], etc.
  Memory transactions are coalesced!
```

### Query Broadcast

Threads with the same queryIdx read the same query:

```
SIMD Group processing query 5:
  Thread 0: reads query[5]
  Thread 1: reads query[5]  (same!)
  Thread 2: reads query[5]  (same!)
  ...

GPU cache handles this efficiently - query stays in L1/L2
```

---

## Strided Access for Non-Contiguous Data

Sometimes vectors aren't stored contiguously (e.g., part of a larger struct):

```swift
// Data layout: each vector embedded in a larger struct
struct VectorRecord {
    var id: UInt64          // 8 bytes
    var timestamp: Double   // 8 bytes
    var vector: [Float]     // D × 4 bytes
    var metadata: [UInt8]   // variable
}
// Stride between vectors > D!

// Use strided parameters
let parameters = L2DistanceParameters(
    numQueries: numQueries,
    numDatabase: numDatabase,
    dimension: dimension,
    strideQuery: queryStride,       // Distance between query starts
    strideDatabase: databaseStride, // Distance between database starts
    strideOutput: numDatabase,      // Output is contiguous
    computeSqrt: false
)
```

---

## Batch Processing for Very Large Matrices

When Q×N is huge, we might need to chunk:

```swift
// 📍 See: Sources/VectorAccelerate/Operations/BatchDistanceOperations.swift

/// Compute distances in chunks for memory efficiency
public func computeDistancesChunked(
    queries: [[Float]],
    database: [[Float]],
    chunkSize: Int = 10_000
) async throws -> [[Float]] {
    var allDistances: [[Float]] = Array(
        repeating: [],
        count: queries.count
    )

    // Process database in chunks
    for chunkStart in stride(from: 0, to: database.count, by: chunkSize) {
        let chunkEnd = min(chunkStart + chunkSize, database.count)
        let chunk = Array(database[chunkStart..<chunkEnd])

        // Compute distances for this chunk
        let chunkDistances = try await kernel.compute(
            queries: queries,
            database: chunk
        )

        // Append to results
        for (i, row) in chunkDistances.enumerated() {
            allDistances[i].append(contentsOf: row)
        }
    }

    return allDistances
}
```

### When to Chunk

```
GPU memory considerations:

Output buffer size: Q × N × 4 bytes
  100 × 1M × 4 = 400 MB    ← Usually fine
  1K × 10M × 4 = 40 GB     ← Too big!

Chunk when:
  - Output buffer exceeds ~25% of GPU memory
  - You're memory constrained
  - You want to overlap compute with CPU processing
```

---

## 🔗 VectorCore Connection

VectorCore's batch operations are conceptually similar:

```swift
// VectorCore: Batch distance computation on CPU
public func batchL2Distances(
    queries: [[Float]],
    database: [[Float]]
) -> [[Float]] {
    queries.map { query in
        database.map { dbVec in
            l2DistanceSquared(query, dbVec)
        }
    }
}
```

The GPU version does the same work but with thousands of threads computing in parallel.

---

## 🔗 VectorIndex Connection

VectorIndex's FlatIndex computes the full distance matrix for brute-force search:

```swift
// VectorIndex: Flat search computes all distances
public func search(query: [Float], k: Int) -> [SearchResult] {
    // Conceptually: distance matrix [1 × N]
    let distances = database.map { l2Distance(query, $0) }

    return distances
        .enumerated()
        .sorted { $0.1 < $1.1 }
        .prefix(k)
        .map { SearchResult(index: $0.0, distance: $0.1) }
}
```

VectorAccelerate batches multiple queries for efficiency:

```swift
// VectorAccelerate: Batch flat search
public func search(queries: [[Float]], k: Int) async throws -> [[SearchResult]] {
    // Distance matrix [Q × N]
    let distances = try await l2Kernel.compute(
        queries: queries,
        database: database
    )

    // Top-K for each query
    return try await topKKernel.select(distances: distances, k: k)
}
```

---

## In VectorAccelerate

The batch distance API in L2DistanceKernel:

📍 See: `Sources/VectorAccelerate/Kernels/Metal4/L2DistanceKernel.swift:334-405`

```swift
/// Compute L2 distances from Float arrays.
public func compute(
    queries: [[Float]],
    database: [[Float]],
    computeSqrt: Bool = true
) async throws -> [[Float]] {
    guard !queries.isEmpty, !database.isEmpty else {
        throw VectorError.invalidInput("Empty input vectors")
    }

    let dimension = queries[0].count
    let numQueries = queries.count
    let numDatabase = database.count

    // Flatten arrays for GPU
    let flatQueries = queries.flatMap { $0 }
    let flatDatabase = database.flatMap { $0 }

    // Create buffers
    let queryBuffer = try context.bufferFactory.makeBuffer(
        from: flatQueries
    )
    let databaseBuffer = try context.bufferFactory.makeBuffer(
        from: flatDatabase
    )

    // Execute
    let parameters = L2DistanceParameters(
        numQueries: numQueries,
        numDatabase: numDatabase,
        dimension: dimension,
        computeSqrt: computeSqrt
    )

    let distanceBuffer = try await execute(
        queries: queryBuffer,
        database: databaseBuffer,
        parameters: parameters
    )

    // Extract results
    return extractResults(
        from: distanceBuffer,
        numQueries: numQueries,
        numDatabase: numDatabase
    )
}
```

---

## Performance Characteristics

### Scaling Analysis

```
Batch size vs. throughput:

Single query (Q=1):
  - GPU launch overhead dominates
  - ~50μs compute + ~100μs overhead = low efficiency

Small batch (Q=10):
  - Overhead amortized across queries
  - Much better throughput

Large batch (Q=100+):
  - GPU fully utilized
  - Maximum throughput achieved

Rule of thumb: Batch at least 10-100 queries for good GPU utilization
```

### Memory Bandwidth Analysis

```
100 queries × 100K database × 768D:

Input data:
  Queries: 100 × 768 × 4 = 307 KB
  Database: 100K × 768 × 4 = 307 MB
  Total input: ~307 MB

Output data:
  Distances: 100 × 100K × 4 = 40 MB

Total memory traffic: ~350 MB

At 400 GB/s bandwidth: ~1 ms just for memory
Actual kernel time: ~4-5 ms (memory bound!)
```

---

## Key Takeaways

1. **2D dispatch maps naturally**: Query index → tid.x, Database index → tid.y

2. **Batch for efficiency**: Amortize launch overhead across many queries

3. **Memory layout matters**: Row-major storage enables coalesced access

4. **Strides provide flexibility**: Support non-contiguous data layouts

5. **Chunk for very large matrices**: Avoid GPU memory exhaustion

---

## Next Up

Let's dive deeper into the dimension-specific optimizations:

**[→ 2.4 Dimension-Optimized Kernels](./04-Dimension-Optimized-Kernels.md)**

---

*Guide 2.3 of 2.4 • Chapter 2: Distance Kernels*
