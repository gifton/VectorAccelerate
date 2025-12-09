# 5.2 Async Compute

> **Overlapping CPU and GPU work for maximum throughput.**

---

## The Concept

By default, CPU waits for GPU to complete before doing more work. But they can work in parallel:

```
Sequential (blocking):
CPU: [Prepare]──wait──────────────────────[Process]──wait──────────────
GPU:           [Compute Batch 1]                    [Compute Batch 2]
Timeline: ├────────────────────────────────────────────────────────────┤

Overlapped (async):
CPU: [Prepare B1][Prepare B2][Process B1][Prepare B3][Process B2]...
GPU:             [Compute B1]           [Compute B2]           [Compute B3]
Timeline: ├───────────────────────────────────────────────────────────┤
                              Much faster!
```

---

## Why It Matters

In many workloads, CPU and GPU spend significant time waiting for each other:

```
Search pipeline (sequential):
  CPU: Prepare query (1 ms)
  GPU: Compute search (5 ms)
  CPU: Process results (2 ms)
  ──────────────────────────
  Total: 8 ms per batch

Search pipeline (overlapped):
  While GPU computes batch N:
    CPU prepares batch N+1
    CPU processes batch N-1

  Throughput: 5 ms per batch (GPU-bound)
  37% improvement!
```

---

## The Technique: Double Buffering

Use two sets of buffers, alternating between them:

```swift
/// Double-buffered search for throughput
func streamingSearch(
    queryBatches: [[Float]],
    database: MTLBuffer,
    k: Int
) async throws -> [[IndexSearchResult]] {

    // Double buffers
    var queryBuffers = [
        try await context.getBuffer(size: batchSize * dimension * 4),
        try await context.getBuffer(size: batchSize * dimension * 4)
    ]
    var resultBuffers = [
        try await context.getBuffer(size: batchSize * k * 8),
        try await context.getBuffer(size: batchSize * k * 8)
    ]

    var allResults: [[IndexSearchResult]] = []
    var pendingCompletion: Task<Void, Error>? = nil
    var activeBuffer = 0

    for (i, batch) in queryBatches.enumerated() {
        // Wait for previous GPU work on this buffer
        if let pending = pendingCompletion, i >= 2 {
            try await pending.value
            // Process results from buffer that just completed
            let results = extractResults(from: resultBuffers[activeBuffer])
            allResults.append(contentsOf: results)
        }

        // Prepare current batch (CPU work)
        fillBuffer(queryBuffers[activeBuffer], with: batch)

        // Launch GPU work (async)
        pendingCompletion = Task {
            try await kernel.execute(
                queries: queryBuffers[activeBuffer],
                database: database,
                results: resultBuffers[activeBuffer]
            )
        }

        // Alternate buffers
        activeBuffer = 1 - activeBuffer
    }

    // Wait for final batch
    if let pending = pendingCompletion {
        try await pending.value
    }

    return allResults
}
```

---

## Metal4Context Async Patterns

VectorAccelerate's Metal4Context supports async execution:

```swift
// 📍 See: Sources/VectorAccelerate/Core/Metal4Context.swift:227-260

/// Execute without waiting (async)
public func execute<T: Sendable>(
    _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T
) async throws -> T {
    guard let commandBuffer = makeCommandBuffer() else {
        throw VectorError.deviceInitializationFailed("Failed to create command buffer")
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw VectorError.encoderCreationFailed()
    }

    let result = try await operation(commandBuffer, encoder)

    encoder.endEncoding()
    commandBuffer.commit()  // Returns immediately!

    return result
}

/// Execute and wait (blocking)
public func executeAndWait(
    _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> Void
) async throws {
    // ... same setup ...

    // Wait for completion using shared event
    await withCheckedContinuation { continuation in
        completionEvent.notify(...) { _, _ in
            continuation.resume()
        }
    }
}
```

---

## Command Buffer Pipelining

Multiple command buffers can be in-flight:

```
Command Buffer Pipeline:
  CB1: [Encode]──[Submit]────[Execute on GPU]────[Complete]
  CB2:           [Encode]──[Submit]────[Execute on GPU]────[Complete]
  CB3:                     [Encode]──[Submit]────[Execute on GPU]────[Complete]

GPU stays busy while CPU prepares next command buffer!
```

```swift
/// Pipeline command buffers for throughput
func pipelinedExecution(
    batches: [BatchWork],
    maxInFlight: Int = 3
) async throws {
    var inFlightTasks: [Task<Void, Error>] = []

    for batch in batches {
        // Limit in-flight work
        if inFlightTasks.count >= maxInFlight {
            try await inFlightTasks.removeFirst().value
        }

        // Launch async
        let task = Task {
            try await context.executeAndWait { cb, encoder in
                // Encode batch
                kernel.encode(into: encoder, batch: batch)
            }
        }
        inFlightTasks.append(task)
    }

    // Wait for all remaining
    for task in inFlightTasks {
        try await task.value
    }
}
```

---

## Shared Events for Synchronization

Metal 4 uses shared events for CPU/GPU synchronization:

```swift
// 📍 See: Sources/VectorAccelerate/Core/Metal4Context.swift:299-313

/// Wait for GPU completion using shared event
private func waitForCompletion(targetValue: UInt64) async {
    await withCheckedContinuation { continuation in
        completionEvent.notify(
            MTLSharedEventListener(dispatchQueue: .global()),
            atValue: targetValue
        ) { _, _ in
            continuation.resume()
        }
    }
}
```

---

## 🔗 VectorCore Connection

VectorCore is synchronous—operations complete before returning:

```swift
// VectorCore: Blocking operations
let dist = l2DistanceSquared(a, b)  // Returns when done
```

---

## 🔗 VectorIndex Connection

VectorIndex's async actor methods can be overlapped:

```swift
// VectorIndex: Actor methods are async
let results1 = try await index.search(query1, k: k)
let results2 = try await index.search(query2, k: k)
// But they execute sequentially on the actor
```

VectorAccelerate enables true GPU parallelism with overlapped batches.

---

## When Async Helps

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ASYNC BENEFIT ANALYSIS                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CPU prep time vs GPU compute time:                                 │
│                                                                      │
│  CPU >> GPU:  CPU-bound, async helps a lot                          │
│               └─ Example: Complex query preprocessing               │
│                                                                      │
│  CPU << GPU:  GPU-bound, async helps moderately                     │
│               └─ Example: Large batch search                        │
│                                                                      │
│  CPU ≈ GPU:   Balanced, async helps maximally                       │
│               └─ Example: Medium batch with result processing       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Overlap CPU and GPU**: Don't let either sit idle

2. **Double buffering**: Alternate buffers to avoid conflicts

3. **Pipeline command buffers**: Keep GPU queue fed

4. **Use shared events**: Efficient CPU/GPU synchronization

5. **Profile to find balance**: Measure CPU vs GPU time

---

## Next Up

Using Instruments to find bottlenecks:

**[→ 5.3 Profiling with Instruments](./03-Profiling-With-Instruments.md)**

---

*Guide 5.2 of 5.4 • Chapter 5: Pipeline Optimization*
