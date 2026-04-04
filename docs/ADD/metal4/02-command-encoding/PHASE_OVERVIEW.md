# Phase 2: Command Encoding

## Objective

Migrate from individual encoder types to Metal 4's unified `MTL4ComputeCommandEncoder` and implement argument tables for efficient resource binding.

## Dependencies

- Phase 1 complete (Metal4Context, ResidencyManager, MetalDevice updates)

## Key Concepts

### Unified Encoder

Metal 4 consolidates compute, blit, and acceleration structure operations into a single encoder type:

```swift
// Metal 3 - Separate encoders
let computeEncoder = commandBuffer.makeComputeCommandEncoder()
computeEncoder.setComputePipelineState(pipeline)
computeEncoder.dispatchThreadgroups(...)
computeEncoder.endEncoding()

let blitEncoder = commandBuffer.makeBlitCommandEncoder()
blitEncoder.copy(from: src, to: dst)
blitEncoder.endEncoding()

// Metal 4 - Unified encoder
let encoder = commandBuffer.makeComputeCommandEncoder() as! MTL4ComputeCommandEncoder
encoder.setComputePipelineState(pipeline)
encoder.dispatchThreadgroups(...)
encoder.barrier(resources: [...], beforeStages: .dispatch, afterStages: .blit)
encoder.copy(from: src, to: dst)
encoder.endEncoding()  // Only one end!
```

### Argument Tables

Replace per-dispatch `setBuffer()` calls with reusable argument tables:

```swift
// Metal 3 - Per-dispatch binding
encoder.setBuffer(queries, offset: 0, index: 0)
encoder.setBuffer(database, offset: 0, index: 1)
encoder.setBuffer(results, offset: 0, index: 2)

// Metal 4 - Argument table
let argTable = device.makeArgumentTable(descriptor: descriptor)
argTable.setAddress(queries.gpuAddress, index: 0)
argTable.setAddress(database.gpuAddress, index: 1)
argTable.setAddress(results.gpuAddress, index: 2)
encoder.setArgumentTable(argTable, stages: .compute)
```

---

## Tasks

| Task | Description | Status | Dependencies |
|------|-------------|--------|--------------|
| task-argument-table-manager.md | ArgumentTablePool for reusable tables | **Complete** | Phase 1 |
| task-metal4-compute-engine.md | Updated ComputeEngine for unified encoder | **Complete** | ArgumentTableManager |
| task-barrier-strategy.md | Barrier placement guidelines | **Complete** | None |
| task-kernel-binding-migration.md | Update all kernel binding patterns | Pending | ArgumentTableManager |

---

## Encoding Patterns by Operation

### Pattern 1: Simple Compute (Distance Calculation)

```
Operation: L2 Distance between query batch and database
Command Buffer: 1
Encoding:
┌─────────────────────────────────────┐
│ setComputePipelineState(l2Distance) │
│ setArgumentTable(distanceArgs)      │
│ dispatchThreadgroups(...)           │
└─────────────────────────────────────┘
Barriers: None (single dispatch)
```

### Pattern 2: Distance + Selection (Fused Pipeline)

```
Operation: L2 Distance → Top-K Selection
Command Buffer: 1 (fused for efficiency)
Encoding:
┌─────────────────────────────────────┐
│ setComputePipelineState(l2Distance) │
│ setArgumentTable(distanceArgs)      │
│ dispatchThreadgroups(...)           │
│                                     │
│ barrier(distances, .dispatch→.dispatch) │
│                                     │
│ setComputePipelineState(topK)       │
│ setArgumentTable(topKArgs)          │
│ dispatchThreadgroups(...)           │
└─────────────────────────────────────┘
Barriers: 1 (distance write → topK read)
```

### Pattern 3: PQ Search (Multi-Stage)

```
Operation: Lookup PQ codes → Compute asymmetric distances → Select Top-K
Command Buffer: 1
Encoding:
┌─────────────────────────────────────┐
│ // Stage 1: Distance table lookup   │
│ setComputePipelineState(pqLookup)   │
│ setArgumentTable(lookupArgs)        │
│ dispatchThreadgroups(...)           │
│                                     │
│ barrier(distTable, .dispatch→.dispatch) │
│                                     │
│ // Stage 2: Asymmetric distance     │
│ setComputePipelineState(pqDistance) │
│ setArgumentTable(distanceArgs)      │
│ dispatchThreadgroups(...)           │
│                                     │
│ barrier(distances, .dispatch→.dispatch) │
│                                     │
│ // Stage 3: Selection               │
│ setComputePipelineState(topK)       │
│ setArgumentTable(topKArgs)          │
│ dispatchThreadgroups(...)           │
└─────────────────────────────────────┘
Barriers: 2
```

### Pattern 4: Streaming Top-K (Batched)

```
Operation: Process large database in chunks, maintain running top-K
Command Buffer: 1 per chunk, or 1 with multiple dispatches
Encoding (per chunk):
┌─────────────────────────────────────┐
│ for chunk in chunks:                │
│   setComputePipelineState(distance) │
│   setArgumentTable(chunkArgs[i])    │
│   dispatchThreadgroups(...)         │
│                                     │
│   barrier(chunkDist, .dispatch→.dispatch) │
│                                     │
│   setComputePipelineState(merge)    │
│   setArgumentTable(mergeArgs)       │
│   dispatchThreadgroups(...)         │
│                                     │
│   barrier(runningTopK, .dispatch→.dispatch) │
└─────────────────────────────────────┘
Barriers: 2 per chunk
```

### Pattern 5: Quantization (Encode + Decode)

```
Operation: Scalar quantize vectors → Later dequantize
Encoding (quantize):
┌─────────────────────────────────────┐
│ setComputePipelineState(quantize)   │
│ setArgumentTable(quantizeArgs)      │
│ dispatchThreadgroups(...)           │
└─────────────────────────────────────┘

Encoding (dequantize + distance):
┌─────────────────────────────────────┐
│ setComputePipelineState(dequantize) │
│ setArgumentTable(dequantArgs)       │
│ dispatchThreadgroups(...)           │
│                                     │
│ barrier(floatVecs, .dispatch→.dispatch) │
│                                     │
│ setComputePipelineState(distance)   │
│ setArgumentTable(distanceArgs)      │
│ dispatchThreadgroups(...)           │
└─────────────────────────────────────┘
Barriers: 1 (dequantize → distance)
```

---

## Hazard Map

### Buffer Hazards by Kernel Type

| Kernel | Inputs (Read) | Outputs (Write) | Hazards |
|--------|---------------|-----------------|---------|
| **L2Distance** | queries, database | distances | None |
| **CosineSimilarity** | queries, database | distances | None |
| **DotProduct** | queries, database | distances | None |
| **TopKSelection** | distances | indices, values | Read→Write on distances |
| **StreamingTopK** | distances, runningTopK | runningTopK | Read/Write on same buffer |
| **FusedL2TopK** | queries, database | indices, values | Internal (fused) |
| **ScalarQuantize** | floatVectors | quantizedVectors | None |
| **ScalarDequantize** | quantizedVectors | floatVectors | None |
| **PQEncode** | vectors, codebook | codes | None |
| **PQDistance** | codes, distTable | distances | distTable must be ready |
| **MatrixMultiply** | matA, matB | matC | None |
| **L2Normalize** | input | output (can be same) | In-place hazard |

### Barrier Requirements

| Transition | Barrier Type | Example |
|------------|--------------|---------|
| Compute → Compute (different buffer) | None | L2 → TopK (different buffers) |
| Compute → Compute (same buffer read) | `.dispatch → .dispatch` | Distance write → Selection read |
| Compute → Blit | `.dispatch → .blit` | Compute → Copy results |
| Blit → Compute | `.blit → .dispatch` | Copy input → Compute |
| In-place update | `.dispatch → .dispatch` + fence | Streaming TopK merge |

### Memory Fence Requirements

```swift
// Standard dispatch barrier
encoder.barrier(resources: [distances],
                beforeStages: .dispatch,
                afterStages: .dispatch)

// Pass barrier (all previous → all subsequent)
encoder.passBarrier()

// Memory barrier (all resources)
encoder.memoryBarrier(beforeStages: .dispatch, afterStages: .dispatch)
```

---

## Command Buffer Strategy

### Option A: One Command Buffer Per Query (Simple)

**Pros:** Simple, isolated, easy debugging
**Cons:** More overhead, less batching opportunity

```swift
for query in queries {
    let cb = device.makeCommandBuffer()
    // encode single query
    queue.commit([cb])
}
```

### Option B: One Command Buffer Per Batch (Recommended)

**Pros:** Better throughput, amortized overhead
**Cons:** Batch failures affect all queries

```swift
let cb = device.makeCommandBuffer()
for query in queryBatch {
    // encode query (with barriers between)
}
queue.commit([cb])
```

### Option C: Pipeline of Command Buffers

**Pros:** Maximum throughput, overlap CPU/GPU
**Cons:** Complex synchronization

```swift
let cbs = [device.makeCommandBuffer(), device.makeCommandBuffer(), device.makeCommandBuffer()]
var idx = 0
for batch in batches {
    let cb = cbs[idx % 3]
    // wait for previous use of this CB
    // encode batch
    queue.commit([cb])
    idx += 1
}
```

**Recommendation:** Start with Option B, move to Option C for performance-critical paths.

---

## ArgumentTablePool Design

```swift
/// Pool of reusable argument tables
public actor ArgumentTablePool {
    private let device: MTLDevice
    private var available: [MTL4ArgumentTable] = []
    private var inUse: Set<ObjectIdentifier> = []
    private let maxTables: Int

    public init(device: MTLDevice, maxTables: Int = 32) {
        self.device = device
        self.maxTables = maxTables
    }

    /// Acquire an argument table (creates if needed)
    public func acquire() throws -> MTL4ArgumentTable {
        if let table = available.popLast() {
            inUse.insert(ObjectIdentifier(table))
            return table
        }

        guard inUse.count < maxTables else {
            throw Metal4Error.argumentTablePoolExhausted
        }

        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = 16
        let table = try device.makeArgumentTable(descriptor: descriptor)
        inUse.insert(ObjectIdentifier(table))
        return table
    }

    /// Return an argument table to the pool
    public func release(_ table: MTL4ArgumentTable) {
        inUse.remove(ObjectIdentifier(table))
        available.append(table)
    }
}
```

---

## Completion Criteria

- [ ] ArgumentTablePool implemented and tested
- [ ] Metal4ComputeEngine uses unified encoder
- [ ] All kernel bindings migrated to argument tables
- [ ] Barrier placement documented for each kernel
- [ ] Fused pipelines working (distance + topK)
- [ ] No regressions in existing tests
- [ ] Benchmark shows no performance regression

## Files Modified

- `Core/ComputeEngine.swift` → Major updates for unified encoder
- `Core/ArgumentTablePool.swift` → NEW FILE
- `Kernels/*.swift` → Update binding patterns (all 25+ files)

## Risk Mitigation

- Test each kernel migration individually
- Keep Metal 3 binding code available for comparison
- Profile before/after to detect regressions
- Start with simple kernels (L2Distance) before complex ones (StreamingTopK)
