# VectorAccelerate Integration Fixes

Tracking document for long-term architectural improvements identified during VectorCore 0.1.5 integration.

---

## Priority 1: Critical Performance Fixes

### 1.1 Eliminate `.toArray()` Anti-Pattern

**Status**: In Progress (Core Kernels Complete)
**Impact**: High (15-40% performance improvement potential)
**Complexity**: High (Swift 6 concurrency constraints)

**Problem**: 55+ instances of `.toArray()` calls create unnecessary allocations when VectorCore's `withUnsafeBufferPointer` provides zero-copy access.

**Locations**:
```
Sources/VectorAccelerate/Operations/BatchDistanceOperations.swift
Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift
Sources/VectorAccelerate/ML/EmbeddingEngine.swift
Sources/VectorAccelerate/Kernels/*.swift
```

**Solution Implemented**: Added zero-copy buffer creation methods to `KernelContext`:

```swift
// KernelContext.swift - New methods added
@inlinable
public func createAlignedBufferFromVectors<V: VectorProtocol>(
    _ vectors: [V],
    options: MTLResourceOptions = .storageModeShared,
    alignment: Int = 16
) -> (any MTLBuffer)? where V.Scalar == Float {
    // Uses withUnsafeBufferPointer internally - no intermediate allocations
    for (i, vector) in vectors.enumerated() {
        vector.withUnsafeBufferPointer { srcPtr in
            // Direct memory copy to Metal buffer
            dst.update(from: srcPtr.baseAddress!, count: dimension)
        }
    }
}
```

**Updated Kernels** (eliminate `.toArray()` + `.flatMap`):
- [x] L2DistanceKernel - `compute<V: VectorProtocol>` now uses `createAlignedBufferFromVectors`
- [x] DotProductKernel - `compute<V: VectorProtocol>` now uses `createAlignedBufferFromVectors`
- [x] CosineSimilarityKernel - `compute<V: VectorProtocol>` now uses `createAlignedBufferFromVectors`
- [x] MinkowskiDistanceKernel - `distance<V>` and `distanceMatrix<V>` use zero-copy
- [x] JaccardDistanceKernel - `computeDistance<V>` and `computeDistanceMatrix<V>` use zero-copy
- [x] MinkowskiCalculator - `calculateDistance<V>` and `calculateDistanceMatrix<V>` use zero-copy
- [x] L2NormalizationKernel - `normalize<V>` uses `createAlignedBufferFromVectors`
- [x] FusedL2TopKKernel - `search<Q,D>` uses `createAlignedBufferFromVectors`
- [x] HammingDistanceKernel - `distance<V>` uses `withUnsafeBufferPointer`
- [x] ParallelReductionKernel - `reduce<V>` uses `createAlignedBufferFromVector`
- [x] ElementwiseKernel - `compute<V>` uses `createAlignedBufferFromVector`
- [x] ScalarQuantizationKernel - `quantize<V>` uses `createAlignedBufferFromVector`
- [x] BinaryQuantizationKernel - `quantize<V>` batch/single both use zero-copy

**Remaining Work**:
- [ ] VectorCoreIntegration.swift (18 calls - requires actor refactoring)
- [ ] StatisticsKernel, MatrixMultiplyKernel, QuantizationStatisticsKernel, etc. (lower priority)

**Tests Passing**: All 529 tests pass

---

### 1.2 Adopt VectorCore's `StaticDimension` Protocol

**Status**: ✅ Complete
**Impact**: Medium (compile-time optimization, type safety)
**Complexity**: Low

**Problem**: Current kernel dispatch uses runtime dimension checks.

**Solution Implemented**: Added `StaticDimension`-constrained methods to core kernels that provide:
- Type safety: Cannot mix vectors of different dimensions at compile time
- Compile-time dimension access via `D.value`
- Automatic optimized kernel selection for dimensions 384, 512, 768, 1536

**Implementation** (added to L2DistanceKernel, DotProductKernel, CosineSimilarityKernel):

```swift
/// Compute using compile-time dimensioned vectors.
/// Example: let distances = try await kernel.compute(queries: queries, database: database)
/// where queries/database are [Vector<Dim384>]
public func compute<D: StaticDimension>(
    queries: [Vector<D>],
    database: [Vector<D>],
    computeSqrt: Bool = true
) async throws -> [[Float]] {
    // Dimension is known at compile time
    let dimension = D.value
    // ... zero-copy buffer creation and GPU dispatch
}

/// Convenience type aliases for common embedding dimensions
public typealias Vector384 = Vector<Dim384>
public typealias Vector512 = Vector<Dim512>
public typealias Vector768 = Vector<Dim768>
public typealias Vector1536 = Vector<Dim1536>
```

**Completed Kernels**:
- [x] L2DistanceKernel - `compute<D: StaticDimension>`
- [x] DotProductKernel - `compute<D: StaticDimension>`
- [x] CosineSimilarityKernel - `compute<D: StaticDimension>`

**Tasks**:
- [x] Add `StaticDimension` constraint to kernel compute methods
- [x] Create specialized overloads for Vector<Dim384>, Vector<Dim512>, etc.
- [x] Add convenience type aliases
- [x] All tests passing (529 tests)

---

### 1.3 Replace `AccelerationError` with `VectorError`

**Status**: ✅ Complete
**Impact**: Low (API consistency)
**Complexity**: Medium (457 throw sites across 39 files)

**Problem**: VectorAccelerate defined custom `AccelerationError` when VectorCore provides `VectorError`.

**Solution Implemented**:

1. Created `VectorError+GPU.swift` with GPU-specific factory methods:
```swift
public extension VectorError {
    static func metalNotAvailable() -> VectorError
    static func deviceInitializationFailed(_ reason: String) -> VectorError
    static func bufferAllocationFailed(size: Int) -> VectorError
    static func bufferCreationFailed(_ reason: String) -> VectorError
    static func shaderNotFound(name: String) -> VectorError
    static func shaderCompilationFailed(_ reason: String) -> VectorError
    static func pipelineCreationFailed(_ reason: String) -> VectorError
    static func computeFailed(reason: String) -> VectorError
    static func bufferPoolExhausted() -> VectorError
    static func memoryPressure() -> VectorError
    static func encoderCreationFailed() -> VectorError
    static func commandQueueCreationFailed() -> VectorError
    static func libraryCreationFailed() -> VectorError
    static func unsupportedGPUOperation(_ operation: String) -> VectorError
    static func invalidInput(_ reason: String) -> VectorError
    static func countMismatch(expected: Int?, actual: Int?) -> VectorError
    static func fileNotFound(_ path: String) -> VectorError
    static func invalidDataFormat(_ reason: String) -> VectorError
    static func invalidOperation(_ reason: String) -> VectorError
}
```

2. Deprecated `AccelerationError` with migration guidance (kept for backward compatibility)

3. Migrated all 457 throw sites to use VectorError factory methods

4. Updated all tests to use VectorError pattern matching

**Completed Tasks**:
- [x] Created VectorError+GPU.swift with 19 GPU-specific factory methods
- [x] Added deprecation to AccelerationError with migration guide
- [x] Migrated 457 throw sites across 39 source files
- [x] Updated 32 catch blocks in 12 test files
- [x] All 529 tests passing

---

## Priority 2: VectorCore Enhancement Requests

These require changes in VectorCore (coordinate with VectorCore team).

### 2.1 Batch Distance Protocol

**Status**: Requested
**Owner**: VectorCore team

**Request**: Standardized protocol for batch distance computation:

```swift
public protocol BatchDistanceComputable {
    associatedtype Vector: VectorProtocol

    func computeBatch(
        query: Vector,
        candidates: [Vector]
    ) -> [Float]

    func computeMatrix(
        queries: [Vector],
        database: [Vector]
    ) -> [[Float]]
}
```

**Benefit**: VectorAccelerate can provide GPU-accelerated conformance while maintaining API compatibility with CPU implementations.

**Tasks**:
- [ ] Draft protocol specification
- [ ] Submit to VectorCore as proposal
- [ ] Implement GPU-accelerated conformance once accepted

---

### 2.2 Streaming Operations Protocol

**Status**: Requested
**Owner**: VectorCore team

**Request**: Iterator-based API for memory-efficient processing:

```swift
public protocol StreamingVectorSource: AsyncSequence
    where Element: VectorProtocol {

    var estimatedCount: Int? { get }
    var dimension: Int { get }
}

public protocol StreamingDistanceComputable {
    func computeStreaming<S: StreamingVectorSource>(
        query: S.Element,
        source: S
    ) -> AsyncStream<(index: Int, distance: Float)>
}
```

**Benefit**: Process datasets larger than memory, enable early termination for top-k searches.

**Tasks**:
- [ ] Draft protocol specification
- [ ] Submit to VectorCore as proposal
- [ ] Implement with GPU batch accumulation

---

### 2.3 Normalization State Tracking

**Status**: Partially Available (IndexableVector)
**Owner**: VectorCore team

**Current**: `IndexableVector` provides `isNormalized` property.

**Enhancement Request**: Mutable state or factory pattern:

```swift
// Option A: Mutable state
public protocol NormalizationTrackable: VectorProtocol {
    var isNormalized: Bool { get set }
    mutating func markNormalized()
}

// Option B: Factory with state
public struct NormalizedVector<Base: VectorProtocol>: VectorProtocol {
    public let base: Base
    // Guaranteed normalized at construction
}
```

**Benefit**: Skip redundant normalization checks in pipelines.

**Tasks**:
- [ ] Evaluate IndexableVector coverage
- [ ] Propose enhancement if gaps exist
- [ ] Integrate into VectorAccelerate pipelines

---

## Priority 3: Architectural Improvements

### 3.1 Non-Actor Buffer Factory

**Status**: ✅ Complete
**Impact**: High (enables zero-copy)
**Complexity**: Medium

**Problem**: `BufferPool` is an actor, forcing all buffer operations through async boundaries.

**Solution Implemented**: Separated buffer creation from pool management:

1. **MetalBufferFactory** (`Core/MetalBufferFactory.swift`):
   - Synchronous, non-actor class (`@unchecked Sendable`)
   - Direct access to raw `MTLDevice` for thread-safe buffer creation
   - Zero-copy methods: `createBuffer(fromVectors:)`, `createBuffer(fromVector:)`
   - Bucket sizing helpers integrated with pool
   - Default resource options based on device capabilities

2. **Updated Components**:
   - `MetalDevice.rawDevice`: Nonisolated access to underlying MTLDevice
   - `MetalDevice.makeBufferFactory()`: Factory creation method
   - `BufferPool`: Now uses factory for synchronous buffer allocation
   - `SmartBufferPool`: Updated to use factory
   - `MetalContext.bufferFactory`: Exposed for direct access

```swift
// Example: Synchronous buffer creation (no await needed)
let factory = context.bufferFactory  // nonisolated access
let buffer = factory.createBuffer(length: 4096)  // synchronous!

// Zero-copy from VectorProtocol
let vectorBuffer = factory.createBuffer(fromVectors: vectors)
```

**Completed Tasks**:
- [x] Design `MetalBufferFactory` class with zero-copy support
- [x] Refactor `BufferPool` to use factory for synchronous allocation
- [x] Refactor `SmartBufferPool` to use factory
- [x] Add `MetalDevice.rawDevice` nonisolated property
- [x] Add `MetalDevice.makeBufferFactory()` method
- [x] Update `MetalContext` to expose factory
- [x] All 529 tests passing

---

### 3.2 Pipeline State Caching Improvements

**Status**: Not Started
**Impact**: Medium
**Complexity**: Low

**Problem**: Current caching is basic string-keyed dictionary.

**Solution**: Use dimension-indexed array for O(1) lookup:

```swift
private struct PipelineCache {
    private var byDimension: [Int: any MTLComputePipelineState] = [:]
    private var byHash: [Int: any MTLComputePipelineState] = [:]

    subscript(dimension: Int) -> (any MTLComputePipelineState)? {
        get { byDimension[dimension] }
        set { byDimension[dimension] = newValue }
    }
}
```

**Tasks**:
- [ ] Implement `PipelineCache` struct
- [ ] Migrate kernel classes to use it
- [ ] Add cache statistics to `PoolStatistics`

---

## Tracking

### Completed
- [x] VectorCore 0.1.5 dependency upgrade
- [x] 384-dimension GPU kernels (L2, DotProduct, Cosine)
- [x] Manhattan distance SIMD integration
- [x] `getBuffer(forVector:)` partial optimization
- [x] `EmbeddingSearchResult` type collision fix
- [x] `KernelContext.createAlignedBufferFromVectors` - zero-copy buffer creation
- [x] L2DistanceKernel zero-copy optimization
- [x] DotProductKernel zero-copy optimization
- [x] CosineSimilarityKernel zero-copy optimization
- [x] MinkowskiDistanceKernel zero-copy optimization
- [x] JaccardDistanceKernel zero-copy optimization
- [x] MinkowskiCalculator zero-copy optimization
- [x] L2NormalizationKernel zero-copy optimization
- [x] FusedL2TopKKernel zero-copy optimization
- [x] HammingDistanceKernel zero-copy optimization
- [x] ParallelReductionKernel zero-copy optimization
- [x] ElementwiseKernel zero-copy optimization
- [x] ScalarQuantizationKernel zero-copy optimization
- [x] BinaryQuantizationKernel zero-copy optimization
- [x] **Priority 1.2**: StaticDimension protocol adoption (L2Distance, DotProduct, CosineSimilarity)
- [x] **Priority 1.3**: Replaced AccelerationError with VectorError (457 throw sites, 39 files)
- [x] Created VectorError+GPU.swift with 19 GPU-specific factory methods
- [x] **Priority 3.1**: Non-Actor Buffer Factory (MetalBufferFactory class)
- [x] All 529 tests passing

### In Progress
- [ ] None currently

### Next Priorities
- [ ] Priority 3.2: Pipeline State Caching Improvements

### Blocked
- [ ] VectorCoreIntegration.swift zero-copy (requires actor architecture refactor)
- [ ] StatisticsKernel, MatrixMultiplyKernel, etc. (very low priority)

---

## Files Modified

### Core Infrastructure
- `Sources/VectorAccelerate/Core/KernelContext.swift` - Added zero-copy buffer creation methods
- `Sources/VectorAccelerate/Core/VectorError+GPU.swift` - NEW: GPU-specific VectorError factory methods
- `Sources/VectorAccelerate/Core/AccelerationError.swift` - Deprecated with migration guidance
- `Sources/VectorAccelerate/Core/MetalBufferFactory.swift` - NEW: Synchronous buffer factory
- `Sources/VectorAccelerate/Core/MetalDevice.swift` - Added `rawDevice`, `makeBufferFactory()`
- `Sources/VectorAccelerate/Core/MetalContext.swift` - Exposed `bufferFactory` property
- `Sources/VectorAccelerate/Core/BufferPool.swift` - Refactored to use factory
- `Sources/VectorAccelerate/Core/SmartBufferPool.swift` - Refactored to use factory

### Distance Kernels (Zero-Copy Optimized)
- `Sources/VectorAccelerate/Kernels/L2DistanceKernel.swift`
- `Sources/VectorAccelerate/Kernels/DotProductKernel.swift`
- `Sources/VectorAccelerate/Kernels/CosineSimilarityKernel.swift`
- `Sources/VectorAccelerate/Kernels/MinkowskiDistanceKernel.swift`
- `Sources/VectorAccelerate/Kernels/JaccardDistanceKernel.swift`
- `Sources/VectorAccelerate/Kernels/MinkowskiCalculator.swift`
- `Sources/VectorAccelerate/Kernels/L2NormalizationKernel.swift`
- `Sources/VectorAccelerate/Kernels/FusedL2TopKKernel.swift`
- `Sources/VectorAccelerate/Kernels/HammingDistanceKernel.swift`
- `Sources/VectorAccelerate/Kernels/ParallelReductionKernel.swift`
- `Sources/VectorAccelerate/Kernels/ElementwiseKernel.swift`
- `Sources/VectorAccelerate/Kernels/ScalarQuantizationKernel.swift`
- `Sources/VectorAccelerate/Kernels/BinaryQuantizationKernel.swift`

---

## References

- VectorCore repository: https://github.com/gifton/VectorCore
- Swift Evolution SE-0302: Sendable and @Sendable closures
- Metal Best Practices Guide: Buffer Management

---

*Last updated: 2025-11-26* (Priority 3.1 complete)
