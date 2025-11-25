# VectorAccelerate ↔ VectorCore Alignment Roadmap

**Generated**: 2025-11-12
**VectorCore Version**: 0.1.4
**Current Integration Score**: 35/100
**Target Integration Score**: 85/100

## Executive Summary

This roadmap outlines the required changes to align VectorAccelerate with VectorCore's latest protocol architecture and public APIs. VectorCore has undergone significant upgrades since initial integration, introducing rich protocol abstractions for compute providers, buffer management, and acceleration strategies.

**Current Status**: VectorAccelerate implements 2 of 7+ available protocols (DistanceProvider, VectorOperationsProvider)
**Target**: Implement 6 core protocols and eliminate API duplication
**Estimated Total Effort**: 20-30 developer hours
**Impact**: Enables VectorCore applications to seamlessly leverage Metal/GPU acceleration

---

## Priority Classification

- 🔴 **HIGH**: Blocking issues preventing VectorCore apps from using GPU acceleration
- 🟡 **MEDIUM**: Important for full feature parity and best practices
- 🟢 **LOW**: Nice-to-have improvements for completeness

---

## Phase 1: Critical Protocol Implementations (HIGH PRIORITY)

### 1.1 🔴 Implement ComputeProvider Protocol

**Objective**: Enable VectorCore to use VectorAccelerate's GPU compute through standardized abstraction

**Files to Modify**:
- `Sources/VectorAccelerate/Core/ComputeEngine.swift`
- `Sources/VectorAccelerate/Core/MetalContext.swift`

**Required Changes**:

```swift
// In ComputeEngine.swift
public actor ComputeEngine: ComputeProvider {
    // Add VectorCore conformance
    public let device: ComputeDevice = .gpu(index: 0)

    public var maxConcurrency: Int {
        // Return based on Metal device capabilities
        return context.device.capabilities.maxThreadsPerThreadgroup
    }

    public var deviceInfo: ComputeDeviceInfo {
        ComputeDeviceInfo(
            name: context.device.name,
            availableMemory: context.device.capabilities.maxBufferLength,
            maxThreads: context.device.capabilities.maxThreadsPerThreadgroup,
            preferredChunkSize: 1024  // Tune based on benchmarks
        )
    }

    public func execute<T: Sendable>(
        _ work: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        // Delegate to existing execution infrastructure
        return try await work()
    }

    public func parallelExecute<T: Sendable>(
        items: Range<Int>,
        _ work: @Sendable @escaping (Int) async throws -> T
    ) async throws -> [T] {
        // Implement using Metal compute pipeline
        // Divide into chunks based on GPU capability
        var results: [T] = []
        results.reserveCapacity(items.count)

        for index in items {
            let result = try await work(index)
            results.append(result)
        }
        return results
    }

    public func parallelForEach(
        items: Range<Int>,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws {
        // Batch execute on GPU
        for index in items {
            try await body(index)
        }
    }

    public func parallelForEach(
        items: Range<Int>,
        minChunk: Int,
        _ body: @Sendable @escaping (Int) async throws -> Void
    ) async throws {
        // Chunked execution for better GPU utilization
        let chunkSize = max(minChunk, items.count / maxConcurrency)
        for chunkStart in stride(from: items.lowerBound, to: items.upperBound, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, items.upperBound)
            for index in chunkStart..<chunkEnd {
                try await body(index)
            }
        }
    }

    public func parallelReduce<R: Sendable>(
        items: Range<Int>,
        initial: R,
        _ rangeWork: @Sendable @escaping (Range<Int>) async throws -> R,
        _ combine: @Sendable @escaping (R, R) -> R
    ) async throws -> R {
        // GPU-accelerated parallel reduction
        let chunkSize = max(1, items.count / maxConcurrency)
        var accumulator = initial

        for chunkStart in stride(from: items.lowerBound, to: items.upperBound, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, items.upperBound)
            let partialResult = try await rangeWork(chunkStart..<chunkEnd)
            accumulator = combine(accumulator, partialResult)
        }

        return accumulator
    }
}
```

**Testing**:
```swift
let engine = try await ComputeEngine.shared()
let provider: any ComputeProvider = engine

// Test parallel execution
let results = try await provider.parallelExecute(items: 0..<1000) { index in
    // GPU work
    return index * 2
}

// Test parallel reduction
let sum = try await provider.parallelReduce(
    items: 0..<10000,
    initial: 0.0
) { range in
    // Compute partial sum on GPU
    return Float(range.reduce(0, +))
} combine: { $0 + $1 }
```

**Estimated Effort**: 6 hours
**Blockers**: None
**Dependencies**: Import VectorCore

---

### 1.2 🔴 Eliminate Distance Metric Duplication

**Objective**: Use VectorCore's `SupportedDistanceMetric` everywhere, eliminate custom enums

**Files to Modify**:
- `Sources/VectorAccelerate/Core/Types.swift` (lines 87-122)
- `Sources/VectorAccelerate/Operations/BatchProcessor.swift` (lines 13-17)
- `Sources/VectorAccelerate/ML/EmbeddingEngine.swift` (line 25)
- All kernel dispatch logic

**Required Changes**:

**Step 1**: Remove custom `DistanceMetric` enum from `Types.swift`
```swift
// DELETE this enum entirely (lines 87-122)
// public enum DistanceMetricType { ... }
```

**Step 2**: Update `BatchProcessor.swift`
```swift
// REPLACE custom enum with VectorCore import
import VectorCore

public actor BatchProcessor {
    // Use VectorCore's SupportedDistanceMetric
    public func process(
        queries: [any VectorProtocol],
        candidates: [any VectorProtocol],
        metric: SupportedDistanceMetric,  // Changed from custom enum
        k: Int
    ) async throws -> [[SearchResult]] {
        // ... implementation
    }
}
```

**Step 3**: Update Metal shader dispatch
```swift
// In ComputeEngine or kernel dispatch code
private func kernelFunctionName(for metric: SupportedDistanceMetric) -> String {
    switch metric {
    case .euclidean: return "euclidean_distance"
    case .cosine: return "cosine_distance"
    case .dotProduct: return "dot_product_distance"
    case .manhattan: return "manhattan_distance"
    case .chebyshev: return "chebyshev_distance"
    }
}
```

**Step 4**: Update `EmbeddingEngine.swift`
```swift
import VectorCore

public struct EmbeddingConfiguration {
    public let metric: SupportedDistanceMetric  // Changed from custom type
    // ... rest of configuration
}
```

**Testing**:
- Verify all distance metric tests still pass
- Confirm Metal shader dispatch works correctly
- Check that API is compatible with VectorCore applications

**Estimated Effort**: 3 hours
**Blockers**: None
**Dependencies**: VectorCore import

---

### 1.3 🔴 Implement BufferProvider Protocol

**Objective**: Enable VectorCore to use VectorAccelerate's buffer pooling through standard abstraction

**Files to Modify**:
- `Sources/VectorAccelerate/Core/BufferPool.swift`
- `Sources/VectorAccelerate/Core/SmartBufferPool.swift`

**Required Changes**:

```swift
import VectorCore

public actor BufferPool: BufferProvider {
    // Add VectorCore conformance
    public var alignment: Int { 256 }  // Metal buffer alignment

    public func acquire(size: Int) async throws -> BufferHandle {
        // Convert internal BufferToken to VectorCore's BufferHandle
        let token = try await acquireBuffer(size: size, type: .compute)

        return BufferHandle(
            id: token.id,
            size: size,
            pointer: token.buffer.contents()
        )
    }

    public func release(_ handle: BufferHandle) async {
        // Find token by ID and release
        if let token = activeBuffers.first(where: { $0.id == handle.id }) {
            await releaseBuffer(token)
        }
    }

    public func statistics() async -> BufferStatistics {
        let stats = await getStatistics()

        return BufferStatistics(
            totalAllocations: stats.totalAllocations,
            reusedBuffers: stats.reusedBuffers,
            currentUsageBytes: stats.currentMemoryUsage,
            peakUsageBytes: stats.peakMemoryUsage
        )
    }

    public func clear() async {
        await reset()
    }
}
```

**Migration Path for BufferToken → BufferHandle**:
- Option A: Keep internal `BufferToken`, add adapter layer
- Option B: Migrate to `BufferHandle` completely (recommended for long-term)

**Testing**:
```swift
let pool: any BufferProvider = await BufferPool.shared()

let handle = try await pool.acquire(size: 1024)
// Use buffer
await pool.release(handle)

let stats = await pool.statistics()
print("Hit rate: \(stats.hitRate)")
```

**Estimated Effort**: 5 hours
**Blockers**: None
**Dependencies**: VectorCore import

---

## Phase 2: Feature Parity (MEDIUM PRIORITY)

### 2.1 🟡 Implement AccelerationProvider Protocol

**Objective**: Enable capability querying and operation dispatch through VectorCore abstraction

**Files to Modify**:
- `Sources/VectorAccelerate/Core/MetalContext.swift`

**Required Changes**:

```swift
import VectorCore

extension MetalContext: AccelerationProvider {
    public typealias Config = MetalConfiguration

    public func isSupported(for operation: AcceleratedOperation) -> Bool {
        switch operation {
        case .distanceComputation:
            return true  // Always supported via Metal
        case .matrixMultiplication:
            return true  // Supported
        case .vectorNormalization:
            return true  // Supported
        case .batchedOperations:
            return true  // Supported via batch kernels
        }
    }

    public func accelerate<T>(
        _ operation: AcceleratedOperation,
        input: T
    ) async throws -> T {
        // Dispatch to appropriate Metal kernel based on operation
        switch operation {
        case .distanceComputation:
            // Route to distance kernels
            break
        case .matrixMultiplication:
            // Route to matrix kernels
            break
        case .vectorNormalization:
            // Route to normalization kernels
            break
        case .batchedOperations:
            // Route to batch processor
            break
        }

        return input  // Placeholder - implement actual acceleration
    }
}
```

**Testing**:
```swift
let context = try await MetalContext.shared(for: device)
let provider: any AccelerationProvider = context

if provider.isSupported(for: .distanceComputation) {
    let result = try await provider.accelerate(.distanceComputation, input: vectors)
}
```

**Estimated Effort**: 4 hours
**Blockers**: None
**Dependencies**: VectorCore import

---

### 2.2 🟡 Adopt ComputeDevice Throughout

**Objective**: Use VectorCore's device abstraction instead of custom detection logic

**Files to Modify**:
- All files using `MetalDevice.isAvailable`
- `Sources/VectorAccelerate/Core/MetalDevice.swift`
- `Sources/VectorAccelerate/Core/ComputeEngine.swift`

**Required Changes**:

```swift
// BEFORE
if await MetalDevice.isAvailable {
    // Use GPU
}

// AFTER
import VectorCore

let device = ComputeDevice.gpu()
if device.isAvailable {
    // Use GPU via VectorCore abstraction
}

// Query capabilities
if let capabilities = device.queryCapabilities() {
    print("Max parallelism: \(capabilities.maxParallelism)")
    print("Has unified memory: \(capabilities.hasUnifiedMemory)")
}
```

**Map Metal Capabilities to VectorCore**:
```swift
extension MetalDevice {
    func asComputeDevice() -> ComputeDevice {
        return .gpu(index: 0)  // Or device index if supporting multiple GPUs
    }

    func deviceCapabilities() -> DeviceCapabilities {
        DeviceCapabilities(
            maxParallelism: capabilities.maxThreadsPerThreadgroup,
            availableMemory: capabilities.maxBufferLength,
            supportedPrecisions: [.float16, .float32, .int8],
            hasUnifiedMemory: capabilities.hasUnifiedMemory
        )
    }
}
```

**Estimated Effort**: 3 hours
**Blockers**: None
**Dependencies**: VectorCore import

---

### 2.3 🟡 Enhance Batch Operations Integration

**Objective**: Integrate with VectorCore's `BatchOperations` for automatic GPU delegation

**Files to Modify**:
- `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift`

**Required Changes**:

```swift
// Add GPU-aware batch operation support
extension BatchOperations {
    /// GPU-accelerated nearest neighbor search
    public static func findNearestGPU<V: VectorProtocol & Sendable>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        // Use VectorAccelerate's GPU kernels
        let engine = try await ComputeEngine.shared()
        // ... GPU implementation
    }
}
```

**Estimated Effort**: 4 hours
**Blockers**: ComputeProvider implementation (1.1)
**Dependencies**: Phase 1 completion

---

## Phase 3: Completeness and Polish (LOW PRIORITY)

### 3.1 🟢 Implement VectorSerializable Support

**Objective**: Enable serialization of GPU-resident data

**Files to Create/Modify**:
- `Sources/VectorAccelerate/Serialization/VectorSerialization.swift` (new)

**Required Changes**:

```swift
import VectorCore

extension QuantizedVector: VectorSerializable {
    public func serialize() throws -> SerializedForm {
        // Serialize quantized data, parameters, and metadata
        let data = Data(/* buffer contents */)
        return SerializedForm(
            format: .custom("quantized-int8"),
            data: data,
            metadata: ["dimension": dimension, "scheme": quantizationScheme]
        )
    }

    public static func deserialize(from form: SerializedForm) throws -> Self {
        // Reconstruct from serialized form
        // ...
    }
}
```

**Estimated Effort**: 3 hours
**Blockers**: None
**Dependencies**: VectorCore import

---

### 3.2 🟢 Create OptimizedVector Implementations

**Objective**: Define SIMD-optimized Metal vectors conforming to VectorCore protocol

**Files to Create**:
- `Sources/VectorAccelerate/Vectors/MetalOptimizedVector512.swift` (new)
- `Sources/VectorAccelerate/Vectors/MetalOptimizedVector768.swift` (new)
- `Sources/VectorAccelerate/Vectors/MetalOptimizedVector1536.swift` (new)

**Required Changes**:

```swift
import VectorCore

public struct MetalOptimizedVector512: OptimizedVector, VectorProtocol {
    public typealias Scalar = Float
    public typealias Storage = ContiguousArray<SIMD4<Float>>

    public static let laneCount = 4
    public static let dimension = 512

    public var storage: Storage
    public var scalarCount: Int { 512 }

    // Specialized Metal kernel dispatch
    public func dotProduct(_ other: Self) -> Float {
        // Use optimized Metal kernel for 512-dim vectors
    }

    // ... other optimized operations
}
```

**Estimated Effort**: 8 hours
**Blockers**: None
**Dependencies**: VectorCore import

---

### 3.3 🟢 Add VectorFactory Conformance

**Objective**: Provide factory methods for creating VectorCore-compatible vectors

**Files to Modify**:
- `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift`

**Required Changes**:

```swift
extension Vector: VectorFactory where D: StaticDimension {
    public static func create(from scalars: [Float]) throws -> Self {
        return try Self(scalars)
    }
}

// Add GPU-specific factory
public struct MetalVectorFactory {
    public static func createOnGPU<D: StaticDimension>(
        from scalars: [Float]
    ) async throws -> Vector<D> {
        // Create vector and upload to GPU buffer
        let vector = try Vector<D>(scalars)
        // Pre-allocate GPU buffer
        return vector
    }
}
```

**Estimated Effort**: 2 hours
**Blockers**: None
**Dependencies**: VectorCore import

---

## Implementation Strategy

### Recommended Execution Order

**Week 1: Critical Foundations**
1. Day 1-2: Implement ComputeProvider (1.1) - 6 hours
2. Day 2-3: Eliminate distance metric duplication (1.2) - 3 hours
3. Day 3-5: Implement BufferProvider (1.3) - 5 hours

**Week 2: Feature Parity**
4. Day 1-2: Implement AccelerationProvider (2.1) - 4 hours
5. Day 3: Adopt ComputeDevice (2.2) - 3 hours
6. Day 4-5: Enhance batch operations (2.3) - 4 hours

**Week 3: Polish (Optional)**
7. Implement VectorSerializable (3.1) - 3 hours
8. Create OptimizedVector implementations (3.2) - 8 hours
9. Add VectorFactory conformance (3.3) - 2 hours

### Testing Strategy

**For Each Phase**:
1. Unit tests for protocol conformance
2. Integration tests with VectorCore applications
3. Performance benchmarks (ensure GPU speedup maintained)
4. Memory leak tests with buffer pooling

**Regression Testing**:
- All existing VectorAccelerate tests must pass
- No performance degradation on existing workloads
- Verify async/await patterns remain correct

---

## Success Metrics

### Phase 1 Completion (Critical)
- ✅ ComputeProvider conformance test passes
- ✅ Zero distance metric duplication across codebase
- ✅ BufferProvider can be used by VectorCore apps
- ✅ Integration score ≥ 65/100

### Phase 2 Completion (Full Integration)
- ✅ AccelerationProvider capability queries work
- ✅ ComputeDevice used throughout
- ✅ BatchOperations automatically delegates to GPU
- ✅ Integration score ≥ 85/100

### Phase 3 Completion (Excellence)
- ✅ Serialization round-trip tests pass
- ✅ OptimizedVector 10%+ faster than generic implementations
- ✅ Factory methods simplify GPU vector creation
- ✅ Integration score ≥ 95/100

---

## Risk Assessment

### Low Risk
- Distance metric consolidation (well-defined, low complexity)
- ComputeDevice adoption (drop-in replacement)
- VectorFactory conformance (simple additions)

### Medium Risk
- ComputeProvider implementation (complex parallel execution logic)
- BufferProvider conformance (need to reconcile BufferToken/BufferHandle)

### High Risk
- None identified - all changes are additive conformances

### Mitigation Strategies
1. Implement behind feature flags if needed
2. Maintain backward compatibility adapters during transition
3. Extensive testing with real VectorCore applications
4. Gradual rollout with beta testing

---

## Breaking Changes

**None Expected** - All changes are additive protocol conformances. Existing VectorAccelerate APIs remain unchanged.

**Migration Path for Users**:
```swift
// OLD (still works)
let engine = try await ComputeEngine.shared()
let distances = try await engine.computeBatchDistances(...)

// NEW (VectorCore-compatible)
let provider: any ComputeProvider = try await ComputeEngine.shared()
let results = try await provider.parallelExecute(items: 0..<1000) { ... }
```

---

## Appendix A: Protocol Checklist

### VectorCore Protocols Status

- [x] **VectorProtocol** - ✅ Already conforms (via extensions)
- [x] **DistanceProvider** - ✅ Implemented (AcceleratedDistanceProvider)
- [x] **VectorOperationsProvider** - ✅ Implemented (AcceleratedVectorOperations)
- [ ] **ComputeProvider** - ❌ NOT implemented (Phase 1.1)
- [ ] **BufferProvider** - ❌ NOT implemented (Phase 1.3)
- [ ] **AccelerationProvider** - ❌ NOT implemented (Phase 2.1)
- [ ] **VectorSerializable** - ❌ NOT implemented (Phase 3.1)
- [ ] **OptimizedVector** - ❌ NOT implemented (Phase 3.2)
- [ ] **VectorFactory** - ❌ NOT implemented (Phase 3.3)

### Type Alignment Status

- [ ] Use `SupportedDistanceMetric` everywhere - ❌ Custom enums exist (Phase 1.2)
- [ ] Use `ComputeDevice` for device selection - ❌ Custom logic (Phase 2.2)
- [ ] Use `BufferHandle` for buffer management - ⚠️ Custom `BufferToken` (Phase 1.3)
- [x] Use `VectorProtocol` for generic vectors - ✅ Widely used
- [ ] Use `ComputeDeviceInfo` for capabilities - ❌ Custom struct (Phase 2.2)

---

## Appendix B: File Modification Summary

### High-Impact Files (Must Change)
1. `Sources/VectorAccelerate/Core/ComputeEngine.swift` - Add ComputeProvider
2. `Sources/VectorAccelerate/Core/BufferPool.swift` - Add BufferProvider
3. `Sources/VectorAccelerate/Core/Types.swift` - Remove DistanceMetricType
4. `Sources/VectorAccelerate/Operations/BatchProcessor.swift` - Use SupportedDistanceMetric
5. `Sources/VectorAccelerate/Core/MetalContext.swift` - Add AccelerationProvider

### Medium-Impact Files (Should Change)
6. `Sources/VectorAccelerate/ML/EmbeddingEngine.swift` - Use SupportedDistanceMetric
7. `Sources/VectorAccelerate/Core/MetalDevice.swift` - ComputeDevice integration
8. `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift` - Enhanced integration

### New Files to Create
9. `Sources/VectorAccelerate/Vectors/MetalOptimizedVector512.swift`
10. `Sources/VectorAccelerate/Vectors/MetalOptimizedVector768.swift`
11. `Sources/VectorAccelerate/Vectors/MetalOptimizedVector1536.swift`
12. `Sources/VectorAccelerate/Serialization/VectorSerialization.swift`

---

## Appendix C: Dependencies and Imports

**Required VectorCore Imports**:
```swift
// Add to Package.swift
.product(name: "VectorCore", package: "VectorCore"),

// Add to affected source files
import VectorCore
```

**Minimum VectorCore Version**: 0.1.4

---

## Contact and Support

**Questions?** Reference the comprehensive VectorCore API analysis in this repository.

**Blockers?** Check VectorCore's documentation at `../../VectorCore/README.md`

**Integration Issues?** Review the integration analysis report for detailed gap descriptions.

---

**End of Roadmap**
