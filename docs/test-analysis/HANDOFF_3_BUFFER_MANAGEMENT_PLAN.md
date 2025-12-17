# Buffer Management Optimization for Neural Decode Path

## Problem Statement

The Neural Quantization decode kernel achieves ~3M vec/s GPU throughput, but end-to-end throughput is only ~13k vec/s due to CPU-side overhead:

| Component | Time (1000 vectors) | Impact |
|-----------|---------------------|--------|
| GPU kernel execution | ~0.3ms | Minimal |
| Buffer allocation (3x) | ~1-2ms | Medium |
| Result extraction (nested loops) | ~50ms | **Critical** |
| **Total** | ~52ms | |

**Target**: Improve end-to-end throughput from ~13k to >50k vec/s.

## Research Findings

### Usage Pattern Analysis
- **100% of test usage** treats decoded results as CPU-side `[[Float]]` arrays
- No GPU pipeline chaining (decoded results not fed to other GPU kernels)
- Results used for: validation, metrics computation, numerical comparison
- **Conclusion**: Keep array-first API, optimize the implementation

### Available Infrastructure
- `BufferPool` exists with token-based RAII, bucket-based reuse
- `Metal4Context.getBuffer(size:)` provides pooled buffers with residency management
- Other kernels use this pattern successfully (L2DistanceKernel, MatrixEngine)

## Implementation Plan

### Phase 1: Use BufferPool for Temporary Allocations (Low Risk)

**File**: `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`

Replace direct `device.makeBuffer()` calls with pooled buffers:

```swift
// BEFORE (decode method, lines 939-971):
guard let inputBuffer = device.makeBuffer(bytes: ...) else { ... }
guard let scaleBuffer = device.makeBuffer(length: ...) else { ... }
guard let outputBuffer = device.makeBuffer(length: ...) else { ... }

// AFTER:
let inputToken = try await context.getBuffer(for: [UInt8](encoded.latentCodes))
let scaleToken = try await context.getBuffer(size: numVectors * MemoryLayout<Float>.size)
let outputToken = try await context.getBuffer(size: numVectors * inputDim * MemoryLayout<Float>.size)
// Tokens auto-return to pool when scope exits (RAII)
```

**Changes**:
1. Store `context` reference in kernel (currently only stores `context.device`)
2. Update `decode()` to use `context.getBuffer()` instead of `device.makeBuffer()`
3. Update `encode()` similarly
4. Buffers auto-return to pool via BufferToken RAII

**Expected Impact**: ~1-2ms savings per call (buffer allocation overhead eliminated)

### Phase 2: Optimize Scale Buffer Initialization (Low Risk)

**File**: `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`

Current code writes same scale value N times in a loop:

```swift
// BEFORE (lines 958-961):
let scalePtr = scaleBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
for i in 0..<numVectors {
    scalePtr[i] = encoded.scale  // Same value repeated
}

// AFTER - use memset-style initialization:
let scalePtr = scaleBuffer.contents().bindMemory(to: Float.self, capacity: numVectors)
let scaleValue = encoded.scale
withUnsafeBytes(of: scaleValue) { srcBytes in
    for i in 0..<numVectors {
        scalePtr[i] = scaleValue
    }
}
// Or use vDSP_vfill for vectorized fill
```

**Better approach**: Use Accelerate framework:
```swift
import Accelerate
var fillValue = encoded.scale
vDSP_vfill(&fillValue, scalePtr, 1, vDSP_Length(numVectors))
```

**Expected Impact**: Minimal (~0.1ms), but cleaner code.

### Phase 3: Optimize Result Extraction (Critical - High Impact)

**File**: `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`

Current nested loop is the primary bottleneck:

```swift
// BEFORE (lines 989-1001):
var results: [[Float]] = []
results.reserveCapacity(numVectors)
for i in 0..<numVectors {
    var row: [Float] = []
    row.reserveCapacity(inputDim)
    for j in 0..<inputDim {
        row.append(outputPtr[i * inputDim + j])
    }
    results.append(row)
}
```

**Problem**: Creates N separate array allocations, element-by-element copy.

**Solution**: Use `Array(UnsafeBufferPointer)` for bulk initialization:

```swift
// AFTER:
let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: numVectors * inputDim)
var results: [[Float]] = []
results.reserveCapacity(numVectors)

for i in 0..<numVectors {
    let rowStart = outputPtr.advanced(by: i * inputDim)
    let row = Array(UnsafeBufferPointer(start: rowStart, count: inputDim))
    results.append(row)
}
```

**Why this is faster**:
- `Array(UnsafeBufferPointer)` uses optimized memcpy internally
- Single allocation per row vs element-by-element append
- Better cache locality

**Expected Impact**: 5-10x faster extraction (~50ms → ~5-10ms)

### Phase 4: Add Flat Array Return Option (Medium Risk)

**File**: `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`

Add alternative API that returns flat array (avoids nested array overhead entirely):

```swift
/// Decode to flat contiguous array [v0d0, v0d1, ..., v0dN, v1d0, v1d1, ...]
public func decodeFlat(_ encoded: Metal4NeuralEncodingResult) async throws -> [Float] {
    // ... same buffer setup ...

    // Single bulk copy instead of nested loops
    let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: numVectors * inputDim)
    return Array(UnsafeBufferPointer(start: outputPtr, count: numVectors * inputDim))
}
```

**Use case**: When caller will process sequentially or reshape themselves.

**Expected Impact**: ~10-20x faster than nested array version.

### Phase 5: Encode Path Optimization (Low Risk)

**File**: `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift`

Apply same optimizations to `encode()` method:
1. Use BufferPool for input/output/scale buffers
2. Optimize `flatMap` with preallocated array
3. Use bulk copy for Data extraction

```swift
// BEFORE (line 912):
let latentCodes = Data(bytes: outputBuffer.contents(), count: outputSize)

// AFTER (already efficient - Data init from pointer is optimized)
// No change needed here
```

### Phase 6: Add Benchmark Tests (Low Risk)

**File**: `Tests/VectorAccelerateTests/MLIntegrationBenchmarkTests.swift`

Add tests to verify improvements:

```swift
func testDecodeExtractionPerformance() async throws {
    // Measure time breakdown: buffer alloc vs GPU vs extraction
    // Compare old vs new extraction method
    // Verify numerical correctness
}

func testDecodeFlatPerformance() async throws {
    // Benchmark decodeFlat() vs decode()
    // Verify same numerical results
}
```

## Critical Files

| File | Changes |
|------|---------|
| `Sources/VectorAccelerate/Kernels/Metal4/NeuralQuantizationKernel.swift` | BufferPool integration, extraction optimization |
| `Tests/VectorAccelerateTests/MLIntegrationBenchmarkTests.swift` | Benchmark tests |
| `Tests/VectorAccelerateTests/NeuralQuantizationTests.swift` | Update tensor count expectations if needed |

## Implementation Order

1. **Phase 3 first** - Extraction optimization (highest impact, lowest risk)
2. **Phase 1 second** - BufferPool integration (medium impact, low risk)
3. **Phase 4 third** - Flat array API (optional, for advanced users)
4. **Phase 2** - Scale init optimization (minor)
5. **Phase 5** - Encode path (completeness)
6. **Phase 6** - Benchmarks (verification)

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| End-to-end decode throughput | ~13k vec/s | >50k vec/s |
| Extraction time (1000 vectors) | ~50ms | <10ms |
| Buffer allocation time | ~1-2ms | ~0ms (pooled) |
| API compatibility | - | 100% backwards compatible |

## Backwards Compatibility

- Existing `decode()` API signature unchanged
- Existing `encode()` API signature unchanged
- New `decodeFlat()` is additive (optional)
- All existing tests should pass without modification (except tensor count if Phase 1 changes it)

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| Phase 1 (BufferPool) | Low | Well-tested infrastructure, used elsewhere |
| Phase 2 (Scale init) | Very Low | Simple optimization |
| Phase 3 (Extraction) | Low | Same data, different copy method |
| Phase 4 (Flat API) | Low | Additive, no breaking changes |
| Phase 5 (Encode) | Low | Mirror of decode changes |
