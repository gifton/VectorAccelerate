# Metal4-Only Migration Plan

**Status:** ✅ COMPLETED
**Created:** 2025-12-15
**Completed:** 2025-12-15
**Target:** iOS 26+ / macOS 26+ / tvOS 26+ / visionOS 3+

## Executive Summary

This document outlines the complete removal of `MetalContext` and migration to `Metal4Context` as the sole GPU compute context. This is a breaking change that simplifies the codebase, eliminates legacy shader loading issues, and provides a unified Metal 4 API.

### Goals
- Remove all legacy `MetalContext` code
- Migrate all components to `Metal4Context`
- Add `AccelerationProvider` conformance to `Metal4Context`
- Update minimum deployment targets to Metal 4 requirements
- Clean up redundant code paths

### Non-Goals
- Backwards compatibility with older OS versions
- Deprecation warnings or migration period
- Support for non-Metal 4 devices

---

## Phase 1: Foundation Changes

### 1.1 Update Package Deployment Targets

**File:** `Package.swift`

Update all platform minimum versions:
```swift
platforms: [
    .macOS(.v26),
    .iOS(.v26),
    .tvOS(.v26),
    .visionOS(.v3)
]
```

### 1.2 Add AccelerationProvider Conformance to Metal4Context

**File:** `Sources/VectorAccelerate/Core/Metal4Context.swift`

Add protocol conformance:
```swift
extension Metal4Context: AccelerationProvider {
    public typealias Config = Metal4Configuration

    public nonisolated func isSupported(for operation: AcceleratedOperation) -> Bool {
        // All operations supported on Metal 4
        switch operation {
        case .distanceComputation, .matrixMultiplication,
             .vectorNormalization, .batchedOperations:
            return true
        }
    }

    public nonisolated func accelerate<T>(_ operation: AcceleratedOperation, input: T) async throws -> T {
        // Route to appropriate Metal 4 kernels
        // This is a compatibility shim - prefer direct kernel usage
        return input
    }
}
```

### 1.3 Remove Metal4Context Metal 4 Verification

Since we're Metal 4 only, remove the runtime check that throws:
```swift
// REMOVE this check - if they're on iOS 26+, Metal 4 is guaranteed
guard capabilities.supportsMetal4Core else {
    throw VectorError.metal4NotSupported(reason: "Device does not support Metal 4")
}
```

---

## Phase 2: Component Migration

### 2.1 MatrixEngine Migration

**Current State:** Uses `MetalContext`, embedded shader strings, legacy loading
**Target State:** Uses `Metal4Context`, existing Metal4 kernels

**File:** `Sources/VectorAccelerate/Operations/MatrixEngine.swift`

**Changes:**
1. Replace `MetalContext` with `Metal4Context`
2. Remove `precompileShaders()` and embedded shader code
3. Use `MatrixMultiplyKernel` and `MatrixTransposeKernel`
4. Remove legacy shader cache variables

**New Implementation Structure:**
```swift
public actor MatrixEngine {
    private let context: Metal4Context
    private let configuration: MatrixConfiguration
    private let logger: Logger

    // Metal 4 Kernels (lazy initialized)
    private var matrixMultiplyKernel: MatrixMultiplyKernel?
    private var matrixTransposeKernel: MatrixTransposeKernel?
    private var batchMatrixKernel: BatchMatrixKernel?

    public init(
        context: Metal4Context,
        configuration: MatrixConfiguration = .default
    ) async {
        self.context = context
        self.configuration = configuration
        self.logger = Logger.shared
    }

    // Lazy kernel initialization
    private func getMatrixMultiplyKernel() async throws -> MatrixMultiplyKernel {
        if let kernel = matrixMultiplyKernel {
            return kernel
        }
        let kernel = try await MatrixMultiplyKernel(context: context)
        matrixMultiplyKernel = kernel
        return kernel
    }

    // multiply() uses kernel instead of embedded shader
    // transpose() uses kernel instead of embedded shader
}
```

### 2.2 ComputeEngine Removal/Migration

**Current State:** `ComputeEngine` uses `MetalContext`
**Target State:** Remove `ComputeEngine`, use `Metal4ComputeEngine` only

**Option A: Remove ComputeEngine entirely**
- Delete `Sources/VectorAccelerate/Core/ComputeEngine.swift`
- Update all references to use `Metal4ComputeEngine`
- Simpler, cleaner

**Option B: Rename Metal4ComputeEngine to ComputeEngine**
- Delete old `ComputeEngine.swift`
- Rename `Metal4ComputeEngine.swift` to `ComputeEngine.swift`
- Update class name
- Less disruptive to existing code

**Recommended:** Option A (full removal) - cleaner separation

**Files to update:**
- `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift` - use Metal4ComputeEngine

### 2.3 BatchProcessor Migration

**File:** `Sources/VectorAccelerate/Operations/BatchProcessor.swift`

**Changes:**
1. Replace `MetalContext?` with `Metal4Context?`
2. Update initialization
3. Use Metal 4 kernels for batch operations

```swift
public actor BatchProcessor {
    private let context: Metal4Context?
    // ...

    public init(
        context: Metal4Context? = nil,
        configuration: Configuration = .default
    ) async {
        self.context = context
        // ...
    }
}
```

### 2.4 BatchDistanceOperations Migration

**File:** `Sources/VectorAccelerate/Operations/BatchDistanceOperations.swift`

**Changes:**
1. Replace `MetalContext` with `Metal4Context`
2. Use Metal 4 distance kernels (L2DistanceKernel, CosineSimilarityKernel, etc.)

```swift
public actor BatchDistanceOperations {
    private let metalContext: Metal4Context
    private let l2Kernel: L2DistanceKernel
    private let cosineKernel: CosineSimilarityKernel

    public init(metalContext: Metal4Context) async throws {
        self.metalContext = metalContext
        self.l2Kernel = try await L2DistanceKernel(context: metalContext)
        self.cosineKernel = try await CosineSimilarityKernel(context: metalContext)
    }
}
```

### 2.5 QuantizationEngine Migration

**File:** `Sources/VectorAccelerate/ML/QuantizationEngine.swift`

**Changes:**
1. Replace `MetalContext?` with `Metal4Context?`
2. Use `ScalarQuantizationKernel` and `BinaryQuantizationKernel`

```swift
public actor QuantizationEngine {
    private let context: Metal4Context?
    private var scalarKernel: ScalarQuantizationKernel?
    private var binaryKernel: BinaryQuantizationKernel?

    public init(
        configuration: QuantizationConfiguration = .default,
        context: Metal4Context? = nil
    ) async {
        self.context = context
        // Lazy kernel initialization
    }
}
```

### 2.6 VectorCoreIntegration Migration

**File:** `Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift`

**Changes:**
1. Replace all `MetalContext` references with `Metal4Context`
2. Update `MetalDistanceProvider`, `GPUEmbeddingProcessor`, `MetalAccelerator`

```swift
public actor MetalDistanceProvider: DistanceProvider {
    private let context: Metal4Context
    private let l2Kernel: L2DistanceKernel

    public init(context: Metal4Context) async throws {
        self.context = context
        self.l2Kernel = try await L2DistanceKernel(context: context)
    }
}

public actor GPUEmbeddingProcessor: EmbeddingProcessor {
    private let context: Metal4Context
    // ...
}

public actor MetalAccelerator: AccelerationProvider {
    private let context: Metal4Context
    // ...
}
```

### 2.7 SharedMetalContext Migration

**File:** `Sources/VectorAccelerate/Core/SharedMetalContext.swift`

**Decision Point:** This file manages shared MetalContext instances.

**Options:**
1. **Remove entirely** - If Metal4Context should always be created fresh
2. **Migrate** - Update to manage Metal4Context instances instead

**Recommended:** Option 2 - Migrate to manage Metal4Context

```swift
// Rename to SharedMetal4Context or keep name
public actor SharedMetalContext {
    private var contexts: [String: Metal4Context] = [:]

    func register(context: Metal4Context, configuration: SharedConfiguration) {
        // ...
    }
}
```

### 2.8 BenchmarkFramework Migration

**File:** `Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift`

**Changes:**
1. Replace `MetalContext` with `Metal4Context`
2. Replace `ComputeEngine` with `Metal4ComputeEngine`

```swift
public actor BenchmarkRunner {
    private let context: Metal4Context

    public init(engine: Metal4ComputeEngine, context: Metal4Context) {
        self.context = context
        // ...
    }
}
```

---

## Phase 3: File Deletions

### 3.1 Files to Delete

| File | Reason |
|------|--------|
| `Sources/VectorAccelerate/Core/MetalContext.swift` | Replaced by Metal4Context |
| `Sources/VectorAccelerate/Core/ComputeEngine.swift` | Replaced by Metal4ComputeEngine |
| `Sources/VectorAccelerate/Core/ShaderManager.swift` | Legacy shader loading, replaced by Metal4ShaderCompiler |

**Note on ShaderManager:** This class has the same `getDefaultLibrary()` issue as MetalContext.
Metal4Context provides `shaderCompiler: Metal4ShaderCompiler` which uses `KernelContext` for
proper runtime shader compilation from `.metal` files. All usages should migrate to use
`context.shaderCompiler` instead.

### 3.2 Metal4ComputeEngine ShaderManager Removal

**File:** `Sources/VectorAccelerate/Core/Metal4ComputeEngine.swift`

Remove ShaderManager dependency:
```swift
// REMOVE:
private let shaderManager: ShaderManager

// In init, REMOVE:
self.shaderManager = try await ShaderManager(device: context.device)

// REPLACE usages with:
context.shaderCompiler
```

### 3.3 Files to Evaluate for Deletion

| File | Evaluation |
|------|------------|
| `Sources/VectorAccelerate/Core/SharedMetalContext.swift` | Keep if useful, migrate to Metal4Context |

---

## Phase 4: Test Updates

### 4.1 Tests to Delete

| Test File | Reason |
|-----------|--------|
| `Tests/VectorAccelerateTests/MetalContextTests.swift` | Tests deleted class |
| `Tests/VectorAccelerateTests/ComputeEngineTests.swift` | Tests deleted class |

### 4.2 Tests to Migrate

| Test File | Changes Needed |
|-----------|----------------|
| `MatrixEngineTests.swift` | Use Metal4Context |
| `MatrixEngineEnhancedTests.swift` | Use Metal4Context |
| `BatchProcessorTests.swift` | Use Metal4Context |
| `QuantizationEngineTests.swift` | Use Metal4Context |
| `QuantizationEngineEnhancedTests.swift` | Use Metal4Context |
| `DistanceMetricsTests.swift` | Use Metal4Context |
| `Phase2Tests.swift` | Use Metal4Context |
| `Phase3Tests.swift` | Use Metal4Context |
| `Priority2IntegrationTests.swift` | Use Metal4Context |

### 4.3 Test Migration Pattern

Replace:
```swift
// Old
let context = try await MetalContext()
let engine = try await ComputeEngine(context: context)
```

With:
```swift
// New
let context = try await Metal4Context()
let engine = try await Metal4ComputeEngine(context: context)
```

---

## Phase 5: Documentation Updates

### 5.1 Files to Update

| File | Changes |
|------|---------|
| `README.md` | Update requirements, examples |
| `docs/metal4_migration_guide.md` | Mark as complete/archived |
| `docs/KERNEL_API_EXPOSURE_PLAN.md` | Update if references MetalContext |
| `RELEASE_NOTES_0.3.0.md` | Document breaking changes |

### 5.2 New Documentation Needed

- Update all code examples to use Metal4Context
- Document iOS 26+ requirement prominently
- Update API reference documentation

---

## Phase 6: Cleanup

### 6.1 Remove Unused Imports

After migration, scan for unused VectorCore imports and clean up.

### 6.2 Remove Dead Code

Look for any remaining references to:
- `MetalContext`
- `MetalConfiguration` (replaced by `Metal4Configuration`)
- `ComputeEngine` (replaced by `Metal4ComputeEngine`)
- `ShaderManager` (replaced by `Metal4ShaderCompiler`)

### 6.3 Consolidate Configuration Types

Consider consolidating:
- `Metal4Configuration` → rename to `MetalConfiguration`?
- `Metal4ComputeEngineConfiguration` → rename to `ComputeEngineConfiguration`?

---

## Implementation Order

### Recommended Execution Sequence

```
1. Package.swift platform updates
2. Metal4Context AccelerationProvider conformance
3. Remove Metal 4 capability check
4. Migrate MatrixEngine (fixes the warning!)
5. Migrate BatchProcessor
6. Migrate BatchDistanceOperations
7. Migrate QuantizationEngine
8. Migrate VectorCoreIntegration
9. Migrate SharedMetalContext
10. Migrate BenchmarkFramework
11. Delete MetalContext.swift
12. Delete ComputeEngine.swift
13. Delete ShaderManager.swift (if no longer needed)
14. Update/delete tests
15. Documentation updates
16. Final cleanup pass
```

### Estimated Effort

| Phase | Estimated Time | Risk |
|-------|---------------|------|
| Phase 1: Foundation | 30 min | Low |
| Phase 2: Component Migration | 2-3 hours | Medium |
| Phase 3: File Deletions | 15 min | Low |
| Phase 4: Test Updates | 1-2 hours | Medium |
| Phase 5: Documentation | 30 min | Low |
| Phase 6: Cleanup | 30 min | Low |
| **Total** | **5-7 hours** | - |

---

## Validation Checklist

### After Each Phase

- [ ] `swift build` succeeds
- [ ] No warnings about MetalContext
- [ ] `swift test` passes

### Final Validation

- [ ] `swift build` succeeds with no warnings
- [ ] `swift test` all tests pass
- [ ] No references to MetalContext in codebase (except docs)
- [ ] No references to ComputeEngine (only Metal4ComputeEngine)
- [ ] README examples work
- [ ] Benchmarks run successfully

---

## Rollback Plan

If issues arise during migration:

1. Git branch: Create `gifton/metal4-only-migration` before starting
2. Incremental commits: Commit after each phase
3. Rollback: `git checkout main` if needed

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missing Metal4 kernel for legacy operation | Low | Medium | Audit all operations before starting |
| Test failures due to API changes | Medium | Low | Fix tests incrementally |
| Performance regression | Low | Medium | Run benchmarks before/after |
| Protocol conformance issues | Low | Medium | Test AccelerationProvider usage |

---

## Success Criteria

1. **Zero MetalContext references** in source code
2. **All tests pass** with Metal4Context only
3. **No legacy shader warnings** during test execution
4. **Clean build** with no deprecation warnings
5. **Benchmarks functional** with Metal4ComputeEngine

---

## Appendix A: Full File Inventory

### Files Requiring Changes

```
Sources/VectorAccelerate/Core/Metal4Context.swift           [ADD AccelerationProvider]
Sources/VectorAccelerate/Operations/MatrixEngine.swift      [MIGRATE]
Sources/VectorAccelerate/Operations/BatchProcessor.swift    [MIGRATE]
Sources/VectorAccelerate/Operations/BatchDistanceOperations.swift [MIGRATE]
Sources/VectorAccelerate/ML/QuantizationEngine.swift        [MIGRATE]
Sources/VectorAccelerate/Integration/VectorCoreIntegration.swift [MIGRATE]
Sources/VectorAccelerate/Core/SharedMetalContext.swift      [MIGRATE or DELETE]
Sources/VectorAccelerate/Benchmarking/BenchmarkFramework.swift [MIGRATE]
Sources/VectorAccelerate/Core/Metal4ComputeEngine.swift      [UPDATE - remove ShaderManager]
Package.swift                                                [UPDATE platforms]
```

### Files to Delete

```
Sources/VectorAccelerate/Core/MetalContext.swift            [DELETE]
Sources/VectorAccelerate/Core/ComputeEngine.swift           [DELETE]
Sources/VectorAccelerate/Core/ShaderManager.swift           [DELETE]
Tests/VectorAccelerateTests/MetalContextTests.swift         [DELETE]
Tests/VectorAccelerateTests/ComputeEngineTests.swift        [DELETE]
```

### Tests to Update

```
Tests/VectorAccelerateTests/MatrixEngineTests.swift
Tests/VectorAccelerateTests/MatrixEngineEnhancedTests.swift
Tests/VectorAccelerateTests/BatchProcessorTests.swift
Tests/VectorAccelerateTests/QuantizationEngineTests.swift
Tests/VectorAccelerateTests/QuantizationEngineEnhancedTests.swift
Tests/VectorAccelerateTests/DistanceMetricsTests.swift
Tests/VectorAccelerateTests/Phase2Tests.swift
Tests/VectorAccelerateTests/Phase3Tests.swift
Tests/VectorAccelerateTests/Priority2IntegrationTests.swift
```

---

## Appendix B: AccelerationProvider Protocol Reference

From VectorCore:

```swift
public protocol AccelerationProvider: Sendable {
    associatedtype Config: Sendable

    func isSupported(for operation: AcceleratedOperation) -> Bool
    func accelerate<T>(_ operation: AcceleratedOperation, input: T) async throws -> T
}

public enum AcceleratedOperation: Sendable {
    case distanceComputation
    case matrixMultiplication
    case vectorNormalization
    case batchedOperations
}
```
