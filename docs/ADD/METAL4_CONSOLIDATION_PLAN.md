# Metal4 Architecture Consolidation Plan

**Status:** ✅ COMPLETED
**Created:** 2025-12-15
**Completed:** 2025-12-15
**Target:** Unified Metal4-only architecture

## Executive Summary

This plan consolidates three remaining legacy/duplicate patterns to achieve a fully unified Metal4 architecture:

1. **BufferPool Unification** - Merge `SmartBufferPool` into `BufferPool`
2. **Legacy Kernel Migration** - Migrate `LearnedDistanceKernel` and `QuantizationStatisticsKernel` to Metal4
3. **KernelContext Evaluation** - Keep as synchronous fallback utility

---

## 1. BufferPool Unification

### Current State

Two buffer pool implementations exist side-by-side:

| Aspect | BufferPool | SmartBufferPool |
|--------|-----------|-----------------|
| **Location** | `Core/BufferPool.swift` | `Core/SmartBufferPool.swift` |
| **Lines** | ~515 | ~283 |
| **Return Type** | `BufferToken` (RAII) | `MetalBuffer` (manual) |
| **Auto-release** | Yes (deinit) | No |
| **Used By** | `Metal4Context` | `BatchDistanceEngine` |
| **Bucket Sizes** | `MetalBufferFactory.standardBucketSizes` | Custom `commonSizes` |
| **Pre-allocation** | No | Yes (on init) |
| **In-use Tracking** | Explicit `Set<ObjectIdentifier>` | None |
| **Configuration** | `BufferPoolConfiguration` struct | `maxMemoryMB: Int` only |

### Decision: Keep BufferPool, Remove SmartBufferPool

**Rationale:**
- BufferPool's RAII token pattern (`BufferToken`) prevents memory leaks
- BufferPool has richer statistics and better tracking
- BufferPool is already used by `Metal4Context`
- SmartBufferPool's pre-allocation feature can be added to BufferPool

### Migration Steps

#### 1.1 Add Pre-allocation to BufferPool

Add SmartBufferPool's pre-allocation feature to BufferPool:

```swift
// Add to BufferPool
public func preallocateCommonSizes() async {
    let commonSizes = [4096, 16384, 65536, 262144, 1048576, 4194304]
    for size in commonSizes.prefix(3) {
        for _ in 0..<2 {
            if let buffer = factory.createBuffer(length: size) {
                var bucket = buckets[size] ?? BufferBucket(size: size)
                bucket.available.append(buffer)
                buckets[size] = bucket
                currentMemoryUsage += size
                allocationCount += 1
            }
        }
    }
}
```

#### 1.2 Add Convenience Methods

Add SmartBufferPool's typed acquire methods to BufferPool:

```swift
// Add to BufferPool
public func getBuffer<T>(for type: T.Type, count: Int) async throws -> BufferToken {
    let byteSize = count * MemoryLayout<T>.stride
    return try await getBuffer(size: byteSize)
}

public func getBuffer<T>(with data: [T]) async throws -> BufferToken {
    let token = try await getBuffer(for: T.self, count: data.count)
    token.write(data: data)
    return token
}
```

#### 1.3 Update BatchDistanceEngine

Migrate `BatchDistanceEngine` from `SmartBufferPool` to `BufferPool`:

**File:** `Sources/VectorAccelerate/Operations/BatchDistanceOperations.swift`

```swift
// Before
private let bufferPool: SmartBufferPool
self.bufferPool = SmartBufferPool(device: context.device)
let buffer = try await bufferPool.acquire(byteSize: size)
await bufferPool.release(buffer)

// After
private let bufferPool: BufferPool
self.bufferPool = BufferPool(device: context.device)
let token = try await bufferPool.getBuffer(size: size)
// Auto-release via RAII when token goes out of scope
```

#### 1.4 Delete SmartBufferPool

After migration:
- Delete `Sources/VectorAccelerate/Core/SmartBufferPool.swift`
- Remove any remaining imports/references

#### 1.5 Update Tests

Migrate any `SmartBufferPool` tests to use `BufferPool`.

### Files Changed

| File | Action |
|------|--------|
| `Core/BufferPool.swift` | ADD pre-allocation, typed methods |
| `Operations/BatchDistanceOperations.swift` | UPDATE to use BufferPool |
| `Core/SmartBufferPool.swift` | DELETE |
| Tests using SmartBufferPool | UPDATE |

---

## 2. Legacy Kernel Migration

### Current State

Two kernels exist outside `Kernels/Metal4/` and use the older `KernelContext` pattern:

| Kernel | Location | Uses |
|--------|----------|------|
| `LearnedDistanceKernel` | `Kernels/LearnedDistanceKernel.swift` | `KernelContext` |
| `QuantizationStatisticsKernel` | `Kernels/QuantizationStatisticsKernel.swift` | `KernelContext` |

These don't conform to `Metal4Kernel` protocol.

### Decision: Migrate Both to Metal4Kernel

**Rationale:**
- Consistent architecture with all other kernels
- Access to Metal4 features (explicit residency, barriers, etc.)
- Unified initialization pattern via `Metal4Context`

### Migration Pattern

Transform from:
```swift
// OLD: KernelContext pattern
public final class LegacyKernel: @unchecked Sendable {
    private let device: any MTLDevice
    private let kernelContext: KernelContext

    public init(device: any MTLDevice) throws {
        self.device = device
        self.kernelContext = try KernelContext.shared(for: device)
        let library = try KernelContext.getSharedLibrary(for: device)
        // ... create pipelines
    }
}
```

To:
```swift
// NEW: Metal4Kernel pattern
public final class Metal4Kernel: @unchecked Sendable, Metal4Kernel {
    private let context: Metal4Context

    public init(context: Metal4Context) async throws {
        self.context = context
        let pipeline = try await context.getPipeline(functionName: "kernel_function")
        // ... store pipelines
    }
}
```

### 2.1 LearnedDistanceKernel Migration

**Current Usage:** Used by `LearnedDistanceService`

```swift
// LearnedDistanceService.swift line 101
self.learnedKernel = try LearnedDistanceKernel(device: context.device.rawDevice)
```

**Migration Steps:**

1. Move file: `Kernels/LearnedDistanceKernel.swift` → `Kernels/Metal4/LearnedDistanceKernel.swift`

2. Update class signature:
```swift
public final class LearnedDistanceKernel: @unchecked Sendable, Metal4Kernel {
    private let context: Metal4Context

    public init(context: Metal4Context) async throws {
        self.context = context
        // Migrate pipeline creation to use context.getPipeline()
    }
}
```

3. Update `LearnedDistanceService`:
```swift
// Before
self.learnedKernel = try LearnedDistanceKernel(device: context.device.rawDevice)

// After
self.learnedKernel = try await LearnedDistanceKernel(context: context)
```

### 2.2 QuantizationStatisticsKernel Migration

**Current Usage:** Appears to be standalone/utility kernel

**Migration Steps:**

1. Move file: `Kernels/QuantizationStatisticsKernel.swift` → `Kernels/Metal4/QuantizationStatisticsKernel.swift`

2. Update class signature:
```swift
public final class QuantizationStatisticsKernel: @unchecked Sendable, Metal4Kernel {
    private let context: Metal4Context

    public init(context: Metal4Context) async throws {
        self.context = context
        let pipeline = try await context.getPipeline(functionName: "quantization_statistics")
        // ...
    }
}
```

3. Update any usages to pass `Metal4Context` instead of raw `MTLDevice`

### Files Changed

| File | Action |
|------|--------|
| `Kernels/LearnedDistanceKernel.swift` | MOVE to Metal4/, UPDATE to Metal4Kernel |
| `Kernels/QuantizationStatisticsKernel.swift` | MOVE to Metal4/, UPDATE to Metal4Kernel |
| `Core/LearnedDistanceService.swift` | UPDATE kernel initialization |
| Any other usages | UPDATE to use Metal4Context |

---

## 3. KernelContext Evaluation

### Current State

`KernelContext` is a lightweight synchronous wrapper providing:
- Synchronous `MTLDevice` and `MTLLibrary` access
- Shared instance caching per device
- Runtime shader compilation fallback

**Used By:**
- `LearnedDistanceKernel` (will be removed after migration)
- `QuantizationStatisticsKernel` (will be removed after migration)
- `Metal4ShaderCompiler` (as fallback for library loading)

### Decision: KEEP as Internal Utility

**Rationale:**
- After kernel migrations, only `Metal4ShaderCompiler` will use it
- It provides valuable fallback for SPM/runtime shader compilation
- It's lightweight (150 lines) and serves a valid purpose
- No external API exposure needed

### Changes

1. **Mark as internal** (not public) after kernel migrations
2. **Document purpose** as Metal4ShaderCompiler utility
3. **No removal** - keep for fallback functionality

### Files Changed

| File | Action |
|------|--------|
| `Core/KernelContext.swift` | MARK methods `internal` where possible |

---

## Implementation Order

### Recommended Sequence

```
Phase 1: BufferPool Unification
├── 1.1 Add pre-allocation to BufferPool
├── 1.2 Add convenience methods to BufferPool
├── 1.3 Update BatchDistanceEngine to use BufferPool
├── 1.4 Delete SmartBufferPool
├── 1.5 Update tests
└── 1.6 Verify build & tests pass

Phase 2: Legacy Kernel Migration
├── 2.1 Migrate LearnedDistanceKernel to Metal4
├── 2.2 Update LearnedDistanceService
├── 2.3 Migrate QuantizationStatisticsKernel to Metal4
├── 2.4 Update any other usages
└── 2.5 Verify build & tests pass

Phase 3: KernelContext Cleanup
├── 3.1 Verify KernelContext only used by Metal4ShaderCompiler
├── 3.2 Mark internal methods appropriately
└── 3.3 Add documentation comments
```

---

## Validation Checklist

### After Phase 1 (BufferPool)
- [ ] `swift build` succeeds
- [ ] No references to `SmartBufferPool` in codebase
- [ ] `BufferPool` has pre-allocation support
- [ ] `BatchDistanceEngine` uses `BufferPool`
- [ ] Tests pass

### After Phase 2 (Kernels)
- [ ] `swift build` succeeds
- [ ] No kernels outside `Kernels/Metal4/` directory
- [ ] All kernels conform to `Metal4Kernel` protocol
- [ ] `LearnedDistanceService` works with migrated kernel
- [ ] Tests pass

### After Phase 3 (KernelContext)
- [ ] `KernelContext` only used by `Metal4ShaderCompiler`
- [ ] Build succeeds
- [ ] All tests pass

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| BufferPool API breaks BatchDistanceEngine | Low | Medium | RAII pattern should be transparent |
| Kernel migration breaks async initialization | Medium | Medium | Careful testing of LearnedDistanceService |
| KernelContext removal breaks shader loading | Low | High | Keep KernelContext, don't remove |
| Performance regression in buffer management | Low | Medium | Benchmark before/after |

---

## Files Summary

### To Modify
- `Core/BufferPool.swift` - Add features
- `Operations/BatchDistanceOperations.swift` - Use BufferPool
- `Kernels/LearnedDistanceKernel.swift` - Migrate to Metal4
- `Kernels/QuantizationStatisticsKernel.swift` - Migrate to Metal4
- `Core/LearnedDistanceService.swift` - Update kernel init
- `Core/KernelContext.swift` - Mark internal

### To Delete
- `Core/SmartBufferPool.swift`

### To Move
- `Kernels/LearnedDistanceKernel.swift` → `Kernels/Metal4/`
- `Kernels/QuantizationStatisticsKernel.swift` → `Kernels/Metal4/`
