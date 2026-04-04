# Metal 4 Lifecycle Management Guide

**Status:** Implemented (Phases 1-3)
**Target:** iOS 26+ / macOS 26+ with UX-first Metal integration

## Overview

VectorAccelerate implements a sophisticated Metal 4 lifecycle management system that ensures GPU operations never block the UI. The system provides:

- **Instant app launch** - No Metal work during initialization
- **Phased warmup** - Critical pipelines first, occasional pipelines during idle
- **Activity awareness** - Pauses warmup during user interaction
- **Thermal respect** - Reduces GPU work when device is hot
- **Binary archive caching** - Near-instant pipeline loading on subsequent launches
- **Graceful fallback** - CPU-based operations when Metal is unavailable

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                             │
│  (JournalEditor, SaveManager, EmbedKit.EmbeddingGenerator)           │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MetalSubsystem                                  │
│  - State machine: dormant → initializing → ready → fullyReady        │
│  - Readiness callbacks                                               │
│  - Archive management                                                │
│  - Fallback coordination                                             │
└─────────────────────────────────────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌────────────────┐  ┌──────────────────────┐
│ PipelineRegistry│  │ WarmupManager  │  │ ArchivePipelineCache │
│ - Critical tier │  │ - Activity     │  │ - Memory cache       │
│ - Occasional    │  │   awareness    │  │ - Binary archive     │
│ - Rare tier     │  │ - Thermal      │  │ - JIT compilation    │
└─────────────────┘  └────────────────┘  └──────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   BinaryArchiveManager                               │
│  - MTLBinaryArchive integration                                      │
│  - Manifest tracking (keys stored in archive)                        │
│  - Graceful corruption handling                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Lifecycle Phases

### Phase A: App Launch (0ms Metal Work)

```swift
// In AppDelegate or App init
let metalSubsystem = MetalSubsystem(configuration: .testing)
// Returns immediately - no Metal work performed
```

**Characteristics:**
- `MetalSubsystem.init()` completes in < 10ms
- State is `.dormant`
- No MTLDevice creation
- No shader compilation
- UI is immediately responsive

### Phase B: Background Initialization

```swift
// After first frame renders
await metalSubsystem.beginBackgroundInitialization()
```

**State Transitions:**
1. `.dormant` → `.initializing` - Metal work begins on background queue
2. `.initializing` → `.deviceReady` - MTLDevice and Metal4Context created
3. `.deviceReady` → `.criticalReady` - Critical pipelines warmed

**Characteristics:**
- Runs on `.utility` priority background queue
- Creates Metal4Context with residency management
- Loads/creates binary archive
- Warms critical pipelines (typically 4-10 pipelines)
- Total time: ~50-200ms depending on archive state

### Phase C: Opportunistic Warmup

```swift
// Automatic after Phase B completes
// Controlled by WarmupManager
```

**State Transition:**
- `.criticalReady` → `.fullyReady` - All occasional pipelines warmed

**Characteristics:**
- Runs one pipeline at a time for granular cancellation
- Pauses on user activity (touch, keyboard, scroll)
- Resumes after `warmupIdleTimeout` seconds of idle (default: 2s)
- Respects thermal state (pauses on `.serious` or `.critical`)
- Saves binary archive on completion

## Pipeline Tiers

Pipelines are categorized into three tiers based on usage patterns:

### Critical Tier
Pipelines that must be ready before first user operation.

```swift
// Default critical pipelines for journaling app
.l2Distance(dimension: 384)
.cosineSimilarity(dimension: 384)
.topK(k: 0)
PipelineCacheKey(operation: "l2_normalize")
```

**Warmup:** Phase B (blocking until ready)

### Occasional Tier
Pipelines used for secondary features.

```swift
// Higher dimension variants
.l2Distance(dimension: 768)
.l2Distance(dimension: 1536)
.cosineSimilarity(dimension: 768)
.dotProduct(dimension: 0)
PipelineCacheKey(operation: "compute_statistics")
```

**Warmup:** Phase C (interruptible, activity-aware)

### Rare Tier
Pipelines that may never run in most sessions.

```swift
// Quantization, matrix ops, advanced ML
PipelineCacheKey(operation: "scalar_quantize_int8")
PipelineCacheKey(operation: "matrix_multiply")
PipelineCacheKey(operation: "attention_similarity")
```

**Warmup:** On-demand only (compiled when first requested)

## Binary Archive Caching

Binary archives store compiled pipeline state objects for near-instant loading.

### First Launch Behavior
1. Archive doesn't exist → created empty
2. Pipelines compile via JIT (~50-200ms each)
3. Compiled pipelines added to archive
4. Archive saved to disk after warmup

### Subsequent Launch Behavior
1. Archive loaded from disk
2. Manifest validates shader source hash and device
3. Pipelines load from archive (~1ms each)
4. Only missing pipelines compiled via JIT

### Archive Location
```
~/Library/Caches/VectorAccelerate/pipelines.metalarchive
~/Library/Caches/VectorAccelerate/pipelines.manifest.json
```

### Invalidation Triggers
- Shader source hash changes
- Device changes (different GPU)
- Corrupted archive file
- Missing manifest

### Three-Tier Lookup
```
┌─────────────────┐
│  Memory Cache   │  ← Fastest (~0ms)
│    (LRU)        │
└────────┬────────┘
         │ miss
         ▼
┌─────────────────┐
│ Binary Archive  │  ← Fast (~1ms)
│ (MTLBinaryArchive)
└────────┬────────┘
         │ miss
         ▼
┌─────────────────┐
│ JIT Compilation │  ← Slow (~50-200ms)
│ (Metal4ShaderCompiler)
└─────────────────┘
```

## Warmup Manager Behavior

### Activity Detection
```swift
// Report user activity (e.g., from gesture recognizer)
await metalSubsystem.reportUserActivity()
```

When activity is reported:
1. Warmup immediately pauses
2. Current pipeline completes (not interrupted mid-compile)
3. Timer starts for `warmupIdleTimeout`
4. Warmup resumes after timeout with no activity

### Thermal State Monitoring

| Thermal State | Warmup Behavior |
|---------------|-----------------|
| `.nominal`    | Normal warmup   |
| `.fair`       | Normal warmup   |
| `.serious`    | Paused          |
| `.critical`   | Paused          |

When thermal throttling ends, warmup automatically resumes.

### Manual Control
```swift
// Pause warmup (e.g., during intensive UI animation)
await metalSubsystem.pauseWarmup()

// Resume warmup
await metalSubsystem.resumeWarmup()

// Check progress (0.0 to 1.0)
let progress = await metalSubsystem.warmupProgress
```

## CPU Fallback

When Metal is unavailable, operations fall back to CPU implementations using Accelerate framework.

### Fallback Triggers
- `MetalSubsystem.state == .failed`
- Individual kernel initialization failure
- No Metal device available

### Available CPU Operations
```swift
let fallback = metalSubsystem.fallback

// Distance operations
let dist = fallback.l2Distance(from: a, to: b)
let sim = fallback.cosineSimilarity(from: a, to: b)

// Batch operations
let distances = fallback.batchL2Distance(from: query, to: candidates)

// Normalization
let normalized = fallback.normalize(vector)

// Top-K selection
let topK = fallback.topKByDistance(distances, k: 10)
```

### Performance Expectations
CPU fallbacks are significantly slower for large batches:

| Operation | GPU | CPU | Ratio |
|-----------|-----|-----|-------|
| 1000 distances (384-dim) | ~2ms | ~50ms | ~25x slower |
| Batch normalize (1000 vectors) | ~1ms | ~10ms | ~10x slower |
| Top-K (k=10, n=10000) | ~0.5ms | ~2ms | ~4x slower |

For single operations or small batches (<100), CPU performance is acceptable.

## Configuration

### Development Configuration
```swift
let config = MetalSubsystemConfiguration.development
// - Runtime compilation: enabled
// - Warmup idle timeout: 1.0s
// - Thermal respect: disabled (faster warmup)
```

### Production Configuration
```swift
let config = MetalSubsystemConfiguration.production(
    metallibURL: Bundle.main.url(forResource: "default", withExtension: "metallib")!,
    archiveURL: cacheDir.appending(path: "pipelines.metalarchive")
)
// - Runtime compilation: disabled (requires precompiled metallib)
// - Uses journaling app pipeline registry
// - Respects thermal state
```

### Testing Configuration
```swift
let config = MetalSubsystemConfiguration.testing
// - Runtime compilation: enabled
// - Minimal pipeline registry
// - Fast warmup timeout (0.1s)
// - Thermal respect: disabled
```

### Custom Configuration
```swift
let config = MetalSubsystemConfiguration(
    allowRuntimeCompilation: false,
    metallibURL: metallibURL,
    binaryArchiveURL: archiveURL,
    pipelineRegistry: .journalingApp,
    backgroundQoS: .utility,
    criticalQoS: .userInitiated,
    warmupIdleTimeout: 2.0,
    respectThermalState: true
)
```

## Usage Examples

### Basic Usage
```swift
// 1. Create subsystem (instant)
let metalSubsystem = MetalSubsystem(configuration: .testing)

// 2. Start background initialization
metalSubsystem.beginBackgroundInitialization()

// 3. Wait for critical pipelines before GPU work
await metalSubsystem.requestCriticalPipelines()

// 4. Use context for GPU operations
if let context = await metalSubsystem.context {
    let pipeline = try await context.getPipeline(for: .l2Distance(dimension: 384))
    // ... use pipeline
} else {
    // Use CPU fallback
    let distance = metalSubsystem.fallback.l2Distance(from: a, to: b)
}
```

### Observing State Changes
```swift
let observerId = await metalSubsystem.addReadinessObserver { state in
    switch state {
    case .dormant:
        print("Metal not started")
    case .initializing:
        print("Metal initializing...")
    case .deviceReady:
        print("Device ready, warming pipelines...")
    case .criticalReady:
        print("Ready for primary operations")
    case .fullyReady:
        print("All pipelines warmed")
    case .failed(let error):
        print("Metal failed: \(error)")
    }
}

// Remove observer when done
await metalSubsystem.removeReadinessObserver(observerId)
```

### Nonisolated Checks
```swift
// Thread-safe, non-blocking checks (no await needed)
if metalSubsystem.isMetalAvailable {
    // Safe to schedule GPU work
}

if metalSubsystem.areCriticalPipelinesReady {
    // Critical pipelines ready for immediate use
}
```

### Archive Statistics
```swift
if let stats = await metalSubsystem.archiveStatistics {
    print("Memory hits: \(stats.memoryHits)")
    print("Archive hits: \(stats.archiveHits)")
    print("JIT compilations: \(stats.compilations)")
    print("Hit rate: \(stats.hitRate * 100)%")
}
```

## Implementation Files

### Core Components

| File | Purpose |
|------|---------|
| `Core/MetalSubsystem.swift` | Central lifecycle coordinator |
| `Core/MetalSubsystemConfiguration.swift` | Configuration with presets |
| `Core/BinaryArchiveManager.swift` | MTLBinaryArchive management |
| `Core/ArchivePipelineCache.swift` | Three-tier pipeline cache |
| `Core/PipelineRegistry.swift` | Pipeline tier definitions |
| `Core/WarmupManager.swift` | Activity-aware warmup |
| `Core/ThermalStateMonitor.swift` | Device thermal monitoring |
| `Core/FallbackProvider.swift` | CPU fallback operations |

### Tests

| File | Coverage |
|------|----------|
| `MetalSubsystemTests.swift` | State machine, observers, configuration |
| `BinaryArchiveManagerTests.swift` | Archive lifecycle, manifest, errors |
| `ArchivePipelineCacheTests.swift` | Cache behavior, statistics |
| `WarmupManagerTests.swift` | Activity awareness, thermal respect |
| `PipelineRegistryTests.swift` | Tier categorization |

## Best Practices

1. **Initialize early, use late** - Create `MetalSubsystem` at app launch, start background init after first frame

2. **Always check state** - Don't assume Metal is available; check state or use `requestCriticalPipelines()`

3. **Report user activity** - Call `reportUserActivity()` from gesture recognizers to ensure warmup doesn't compete with user operations

4. **Use appropriate tier** - Put critical-path pipelines in `.critical` tier; everything else in `.occasional` or `.rare`

5. **Handle fallback gracefully** - Always have a CPU fallback path for when Metal fails

6. **Don't block on Metal** - Use async patterns; never synchronously wait for Metal initialization

7. **Test with archive** - Test both cold launch (no archive) and warm launch (with archive) scenarios

## Troubleshooting

### Warmup seems slow
- Check if thermal throttling is active (`ThermalStateMonitor.shared.shouldThrottle`)
- Verify `warmupIdleTimeout` isn't too long
- Check if user activity is being reported too frequently

### Archive not loading
- Verify archive URL is writable
- Check manifest for shader source hash mismatch
- Look for corruption errors in logs

### High JIT compilation count
- Verify archive is being saved (check file exists)
- Ensure manifest persists between launches
- Check for device changes invalidating archive

### Memory pressure
- Reduce `maxMemoryCacheSize` in ArchivePipelineCache
- Clear memory cache during memory warnings: `await context.pipelineCache.clearMemory()`
