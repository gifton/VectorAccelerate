# Phase 3: Shader Compilation

## Objective

Adopt `MTL4Compiler` for shader compilation, implement pipeline harvesting for faster startup, and migrate all shaders to MSL 4.0.

## Dependencies

- Phase 1 complete (Metal4Context, MetalDevice with compiler factory)
- Phase 2 in progress (can be parallel)

## Key Concepts

### MTL4Compiler

Dedicated compiler decoupled from device, with QoS support:

```swift
// Metal 3 - Device-coupled compilation
device.makeLibrary(source: source, options: options) { library, error in ... }

// Metal 4 - Dedicated compiler
let compiler = try device.makeCompiler(descriptor: compilerDescriptor)
let library = try await compiler.makeLibrary(source: source, options: options)
```

### Pipeline Harvesting

Capture compiled pipelines for faster future loads:

```swift
// Harvest after initial compilation
let harvestData = try compiler.harvest(descriptor: harvestDescriptor)
try harvestData.write(to: cacheURL)

// Load from harvest on subsequent launches
let pipelines = try compiler.makeComputePipelineStates(harvestData: data)
```

---

## Tasks

| Task | Description | Status | Dependencies |
|------|-------------|--------|--------------|
| task-metal4-compiler.md | Metal4ShaderCompiler wrapper | **Complete** | Phase 1 |
| task-pipeline-cache.md | Enhanced pipeline caching with keys | **Complete** | Metal4Compiler |
| task-pipeline-harvesting.md | AOT compilation and cache | **Complete** | PipelineCache |
| task-msl4-migration.md | Update all shaders to MSL 4.0 | **Complete** | None |

---

## Pipeline Cache Design

### Cache Key Structure

```swift
/// Unique identifier for a compiled pipeline
public struct PipelineCacheKey: Hashable, Codable {
    /// Operation type (l2Distance, cosine, topK, etc.)
    public let operation: String

    /// Target dimension (384, 512, 768, 1536, or 0 for generic)
    public let dimension: Int

    /// Data type
    public let dataType: DataType

    /// Quantization mode if applicable
    public let quantizationMode: QuantizationMode?

    /// Feature flags
    public let features: FeatureFlags

    public enum DataType: String, Codable {
        case float32
        case float16
        case int8
        case uint8
    }

    public enum QuantizationMode: String, Codable {
        case scalar4
        case scalar8
        case binary
        case productQuantization
    }

    public struct FeatureFlags: OptionSet, Hashable, Codable {
        public let rawValue: UInt32

        public static let fusedNormalize = FeatureFlags(rawValue: 1 << 0)
        public static let fusedTopK = FeatureFlags(rawValue: 1 << 1)
        public static let simdgroupMatrix = FeatureFlags(rawValue: 1 << 2)
        public static let mlTensor = FeatureFlags(rawValue: 1 << 3)

        public init(rawValue: UInt32) { self.rawValue = rawValue }
    }
}

extension PipelineCacheKey {
    /// Create key for standard distance kernel
    static func distance(_ op: String, dimension: Int) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: op,
            dimension: dimension,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )
    }

    /// Create key for quantized operation
    static func quantized(_ op: String, mode: QuantizationMode) -> PipelineCacheKey {
        PipelineCacheKey(
            operation: op,
            dimension: 0,
            dataType: .uint8,
            quantizationMode: mode,
            features: []
        )
    }
}
```

### Cache Storage

```swift
/// Pipeline cache with persistence support
public actor PipelineCache {
    private var memoryCache: [PipelineCacheKey: any MTLComputePipelineState] = [:]
    private let compiler: Metal4ShaderCompiler
    private let cacheDirectory: URL?

    // Statistics
    private var hits: Int = 0
    private var misses: Int = 0
    private var compilations: Int = 0

    public init(compiler: Metal4ShaderCompiler, cacheDirectory: URL? = nil) {
        self.compiler = compiler
        self.cacheDirectory = cacheDirectory
    }

    /// Get or compile pipeline
    public func getPipeline(for key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
        // Check memory cache
        if let cached = memoryCache[key] {
            hits += 1
            return cached
        }

        misses += 1

        // Try to load from disk cache
        if let diskCached = try await loadFromDisk(key: key) {
            memoryCache[key] = diskCached
            return diskCached
        }

        // Compile
        compilations += 1
        let pipeline = try await compiler.compilePipeline(for: key)
        memoryCache[key] = pipeline

        // Save to disk asynchronously
        Task { try? await saveToDisk(key: key, pipeline: pipeline) }

        return pipeline
    }

    /// Cache hit rate
    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }

    /// Warm up cache with common pipelines
    public func warmUp(keys: [PipelineCacheKey]) async {
        await withTaskGroup(of: Void.self) { group in
            for key in keys {
                group.addTask {
                    _ = try? await self.getPipeline(for: key)
                }
            }
        }
    }
}
```

---

## Harvesting Strategy

### When to Harvest

1. **Development builds:** After each shader change
2. **Release builds:** Include pre-harvested data in bundle
3. **First launch:** Harvest if no cache exists
4. **Cache invalidation:** Re-harvest when shaders updated

### Harvest File Format

```
VectorAccelerate/
├── HarvestedPipelines/
│   ├── manifest.json           # Version, hardware, pipeline list
│   ├── distance-384.harvest    # Harvested binary data
│   ├── distance-768.harvest
│   ├── topk-generic.harvest
│   └── ...
```

### Manifest Structure

```json
{
  "version": "1.0.0",
  "vectorAccelerateVersion": "0.2.0",
  "metalSDKVersion": "4.0",
  "generatedAt": "2025-11-29T12:00:00Z",
  "targetGPUFamily": "apple9",
  "pipelines": [
    {
      "key": "l2Distance-384-float32",
      "file": "distance-384.harvest",
      "size": 12345
    }
  ]
}
```

### Versioning & Invalidation

```swift
/// Check if harvested data is compatible
func isHarvestValid(_ manifest: HarvestManifest) -> Bool {
    // Must match:
    // 1. VectorAccelerate version
    // 2. Metal SDK version
    // 3. GPU family (can be more specific if needed)

    return manifest.vectorAccelerateVersion == currentVersion
        && manifest.metalSDKVersion == Metal.sdkVersion
        && device.supportsFamily(manifest.targetGPUFamily)
}
```

---

## Metal4ShaderCompiler Design

```swift
/// Wrapper around MTL4Compiler with caching and harvesting
public actor Metal4ShaderCompiler {
    private let compiler: MTL4Compiler
    private let device: MTLDevice
    private var libraryCache: [String: any MTLLibrary] = [:]

    public init(device: MTLDevice, qos: QualityOfService = .userInteractive) throws {
        self.device = device

        let descriptor = MTL4CompilerDescriptor()
        descriptor.qualityOfService = qos
        self.compiler = try device.makeCompiler(descriptor: descriptor)
    }

    /// Compile library from source
    public func makeLibrary(source: String, label: String? = nil) async throws -> any MTLLibrary {
        let cacheKey = source.hashValue.description

        if let cached = libraryCache[cacheKey] {
            return cached
        }

        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        options.fastMathEnabled = true
        if let label = label {
            options.libraryType = .dynamic
        }

        do {
            let library = try await compiler.makeLibrary(source: source, options: options)
            libraryCache[cacheKey] = library
            return library
        } catch {
            throw Metal4Error.shaderCompilationFailed(
                shader: label ?? "unknown",
                error: error.localizedDescription
            )
        }
    }

    /// Compile library from URL
    public func makeLibrary(url: URL) async throws -> any MTLLibrary {
        let source = try String(contentsOf: url, encoding: .utf8)
        return try await makeLibrary(source: source, label: url.lastPathComponent)
    }

    /// Create function with specialization
    public func makeFunction(
        name: String,
        library: any MTLLibrary,
        constantValues: MTLFunctionConstantValues? = nil
    ) async throws -> any MTLFunction {
        let functionDescriptor = MTL4LibraryFunctionDescriptor()
        functionDescriptor.name = name
        functionDescriptor.library = library

        if let constants = constantValues {
            let specializedDescriptor = MTL4SpecializedFunctionDescriptor()
            specializedDescriptor.functionDescriptor = functionDescriptor
            specializedDescriptor.constantValues = constants
            return try compiler.makeFunction(descriptor: specializedDescriptor)
        }

        return try compiler.makeFunction(descriptor: functionDescriptor)
    }

    /// Compile pipeline for cache key
    public func compilePipeline(for key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
        let functionName = key.functionName
        let library = try await getLibrary(for: key)
        let function = try await makeFunction(name: functionName, library: library)

        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function

        return try await device.makeComputePipelineState(descriptor: descriptor,
                                                          options: [],
                                                          reflection: nil)
    }

    /// Harvest pipelines for AOT loading
    public func harvest(pipelines: [any MTLComputePipelineState]) throws -> Data {
        let harvestDescriptor = MTL4PipelineHarvestDescriptor()
        harvestDescriptor.pipelines = pipelines
        return try compiler.harvest(descriptor: harvestDescriptor)
    }

    /// Load pipelines from harvest data
    public func loadHarvested(data: Data) throws -> [any MTLComputePipelineState] {
        try compiler.makeComputePipelineStates(harvestData: data)
    }
}

extension PipelineCacheKey {
    var functionName: String {
        switch (operation, dimension) {
        case ("l2Distance", 384): return "l2_distance_384"
        case ("l2Distance", 512): return "l2_distance_512"
        case ("l2Distance", 768): return "l2_distance_768"
        case ("l2Distance", 1536): return "l2_distance_1536"
        case ("l2Distance", _): return "l2_distance_generic"
        case ("cosineSimilarity", let d): return "cosine_similarity_\(d)"
        case ("topK", _): return "top_k_selection"
        // ... etc
        default: return operation
        }
    }
}
```

---

## MSL 4.0 Migration

### Header Updates

All shaders need:

```metal
// Before
#include <metal_stdlib>
using namespace metal;

// After
#include <metal_stdlib>
#include <metal_tensor>  // Only if using tensor ops
using namespace metal;
```

### Compile with MSL 4.0

```bash
# CI validation
xcrun metal -c shader.metal -std=metal4.0 -mmacosx-version-min=26.0 -o shader.air
```

### Shader Changes Checklist

| Shader File | Header Update | Tensor Integration | Barrier Review | Status |
|-------------|---------------|--------------------| ---------------|--------|
| BasicOperations.metal | ✅ Done | Future | N/A | **Complete** |
| L2Distance.metal | ✅ Done | Future | N/A | **Complete** |
| CosineSimilarity.metal | ✅ Done | Future | N/A | **Complete** |
| DotProduct.metal | ✅ Done | Future | N/A | **Complete** |
| AdvancedTopK.metal | ✅ Done | No | N/A | **Complete** |
| ProductQuantization.metal | ✅ Done | Future | N/A | **Complete** |
| QuantizationShaders.metal | ✅ Done | Future | N/A | **Complete** |
| ChebyshevDistance.metal | ✅ Done | No | N/A | **Complete** |
| ClusteringShaders.metal | ✅ Done | No | N/A | **Complete** |
| DataTransformations.metal | ✅ Done | No | N/A | **Complete** |
| DistanceShaders.metal | ✅ Done | No | N/A | **Complete** |
| HammingDistance.metal | ✅ Done | No | N/A | **Complete** |
| L2Normalization.metal | ✅ Done | No | N/A | **Complete** |
| ManhattanDistance.metal | ✅ Done | No | N/A | **Complete** |
| MinkowskiDistance.metal | ✅ Done | No | N/A | **Complete** |
| OptimizedMatrixOps.metal | ✅ Done | Future | N/A | **Complete** |
| SearchAndRetrieval.metal | ✅ Done | No | N/A | **Complete** |
| StatisticsShaders.metal | ✅ Done | No | N/A | **Complete** |

All 18 shader files migrated to MSL 4.0 with Metal4Common.h header.

---

## Error Handling

### Compilation Failures

```swift
public func handleCompilationError(_ error: Error, shader: String) -> Metal4Error {
    // Parse error message for common issues
    let message = error.localizedDescription

    if message.contains("undeclared identifier") {
        return .shaderCompilationFailed(
            shader: shader,
            error: "Undeclared identifier - check MSL 4.0 compatibility"
        )
    }

    if message.contains("metal_tensor") {
        return .shaderCompilationFailed(
            shader: shader,
            error: "Missing #include <metal_tensor> header"
        )
    }

    return .shaderCompilationFailed(shader: shader, error: message)
}
```

### Fallback Strategy

```swift
/// Try to load harvested, fall back to JIT compilation
public func loadPipeline(key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
    // 1. Try harvested cache
    if let harvested = try? await loadFromHarvest(key) {
        return harvested
    }

    // 2. Try disk cache
    if let cached = try? await loadFromDisk(key) {
        return cached
    }

    // 3. JIT compile
    Logger.debug("JIT compiling pipeline: \(key)")
    return try await compile(key)
}
```

---

## Completion Criteria

- [ ] Metal4ShaderCompiler implemented
- [ ] PipelineCache with cache keys working
- [ ] Pipeline harvesting implemented
- [ ] All shaders compile with MSL 4.0
- [ ] CI validates all shaders
- [ ] Cache hit rate > 95% in benchmarks
- [ ] Startup time not regressed

## Files Modified

- `Core/ShaderManager.swift` → Major refactor to Metal4ShaderCompiler
- `Core/PipelineCache.swift` → NEW FILE with cache key design
- `Metal/Shaders/*.metal` → MSL 4.0 headers (all 16 files)

## Risk Mitigation

- Keep original ShaderManager for Metal 3 path
- Test each shader individually after MSL 4.0 migration
- Profile compilation times before/after
- Implement harvesting incrementally (start with most-used pipelines)
