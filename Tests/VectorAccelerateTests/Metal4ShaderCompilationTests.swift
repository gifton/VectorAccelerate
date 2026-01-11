//
//  Metal4ShaderCompilationTests.swift
//  VectorAccelerateTests
//
//  Tests for Metal 4 Phase 3: Shader Compilation Infrastructure
//  - PipelineCacheKey
//  - Metal4ShaderCompiler
//  - PipelineCache
//  - PipelineHarvester
//

import XCTest
@testable import VectorAccelerate
import Metal

// MARK: - PipelineCacheKey Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class PipelineCacheKeyTests: XCTestCase {

    // MARK: - Basic Initialization

    func testBasicInitialization() {
        let key = PipelineCacheKey(
            operation: "l2Distance",
            dimension: 384,
            dataType: .float32,
            quantizationMode: nil,
            features: []
        )

        XCTAssertEqual(key.operation, "l2Distance")
        XCTAssertEqual(key.dimension, 384)
        XCTAssertEqual(key.dataType, .float32)
        XCTAssertNil(key.quantizationMode)
        XCTAssertTrue(key.features.isEmpty)
    }

    // MARK: - Factory Methods

    func testL2DistanceFactory() {
        let key = PipelineCacheKey.l2Distance(dimension: 768)

        XCTAssertEqual(key.operation, "l2Distance")
        XCTAssertEqual(key.dimension, 768)
        XCTAssertEqual(key.dataType, .float32)
    }

    func testCosineSimilarityFactory() {
        let key = PipelineCacheKey.cosineSimilarity(dimension: 384)

        XCTAssertEqual(key.operation, "cosineSimilarity")
        XCTAssertEqual(key.dimension, 384)
    }

    func testDotProductFactory() {
        let key = PipelineCacheKey.dotProduct(dimension: 1536)

        XCTAssertEqual(key.operation, "dotProduct")
        XCTAssertEqual(key.dimension, 1536)
    }

    func testTopKFactory() {
        let key = PipelineCacheKey.topK(k: 10)

        XCTAssertEqual(key.operation, "topK")
        XCTAssertEqual(key.dimension, 10)
    }

    func testQuantizedFactory() {
        let key = PipelineCacheKey.quantized("scalarQuantize", mode: .scalar8)

        XCTAssertEqual(key.operation, "scalarQuantize")
        XCTAssertEqual(key.dataType, .uint8)
        XCTAssertEqual(key.quantizationMode, .scalar8)
    }

    func testFusedDistanceTopKFactory() {
        let key = PipelineCacheKey.fusedDistanceTopK(metric: "l2", dimension: 384, k: 10)

        XCTAssertEqual(key.operation, "fused_l2_topk")
        XCTAssertEqual(key.dimension, 384)
        XCTAssertTrue(key.features.contains(.fusedTopK))
    }

    func testBatchFactory() {
        let key = PipelineCacheKey.batch("euclideanDistance", dimension: 512)

        XCTAssertEqual(key.operation, "batch_euclideanDistance")
        XCTAssertEqual(key.dimension, 512)
    }

    // MARK: - Function Name Resolution

    func testFunctionNameForL2Distance() {
        XCTAssertEqual(PipelineCacheKey.l2Distance(dimension: 384).functionName, "l2_distance_384_kernel")
        XCTAssertEqual(PipelineCacheKey.l2Distance(dimension: 512).functionName, "l2_distance_512_kernel")
        XCTAssertEqual(PipelineCacheKey.l2Distance(dimension: 768).functionName, "l2_distance_768_kernel")
        XCTAssertEqual(PipelineCacheKey.l2Distance(dimension: 1536).functionName, "l2_distance_1536_kernel")
        XCTAssertEqual(PipelineCacheKey.l2Distance(dimension: 0).functionName, "l2_distance_kernel")
    }

    func testFunctionNameForCosineSimilarity() {
        XCTAssertEqual(PipelineCacheKey.cosineSimilarity(dimension: 384).functionName, "cosine_similarity_384_kernel")
        XCTAssertEqual(PipelineCacheKey.cosineSimilarity(dimension: 0).functionName, "cosine_similarity_kernel")
    }

    func testFunctionNameForDotProduct() {
        XCTAssertEqual(PipelineCacheKey.dotProduct(dimension: 768).functionName, "dot_product_768_kernel")
    }

    func testFunctionNameForTopK() {
        XCTAssertEqual(PipelineCacheKey.topK(k: 10).functionName, "top_k_selection")
    }

    func testFunctionNameWithQuantization() {
        let key = PipelineCacheKey.quantized("scalarQuantize", mode: .scalar8)
        XCTAssertEqual(key.functionName, "scalarQuantize_q8")
    }

    // MARK: - Cache String

    func testCacheString() {
        let key = PipelineCacheKey.l2Distance(dimension: 384)
        let cacheString = key.cacheString

        XCTAssertTrue(cacheString.contains("l2Distance"))
        XCTAssertTrue(cacheString.contains("d384"))
        XCTAssertTrue(cacheString.contains("float32"))
    }

    func testCacheStringWithFeatures() {
        let key = PipelineCacheKey(
            operation: "fused",
            dimension: 384,
            dataType: .float32,
            quantizationMode: nil,
            features: [.fusedTopK, .simdgroupMatrix]
        )

        let cacheString = key.cacheString
        XCTAssertTrue(cacheString.contains("f"))  // Feature flag indicator
    }

    // MARK: - Equality and Hashing

    func testEquality() {
        let key1 = PipelineCacheKey.l2Distance(dimension: 384)
        let key2 = PipelineCacheKey.l2Distance(dimension: 384)
        let key3 = PipelineCacheKey.l2Distance(dimension: 768)

        XCTAssertEqual(key1, key2)
        XCTAssertNotEqual(key1, key3)
    }

    func testHashing() {
        let key1 = PipelineCacheKey.l2Distance(dimension: 384)
        let key2 = PipelineCacheKey.l2Distance(dimension: 384)

        XCTAssertEqual(key1.hashValue, key2.hashValue)

        // Test use in dictionary
        var dict: [PipelineCacheKey: String] = [:]
        dict[key1] = "test"
        XCTAssertEqual(dict[key2], "test")
    }

    // MARK: - Codable

    func testCodable() throws {
        let key = PipelineCacheKey(
            operation: "l2Distance",
            dimension: 384,
            dataType: .float32,
            quantizationMode: .scalar8,
            features: [.fusedTopK]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(key)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PipelineCacheKey.self, from: data)

        XCTAssertEqual(key, decoded)
        XCTAssertEqual(key.operation, decoded.operation)
        XCTAssertEqual(key.dimension, decoded.dimension)
        XCTAssertEqual(key.dataType, decoded.dataType)
        XCTAssertEqual(key.quantizationMode, decoded.quantizationMode)
        XCTAssertEqual(key.features, decoded.features)
    }

    // MARK: - Common Keys

    func testCommonKeys() {
        let commonKeys = PipelineCacheKey.commonKeys

        XCTAssertFalse(commonKeys.isEmpty)
        XCTAssertTrue(commonKeys.contains(where: { $0.operation == "l2Distance" && $0.dimension == 384 }))
        XCTAssertTrue(commonKeys.contains(where: { $0.operation == "l2Distance" && $0.dimension == 768 }))
        XCTAssertTrue(commonKeys.contains(where: { $0.operation == "cosineSimilarity" }))
    }

    func testEmbeddingModelKeys() {
        let embeddingKeys = PipelineCacheKey.embeddingModelKeys

        XCTAssertFalse(embeddingKeys.isEmpty)
        // Should include common embedding dimensions
        XCTAssertTrue(embeddingKeys.contains(where: { $0.dimension == 384 }))  // MiniLM
        XCTAssertTrue(embeddingKeys.contains(where: { $0.dimension == 768 }))  // BERT
        XCTAssertTrue(embeddingKeys.contains(where: { $0.dimension == 1536 })) // OpenAI ada-002
    }

    // MARK: - PipelineCacheKeySet

    func testPipelineCacheKeySet() {
        let keySet = PipelineCacheKeySet.standard

        XCTAssertFalse(keySet.keys.isEmpty)
        XCTAssertEqual(keySet.version, "1.0.0")
        XCTAssertNotNil(keySet.createdAt)
    }

    func testPipelineCacheKeySetCodable() throws {
        let keySet = PipelineCacheKeySet.standard

        let encoder = JSONEncoder()
        let data = try encoder.encode(keySet)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(PipelineCacheKeySet.self, from: data)

        XCTAssertEqual(keySet.keys.count, decoded.keys.count)
        XCTAssertEqual(keySet.version, decoded.version)
    }
}

// MARK: - Metal4ShaderCompiler Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4ShaderCompilerTests: XCTestCase {

    var device: MTLDevice!
    var compiler: Metal4ShaderCompiler!

    override func setUp() async throws {
        try await super.setUp()

        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        device = mtlDevice
        compiler = try Metal4ShaderCompiler(device: device)
    }

    override func tearDown() async throws {
        compiler = nil
        device = nil
        try await super.tearDown()
    }

    // MARK: - Initialization

    func testInitialization() async throws {
        let compiler = try Metal4ShaderCompiler(device: device)
        let stats = await compiler.getStatistics()

        XCTAssertEqual(stats.totalCompilations, 0)
        XCTAssertEqual(stats.successfulCompilations, 0)
        XCTAssertEqual(stats.failedCompilations, 0)
    }

    func testInitializationWithConfiguration() async throws {
        let config = Metal4CompilerConfiguration(
            qualityOfService: .utility,
            fastMathEnabled: true,
            optimizationLevel: .performance,
            maxConcurrentCompilations: 2
        )

        let compiler = try Metal4ShaderCompiler(device: device, configuration: config)
        let stats = await compiler.getStatistics()

        XCTAssertEqual(stats.totalCompilations, 0)
    }

    // MARK: - Library Compilation

    func testMakeLibraryFromSource() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void test_kernel(device float* output [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
            output[id] = float(id);
        }
        """

        let library = try await compiler.makeLibrary(source: source, label: "TestLibrary")

        XCTAssertNotNil(library)

        let stats = await compiler.getStatistics()
        XCTAssertEqual(stats.successfulCompilations, 1)
        XCTAssertEqual(stats.cachedLibraries, 1)
    }

    func testLibraryCaching() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void test_kernel(device float* output [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
            output[id] = 1.0;
        }
        """

        // Compile twice
        let library1 = try await compiler.makeLibrary(source: source)
        let library2 = try await compiler.makeLibrary(source: source)

        // Should be same cached library
        XCTAssertEqual(ObjectIdentifier(library1), ObjectIdentifier(library2))

        let stats = await compiler.getStatistics()
        XCTAssertEqual(stats.totalCompilations, 1)  // Only compiled once
    }

    func testMakeLibraryFailure() async {
        let invalidSource = "this is not valid metal code {"

        do {
            _ = try await compiler.makeLibrary(source: invalidSource)
            XCTFail("Should have thrown an error")
        } catch {
            // Expected
            let stats = await compiler.getStatistics()
            XCTAssertEqual(stats.failedCompilations, 1)
        }
    }

    // MARK: - Function Creation

    func testMakeFunction() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void my_function(device float* output [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
            output[id] = 42.0;
        }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        let function = try await compiler.makeFunction(name: "my_function")

        XCTAssertNotNil(function)
        XCTAssertEqual(function.name, "my_function")
    }

    func testMakeFunctionNotFound() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void existing_function(device float* output [[buffer(0)]],
                                     uint id [[thread_position_in_grid]]) {
            output[id] = 1.0;
        }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        do {
            _ = try await compiler.makeFunction(name: "nonexistent_function")
            XCTFail("Should have thrown an error")
        } catch {
            // Expected - function not found
        }
    }

    // MARK: - Pipeline Compilation

    func testGetPipelineState() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void test_pipeline(device float* output [[buffer(0)]],
                                 uint id [[thread_position_in_grid]]) {
            output[id] = float(id) * 2.0;
        }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        let pipeline = try await compiler.getPipelineState(functionName: "test_pipeline")

        XCTAssertNotNil(pipeline)
        XCTAssertGreaterThan(pipeline.maxTotalThreadsPerThreadgroup, 0)
    }

    // MARK: - Batch Compilation

    func testCompileMultiple() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_a(device float* output [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) {
            output[id] = 1.0;
        }

        kernel void kernel_b(device float* output [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) {
            output[id] = 2.0;
        }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        let keys = [
            PipelineCacheKey(operation: "kernel_a"),
            PipelineCacheKey(operation: "kernel_b")
        ]

        let results = try await compiler.compileMultiple(keys: keys)

        XCTAssertEqual(results.count, 2)
        XCTAssertNotNil(results[keys[0]])
        XCTAssertNotNil(results[keys[1]])
    }

    // MARK: - Cache Management

    func testCacheInfo() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void cached_kernel(device float* output [[buffer(0)]],
                                 uint id [[thread_position_in_grid]]) {
            output[id] = 0.0;
        }
        """

        _ = try await compiler.makeLibrary(source: source)

        let info = await compiler.cacheInfo

        XCTAssertEqual(info.libraries, 1)
    }

    func testClearCache() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void to_clear(device float* output [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) {
            output[id] = 0.0;
        }
        """

        _ = try await compiler.makeLibrary(source: source)

        var info = await compiler.cacheInfo
        XCTAssertEqual(info.libraries, 1)

        await compiler.clearCache()

        info = await compiler.cacheInfo
        XCTAssertEqual(info.libraries, 0)
        XCTAssertEqual(info.pipelines, 0)
    }

    // MARK: - Statistics

    func testStatistics() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void stats_test(device float* output [[buffer(0)]],
                              uint id [[thread_position_in_grid]]) {
            output[id] = 1.0;
        }
        """

        _ = try await compiler.makeLibrary(source: source)

        let stats = await compiler.getStatistics()

        XCTAssertEqual(stats.successfulCompilations, 1)
        XCTAssertGreaterThan(stats.totalCompilationTime, 0)
        XCTAssertGreaterThan(stats.averageCompilationTime, 0)
    }

    func testResetStatistics() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void reset_test(device float* output [[buffer(0)]],
                              uint id [[thread_position_in_grid]]) {
            output[id] = 1.0;
        }
        """

        _ = try await compiler.makeLibrary(source: source)

        await compiler.resetStatistics()

        let stats = await compiler.getStatistics()
        XCTAssertEqual(stats.totalCompilations, 0)
        XCTAssertEqual(stats.successfulCompilations, 0)
    }
}

// MARK: - PipelineCache Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class PipelineCacheTests: XCTestCase {

    var device: MTLDevice!
    var compiler: Metal4ShaderCompiler!
    var cache: PipelineCache!

    override func setUp() async throws {
        try await super.setUp()

        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        device = mtlDevice
        compiler = try Metal4ShaderCompiler(device: device)
        cache = PipelineCacheFactory.createMemoryOnly(compiler: compiler, maxSize: 10)

        // Set up a default library with test kernels
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void test_kernel(device float* output [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
            output[id] = float(id);
        }

        kernel void euclideanDistance(device float* output [[buffer(0)]],
                                     uint id [[thread_position_in_grid]]) {
            output[id] = 0.0;
        }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)
    }

    override func tearDown() async throws {
        cache = nil
        compiler = nil
        device = nil
        try await super.tearDown()
    }

    // MARK: - Basic Operations

    func testGetPipeline() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        let pipeline = try await cache.getPipeline(for: key)

        XCTAssertNotNil(pipeline)
    }

    func testGetPipelineByFunctionName() async throws {
        let pipeline = try await cache.getPipeline(functionName: "test_kernel")

        XCTAssertNotNil(pipeline)
    }

    func testCaching() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        // First call - cache miss
        _ = try await cache.getPipeline(for: key)

        let stats1 = await cache.getStatistics()
        XCTAssertEqual(stats1.misses, 1)
        XCTAssertEqual(stats1.compilations, 1)

        // Second call - cache hit
        _ = try await cache.getPipeline(for: key)

        let stats2 = await cache.getStatistics()
        XCTAssertEqual(stats2.hits, 1)
        XCTAssertEqual(stats2.compilations, 1)  // No new compilation
    }

    func testIsCached() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        let cachedBefore = await cache.isCached(key)
        XCTAssertFalse(cachedBefore)

        _ = try await cache.getPipeline(for: key)

        let cachedAfter = await cache.isCached(key)
        XCTAssertTrue(cachedAfter)
    }

    // MARK: - LRU Eviction

    func testLRUEviction() async throws {
        // Create cache with max size of 2
        let smallCache = PipelineCacheFactory.createMemoryOnly(compiler: compiler, maxSize: 2)

        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void kernel_1(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 1.0; }
        kernel void kernel_2(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 2.0; }
        kernel void kernel_3(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 3.0; }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        let key1 = PipelineCacheKey(operation: "kernel_1")
        let key2 = PipelineCacheKey(operation: "kernel_2")
        let key3 = PipelineCacheKey(operation: "kernel_3")

        // Fill cache
        _ = try await smallCache.getPipeline(for: key1)
        _ = try await smallCache.getPipeline(for: key2)

        var count = await smallCache.count
        XCTAssertEqual(count, 2)

        // Add third - should evict oldest (key1)
        _ = try await smallCache.getPipeline(for: key3)

        count = await smallCache.count
        XCTAssertEqual(count, 2)

        let key1Cached = await smallCache.isCached(key1)
        let key2Cached = await smallCache.isCached(key2)
        let key3Cached = await smallCache.isCached(key3)
        XCTAssertFalse(key1Cached)  // Evicted
        XCTAssertTrue(key2Cached)
        XCTAssertTrue(key3Cached)
    }

    // MARK: - Cache Management

    func testPreCache() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void precached(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 1.0; }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        let function = try await compiler.makeFunction(name: "precached")
        let pipeline = try await device.makeComputePipelineState(function: function)

        let key = PipelineCacheKey(operation: "precached")
        await cache.preCache(key: key, pipeline: pipeline)

        let isCached = await cache.isCached(key)
        XCTAssertTrue(isCached)
    }

    func testRemove() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        _ = try await cache.getPipeline(for: key)
        var isCached = await cache.isCached(key)
        XCTAssertTrue(isCached)

        await cache.remove(key)
        isCached = await cache.isCached(key)
        XCTAssertFalse(isCached)
    }

    func testClear() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        _ = try await cache.getPipeline(for: key)
        var count = await cache.count
        XCTAssertEqual(count, 1)

        await cache.clear()
        count = await cache.count
        XCTAssertEqual(count, 0)
    }

    // MARK: - Statistics

    func testStatistics() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        _ = try await cache.getPipeline(for: key)
        _ = try await cache.getPipeline(for: key)

        let stats = await cache.getStatistics()

        XCTAssertEqual(stats.hits, 1)
        XCTAssertEqual(stats.misses, 1)
        XCTAssertEqual(stats.compilations, 1)
        XCTAssertEqual(stats.memoryCacheSize, 1)
        XCTAssertEqual(stats.hitRate, 0.5, accuracy: 0.01)
    }

    func testResetStatistics() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        _ = try await cache.getPipeline(for: key)

        await cache.resetStatistics()

        let stats = await cache.getStatistics()
        XCTAssertEqual(stats.hits, 0)
        XCTAssertEqual(stats.misses, 0)
    }

    // MARK: - Metadata

    func testGetMetadata() async throws {
        let key = PipelineCacheKey(operation: "test_kernel")

        _ = try await cache.getPipeline(for: key)

        let metadata = await cache.getMetadata(for: key)

        XCTAssertNotNil(metadata)
        XCTAssertEqual(metadata?.key, key)
        XCTAssertEqual(metadata?.accessCount, 1)
        XCTAssertGreaterThanOrEqual(metadata?.compilationTime ?? 0, 0)
    }

    func testMostAccessed() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void popular(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 1.0; }
        kernel void unpopular(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 2.0; }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        let popularKey = PipelineCacheKey(operation: "popular")
        let unpopularKey = PipelineCacheKey(operation: "unpopular")

        // Access popular more times
        _ = try await cache.getPipeline(for: popularKey)
        _ = try await cache.getPipeline(for: popularKey)
        _ = try await cache.getPipeline(for: popularKey)
        _ = try await cache.getPipeline(for: unpopularKey)

        let mostAccessed = await cache.getMostAccessed(limit: 1)

        XCTAssertEqual(mostAccessed.first, popularKey)
    }

    // MARK: - Warm-up

    func testWarmUp() async throws {
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void warmup_a(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 1.0; }
        kernel void warmup_b(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 2.0; }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)

        let keys = [
            PipelineCacheKey(operation: "warmup_a"),
            PipelineCacheKey(operation: "warmup_b")
        ]

        await cache.warmUp(keys: keys)

        let key0Cached = await cache.isCached(keys[0])
        let key1Cached = await cache.isCached(keys[1])
        XCTAssertTrue(key0Cached)
        XCTAssertTrue(key1Cached)
    }
}

// MARK: - PipelineHarvester Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class PipelineHarvesterTests: XCTestCase {

    var device: MTLDevice!
    var compiler: Metal4ShaderCompiler!
    var tempDirectory: URL!

    override func setUp() async throws {
        try await super.setUp()

        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        device = mtlDevice
        compiler = try Metal4ShaderCompiler(device: device)

        // Create temp directory for harvest files
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("VectorAccelerateTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)

        // Set up default library
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void harvest_test(device float* o [[buffer(0)]], uint id [[thread_position_in_grid]]) { o[id] = 1.0; }
        """

        let library = try await compiler.makeLibrary(source: source)
        await compiler.setDefaultLibrary(library)
    }

    override func tearDown() async throws {
        // Clean up temp directory
        if let tempDirectory = tempDirectory {
            try? FileManager.default.removeItem(at: tempDirectory)
        }

        compiler = nil
        device = nil
        try await super.tearDown()
    }

    // MARK: - Initialization

    func testInitialization() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        XCTAssertNotNil(harvester)
    }

    // MARK: - Harvesting

    func testHarvest() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        let keys = [PipelineCacheKey(operation: "harvest_test")]

        let result = try await harvester.harvest(keys: keys)

        XCTAssertEqual(result.successCount, 1)
        XCTAssertEqual(result.failureCount, 0)
        XCTAssertTrue(result.wasSuccessful)
        XCTAssertGreaterThan(result.totalTime, 0)
        XCTAssertEqual(result.manifest.pipelines.count, 1)
    }

    func testHarvestManifest() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory,
            vectorAccelerateVersion: "0.3.6"
        )

        let keys = [PipelineCacheKey(operation: "harvest_test")]
        let result = try await harvester.harvest(keys: keys)

        let manifest = result.manifest

        XCTAssertEqual(manifest.version, "1.0.0")
        XCTAssertEqual(manifest.vectorAccelerateVersion, "0.3.6")
        XCTAssertFalse(manifest.targetGPUFamily.isEmpty)
        XCTAssertFalse(manifest.deviceName.isEmpty)
        XCTAssertNotNil(manifest.generatedAt)
    }

    func testHarvestCreatesManifestFile() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        let keys = [PipelineCacheKey(operation: "harvest_test")]
        _ = try await harvester.harvest(keys: keys)

        // Check manifest file exists
        let manifestURL = tempDirectory.appendingPathComponent("manifest.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))

        // Verify manifest is valid JSON
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(HarvestManifest.self, from: data)

        XCTAssertEqual(manifest.pipelines.count, 1)
    }

    func testHarvestWithInvalidKey() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        let keys = [
            PipelineCacheKey(operation: "harvest_test"),
            PipelineCacheKey(operation: "nonexistent_kernel")
        ]

        let result = try await harvester.harvest(keys: keys)

        XCTAssertEqual(result.successCount, 1)
        XCTAssertEqual(result.failureCount, 1)
        XCTAssertFalse(result.wasSuccessful)
    }

    // MARK: - Manifest Operations

    func testGetManifest() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        // Before harvest - no manifest
        let manifestBefore = try await harvester.getManifest()
        XCTAssertNil(manifestBefore)

        // After harvest
        let keys = [PipelineCacheKey(operation: "harvest_test")]
        _ = try await harvester.harvest(keys: keys)

        let manifestAfter = try await harvester.getManifest()
        XCTAssertNotNil(manifestAfter)
        XCTAssertEqual(manifestAfter?.pipelines.count, 1)
    }

    func testIsManifestCompatible() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        let keys = [PipelineCacheKey(operation: "harvest_test")]
        let result = try await harvester.harvest(keys: keys)

        // Should be compatible with itself
        let isCompatible = await harvester.isManifestCompatible(result.manifest)
        XCTAssertTrue(isCompatible)
    }

    // MARK: - Cache Loading

    func testLoadHarvestedIntoCache() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        let keys = [PipelineCacheKey(operation: "harvest_test")]
        _ = try await harvester.harvest(keys: keys)

        // Create fresh cache
        let cache = PipelineCacheFactory.createMemoryOnly(compiler: compiler, maxSize: 10)

        let loadedCount = try await harvester.loadHarvested(into: cache)

        XCTAssertEqual(loadedCount, 1)
        let isCached = await cache.isCached(keys[0])
        XCTAssertTrue(isCached)
    }

    // MARK: - Cleanup

    func testClearHarvest() async throws {
        let harvester = try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: tempDirectory
        )

        let keys = [PipelineCacheKey(operation: "harvest_test")]
        _ = try await harvester.harvest(keys: keys)

        // Verify files exist
        let manifestURL = tempDirectory.appendingPathComponent("manifest.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifestURL.path))

        // Clear
        try await harvester.clearHarvest()

        // Verify files removed
        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestURL.path))
    }

    // MARK: - Factory

    func testHarvesterFactory() async throws {
        let harvester = try PipelineHarvesterFactory.create(
            device: device,
            compiler: compiler
        )

        XCTAssertNotNil(harvester)
    }

    func testHarvesterFactoryWithCustomDirectory() async throws {
        let customDir = tempDirectory.appendingPathComponent("custom")

        let harvester = try PipelineHarvesterFactory.create(
            device: device,
            compiler: compiler,
            outputDirectory: customDir
        )

        XCTAssertNotNil(harvester)
        XCTAssertTrue(FileManager.default.fileExists(atPath: customDir.path))
    }
}

// MARK: - HarvestManifest Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class HarvestManifestTests: XCTestCase {

    func testManifestCodable() throws {
        let pipelineInfo = HarvestedPipelineInfo(
            key: PipelineCacheKey.l2Distance(dimension: 384),
            fileName: "l2-384.harvest",
            size: 12345,
            checksum: "abc123"
        )

        let manifest = HarvestManifest(
            vectorAccelerateVersion: "0.3.6",
            metalSDKVersion: "4.0",
            targetGPUFamily: "apple9",
            deviceName: "Test Device",
            pipelines: [pipelineInfo]
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(manifest)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(HarvestManifest.self, from: data)

        XCTAssertEqual(manifest.version, decoded.version)
        XCTAssertEqual(manifest.vectorAccelerateVersion, decoded.vectorAccelerateVersion)
        XCTAssertEqual(manifest.metalSDKVersion, decoded.metalSDKVersion)
        XCTAssertEqual(manifest.targetGPUFamily, decoded.targetGPUFamily)
        XCTAssertEqual(manifest.deviceName, decoded.deviceName)
        XCTAssertEqual(manifest.pipelines.count, decoded.pipelines.count)
    }

    func testHarvestedPipelineInfoCodable() throws {
        let info = HarvestedPipelineInfo(
            key: PipelineCacheKey(
                operation: "l2Distance",
                dimension: 768,
                dataType: .float32,
                quantizationMode: nil,
                features: [.fusedTopK]
            ),
            fileName: "l2-768.harvest",
            size: 54321,
            checksum: "xyz789"
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(info)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(HarvestedPipelineInfo.self, from: data)

        XCTAssertEqual(info.key, decoded.key)
        XCTAssertEqual(info.fileName, decoded.fileName)
        XCTAssertEqual(info.size, decoded.size)
        XCTAssertEqual(info.checksum, decoded.checksum)
    }
}

// MARK: - Metal4CompilerConfiguration Tests

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class Metal4CompilerConfigurationTests: XCTestCase {

    func testDefaultConfiguration() {
        let config = Metal4CompilerConfiguration.default

        XCTAssertEqual(config.qualityOfService, .userInteractive)
        XCTAssertTrue(config.fastMathEnabled)
        XCTAssertEqual(config.maxConcurrentCompilations, 4)
    }

    func testBatchConfiguration() {
        let config = Metal4CompilerConfiguration.batch

        XCTAssertEqual(config.qualityOfService, .utility)
        XCTAssertEqual(config.maxConcurrentCompilations, 8)
    }

    func testRealTimeConfiguration() {
        let config = Metal4CompilerConfiguration.realTime

        XCTAssertEqual(config.qualityOfService, .userInteractive)
        XCTAssertEqual(config.maxConcurrentCompilations, 2)
    }

    func testCustomConfiguration() {
        let config = Metal4CompilerConfiguration(
            qualityOfService: .background,
            fastMathEnabled: false,
            optimizationLevel: .size,
            maxConcurrentCompilations: 1
        )

        XCTAssertEqual(config.qualityOfService, .background)
        XCTAssertFalse(config.fastMathEnabled)
        XCTAssertEqual(config.maxConcurrentCompilations, 1)
    }
}
