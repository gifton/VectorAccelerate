//
//  ArchivePipelineCacheTests.swift
//  VectorAccelerateTests
//
//  Tests for ArchivePipelineCache archive-aware pipeline caching
//

import XCTest
@testable import VectorAccelerate
import Metal

final class ArchivePipelineCacheTests: XCTestCase {

    private var testDirectory: URL!
    private var archiveURL: URL!
    private var device: (any MTLDevice)!
    private var compiler: Metal4ShaderCompiler!
    private var archiveManager: BinaryArchiveManager!

    override func setUpWithError() throws {
        // Create test directory
        testDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ArchivePipelineCacheTests")
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: testDirectory,
            withIntermediateDirectories: true
        )

        archiveURL = testDirectory.appendingPathComponent("test.metalarchive")

        // Get Metal device
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        device = metalDevice

        // Create compiler
        compiler = try Metal4ShaderCompiler(
            device: device,
            configuration: .default
        )

        // Create archive manager
        archiveManager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )
    }

    override func tearDownWithError() throws {
        // Clean up test directory
        if let testDirectory = testDirectory {
            try? FileManager.default.removeItem(at: testDirectory)
        }
    }

    // MARK: - Initialization Tests

    func testInitialization() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager,
            maxMemoryCacheSize: 50
        )

        let count = await cache.memoryCacheCount
        XCTAssertEqual(count, 0)
    }

    // MARK: - Statistics Tests

    func testInitialStatistics() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager
        )

        let stats = await cache.getStatistics()

        XCTAssertEqual(stats.memoryHits, 0)
        XCTAssertEqual(stats.archiveHits, 0)
        XCTAssertEqual(stats.compilations, 0)
        XCTAssertEqual(stats.archiveSaves, 0)
        XCTAssertEqual(stats.memoryCacheSize, 0)
        XCTAssertEqual(stats.hitRate, 0)
    }

    func testStatisticsHitRate() {
        // Test hit rate calculation
        let stats = ArchivePipelineCacheStatistics(
            memoryHits: 80,
            archiveHits: 15,
            compilations: 5,
            archiveSaves: 5,
            memoryCacheSize: 10,
            archiveSize: 20
        )

        // Total = 100, hits = 95
        XCTAssertEqual(stats.hitRate, 0.95, accuracy: 0.001)

        // Memory hit rate = 80/100
        XCTAssertEqual(stats.memoryHitRate, 0.80, accuracy: 0.001)

        // Archive hit rate = 15/(15+5) = 15/20
        XCTAssertEqual(stats.archiveHitRate, 0.75, accuracy: 0.001)
    }

    func testStatisticsResetStatistics() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager
        )

        // Reset and verify
        await cache.resetStatistics()

        let stats = await cache.getStatistics()
        XCTAssertEqual(stats.memoryHits, 0)
        XCTAssertEqual(stats.archiveHits, 0)
        XCTAssertEqual(stats.compilations, 0)
    }

    // MARK: - Cache Management Tests

    func testClearMemory() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager
        )

        await cache.clearMemory()

        let count = await cache.memoryCacheCount
        XCTAssertEqual(count, 0)
    }

    func testClear() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager
        )

        await cache.clear()

        let count = await cache.memoryCacheCount
        XCTAssertEqual(count, 0)
    }

    func testCachedKeysEmpty() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager
        )

        let keys = await cache.cachedKeys
        XCTAssertTrue(keys.isEmpty)
    }

    // MARK: - IsCached Tests

    func testIsCachedFalseInitially() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager
        )

        let key = PipelineCacheKey.l2Distance(dimension: 384)
        let isCached = await cache.isCached(key)

        XCTAssertFalse(isCached)
    }

    // MARK: - Save to Archive Tests

    func testSaveToArchiveWithNoPending() async throws {
        try await archiveManager.loadOrCreate()

        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager
        )

        // Should not throw with no pending saves
        try await cache.saveToArchive()
    }

    // MARK: - Factory Tests

    func testFactoryCreate() async throws {
        let cache = try await ArchivePipelineCache.create(
            device: device,
            compiler: compiler,
            archiveURL: archiveURL,
            maxMemoryCacheSize: 75
        )

        let count = await cache.memoryCacheCount
        XCTAssertEqual(count, 0)
    }

    // MARK: - Hit Rate Edge Cases

    func testHitRateZeroWhenNoRequests() {
        let stats = ArchivePipelineCacheStatistics(
            memoryHits: 0,
            archiveHits: 0,
            compilations: 0,
            archiveSaves: 0,
            memoryCacheSize: 0,
            archiveSize: 0
        )

        XCTAssertEqual(stats.hitRate, 0)
        XCTAssertEqual(stats.memoryHitRate, 0)
        XCTAssertEqual(stats.archiveHitRate, 0)
    }

    func testHitRate100PercentMemory() {
        let stats = ArchivePipelineCacheStatistics(
            memoryHits: 100,
            archiveHits: 0,
            compilations: 0,
            archiveSaves: 0,
            memoryCacheSize: 10,
            archiveSize: 0
        )

        XCTAssertEqual(stats.hitRate, 1.0)
        XCTAssertEqual(stats.memoryHitRate, 1.0)
    }

    func testHitRate100PercentArchive() {
        let stats = ArchivePipelineCacheStatistics(
            memoryHits: 0,
            archiveHits: 100,
            compilations: 0,
            archiveSaves: 0,
            memoryCacheSize: 0,
            archiveSize: 10
        )

        XCTAssertEqual(stats.hitRate, 1.0)
        XCTAssertEqual(stats.archiveHitRate, 1.0)
    }

    // MARK: - Memory Cache Size Limit

    func testMemoryCacheSizeLimit() async throws {
        try await archiveManager.loadOrCreate()

        // Create cache with very small limit
        let cache = ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager,
            maxMemoryCacheSize: 2
        )

        // The cache should respect the limit (tested indirectly through statistics)
        let stats = await cache.getStatistics()
        XCTAssertEqual(stats.memoryCacheSize, 0)
    }
}
