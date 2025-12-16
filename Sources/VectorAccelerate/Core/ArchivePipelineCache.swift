//
//  ArchivePipelineCache.swift
//  VectorAccelerate
//
//  Enhanced pipeline cache with binary archive support
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Statistics for archive-aware pipeline cache.
public struct ArchivePipelineCacheStatistics: Sendable {
    /// Hits from memory cache
    public let memoryHits: Int

    /// Hits from binary archive
    public let archiveHits: Int

    /// JIT compilations (cache misses)
    public let compilations: Int

    /// Number of pipelines saved to archive
    public let archiveSaves: Int

    /// Current size of memory cache
    public let memoryCacheSize: Int

    /// Number of pipelines in archive
    public let archiveSize: Int

    /// Combined hit rate (memory + archive)
    public var hitRate: Double {
        let total = memoryHits + archiveHits + compilations
        guard total > 0 else { return 0 }
        return Double(memoryHits + archiveHits) / Double(total)
    }

    /// Memory-only hit rate
    public var memoryHitRate: Double {
        let total = memoryHits + archiveHits + compilations
        guard total > 0 else { return 0 }
        return Double(memoryHits) / Double(total)
    }

    /// Archive hit rate (excluding memory hits)
    public var archiveHitRate: Double {
        let archiveAndCompile = archiveHits + compilations
        guard archiveAndCompile > 0 else { return 0 }
        return Double(archiveHits) / Double(archiveAndCompile)
    }
}

/// Enhanced pipeline cache that uses binary archives for persistence.
///
/// `ArchivePipelineCache` provides three-tier lookup:
/// 1. **Memory cache:** Fastest, holds recently used pipelines
/// 2. **Binary archive:** Persisted compiled pipelines from previous launches
/// 3. **JIT compilation:** Fallback when not in memory or archive
///
/// After JIT compilation, pipelines are automatically added to the binary archive
/// for instant loading on subsequent launches.
///
/// ## Benefits
///
/// - **First launch:** Pipelines compile and are saved to archive
/// - **Subsequent launches:** Pipelines load from archive in ~1ms instead of ~50-200ms
/// - **Reduced thermal load:** No shader compilation heat on warm starts
///
/// ## Usage
///
/// ```swift
/// let cache = ArchivePipelineCache(
///     compiler: compiler,
///     archiveManager: archiveManager
/// )
///
/// // Pipeline retrieval uses three-tier lookup
/// let pipeline = try await cache.getPipeline(for: .l2Distance(dimension: 384))
///
/// // Warm up populates both memory cache and archive
/// await cache.warmUp(keys: registry.criticalKeys)
///
/// // Save archive to disk after warmup
/// try await cache.saveToArchive()
/// ```
public actor ArchivePipelineCache {

    // MARK: - Properties

    /// Metal 4 shader compiler for JIT compilation
    private let compiler: Metal4ShaderCompiler

    /// Binary archive manager for persistent storage
    private let archiveManager: BinaryArchiveManager

    /// In-memory pipeline cache
    private var memoryCache: [PipelineCacheKey: any MTLComputePipelineState] = [:]

    /// Maximum pipelines in memory cache
    private let maxMemoryCacheSize: Int

    /// Metadata for LRU eviction
    private var accessTimes: [PipelineCacheKey: Date] = [:]

    // MARK: - Statistics

    private var memoryHits: Int = 0
    private var archiveHits: Int = 0
    private var compilations: Int = 0
    private var archiveSaves: Int = 0

    // MARK: - Pending Archive Saves

    /// Keys that need to be saved to archive (batched)
    private var pendingArchiveSaves: Set<PipelineCacheKey> = []

    /// Threshold for batch save
    private let batchSaveThreshold: Int = 5

    // MARK: - Initialization

    /// Create an archive-aware pipeline cache.
    ///
    /// - Parameters:
    ///   - compiler: Shader compiler for JIT compilation
    ///   - archiveManager: Binary archive manager
    ///   - maxMemoryCacheSize: Maximum pipelines in memory (default: 100)
    public init(
        compiler: Metal4ShaderCompiler,
        archiveManager: BinaryArchiveManager,
        maxMemoryCacheSize: Int = 100
    ) {
        self.compiler = compiler
        self.archiveManager = archiveManager
        self.maxMemoryCacheSize = maxMemoryCacheSize
    }

    // MARK: - Pipeline Retrieval

    /// Get or compile pipeline for a cache key.
    ///
    /// Lookup order:
    /// 1. Memory cache (fastest)
    /// 2. Binary archive (fast, ~1ms)
    /// 3. JIT compilation (slow, ~50-200ms)
    ///
    /// After JIT compilation, the pipeline is added to the archive for next launch.
    ///
    /// - Parameter key: Pipeline cache key
    /// - Returns: Compiled pipeline state object
    /// - Throws: Compilation errors
    public func getPipeline(for key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
        // 1. Check memory cache
        if let cached = memoryCache[key] {
            memoryHits += 1
            accessTimes[key] = Date()
            return cached
        }

        // 2. Check binary archive
        if await archiveManager.containsPipeline(for: key) {
            do {
                let pipeline = try await compilePipelineFromArchive(for: key)
                archiveHits += 1
                addToMemoryCache(key: key, pipeline: pipeline)
                return pipeline
            } catch {
                // Archive entry invalid - fall through to JIT
            }
        }

        // 3. JIT compile
        compilations += 1
        let pipeline = try await compiler.compilePipeline(for: key)
        addToMemoryCache(key: key, pipeline: pipeline)

        // Queue for archive save
        await queueForArchiveSave(key: key)

        return pipeline
    }

    /// Get pipeline by function name.
    ///
    /// - Parameter functionName: Metal function name
    /// - Returns: Compiled pipeline state object
    public func getPipeline(functionName: String) async throws -> any MTLComputePipelineState {
        let key = PipelineCacheKey(operation: functionName)
        return try await getPipeline(for: key)
    }

    /// Check if a pipeline is cached (memory or archive).
    ///
    /// - Parameter key: Pipeline cache key
    /// - Returns: true if cached
    public func isCached(_ key: PipelineCacheKey) async -> Bool {
        if memoryCache[key] != nil {
            return true
        }
        return await archiveManager.containsPipeline(for: key)
    }

    // MARK: - Warm-up

    /// Warm up cache with pipeline keys.
    ///
    /// Compiles pipelines concurrently and adds them to both memory cache
    /// and binary archive.
    ///
    /// - Parameter keys: Pipeline keys to warm up
    public func warmUp(keys: [PipelineCacheKey]) async {
        await withTaskGroup(of: Void.self) { group in
            for key in keys {
                group.addTask {
                    _ = try? await self.getPipeline(for: key)
                }
            }
        }
    }

    /// Warm up with standard pipeline keys.
    public func warmUpStandard() async {
        await warmUp(keys: PipelineCacheKey.commonKeys)
    }

    /// Warm up with embedding model keys.
    public func warmUpEmbeddings() async {
        await warmUp(keys: PipelineCacheKey.embeddingModelKeys)
    }

    // MARK: - Archive Operations

    /// Save pending pipelines to the binary archive.
    ///
    /// This method:
    /// 1. Creates pipeline descriptors for pending pipelines
    /// 2. Adds them to the archive
    /// 3. Serializes the archive to disk
    ///
    /// - Throws: Archive errors
    public func saveToArchive() async throws {
        guard !pendingArchiveSaves.isEmpty else { return }

        // Process pending saves
        for key in pendingArchiveSaves {
            if memoryCache[key] != nil {
                await addPipelineToArchive(key: key)
            }
        }

        pendingArchiveSaves.removeAll()

        // Save to disk
        try await archiveManager.save()
    }

    /// Load pipelines from the binary archive into memory cache.
    ///
    /// This pre-loads frequently used pipelines for instant access.
    ///
    /// - Parameter keys: Keys to load (defaults to all stored keys)
    public func loadFromArchive(keys: [PipelineCacheKey]? = nil) async throws {
        let keysToLoad: [PipelineCacheKey]
        if let providedKeys = keys {
            keysToLoad = providedKeys
        } else {
            keysToLoad = await archiveManager.storedKeys
        }

        for key in keysToLoad {
            guard memoryCache[key] == nil else { continue }
            let containsPipeline = await archiveManager.containsPipeline(for: key)
            guard containsPipeline else { continue }

            do {
                let pipeline = try await compilePipelineFromArchive(for: key)
                addToMemoryCache(key: key, pipeline: pipeline)
            } catch {
                // Skip invalid entries
                continue
            }
        }
    }

    // MARK: - Cache Management

    /// Clear memory cache.
    public func clearMemory() {
        memoryCache.removeAll()
        accessTimes.removeAll()
    }

    /// Clear all caches including pending saves.
    public func clear() {
        clearMemory()
        pendingArchiveSaves.removeAll()
    }

    /// Get number of pipelines in memory cache.
    public var memoryCacheCount: Int {
        memoryCache.count
    }

    /// Get all keys in memory cache.
    public var cachedKeys: [PipelineCacheKey] {
        Array(memoryCache.keys)
    }

    // MARK: - Statistics

    /// Get cache statistics.
    public func getStatistics() async -> ArchivePipelineCacheStatistics {
        ArchivePipelineCacheStatistics(
            memoryHits: memoryHits,
            archiveHits: archiveHits,
            compilations: compilations,
            archiveSaves: archiveSaves,
            memoryCacheSize: memoryCache.count,
            archiveSize: await archiveManager.pipelineCount
        )
    }

    /// Reset statistics counters.
    public func resetStatistics() {
        memoryHits = 0
        archiveHits = 0
        compilations = 0
        archiveSaves = 0
    }

    // MARK: - Private Implementation

    /// Add pipeline to memory cache with LRU eviction.
    private func addToMemoryCache(key: PipelineCacheKey, pipeline: any MTLComputePipelineState) {
        // Evict if at capacity
        if maxMemoryCacheSize > 0 && memoryCache.count >= maxMemoryCacheSize {
            evictLRU()
        }

        memoryCache[key] = pipeline
        accessTimes[key] = Date()
    }

    /// Evict least recently used entry.
    private func evictLRU() {
        guard let oldest = accessTimes.min(by: { $0.value < $1.value }) else {
            return
        }
        memoryCache.removeValue(forKey: oldest.key)
        accessTimes.removeValue(forKey: oldest.key)
    }

    /// Compile pipeline using archive hint.
    private func compilePipelineFromArchive(for key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
        // Get function from compiler
        let functionName = key.functionName
        let function = try await compiler.makeFunction(name: functionName)

        // Create descriptor with archive hint
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.label = key.cacheString

        // Add archive for potential acceleration
        if let archive = await archiveManager.underlyingArchive {
            descriptor.binaryArchives = [archive]
        }

        // Create pipeline (should be fast if archive is valid)
        let device = function.device
        let pipeline = try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<any MTLComputePipelineState, any Error>) in
            device.makeComputePipelineState(
                descriptor: descriptor,
                options: []
            ) { pipeline, _, error in
                if let pipeline = pipeline {
                    continuation.resume(returning: pipeline)
                } else {
                    continuation.resume(
                        throwing: error ?? VectorError.shaderCompilationFailed("Pipeline creation failed")
                    )
                }
            }
        }

        return pipeline
    }

    /// Queue a pipeline for archive save.
    private func queueForArchiveSave(key: PipelineCacheKey) async {
        // Skip if already in archive
        if await archiveManager.containsPipeline(for: key) {
            return
        }

        pendingArchiveSaves.insert(key)

        // Auto-save when threshold reached
        if pendingArchiveSaves.count >= batchSaveThreshold {
            try? await saveToArchive()
        }
    }

    /// Add a single pipeline to the archive.
    private func addPipelineToArchive(key: PipelineCacheKey) async {
        do {
            // Get function
            let functionName = key.functionName
            let function = try await compiler.makeFunction(name: functionName)

            // Create descriptor
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.label = key.cacheString

            // Add to archive
            try await archiveManager.addPipeline(descriptor: descriptor, for: key)
            archiveSaves += 1
        } catch {
            // Log but don't fail - archive save is best-effort
        }
    }
}

// MARK: - Factory

extension ArchivePipelineCache {
    /// Create an archive pipeline cache with a new archive manager.
    ///
    /// - Parameters:
    ///   - device: Metal device
    ///   - compiler: Shader compiler
    ///   - archiveURL: Optional archive URL (uses default if nil)
    ///   - maxMemoryCacheSize: Maximum memory cache size
    /// - Returns: Configured cache
    public static func create(
        device: any MTLDevice,
        compiler: Metal4ShaderCompiler,
        archiveURL: URL? = nil,
        maxMemoryCacheSize: Int = 100
    ) async throws -> ArchivePipelineCache {
        let archiveManager = BinaryArchiveManager(
            device: device,
            archiveURL: archiveURL
        )

        try await archiveManager.loadOrCreate()

        return ArchivePipelineCache(
            compiler: compiler,
            archiveManager: archiveManager,
            maxMemoryCacheSize: maxMemoryCacheSize
        )
    }
}
