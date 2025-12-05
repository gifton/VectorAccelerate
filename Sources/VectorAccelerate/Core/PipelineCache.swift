//
//  PipelineCache.swift
//  VectorAccelerate
//
//  Pipeline cache with memory and disk persistence
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Pipeline Cache Statistics

/// Statistics for pipeline cache monitoring
public struct PipelineCacheStatistics: Sendable {
    public let hits: Int
    public let misses: Int
    public let compilations: Int
    public let diskLoads: Int
    public let diskSaves: Int
    public let memoryCacheSize: Int
    public let diskCacheSize: Int

    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }

    public var missRate: Double {
        1.0 - hitRate
    }
}

// MARK: - Cache Entry Metadata

/// Metadata for cached pipeline entries
public struct PipelineCacheEntry: Codable, Sendable {
    public let key: PipelineCacheKey
    public let createdAt: Date
    public let lastAccessedAt: Date
    public let accessCount: Int
    public let compilationTime: TimeInterval
    public let diskFileName: String?

    public init(
        key: PipelineCacheKey,
        compilationTime: TimeInterval = 0,
        diskFileName: String? = nil
    ) {
        self.key = key
        self.createdAt = Date()
        self.lastAccessedAt = Date()
        self.accessCount = 1
        self.compilationTime = compilationTime
        self.diskFileName = diskFileName
    }

    public func touched() -> PipelineCacheEntry {
        PipelineCacheEntry(
            key: key,
            createdAt: createdAt,
            lastAccessedAt: Date(),
            accessCount: accessCount + 1,
            compilationTime: compilationTime,
            diskFileName: diskFileName
        )
    }

    init(
        key: PipelineCacheKey,
        createdAt: Date,
        lastAccessedAt: Date,
        accessCount: Int,
        compilationTime: TimeInterval,
        diskFileName: String?
    ) {
        self.key = key
        self.createdAt = createdAt
        self.lastAccessedAt = lastAccessedAt
        self.accessCount = accessCount
        self.compilationTime = compilationTime
        self.diskFileName = diskFileName
    }
}

// MARK: - Pipeline Cache

/// Pipeline cache with memory and optional disk persistence
///
/// Features:
/// - In-memory LRU cache
/// - Optional disk persistence
/// - Automatic compilation on miss
/// - Statistics tracking
/// - Warm-up support
///
/// Example:
/// ```swift
/// let cache = try await PipelineCache(compiler: compiler)
/// let pipeline = try await cache.getPipeline(for: .l2Distance(dimension: 384))
/// ```
public actor PipelineCache {
    // MARK: - Properties

    private let compiler: Metal4ShaderCompiler
    private let cacheDirectory: URL?
    private let maxMemoryCacheSize: Int

    // Memory cache
    private var memoryCache: [PipelineCacheKey: any MTLComputePipelineState] = [:]
    private var metadata: [PipelineCacheKey: PipelineCacheEntry] = [:]

    // Statistics
    private var hits: Int = 0
    private var misses: Int = 0
    private var compilations: Int = 0
    private var diskLoads: Int = 0
    private var diskSaves: Int = 0

    // MARK: - Initialization

    /// Create a pipeline cache
    ///
    /// - Parameters:
    ///   - compiler: Shader compiler for on-demand compilation
    ///   - cacheDirectory: Optional directory for disk persistence
    ///   - maxMemoryCacheSize: Maximum pipelines in memory (0 = unlimited)
    public init(
        compiler: Metal4ShaderCompiler,
        cacheDirectory: URL? = nil,
        maxMemoryCacheSize: Int = 100
    ) {
        self.compiler = compiler
        self.cacheDirectory = cacheDirectory
        self.maxMemoryCacheSize = maxMemoryCacheSize
    }

    /// Create with automatic cache directory
    public init(
        compiler: Metal4ShaderCompiler,
        enableDiskCache: Bool = true,
        maxMemoryCacheSize: Int = 100
    ) throws {
        self.compiler = compiler
        self.maxMemoryCacheSize = maxMemoryCacheSize

        if enableDiskCache {
            let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            self.cacheDirectory = cachesDir.appendingPathComponent("VectorAccelerate/PipelineCache", isDirectory: true)

            // Create directory if needed
            try FileManager.default.createDirectory(
                at: self.cacheDirectory!,
                withIntermediateDirectories: true
            )
        } else {
            self.cacheDirectory = nil
        }
    }

    // MARK: - Pipeline Retrieval

    /// Get or compile pipeline for a cache key
    ///
    /// Lookup order:
    /// 1. Memory cache
    /// 2. Disk cache (if enabled)
    /// 3. JIT compilation
    public func getPipeline(for key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
        // 1. Check memory cache
        if let cached = memoryCache[key] {
            hits += 1
            if let entry = metadata[key] {
                metadata[key] = entry.touched()
            }
            return cached
        }

        misses += 1

        // 2. Check if compiler already has it cached
        if await compiler.isCached(key) {
            let pipeline = try await compiler.compilePipeline(for: key)
            addToMemoryCache(key: key, pipeline: pipeline)
            return pipeline
        }

        // 3. JIT compile
        compilations += 1
        let startTime = CFAbsoluteTimeGetCurrent()
        let pipeline = try await compiler.compilePipeline(for: key)
        let compilationTime = CFAbsoluteTimeGetCurrent() - startTime

        // Add to cache
        addToMemoryCache(key: key, pipeline: pipeline, compilationTime: compilationTime)

        return pipeline
    }

    /// Get pipeline by function name (convenience)
    public func getPipeline(functionName: String) async throws -> any MTLComputePipelineState {
        let key = PipelineCacheKey(operation: functionName)
        return try await getPipeline(for: key)
    }

    /// Check if a pipeline is cached (memory or disk)
    public func isCached(_ key: PipelineCacheKey) -> Bool {
        memoryCache[key] != nil
    }

    // MARK: - Cache Management

    /// Add pipeline to memory cache with LRU eviction
    private func addToMemoryCache(
        key: PipelineCacheKey,
        pipeline: any MTLComputePipelineState,
        compilationTime: TimeInterval = 0
    ) {
        // Evict if at capacity
        if maxMemoryCacheSize > 0 && memoryCache.count >= maxMemoryCacheSize {
            evictLRU()
        }

        memoryCache[key] = pipeline
        metadata[key] = PipelineCacheEntry(key: key, compilationTime: compilationTime)
    }

    /// Evict least recently used entry
    private func evictLRU() {
        guard let oldest = metadata.values.min(by: { $0.lastAccessedAt < $1.lastAccessedAt }) else {
            return
        }
        memoryCache.removeValue(forKey: oldest.key)
        metadata.removeValue(forKey: oldest.key)
    }

    /// Pre-cache a pipeline
    public func preCache(key: PipelineCacheKey, pipeline: any MTLComputePipelineState) {
        addToMemoryCache(key: key, pipeline: pipeline)
    }

    /// Remove a specific pipeline from cache
    public func remove(_ key: PipelineCacheKey) {
        memoryCache.removeValue(forKey: key)
        metadata.removeValue(forKey: key)
    }

    /// Clear all cached pipelines
    public func clear() {
        memoryCache.removeAll()
        metadata.removeAll()
    }

    /// Clear memory cache but keep disk cache
    public func clearMemory() {
        memoryCache.removeAll()
        metadata.removeAll()
    }

    // MARK: - Warm-up

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

    /// Warm up with standard keys
    public func warmUpStandard() async {
        await warmUp(keys: PipelineCacheKey.commonKeys)
    }

    /// Warm up with embedding model keys
    public func warmUpEmbeddings() async {
        await warmUp(keys: PipelineCacheKey.embeddingModelKeys)
    }

    // MARK: - Statistics

    /// Get cache statistics
    public func getStatistics() -> PipelineCacheStatistics {
        PipelineCacheStatistics(
            hits: hits,
            misses: misses,
            compilations: compilations,
            diskLoads: diskLoads,
            diskSaves: diskSaves,
            memoryCacheSize: memoryCache.count,
            diskCacheSize: 0  // Would need to scan disk
        )
    }

    /// Reset statistics
    public func resetStatistics() {
        hits = 0
        misses = 0
        compilations = 0
        diskLoads = 0
        diskSaves = 0
    }

    /// Get all cached keys
    public var cachedKeys: [PipelineCacheKey] {
        Array(memoryCache.keys)
    }

    /// Get cache size
    public var count: Int {
        memoryCache.count
    }

    // MARK: - Metadata Access

    /// Get metadata for a cached pipeline
    public func getMetadata(for key: PipelineCacheKey) -> PipelineCacheEntry? {
        metadata[key]
    }

    /// Get most frequently accessed pipelines
    public func getMostAccessed(limit: Int = 10) -> [PipelineCacheKey] {
        metadata.values
            .sorted { $0.accessCount > $1.accessCount }
            .prefix(limit)
            .map { $0.key }
    }

    /// Get recently accessed pipelines
    public func getRecentlyAccessed(limit: Int = 10) -> [PipelineCacheKey] {
        metadata.values
            .sorted { $0.lastAccessedAt > $1.lastAccessedAt }
            .prefix(limit)
            .map { $0.key }
    }
}

// MARK: - Disk Persistence Extension

extension PipelineCache {
    /// Save cache manifest to disk
    public func saveManifest() async throws {
        guard let cacheDir = cacheDirectory else { return }

        let manifest = CacheManifest(
            version: "1.0.0",
            entries: Array(metadata.values),
            createdAt: Date()
        )

        let manifestURL = cacheDir.appendingPathComponent("manifest.json")
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL)
    }

    /// Load cache manifest from disk
    public func loadManifest() async throws -> CacheManifest? {
        guard let cacheDir = cacheDirectory else { return nil }

        let manifestURL = cacheDir.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return nil
        }

        let data = try Data(contentsOf: manifestURL)
        return try JSONDecoder().decode(CacheManifest.self, from: data)
    }

    /// Cache manifest for disk persistence
    public struct CacheManifest: Codable, Sendable {
        public let version: String
        public let entries: [PipelineCacheEntry]
        public let createdAt: Date
    }
}

// MARK: - Pipeline Cache Factory

public enum PipelineCacheFactory {
    /// Create a memory-only cache
    public static func createMemoryOnly(
        compiler: Metal4ShaderCompiler,
        maxSize: Int = 100
    ) -> PipelineCache {
        PipelineCache(compiler: compiler, cacheDirectory: nil, maxMemoryCacheSize: maxSize)
    }

    /// Create a cache with disk persistence
    public static func createPersistent(
        compiler: Metal4ShaderCompiler,
        maxMemorySize: Int = 100
    ) throws -> PipelineCache {
        try PipelineCache(compiler: compiler, enableDiskCache: true, maxMemoryCacheSize: maxMemorySize)
    }

    /// Create a development cache (no limits, no disk)
    public static func createDevelopment(
        compiler: Metal4ShaderCompiler
    ) -> PipelineCache {
        PipelineCache(compiler: compiler, cacheDirectory: nil, maxMemoryCacheSize: 0)
    }
}
