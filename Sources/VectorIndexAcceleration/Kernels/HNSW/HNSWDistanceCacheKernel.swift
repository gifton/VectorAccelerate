//
//  HNSWDistanceCacheKernel.swift
//  VectorIndexAcceleration
//
//  Metal 4 kernel for HNSW distance cache management.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import simd
import QuartzCore
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Cache Update Batch

/// Batch of distance cache updates.
public struct CacheUpdateBatch: Sendable {
    /// Distances to cache
    public let distances: [Float]

    /// Node pairs (nodeA, nodeB)
    public let nodePairs: [(UInt32, UInt32)]

    public init(distances: [Float], nodePairs: [(UInt32, UInt32)]) {
        precondition(distances.count == nodePairs.count,
                    "Distances and node pairs must have same count")
        self.distances = distances
        self.nodePairs = nodePairs
    }

    /// Number of updates
    public var count: Int { distances.count }
}

// MARK: - Cache Update Result

/// Result from cache update operation.
public struct CacheUpdateResult: Sendable {
    /// Number of updates processed
    public let updatesProcessed: Int

    /// Execution time
    public let executionTime: TimeInterval

    /// Throughput in operations per second
    public let throughputOps: Double?

    /// Cache utilization percentage
    public let cacheUtilization: Float?

    public init(
        updatesProcessed: Int,
        executionTime: TimeInterval,
        throughputOps: Double? = nil,
        cacheUtilization: Float? = nil
    ) {
        self.updatesProcessed = updatesProcessed
        self.executionTime = executionTime
        self.throughputOps = throughputOps ?? (
            executionTime > 0 ? Double(updatesProcessed * 2) / executionTime : nil
        )
        self.cacheUtilization = cacheUtilization
    }
}

// MARK: - HNSW Distance Cache Kernel

/// Metal 4 kernel for GPU-accelerated distance cache management.
///
/// Provides GPU-resident distance caching for HNSW neighbor connections
/// with symmetric update support.
///
/// ## Features
/// - GPU-resident cache structure
/// - Atomic counter management
/// - Symmetric (bidirectional) updates
/// - Cache utilization tracking
///
/// ## Usage
/// ```swift
/// let kernel = try await HNSWDistanceCacheKernel(context: context, configuration: config)
/// let cache = try await kernel.createCache()
/// let result = try await kernel.updateCache(cache, batch: updates)
/// ```
public final class HNSWDistanceCacheKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "HNSWDistanceCacheKernel"

    // MARK: - Properties

    /// The configuration used by this kernel
    public let configuration: DistanceCacheConfiguration

    private let pipeline: any MTLComputePipelineState

    // MARK: - Initialization

    public init(
        context: Metal4Context,
        configuration: DistanceCacheConfiguration
    ) async throws {
        self.context = context
        self.configuration = configuration

        // Load library and create pipeline
        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let function = library.makeFunction(name: "hnsw_update_distance_cache") else {
            throw VectorError.shaderNotFound(
                name: "hnsw_update_distance_cache. Ensure HNSW shaders are compiled."
            )
        }

        self.pipeline = try await context.device.rawDevice.makeComputePipelineState(function: function)
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        // Pipeline is created in init
    }

    // MARK: - Cache Creation

    /// Create a new distance cache.
    ///
    /// - Returns: The created cache
    public func createCache() async throws -> DistanceCache {
        let N = configuration.totalNodes
        let C = configuration.cacheCapacity

        let device = context.device.rawDevice

        // Allocate distance buffer [N, C]
        let distanceSize = N * C * MemoryLayout<Float>.size
        guard let distanceBuffer = device.makeBuffer(
            length: distanceSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: distanceSize)
        }
        distanceBuffer.label = "HNSWDistanceCache.distances"
        distanceBuffer.contents().initializeMemory(
            as: Float.self,
            repeating: Float.infinity,
            count: N * C
        )

        // Allocate indices buffer [N, C]
        let indicesSize = N * C * MemoryLayout<UInt32>.size
        guard let indicesBuffer = device.makeBuffer(
            length: indicesSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: indicesSize)
        }
        indicesBuffer.label = "HNSWDistanceCache.indices"
        indicesBuffer.contents().initializeMemory(
            as: UInt32.self,
            repeating: UInt32.max,
            count: N * C
        )

        // Allocate counters buffer [N]
        let countersSize = N * MemoryLayout<UInt32>.size
        guard let countersBuffer = device.makeBuffer(
            length: countersSize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: countersSize)
        }
        countersBuffer.label = "HNSWDistanceCache.counters"
        countersBuffer.contents().initializeMemory(
            as: UInt32.self,
            repeating: 0,
            count: N
        )

        return DistanceCache(
            distanceBuffer: distanceBuffer,
            indicesBuffer: indicesBuffer,
            countersBuffer: countersBuffer,
            configuration: configuration
        )
    }

    // MARK: - Cache Updates

    /// Update distance cache with batch of new distances.
    ///
    /// - Parameters:
    ///   - cache: The cache to update
    ///   - batch: The updates to apply
    /// - Returns: Update result
    public func updateCache(
        _ cache: DistanceCache,
        batch: CacheUpdateBatch
    ) async throws -> CacheUpdateResult {
        let startTime = CACurrentMediaTime()

        guard batch.count > 0 else {
            return CacheUpdateResult(
                updatesProcessed: 0,
                executionTime: 0
            )
        }

        let device = context.device.rawDevice

        // Create distances buffer
        guard let distancesBuffer = device.makeBuffer(
            bytes: batch.distances,
            length: batch.distances.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: batch.distances.count * MemoryLayout<Float>.size)
        }
        distancesBuffer.label = "HNSWDistanceCache.updateDistances"

        // Convert node pairs to SIMD2<UInt32> format
        let uint2Pairs = batch.nodePairs.map { pair in
            SIMD2<UInt32>(pair.0, pair.1)
        }
        guard let pairsBuffer = device.makeBuffer(
            bytes: uint2Pairs,
            length: uint2Pairs.count * MemoryLayout<SIMD2<UInt32>>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: uint2Pairs.count * MemoryLayout<SIMD2<UInt32>>.size)
        }
        pairsBuffer.label = "HNSWDistanceCache.updatePairs"

        // Execute kernel
        try await context.executeAndWait { [self] _, encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.label = "Distance Cache Update"

            encoder.setBuffer(cache.distanceBuffer, offset: 0, index: 0)
            encoder.setBuffer(cache.indicesBuffer, offset: 0, index: 1)
            encoder.setBuffer(distancesBuffer, offset: 0, index: 2)
            encoder.setBuffer(pairsBuffer, offset: 0, index: 3)
            encoder.setBuffer(cache.countersBuffer, offset: 0, index: 4)

            var B = UInt32(batch.count)
            var C = UInt32(cache.configuration.cacheCapacity)

            encoder.setBytes(&B, length: MemoryLayout<UInt32>.size, index: 5)
            encoder.setBytes(&C, length: MemoryLayout<UInt32>.size, index: 6)

            let threadsPerGroup = configuration.threadgroupSize
            let threadgroups = (batch.count + threadsPerGroup - 1) / threadsPerGroup

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
        }

        let executionTime = CACurrentMediaTime() - startTime
        let throughput = Double(batch.count * 2) / executionTime

        // Calculate cache utilization if profiling enabled
        let utilization: Float? = configuration.enableProfiling ?
            calculateCacheUtilization(cache) : nil

        return CacheUpdateResult(
            updatesProcessed: batch.count * 2,
            executionTime: executionTime,
            throughputOps: throughput,
            cacheUtilization: utilization
        )
    }

    /// Batch update with multiple update batches.
    ///
    /// - Parameters:
    ///   - cache: The cache to update
    ///   - batches: Multiple update batches
    /// - Returns: Results for each batch
    public func batchUpdate(
        _ cache: DistanceCache,
        batches: [CacheUpdateBatch]
    ) async throws -> [CacheUpdateResult] {
        guard !batches.isEmpty else { return [] }

        var results: [CacheUpdateResult] = []
        for batch in batches {
            let result = try await updateCache(cache, batch: batch)
            results.append(result)
        }
        return results
    }

    // MARK: - Private Helpers

    private func calculateCacheUtilization(_ cache: DistanceCache) -> Float {
        let N = cache.configuration.totalNodes
        let C = cache.configuration.cacheCapacity

        let distPtr = cache.distanceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: N * C
        )

        var filledSlots = 0
        for i in 0..<(N * C) {
            if distPtr[i] != Float.infinity {
                filledSlots += 1
            }
        }

        return Float(filledSlots) / Float(N * C) * 100
    }
}

// MARK: - Distance Cache Extensions

extension DistanceCache {
    /// Get cache entry for a node (CPU operation).
    public func getCacheEntry(nodeID: Int, slot: Int) -> (distance: Float, neighborID: UInt32)? {
        let C = configuration.cacheCapacity
        guard nodeID < configuration.totalNodes, slot < C else { return nil }

        let index = nodeID * C + slot

        let distPtr = distanceBuffer.contents().bindMemory(
            to: Float.self,
            capacity: configuration.totalNodes * C
        )
        let idPtr = indicesBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: configuration.totalNodes * C
        )

        return (distPtr[index], idPtr[index])
    }

    /// Get current insertion counter for a node.
    public func getInsertionCount(nodeID: Int) -> UInt32 {
        guard nodeID < configuration.totalNodes else { return 0 }

        let ptr = countersBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: configuration.totalNodes
        )
        return ptr[nodeID]
    }

    /// Reset all counters (CPU operation).
    public func resetCounters() {
        let ptr = countersBuffer.contents().bindMemory(
            to: UInt32.self,
            capacity: configuration.totalNodes
        )
        for i in 0..<configuration.totalNodes {
            ptr[i] = 0
        }
    }

    /// Get all valid cache entries for a node.
    public func getNodeCache(nodeID: Int) -> [(distance: Float, neighborID: UInt32)] {
        let C = configuration.cacheCapacity
        guard nodeID < configuration.totalNodes else { return [] }

        var entries: [(Float, UInt32)] = []
        let insertionCount = getInsertionCount(nodeID: nodeID)
        let validSlots = min(Int(insertionCount), C)

        for slot in 0..<validSlots {
            if let entry = getCacheEntry(nodeID: nodeID, slot: slot),
               entry.distance != Float.infinity {
                entries.append(entry)
            }
        }

        return entries
    }
}
