//
//  HNSWKernels.swift
//  VectorIndexAcceleration
//
//  Base types, protocols, and shared utilities for HNSW Metal 4 kernels.
//
//  Migrated from VectorIndexAccelerated Phase 2.
//

import Foundation
@preconcurrency import Metal
import VectorAccelerate
import VectorIndex
import VectorCore

// MARK: - Design Overview
//
// This file defines two categories of types that serve distinct purposes:
//
// ## Configuration Types (*Configuration)
// Developer-facing types used to construct and customize kernel behavior.
// - Stored in kernels at initialization time
// - Use Swift-native types (Int, Bool, Float)
// - Include validation methods and static presets
// - Control threading, feature flags, and algorithm parameters
// - Examples: HNSWSearchConfiguration, HNSWLevelConfiguration
//
// ## Shader Argument Types (*ShaderArgs)
// GPU-bound types passed directly to Metal shaders via setBytes().
// - Created fresh for each dispatch operation
// - Use UInt32 to match Metal shader expectations
// - Include explicit _padding fields for memory alignment
// - Memory layout MUST match corresponding Metal shader structs
// - Examples: HNSWSearchShaderArgs, HNSWDistanceMatrixShaderArgs
//
// These are intentionally separate: Configuration controls HOW a kernel
// operates, while ShaderArgs specify WHAT to compute on each dispatch.

// MARK: - Shader Arguments
//
// These structs are passed directly to Metal shaders. Their memory layout
// must match the corresponding structs in HNSWShaders.metal.
// All use explicit padding for alignment.

// MARK: - HNSW Search Shader Args

/// Shader arguments for HNSW search operations.
/// Memory layout matches `HNSWSearchArgs` in HNSWShaders.metal.
public struct HNSWSearchShaderArgs: Sendable {
    /// Query vector dimension
    public let dimension: UInt32

    /// Number of candidates to explore (ef parameter)
    public let efSearch: UInt32

    /// Number of results to return
    public let k: UInt32

    /// Current layer being searched
    public let layer: UInt32

    /// Maximum connections per node
    public let maxConnections: UInt32

    /// Padding for alignment
    private let _padding: (UInt32, UInt32, UInt32) = (0, 0, 0)

    public init(
        dimension: Int,
        efSearch: Int,
        k: Int,
        layer: Int,
        maxConnections: Int
    ) {
        self.dimension = UInt32(dimension)
        self.efSearch = UInt32(efSearch)
        self.k = UInt32(k)
        self.layer = UInt32(layer)
        self.maxConnections = UInt32(maxConnections)
    }
}

// MARK: - HNSW Distance Matrix Shader Args

/// Shader arguments for HNSW distance matrix computation.
/// Memory layout matches `HNSWDistanceMatrixArgs` in HNSWShaders.metal.
public struct HNSWDistanceMatrixShaderArgs: Sendable {
    /// Query vector dimension
    public let dimension: UInt32

    /// Number of query vectors
    public let numQueries: UInt32

    /// Number of candidate vectors
    public let numCandidates: UInt32

    /// Distance metric (0 = L2, 1 = Cosine, 2 = DotProduct)
    public let metric: UInt32

    /// Tile size for shared memory caching
    public let tileSize: UInt32

    /// Padding for alignment
    private let _padding: (UInt32, UInt32, UInt32) = (0, 0, 0)

    public init(
        dimension: Int,
        numQueries: Int,
        numCandidates: Int,
        metric: SupportedDistanceMetric = .euclidean,
        tileSize: Int = 32
    ) {
        self.dimension = UInt32(dimension)
        self.numQueries = UInt32(numQueries)
        self.numCandidates = UInt32(numCandidates)
        self.metric = metric.metalRawValue
        self.tileSize = UInt32(tileSize)
    }
}

// MARK: - HNSW Level Configuration

/// Configuration for HNSW level assignment operations.
public struct HNSWLevelConfiguration: Sendable {
    /// Number of connections per node (M parameter)
    public let M: Int

    /// Normalization factor: 1/ln(M)
    public let mlFactor: Float

    /// Maximum hierarchy level
    public let maxLevel: Int

    /// Global random seed
    public let globalSeed: UInt32

    /// Use per-node seeds for reproducibility
    public let useNodeSeeds: Bool

    /// Track probabilities for debugging
    public let trackProbabilities: Bool

    /// Threads per threadgroup
    public let threadgroupSize: Int

    public init(
        M: Int = 16,
        maxLevel: Int = 16,
        globalSeed: UInt32 = UInt32.random(in: 0..<UInt32.max),
        useNodeSeeds: Bool = true,
        trackProbabilities: Bool = false,
        threadgroupSize: Int = 256
    ) {
        self.M = M
        self.mlFactor = 1.0 / log(Float(M))
        self.maxLevel = maxLevel
        self.globalSeed = globalSeed
        self.useNodeSeeds = useNodeSeeds
        self.trackProbabilities = trackProbabilities
        self.threadgroupSize = threadgroupSize
    }

    /// Create with explicit mlFactor
    public init(
        mlFactor: Float,
        maxLevel: Int = 16,
        globalSeed: UInt32 = UInt32.random(in: 0..<UInt32.max),
        useNodeSeeds: Bool = true,
        trackProbabilities: Bool = false,
        threadgroupSize: Int = 256
    ) {
        self.M = Int(exp(1.0 / mlFactor))
        self.mlFactor = mlFactor
        self.maxLevel = maxLevel
        self.globalSeed = globalSeed
        self.useNodeSeeds = useNodeSeeds
        self.trackProbabilities = trackProbabilities
        self.threadgroupSize = threadgroupSize
    }

    /// Validate configuration parameters
    public func validate() throws {
        guard M >= 2 else {
            throw IndexAccelerationError.invalidInput(message: "M must be at least 2, got \(M)")
        }
        guard M <= 128 else {
            throw IndexAccelerationError.invalidInput(message: "M must be at most 128, got \(M)")
        }
        guard maxLevel >= 1 else {
            throw IndexAccelerationError.invalidInput(message: "maxLevel must be at least 1, got \(maxLevel)")
        }
        guard maxLevel <= 32 else {
            throw IndexAccelerationError.invalidInput(message: "maxLevel must be at most 32, got \(maxLevel)")
        }
        guard threadgroupSize > 0 && threadgroupSize <= 1024 else {
            throw IndexAccelerationError.invalidInput(message: "threadgroupSize must be 1-1024, got \(threadgroupSize)")
        }
    }

    // MARK: - Presets

    /// Small index preset (M=8) - lower memory, faster construction, lower recall
    public static func small(seed: UInt32 = UInt32.random(in: 0..<UInt32.max)) -> HNSWLevelConfiguration {
        HNSWLevelConfiguration(M: 8, maxLevel: 12, globalSeed: seed)
    }

    /// Standard preset (M=16) - balanced memory/quality tradeoff
    public static func standard(seed: UInt32 = UInt32.random(in: 0..<UInt32.max)) -> HNSWLevelConfiguration {
        HNSWLevelConfiguration(M: 16, maxLevel: 16, globalSeed: seed)
    }

    /// Large index preset (M=32) - higher memory, better recall
    public static func highQuality(seed: UInt32 = UInt32.random(in: 0..<UInt32.max)) -> HNSWLevelConfiguration {
        HNSWLevelConfiguration(M: 32, maxLevel: 20, globalSeed: seed)
    }

    /// Maximum quality preset (M=48) - highest memory, best recall
    public static func maximum(seed: UInt32 = UInt32.random(in: 0..<UInt32.max)) -> HNSWLevelConfiguration {
        HNSWLevelConfiguration(M: 48, maxLevel: 24, globalSeed: seed)
    }
}

// MARK: - HNSW Edge Insertion Configuration

/// Configuration for HNSW edge insertion operations.
///
/// Controls how edges are inserted into the HNSW graph, including
/// bidirectionality, capacity management, and GPU dispatch settings.
public struct HNSWEdgeInsertionConfiguration: Sendable {
    /// Insert reverse edges for bidirectional graph
    public let bidirectional: Bool

    /// Allow self-connections (usually false for HNSW)
    public let allowSelfLoops: Bool

    /// Initial edge capacity per node
    public let initialCapacity: Int

    /// Growth factor when reallocating capacity (must be > 1.0)
    public let capacityGrowthFactor: Float

    /// Threadgroup width for 2D dispatch
    public let threadgroupWidth: Int

    /// Threadgroup height for 2D dispatch
    public let threadgroupHeight: Int

    public init(
        bidirectional: Bool = true,
        allowSelfLoops: Bool = false,
        initialCapacity: Int = 32,
        capacityGrowthFactor: Float = 1.5,
        threadgroupWidth: Int = 16,
        threadgroupHeight: Int = 16
    ) {
        self.bidirectional = bidirectional
        self.allowSelfLoops = allowSelfLoops
        self.initialCapacity = initialCapacity
        self.capacityGrowthFactor = max(1.1, capacityGrowthFactor)
        self.threadgroupWidth = min(max(1, threadgroupWidth), 32)
        self.threadgroupHeight = min(max(1, threadgroupHeight), 32)
    }

    /// Total threads per threadgroup
    public var threadsPerThreadgroup: Int {
        threadgroupWidth * threadgroupHeight
    }

    /// Validate configuration parameters
    public func validate() throws {
        guard initialCapacity >= 1 else {
            throw IndexAccelerationError.invalidInput(message: "initialCapacity must be at least 1, got \(initialCapacity)")
        }
        guard initialCapacity <= 1024 else {
            throw IndexAccelerationError.invalidInput(message: "initialCapacity must be at most 1024, got \(initialCapacity)")
        }
        guard capacityGrowthFactor > 1.0 else {
            throw IndexAccelerationError.invalidInput(message: "capacityGrowthFactor must be > 1.0, got \(capacityGrowthFactor)")
        }
        guard capacityGrowthFactor <= 4.0 else {
            throw IndexAccelerationError.invalidInput(message: "capacityGrowthFactor must be <= 4.0, got \(capacityGrowthFactor)")
        }
        guard threadsPerThreadgroup <= 1024 else {
            throw IndexAccelerationError.invalidInput(message: "Total threads per threadgroup must be <= 1024, got \(threadsPerThreadgroup)")
        }
    }

    // MARK: - Presets

    /// Default preset for most use cases
    public static let `default` = HNSWEdgeInsertionConfiguration()

    /// Preset for small graphs (< 10K nodes)
    public static let smallGraph = HNSWEdgeInsertionConfiguration(
        initialCapacity: 16,
        capacityGrowthFactor: 2.0
    )

    /// Preset for large graphs (> 100K nodes) - larger initial capacity to reduce reallocations
    public static let largeGraph = HNSWEdgeInsertionConfiguration(
        initialCapacity: 64,
        capacityGrowthFactor: 1.5
    )

    /// Preset for directed graphs (no reverse edges)
    public static let directed = HNSWEdgeInsertionConfiguration(
        bidirectional: false
    )
}

// MARK: - HNSW Pruning Configuration

/// Configuration for HNSW edge pruning operations.
public struct HNSWPruningConfiguration: Sendable {
    /// Pruning strategy enumeration
    public enum PruningStrategy: Sendable {
        case distance       // Prune furthest edges
        case heuristic      // Use HNSW heuristic
        case random         // Random pruning
        case custom         // User-provided flags
    }

    /// Maximum edges per node (must be ≤ 256)
    public let maxDegree: Int

    /// Threadgroup size (must be 256 for Blelloch scan)
    public let threadgroupSize: Int

    /// Maintain distance values during pruning
    public let preserveDistances: Bool

    /// Pruning strategy to use
    public let pruningStrategy: PruningStrategy

    public init(
        maxDegree: Int = 256,
        threadgroupSize: Int = 256,
        preserveDistances: Bool = true,
        pruningStrategy: PruningStrategy = .heuristic
    ) {
        self.maxDegree = min(maxDegree, 256)
        self.threadgroupSize = 256  // Fixed for Blelloch scan
        self.preserveDistances = preserveDistances
        self.pruningStrategy = pruningStrategy
    }

    /// Validate configuration parameters
    public func validate() throws {
        guard maxDegree >= 1 else {
            throw IndexAccelerationError.invalidInput(message: "maxDegree must be at least 1, got \(maxDegree)")
        }
        guard maxDegree <= 256 else {
            throw IndexAccelerationError.invalidInput(message: "maxDegree must be at most 256, got \(maxDegree)")
        }
        // threadgroupSize is fixed at 256 for Blelloch scan, no validation needed
    }

    // MARK: - Presets

    /// Default preset with heuristic pruning
    public static let `default` = HNSWPruningConfiguration()

    /// Distance-based pruning - keeps nearest neighbors
    public static let distanceBased = HNSWPruningConfiguration(
        pruningStrategy: .distance
    )

    /// Fast pruning without distance preservation
    public static let fast = HNSWPruningConfiguration(
        preserveDistances: false,
        pruningStrategy: .distance
    )

    /// Preset for M=16 HNSW (maxDegree = 2*M = 32)
    public static func forM(_ M: Int) -> HNSWPruningConfiguration {
        HNSWPruningConfiguration(maxDegree: min(M * 2, 256))
    }
}

// MARK: - HNSW Search Configuration

/// Configuration for HNSW search operations.
public struct HNSWSearchConfiguration: Sendable {
    /// Search parameter (size of dynamic list)
    public let efSearch: Int

    /// Maximum vector dimension supported
    public let maxDimension: Int

    /// Maximum ef_search value
    public let maxEF: Int

    /// Threads per threadgroup
    public let threadgroupSize: Int

    /// Track performance metrics
    public let enableProfiling: Bool

    public init(
        efSearch: Int = 64,
        maxDimension: Int = 512,
        maxEF: Int = 128,
        threadgroupSize: Int = 256,
        enableProfiling: Bool = false
    ) {
        self.efSearch = min(efSearch, maxEF)
        self.maxDimension = min(maxDimension, 512)
        self.maxEF = min(maxEF, 128)
        self.threadgroupSize = min(threadgroupSize, 256)
        self.enableProfiling = enableProfiling
    }

    /// Validate configuration parameters
    public func validate() throws {
        guard efSearch >= 1 else {
            throw IndexAccelerationError.invalidInput(message: "efSearch must be at least 1, got \(efSearch)")
        }
        guard efSearch <= maxEF else {
            throw IndexAccelerationError.invalidInput(message: "efSearch (\(efSearch)) must be <= maxEF (\(maxEF))")
        }
        guard maxDimension >= 1 else {
            throw IndexAccelerationError.invalidInput(message: "maxDimension must be at least 1, got \(maxDimension)")
        }
        guard threadgroupSize > 0 && threadgroupSize <= 1024 else {
            throw IndexAccelerationError.invalidInput(message: "threadgroupSize must be 1-1024, got \(threadgroupSize)")
        }
    }

    // MARK: - Presets

    /// Default preset - balanced speed/recall
    public static let `default` = HNSWSearchConfiguration()

    /// Fast search preset (ef=32) - lower recall, higher speed
    public static let fast = HNSWSearchConfiguration(efSearch: 32, maxEF: 64)

    /// High recall preset (ef=128) - higher recall, lower speed
    public static let highRecall = HNSWSearchConfiguration(efSearch: 128, maxEF: 256)

    /// Maximum recall preset (ef=256) - best recall, lowest speed
    public static let maxRecall = HNSWSearchConfiguration(efSearch: 256, maxEF: 512)

    /// Preset for high-dimensional vectors (768+)
    public static let highDimensional = HNSWSearchConfiguration(
        efSearch: 64,
        maxDimension: 1024,
        maxEF: 128
    )

    /// Preset with profiling enabled
    public static let profiled = HNSWSearchConfiguration(enableProfiling: true)
}

// MARK: - Visited Set Configuration

/// Configuration for visited set operations.
public struct VisitedSetConfiguration: Sendable {
    /// Use uint4 vectorization for better throughput
    public let useVectorization: Bool

    /// Threads per threadgroup
    public let threadgroupSize: Int

    /// Track performance metrics
    public let enableProfiling: Bool

    public init(
        useVectorization: Bool = true,
        threadgroupSize: Int = 256,
        enableProfiling: Bool = false
    ) {
        self.useVectorization = useVectorization
        self.threadgroupSize = threadgroupSize
        self.enableProfiling = enableProfiling
    }

    /// Validate configuration parameters
    public func validate() throws {
        guard threadgroupSize > 0 && threadgroupSize <= 1024 else {
            throw IndexAccelerationError.invalidInput(message: "threadgroupSize must be 1-1024, got \(threadgroupSize)")
        }
    }

    // MARK: - Presets

    /// Default preset with vectorization enabled
    public static let `default` = VisitedSetConfiguration()

    /// Preset with profiling enabled
    public static let profiled = VisitedSetConfiguration(enableProfiling: true)

    /// Preset without vectorization (for debugging)
    public static let noVectorization = VisitedSetConfiguration(useVectorization: false)
}

// MARK: - Distance Cache Configuration

/// Configuration for distance cache operations.
public struct DistanceCacheConfiguration: Sendable {
    /// Cache capacity per node
    public let cacheCapacity: Int

    /// Total number of nodes
    public let totalNodes: Int

    /// Use LRU eviction (future)
    public let useLRU: Bool

    /// Maintain sorted order (future)
    public let maintainSorted: Bool

    /// Threads per threadgroup
    public let threadgroupSize: Int

    /// Track performance metrics
    public let enableProfiling: Bool

    public init(
        cacheCapacity: Int = 64,
        totalNodes: Int,
        useLRU: Bool = false,
        maintainSorted: Bool = false,
        threadgroupSize: Int = 256,
        enableProfiling: Bool = false
    ) {
        self.cacheCapacity = cacheCapacity
        self.totalNodes = totalNodes
        self.useLRU = useLRU
        self.maintainSorted = maintainSorted
        self.threadgroupSize = threadgroupSize
        self.enableProfiling = enableProfiling
    }

    /// Validate configuration parameters
    public func validate() throws {
        guard cacheCapacity >= 1 else {
            throw IndexAccelerationError.invalidInput(message: "cacheCapacity must be at least 1, got \(cacheCapacity)")
        }
        guard cacheCapacity <= 256 else {
            throw IndexAccelerationError.invalidInput(message: "cacheCapacity must be at most 256, got \(cacheCapacity)")
        }
        guard totalNodes >= 1 else {
            throw IndexAccelerationError.invalidInput(message: "totalNodes must be at least 1, got \(totalNodes)")
        }
        guard threadgroupSize > 0 && threadgroupSize <= 1024 else {
            throw IndexAccelerationError.invalidInput(message: "threadgroupSize must be 1-1024, got \(threadgroupSize)")
        }
    }

    /// Memory footprint in bytes for this configuration
    public var estimatedMemoryBytes: Int {
        // distances [N, C] + indices [N, C] + counters [N]
        let distanceBytes = totalNodes * cacheCapacity * MemoryLayout<Float>.size
        let indicesBytes = totalNodes * cacheCapacity * MemoryLayout<UInt32>.size
        let countersBytes = totalNodes * MemoryLayout<UInt32>.size
        return distanceBytes + indicesBytes + countersBytes
    }

    // MARK: - Presets

    /// Small cache preset (32 slots per node)
    public static func small(totalNodes: Int) -> DistanceCacheConfiguration {
        DistanceCacheConfiguration(cacheCapacity: 32, totalNodes: totalNodes)
    }

    /// Standard cache preset (64 slots per node)
    public static func standard(totalNodes: Int) -> DistanceCacheConfiguration {
        DistanceCacheConfiguration(cacheCapacity: 64, totalNodes: totalNodes)
    }

    /// Large cache preset (128 slots per node)
    public static func large(totalNodes: Int) -> DistanceCacheConfiguration {
        DistanceCacheConfiguration(cacheCapacity: 128, totalNodes: totalNodes)
    }

    /// Preset with profiling enabled
    public static func profiled(totalNodes: Int, cacheCapacity: Int = 64) -> DistanceCacheConfiguration {
        DistanceCacheConfiguration(cacheCapacity: cacheCapacity, totalNodes: totalNodes, enableProfiling: true)
    }
}

// MARK: - Distance Metric Extension

extension SupportedDistanceMetric {
    /// Convert to Metal kernel constant
    var metalRawValue: UInt32 {
        switch self {
        case .euclidean:
            return 0
        case .cosine:
            return 1
        case .dotProduct:
            return 2
        case .manhattan:
            return 3
        case .chebyshev:
            return 4
        }
    }
}

// MARK: - HNSW Graph Layer

/// Represents an immutable HNSW graph layer for GPU operations.
///
/// This struct holds references to GPU buffers containing the graph structure.
/// The buffers themselves are reference types, so copying this struct is cheap.
///
/// ## Memory Layout
/// - `vectors`: Node vectors in row-major format [N × D]
/// - `edges`: CSR edge data (neighbor indices)
/// - `edgeOffsets`: CSR row pointers [N+1]
public struct HNSWGraphLayer: Sendable {
    /// Node vectors [N × D]
    public let vectors: any MTLBuffer

    /// CSR edge data (neighbor indices)
    public let edges: any MTLBuffer

    /// CSR offsets [N+1] - edgeOffsets[i] is the start index for node i's edges
    public let edgeOffsets: any MTLBuffer

    /// Number of nodes in this layer
    public let nodeCount: Int

    /// Vector dimension
    public let dimension: Int

    public init(
        vectors: any MTLBuffer,
        edges: any MTLBuffer,
        edgeOffsets: any MTLBuffer,
        nodeCount: Int,
        dimension: Int
    ) {
        self.vectors = vectors
        self.edges = edges
        self.edgeOffsets = edgeOffsets
        self.nodeCount = nodeCount
        self.dimension = dimension
    }

    /// Total number of edges in this layer
    public var edgeCount: Int {
        guard nodeCount > 0 else { return 0 }
        let ptr = edgeOffsets.contents().bindMemory(to: UInt32.self, capacity: nodeCount + 1)
        return Int(ptr[nodeCount])
    }
}

// MARK: - HNSW Mutable Graph

/// Mutable HNSW graph structure for edge insertion operations.
///
/// This struct manages GPU buffers for a graph that supports atomic edge insertion.
/// The buffers themselves are reference types, so the graph contents can be modified
/// through GPU operations even though this is a struct.
///
/// ## Buffer Layout
/// - `edges`: Flattened edge storage, indexed via `edgeOffsets`
/// - `edgeCounts`: Atomic counters tracking current edge count per node
/// - `edgeCapacity`: Maximum edges allocated per node
/// - `edgeOffsets`: CSR row pointers for edge lookup
///
/// ## Usage
/// ```swift
/// let graph = try await kernel.createGraph(nodeCount: 10000)
/// let result = try await kernel.insertEdges(batch: edges, graph: graph)
/// if result.needsReallocation {
///     graph = try await kernel.reallocateGraph(graph: graph, ...)
/// }
/// ```
public struct HNSWMutableGraph: Sendable {
    /// CSR edge data (neighbor indices)
    public let edges: any MTLBuffer

    /// Current edge counts per node (atomic counters)
    public let edgeCounts: any MTLBuffer

    /// Allocated capacity per node
    public let edgeCapacity: any MTLBuffer

    /// CSR row pointers
    public let edgeOffsets: any MTLBuffer

    /// Number of nodes in the graph
    public let nodeCount: Int

    /// Total edge slots allocated
    public let totalCapacity: Int

    public init(
        edges: any MTLBuffer,
        edgeCounts: any MTLBuffer,
        edgeCapacity: any MTLBuffer,
        edgeOffsets: any MTLBuffer,
        nodeCount: Int,
        totalCapacity: Int
    ) {
        self.edges = edges
        self.edgeCounts = edgeCounts
        self.edgeCapacity = edgeCapacity
        self.edgeOffsets = edgeOffsets
        self.nodeCount = nodeCount
        self.totalCapacity = totalCapacity
    }

    /// Current total edge count across all nodes (CPU read)
    public var currentEdgeCount: Int {
        let ptr = edgeCounts.contents().bindMemory(to: UInt32.self, capacity: nodeCount)
        var total = 0
        for i in 0..<nodeCount {
            total += Int(ptr[i])
        }
        return total
    }

    /// Memory footprint in bytes
    public var sizeInBytes: Int {
        edges.length + edgeCounts.length + edgeCapacity.length + edgeOffsets.length
    }
}

// MARK: - Visited Set

/// Bit-packed visited set for tracking visited nodes during graph traversal.
///
/// Uses a compact bit-array representation where each node is represented by a single bit.
/// The underlying buffer is 16-byte aligned for efficient vectorized GPU operations.
///
/// ## Memory Layout
/// - 1 bit per node, packed into 32-bit words
/// - Word i contains nodes [i*32, i*32+31]
/// - Bit j within word represents node (wordIndex * 32 + j)
///
/// ## Thread Safety
/// CPU operations (`isVisited`, `markVisited`, `countVisited`) are not thread-safe.
/// For concurrent access, use GPU kernels or external synchronization.
public struct VisitedSet: Sendable {
    /// GPU buffer containing bit-packed flags
    public let buffer: any MTLBuffer

    /// Total number of nodes this set can track
    public let nodeCount: Int

    /// Number of 32-bit words in the buffer
    public let wordCount: Int

    public init(buffer: any MTLBuffer, nodeCount: Int) {
        self.buffer = buffer
        self.nodeCount = nodeCount
        self.wordCount = (nodeCount + 31) / 32
    }

    /// Size in bytes
    public var sizeInBytes: Int {
        wordCount * MemoryLayout<UInt32>.size
    }

    /// Check if a node is visited (CPU operation - not thread-safe)
    public func isVisited(_ nodeID: Int) -> Bool {
        guard nodeID >= 0, nodeID < nodeCount else { return false }

        let wordIndex = nodeID / 32
        let bitIndex = nodeID % 32

        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: wordCount)
        return (ptr[wordIndex] & (1 << bitIndex)) != 0
    }

    /// Mark a node as visited (CPU operation - not thread-safe)
    ///
    /// Note: This mutates the buffer contents, not the struct itself.
    public func markVisited(_ nodeID: Int) {
        guard nodeID >= 0, nodeID < nodeCount else { return }

        let wordIndex = nodeID / 32
        let bitIndex = nodeID % 32

        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: wordCount)
        ptr[wordIndex] |= (1 << bitIndex)
    }

    /// Count total visited nodes (CPU operation)
    public func countVisited() -> Int {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: wordCount)

        var count = 0
        for i in 0..<wordCount {
            count += ptr[i].nonzeroBitCount
        }
        return count
    }

    /// Clear all visited flags (CPU operation)
    public func clear() {
        let ptr = buffer.contents().bindMemory(to: UInt32.self, capacity: wordCount)
        for i in 0..<wordCount {
            ptr[i] = 0
        }
    }
}

// MARK: - Distance Cache

/// GPU-resident distance cache for HNSW neighbor connections.
///
/// Caches computed distances to avoid redundant computation during graph traversal.
/// Each node has a fixed-size cache of its most recently computed distances.
///
/// ## Buffer Layout
/// - `distanceBuffer`: [N, C] - Distance values, initialized to infinity
/// - `indicesBuffer`: [N, C] - Neighbor node indices, initialized to UInt32.max
/// - `countersBuffer`: [N] - Insertion counters for round-robin replacement
///
/// Where N = totalNodes and C = cacheCapacity.
public struct DistanceCache: Sendable {
    /// Distance values [N, C]
    public let distanceBuffer: any MTLBuffer

    /// Neighbor node indices [N, C]
    public let indicesBuffer: any MTLBuffer

    /// Insertion counters [N] for round-robin slot selection
    public let countersBuffer: any MTLBuffer

    /// Configuration used to create this cache
    public let configuration: DistanceCacheConfiguration

    public init(
        distanceBuffer: any MTLBuffer,
        indicesBuffer: any MTLBuffer,
        countersBuffer: any MTLBuffer,
        configuration: DistanceCacheConfiguration
    ) {
        self.distanceBuffer = distanceBuffer
        self.indicesBuffer = indicesBuffer
        self.countersBuffer = countersBuffer
        self.configuration = configuration
    }

    /// Total memory footprint in bytes
    public var sizeInBytes: Int {
        distanceBuffer.length + indicesBuffer.length + countersBuffer.length
    }

    /// Number of cache slots per node
    public var slotsPerNode: Int {
        configuration.cacheCapacity
    }

    /// Total number of nodes
    public var totalNodes: Int {
        configuration.totalNodes
    }
}
