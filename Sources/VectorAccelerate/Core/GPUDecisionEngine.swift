//
//  GPUDecisionEngine.swift
//  VectorAccelerate
//
//  Adaptive GPU/CPU routing based on workload characteristics and runtime performance history.
//
//  Migrated from VectorIndexAccelerated with adaptations for Metal 4 architecture.
//

import Foundation
@preconcurrency import Metal

// MARK: - GPU Activation Thresholds

/// Configuration thresholds for GPU activation decisions.
///
/// These thresholds determine when operations should be routed to GPU vs CPU
/// based on workload characteristics. The values can be tuned based on specific
/// hardware and workload patterns.
public struct GPUActivationThresholds: Sendable {
    // MARK: - Dataset Size Thresholds

    /// Minimum number of vectors to consider GPU acceleration
    public let minVectorsForGPU: Int

    /// Minimum number of candidate vectors for GPU acceleration
    public let minCandidatesForGPU: Int

    // MARK: - Operation Complexity Thresholds

    /// Minimum k value for top-k operations to use GPU
    public let minKForGPU: Int

    /// Maximum k value for GPU (beyond this, CPU may be more efficient)
    public let maxKForGPU: Int

    /// Minimum operation count (query_count × candidate_count × k) for GPU
    public let minOperationsForGPU: Int

    // MARK: - Memory Thresholds

    /// Maximum GPU memory to use in megabytes
    public let maxGPUMemoryMB: Int

    /// Minimum batch size to trigger GPU acceleration
    public let minBatchSizeForGPU: Int

    // MARK: - Adaptive Thresholds

    /// Performance ratio updated based on runtime measurements (GPU time / CPU time)
    /// Values > 1.0 indicate GPU is faster
    public var gpuPerformanceRatio: Float

    /// Number of performance samples to keep for adaptive decisions
    public let performanceHistorySize: Int

    // MARK: - Initialization

    public init(
        minVectorsForGPU: Int = 1000,
        minCandidatesForGPU: Int = 500,
        minKForGPU: Int = 10,
        maxKForGPU: Int = 1000,
        minOperationsForGPU: Int = 50_000,
        maxGPUMemoryMB: Int = 1024,
        minBatchSizeForGPU: Int = 4,
        gpuPerformanceRatio: Float = 1.0,
        performanceHistorySize: Int = 20
    ) {
        self.minVectorsForGPU = minVectorsForGPU
        self.minCandidatesForGPU = minCandidatesForGPU
        self.minKForGPU = minKForGPU
        self.maxKForGPU = maxKForGPU
        self.minOperationsForGPU = minOperationsForGPU
        self.maxGPUMemoryMB = maxGPUMemoryMB
        self.minBatchSizeForGPU = minBatchSizeForGPU
        self.gpuPerformanceRatio = gpuPerformanceRatio
        self.performanceHistorySize = performanceHistorySize
    }

    /// Thresholds optimized for large batch workloads
    public static let batchOptimized = GPUActivationThresholds(
        minVectorsForGPU: 500,
        minCandidatesForGPU: 256,
        minKForGPU: 5,
        maxKForGPU: 2000,
        minOperationsForGPU: 25_000,
        maxGPUMemoryMB: 2048,
        minBatchSizeForGPU: 2
    )

    /// Thresholds optimized for real-time single-query workloads
    public static let realTimeOptimized = GPUActivationThresholds(
        minVectorsForGPU: 2000,
        minCandidatesForGPU: 1000,
        minKForGPU: 20,
        maxKForGPU: 500,
        minOperationsForGPU: 100_000,
        maxGPUMemoryMB: 512,
        minBatchSizeForGPU: 8
    )
}

// MARK: - GPU Operation Types

/// Operations that can be accelerated by GPU in VectorAccelerate.
///
/// Each operation type has different characteristics that affect GPU vs CPU routing decisions.
public enum GPUOperation: String, CaseIterable, Sendable {
    // MARK: - Distance Operations

    /// L2 (Euclidean) distance computation
    case l2Distance = "l2_distance"

    /// Cosine similarity computation
    case cosineSimilarity = "cosine_similarity"

    /// Dot product computation
    case dotProduct = "dot_product"

    /// Manhattan (L1) distance computation
    case manhattanDistance = "manhattan_distance"

    /// Batch distance matrix computation
    case distanceMatrix = "distance_matrix"

    // MARK: - Selection Operations

    /// Top-k selection (finding k nearest neighbors)
    case topKSelection = "top_k_selection"

    /// Bitonic sort for parallel sorting
    case bitonicSort = "bitonic_sort"

    // MARK: - Vector Operations

    /// L2 normalization of vectors
    case normalization = "normalization"

    /// Vector reduction operations (sum, mean, etc.)
    case reduction = "reduction"

    // MARK: - Quantization Operations

    /// Product quantization encoding
    case pqEncode = "pq_encode"

    /// Product quantization search
    case pqSearch = "pq_search"

    /// Scalar quantization
    case scalarQuantization = "scalar_quantization"

    // MARK: - Index Operations

    /// IVF cluster assignment
    case ivfAssignment = "ivf_assignment"

    /// IVF search within clusters
    case ivfSearch = "ivf_search"

    /// IVF training (k-means clustering)
    case ivfTraining = "ivf_training"

    /// IVF list compaction
    case ivfCompaction = "ivf_compaction"

    // MARK: - Batch Operations

    /// Batch vector search
    case batchSearch = "batch_search"

    /// Batch normalization
    case batchNormalization = "batch_normalization"

    /// Batch matrix multiplication
    case batchMatrixMultiply = "batch_matrix_multiply"

    // MARK: - Properties

    /// Whether this operation typically benefits from GPU at small scales
    public var benefitsFromGPUAtSmallScale: Bool {
        switch self {
        case .bitonicSort, .batchMatrixMultiply, .distanceMatrix:
            return true
        default:
            return false
        }
    }

    /// Minimum candidate count for this specific operation
    public var minimumCandidates: Int {
        switch self {
        case .bitonicSort, .topKSelection:
            return 1024
        case .distanceMatrix, .batchMatrixMultiply:
            return 256
        case .ivfTraining:
            return 5000
        default:
            return 500
        }
    }
}

// MARK: - GPU Decision Engine

/// Actor responsible for making adaptive GPU/CPU routing decisions.
///
/// The decision engine tracks performance history for different operation types
/// and adjusts routing decisions based on observed performance. This allows
/// the system to adapt to different hardware capabilities and workload patterns.
///
/// ## Usage
/// ```swift
/// let engine = GPUDecisionEngine()
///
/// if await engine.shouldUseGPU(
///     operation: .l2Distance,
///     vectorCount: 10000,
///     candidateCount: 1000,
///     k: 10,
///     dimension: 128
/// ) {
///     // Use GPU path
/// } else {
///     // Use CPU path
/// }
///
/// // Record performance for adaptive learning
/// await engine.recordPerformance(
///     operation: .l2Distance,
///     cpuTime: cpuDuration,
///     gpuTime: gpuDuration
/// )
/// ```
public actor GPUDecisionEngine {
    // MARK: - State

    /// Performance history for adaptive decisions
    private var performanceHistory: [PerformanceRecord] = []

    /// Current activation thresholds
    private var currentThresholds: GPUActivationThresholds

    /// Metal device for capability queries
    private let device: (any MTLDevice)?

    /// Available GPU memory in MB
    private let deviceMemoryMB: Int

    /// Adaptive performance ratios per operation type
    private var adaptiveRatios: [String: Float] = [:]

    // MARK: - Types

    /// Record of a single performance measurement
    private struct PerformanceRecord: Sendable {
        let operation: String
        let cpuTime: Double
        let gpuTime: Double
        let timestamp: Date
    }

    // MARK: - Initialization

    /// Create a decision engine with default thresholds
    public init() {
        self.init(thresholds: GPUActivationThresholds())
    }

    /// Create a decision engine with custom thresholds
    ///
    /// - Parameter thresholds: Custom activation thresholds
    public init(thresholds: GPUActivationThresholds) {
        self.currentThresholds = thresholds
        self.device = MTLCreateSystemDefaultDevice()

        // Calculate available GPU memory
        if let device = device {
            self.deviceMemoryMB = Int(device.recommendedMaxWorkingSetSize / (1024 * 1024))
        } else {
            self.deviceMemoryMB = 0
        }

        // Initialize adaptive ratios for each operation type
        for operation in GPUOperation.allCases {
            adaptiveRatios[operation.rawValue] = 1.0
        }
    }

    /// Create a decision engine using a Metal4Context's device
    ///
    /// - Parameters:
    ///   - context: The Metal4Context to use for device queries
    ///   - thresholds: Custom activation thresholds
    public init(context: Metal4Context, thresholds: GPUActivationThresholds = GPUActivationThresholds()) async {
        self.currentThresholds = thresholds
        // context.device is nonisolated, so we can access rawDevice directly
        let rawDevice = context.device.rawDevice
        self.device = rawDevice

        // Calculate available GPU memory
        self.deviceMemoryMB = Int(rawDevice.recommendedMaxWorkingSetSize / (1024 * 1024))

        // Initialize adaptive ratios
        for operation in GPUOperation.allCases {
            adaptiveRatios[operation.rawValue] = 1.0
        }
    }

    // MARK: - Decision Making

    /// Determines whether to use GPU for a given operation.
    ///
    /// This method considers multiple factors:
    /// - Hardware capabilities and availability
    /// - Workload characteristics (size, complexity)
    /// - Historical performance data
    /// - Memory requirements
    ///
    /// - Parameters:
    ///   - operation: The type of operation to perform
    ///   - vectorCount: Total number of vectors in the dataset
    ///   - candidateCount: Number of candidate vectors to consider
    ///   - k: Number of results to return (for top-k operations)
    ///   - queryCount: Number of queries in a batch
    ///   - dimension: Vector dimensionality
    /// - Returns: `true` if GPU should be used, `false` for CPU
    public func shouldUseGPU(
        operation: GPUOperation,
        vectorCount: Int,
        candidateCount: Int,
        k: Int,
        queryCount: Int = 1,
        dimension: Int = 128
    ) async -> Bool {
        // Check if GPU is available
        guard device != nil else { return false }

        // Check basic thresholds
        guard vectorCount >= currentThresholds.minVectorsForGPU else { return false }
        guard candidateCount >= currentThresholds.minCandidatesForGPU else { return false }
        guard k >= currentThresholds.minKForGPU && k <= currentThresholds.maxKForGPU else { return false }

        // Calculate operation complexity
        let operationCount = queryCount * candidateCount * k
        guard operationCount >= currentThresholds.minOperationsForGPU else { return false }

        // Check batch size for batch operations
        if operation == .batchSearch || operation == .batchNormalization || operation == .batchMatrixMultiply {
            guard queryCount >= currentThresholds.minBatchSizeForGPU else { return false }
        }

        // Estimate memory requirement
        let estimatedMemoryMB = estimateMemoryRequirement(
            vectorCount: vectorCount,
            candidateCount: candidateCount,
            dimension: dimension,
            k: k,
            queryCount: queryCount
        )

        guard estimatedMemoryMB <= min(currentThresholds.maxGPUMemoryMB, deviceMemoryMB) else {
            return false
        }

        // Check adaptive performance ratio for this operation type
        let adaptiveRatio = adaptiveRatios[operation.rawValue] ?? 1.0
        guard adaptiveRatio > 0.8 else { // GPU must be at least 80% as fast as CPU
            return false
        }

        // Operation-specific considerations
        return evaluateOperationSpecificCriteria(
            operation: operation,
            vectorCount: vectorCount,
            candidateCount: candidateCount,
            k: k,
            queryCount: queryCount,
            dimension: dimension
        )
    }

    /// Evaluates operation-specific criteria for GPU routing.
    private func evaluateOperationSpecificCriteria(
        operation: GPUOperation,
        vectorCount: Int,
        candidateCount: Int,
        k: Int,
        queryCount: Int,
        dimension: Int
    ) -> Bool {
        switch operation {
        case .ivfTraining:
            // Training benefits from GPU with larger datasets
            return vectorCount >= 5000

        case .ivfSearch:
            // Search benefits from GPU with many candidates
            return candidateCount >= 1000 || queryCount >= 10

        case .topKSelection:
            // K-selection benefits depend on K and N
            if k <= 32 && candidateCount <= 10000 {
                return true // Warp-optimized kernel is very efficient
            }
            return candidateCount >= 5000

        case .l2Distance, .cosineSimilarity, .dotProduct, .manhattanDistance:
            // Distance computation benefits from GPU for large datasets
            return candidateCount * dimension >= 100_000

        case .distanceMatrix:
            // Distance matrix always benefits from GPU
            return queryCount * candidateCount >= 10_000

        case .normalization:
            // Normalization benefits from GPU for large batches
            return vectorCount * dimension >= 50_000

        case .ivfCompaction:
            // List compaction benefits from GPU for large lists
            return candidateCount >= 10_000

        case .batchSearch, .batchNormalization:
            // Batch operations require sufficient batch size
            return queryCount >= 8 && candidateCount >= 500

        case .pqEncode, .pqSearch:
            // PQ operations benefit from GPU with enough vectors
            return vectorCount >= 1000

        case .scalarQuantization:
            // Scalar quantization is memory-bound, benefits from GPU
            return vectorCount * dimension >= 100_000

        case .bitonicSort:
            // Bitonic sort is efficient on GPU for power-of-2 sizes
            return candidateCount >= 1024

        case .batchMatrixMultiply:
            // Matrix multiplication benefits from GPU for larger matrices
            return queryCount * candidateCount >= 10_000

        case .reduction:
            // Reduction operations benefit from GPU for large inputs
            return vectorCount * dimension >= 50_000

        case .ivfAssignment:
            // Cluster assignment benefits from GPU with many clusters/vectors
            return vectorCount >= 1000
        }
    }

    // MARK: - Performance Recording

    /// Records performance metrics for adaptive threshold adjustment.
    ///
    /// Call this method after completing an operation to help the engine
    /// learn optimal routing decisions for future operations.
    ///
    /// - Parameters:
    ///   - operation: The operation that was performed
    ///   - cpuTime: Time taken by CPU implementation in seconds
    ///   - gpuTime: Time taken by GPU implementation in seconds
    public func recordPerformance(
        operation: GPUOperation,
        cpuTime: Double,
        gpuTime: Double
    ) async {
        await recordPerformance(operation: operation.rawValue, cpuTime: cpuTime, gpuTime: gpuTime)
    }

    /// Records performance metrics using operation string identifier.
    ///
    /// - Parameters:
    ///   - operation: The operation identifier string
    ///   - cpuTime: Time taken by CPU implementation in seconds
    ///   - gpuTime: Time taken by GPU implementation in seconds
    public func recordPerformance(
        operation: String,
        cpuTime: Double,
        gpuTime: Double
    ) async {
        // Add to history
        let record = PerformanceRecord(
            operation: operation,
            cpuTime: cpuTime,
            gpuTime: gpuTime,
            timestamp: Date()
        )
        performanceHistory.append(record)

        // Trim history if too large
        if performanceHistory.count > currentThresholds.performanceHistorySize {
            performanceHistory.removeFirst()
        }

        // Update adaptive ratio for this operation
        let operationHistory = performanceHistory.filter { $0.operation == operation }
        if operationHistory.count >= 3 {
            let avgCPUTime = operationHistory.map { $0.cpuTime }.reduce(0, +) / Double(operationHistory.count)
            let avgGPUTime = operationHistory.map { $0.gpuTime }.reduce(0, +) / Double(operationHistory.count)

            if avgGPUTime > 0 {
                let ratio = Float(avgCPUTime / avgGPUTime)
                adaptiveRatios[operation] = ratio

                // Update global performance ratio
                let globalRatio = adaptiveRatios.values.reduce(0, +) / Float(adaptiveRatios.count)
                currentThresholds.gpuPerformanceRatio = globalRatio
            }
        }
    }

    // MARK: - Threshold Management

    /// Updates the activation thresholds.
    ///
    /// - Parameter newThresholds: New thresholds to apply
    public func updateThresholds(_ newThresholds: GPUActivationThresholds) async {
        currentThresholds = newThresholds
    }

    /// Gets the current activation thresholds.
    public func getThresholds() async -> GPUActivationThresholds {
        currentThresholds
    }

    // MARK: - Statistics

    /// Gets current performance statistics.
    ///
    /// - Returns: Aggregated performance statistics
    public func getPerformanceStats() async -> GPUPerformanceStats {
        var stats = GPUPerformanceStats()

        for operation in GPUOperation.allCases {
            let operationHistory = performanceHistory.filter { $0.operation == operation.rawValue }
            if !operationHistory.isEmpty {
                let avgCPUTime = operationHistory.map { $0.cpuTime }.reduce(0, +) / Double(operationHistory.count)
                let avgGPUTime = operationHistory.map { $0.gpuTime }.reduce(0, +) / Double(operationHistory.count)
                let speedup = avgGPUTime > 0 ? avgCPUTime / avgGPUTime : 0

                stats.operationSpeedups[operation.rawValue] = Float(speedup)
                stats.avgGPUTimes[operation.rawValue] = avgGPUTime
                stats.avgCPUTimes[operation.rawValue] = avgCPUTime
            }
        }

        stats.globalSpeedup = currentThresholds.gpuPerformanceRatio
        stats.totalOperations = performanceHistory.count

        return stats
    }

    /// Resets performance history and adaptive ratios.
    public func reset() async {
        performanceHistory.removeAll()
        for operation in GPUOperation.allCases {
            adaptiveRatios[operation.rawValue] = 1.0
        }
        currentThresholds.gpuPerformanceRatio = 1.0
    }

    // MARK: - Memory Estimation

    /// Estimates memory requirement for an operation in megabytes.
    private func estimateMemoryRequirement(
        vectorCount: Int,
        candidateCount: Int,
        dimension: Int,
        k: Int,
        queryCount: Int
    ) -> Int {
        let bytesPerFloat = 4

        // Vector storage
        let vectorMemory = candidateCount * dimension * bytesPerFloat

        // Distance matrix (for batch operations)
        let distanceMemory = queryCount * candidateCount * bytesPerFloat

        // Index buffers
        let indexMemory = candidateCount * 4 // UInt32

        // Result buffers
        let resultMemory = queryCount * k * (bytesPerFloat + 4) // distances + indices

        // Working memory (approximate 2x for intermediate buffers)
        let workingMemory = (vectorMemory + distanceMemory) * 2

        let totalBytes = vectorMemory + distanceMemory + indexMemory + resultMemory + workingMemory
        return totalBytes / (1024 * 1024)
    }

    // MARK: - Device Information

    /// Whether GPU is available for acceleration.
    public var isGPUAvailable: Bool {
        device != nil
    }

    /// Available GPU memory in megabytes.
    public var availableGPUMemoryMB: Int {
        deviceMemoryMB
    }

    /// Device name for debugging.
    public var deviceName: String? {
        device?.name
    }
}

// MARK: - Performance Statistics

/// Aggregated performance statistics from the decision engine.
public struct GPUPerformanceStats: Sendable {
    /// Speedup ratios per operation type (CPU time / GPU time)
    public var operationSpeedups: [String: Float] = [:]

    /// Average GPU times per operation in seconds
    public var avgGPUTimes: [String: Double] = [:]

    /// Average CPU times per operation in seconds
    public var avgCPUTimes: [String: Double] = [:]

    /// Global speedup across all operations
    public var globalSpeedup: Float = 1.0

    /// Total number of operations recorded
    public var totalOperations: Int = 0

    /// Human-readable summary of performance statistics
    public var summary: String {
        var result = "GPU Performance Statistics:\n"
        result += "Total Operations: \(totalOperations)\n"
        result += "Global Speedup: \(String(format: "%.2f", globalSpeedup))x\n\n"

        for (operation, speedup) in operationSpeedups.sorted(by: { $0.key < $1.key }) {
            let gpuTime = avgGPUTimes[operation] ?? 0
            let cpuTime = avgCPUTimes[operation] ?? 0
            result += "\(operation):\n"
            result += "  CPU: \(String(format: "%.3f", cpuTime * 1000))ms\n"
            result += "  GPU: \(String(format: "%.3f", gpuTime * 1000))ms\n"
            result += "  Speedup: \(String(format: "%.2f", speedup))x\n"
        }

        return result
    }
}
