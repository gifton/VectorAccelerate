// VectorAccelerate: Adaptive Threshold Management
//
// Dynamic CPU/GPU routing based on performance history

import Foundation
@preconcurrency import Metal

/// Adaptive threshold management for CPU/GPU routing.
///
/// - Note: This class now delegates to `GPUDecisionEngine` internally.
///   For new code, consider using `GPUDecisionEngine` directly.
///
/// ## Migration
/// ```swift
/// // Old pattern:
/// let manager = AdaptiveThresholdManager(configuration: config)
/// let path = await manager.recommendExecutionPath(for: .distanceComputation, workloadSize: 10000)
///
/// // New pattern:
/// let engine = GPUDecisionEngine()
/// let useGPU = await engine.shouldUseGPU(operation: .l2Distance, vectorCount: 10000, ...)
/// ```
@available(*, deprecated, message: "Use GPUDecisionEngine directly for new code. AdaptiveThresholdManager now delegates to GPUDecisionEngine internally.")
public actor AdaptiveThresholdManager {
    private var thresholds: [OperationType: OperationThreshold]
    private let configuration: AccelerationConfiguration
    private var performanceHistory: [OperationType: [PerformanceRecord]] = [:]
    private let historyLimit = 100

    /// Internal decision engine for delegated routing decisions
    private let decisionEngine: GPUDecisionEngine
    
    public struct OperationThreshold {
        public var minGPUWorkload: Int
        public var hybridThreshold: Int
        public var cpuThroughput: Double
        public var gpuThroughput: Double
        
        mutating func updateFromPerformance(path: ExecutionPath, throughput: Double) {
            switch path {
            case .cpu:
                cpuThroughput = (cpuThroughput * 0.9) + (throughput * 0.1)
            case .gpu:
                gpuThroughput = (gpuThroughput * 0.9) + (throughput * 0.1)
            case .hybrid:
                // Update both based on hybrid performance
                let hybridFactor = throughput / max(cpuThroughput, gpuThroughput)
                if hybridFactor > 1.2 {
                    // Hybrid is significantly better, lower threshold
                    hybridThreshold = Int(Double(hybridThreshold) * 0.9)
                }
            }
            
            // Adjust GPU threshold based on relative performance
            if cpuThroughput > 0 && gpuThroughput > 0 {
                let ratio = gpuThroughput / cpuThroughput
                if ratio < 1.5 {
                    // GPU not much faster, increase threshold
                    minGPUWorkload = Int(Double(minGPUWorkload) * 1.1)
                } else if ratio > 3.0 {
                    // GPU much faster, decrease threshold
                    minGPUWorkload = Int(Double(minGPUWorkload) * 0.9)
                }
            }
        }
    }
    
    struct PerformanceRecord {
        let timestamp: Date
        let path: ExecutionPath
        let throughput: Double
        let workloadSize: Int
    }
    
    public init(configuration: AccelerationConfiguration) {
        self.configuration = configuration

        // Create internal decision engine with thresholds from config
        self.decisionEngine = GPUDecisionEngine(thresholds: configuration.toGPUActivationThresholds())

        // Initialize default thresholds for backward compatibility
        self.thresholds = [
            .distanceComputation: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold,
                hybridThreshold: configuration.hybridThreshold,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            ),
            .batchDistanceComputation: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold / 2,
                hybridThreshold: configuration.hybridThreshold / 2,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            ),
            .matrixMultiplication: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold / 4,
                hybridThreshold: configuration.hybridThreshold / 4,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            ),
            .quantization: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold * 2,
                hybridThreshold: configuration.hybridThreshold * 2,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            )
        ]
    }

    /// Create with explicit decision engine (for sharing engine across components)
    public init(configuration: AccelerationConfiguration, decisionEngine: GPUDecisionEngine) {
        self.configuration = configuration
        self.decisionEngine = decisionEngine

        // Initialize default thresholds for backward compatibility
        self.thresholds = [
            .distanceComputation: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold,
                hybridThreshold: configuration.hybridThreshold,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            ),
            .batchDistanceComputation: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold / 2,
                hybridThreshold: configuration.hybridThreshold / 2,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            ),
            .matrixMultiplication: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold / 4,
                hybridThreshold: configuration.hybridThreshold / 4,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            ),
            .quantization: OperationThreshold(
                minGPUWorkload: configuration.gpuThreshold * 2,
                hybridThreshold: configuration.hybridThreshold * 2,
                cpuThroughput: 1.0,
                gpuThroughput: 1.0
            )
        ]
    }

    /// Access the underlying GPUDecisionEngine for direct use.
    ///
    /// Use this to share performance history across components or
    /// to migrate to direct GPUDecisionEngine usage.
    public var underlyingDecisionEngine: GPUDecisionEngine {
        decisionEngine
    }
    
    public func getThreshold(for operation: OperationType) -> OperationThreshold {
        return thresholds[operation] ?? OperationThreshold(
            minGPUWorkload: configuration.gpuThreshold,
            hybridThreshold: configuration.hybridThreshold,
            cpuThroughput: 1.0,
            gpuThroughput: 1.0
        )
    }

    // MARK: - Type Mapping

    /// Maps OperationType to GPUOperation for delegation
    private func mapToGPUOperation(_ operationType: OperationType) -> GPUOperation {
        switch operationType {
        case .distanceComputation:
            return .l2Distance
        case .batchDistanceComputation:
            return .distanceMatrix
        case .matrixMultiplication:
            return .batchMatrixMultiply
        case .quantization:
            return .pqEncode
        case .clustering:
            return .ivfTraining
        case .indexing:
            return .ivfAssignment
        case .search:
            return .batchSearch
        }
    }

    // MARK: - Execution Path Recommendation

    /// Recommends execution path by delegating to GPUDecisionEngine.
    ///
    /// This async version uses the internal GPUDecisionEngine for adaptive routing.
    ///
    /// - Parameters:
    ///   - operation: The operation type to route
    ///   - workloadSize: Size of the workload (vector count)
    ///   - dimension: Vector dimension (default 128)
    /// - Returns: Recommended execution path
    public func recommendExecutionPath(
        for operation: OperationType,
        workloadSize: Int,
        dimension: Int = 128
    ) async -> ExecutionPath {
        guard configuration.adaptiveThresholds else {
            return determineStaticPath(workloadSize: workloadSize)
        }

        // Map OperationType to GPUOperation
        let gpuOp = mapToGPUOperation(operation)

        // Delegate to GPUDecisionEngine
        let shouldUseGPU = await decisionEngine.shouldUseGPU(
            operation: gpuOp,
            vectorCount: workloadSize,
            candidateCount: workloadSize,
            k: 10,  // Default k
            dimension: dimension
        )

        // Convert boolean to ExecutionPath
        if shouldUseGPU {
            // Check if workload is large enough for hybrid
            let threshold = getThreshold(for: operation)
            if workloadSize >= threshold.hybridThreshold {
                return .hybrid
            }
            return .gpu
        } else {
            return .cpu
        }
    }

    /// Backward-compatible synchronous version.
    ///
    /// - Note: This synchronous version uses the original threshold-based logic
    ///   and does not delegate to GPUDecisionEngine. Use the async version for
    ///   full decision engine integration.
    @available(*, deprecated, message: "Use async version for GPUDecisionEngine integration")
    public func recommendExecutionPath(
        for operation: OperationType,
        workloadSize: Int
    ) -> ExecutionPath {
        guard configuration.adaptiveThresholds else {
            return determineStaticPath(workloadSize: workloadSize)
        }

        // Fall back to original logic for sync callers
        let threshold = getThreshold(for: operation)

        if workloadSize < configuration.cpuThreshold {
            return .cpu
        } else if workloadSize < threshold.minGPUWorkload {
            // Consider CPU vs GPU based on historical performance
            if threshold.cpuThroughput > threshold.gpuThroughput * 0.8 {
                return .cpu
            }
            return .gpu
        } else if workloadSize < threshold.hybridThreshold {
            return .gpu
        } else {
            return .hybrid
        }
    }
    
    private func determineStaticPath(workloadSize: Int) -> ExecutionPath {
        switch configuration.preferredDevice {
        case .cpu:
            return .cpu
        case .gpu:
            return .gpu
        case .hybrid:
            return .hybrid
        case .auto:
            if workloadSize < configuration.cpuThreshold {
                return .cpu
            } else if workloadSize < configuration.gpuThreshold {
                return .gpu
            } else {
                return .hybrid
            }
        }
    }
    
    /// Updates performance metrics and delegates to GPUDecisionEngine.
    ///
    /// This async version delegates to the internal GPUDecisionEngine for
    /// adaptive learning while maintaining backward compatibility with the
    /// local threshold system.
    ///
    /// - Parameters:
    ///   - operation: The operation type that was performed
    ///   - path: The execution path that was used
    ///   - throughput: Achieved throughput
    ///   - workloadSize: Size of the workload
    public func updatePerformance(
        operation: OperationType,
        path: ExecutionPath,
        throughput: Double,
        workloadSize: Int
    ) async {
        guard configuration.adaptiveThresholds else { return }

        // Delegate to GPUDecisionEngine
        let gpuOp = mapToGPUOperation(operation)

        // Convert throughput to time (inverse relationship)
        // Higher throughput = lower time
        let estimatedTime = workloadSize > 0 ? Double(workloadSize) / throughput : 1.0

        switch path {
        case .cpu:
            await decisionEngine.recordPerformance(
                operation: gpuOp,
                cpuTime: estimatedTime,
                gpuTime: estimatedTime * 2  // Assume GPU was slower if CPU was chosen
            )
        case .gpu, .hybrid:
            await decisionEngine.recordPerformance(
                operation: gpuOp,
                cpuTime: estimatedTime * 2,  // Assume CPU was slower
                gpuTime: estimatedTime
            )
        }

        // Keep existing threshold update logic for backward compatibility
        if var threshold = thresholds[operation] {
            threshold.updateFromPerformance(path: path, throughput: throughput)
            thresholds[operation] = threshold
        }

        // Keep existing history recording
        let record = PerformanceRecord(
            timestamp: Date(),
            path: path,
            throughput: throughput,
            workloadSize: workloadSize
        )

        if performanceHistory[operation] == nil {
            performanceHistory[operation] = []
        }
        performanceHistory[operation]?.append(record)

        // Keep only recent history
        if let count = performanceHistory[operation]?.count, count > historyLimit {
            performanceHistory[operation]?.removeFirst(count - historyLimit)
        }
    }

    /// Backward-compatible synchronous version.
    ///
    /// - Note: This synchronous version does not delegate to GPUDecisionEngine.
    ///   Use the async version for full decision engine integration.
    @available(*, deprecated, message: "Use async version for GPUDecisionEngine integration")
    public func updatePerformanceSync(
        operation: OperationType,
        path: ExecutionPath,
        throughput: Double,
        workloadSize: Int
    ) {
        guard configuration.adaptiveThresholds else { return }

        // Update threshold
        if var threshold = thresholds[operation] {
            threshold.updateFromPerformance(path: path, throughput: throughput)
            thresholds[operation] = threshold
        }

        // Record performance history
        let record = PerformanceRecord(
            timestamp: Date(),
            path: path,
            throughput: throughput,
            workloadSize: workloadSize
        )

        if performanceHistory[operation] == nil {
            performanceHistory[operation] = []
        }
        performanceHistory[operation]?.append(record)

        // Keep only recent history
        if let count = performanceHistory[operation]?.count, count > historyLimit {
            performanceHistory[operation]?.removeFirst(count - historyLimit)
        }
    }
    
    public func getPerformanceStats(for operation: OperationType) -> AdaptivePerformanceStats? {
        guard let history = performanceHistory[operation], !history.isEmpty else {
            return nil
        }
        
        let cpuRecords = history.filter { $0.path == .cpu }
        let gpuRecords = history.filter { $0.path == .gpu }
        let hybridRecords = history.filter { $0.path == .hybrid }
        
        return AdaptivePerformanceStats(
            averageCPUThroughput: cpuRecords.isEmpty ? 0 : cpuRecords.map(\.throughput).reduce(0, +) / Double(cpuRecords.count),
            averageGPUThroughput: gpuRecords.isEmpty ? 0 : gpuRecords.map(\.throughput).reduce(0, +) / Double(gpuRecords.count),
            averageHybridThroughput: hybridRecords.isEmpty ? 0 : hybridRecords.map(\.throughput).reduce(0, +) / Double(hybridRecords.count),
            totalOperations: history.count
        )
    }
}

public struct AdaptivePerformanceStats: Sendable {
    public let averageCPUThroughput: Double
    public let averageGPUThroughput: Double
    public let averageHybridThroughput: Double
    public let totalOperations: Int
}