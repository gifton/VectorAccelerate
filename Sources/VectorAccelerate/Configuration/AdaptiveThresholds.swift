// VectorAccelerate: Adaptive Threshold Management
//
// Dynamic CPU/GPU routing based on performance history

import Foundation
@preconcurrency import Metal

/// Adaptive threshold management for CPU/GPU routing
public actor AdaptiveThresholdManager {
    private var thresholds: [OperationType: OperationThreshold]
    private let configuration: AccelerationConfiguration
    private var performanceHistory: [OperationType: [PerformanceRecord]] = [:]
    private let historyLimit = 100
    
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
        
        // Initialize default thresholds
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
    
    public func getThreshold(for operation: OperationType) -> OperationThreshold {
        return thresholds[operation] ?? OperationThreshold(
            minGPUWorkload: configuration.gpuThreshold,
            hybridThreshold: configuration.hybridThreshold,
            cpuThroughput: 1.0,
            gpuThroughput: 1.0
        )
    }
    
    public func recommendExecutionPath(
        for operation: OperationType,
        workloadSize: Int
    ) -> ExecutionPath {
        guard configuration.adaptiveThresholds else {
            return determineStaticPath(workloadSize: workloadSize)
        }
        
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
    
    public func updatePerformance(
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