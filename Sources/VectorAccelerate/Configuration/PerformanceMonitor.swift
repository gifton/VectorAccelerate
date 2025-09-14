// VectorAccelerate: Performance Monitoring System
//
// Tracks resource utilization and operation metrics

import Foundation
@preconcurrency import Metal

/// Performance monitoring system
public actor PerformanceMonitor {
    private var operationHistory: [OperationType: [OperationRecord]] = [:]
    private var currentMemoryPressure: MemoryPressure = .normal
    private var gpuUtilization: Double = 0.0
    private var cpuUtilization: Double = 0.0
    private let historyLimit = 1000
    private var monitoringTask: Task<Void, Never>?
    
    struct OperationRecord {
        let timestamp: Date
        let duration: TimeInterval
        let dataSize: Int
        let path: ExecutionPath
        let throughput: Double
    }
    
    public enum MemoryPressure: Sendable {
        case normal
        case warning
        case critical
    }
    
    public init() {
        // Start monitoring system resources in a separate initialization
        Task {
            await self.startMonitoring()
        }
    }
    
    private func startMonitoring() {
        self.monitoringTask = Task {
            await self.startResourceMonitoring()
        }
    }
    
    deinit {
        monitoringTask?.cancel()
    }
    
    public func recordOperation(
        type: OperationType,
        path: ExecutionPath,
        duration: TimeInterval,
        dataSize: Int
    ) {
        let throughput = Double(dataSize) / duration
        let record = OperationRecord(
            timestamp: Date(),
            duration: duration,
            dataSize: dataSize,
            path: path,
            throughput: throughput
        )
        
        if operationHistory[type] == nil {
            operationHistory[type] = []
        }
        operationHistory[type]?.append(record)
        
        // Keep only recent history
        if let count = operationHistory[type]?.count, count > historyLimit {
            operationHistory[type]?.removeFirst(count - historyLimit)
        }
    }
    
    public func getCurrentMemoryPressure() -> MemoryPressure {
        return currentMemoryPressure
    }
    
    public func getCurrentGPUUtilization() -> Double {
        return gpuUtilization
    }
    
    public func getCurrentCPUUtilization() -> Double {
        return cpuUtilization
    }
    
    public func getOperationMetrics(for type: OperationType) -> OperationMetrics? {
        guard let history = operationHistory[type], !history.isEmpty else {
            return nil
        }
        
        let durations = history.map(\.duration)
        let throughputs = history.map(\.throughput)
        
        let sortedDurations = durations.sorted()
        let medianDuration = sortedDurations[sortedDurations.count / 2]
        let p95Duration = sortedDurations[Int(Double(sortedDurations.count) * 0.95)]
        
        return OperationMetrics(
            averageDuration: durations.reduce(0, +) / Double(durations.count),
            medianDuration: medianDuration,
            p95Duration: p95Duration,
            averageThroughput: throughputs.reduce(0, +) / Double(throughputs.count),
            operationCount: history.count
        )
    }
    
    public func getResourceUtilization() -> ResourceUtilization {
        return ResourceUtilization(
            memoryPressure: currentMemoryPressure,
            gpuUtilization: gpuUtilization,
            cpuUtilization: cpuUtilization,
            timestamp: Date()
        )
    }
    
    private func startResourceMonitoring() async {
        while !Task.isCancelled {
            updateResourceMetrics()
            try? await Task.sleep(for: .seconds(1))
        }
    }
    
    private func updateResourceMetrics() {
        // Update memory pressure based on available memory
        let memInfo = ProcessInfo.processInfo
        let physicalMemory = memInfo.physicalMemory
        
        // Get memory usage (this is a simplified approach)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let memoryUsage = info.resident_size
            let memoryRatio = Double(memoryUsage) / Double(physicalMemory)
            
            if memoryRatio > 0.9 {
                currentMemoryPressure = .critical
            } else if memoryRatio > 0.7 {
                currentMemoryPressure = .warning
            } else {
                currentMemoryPressure = .normal
            }
        }
        
        // Update CPU utilization (simplified)
        var cpuInfo: processor_info_array_t!
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCpus: natural_t = 0
        
        let cpuResult = host_processor_info(mach_host_self(),
                                           PROCESSOR_CPU_LOAD_INFO,
                                           &numCpus,
                                           &cpuInfo,
                                           &numCpuInfo)
        
        if cpuResult == KERN_SUCCESS {
            // Calculate average CPU usage across all cores
            var totalUsage: Double = 0
            for i in 0..<Int(numCpus) {
                let offset = Int32(CPU_STATE_MAX) * Int32(i)
                let user = Double(cpuInfo[Int(offset + Int32(CPU_STATE_USER))])
                let system = Double(cpuInfo[Int(offset + Int32(CPU_STATE_SYSTEM))])
                let idle = Double(cpuInfo[Int(offset + Int32(CPU_STATE_IDLE))])
                let nice = Double(cpuInfo[Int(offset + Int32(CPU_STATE_NICE))])
                
                let total = user + system + idle + nice
                if total > 0 {
                    totalUsage += (user + system + nice) / total
                }
            }
            cpuUtilization = totalUsage / Double(numCpus)
            
            // Deallocate the CPU info array
            let cpuInfoSize = MemoryLayout<integer_t>.stride * Int(numCpuInfo)
            vm_deallocate(mach_task_self_, vm_address_t(bitPattern: cpuInfo), vm_size_t(cpuInfoSize))
        }
        
        // GPU utilization would require more complex system APIs
        // For now, estimate based on operation history
        let recentGPUOps = operationHistory.values
            .flatMap { $0 }
            .filter { $0.path == .gpu && Date().timeIntervalSince($0.timestamp) < 5 }
        
        gpuUtilization = min(Double(recentGPUOps.count) * 0.1, 1.0)
    }
}

public struct OperationMetrics: Sendable {
    public let averageDuration: TimeInterval
    public let medianDuration: TimeInterval
    public let p95Duration: TimeInterval
    public let averageThroughput: Double
    public let operationCount: Int
}

public struct ResourceUtilization: Sendable {
    public let memoryPressure: PerformanceMonitor.MemoryPressure
    public let gpuUtilization: Double
    public let cpuUtilization: Double
    public let timestamp: Date
}
