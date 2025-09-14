//
//  VectorCoreIntegration.swift
//  VectorAccelerate
//
//  Integration protocols and extensions for VectorCore types
//

import Foundation
import VectorCore
@preconcurrency import Metal

// MARK: - Accelerated Protocol Conformance

/// Protocol for types that can be accelerated with Metal
public protocol MetalAccelerable {
    associatedtype Element: Numeric
    
    /// Convert to Metal-compatible buffer format
    func toMetalBuffer() async throws -> BufferToken
    
    /// Create from Metal buffer result
    static func fromMetalBuffer(_ buffer: BufferToken, dimension: Int) -> Self
}

// MARK: - Distance Provider Implementation

/// GPU-accelerated distance computation provider
public actor AcceleratedDistanceProvider: DistanceProvider {
    private let engine: ComputeEngine
    private let context: MetalContext
    
    public init() async throws {
        guard MetalDevice.isAvailable else {
            throw AccelerationError.metalNotAvailable
        }
        
        self.context = try await VectorAccelerate.MetalContext()
        self.engine = try await ComputeEngine(context: context)
    }
    
    public init(context: MetalContext) async throws {
        self.context = context
        self.engine = try await ComputeEngine(context: context)
    }
    
    public func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float {
        let v1Array = vector1.toArray()
        let v2Array = vector2.toArray()
        
        switch metric {
        case .euclidean:
            return try await engine.euclideanDistance(v1Array, v2Array)
        case .cosine:
            return try await engine.cosineDistance(v1Array, v2Array)
        case .dotProduct:
            return try await engine.dotProduct(v1Array, v2Array)
        default:
            // Fall back to CPU implementation for unsupported metrics
            throw AccelerationError.unsupportedOperation("Metric \(metric) not accelerated")
        }
    }
    
    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float {
        let queryArray = query.toArray()
        let candidateArrays = candidates.map { $0.toArray() }
        
        switch metric {
        case .euclidean:
            return try await engine.batchEuclideanDistance(
                query: queryArray,
                candidates: candidateArrays
            )
        default:
            // Fall back to sequential computation for unsupported metrics
            var distances: [Float] = []
            for candidate in candidates {
                let distance = try await self.distance(from: query, to: candidate, metric: metric)
                distances.append(distance)
            }
            return distances
        }
    }
}

// MARK: - Vector Operations Provider

/// GPU-accelerated vector operations provider
public actor AcceleratedVectorOperations: VectorOperationsProvider {
    private let engine: ComputeEngine
    private let context: MetalContext
    
    public init() async throws {
        guard MetalDevice.isAvailable else {
            throw AccelerationError.metalNotAvailable
        }
        
        self.context = try await VectorAccelerate.MetalContext()
        self.engine = try await ComputeEngine(context: context)
    }
    
    public init(context: MetalContext) async throws {
        self.context = context
        self.engine = try await ComputeEngine(context: context)
    }
    
    public func add<T: VectorProtocol>(_ v1: T, _ v2: T) async throws -> T where T.Scalar == Float {
        let a = v1.toArray()
        let b = v2.toArray()
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: a)
        let bufferB = try await context.getBuffer(for: b)
        let resultBuffer = try await context.getBuffer(size: a.count * MemoryLayout<Float>.size)
        
        // Get shader
        let shaderManager = try await ShaderManager(device: context.device)
        let pipelineState = try await shaderManager.getPipelineState(functionName: "vectorAdd")
        
        // Execute
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(a.count)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            let (threadsPerGroup, threadgroups) = await context.calculateThreadGroups(for: a.count)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        let result = resultBuffer.copyData(as: Float.self)
        return try T(result)
    }
    
    public func multiply<T: VectorProtocol>(_ v1: T, _ v2: T) async throws -> T where T.Scalar == Float {
        let a = v1.toArray()
        let b = v2.toArray()
        
        // Get buffers
        let bufferA = try await context.getBuffer(for: a)
        let bufferB = try await context.getBuffer(for: b)
        let resultBuffer = try await context.getBuffer(size: a.count * MemoryLayout<Float>.size)
        
        // Get shader
        let shaderManager = try await ShaderManager(device: context.device)
        let pipelineState = try await shaderManager.getPipelineState(functionName: "vectorMultiply")
        
        // Execute
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(bufferA.buffer, offset: 0, index: 0)
            encoder.setBuffer(bufferB.buffer, offset: 0, index: 1)
            encoder.setBuffer(resultBuffer.buffer, offset: 0, index: 2)
            
            var dim = UInt32(a.count)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.size, index: 3)
            
            let (threadsPerGroup, threadgroups) = await context.calculateThreadGroups(for: a.count)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        }
        
        let result = resultBuffer.copyData(as: Float.self)
        return try T(result)
    }
    
    public func scale<T: VectorProtocol>(_ vector: T, by scalar: Float) async throws -> T where T.Scalar == Float {
        let array = vector.toArray()
        let scaled = try await engine.scale(array, by: scalar)
        return try T(scaled)
    }
    
    public func normalize<T: VectorProtocol>(_ vector: T) async throws -> T where T.Scalar == Float {
        let array = vector.toArray()
        let normalized = try await engine.normalize(array)
        return try T(normalized)
    }
    
    public func dotProduct<T: VectorProtocol>(_ v1: T, _ v2: T) async throws -> Float where T.Scalar == Float {
        let a = v1.toArray()
        let b = v2.toArray()
        return try await engine.dotProduct(a, b)
    }
}

// MARK: - Accelerated Vector Factory

/// Factory for creating GPU-accelerated vector computation providers
public enum AcceleratedVectorFactory {
    
    /// Create default accelerated providers
    public static func createDefaultProviders() async throws -> (
        distance: AcceleratedDistanceProvider,
        operations: AcceleratedVectorOperations
    ) {
        let context = try await VectorAccelerate.MetalContext()
        
        let distance = try await AcceleratedDistanceProvider(context: context)
        let operations = try await AcceleratedVectorOperations(context: context)
        
        return (distance, operations)
    }
    
    /// Check if acceleration is available
    public static var isAccelerationAvailable: Bool {
        MetalDevice.isAvailable
    }
    
    /// Create providers with custom configuration
    public static func createProviders(
        configuration: MetalConfiguration
    ) async throws -> (
        distance: AcceleratedDistanceProvider,
        operations: AcceleratedVectorOperations
    ) {
        let context = try await VectorAccelerate.MetalContext(configuration: configuration)
        
        let distance = try await AcceleratedDistanceProvider(context: context)
        let operations = try await AcceleratedVectorOperations(context: context)
        
        return (distance, operations)
    }
}

// MARK: - Performance Monitoring

/// Protocol for monitoring GPU acceleration performance
public protocol AccelerationMonitor {
    func recordOperation(name: String, duration: TimeInterval)
    func getStatistics() -> AccelerationStatistics
}

public struct AccelerationStatistics: Sendable {
    public let totalOperations: Int
    public let totalTime: TimeInterval
    public let averageTime: TimeInterval
    public let speedupFactor: Double // Compared to CPU baseline
    
    public init(
        totalOperations: Int,
        totalTime: TimeInterval,
        averageTime: TimeInterval,
        speedupFactor: Double
    ) {
        self.totalOperations = totalOperations
        self.totalTime = totalTime
        self.averageTime = averageTime
        self.speedupFactor = speedupFactor
    }
}

// MARK: - Convenience Extensions

public extension VectorProtocol where Scalar == Float {
    
    /// Compute distance using GPU acceleration if available
    func acceleratedDistance(
        to other: Self,
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> Float {
        if AcceleratedVectorFactory.isAccelerationAvailable {
            let provider = try await AcceleratedDistanceProvider()
            return try await provider.distance(from: self, to: other, metric: metric)
        } else {
            // Fall back to CPU implementation
            return distance(to: other, metric: metric)
        }
    }
    
    /// Normalize using GPU acceleration if available
    func acceleratedNormalize() async throws -> Self {
        if AcceleratedVectorFactory.isAccelerationAvailable {
            let provider = try await AcceleratedVectorOperations()
            return try await provider.normalize(self)
        } else {
            // Fall back to CPU implementation
            return try normalized().get()
        }
    }
}

// MARK: - Hybrid Execution Strategy

/// Strategy for choosing between CPU and GPU execution
public struct HybridExecutionStrategy {
    public let gpuThreshold: Int // Minimum dimension for GPU execution
    public let batchThreshold: Int // Minimum batch size for GPU execution
    
    public init(gpuThreshold: Int = 128, batchThreshold: Int = 10) {
        self.gpuThreshold = gpuThreshold
        self.batchThreshold = batchThreshold
    }
    
    public func shouldUseGPU<T: VectorProtocol>(for vector: T) -> Bool {
        guard AcceleratedVectorFactory.isAccelerationAvailable else { return false }
        return vector.scalarCount >= gpuThreshold
    }
    
    public func shouldUseGPU<T: VectorProtocol>(for vectors: [T]) -> Bool {
        guard AcceleratedVectorFactory.isAccelerationAvailable else { return false }
        guard let first = vectors.first else { return false }
        return vectors.count >= batchThreshold || first.scalarCount >= gpuThreshold
    }
}