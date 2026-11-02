//
//  VectorCoreIntegration.swift
//  VectorAccelerate
//
//  Integration protocols and extensions for VectorCore types.
//  Uses Metal 4 exclusively (iOS 26+, macOS 26+).
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

// MARK: - Vector Operations Provider

/// GPU-accelerated vector operations provider using Metal 4
public actor AcceleratedVectorOperations: VectorOperationsProvider {
    private let engine: Metal4ComputeEngine
    private let context: Metal4Context

    public init() async throws {
        guard ComputeDevice.gpu().isAvailable else {
            throw VectorError.metalNotAvailable()
        }

        self.context = try await Metal4Context()
        self.engine = try await Metal4ComputeEngine(context: context)
    }

    public init(context: Metal4Context) async throws {
        self.context = context
        self.engine = try await Metal4ComputeEngine(context: context)
    }

    public func add<T: VectorProtocol>(_ v1: T, _ v2: T) async throws -> T where T.Scalar == Float {
        let a = v1.toArray()
        let b = v2.toArray()

        // Get buffers
        let bufferA = try await context.getBuffer(for: a)
        let bufferB = try await context.getBuffer(for: b)
        let resultBuffer = try await context.getBuffer(size: a.count * MemoryLayout<Float>.size)

        // Get pipeline using Metal 4 shader compiler
        let pipeline = try await context.getPipeline(functionName: "vectorAdd")

        // Execute
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
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

        // Get pipeline
        let pipeline = try await context.getPipeline(functionName: "vectorMultiply")

        // Execute
        try await context.executeAndWait { commandBuffer, encoder in
            encoder.setComputePipelineState(pipeline)
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

    /// Normalize a vector on the GPU: `v / ‖v‖₂`.
    ///
    /// Matches VectorCore's CPU normalization for every input class (subnormal,
    /// huge, degenerate) — see ``Metal4ComputeEngine/normalize(_:)``.
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

    /// Check if acceleration is available
    public static var isAccelerationAvailable: Bool {
        ComputeDevice.gpu().isAvailable
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

    /// Fast normalization without runtime validation, using VectorCore's unchecked normalization path
    func acceleratedNormalizeUnchecked() async throws -> Self {
        if AcceleratedVectorFactory.isAccelerationAvailable {
            let provider = try await AcceleratedVectorOperations()
            return try await provider.normalize(self)
        } else {
            return normalizedUnchecked()
        }
    }
}

// MARK: - Hybrid Execution Strategy

/// Strategy for choosing between CPU and GPU execution
public struct HybridExecutionStrategy {
    public let gpuThreshold: Int
    public let batchThreshold: Int

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

// MARK: - High-level Integration Facade

/// Lightweight facade to integrate VectorAccelerate with VectorCore
public struct VectorCoreIntegration: Sendable {
    public struct Configuration: Sendable {
        public var preferGPU: Bool
        public var fallbackToCPU: Bool
        public var cachingEnabled: Bool
        public var batchThreshold: Int

        public init(
            preferGPU: Bool = true,
            fallbackToCPU: Bool = true,
            cachingEnabled: Bool = true,
            batchThreshold: Int = 100
        ) {
            self.preferGPU = preferGPU
            self.fallbackToCPU = fallbackToCPU
            self.cachingEnabled = cachingEnabled
            self.batchThreshold = batchThreshold
        }
    }

    public enum IntegrationError: Error, Sendable {
        case metalUnavailable
    }

    private let context: Metal4Context
    public let configuration: Configuration

    public init(context: Metal4Context, configuration: Configuration = .init()) {
        self.context = context
        self.configuration = configuration
    }

    /// Create a GPU-accelerated vector operations provider
    public func createVectorOperations() async throws -> AcceleratedVectorOperations {
        guard ComputeDevice.gpu().isAvailable else { throw IntegrationError.metalUnavailable }
        return try await AcceleratedVectorOperations(context: context)
    }
}
