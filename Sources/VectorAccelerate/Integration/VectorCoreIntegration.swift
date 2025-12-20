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

// MARK: - Distance Provider Implementation

/// GPU-accelerated distance computation provider using Metal 4
public actor AcceleratedDistanceProvider: DistanceProvider {
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
        case .manhattan:
            return try await engine.manhattanDistance(v1Array, v2Array)
        case .chebyshev:
            return try await engine.chebyshevDistance(v1Array, v2Array)
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
            return try await engine.batchEuclideanDistance(query: queryArray, candidates: candidateArrays)
        case .cosine:
            return try await engine.batchCosineDistance(query: queryArray, candidates: candidateArrays)
        case .dotProduct, .manhattan, .chebyshev:
            // No specialized batch kernels yet - compute sequentially
            var distances: [Float] = []
            distances.reserveCapacity(candidateArrays.count)
            for candidate in candidates {
                distances.append(try await self.distance(from: query, to: candidate, metric: metric))
            }
            return distances
        }
    }
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

    /// Create default accelerated providers using Metal 4
    public static func createDefaultProviders() async throws -> (
        distance: AcceleratedDistanceProvider,
        operations: AcceleratedVectorOperations
    ) {
        let context = try await Metal4Context()

        let distance = try await AcceleratedDistanceProvider(context: context)
        let operations = try await AcceleratedVectorOperations(context: context)

        return (distance, operations)
    }

    /// Check if acceleration is available
    public static var isAccelerationAvailable: Bool {
        ComputeDevice.gpu().isAvailable
    }

    /// Create providers with custom configuration
    public static func createProviders(
        configuration: Metal4Configuration
    ) async throws -> (
        distance: AcceleratedDistanceProvider,
        operations: AcceleratedVectorOperations
    ) {
        let context = try await Metal4Context(configuration: configuration)

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

    /// Fast normalization without runtime validation (VectorCore 0.1.5)
    func acceleratedNormalizeUnchecked() async throws -> Self {
        if AcceleratedVectorFactory.isAccelerationAvailable {
            let provider = try await AcceleratedVectorOperations()
            return try await provider.normalize(self)
        } else {
            return normalizedUnchecked()
        }
    }
}

// MARK: - IndexableVector Extensions (VectorCore 0.1.5)

public extension IndexableVector where Scalar == Float {

    /// Compute distance with GPU acceleration, skipping normalization if already normalized
    func acceleratedDistanceOptimized(
        to other: Self,
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> Float {
        // For cosine with normalized vectors, use dot product directly
        if metric == .cosine && self.isNormalized && other.isNormalized {
            if AcceleratedVectorFactory.isAccelerationAvailable {
                let provider = try await AcceleratedVectorOperations()
                let dot = try await provider.dotProduct(self, other)
                return 1.0 - dot  // Convert similarity to distance
            } else {
                return 1.0 - DotProductDistance().distance(self, other)
            }
        }

        return try await acceleratedDistance(to: other, metric: metric)
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

    /// Create a GPU-accelerated distance provider
    public func createDistanceProvider() async throws -> AcceleratedDistanceProvider {
        guard ComputeDevice.gpu().isAvailable else { throw IntegrationError.metalUnavailable }
        return try await AcceleratedDistanceProvider(context: context)
    }

    /// Create a GPU-accelerated vector operations provider
    public func createVectorOperations() async throws -> AcceleratedVectorOperations {
        guard ComputeDevice.gpu().isAvailable else { throw IntegrationError.metalUnavailable }
        return try await AcceleratedVectorOperations(context: context)
    }
}

// MARK: - GPU-Accelerated Batch Operations Extension

public extension BatchOperations {

    // MARK: - GPU-Accelerated k-NN Search

    /// Find k nearest neighbors using GPU acceleration (Metal 4)
    static func findNearestGPU<V: VectorProtocol & Sendable>(
        to query: V,
        in vectors: [V],
        k: Int,
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        guard ComputeDevice.gpu().isAvailable else {
            throw VectorError.metalNotAvailable()
        }

        guard k > 0 && !vectors.isEmpty else {
            return []
        }

        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)

        let queryArray = query.toArray()
        let candidateArrays = vectors.map { $0.toArray() }

        // Compute distances using GPU
        let distances: [Float]
        switch metric {
        case .euclidean:
            distances = try await engine.batchEuclideanDistance(query: queryArray, candidates: candidateArrays)
        case .cosine:
            distances = try await engine.batchCosineDistance(query: queryArray, candidates: candidateArrays)
        default:
            // For other metrics, compute individually
            distances = try await withThrowingTaskGroup(of: (Int, Float).self) { group in
                for (index, candidate) in candidateArrays.enumerated() {
                    group.addTask {
                        let dist: Float
                        switch metric {
                        case .dotProduct:
                            dist = try await engine.dotProduct(queryArray, candidate)
                        case .manhattan:
                            dist = try await engine.manhattanDistance(queryArray, candidate)
                        case .chebyshev:
                            dist = try await engine.chebyshevDistance(queryArray, candidate)
                        default:
                            dist = try await engine.euclideanDistance(queryArray, candidate)
                        }
                        return (index, dist)
                    }
                }

                var results = [(Int, Float)]()
                for try await (index, dist) in group {
                    results.append((index, dist))
                }
                return results.sorted { $0.0 < $1.0 }.map { $0.1 }
            }
        }

        // Create (index, distance) pairs
        let pairs = distances.enumerated().map { (index: $0.offset, distance: $0.element) }

        // Select top-k
        return selectTopK(pairs, k: k)
    }

    // MARK: - GPU-Accelerated Batch Distance Computation

    /// Compute distances from a query to multiple candidates using GPU
    static func batchDistancesGPU<V: VectorProtocol & Sendable>(
        from query: V,
        to candidates: [V],
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> [Float] where V.Scalar == Float {
        guard ComputeDevice.gpu().isAvailable else {
            throw VectorError.metalNotAvailable()
        }

        guard !candidates.isEmpty else {
            return []
        }

        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)

        let queryArray = query.toArray()
        let candidateArrays = candidates.map { $0.toArray() }

        switch metric {
        case .euclidean:
            return try await engine.batchEuclideanDistance(query: queryArray, candidates: candidateArrays)
        case .cosine:
            return try await engine.batchCosineDistance(query: queryArray, candidates: candidateArrays)
        default:
            // For other metrics, compute individually
            return try await withThrowingTaskGroup(of: (Int, Float).self) { group in
                for (index, candidate) in candidateArrays.enumerated() {
                    group.addTask {
                        let dist: Float
                        switch metric {
                        case .dotProduct:
                            dist = try await engine.dotProduct(queryArray, candidate)
                        case .manhattan:
                            dist = try await engine.manhattanDistance(queryArray, candidate)
                        case .chebyshev:
                            dist = try await engine.chebyshevDistance(queryArray, candidate)
                        default:
                            dist = try await engine.euclideanDistance(queryArray, candidate)
                        }
                        return (index, dist)
                    }
                }

                var results = [(Int, Float)]()
                for try await (index, dist) in group {
                    results.append((index, dist))
                }
                return results.sorted { $0.0 < $1.0 }.map { $0.1 }
            }
        }
    }

    // MARK: - GPU-Accelerated Batch Vector Operations

    /// Normalize vectors in batch using GPU
    static func normalizeGPU<V: VectorProtocol & Sendable>(
        _ vectors: [V]
    ) async throws -> [V] where V.Scalar == Float {
        guard ComputeDevice.gpu().isAvailable else {
            throw VectorError.metalNotAvailable()
        }

        guard !vectors.isEmpty else {
            return []
        }

        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)

        var results: [V] = []
        results.reserveCapacity(vectors.count)

        for vector in vectors {
            let array = vector.toArray()
            let normalized = try await engine.normalize(array)

            if let result = try? V(normalized) {
                results.append(result)
            }
        }

        return results
    }

    /// Fast batch normalization using normalizedUnchecked() (VectorCore 0.1.5)
    static func normalizeGPUUnchecked<V: VectorProtocol & Sendable>(
        _ vectors: [V]
    ) async throws -> [V] where V.Scalar == Float {
        guard !vectors.isEmpty else {
            return []
        }

        if ComputeDevice.gpu().isAvailable {
            return try await normalizeGPU(vectors)
        }

        // CPU fallback
        var results: [V] = []
        results.reserveCapacity(vectors.count)

        for vector in vectors {
            results.append(vector.normalizedUnchecked())
        }

        return results
    }

    /// Scale vectors by a constant using GPU
    static func scaleGPU<V: VectorProtocol & Sendable>(
        _ vectors: [V],
        by scalar: Float
    ) async throws -> [V] where V.Scalar == Float {
        guard ComputeDevice.gpu().isAvailable else {
            throw VectorError.metalNotAvailable()
        }

        guard !vectors.isEmpty else {
            return []
        }

        let context = try await Metal4Context()
        let operations = try await AcceleratedVectorOperations(context: context)

        var results: [V] = []
        results.reserveCapacity(vectors.count)

        for vector in vectors {
            let scaled = try await operations.scale(vector, by: scalar)
            results.append(scaled)
        }

        return results
    }

    /// Compute pairwise distances using GPU acceleration
    static func pairwiseDistancesGPU<V: VectorProtocol & Sendable>(
        _ vectors: [V],
        metric: SupportedDistanceMetric = .euclidean
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard ComputeDevice.gpu().isAvailable else {
            throw VectorError.metalNotAvailable()
        }

        let n = vectors.count
        guard n > 0 else {
            return []
        }

        // For small matrices, CPU is more efficient
        if n < 100 {
            switch metric {
            case .euclidean:
                return await pairwiseDistances(vectors, metric: EuclideanDistance())
            case .cosine:
                return await pairwiseDistances(vectors, metric: CosineDistance())
            case .dotProduct:
                return await pairwiseDistances(vectors, metric: DotProductDistance())
            case .manhattan:
                return await pairwiseDistances(vectors, metric: ManhattanDistance())
            case .chebyshev:
                return await pairwiseDistances(vectors, metric: EuclideanDistance())
            }
        }

        let context = try await Metal4Context()
        let engine = try await Metal4ComputeEngine(context: context)

        let arrays = vectors.map { $0.toArray() }
        var matrix = Array(repeating: Array(repeating: Float(0), count: n), count: n)

        // Process rows
        for i in 0..<n {
            let query = arrays[i]
            let candidates = Array(arrays[i..<n])

            let distances: [Float]
            switch metric {
            case .euclidean:
                distances = try await engine.batchEuclideanDistance(query: query, candidates: candidates)
            case .cosine:
                distances = try await engine.batchCosineDistance(query: query, candidates: candidates)
            default:
                // Compute individually for other metrics
                var dists: [Float] = []
                for candidate in candidates {
                    switch metric {
                    case .dotProduct:
                        dists.append(try await engine.dotProduct(query, candidate))
                    case .manhattan:
                        dists.append(try await engine.manhattanDistance(query, candidate))
                    case .chebyshev:
                        dists.append(try await engine.chebyshevDistance(query, candidate))
                    default:
                        dists.append(try await engine.euclideanDistance(query, candidate))
                    }
                }
                distances = dists
            }

            // Fill symmetric matrix
            for j in i..<n {
                let dist = distances[j - i]
                matrix[i][j] = dist
                matrix[j][i] = dist
            }
        }

        return matrix
    }

    // MARK: - Helper Methods

    /// Select top-k elements using heap selection
    private static func selectTopK(
        _ elements: [(index: Int, distance: Float)],
        k: Int
    ) -> [(index: Int, distance: Float)] {
        guard k < elements.count else {
            return elements.sorted { $0.distance < $1.distance }
        }

        var heap = [(index: Int, distance: Float)]()
        heap.reserveCapacity(k + 1)

        for element in elements {
            if heap.count < k {
                heap.append(element)
                if heap.count == k {
                    heap.sort { $0.distance > $1.distance }
                }
            } else if element.distance < heap[0].distance {
                heap[0] = element
                var i = 0
                while i < k {
                    let left = 2 * i + 1
                    let right = 2 * i + 2
                    var largest = i

                    if left < heap.count && heap[left].distance > heap[largest].distance {
                        largest = left
                    }
                    if right < heap.count && heap[right].distance > heap[largest].distance {
                        largest = right
                    }

                    if largest == i {
                        break
                    }

                    heap.swapAt(i, largest)
                    i = largest
                }
            }
        }

        return heap.sorted { $0.distance < $1.distance }
    }
}
