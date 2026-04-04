//
//  KernelDistanceProviders.swift
//  VectorAccelerate
//
//  VectorCore DistanceProvider implementations backed by Metal4 kernels.
//
//  These providers give direct access to the high-performance Metal4 distance
//  kernels through VectorCore's DistanceProvider protocol. Use these when you
//  need the full performance of the GPU kernels with VectorCore type compatibility.
//
//  For most use cases, AcceleratedDistanceProvider (which uses ComputeEngine)
//  is sufficient. Use these kernel providers when:
//  - You need maximum GPU performance for large batches
//  - You want to avoid the ComputeEngine abstraction layer
//  - You need access to kernel-specific features (dimension optimizations, etc.)
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - L2 Kernel Distance Provider

/// VectorCore DistanceProvider backed by Metal4 L2DistanceKernel.
///
/// Provides GPU-accelerated Euclidean distance computation with dimension-specific
/// optimizations for 384, 512, 768, and 1536 dimensions.
///
/// ## Usage
/// ```swift
/// let context = try await Metal4Context()
/// let provider = try await L2KernelDistanceProvider(context: context)
///
/// // Single distance
/// let distance = try await provider.distance(from: v1, to: v2, metric: .euclidean)
///
/// // Batch distances
/// let distances = try await provider.batchDistance(from: query, to: candidates, metric: .euclidean)
/// ```
public actor L2KernelDistanceProvider: DistanceProvider {
    private let kernel: L2DistanceKernel
    private let context: Metal4Context

    /// Create an L2 distance provider with a new context.
    public init() async throws {
        self.context = try await Metal4Context()
        self.kernel = try await L2DistanceKernel(context: context)
    }

    /// Create an L2 distance provider with an existing context.
    public init(context: Metal4Context) async throws {
        self.context = context
        self.kernel = try await L2DistanceKernel(context: context)
    }

    /// Access the underlying kernel for advanced usage.
    public var underlyingKernel: L2DistanceKernel { kernel }

    public func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float {
        guard metric == .euclidean else {
            throw VectorError.invalidInput("L2KernelDistanceProvider only supports euclidean metric")
        }

        let a = vector1.toArray()
        let b = vector2.toArray()
        let dimension: Int = a.count

        let qToken = try await context.getBuffer(for: a)
        let tToken = try await context.getBuffer(for: b)
        let outToken = try await context.getBuffer(size: MemoryLayout<Float>.size)

        let parameters = L2DistanceParameters(
            numQueries: 1,
            numDatabase: 1,
            dimension: dimension,
            computeSqrt: true
        )

        try await context.executeAndWait { commandBuffer, encoder in
            kernel.encode(
                into: encoder,
                queries: qToken.buffer,
                database: tToken.buffer,
                distances: outToken.buffer,
                parameters: parameters
            )
            qToken.keepAlive(until: commandBuffer)
            tToken.keepAlive(until: commandBuffer)
            outToken.keepAlive(until: commandBuffer)
        }

        return outToken.copyData(as: Float.self, count: 1)[0]
    }

    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float {
        guard metric == .euclidean else {
            throw VectorError.invalidInput("L2KernelDistanceProvider only supports euclidean metric")
        }

        guard !candidates.isEmpty else { return [] }

        let queryArray = query.toArray()
        let dimension: Int = queryArray.count
        let n: Int = candidates.count

        let flatTargets: [Float] = candidates.flatMap { $0.toArray() }

        let qToken = try await context.getBuffer(for: queryArray)
        let tToken = try await context.getBuffer(for: flatTargets)
        let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

        let parameters = L2DistanceParameters(
            numQueries: 1,
            numDatabase: n,
            dimension: dimension,
            computeSqrt: true
        )

        try await context.executeAndWait { commandBuffer, encoder in
            kernel.encode(
                into: encoder,
                queries: qToken.buffer,
                database: tToken.buffer,
                distances: outToken.buffer,
                parameters: parameters
            )
            qToken.keepAlive(until: commandBuffer)
            tToken.keepAlive(until: commandBuffer)
            outToken.keepAlive(until: commandBuffer)
        }

        return outToken.copyData(as: Float.self, count: n)
    }
}

// MARK: - Cosine Kernel Distance Provider

/// VectorCore DistanceProvider backed by Metal4 CosineSimilarityKernel.
///
/// Provides GPU-accelerated cosine distance/similarity computation with optional
/// input normalization leveraging IndexableVector hints.
public actor CosineKernelDistanceProvider: DistanceProvider {
    private let kernel: CosineSimilarityKernel
    private let dotProductKernel: DotProductKernel
    private let context: Metal4Context

    public init() async throws {
        self.context = try await Metal4Context()
        self.kernel = try await CosineSimilarityKernel(context: context)
        self.dotProductKernel = try await DotProductKernel(context: context)
    }

    public init(context: Metal4Context) async throws {
        self.context = context
        self.kernel = try await CosineSimilarityKernel(context: context)
        self.dotProductKernel = try await DotProductKernel(context: context)
    }

    public var underlyingKernel: CosineSimilarityKernel { kernel }

    public func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float {
        guard metric == .cosine else {
            throw VectorError.invalidInput("CosineKernelDistanceProvider only supports cosine metric")
        }

        let a = vector1.toArray()
        let b = vector2.toArray()
        let dimension: Int = a.count

        // Check if we can use optimized dot product path (requires both vectors to be normalized)
        let isNormalized = (vector1 as? any IndexableVector)?.isNormalized == true &&
                           (vector2 as? any IndexableVector)?.isNormalized == true

        let qToken = try await context.getBuffer(for: a)
        let tToken = try await context.getBuffer(for: b)
        let outToken = try await context.getBuffer(size: MemoryLayout<Float>.size)

        try await context.executeAndWait { commandBuffer, encoder in
            if isNormalized {
                // For normalized vectors, cosine distance = 1 - dot product
                let params = DotProductParameters(numQueries: 1, numDatabase: 1, dimension: dimension)
                self.dotProductKernel.encode(into: encoder, queries: qToken.buffer, database: tToken.buffer, output: outToken.buffer, parameters: params)
            } else {
                let params = CosineSimilarityParameters(
                    numQueries: 1,
                    numDatabase: 1,
                    dimension: dimension,
                    outputDistance: true
                )
                self.kernel.encode(
                    into: encoder,
                    queries: qToken.buffer,
                    database: tToken.buffer,
                    output: outToken.buffer,
                    parameters: params
                )
            }
            qToken.keepAlive(until: commandBuffer)
            tToken.keepAlive(until: commandBuffer)
            outToken.keepAlive(until: commandBuffer)
        }

        let result = outToken.copyData(as: Float.self, count: 1)[0]
        // If we used dot product, result is the similarity, so we return 1 - result
        return isNormalized ? (1.0 - result) : result
    }

    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float {
        guard metric == .cosine else {
            throw VectorError.invalidInput("CosineKernelDistanceProvider only supports cosine metric")
        }

        guard !candidates.isEmpty else { return [] }

        let queryArray = query.toArray()
        let dimension: Int = queryArray.count
        let n: Int = candidates.count

        // Optimization: check if all candidates are pre-normalized
        let queryNormalized = (query as? any IndexableVector)?.isNormalized == true
        let allCandidatesNormalized = candidates.allSatisfy { ($0 as? any IndexableVector)?.isNormalized == true }
        let useDotProduct = queryNormalized && allCandidatesNormalized

        let qToken = try await context.getBuffer(for: queryArray)
        let tToken = try await context.getBuffer(size: n * dimension * MemoryLayout<Float>.size)
        let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

        let destination = tToken.buffer.contents().bindMemory(to: Float.self, capacity: n * dimension)
        for (i, candidate) in candidates.enumerated() {
            let offset = i * dimension
            candidate.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                destination.advanced(by: offset).update(from: srcBase, count: dimension)
            }
        }

        try await context.executeAndWait { commandBuffer, encoder in
            if useDotProduct {
                let params = DotProductParameters(numQueries: 1, numDatabase: n, dimension: dimension)
                self.dotProductKernel.encode(into: encoder, queries: qToken.buffer, database: tToken.buffer, output: outToken.buffer, parameters: params)
            } else {
                let params = CosineSimilarityParameters(
                    numQueries: n,
                    numDatabase: 1,
                    dimension: dimension,
                    outputDistance: true
                )
                self.kernel.encode(
                    into: encoder,
                    queries: qToken.buffer,
                    database: tToken.buffer,
                    output: outToken.buffer,
                    parameters: params
                )
            }
            qToken.keepAlive(until: commandBuffer)
            tToken.keepAlive(until: commandBuffer)
            outToken.keepAlive(until: commandBuffer)
        }

        let results = outToken.copyData(as: Float.self, count: n)
        return useDotProduct ? results.map { 1.0 - $0 } : results
    }
}

// MARK: - Dot Product Kernel Distance Provider

/// VectorCore DistanceProvider backed by Metal4 DotProductKernel.
///
/// Note: Dot product is typically used as a similarity measure (higher = more similar),
/// not a distance. This provider returns the negative dot product for distance semantics.
public actor DotProductKernelDistanceProvider: DistanceProvider {
    private let kernel: DotProductKernel
    private let context: Metal4Context

    public init() async throws {
        self.context = try await Metal4Context()
        self.kernel = try await DotProductKernel(context: context)
    }

    public init(context: Metal4Context) async throws {
        self.context = context
        self.kernel = try await DotProductKernel(context: context)
    }

    public var underlyingKernel: DotProductKernel { kernel }

    public func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float {
        guard metric == .dotProduct else {
            throw VectorError.invalidInput("DotProductKernelDistanceProvider only supports dotProduct metric")
        }

        let a = vector1.toArray()
        let b = vector2.toArray()

        let products = try await kernel.compute(queries: [a], database: [b])
        // Negate for distance semantics (lower = more similar)
        return -products[0][0]
    }

    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float {
        guard metric == .dotProduct else {
            throw VectorError.invalidInput("DotProductKernelDistanceProvider only supports dotProduct metric")
        }

        guard !candidates.isEmpty else { return [] }

        let queryArray = query.toArray()
        let candidateArrays = candidates.map { $0.toArray() }

        let products = try await kernel.compute(queries: [queryArray], database: candidateArrays)
        // Negate for distance semantics
        return products[0].map { -$0 }
    }
}

// MARK: - Minkowski Kernel Distance Provider

/// VectorCore DistanceProvider backed by Metal4 MinkowskiDistanceKernel.
///
/// Supports Manhattan (L1, p=1), Euclidean (L2, p=2), and Chebyshev (L∞, p→∞) distances.
public actor MinkowskiKernelDistanceProvider: DistanceProvider {
    private let kernel: MinkowskiDistanceKernel
    private let context: Metal4Context

    public init() async throws {
        self.context = try await Metal4Context()
        self.kernel = try await MinkowskiDistanceKernel(context: context)
    }

    public init(context: Metal4Context) async throws {
        self.context = context
        self.kernel = try await MinkowskiDistanceKernel(context: context)
    }

    public var underlyingKernel: MinkowskiDistanceKernel { kernel }

    public func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float {
        let p: Float
        switch metric {
        case .manhattan: p = 1.0
        case .euclidean: p = 2.0
        case .chebyshev: p = 100.0  // Approximates L∞
        default:
            throw VectorError.invalidInput("MinkowskiKernelDistanceProvider supports manhattan, euclidean, chebyshev")
        }

        let a = vector1.toArray()
        let b = vector2.toArray()

        return try await kernel.distance(a, b, p: p)
    }

    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float {
        let p: Float
        switch metric {
        case .manhattan: p = 1.0
        case .euclidean: p = 2.0
        case .chebyshev: p = 100.0
        default:
            throw VectorError.invalidInput("MinkowskiKernelDistanceProvider supports manhattan, euclidean, chebyshev")
        }

        guard !candidates.isEmpty else { return [] }

        let queryArray = query.toArray()
        let candidateArrays = candidates.map { $0.toArray() }

        let config = Metal4MinkowskiConfig(p: p)
        let result = try await kernel.computeDistances(
            queries: [queryArray],
            dataset: candidateArrays,
            config: config
        )
        return result.asMatrix()[0]
    }
}

// MARK: - Jaccard Kernel Distance Provider

/// VectorCore DistanceProvider backed by Metal4 JaccardDistanceKernel.
///
/// Computes Jaccard distance (1 - Jaccard similarity) for set-based vectors.
/// Useful for document fingerprints and near-duplicate detection.
public actor JaccardKernelDistanceProvider {
    private let kernel: JaccardDistanceKernel
    private let context: Metal4Context

    public init() async throws {
        self.context = try await Metal4Context()
        self.kernel = try await JaccardDistanceKernel(context: context)
    }

    public init(context: Metal4Context) async throws {
        self.context = context
        self.kernel = try await JaccardDistanceKernel(context: context)
    }

    public var underlyingKernel: JaccardDistanceKernel { kernel }

    /// Compute Jaccard distance between two vectors.
    public func distance<T: VectorProtocol>(
        _ vector1: T,
        _ vector2: T
    ) async throws -> Float where T.Scalar == Float {
        let a = vector1.toArray()
        let b = vector2.toArray()

        let result = try await kernel.computeDistance(vectorA: a, vectorB: b)
        return result.distance
    }

    /// Compute Jaccard distances from query to candidates.
    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T]
    ) async throws -> [Float] where T.Scalar == Float {
        guard !candidates.isEmpty else { return [] }

        let queryArray = query.toArray()
        let candidateArrays = candidates.map { $0.toArray() }

        var distances: [Float] = []
        distances.reserveCapacity(candidates.count)

        for candidate in candidateArrays {
            let result = try await kernel.computeDistance(vectorA: queryArray, vectorB: candidate)
            distances.append(result.distance)
        }

        return distances
    }
}

// MARK: - Hamming Kernel Distance Provider

/// Distance provider backed by Metal4 HammingDistanceKernel.
///
/// Computes Hamming distance for binary vectors. Best used with
/// BinaryQuantizationKernel for compressed similarity search.
public actor HammingKernelDistanceProvider {
    private let kernel: HammingDistanceKernel
    private let context: Metal4Context

    public init() async throws {
        self.context = try await Metal4Context()
        self.kernel = try await HammingDistanceKernel(context: context)
    }

    public init(context: Metal4Context) async throws {
        self.context = context
        self.kernel = try await HammingDistanceKernel(context: context)
    }

    public var underlyingKernel: HammingDistanceKernel { kernel }

    /// Compute Hamming distance between two binary vectors.
    public func distance(
        _ vector1: Metal4BinaryVector,
        _ vector2: Metal4BinaryVector
    ) -> Int {
        vector1.hammingDistance(to: vector2)
    }

    /// Compute Hamming distances from query to candidates.
    public func batchDistance(
        from query: Metal4BinaryVector,
        to candidates: [Metal4BinaryVector]
    ) -> [Int] {
        candidates.map { query.hammingDistance(to: $0) }
    }
}

// MARK: - Universal Kernel Distance Provider

/// Universal distance provider that dispatches to the appropriate Metal4 kernel
/// based on the requested metric.
///
/// This provider maintains a cache of kernels and routes requests to the most
/// efficient implementation for each metric.
///
/// ## Supported Metrics
/// - `.euclidean` → L2DistanceKernel (with dimension optimizations)
/// - `.cosine` → CosineSimilarityKernel (optimized via DotProduct if pre-normalized)
/// - `.dotProduct` → DotProductKernel
/// - `.manhattan` → MinkowskiDistanceKernel (p=1)
/// - `.chebyshev` → MinkowskiDistanceKernel (p=∞)
///
/// ## Usage
/// ```swift
/// let provider = try await UniversalKernelDistanceProvider()
///
/// // Works with any supported metric
/// let euclidean = try await provider.distance(from: v1, to: v2, metric: .euclidean)
/// let cosine = try await provider.distance(from: v1, to: v2, metric: .cosine)
/// ```
public actor UniversalKernelDistanceProvider: DistanceProvider {
    private let context: Metal4Context

    // Lazy kernel cache
    private var l2Kernel: L2DistanceKernel?
    private var cosineKernel: CosineSimilarityKernel?
    private var dotKernel: DotProductKernel?
    private var minkowskiKernel: MinkowskiDistanceKernel?

    public init() async throws {
        self.context = try await Metal4Context()
    }

    public init(context: Metal4Context) {
        self.context = context
    }

    // MARK: - Kernel Accessors

    private func getL2Kernel() async throws -> L2DistanceKernel {
        if let kernel = l2Kernel { return kernel }
        let kernel = try await L2DistanceKernel(context: context)
        l2Kernel = kernel
        return kernel
    }

    private func getCosineKernel() async throws -> CosineSimilarityKernel {
        if let kernel = cosineKernel { return kernel }
        let kernel = try await CosineSimilarityKernel(context: context)
        cosineKernel = kernel
        return kernel
    }

    private func getDotKernel() async throws -> DotProductKernel {
        if let kernel = dotKernel { return kernel }
        let kernel = try await DotProductKernel(context: context)
        dotKernel = kernel
        return kernel
    }

    private func getMinkowskiKernel() async throws -> MinkowskiDistanceKernel {
        if let kernel = minkowskiKernel { return kernel }
        let kernel = try await MinkowskiDistanceKernel(context: context)
        minkowskiKernel = kernel
        return kernel
    }

    // MARK: - DistanceProvider

    public func distance<T: VectorProtocol>(
        from vector1: T,
        to vector2: T,
        metric: SupportedDistanceMetric
    ) async throws -> Float where T.Scalar == Float {
        let a = vector1.toArray()
        let b = vector2.toArray()

        switch metric {
        case .euclidean:
            let kernel = try await getL2Kernel()
            let dimension: Int = a.count

            let qToken = try await context.getBuffer(for: a)
            let tToken = try await context.getBuffer(for: b)
            let outToken = try await context.getBuffer(size: MemoryLayout<Float>.size)

            let params = L2DistanceParameters(
                numQueries: 1,
                numDatabase: 1,
                dimension: dimension,
                computeSqrt: true
            )

            try await context.executeAndWait { commandBuffer, encoder in
                kernel.encode(
                    into: encoder,
                    queries: qToken.buffer,
                    database: tToken.buffer,
                    distances: outToken.buffer,
                    parameters: params
                )
                qToken.keepAlive(until: commandBuffer)
                tToken.keepAlive(until: commandBuffer)
                outToken.keepAlive(until: commandBuffer)
            }

            return outToken.copyData(as: Float.self, count: 1)[0]

        case .cosine:
            // Optimization: Use dot product if both are pre-normalized
            let isNormalized = (vector1 as? any IndexableVector)?.isNormalized == true &&
                               (vector2 as? any IndexableVector)?.isNormalized == true
            
            if isNormalized {
                let kernel = try await getDotKernel()
                let products = try await kernel.compute(queries: [a], database: [b])
                return 1.0 - products[0][0]
            } else {
                let kernel = try await getCosineKernel()
                let dimension: Int = a.count

                let qToken = try await context.getBuffer(for: a)
                let tToken = try await context.getBuffer(for: b)
                let outToken = try await context.getBuffer(size: MemoryLayout<Float>.size)

                let params = CosineSimilarityParameters(
                    numQueries: 1,
                    numDatabase: 1,
                    dimension: dimension,
                    outputDistance: true
                )

                try await context.executeAndWait { commandBuffer, encoder in
                    kernel.encode(
                        into: encoder,
                        queries: qToken.buffer,
                        database: tToken.buffer,
                        output: outToken.buffer,
                        parameters: params
                    )
                    qToken.keepAlive(until: commandBuffer)
                    tToken.keepAlive(until: commandBuffer)
                    outToken.keepAlive(until: commandBuffer)
                }

                return outToken.copyData(as: Float.self, count: 1)[0]
            }

        case .dotProduct:
            let kernel = try await getDotKernel()
            let products = try await kernel.compute(queries: [a], database: [b])
            return -products[0][0]  // Negate for distance semantics

        case .manhattan:
            let kernel = try await getMinkowskiKernel()
            return try await kernel.distance(a, b, p: 1.0)

        case .chebyshev:
            let kernel = try await getMinkowskiKernel()
            return try await kernel.distance(a, b, p: 100.0)  // Approximates L∞
        }
    }

    public func batchDistance<T: VectorProtocol>(
        from query: T,
        to candidates: [T],
        metric: SupportedDistanceMetric
    ) async throws -> [Float] where T.Scalar == Float {
        guard !candidates.isEmpty else { return [] }

        let queryArray = query.toArray()
        let dimension: Int = queryArray.count
        let n: Int = candidates.count

        switch metric {
        case .euclidean:
            let kernel = try await getL2Kernel()
            let flatTargets: [Float] = candidates.flatMap { $0.toArray() }

            let qToken = try await context.getBuffer(for: queryArray)
            let tToken = try await context.getBuffer(for: flatTargets)
            let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

            let params = L2DistanceParameters(
                numQueries: 1,
                numDatabase: n,
                dimension: dimension,
                computeSqrt: true
            )

            try await context.executeAndWait { commandBuffer, encoder in
                kernel.encode(
                    into: encoder,
                    queries: qToken.buffer,
                    database: tToken.buffer,
                    distances: outToken.buffer,
                    parameters: params
                )
                qToken.keepAlive(until: commandBuffer)
                tToken.keepAlive(until: commandBuffer)
                outToken.keepAlive(until: commandBuffer)
            }

            return outToken.copyData(as: Float.self, count: n)

        case .cosine:
            let queryNormalized = (query as? any IndexableVector)?.isNormalized == true
            let allCandidatesNormalized = candidates.allSatisfy { ($0 as? any IndexableVector)?.isNormalized == true }
            
            if queryNormalized && allCandidatesNormalized {
                let kernel = try await getDotKernel()
                let products = try await kernel.compute(queries: [queryArray], database: candidates.map { $0.toArray() })
                return products[0].map { 1.0 - $0 }
            } else {
                let kernel = try await getCosineKernel()
                let flatTargets: [Float] = candidates.flatMap { $0.toArray() }

                let qToken = try await context.getBuffer(for: queryArray)
                let tToken = try await context.getBuffer(for: flatTargets)
                let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

                let params = CosineSimilarityParameters(
                    numQueries: 1,
                    numDatabase: n,
                    dimension: dimension,
                    outputDistance: true
                )

                try await context.executeAndWait { commandBuffer, encoder in
                    kernel.encode(
                        into: encoder,
                        queries: qToken.buffer,
                        database: tToken.buffer,
                        output: outToken.buffer,
                        parameters: params
                    )
                    qToken.keepAlive(until: commandBuffer)
                    tToken.keepAlive(until: commandBuffer)
                    outToken.keepAlive(until: commandBuffer)
                }

                return outToken.copyData(as: Float.self, count: n)
            }

        case .dotProduct:
            let kernel = try await getDotKernel()
            let products = try await kernel.compute(queries: [queryArray], database: candidates.map { $0.toArray() })
            return products[0].map { -$0 }

        case .manhattan:
            let kernel = try await getMinkowskiKernel()
            let config = Metal4MinkowskiConfig(p: 1.0)
            let result = try await kernel.computeDistances(
                queries: [queryArray],
                dataset: candidates.map { $0.toArray() },
                config: config
            )
            return result.asMatrix()[0]

        case .chebyshev:
            let kernel = try await getMinkowskiKernel()
            let config = Metal4MinkowskiConfig(p: 100.0)
            let result = try await kernel.computeDistances(
                queries: [queryArray],
                dataset: candidates.map { $0.toArray() },
                config: config
            )
            return result.asMatrix()[0]
        }
    }
}

// MARK: - Metal4Context Extensions

public extension Metal4Context {

    /// Create a distance kernel for the specified metric.
    ///
    /// Returns the most efficient kernel implementation for the given metric:
    /// - `.euclidean` → L2DistanceKernel (dimension-optimized)
    /// - `.cosine` → CosineSimilarityKernel
    /// - `.dotProduct` → DotProductKernel
    /// - `.manhattan`, `.chebyshev` → MinkowskiDistanceKernel
    ///
    /// ## Usage
    /// ```swift
    /// let context = try await Metal4Context()
    /// let kernel = try await context.distanceKernel(for: .euclidean)
    /// // Use kernel directly...
    /// ```
    func distanceKernel(for metric: SupportedDistanceMetric) async throws -> any Metal4Kernel {
        switch metric {
        case .euclidean:
            return try await L2DistanceKernel(context: self)
        case .cosine:
            return try await CosineSimilarityKernel(context: self)
        case .dotProduct:
            return try await DotProductKernel(context: self)
        case .manhattan, .chebyshev:
            return try await MinkowskiDistanceKernel(context: self)
        }
    }

    /// Create a universal distance provider backed by this context.
    ///
    /// The universal provider handles all supported metrics and caches
    /// kernels for efficient reuse.
    func universalDistanceProvider() -> UniversalKernelDistanceProvider {
        UniversalKernelDistanceProvider(context: self)
    }
}
