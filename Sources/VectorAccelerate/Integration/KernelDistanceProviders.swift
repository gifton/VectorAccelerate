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
public actor L2KernelDistanceProvider: DistanceProvider {
    private let kernel: L2DistanceKernel
    private let context: Metal4Context

    public init() async throws {
        self.context = try await Metal4Context()
        self.kernel = try await L2DistanceKernel(context: context)
    }

    public init(context: Metal4Context) async throws {
        self.context = context
        self.kernel = try await L2DistanceKernel(context: context)
    }

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

        try await context.executeAndWait { commandBuffer, encoder in
            kernel.encode(
                into: encoder,
                commandBuffer: commandBuffer,
                queriesToken: qToken,
                targetsToken: tToken,
                distancesToken: outToken,
                numQueries: 1,
                dimension: dimension,
                computeSqrt: true
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

        // Replicate query for 1:1 pairing with each candidate
        var flatQueries = [Float]()
        flatQueries.reserveCapacity(n * dimension)
        for _ in 0..<n { flatQueries.append(contentsOf: queryArray) }
        let flatTargets: [Float] = candidates.flatMap { $0.toArray() }

        let qToken = try await context.getBuffer(for: flatQueries)
        let tToken = try await context.getBuffer(for: flatTargets)
        let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

        try await context.executeAndWait { commandBuffer, encoder in
            kernel.encode(
                into: encoder,
                commandBuffer: commandBuffer,
                queriesToken: qToken,
                targetsToken: tToken,
                distancesToken: outToken,
                numQueries: n,
                dimension: dimension,
                computeSqrt: true
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
                let params = DotProductParameters(numQueries: 1, numDatabase: 1, dimension: dimension)
                self.dotProductKernel.encode(into: encoder, queries: qToken.buffer, database: tToken.buffer, output: outToken.buffer, parameters: params)
            } else {
                self.kernel.encode(
                    into: encoder,
                    commandBuffer: commandBuffer,
                    queriesToken: qToken,
                    targetsToken: tToken,
                    similaritiesToken: outToken,
                    numQueries: 1,
                    dimension: dimension,
                    outputDistance: true
                )
            }
            qToken.keepAlive(until: commandBuffer)
            tToken.keepAlive(until: commandBuffer)
            outToken.keepAlive(until: commandBuffer)
        }

        let result = outToken.copyData(as: Float.self, count: 1)[0]
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

        let queryNormalized = (query as? any IndexableVector)?.isNormalized == true
        let allCandidatesNormalized = candidates.allSatisfy { ($0 as? any IndexableVector)?.isNormalized == true }
        let useDotProduct = queryNormalized && allCandidatesNormalized

        let qToken: BufferToken
        if useDotProduct {
            qToken = try await context.getBuffer(for: queryArray)
        } else {
            var flatQueries = [Float]()
            flatQueries.reserveCapacity(n * dimension)
            for _ in 0..<n { flatQueries.append(contentsOf: queryArray) }
            qToken = try await context.getBuffer(for: flatQueries)
        }

        let tToken = try await context.getBuffer(for: candidates.flatMap { $0.toArray() })
        let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

        try await context.executeAndWait { commandBuffer, encoder in
            if useDotProduct {
                let params = DotProductParameters(numQueries: 1, numDatabase: n, dimension: dimension)
                self.dotProductKernel.encode(into: encoder, queries: qToken.buffer, database: tToken.buffer, output: outToken.buffer, parameters: params)
            } else {
                self.kernel.encode(
                    into: encoder,
                    commandBuffer: commandBuffer,
                    queriesToken: qToken,
                    targetsToken: tToken,
                    similaritiesToken: outToken,
                    numQueries: n,
                    dimension: dimension,
                    outputDistance: true
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
        return products[0].map { -$0 }
    }
}

// MARK: - Minkowski Kernel Distance Provider

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
        case .chebyshev: p = 100.0
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

    public func distance<T: VectorProtocol>(
        _ vector1: T,
        _ vector2: T
    ) async throws -> Float where T.Scalar == Float {
        let a = vector1.toArray()
        let b = vector2.toArray()

        let result = try await kernel.computeDistance(vectorA: a, vectorB: b)
        return result.distance
    }

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

// MARK: - Universal Kernel Distance Provider

public actor UniversalKernelDistanceProvider: DistanceProvider {
    private let context: Metal4Context

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

            try await context.executeAndWait { commandBuffer, encoder in
                kernel.encode(
                    into: encoder,
                    commandBuffer: commandBuffer,
                    queriesToken: qToken,
                    targetsToken: tToken,
                    distancesToken: outToken,
                    numQueries: 1,
                    dimension: dimension,
                    computeSqrt: true
                )
                qToken.keepAlive(until: commandBuffer)
                tToken.keepAlive(until: commandBuffer)
                outToken.keepAlive(until: commandBuffer)
            }

            return outToken.copyData(as: Float.self, count: 1)[0]

        case .cosine:
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

                try await context.executeAndWait { commandBuffer, encoder in
                    kernel.encode(
                        into: encoder,
                        commandBuffer: commandBuffer,
                        queriesToken: qToken,
                        targetsToken: tToken,
                        similaritiesToken: outToken,
                        numQueries: 1,
                        dimension: dimension,
                        outputDistance: true
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
            return -products[0][0]

        case .manhattan:
            let kernel = try await getMinkowskiKernel()
            return try await kernel.distance(a, b, p: 1.0)

        case .chebyshev:
            let kernel = try await getMinkowskiKernel()
            return try await kernel.distance(a, b, p: 100.0)
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
            
            var flatQueries = [Float]()
            flatQueries.reserveCapacity(n * dimension)
            for _ in 0..<n { flatQueries.append(contentsOf: queryArray) }
            let flatTargets: [Float] = candidates.flatMap { $0.toArray() }

            let qToken = try await context.getBuffer(for: flatQueries)
            let tToken = try await context.getBuffer(for: flatTargets)
            let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

            try await context.executeAndWait { commandBuffer, encoder in
                kernel.encode(
                    into: encoder,
                    commandBuffer: commandBuffer,
                    queriesToken: qToken,
                    targetsToken: tToken,
                    distancesToken: outToken,
                    numQueries: n,
                    dimension: dimension,
                    computeSqrt: true
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
                
                var flatQueries = [Float]()
                flatQueries.reserveCapacity(n * dimension)
                for _ in 0..<n { flatQueries.append(contentsOf: queryArray) }
                let flatTargets: [Float] = candidates.flatMap { $0.toArray() }

                let qToken = try await context.getBuffer(for: flatQueries)
                let tToken = try await context.getBuffer(for: flatTargets)
                let outToken = try await context.getBuffer(size: n * MemoryLayout<Float>.size)

                try await context.executeAndWait { commandBuffer, encoder in
                    kernel.encode(
                        into: encoder,
                        commandBuffer: commandBuffer,
                        queriesToken: qToken,
                        targetsToken: tToken,
                        similaritiesToken: outToken,
                        numQueries: n,
                        dimension: dimension,
                        outputDistance: true
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

    func universalDistanceProvider() -> UniversalKernelDistanceProvider {
        UniversalKernelDistanceProvider(context: self)
    }
}
