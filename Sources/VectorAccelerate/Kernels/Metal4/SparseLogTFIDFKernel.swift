//
//  SparseLogTFIDFKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for class-based TF-IDF (c-TF-IDF) computation.
//
//  Features:
//  - Sparse c-TF-IDF score computation
//  - Vectorized float4 variant for aligned data
//  - Per-cluster top-K term extraction
//  - FusibleKernel conformance
//
//  Mathematical Background:
//  c-TF-IDF highlights terms that are frequent within a cluster but rare
//  across the corpus:
//    c-TF-IDF(term, cluster) = tf(term, cluster) * log(1 + avgClusterSize / tf(term, corpus))
//
//  Primary use cases:
//  - Topic keyword extraction
//  - Document clustering analysis
//  - Semantic similarity via term overlap
//

import Foundation
@preconcurrency import Metal
import QuartzCore
import VectorCore

// MARK: - Supporting Types

/// Represents sparse term frequencies for a cluster.
public struct ClusterTermFrequencies: Sendable {
    /// Term indices (vocabulary IDs).
    public var termIndices: [UInt32]
    /// Corresponding term frequencies.
    public var frequencies: [Float]

    public init(termIndices: [UInt32], frequencies: [Float]) {
        precondition(termIndices.count == frequencies.count,
                     "termIndices and frequencies must have the same length")
        self.termIndices = termIndices
        self.frequencies = frequencies
    }

    /// Initialize from a dictionary mapping term index to count.
    public init(from dictionary: [Int: Int]) {
        self.termIndices = dictionary.keys.map { UInt32($0) }
        self.frequencies = dictionary.values.map { Float($0) }
    }

    /// Initialize from a dictionary with Float values.
    public init(fromFloats dictionary: [Int: Float]) {
        self.termIndices = dictionary.keys.map { UInt32($0) }
        self.frequencies = Array(dictionary.values)
    }

    /// Number of non-zero terms in this cluster.
    public var count: Int { termIndices.count }

    /// Whether this cluster has no terms.
    public var isEmpty: Bool { termIndices.isEmpty }
}

// MARK: - Result Types

/// Result from c-TF-IDF score computation.
public struct CTFIDFResult: Sendable {
    /// c-TF-IDF scores per cluster, matching input sparsity pattern.
    public let scoresPerCluster: [[Float]]
    /// Total non-zero entries processed.
    public let nnz: Int
    /// Number of clusters processed.
    public let clusterCount: Int
    /// Execution time in seconds.
    public let executionTime: TimeInterval
    /// Memory throughput in GB/s.
    public let throughputGBps: Double
}

/// Result from top-K extraction.
public struct CTFIDFTopKResult: Sendable {
    /// Top-K terms per cluster: [(termIndex, score)]
    public let topKPerCluster: [[(termIndex: Int, score: Float)]]
    /// Number of clusters processed.
    public let clusterCount: Int
    /// K value used.
    public let k: Int
    /// Execution time in seconds.
    public let executionTime: TimeInterval
    /// Memory throughput in GB/s.
    public let throughputGBps: Double
}

// MARK: - GPU Structures

/// GPU parameter structure for c-TF-IDF computation.
struct CTFIDFParamsGPU {
    var avgClusterSize: Float
    var nnz: UInt32
}

// MARK: - Kernel Implementation

/// Metal 4 kernel for class-based TF-IDF (c-TF-IDF) computation.
///
/// Computes c-TF-IDF scores for sparse cluster term frequencies:
/// ```
/// c-TF-IDF(term, cluster) = tf(term, cluster) * log(1 + avgClusterSize / tf(term, corpus))
/// ```
///
/// This metric highlights terms that are:
/// - Frequent within the cluster (high tf in cluster)
/// - Rare across the corpus (low tf in corpus)
///
/// ## Example Usage
///
/// ```swift
/// let kernel = try await SparseLogTFIDFKernel(context: context)
///
/// let clusterTerms = [
///     ClusterTermFrequencies(termIndices: [0, 1, 5], frequencies: [3.0, 2.0, 5.0]),
///     ClusterTermFrequencies(termIndices: [2, 3], frequencies: [4.0, 1.0])
/// ]
/// let corpusFreqs: [Float] = [10.0, 20.0, 5.0, 15.0, 8.0, 3.0]  // 6-term vocabulary
/// let avgClusterSize: Float = 10.0
///
/// // Compute c-TF-IDF scores
/// let result = try await kernel.compute(
///     clusterTerms: clusterTerms,
///     corpusFrequencies: corpusFreqs,
///     avgClusterSize: avgClusterSize
/// )
///
/// // Or extract top-K keywords per cluster
/// let topK = try await kernel.topKPerCluster(
///     clusterTerms: clusterTerms,
///     corpusFrequencies: corpusFreqs,
///     avgClusterSize: avgClusterSize,
///     k: 10
/// )
/// ```
///
/// ## Performance Notes
///
/// This kernel is optimized for typical topic modeling workloads:
/// - 10-200 clusters
/// - Vocabulary size 5K-50K
/// - ~50 terms per cluster on average
///
/// For small workloads (< 10K total terms), CPU may be faster due to GPU overhead.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class SparseLogTFIDFKernel: @unchecked Sendable, Metal4Kernel, FusibleKernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "SparseLogTFIDFKernel"
    public let fusibleWith: [String] = ["Any"]
    public let requiresBarrierAfter: Bool = true

    // MARK: - Pipelines

    private let ctfidfPipeline: any MTLComputePipelineState
    private let ctfidfVectorizedPipeline: any MTLComputePipelineState
    private let topKPipeline: any MTLComputePipelineState

    // MARK: - Initialization

    /// Create a SparseLogTFIDF kernel.
    ///
    /// - Parameter context: The Metal 4 context to use.
    /// - Throws: If pipeline creation fails.
    public init(context: Metal4Context) async throws {
        self.context = context

        let library = try await context.shaderCompiler.getDefaultLibrary()

        guard let ctfidfFunc = library.makeFunction(name: "sparse_ctfidf_kernel"),
              let ctfidfVecFunc = library.makeFunction(name: "sparse_ctfidf_vectorized_kernel"),
              let topKFunc = library.makeFunction(name: "ctfidf_topk_per_cluster_kernel") else {
            throw VectorError.shaderNotFound(
                name: "SparseLogTFIDF kernels. Ensure SparseLogTFIDF.metal is compiled."
            )
        }

        let device = context.device.rawDevice
        self.ctfidfPipeline = try await device.makeComputePipelineState(function: ctfidfFunc)
        self.ctfidfVectorizedPipeline = try await device.makeComputePipelineState(function: ctfidfVecFunc)
        self.topKPipeline = try await device.makeComputePipelineState(function: topKFunc)
    }

    // MARK: - Warm Up

    /// Warm up the kernel pipelines.
    ///
    /// All pipelines are created in init, so this is a no-op.
    public func warmUp() async throws {
        // Pipelines created in init
    }

    // MARK: - Encode API (for Kernel Fusion)

    /// Encode c-TF-IDF computation into an existing encoder.
    ///
    /// This allows fusing c-TF-IDF with other operations in a single command buffer.
    /// Automatically selects vectorized kernel when nnz is divisible by 4 and >= 16.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder.
    ///   - termIndices: Buffer of term indices [nnz].
    ///   - termFreqs: Buffer of term frequencies [nnz].
    ///   - corpusFreqs: Buffer of corpus frequencies [vocabSize].
    ///   - scores: Output buffer for c-TF-IDF scores [nnz].
    ///   - avgClusterSize: Average tokens per cluster.
    ///   - nnz: Number of non-zero entries.
    /// - Returns: Encoding result for debugging.
    @discardableResult
    public func encode(
        into encoder: any MTLComputeCommandEncoder,
        termIndices: any MTLBuffer,
        termFreqs: any MTLBuffer,
        corpusFreqs: any MTLBuffer,
        scores: any MTLBuffer,
        avgClusterSize: Float,
        nnz: Int
    ) -> Metal4EncodingResult {
        var params = CTFIDFParamsGPU(avgClusterSize: avgClusterSize, nnz: UInt32(nnz))

        // Use vectorized kernel if data is aligned
        if nnz % 4 == 0 && nnz >= 16 {
            encoder.setComputePipelineState(ctfidfVectorizedPipeline)
            encoder.setBuffer(termIndices, offset: 0, index: 0)
            encoder.setBuffer(termFreqs, offset: 0, index: 1)
            encoder.setBuffer(corpusFreqs, offset: 0, index: 2)
            encoder.setBuffer(scores, offset: 0, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<CTFIDFParamsGPU>.size, index: 4)

            let threadCount = nnz / 4
            let config = Metal4ThreadConfiguration.linear(count: threadCount, pipeline: ctfidfVectorizedPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "sparse_ctfidf_vectorized_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        } else {
            encoder.setComputePipelineState(ctfidfPipeline)
            encoder.setBuffer(termIndices, offset: 0, index: 0)
            encoder.setBuffer(termFreqs, offset: 0, index: 1)
            encoder.setBuffer(corpusFreqs, offset: 0, index: 2)
            encoder.setBuffer(scores, offset: 0, index: 3)
            encoder.setBytes(&params, length: MemoryLayout<CTFIDFParamsGPU>.size, index: 4)

            let config = Metal4ThreadConfiguration.linear(count: nnz, pipeline: ctfidfPipeline)
            encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

            return Metal4EncodingResult(
                pipelineName: "sparse_ctfidf_kernel",
                threadgroups: config.threadgroups,
                threadsPerThreadgroup: config.threadsPerThreadgroup
            )
        }
    }

    /// Encode top-K extraction into an existing encoder.
    ///
    /// - Parameters:
    ///   - encoder: The compute command encoder.
    ///   - scores: Buffer of c-TF-IDF scores [nnz].
    ///   - termIndices: Buffer of term indices [nnz].
    ///   - clusterOffsets: Buffer of cluster offsets [numClusters+1].
    ///   - topKIndices: Output buffer for top-K indices [numClusters, k].
    ///   - topKScores: Output buffer for top-K scores [numClusters, k].
    ///   - numClusters: Number of clusters.
    ///   - k: Number of top terms per cluster.
    /// - Returns: Encoding result.
    @discardableResult
    public func encodeTopK(
        into encoder: any MTLComputeCommandEncoder,
        scores: any MTLBuffer,
        termIndices: any MTLBuffer,
        clusterOffsets: any MTLBuffer,
        topKIndices: any MTLBuffer,
        topKScores: any MTLBuffer,
        numClusters: Int,
        k: Int
    ) -> Metal4EncodingResult {
        encoder.setComputePipelineState(topKPipeline)
        encoder.setBuffer(scores, offset: 0, index: 0)
        encoder.setBuffer(termIndices, offset: 0, index: 1)
        encoder.setBuffer(clusterOffsets, offset: 0, index: 2)
        encoder.setBuffer(topKIndices, offset: 0, index: 3)
        encoder.setBuffer(topKScores, offset: 0, index: 4)

        var numClustersU32 = UInt32(numClusters)
        var kU32 = UInt32(k)
        encoder.setBytes(&numClustersU32, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&kU32, length: MemoryLayout<UInt32>.size, index: 6)

        let config = Metal4ThreadConfiguration.linear(count: numClusters, pipeline: topKPipeline)
        encoder.dispatchThreadgroups(config.threadgroups, threadsPerThreadgroup: config.threadsPerThreadgroup)

        return Metal4EncodingResult(
            pipelineName: "ctfidf_topk_per_cluster_kernel",
            threadgroups: config.threadgroups,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
    }

    // MARK: - Execute API (Standalone)

    /// Compute c-TF-IDF scores for sparse cluster term frequencies.
    ///
    /// - Parameters:
    ///   - clusterTerms: Per-cluster term indices and frequencies.
    ///   - corpusFrequencies: Corpus-wide term frequency counts [vocabSize].
    ///   - avgClusterSize: Average number of tokens per cluster.
    /// - Returns: c-TF-IDF scores per cluster.
    /// - Throws: If execution fails.
    public func compute(
        clusterTerms: [ClusterTermFrequencies],
        corpusFrequencies: [Float],
        avgClusterSize: Float
    ) async throws -> CTFIDFResult {
        // Flatten cluster data
        var allIndices: [UInt32] = []
        var allFreqs: [Float] = []
        var offsets: [UInt32] = [0]

        for cluster in clusterTerms {
            allIndices.append(contentsOf: cluster.termIndices)
            allFreqs.append(contentsOf: cluster.frequencies)
            offsets.append(UInt32(allIndices.count))
        }

        let nnz = allIndices.count
        if nnz == 0 {
            return CTFIDFResult(
                scoresPerCluster: clusterTerms.map { _ in [] },
                nnz: 0,
                clusterCount: clusterTerms.count,
                executionTime: 0,
                throughputGBps: 0
            )
        }

        let device = context.device.rawDevice

        // Create input buffers
        guard let indicesBuffer = allIndices.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: allIndices.count * MemoryLayout<UInt32>.size)
        }
        indicesBuffer.label = "CTFIDF.indices"

        guard let freqsBuffer = allFreqs.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: allFreqs.count * MemoryLayout<Float>.size)
        }
        freqsBuffer.label = "CTFIDF.freqs"

        guard let corpusBuffer = corpusFrequencies.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: corpusFrequencies.count * MemoryLayout<Float>.size)
        }
        corpusBuffer.label = "CTFIDF.corpus"

        let scoresSize = nnz * MemoryLayout<Float>.size
        guard let scoresBuffer = device.makeBuffer(length: scoresSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: scoresSize)
        }
        scoresBuffer.label = "CTFIDF.scores"

        // Execute
        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                termIndices: indicesBuffer,
                termFreqs: freqsBuffer,
                corpusFreqs: corpusBuffer,
                scores: scoresBuffer,
                avgClusterSize: avgClusterSize,
                nnz: nnz
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Read back and split by cluster
        let scoresPtr = scoresBuffer.contents().bindMemory(to: Float.self, capacity: nnz)
        var result: [[Float]] = []
        result.reserveCapacity(clusterTerms.count)

        for i in 0..<clusterTerms.count {
            let start = Int(offsets[i])
            let end = Int(offsets[i + 1])
            result.append(Array(UnsafeBufferPointer(start: scoresPtr.advanced(by: start), count: end - start)))
        }

        // Throughput: read indices + freqs + corpus lookups, write scores
        let totalBytes = (nnz * (MemoryLayout<UInt32>.size + 2 * MemoryLayout<Float>.size))
                       + (nnz * MemoryLayout<Float>.size)
        let throughputGBps = Double(totalBytes) / (1e9 * max(executionTime, 1e-9))

        return CTFIDFResult(
            scoresPerCluster: result,
            nnz: nnz,
            clusterCount: clusterTerms.count,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    /// Compute c-TF-IDF and extract top-K terms per cluster.
    ///
    /// - Parameters:
    ///   - clusterTerms: Per-cluster term indices and frequencies.
    ///   - corpusFrequencies: Corpus-wide term frequencies [vocabSize].
    ///   - avgClusterSize: Average tokens per cluster.
    ///   - k: Number of top terms to extract per cluster.
    /// - Returns: Top-K (termIndex, score) pairs per cluster.
    /// - Throws: If execution fails.
    public func topKPerCluster(
        clusterTerms: [ClusterTermFrequencies],
        corpusFrequencies: [Float],
        avgClusterSize: Float,
        k: Int
    ) async throws -> CTFIDFTopKResult {
        let numClusters = clusterTerms.count
        if numClusters == 0 {
            return CTFIDFTopKResult(
                topKPerCluster: [],
                clusterCount: 0,
                k: k,
                executionTime: 0,
                throughputGBps: 0
            )
        }

        // Flatten data
        var allIndices: [UInt32] = []
        var allFreqs: [Float] = []
        var offsets: [UInt32] = [0]

        for cluster in clusterTerms {
            allIndices.append(contentsOf: cluster.termIndices)
            allFreqs.append(contentsOf: cluster.frequencies)
            offsets.append(UInt32(allIndices.count))
        }

        let nnz = allIndices.count
        let device = context.device.rawDevice

        // Handle empty case
        if nnz == 0 {
            return CTFIDFTopKResult(
                topKPerCluster: clusterTerms.map { _ in [] },
                clusterCount: numClusters,
                k: k,
                executionTime: 0,
                throughputGBps: 0
            )
        }

        // Create buffers
        guard let indicesBuffer = allIndices.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: allIndices.count * MemoryLayout<UInt32>.size)
        }
        indicesBuffer.label = "CTFIDF.indices"

        guard let freqsBuffer = allFreqs.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: allFreqs.count * MemoryLayout<Float>.size)
        }
        freqsBuffer.label = "CTFIDF.freqs"

        guard let corpusBuffer = corpusFrequencies.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: corpusFrequencies.count * MemoryLayout<Float>.size)
        }
        corpusBuffer.label = "CTFIDF.corpus"

        guard let offsetsBuffer = offsets.withUnsafeBytes({ bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }) else {
            throw VectorError.bufferAllocationFailed(size: offsets.count * MemoryLayout<UInt32>.size)
        }
        offsetsBuffer.label = "CTFIDF.offsets"

        let scoresSize = nnz * MemoryLayout<Float>.size
        guard let scoresBuffer = device.makeBuffer(length: scoresSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: scoresSize)
        }
        scoresBuffer.label = "CTFIDF.scores"

        let topKIndicesSize = numClusters * k * MemoryLayout<UInt32>.size
        guard let topKIndicesBuffer = device.makeBuffer(length: topKIndicesSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: topKIndicesSize)
        }
        topKIndicesBuffer.label = "CTFIDF.topKIndices"

        let topKScoresSize = numClusters * k * MemoryLayout<Float>.size
        guard let topKScoresBuffer = device.makeBuffer(length: topKScoresSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: topKScoresSize)
        }
        topKScoresBuffer.label = "CTFIDF.topKScores"

        // Execute both kernels
        let startTime = CACurrentMediaTime()
        try await context.executeAndWait { [self] _, encoder in
            // Step 1: Compute c-TF-IDF scores
            self.encode(
                into: encoder,
                termIndices: indicesBuffer,
                termFreqs: freqsBuffer,
                corpusFreqs: corpusBuffer,
                scores: scoresBuffer,
                avgClusterSize: avgClusterSize,
                nnz: nnz
            )

            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Extract top-K per cluster
            self.encodeTopK(
                into: encoder,
                scores: scoresBuffer,
                termIndices: indicesBuffer,
                clusterOffsets: offsetsBuffer,
                topKIndices: topKIndicesBuffer,
                topKScores: topKScoresBuffer,
                numClusters: numClusters,
                k: k
            )
        }
        let executionTime = CACurrentMediaTime() - startTime

        // Read back results
        let indicesPtr = topKIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: numClusters * k)
        let scoresPtr = topKScoresBuffer.contents().bindMemory(to: Float.self, capacity: numClusters * k)

        var result: [[(termIndex: Int, score: Float)]] = []
        result.reserveCapacity(numClusters)

        for c in 0..<numClusters {
            var clusterTopK: [(termIndex: Int, score: Float)] = []
            for j in 0..<k {
                let idx = c * k + j
                let score = scoresPtr[idx]
                let termIdx = indicesPtr[idx]
                // Check for valid entry (not -inf score, not sentinel index)
                if score > -.infinity && termIdx != 0xFFFFFFFF {
                    clusterTopK.append((termIndex: Int(termIdx), score: score))
                }
            }
            result.append(clusterTopK)
        }

        // Throughput calculation
        let totalBytes = (nnz * (MemoryLayout<UInt32>.size + 2 * MemoryLayout<Float>.size))
                       + (nnz * MemoryLayout<Float>.size)
                       + (numClusters * k * (MemoryLayout<UInt32>.size + MemoryLayout<Float>.size))
        let throughputGBps = Double(totalBytes) / (1e9 * max(executionTime, 1e-9))

        return CTFIDFTopKResult(
            topKPerCluster: result,
            clusterCount: numClusters,
            k: k,
            executionTime: executionTime,
            throughputGBps: throughputGBps
        )
    }

    // MARK: - Buffer API

    /// Compute c-TF-IDF scores using pre-created buffers.
    ///
    /// - Parameters:
    ///   - termIndices: Buffer of term indices [nnz].
    ///   - termFreqs: Buffer of term frequencies [nnz].
    ///   - corpusFreqs: Buffer of corpus frequencies [vocabSize].
    ///   - avgClusterSize: Average tokens per cluster.
    ///   - nnz: Number of non-zero entries.
    /// - Returns: Buffer containing c-TF-IDF scores [nnz].
    /// - Throws: If execution fails.
    public func compute(
        termIndices: any MTLBuffer,
        termFreqs: any MTLBuffer,
        corpusFreqs: any MTLBuffer,
        avgClusterSize: Float,
        nnz: Int
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice
        let scoresSize = nnz * MemoryLayout<Float>.size

        guard let scoresBuffer = device.makeBuffer(length: scoresSize, options: .storageModeShared) else {
            throw VectorError.bufferAllocationFailed(size: scoresSize)
        }
        scoresBuffer.label = "CTFIDF.scores"

        try await context.executeAndWait { [self] _, encoder in
            self.encode(
                into: encoder,
                termIndices: termIndices,
                termFreqs: termFreqs,
                corpusFreqs: corpusFreqs,
                scores: scoresBuffer,
                avgClusterSize: avgClusterSize,
                nnz: nnz
            )
        }

        return scoresBuffer
    }

    // MARK: - Convenience APIs

    /// Compute c-TF-IDF and return flattened scores array.
    ///
    /// - Parameters:
    ///   - clusterTerms: Per-cluster term indices and frequencies.
    ///   - corpusFrequencies: Corpus-wide term frequencies.
    ///   - avgClusterSize: Average tokens per cluster.
    /// - Returns: Flattened array of all c-TF-IDF scores.
    public func computeFlat(
        clusterTerms: [ClusterTermFrequencies],
        corpusFrequencies: [Float],
        avgClusterSize: Float
    ) async throws -> [Float] {
        let result = try await compute(
            clusterTerms: clusterTerms,
            corpusFrequencies: corpusFrequencies,
            avgClusterSize: avgClusterSize
        )
        return result.scoresPerCluster.flatMap { $0 }
    }
}
