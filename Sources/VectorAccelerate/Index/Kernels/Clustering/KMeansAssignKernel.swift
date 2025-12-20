//
//  KMeansAssignKernel.swift
//  VectorAccelerate
//
//  Metal 4 kernel for K-Means point assignment phase.
//
//  Assigns each vector to its nearest centroid using GPU-accelerated
//  distance computation. Supports both optimized SIMD-tiled kernels
//  and fallback FusedL2TopK path.
//

import Foundation
@preconcurrency import Metal
import QuartzCore

import VectorCore

// MARK: - K-Means Assign Kernel

/// Metal 4 kernel for K-Means point assignment.
///
/// Assigns each vector to its nearest centroid using GPU-accelerated
/// distance computation. This is the E-step of the EM algorithm.
///
/// ## Kernel Selection
/// - **SIMD-Tiled (Default)**: Uses optimized simdgroup-cooperative kernels with
///   tiled memory access for ~10-20x speedup on typical PQ training workloads.
/// - **FusedL2TopK (Fallback)**: Uses generic fused L2 + Top-K kernel with k=1.
///
/// ## Usage
/// ```swift
/// let kernel = try await KMeansAssignKernel(context: context)
/// let result = try await kernel.assign(
///     vectors: vectorBuffer,
///     centroids: centroidBuffer,
///     numVectors: 10000,
///     numCentroids: 256,
///     dimension: 128
/// )
/// ```
public final class KMeansAssignKernel: @unchecked Sendable, Metal4Kernel {

    // MARK: - Protocol Properties

    public let context: Metal4Context
    public let name: String = "KMeansAssignKernel"

    // MARK: - Private Properties

    /// Fallback kernel using FusedL2TopK with k=1
    private let fusedL2TopK: FusedL2TopKKernel

    /// Optimized SIMD-tiled pipeline (512 threads per threadgroup)
    private var simdTiled512Pipeline: (any MTLComputePipelineState)?

    /// Optimized SIMD-tiled pipeline (256 threads per threadgroup)
    private var simdTiled256Pipeline: (any MTLComputePipelineState)?

    /// Whether to use the optimized SIMD-tiled kernels
    private let useOptimizedKernels: Bool

    // MARK: - Constants

    /// Vectors per threadgroup for 512-thread variant
    private static let vectorsPerTG512 = 16

    /// Vectors per threadgroup for 256-thread variant
    private static let vectorsPerTG256 = 8

    // MARK: - Initialization

    /// Create a K-Means assignment kernel.
    ///
    /// - Parameter context: The Metal 4 context to use
    public init(context: Metal4Context) async throws {
        self.context = context
        self.fusedL2TopK = try await FusedL2TopKKernel(context: context)

        // Try to load optimized SIMD-tiled kernels
        var loadedOptimized = false
        do {
            let library = try await context.shaderCompiler.getDefaultLibrary()

            if let function512 = library.makeFunction(name: "assign_to_centroids_simd_tiled_512") {
                self.simdTiled512Pipeline = try await context.device.rawDevice.makeComputePipelineState(function: function512)
            }

            if let function256 = library.makeFunction(name: "assign_to_centroids_simd_tiled_256") {
                self.simdTiled256Pipeline = try await context.device.rawDevice.makeComputePipelineState(function: function256)
            }

            loadedOptimized = (simdTiled512Pipeline != nil || simdTiled256Pipeline != nil)

            if loadedOptimized {
                VectorLogDebug("Loaded optimized SIMD-tiled KMeans kernels", category: "KMeansAssign")
            }
        } catch {
            VectorLogDebug("Failed to load optimized kernels, using fallback: \(error)", category: "KMeansAssign")
        }

        self.useOptimizedKernels = loadedOptimized
    }

    // MARK: - Warm Up

    public func warmUp() async throws {
        try await fusedL2TopK.warmUp()

        // Warm up optimized kernels with a small dispatch if available
        if let pipeline = simdTiled512Pipeline ?? simdTiled256Pipeline {
            guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return
            }
            encoder.setComputePipelineState(pipeline)
            encoder.endEncoding()
            commandBuffer.commit()
        }
    }

    // MARK: - Assignment

    /// Assign vectors to nearest centroids.
    ///
    /// - Parameters:
    ///   - vectors: Vector buffer [numVectors × dimension]
    ///   - centroids: Centroid buffer [numCentroids × dimension]
    ///   - numVectors: Number of vectors to assign
    ///   - numCentroids: Number of centroids (K)
    ///   - dimension: Vector dimension
    /// - Returns: Assignment result with cluster assignments and distances
    public func assign(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int
    ) async throws -> KMeansAssignmentResult {
        let startTime = CACurrentMediaTime()

        guard numVectors > 0 else {
            throw IndexError.invalidInput(message: "numVectors must be positive")
        }
        guard numCentroids > 0 else {
            throw IndexError.invalidInput(message: "numCentroids must be positive")
        }

        // Use optimized SIMD-tiled kernels when available
        if useOptimizedKernels {
            return try await assignSimdTiled(
                vectors: vectors,
                centroids: centroids,
                numVectors: numVectors,
                numCentroids: numCentroids,
                dimension: dimension,
                startTime: startTime
            )
        }

        // Fallback: Use fused L2 + Top-K with k=1 to find nearest centroid
        return try await assignFusedL2TopK(
            vectors: vectors,
            centroids: centroids,
            numVectors: numVectors,
            numCentroids: numCentroids,
            dimension: dimension,
            startTime: startTime
        )
    }

    // MARK: - Optimized SIMD-Tiled Assignment

    /// Assign vectors using the optimized SIMD-tiled kernels.
    private func assignSimdTiled(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int,
        startTime: CFTimeInterval
    ) async throws -> KMeansAssignmentResult {
        let device = context.device.rawDevice

        // Allocate output buffers
        guard let assignmentsBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numVectors * MemoryLayout<UInt32>.size)
        }
        assignmentsBuffer.label = "KMeansAssign.assignments"

        guard let distancesBuffer = device.makeBuffer(
            length: numVectors * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numVectors * MemoryLayout<Float>.size)
        }
        distancesBuffer.label = "KMeansAssign.distances"

        // Select pipeline variant based on device capabilities
        let pipeline: any MTLComputePipelineState
        let threadsPerThreadgroup: MTLSize
        let vectorsPerTG: Int

        if let pipeline512 = simdTiled512Pipeline,
           pipeline512.maxTotalThreadsPerThreadgroup >= 512 {
            pipeline = pipeline512
            threadsPerThreadgroup = MTLSize(width: 512, height: 1, depth: 1)
            vectorsPerTG = Self.vectorsPerTG512
        } else if let pipeline256 = simdTiled256Pipeline {
            pipeline = pipeline256
            threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            vectorsPerTG = Self.vectorsPerTG256
        } else {
            // Fall back to FusedL2TopK if no optimized kernel available
            return try await assignFusedL2TopK(
                vectors: vectors,
                centroids: centroids,
                numVectors: numVectors,
                numCentroids: numCentroids,
                dimension: dimension,
                startTime: startTime
            )
        }

        // Calculate threadgroups
        let tgCount = (numVectors + vectorsPerTG - 1) / vectorsPerTG
        let threadgroups = MTLSize(width: tgCount, height: 1, depth: 1)

        // Prepare args struct
        var args = KMeansAssignShaderArgs(
            dimension: dimension,
            numVectors: numVectors,
            numCentroids: numCentroids
        )

        // Create command buffer and encoder
        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw IndexError.bufferError(
                operation: "assignSimdTiled",
                reason: "Failed to create command buffer or encoder"
            )
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(vectors, offset: 0, index: 0)
        encoder.setBuffer(centroids, offset: 0, index: 1)
        encoder.setBuffer(assignmentsBuffer, offset: 0, index: 2)
        encoder.setBuffer(distancesBuffer, offset: 0, index: 3)
        encoder.setBytes(&args, length: MemoryLayout<KMeansAssignShaderArgs>.stride, index: 4)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        await commandBuffer.commitAndWait()

        let executionTime = CACurrentMediaTime() - startTime

        return KMeansAssignmentResult(
            assignments: assignmentsBuffer,
            distances: distancesBuffer,
            numVectors: numVectors,
            executionTime: executionTime
        )
    }

    // MARK: - Fallback FusedL2TopK Assignment

    /// Assign vectors using FusedL2TopK with k=1 (fallback path).
    private func assignFusedL2TopK(
        vectors: any MTLBuffer,
        centroids: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int,
        dimension: Int,
        startTime: CFTimeInterval
    ) async throws -> KMeansAssignmentResult {
        let params = try FusedL2TopKParameters(
            numQueries: numVectors,
            numDataset: numCentroids,
            dimension: dimension,
            k: 1  // We only need the nearest centroid
        )

        let result = try await fusedL2TopK.execute(
            queries: vectors,
            dataset: centroids,
            parameters: params,
            config: Metal4FusedL2Config(includeDistances: true)
        )

        let executionTime = CACurrentMediaTime() - startTime

        return KMeansAssignmentResult(
            assignments: result.indices,
            distances: result.distances!,
            numVectors: numVectors,
            executionTime: executionTime
        )
    }

    /// Assign vectors from Float arrays.
    ///
    /// - Parameters:
    ///   - vectors: Vector data as 2D Float array
    ///   - centroids: Centroid data as 2D Float array
    /// - Returns: Tuple of (assignments, distances)
    public func assign(
        vectors: [[Float]],
        centroids: [[Float]]
    ) async throws -> (assignments: [Int], distances: [Float]) {
        guard !vectors.isEmpty else { return ([], []) }
        guard !centroids.isEmpty else {
            throw IndexError.invalidInput(message: "Centroids cannot be empty")
        }

        let dimension = vectors[0].count
        guard vectors.allSatisfy({ $0.count == dimension }),
              centroids.allSatisfy({ $0.count == dimension }) else {
            throw IndexError.invalidInput(message: "All vectors must have same dimension")
        }

        let device = context.device.rawDevice

        // Create vector buffer
        let flatVectors = vectors.flatMap { $0 }
        guard let vectorBuffer = device.makeBuffer(
            bytes: flatVectors,
            length: flatVectors.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatVectors.count * MemoryLayout<Float>.size)
        }
        vectorBuffer.label = "KMeansAssign.vectors"

        // Create centroid buffer
        let flatCentroids = centroids.flatMap { $0 }
        guard let centroidBuffer = device.makeBuffer(
            bytes: flatCentroids,
            length: flatCentroids.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatCentroids.count * MemoryLayout<Float>.size)
        }
        centroidBuffer.label = "KMeansAssign.centroids"

        // Execute
        let result = try await assign(
            vectors: vectorBuffer,
            centroids: centroidBuffer,
            numVectors: vectors.count,
            numCentroids: centroids.count,
            dimension: dimension
        )

        // Extract results
        let assignPtr = result.assignments.contents().bindMemory(to: UInt32.self, capacity: vectors.count)
        let distPtr = result.distances.contents().bindMemory(to: Float.self, capacity: vectors.count)

        let assignments = (0..<vectors.count).map { Int(assignPtr[$0]) }
        let distances = (0..<vectors.count).map { distPtr[$0] }

        return (assignments, distances)
    }

    /// Compute cluster counts from assignments.
    ///
    /// - Parameters:
    ///   - assignments: Assignment buffer [numVectors]
    ///   - numVectors: Number of vectors
    ///   - numCentroids: Number of centroids
    /// - Returns: Buffer with count per cluster [numCentroids]
    public func computeClusterCounts(
        assignments: any MTLBuffer,
        numVectors: Int,
        numCentroids: Int
    ) async throws -> any MTLBuffer {
        let device = context.device.rawDevice

        // Allocate counts buffer
        guard let countsBuffer = device.makeBuffer(
            length: numCentroids * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: numCentroids * MemoryLayout<UInt32>.size)
        }
        countsBuffer.label = "KMeansAssign.counts"

        // Initialize to zero
        let countsPtr = countsBuffer.contents().bindMemory(to: UInt32.self, capacity: numCentroids)
        for i in 0..<numCentroids {
            countsPtr[i] = 0
        }

        // Count assignments (CPU for now - could be GPU with atomics)
        let assignPtr = assignments.contents().bindMemory(to: UInt32.self, capacity: numVectors)
        for i in 0..<numVectors {
            let cluster = Int(assignPtr[i])
            if cluster < numCentroids {
                countsPtr[cluster] += 1
            }
        }

        return countsBuffer
    }
}
