//
//  KernelUsageExamples.swift
//  VectorAccelerateBenchmarks
//
//  Comprehensive usage examples for VectorAccelerate GPU kernel primitives.
//
//  This file demonstrates the two-layer API architecture:
//  1. High-Level API: Simple array-based methods for quick prototyping
//  2. Low-Level API: Buffer-based methods for maximum performance and fusion
//
//  Each example category includes:
//  - Basic usage with high-level compute() methods
//  - VectorCore type integration
//  - Pipeline composition with encode() methods
//

import Foundation
import VectorAccelerate
import VectorCore

// MARK: - Example Runner

/// Runner for kernel usage examples.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct KernelUsageExamples {

    /// Run all kernel examples.
    public static func runAllExamples() async throws {
        print("\n" + String(repeating: "=", count: 60))
        print("VectorAccelerate Kernel Usage Examples")
        print(String(repeating: "=", count: 60) + "\n")

        // Initialize context
        let context = try await Metal4Context()
        print("✅ Metal4Context initialized\n")

        // Run example categories
        try await runDistanceExamples(context: context)
        try await runSelectionExamples(context: context)
        try await runQuantizationExamples(context: context)
        try await runStatisticsExamples(context: context)
        try await runPipelineCompositionExamples(context: context)

        print("\n" + String(repeating: "=", count: 60))
        print("All examples completed successfully!")
        print(String(repeating: "=", count: 60) + "\n")
    }
}

// MARK: - Distance Kernel Examples

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension KernelUsageExamples {

    /// Examples for distance computation kernels.
    static func runDistanceExamples(context: Metal4Context) async throws {
        print("📏 DISTANCE KERNELS")
        print(String(repeating: "-", count: 40))

        // ========================================
        // Example 1: L2 (Euclidean) Distance
        // ========================================
        print("\n1. L2 Distance Kernel")

        // Create kernel
        let l2Kernel = try await L2DistanceKernel(context: context)

        // High-level API: [[Float]] → [[Float]]
        let queries: [[Float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ]
        let database: [[Float]] = [
            [1.0, 0.0, 0.0],  // Same as query 0
            [0.0, 0.0, 1.0],  // Unit vector on z-axis
            [0.5, 0.5, 0.5]   // Diagonal
        ]

        let l2Distances = try await l2Kernel.compute(
            queries: queries,
            database: database,
            computeSqrt: true  // Euclidean distance (not squared)
        )

        print("   Query 0 → Database:")
        for (j, dist) in l2Distances[0].enumerated() {
            print("     → db[\(j)]: \(String(format: "%.4f", dist))")
        }
        print("   ✓ Distance from [1,0,0] to itself = \(String(format: "%.4f", l2Distances[0][0]))")

        // ========================================
        // Example 2: Cosine Similarity
        // ========================================
        print("\n2. Cosine Similarity Kernel")

        let cosineKernel = try await CosineSimilarityKernel(context: context)

        // Compute similarity (higher = more similar)
        let similarities = try await cosineKernel.compute(
            queries: queries,
            database: database,
            outputDistance: false,      // Similarity mode
            inputsNormalized: false     // Kernel will normalize
        )

        print("   Query [1,0,0] similarities:")
        print("     → [1,0,0]: \(String(format: "%.4f", similarities[0][0])) (same)")
        print("     → [0,0,1]: \(String(format: "%.4f", similarities[0][1])) (orthogonal)")
        print("     → [.5,.5,.5]: \(String(format: "%.4f", similarities[0][2])) (partial)")

        // ========================================
        // Example 3: Dot Product
        // ========================================
        print("\n3. Dot Product Kernel")

        let dotKernel = try await DotProductKernel(context: context)

        // Single query (uses optimized GEMV path)
        let dotProducts = try await dotKernel.computeSingle(
            query: [1.0, 2.0, 3.0],
            database: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0]
            ]
        )

        print("   [1,2,3] · [1,0,0] = \(String(format: "%.1f", dotProducts[0]))")
        print("   [1,2,3] · [0,1,0] = \(String(format: "%.1f", dotProducts[1]))")
        print("   [1,2,3] · [1,1,1] = \(String(format: "%.1f", dotProducts[2]))")

        // ========================================
        // Example 4: Minkowski Distance
        // ========================================
        print("\n4. Minkowski Distance Kernel")

        // L1 (Manhattan) distance: p = 1
        let minkowskiKernel = try await MinkowskiDistanceKernel(context: context)
        let manhattanDistance = try await minkowskiKernel.distance(
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            p: 1.0  // L1 norm
        )
        print("   Manhattan distance [1,2,3] → [4,5,6] = \(String(format: "%.1f", manhattanDistance))")
        // |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9

        print("\n   ✓ Distance kernels complete")
    }
}

// MARK: - Selection Kernel Examples

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension KernelUsageExamples {

    /// Examples for top-k selection kernels.
    static func runSelectionExamples(context: Metal4Context) async throws {
        print("\n\n🔝 SELECTION KERNELS")
        print(String(repeating: "-", count: 40))

        // ========================================
        // Example 1: Top-K Selection
        // ========================================
        print("\n1. Top-K Selection Kernel")

        let topKKernel = try await TopKSelectionKernel(context: context)

        // Find 3 smallest values from each row
        let valueMatrix: [[Float]] = [
            [5.0, 2.0, 8.0, 1.0, 9.0, 3.0],  // Row 0
            [7.0, 4.0, 0.0, 6.0, 2.0, 8.0]   // Row 1
        ]

        let topKResults = try await topKKernel.select(
            from: valueMatrix,
            k: 3,
            mode: .minimum,  // Find smallest (nearest neighbors)
            sorted: true
        )

        print("   Row 0 - Top 3 smallest:")
        for (idx, val) in topKResults[0] {
            print("     index \(idx): value = \(String(format: "%.1f", val))")
        }

        // ========================================
        // Example 2: Fused L2 + Top-K
        // ========================================
        print("\n2. Fused L2 + Top-K Kernel (Memory Efficient)")

        let fusedKernel = try await FusedL2TopKKernel(context: context)

        // Generate sample data
        let queryVectors: [[Float]] = [
            [0.5, 0.5, 0.5, 0.5]
        ]
        let datasetVectors: [[Float]] = (0..<10).map { i in
            // Create vectors at varying distances from query
            let offset = Float(i) * 0.1
            return [0.5 + offset, 0.5, 0.5, 0.5]
        }

        // Single-pass nearest neighbor (avoids full distance matrix)
        let neighbors = try await fusedKernel.findNearestNeighbors(
            queries: queryVectors,
            dataset: datasetVectors,
            k: 3
        )

        print("   Query [0.5, 0.5, 0.5, 0.5] - 3 nearest neighbors:")
        for (idx, dist) in neighbors[0] {
            print("     Dataset[\(idx)]: distance = \(String(format: "%.4f", dist))")
        }

        print("\n   ✓ Selection kernels complete")
    }
}

// MARK: - Quantization Kernel Examples

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension KernelUsageExamples {

    /// Examples for quantization kernels.
    static func runQuantizationExamples(context: Metal4Context) async throws {
        print("\n\n📦 QUANTIZATION KERNELS")
        print(String(repeating: "-", count: 40))

        // ========================================
        // Example 1: Binary Quantization
        // ========================================
        print("\n1. Binary Quantization Kernel (32x Compression)")

        let binaryKernel = try await BinaryQuantizationKernel(context: context)

        // Sample embedding vectors
        let embeddings: [[Float]] = [
            [0.5, -0.3, 0.8, -0.1, 0.2, -0.9, 0.4, -0.6],
            [-0.2, 0.7, -0.5, 0.3, -0.8, 0.1, -0.4, 0.9]
        ]

        // Quantize to binary (sign-based)
        let quantized = try await binaryKernel.quantize(
            vectors: embeddings,
            config: .init(useSignBit: true)
        )

        print("   Original size: \(quantized.originalBytes) bytes")
        print("   Compressed size: \(quantized.compressedBytes) bytes")
        print("   Compression ratio: \(String(format: "%.1f", quantized.compressionRatio))x")
        print("   Space savings: \(String(format: "%.1f", quantized.spaceSavings))%")

        // ========================================
        // Example 2: Scalar Quantization
        // ========================================
        print("\n2. Scalar Quantization Kernel (INT8)")

        let scalarKernel = try await ScalarQuantizationKernel(context: context)

        // Generate random float vector
        let floatVector: [Float] = (0..<128).map { _ in Float.random(in: -1...1) }

        let scalarResult = try await scalarKernel.quantize(
            floatVector,
            bitWidth: .int8
        )

        print("   Original: Float32 (\(floatVector.count * 4) bytes)")
        print("   Quantized: INT8 (\(scalarResult.quantizedData.count) bytes)")
        print("   Compression: \(String(format: "%.1f", scalarResult.compressionRatio))x")

        // Dequantize and check reconstruction error
        let reconstructed = try await scalarKernel.dequantize(scalarResult, count: floatVector.count)
        var maxError: Float = 0
        for i in 0..<min(floatVector.count, reconstructed.count) {
            let error = abs(floatVector[i] - reconstructed[i])
            maxError = max(maxError, error)
        }
        print("   Max reconstruction error: \(String(format: "%.6f", maxError))")

        print("\n   ✓ Quantization kernels complete")
    }
}

// MARK: - Statistics Kernel Examples

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension KernelUsageExamples {

    /// Examples for statistics and utility kernels.
    static func runStatisticsExamples(context: Metal4Context) async throws {
        print("\n\n📊 STATISTICS & UTILITY KERNELS")
        print(String(repeating: "-", count: 40))

        // ========================================
        // Example 1: Statistics Kernel
        // ========================================
        print("\n1. Statistics Kernel")

        let statsKernel = try await StatisticsKernel(context: context)

        // Generate sample data
        let data: [Float] = (0..<1000).map { _ in Float.random(in: 0...100) }

        // Full statistics including quantiles
        let stats = try await statsKernel.computeStatistics(data, config: .full)

        print("   Count: \(stats.basic.count)")
        print("   Mean: \(String(format: "%.2f", stats.basic.mean))")
        print("   Std Dev: \(String(format: "%.2f", stats.basic.standardDeviation))")
        print("   Min: \(String(format: "%.2f", stats.basic.minimum))")
        print("   Max: \(String(format: "%.2f", stats.basic.maximum))")

        if let quantiles = stats.quantiles {
            print("   Median (Q50): \(String(format: "%.2f", quantiles.median ?? 0))")
            print("   Q25: \(String(format: "%.2f", quantiles.firstQuartile ?? 0))")
            print("   Q75: \(String(format: "%.2f", quantiles.thirdQuartile ?? 0))")
        }

        if let moments = stats.moments {
            print("   Skewness: \(String(format: "%.4f", moments.skewness))")
            print("   Kurtosis: \(String(format: "%.4f", moments.kurtosis))")
        }

        // ========================================
        // Example 2: Parallel Reduction
        // ========================================
        print("\n2. Parallel Reduction Kernel")

        let reductionKernel = try await ParallelReductionKernel(context: context)

        let values: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        let sum = try await reductionKernel.sum(values)
        let (minVal, minIdx) = try await reductionKernel.argMin(values)
        let (maxVal, maxIdx) = try await reductionKernel.argMax(values)

        print("   Sum: \(String(format: "%.0f", sum))")
        print("   Min: \(String(format: "%.0f", minVal)) at index \(minIdx)")
        print("   Max: \(String(format: "%.0f", maxVal)) at index \(maxIdx)")

        // ========================================
        // Example 3: Elementwise Operations
        // ========================================
        print("\n3. Elementwise Operations Kernel")

        let elementwiseKernel = try await ElementwiseKernel(context: context)

        let a: [Float] = [1, 2, 3, 4, 5]
        let b: [Float] = [5, 4, 3, 2, 1]

        let added = try await elementwiseKernel.add(a, b)
        let scaled = try await elementwiseKernel.scale(a, by: 2.0)
        let clamped = try await elementwiseKernel.clamp(a, min: 2.0, max: 4.0)

        print("   a + b = \(added.map { String(format: "%.0f", $0) })")
        print("   a * 2 = \(scaled.map { String(format: "%.0f", $0) })")
        print("   clamp(a, 2, 4) = \(clamped.map { String(format: "%.0f", $0) })")

        // ========================================
        // Example 4: L2 Normalization
        // ========================================
        print("\n4. L2 Normalization Kernel")

        let normKernel = try await L2NormalizationKernel(context: context)

        let vectors: [[Float]] = [
            [3.0, 4.0, 0.0],  // norm = 5
            [1.0, 1.0, 1.0]   // norm = sqrt(3)
        ]

        let normResult = try await normKernel.normalize(vectors, storeNorms: true)
        let normalized = normResult.asArrays()
        let norms = normResult.normsAsArray() ?? []

        print("   Vector [3, 4, 0] (norm = \(String(format: "%.2f", norms[0]))):")
        print("     → normalized: [\(normalized[0].map { String(format: "%.2f", $0) }.joined(separator: ", "))]")

        print("\n   ✓ Statistics & utility kernels complete")
    }
}

// MARK: - Pipeline Composition Examples

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension KernelUsageExamples {

    /// Examples for composing multiple kernels in a single command buffer.
    static func runPipelineCompositionExamples(context: Metal4Context) async throws {
        print("\n\n🔗 PIPELINE COMPOSITION")
        print(String(repeating: "-", count: 40))

        // ========================================
        // Example 1: Normalize → Cosine → Top-K
        // ========================================
        print("\n1. Fused Normalize + Cosine + Top-K Pipeline")

        let normKernel = try await L2NormalizationKernel(context: context)
        let cosineKernel = try await CosineSimilarityKernel(context: context)
        let topKKernel = try await TopKSelectionKernel(context: context)

        // Prepare data
        let queryVectors: [[Float]] = [[1.0, 2.0, 3.0, 4.0]]
        let databaseVectors: [[Float]] = (0..<100).map { _ in
            (0..<4).map { _ in Float.random(in: -1...1) }
        }

        let device = context.device.rawDevice

        // Flatten and create buffers
        let flatQueries = queryVectors.flatMap { $0 }
        let flatDatabase = databaseVectors.flatMap { $0 }

        guard let queryBufferRaw = device.makeBuffer(
            bytes: flatQueries,
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        nonisolated(unsafe) let queryBuffer = queryBufferRaw

        guard let databaseBufferRaw = device.makeBuffer(
            bytes: flatDatabase,
            length: flatDatabase.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatDatabase.count * MemoryLayout<Float>.size)
        }
        nonisolated(unsafe) let databaseBuffer = databaseBufferRaw

        // Allocate intermediate buffers
        guard let normalizedQueriesRaw = device.makeBuffer(
            length: flatQueries.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatQueries.count * MemoryLayout<Float>.size)
        }
        nonisolated(unsafe) let normalizedQueries = normalizedQueriesRaw

        guard let normalizedDatabaseRaw = device.makeBuffer(
            length: flatDatabase.count * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: flatDatabase.count * MemoryLayout<Float>.size)
        }
        nonisolated(unsafe) let normalizedDatabase = normalizedDatabaseRaw

        let similaritySize = queryVectors.count * databaseVectors.count * MemoryLayout<Float>.size
        guard let similarityBufferRaw = device.makeBuffer(
            length: similaritySize,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: similaritySize)
        }
        nonisolated(unsafe) let similarityBuffer = similarityBufferRaw

        let k = 5
        guard let topKValuesRaw = device.makeBuffer(
            length: queryVectors.count * k * MemoryLayout<Float>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: queryVectors.count * k * MemoryLayout<Float>.size)
        }
        nonisolated(unsafe) let topKValues = topKValuesRaw

        guard let topKIndicesRaw = device.makeBuffer(
            length: queryVectors.count * k * MemoryLayout<UInt32>.size,
            options: .storageModeShared
        ) else {
            throw VectorError.bufferAllocationFailed(size: queryVectors.count * k * MemoryLayout<UInt32>.size)
        }
        nonisolated(unsafe) let topKIndices = topKIndicesRaw

        // Execute fused pipeline
        try await context.executeAndWait { _, encoder in
            // Step 1: Normalize queries
            normKernel.encode(
                into: encoder,
                input: queryBuffer,
                output: normalizedQueries,
                norms: nil,
                parameters: L2NormalizationParameters(
                    numVectors: queryVectors.count,
                    dimension: 4
                )
            )
            encoder.memoryBarrier(scope: .buffers)

            // Step 2: Normalize database
            normKernel.encode(
                into: encoder,
                input: databaseBuffer,
                output: normalizedDatabase,
                norms: nil,
                parameters: L2NormalizationParameters(
                    numVectors: databaseVectors.count,
                    dimension: 4
                )
            )
            encoder.memoryBarrier(scope: .buffers)

            // Step 3: Compute cosine similarity (inputs already normalized)
            cosineKernel.encode(
                into: encoder,
                queries: normalizedQueries,
                database: normalizedDatabase,
                output: similarityBuffer,
                parameters: CosineSimilarityParameters(
                    numQueries: queryVectors.count,
                    numDatabase: databaseVectors.count,
                    dimension: 4,
                    outputDistance: false,
                    inputsNormalized: true
                )
            )
            encoder.memoryBarrier(scope: .buffers)

            // Step 4: Select top-k (maximum similarity)
            topKKernel.encode(
                into: encoder,
                input: similarityBuffer,
                outputValues: topKValues,
                outputIndices: topKIndices,
                parameters: TopKParameters(
                    batchSize: queryVectors.count,
                    numElements: databaseVectors.count,
                    k: k,
                    mode: .maximum  // Maximum similarity
                )
            )
        }

        // Extract results
        let valuePtr = topKValues.contents().bindMemory(to: Float.self, capacity: k)
        let indexPtr = topKIndices.contents().bindMemory(to: UInt32.self, capacity: k)

        print("   Top \(k) most similar vectors:")
        for i in 0..<k {
            let idx = indexPtr[i]
            let sim = valuePtr[i]
            if idx != 0xFFFFFFFF {
                print("     Rank \(i+1): database[\(idx)] similarity = \(String(format: "%.4f", sim))")
            }
        }

        // ========================================
        // Example 2: Using TopKSelectionKernel.fusedDistanceTopK
        // ========================================
        print("\n2. Built-in Fused Distance + Top-K Helper")

        let l2Kernel = try await L2DistanceKernel(context: context)

        // Use the convenience method for common pattern
        let fusedResult = try await topKKernel.fusedDistanceTopK(
            distanceKernel: l2Kernel,
            queries: queryBuffer,
            database: databaseBuffer,
            distanceParams: L2DistanceParameters(
                numQueries: queryVectors.count,
                numDatabase: databaseVectors.count,
                dimension: 4
            ),
            k: 3,
            mode: .minimum  // Minimum distance (nearest)
        )

        print("   Top 3 nearest (by L2 distance):")
        for (idx, dist) in fusedResult.results(for: 0).prefix(3) {
            print("     database[\(idx)]: distance = \(String(format: "%.4f", dist))")
        }

        print("\n   ✓ Pipeline composition complete")
    }
}

// MARK: - VectorCore Integration Examples

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension KernelUsageExamples {

    /// Run VectorCore integration examples.
    public static func runVectorCoreExamples() async throws {
        print("\n\n🔌 VECTORCORE INTEGRATION")
        print(String(repeating: "-", count: 40))

        let context = try await Metal4Context()

        // Using DynamicVector from VectorCore
        let l2Kernel = try await L2DistanceKernel(context: context)

        // Create VectorCore vectors
        let query = DynamicVector([1.0, 0.0, 0.0])
        let database = [
            DynamicVector([1.0, 0.0, 0.0]),
            DynamicVector([0.0, 1.0, 0.0]),
            DynamicVector([0.707, 0.707, 0.0])
        ]

        // Compute distances using VectorProtocol overload
        let distances = try await l2Kernel.compute(
            queries: [query],
            database: database
        )

        print("   Using DynamicVector from VectorCore:")
        for (i, dist) in distances[0].enumerated() {
            print("     → database[\(i)]: \(String(format: "%.4f", dist))")
        }

        print("\n   ✓ VectorCore integration complete")
    }
}
