//
//  VectorAccelerate.swift
//  VectorAccelerate
//
//  Metal 4 GPU acceleration for vector operations and similarity search.
//
//  VectorAccelerate provides a two-layer architecture:
//
//  1. HIGH-LEVEL API: AcceleratedVectorIndex
//     A complete GPU-first vector search solution with insert, search, and
//     index management. Ideal for most use cases.
//
//  2. LOW-LEVEL API: GPU Kernel Primitives
//     Fine-grained control over GPU operations for custom ML pipelines,
//     specialized distance metrics, and advanced quantization workflows.
//

// MARK: - Module Exports

/// Re-export VectorCore for convenience.
/// This allows consumers to use VectorCore types without an additional import.
@_exported import VectorCore

// ==============================================================================
// MARK: - HIGH-LEVEL API: AcceleratedVectorIndex
// ==============================================================================
//
// For users who want a complete vector search solution:
//
// ## Main Types
//
// - ``AcceleratedVectorIndex``: Actor-based GPU vector index with automatic
//   GPU/CPU routing, IVF clustering, and compaction support.
//
// - ``VectorHandle``: Opaque, generation-based handle to a stored vector.
//   Handles remain valid across compaction operations.
//
// - ``IndexSearchResult``: Search result containing handle and distance.
//
// - ``IndexConfiguration``: Index configuration supporting flat and IVF modes.
//   Use `.flat(dimension:capacity:)` for small datasets or
//   `.ivf(dimension:nlist:nprobe:)` for large-scale approximate search.
//
// - ``IndexError``: Error type for index operations.
//
// - ``GPUIndexStats``: Statistics about index state, memory usage, and
//   fragmentation for monitoring and optimization decisions.
//
// ## Quick Start
//
// ```swift
// import VectorAccelerate
//
// // Create a flat index for 768-dimensional embeddings
// let index = try await AcceleratedVectorIndex(
//     configuration: .flat(dimension: 768, capacity: 10_000)
// )
//
// // Insert vectors
// let handle = try await index.insert(embedding)
//
// // Search for similar vectors
// let results = try await index.search(query: queryVector, k: 10)
// for result in results {
//     print("Handle: \(result.handle), Distance: \(result.distance)")
// }
//
// // Check index statistics
// let stats = await index.statistics()
// if stats.shouldCompact {
//     _ = try await index.compact()
// }
// ```
//
// ## IVF Index for Large-Scale Search
//
// ```swift
// // Create an IVF index with 256 clusters, probing 16 during search
// let ivfIndex = try await AcceleratedVectorIndex(
//     configuration: .ivf(dimension: 768, nlist: 256, nprobe: 16)
// )
//
// // Insert training data
// for vector in trainingVectors {
//     _ = try await ivfIndex.insert(vector)
// }
//
// // Train the index (builds cluster centroids)
// try await ivfIndex.train()
//
// // Now searches use approximate nearest neighbor
// let results = try await ivfIndex.search(query: query, k: 10)
// ```

// ==============================================================================
// MARK: - LOW-LEVEL API: GPU Kernel Primitives
// ==============================================================================
//
// For users building custom ML pipelines or requiring fine-grained control
// over GPU operations. All kernels use Metal 4 features including ArgumentTable
// for efficient parameter passing.
//
// ## Getting Started with Kernels
//
// ```swift
// import VectorAccelerate
//
// // Create a Metal 4 context (manages device, queues, and buffers)
// let context = try await Metal4Context()
//
// // Create a kernel
// let kernel = try await L2DistanceKernel(context: context)
//
// // Compute distances
// let distances = try await kernel.compute(queries: queries, database: database)
// ```
//
// ## Distance Kernels
//
// GPU-accelerated distance computation between vector sets.
//
// - ``L2DistanceKernel``: Euclidean (L2) distance with dimension-optimized
//   paths for 384, 512, 768, and 1536 dimensions.
//
// - ``CosineSimilarityKernel``: Cosine similarity/distance with optional
//   input normalization. Supports both similarity (higher = more similar)
//   and distance (lower = more similar) output modes.
//
// - ``DotProductKernel``: Dot product computation for inner product search.
//
// - ``MinkowskiDistanceKernel``: Generalized Minkowski distance supporting
//   L1 (Manhattan), L2 (Euclidean), and L∞ (Chebyshev) via the p parameter.
//
// - ``HammingDistanceKernel``: Hamming distance for binary vectors.
//   Pairs with ``BinaryQuantizationKernel`` for compressed similarity search.
//
// - ``JaccardDistanceKernel``: Jaccard distance for set similarity.
//   Useful for document fingerprints and near-duplicate detection.
//
// - ``AttentionSimilarityKernel``: Learned attention-based similarity
//   for asymmetric query-document matching.
//
// ## Selection Kernels
//
// GPU-accelerated top-k selection algorithms.
//
// - ``TopKSelectionKernel``: Batch top-k selection from distance matrices.
//   Use after distance computation for two-pass search.
//
// - ``FusedL2TopKKernel``: Single-pass L2 distance + top-k selection.
//   Avoids materializing the full Q×N distance matrix for memory efficiency.
//
// - ``StreamingTopKKernel``: Streaming top-k for datasets larger than GPU
//   memory. Maintains running top-k across batches.
//
// - ``WarpOptimizedSelectionKernel``: Warp-cooperative selection using
//   SIMD group operations for maximum throughput.
//
// ## Quantization Kernels
//
// Vector compression for memory-efficient storage and fast approximate search.
//
// - ``ScalarQuantizationKernel``: Scalar quantization to INT8/INT4 with
//   configurable symmetric/asymmetric modes. Achieves 4-8x compression.
//
// - ``BinaryQuantizationKernel``: Binary (1-bit) quantization achieving
//   32x compression. Use with ``HammingDistanceKernel`` for fast search.
//
// - ``ProductQuantizationKernel``: Product quantization with configurable
//   codebooks for high compression ratios (32-64x) with learned codebooks.
//
// - ``NeuralQuantizationKernel``: Learned neural quantization using
//   MLTensor for encoding vectors through projection matrices.
//
// ## Matrix Kernels
//
// GPU-accelerated linear algebra primitives.
//
// - ``MatrixMultiplyKernel``: Tiled GEMM with 32×32×8 tiles and optional
//   simdgroup matrix operations for large matrices.
//
// - ``MatrixVectorKernel``: Optimized matrix-vector multiplication.
//
// - ``MatrixTransposeKernel``: In-place and out-of-place matrix transpose
//   with tiled memory access patterns.
//
// - ``BatchMatrixKernel``: Batched matrix operations for processing
//   multiple independent matrix computations in parallel.
//
// ## Utility Kernels
//
// Statistical, normalization, and element-wise operations.
//
// - ``StatisticsKernel``: Comprehensive statistics including mean, variance,
//   skewness, kurtosis, quantiles, and correlation.
//
// - ``HistogramKernel``: GPU-accelerated histogram computation with
//   uniform, adaptive, logarithmic, and custom binning strategies.
//
// - ``ElementwiseKernel``: Element-wise math operations (add, multiply,
//   subtract, divide, power, exp, log, etc.) with broadcasting support.
//
// - ``ParallelReductionKernel``: Parallel reduction operations including
//   sum, min, max, and product with multi-stage reduction.
//
// - ``L2NormalizationKernel``: Batch vector normalization to unit length.
//   Often used before cosine similarity computation.
//
// ## Context and Infrastructure
//
// - ``Metal4Context``: GPU context managing device, command queues, buffer
//   pools, and residency. Create one context and share across kernels.
//
// - ``Metal4Configuration``: Configuration options for context creation
//   including profiling, buffer pre-allocation, and queue priority.
//
// - ``Metal4Capabilities``: Query device capabilities including MLTensor
//   support, maximum buffer sizes, and optimal thread configurations.
//
// ## Pipeline Composition Example
//
// ```swift
// let context = try await Metal4Context()
// let normalizeKernel = try await L2NormalizationKernel(context: context)
// let cosineKernel = try await CosineSimilarityKernel(context: context)
// let topKKernel = try await TopKSelectionKernel(context: context)
//
// // Prepare buffers
// let queryBuffer = try context.makeBuffer(queries)
// let dbBuffer = try context.makeBuffer(database)
//
// // Fused execution in single command buffer
// let results = try await context.withCommandBuffer { encoder in
//     // Normalize queries
//     let normalized = normalizeKernel.encode(into: encoder, input: queryBuffer)
//     encoder.memoryBarrier(scope: .buffers)
//
//     // Compute similarities
//     let similarities = cosineKernel.encode(into: encoder, queries: normalized, ...)
//     encoder.memoryBarrier(scope: .buffers)
//
//     // Select top-k
//     return topKKernel.encode(into: encoder, input: similarities, k: 10)
// }
// ```

// ==============================================================================
// MARK: - Type Aliases
// ==============================================================================

// MARK: Context Aliases

/// Alias for Metal4Context - the GPU context for all kernel operations.
public typealias GPUContext = Metal4Context

/// Alias for Metal4Configuration - context configuration options.
public typealias GPUConfiguration = Metal4Configuration

/// Alias for Metal4Capabilities - device capability queries.
public typealias GPUCapabilities = Metal4Capabilities

// MARK: Distance Kernel Aliases

/// Alias for L2DistanceKernel - Euclidean distance computation.
public typealias L2Kernel = L2DistanceKernel

/// Alias for CosineSimilarityKernel - cosine similarity/distance.
public typealias CosineKernel = CosineSimilarityKernel

/// Alias for DotProductKernel - dot product computation.
public typealias DotKernel = DotProductKernel

/// Alias for MinkowskiDistanceKernel - generalized Minkowski distance.
public typealias MinkowskiKernel = MinkowskiDistanceKernel

/// Alias for HammingDistanceKernel - Hamming distance for binary vectors.
public typealias HammingKernel = HammingDistanceKernel

/// Alias for JaccardDistanceKernel - Jaccard distance for sets.
public typealias JaccardKernel = JaccardDistanceKernel

// MARK: Selection Kernel Aliases

/// Alias for TopKSelectionKernel - batch top-k selection.
public typealias TopKKernel = TopKSelectionKernel

/// Alias for FusedL2TopKKernel - fused distance + selection.
public typealias FusedTopKKernel = FusedL2TopKKernel

/// Alias for StreamingTopKKernel - streaming top-k for large datasets.
public typealias StreamingKernel = StreamingTopKKernel

// MARK: Quantization Kernel Aliases

/// Alias for ScalarQuantizationKernel - INT8/INT4 quantization.
public typealias ScalarQuantKernel = ScalarQuantizationKernel

/// Alias for BinaryQuantizationKernel - 1-bit quantization.
public typealias BinaryQuantKernel = BinaryQuantizationKernel

/// Alias for ProductQuantizationKernel - PQ with codebooks.
public typealias PQKernel = ProductQuantizationKernel

// MARK: Result Type Aliases

/// Alias for Metal4TopKResult - top-k selection results.
public typealias TopKResult = Metal4TopKResult

/// Alias for Metal4FusedL2TopKResult - fused distance + top-k results.
public typealias FusedTopKResult = Metal4FusedL2TopKResult

/// Alias for Metal4QuantizationResult - scalar quantization results.
public typealias QuantizationResult = Metal4QuantizationResult

/// Alias for Metal4StatisticsResult - statistics computation results.
public typealias StatisticsResult = Metal4StatisticsResult

/// Alias for Metal4HistogramResult - histogram computation results.
public typealias HistogramResult = Metal4HistogramResult

/// Alias for Metal4JaccardResult - Jaccard distance results.
public typealias JaccardResult = Metal4JaccardResult

/// Alias for Metal4MinkowskiResult - Minkowski distance results.
public typealias MinkowskiResult = Metal4MinkowskiResult

/// Alias for Metal4HammingResult - Hamming distance results.
public typealias HammingResult = Metal4HammingResult
