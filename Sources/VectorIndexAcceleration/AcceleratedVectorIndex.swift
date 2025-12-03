//
//  AcceleratedVectorIndex.swift
//  VectorIndexAcceleration
//
//  GPU-first vector index with direct GPU data ownership.
//
//  This is the main entry point for the clean API. It provides:
//  - Direct GPU buffer ownership (no CPU copy)
//  - Opaque handle-based vector identification
//  - Generation-based stale handle detection
//  - Lazy deletion with on-demand compaction
//  - Native GPU distances (L2² for euclidean)
//  - Support for both Flat and IVF index types
//

import Foundation
@preconcurrency import Metal
import VectorCore
import VectorAccelerate

// MARK: - Metadata Type

/// Metadata for a vector (optional per-vector key-value pairs).
public typealias VectorMetadata = [String: String]

// MARK: - Accelerated Vector Index

/// GPU-accelerated vector index with direct data ownership.
///
/// `AcceleratedVectorIndex` provides a clean, GPU-first API for similarity search.
/// Unlike wrapper-based approaches, this index owns the vector data directly on
/// the GPU, eliminating redundant CPU copies.
///
/// ## Index Types
/// - **Flat**: Exhaustive search, best for < 10K vectors
/// - **IVF**: Inverted file index with clustering, best for > 10K vectors
///
/// ## Features
/// - **Direct GPU ownership**: Vectors live on GPU, no duplication
/// - **Opaque handles**: `VectorHandle` instead of string IDs
/// - **Native distances**: Returns L2² for euclidean (no expensive sqrt)
/// - **Lazy deletion**: Fast removes with on-demand compaction
/// - **Filtered search**: CPU-side metadata filtering with iterative fetch
///
/// ## Supported Metrics
/// Currently, only `.euclidean` metric is supported for GPU-accelerated search.
/// The search returns L2² (squared euclidean distance) for efficiency.
///
/// ## Flat Index Usage
/// ```swift
/// let index = try await AcceleratedVectorIndex(
///     configuration: .flat(dimension: 768, capacity: 10_000)
/// )
/// let handle = try await index.insert(embedding)
/// let results = try await index.search(query: queryVector, k: 10)
/// ```
///
/// ## IVF Index Usage
/// ```swift
/// let index = try await AcceleratedVectorIndex(
///     configuration: .ivf(dimension: 768, nlist: 256, nprobe: 16, capacity: 100_000)
/// )
/// // Insert vectors - training happens automatically
/// for vector in vectors {
///     _ = try await index.insert(vector)
/// }
/// // Or trigger training manually
/// try await index.train()
/// // Search
/// let results = try await index.search(query: queryVector, k: 10)
/// ```
public actor AcceleratedVectorIndex {

    // MARK: - Configuration

    /// Index configuration.
    public let configuration: IndexConfiguration

    // MARK: - Internal Components

    /// GPU context for compute operations.
    private let context: Metal4Context

    /// GPU buffer storage for vectors.
    private var storage: GPUVectorStorage

    /// Handle allocation and generation tracking.
    private var handleAllocator: HandleAllocator

    /// Deletion mask for lazy deletion.
    private var deletionMask: DeletionMask

    /// CPU-side metadata storage.
    private var metadataStore: MetadataStore

    /// Fused L2 Top-K kernel for flat search.
    private var fusedL2TopKKernel: FusedL2TopKKernel?

    // MARK: - IVF Components (Optional)

    /// IVF structure for inverted file index (nil for flat index).
    private var ivfStructure: IVFStructure?

    /// IVF search pipeline (nil for flat index).
    private var ivfSearchPipeline: IVFSearchPipeline?

    /// Whether auto-training is enabled for IVF.
    private var autoTrainEnabled: Bool = true

    // MARK: - Initialization

    /// Create an accelerated vector index with the given configuration.
    ///
    /// - Parameter configuration: Index configuration specifying dimension, metric, capacity, and type
    /// - Throws: `IndexAccelerationError` if configuration is invalid or GPU initialization fails
    ///
    /// - Note: Currently only `.euclidean` metric is supported for GPU-accelerated search.
    public init(configuration: IndexConfiguration) async throws {
        try configuration.validate()
        try Self.validateMetricSupport(configuration.metric)

        self.configuration = configuration
        self.context = try await Metal4Context()

        // Initialize core components
        self.storage = try GPUVectorStorage(
            device: context.device.rawDevice,
            dimension: configuration.dimension,
            capacity: configuration.capacity
        )
        self.handleAllocator = HandleAllocator(initialCapacity: configuration.capacity)
        self.deletionMask = DeletionMask(capacity: configuration.capacity)
        self.metadataStore = MetadataStore(capacity: configuration.capacity / 10)

        // Initialize index-type-specific components
        switch configuration.indexType {
        case .flat:
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)

        case .ivf(let nlist, let nprobe):
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)
            self.ivfStructure = IVFStructure(
                numClusters: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension
            )
            let ivfConfig = IVFSearchConfiguration(
                numCentroids: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension,
                metric: configuration.metric
            )
            self.ivfSearchPipeline = try await IVFSearchPipeline(
                context: context,
                configuration: ivfConfig
            )
        }
    }

    /// Create an accelerated vector index with an existing Metal context.
    ///
    /// Use this initializer when you want to share a context across multiple indexes.
    ///
    /// - Parameters:
    ///   - configuration: Index configuration
    ///   - context: Existing Metal 4 context to use
    /// - Throws: `IndexAccelerationError` if configuration is invalid
    public init(configuration: IndexConfiguration, context: Metal4Context) async throws {
        try configuration.validate()
        try Self.validateMetricSupport(configuration.metric)

        self.configuration = configuration
        self.context = context

        // Initialize core components
        self.storage = try GPUVectorStorage(
            device: context.device.rawDevice,
            dimension: configuration.dimension,
            capacity: configuration.capacity
        )
        self.handleAllocator = HandleAllocator(initialCapacity: configuration.capacity)
        self.deletionMask = DeletionMask(capacity: configuration.capacity)
        self.metadataStore = MetadataStore(capacity: configuration.capacity / 10)

        // Initialize index-type-specific components
        switch configuration.indexType {
        case .flat:
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)

        case .ivf(let nlist, let nprobe):
            self.fusedL2TopKKernel = try await FusedL2TopKKernel(context: context)
            self.ivfStructure = IVFStructure(
                numClusters: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension
            )
            let ivfConfig = IVFSearchConfiguration(
                numCentroids: nlist,
                nprobe: nprobe,
                dimension: configuration.dimension,
                metric: configuration.metric
            )
            self.ivfSearchPipeline = try await IVFSearchPipeline(
                context: context,
                configuration: ivfConfig
            )
        }
    }

    /// Validate that the metric is supported for GPU acceleration.
    private static func validateMetricSupport(_ metric: SupportedDistanceMetric) throws {
        switch metric {
        case .euclidean:
            break // Supported
        case .cosine, .dotProduct, .manhattan, .chebyshev:
            throw IndexAccelerationError.invalidConfiguration(
                parameter: "metric",
                reason: "'\(metric)' is not yet supported. Currently only .euclidean is available for GPU-accelerated search."
            )
        }
    }

    // MARK: - Properties

    /// Number of active (non-deleted) vectors in the index.
    public var count: Int {
        handleAllocator.occupiedCount
    }

    /// Vector dimension.
    public var dimension: Int {
        configuration.dimension
    }

    /// Whether the index is empty.
    public var isEmpty: Bool {
        handleAllocator.occupiedCount == 0
    }

    /// Distance metric used by this index.
    public var metric: SupportedDistanceMetric {
        configuration.metric
    }

    /// Current capacity of the index.
    public var capacity: Int {
        storage.capacity
    }

    /// Whether the index uses IVF.
    public var isIVF: Bool {
        ivfStructure != nil
    }

    /// Whether the IVF index is trained (always true for flat index).
    public var isTrained: Bool {
        ivfStructure?.isTrained ?? true
    }

    // MARK: - Statistics

    /// Get statistics about the index.
    public func statistics() -> GPUIndexStats {
        let ivfStats = ivfStructure?.getStats()

        return GPUIndexStats(
            vectorCount: handleAllocator.occupiedCount,
            allocatedSlots: storage.allocatedSlots,
            deletedSlots: deletionMask.deletedCount,
            dimension: configuration.dimension,
            metric: configuration.metric,
            indexType: configuration.indexType,
            capacity: storage.capacity,
            gpuVectorMemoryBytes: storage.usedBytes,
            gpuIndexStructureBytes: 0,
            cpuMetadataMemoryBytes: metadataStore.estimatedMemoryBytes,
            ivfStats: ivfStats
        )
    }

    // MARK: - IVF Training

    /// Train the IVF index.
    ///
    /// For IVF indexes, this runs K-Means clustering on the inserted vectors
    /// to compute centroids. Training is required before search returns accurate results.
    ///
    /// For flat indexes, this method does nothing.
    ///
    /// - Note: Training happens automatically when enough vectors are inserted
    ///         (if auto-training is enabled).
    /// - Throws: `IndexAccelerationError` if training fails
    public func train() async throws {
        guard let ivf = ivfStructure else { return }
        guard !ivf.isTrained else { return }

        // Gather all vectors for training
        var trainingVectors: [[Float]] = []
        trainingVectors.reserveCapacity(handleAllocator.occupiedCount)

        for slotIndex in deletionMask {
            let vector = try storage.readVector(at: slotIndex)
            trainingVectors.append(vector)
        }

        guard !trainingVectors.isEmpty else {
            throw IndexAccelerationError.invalidInput(
                message: "Cannot train IVF index with no vectors"
            )
        }

        try await ivf.train(vectors: trainingVectors, context: context)

        // Reassign all vectors to clusters
        for slotIndex in deletionMask {
            let vector = try storage.readVector(at: slotIndex)
            if let handle = handleAllocator.handle(for: UInt32(slotIndex)) {
                _ = ivf.assignToCluster(
                    vector: vector,
                    slotIndex: UInt32(slotIndex),
                    generation: handle.generation
                )
            }
        }
    }

    /// Enable or disable auto-training for IVF indexes.
    ///
    /// When enabled, the index automatically trains when enough vectors are inserted.
    ///
    /// - Parameter enabled: Whether auto-training should be enabled
    public func setAutoTraining(_ enabled: Bool) {
        autoTrainEnabled = enabled
    }

    // MARK: - Insert Operations

    /// Insert a single vector.
    ///
    /// - Parameters:
    ///   - vector: Vector data (must match index dimension)
    ///   - metadata: Optional metadata for filtering
    /// - Returns: Handle to the inserted vector
    /// - Throws: `IndexAccelerationError` if dimension mismatch or GPU error
    public func insert(
        _ vector: consuming [Float],
        metadata: VectorMetadata? = nil
    ) async throws -> VectorHandle {
        guard vector.count == configuration.dimension else {
            throw IndexAccelerationError.dimensionMismatch(
                expected: configuration.dimension,
                got: vector.count
            )
        }

        // Ensure capacity
        try storage.ensureCapacity(storage.allocatedSlots + 1)
        deletionMask.ensureCapacity(storage.allocatedSlots + 1)

        // Allocate handle
        let handle = handleAllocator.allocate()
        let slotIndex = Int(handle.index)

        // Write vector to GPU
        try storage.writeVector(vector, at: slotIndex)

        // Store metadata if provided
        if let metadata = metadata {
            metadataStore[handle.index] = metadata
        }

        // Handle IVF-specific logic
        if let ivf = ivfStructure {
            if ivf.isTrained {
                // Assign to cluster
                _ = ivf.assignToCluster(
                    vector: vector,
                    slotIndex: UInt32(slotIndex),
                    generation: handle.generation
                )
            } else {
                // Add to staging
                ivf.addToStaging(UInt32(slotIndex))

                // Auto-train if we have enough vectors
                if autoTrainEnabled && ivf.canTrain {
                    try await train()
                }
            }
        }

        return handle
    }

    /// Batch insert multiple vectors.
    ///
    /// More efficient than individual inserts for large batches.
    /// Uses optimized batch GPU write operations.
    ///
    /// - Parameters:
    ///   - vectors: Array of vectors (each must match index dimension)
    ///   - metadata: Optional metadata array (must match vectors count if provided)
    /// - Returns: Array of handles for inserted vectors
    /// - Throws: `IndexAccelerationError` if dimension mismatch or GPU error
    public func insert(
        _ vectors: [[Float]],
        metadata: [VectorMetadata?]? = nil
    ) async throws -> [VectorHandle] {
        guard !vectors.isEmpty else { return [] }

        // Validate dimensions
        for vector in vectors {
            guard vector.count == configuration.dimension else {
                throw IndexAccelerationError.dimensionMismatch(
                    expected: configuration.dimension,
                    got: vector.count
                )
            }
        }

        if let metadata = metadata, metadata.count != vectors.count {
            throw IndexAccelerationError.invalidInput(
                message: "Metadata count (\(metadata.count)) must match vectors count (\(vectors.count))"
            )
        }

        // Ensure capacity
        try storage.ensureCapacity(storage.allocatedSlots + vectors.count)
        deletionMask.ensureCapacity(storage.allocatedSlots + vectors.count)

        // Allocate handles
        let handles = handleAllocator.allocate(count: vectors.count)

        // Batch write vectors to GPU (optimized)
        let startSlot = Int(handles[0].index)
        try storage.writeVectors(vectors, startingAt: startSlot)

        // Store metadata
        if let metadata = metadata {
            for (i, meta) in metadata.enumerated() {
                if let meta = meta {
                    metadataStore[handles[i].index] = meta
                }
            }
        }

        // Handle IVF-specific logic
        if let ivf = ivfStructure {
            if ivf.isTrained {
                // Assign each vector to a cluster
                for (i, vector) in vectors.enumerated() {
                    _ = ivf.assignToCluster(
                        vector: vector,
                        slotIndex: handles[i].index,
                        generation: handles[i].generation
                    )
                }
            } else {
                // Add all to staging
                for handle in handles {
                    ivf.addToStaging(handle.index)
                }

                // Auto-train if we have enough vectors
                if autoTrainEnabled && ivf.canTrain {
                    try await train()
                }
            }
        }

        return handles
    }

    // MARK: - Vector Retrieval

    /// Retrieve the vector data for a handle.
    ///
    /// - Parameter handle: Handle to the vector
    /// - Returns: Vector data, or nil if handle is invalid/stale
    /// - Throws: `IndexAccelerationError` if GPU read fails
    public func vector(for handle: VectorHandle) throws -> [Float]? {
        guard handleAllocator.validate(handle) else { return nil }
        return try storage.readVector(at: Int(handle.index))
    }

    /// Retrieve multiple vectors by their handles.
    ///
    /// - Parameter handles: Array of handles
    /// - Returns: Array of optional vectors (nil for invalid handles)
    /// - Throws: `IndexAccelerationError` if GPU read fails
    public func vectors(for handles: [VectorHandle]) throws -> [[Float]?] {
        try handles.map { handle in
            guard handleAllocator.validate(handle) else { return nil }
            return try storage.readVector(at: Int(handle.index))
        }
    }

    // MARK: - Metadata Operations

    /// Get metadata for a handle.
    ///
    /// - Parameter handle: Handle to the vector
    /// - Returns: Metadata if set, nil otherwise. Returns nil for invalid/stale handles.
    public func metadata(for handle: VectorHandle) -> VectorMetadata? {
        guard handleAllocator.validate(handle) else { return nil }
        return metadataStore[handle.index]
    }

    /// Update metadata for a handle.
    ///
    /// - Parameters:
    ///   - metadata: New metadata (nil to remove)
    ///   - handle: Handle to the vector
    /// - Throws: `IndexAccelerationError.invalidInput` if handle is invalid/stale
    public func setMetadata(_ metadata: VectorMetadata?, for handle: VectorHandle) throws {
        guard handleAllocator.validate(handle) else {
            throw IndexAccelerationError.invalidInput(
                message: "Invalid or stale handle: \(handle)"
            )
        }
        metadataStore[handle.index] = metadata
    }

    // MARK: - Remove Operations

    /// Mark a vector as deleted (lazy deletion).
    ///
    /// The vector is not immediately removed from the GPU buffer.
    /// Call `compact()` to reclaim space from deleted vectors.
    ///
    /// - Parameter handle: Handle to the vector to remove
    /// - Throws: `IndexAccelerationError.invalidInput` if handle is invalid/stale
    public func remove(_ handle: VectorHandle) throws {
        guard handleAllocator.validate(handle) else {
            throw IndexAccelerationError.invalidInput(
                message: "Invalid or stale handle: \(handle)"
            )
        }

        handleAllocator.markDeleted(handle)
        deletionMask.markDeleted(Int(handle.index))
        metadataStore.remove(handle.index)

        // Remove from IVF structure
        ivfStructure?.removeVector(slotIndex: handle.index, generation: handle.generation)
    }

    /// Mark multiple vectors as deleted (lazy deletion).
    ///
    /// More efficient than individual removes for bulk operations.
    ///
    /// - Parameter handles: Handles to remove
    /// - Returns: Number of vectors actually removed (excludes invalid handles)
    @discardableResult
    public func remove(_ handles: [VectorHandle]) -> Int {
        var removedCount = 0
        for handle in handles {
            if handleAllocator.validate(handle) {
                handleAllocator.markDeleted(handle)
                deletionMask.markDeleted(Int(handle.index))
                metadataStore.remove(handle.index)
                ivfStructure?.removeVector(slotIndex: handle.index, generation: handle.generation)
                removedCount += 1
            }
        }
        return removedCount
    }

    /// Compact the index to reclaim space from deleted vectors.
    ///
    /// This operation rebuilds the GPU buffer without deleted vectors.
    /// After compaction, old handles are invalid (generation mismatch).
    /// Use the returned mapping to update any external handle references.
    ///
    /// - Returns: Mapping from old handles to new handles
    /// - Note: This is an expensive operation. Call sparingly when fragmentation is high.
    @discardableResult
    public func compact() async throws -> [VectorHandle: VectorHandle] {
        guard deletionMask.deletedCount > 0 else { return [:] }

        // Get keep mask
        let keepMask = deletionMask.keepMask()

        // Compact storage
        _ = try storage.compact(keepMask: keepMask)

        // Compact handle allocator
        let compactionResult = handleAllocator.compact(keepMask: keepMask)

        // Compact metadata store
        let uint32Mapping = compactionResult.indexMapping
        metadataStore.compact(using: uint32Mapping)

        // Compact IVF structure
        ivfStructure?.compact(using: uint32Mapping)

        // Reset deletion mask
        deletionMask.resetAfterCompaction(newCapacity: compactionResult.newSlotCount)

        // Build handle mapping for return
        var handleMapping: [VectorHandle: VectorHandle] = [:]
        for (oldIdx, newIdx) in uint32Mapping {
            let newHandle = compactionResult.newHandles[Int(newIdx)]
            let oldGeneration = newHandle.generation &- 1
            let oldHandle = VectorHandle(index: oldIdx, generation: oldGeneration)
            handleMapping[oldHandle] = newHandle
        }

        return handleMapping
    }

    // MARK: - Search Operations

    /// Search for nearest neighbors.
    ///
    /// - Parameters:
    ///   - query: Query vector
    ///   - k: Number of results to return
    ///   - filter: Optional filter predicate on metadata
    /// - Returns: Array of search results sorted by distance (ascending)
    /// - Throws: `IndexAccelerationError` if dimension mismatch or GPU error
    public func search(
        query: consuming [Float],
        k: Int,
        filter: (@Sendable (VectorHandle, VectorMetadata?) -> Bool)? = nil
    ) async throws -> [SearchResult] {
        guard query.count == configuration.dimension else {
            throw IndexAccelerationError.dimensionMismatch(
                expected: configuration.dimension,
                got: query.count
            )
        }

        guard handleAllocator.occupiedCount > 0 else {
            return []
        }

        // Use IVF search if available and trained
        if let ivf = ivfStructure, ivf.isTrained {
            return try await searchIVF(query: query, k: k, filter: filter)
        }

        // Fall back to flat search
        guard let filter = filter else {
            return try await searchUnfiltered(query: query, k: k)
        }

        return try await searchFiltered(query: query, k: k, filter: filter)
    }

    /// Batch search for multiple queries.
    ///
    /// More efficient than individual searches for multiple queries.
    ///
    /// - Parameters:
    ///   - queries: Array of query vectors
    ///   - k: Number of results per query
    ///   - filter: Optional filter predicate on metadata
    /// - Returns: Array of result arrays (one per query)
    /// - Throws: `IndexAccelerationError` if dimension mismatch or GPU error
    public func search(
        queries: [[Float]],
        k: Int,
        filter: (@Sendable (VectorHandle, VectorMetadata?) -> Bool)? = nil
    ) async throws -> [[SearchResult]] {
        guard !queries.isEmpty else { return [] }

        // Validate dimensions
        for query in queries {
            guard query.count == configuration.dimension else {
                throw IndexAccelerationError.dimensionMismatch(
                    expected: configuration.dimension,
                    got: query.count
                )
            }
        }

        guard handleAllocator.occupiedCount > 0 else {
            return queries.map { _ in [] }
        }

        // Use IVF batch search if available and trained
        if let ivf = ivfStructure, ivf.isTrained, filter == nil {
            return try await searchIVFBatch(queries: queries, k: k)
        }

        // Fall back to sequential search
        var results: [[SearchResult]] = []
        results.reserveCapacity(queries.count)

        for query in queries {
            let result = try await search(query: query, k: k, filter: filter)
            results.append(result)
        }

        return results
    }

    // MARK: - Index Introspection

    /// Get all valid handles in the index.
    ///
    /// - Returns: Array of all valid handles
    public func allHandles() -> [VectorHandle] {
        var handles: [VectorHandle] = []
        handles.reserveCapacity(handleAllocator.occupiedCount)

        for slotIndex in deletionMask {
            if let handle = handleAllocator.handle(for: UInt32(slotIndex)) {
                handles.append(handle)
            }
        }

        return handles
    }

    /// Check if a handle exists in the index.
    ///
    /// - Parameter handle: Handle to check
    /// - Returns: true if handle is valid and exists
    public func contains(_ handle: VectorHandle) -> Bool {
        handleAllocator.validate(handle)
    }

    /// Iterate over all vectors with their handles and metadata.
    ///
    /// - Parameter body: Closure called for each vector
    /// - Throws: Rethrows any error from the closure or GPU read
    public func forEach(_ body: (VectorHandle, [Float], VectorMetadata?) throws -> Void) throws {
        for slotIndex in deletionMask {
            guard let handle = handleAllocator.handle(for: UInt32(slotIndex)) else { continue }
            let vector = try storage.readVector(at: slotIndex)
            let metadata = metadataStore[handle.index]
            try body(handle, vector, metadata)
        }
    }

    /// Get handles matching a metadata predicate.
    ///
    /// - Parameter predicate: Filter function on metadata
    /// - Returns: Array of matching handles
    public func handles(where predicate: (VectorMetadata?) -> Bool) -> [VectorHandle] {
        var result: [VectorHandle] = []

        for slotIndex in deletionMask {
            guard let handle = handleAllocator.handle(for: UInt32(slotIndex)) else { continue }
            let metadata = metadataStore[handle.index]
            if predicate(metadata) {
                result.append(handle)
            }
        }

        return result
    }

    // MARK: - Lifecycle

    /// Release GPU resources.
    ///
    /// After calling this, the index cannot be used.
    public func releaseResources() {
        storage.release()
        fusedL2TopKKernel = nil
        ivfSearchPipeline = nil
        ivfStructure?.reset()
        metadataStore.reset()
        handleAllocator.reset()
        deletionMask.reset()
    }

    // MARK: - Handle Validation

    /// Check if a handle is valid (not stale).
    ///
    /// - Parameter handle: Handle to validate
    /// - Returns: true if handle is valid for this index
    public func isHandleValid(_ handle: VectorHandle) -> Bool {
        handleAllocator.validate(handle)
    }

    /// Get a valid handle for a slot index (if occupied).
    ///
    /// - Parameter slotIndex: Slot index
    /// - Returns: Valid handle, or nil if slot is not occupied
    public func handle(for slotIndex: UInt32) -> VectorHandle? {
        handleAllocator.handle(for: slotIndex)
    }

    // MARK: - Private Implementation - Flat Search

    private func searchUnfiltered(query: [Float], k: Int) async throws -> [SearchResult] {
        guard let kernel = fusedL2TopKKernel,
              let datasetBuffer = storage.buffer else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "search")
        }

        let device = context.device.rawDevice
        let effectiveK = min(k, handleAllocator.occupiedCount)

        // Create query buffer
        let queryBufferSize = configuration.dimension * MemoryLayout<Float>.size
        guard let queryBuffer = device.makeBuffer(
            bytes: query,
            length: queryBufferSize,
            options: .storageModeShared
        ) else {
            throw IndexAccelerationError.bufferError(
                operation: "search",
                reason: "Failed to create query buffer"
            )
        }

        // Request more results than needed to account for deleted vectors
        let fetchK = min(effectiveK + deletionMask.deletedCount, storage.allocatedSlots)

        let parameters = FusedL2TopKParameters(
            numQueries: 1,
            numDataset: storage.allocatedSlots,
            dimension: configuration.dimension,
            k: fetchK
        )

        let gpuResult = try await kernel.execute(
            queries: queryBuffer,
            dataset: datasetBuffer,
            parameters: parameters
        )

        // Convert GPU results to SearchResults, filtering deleted vectors
        var results: [SearchResult] = []
        results.reserveCapacity(effectiveK)

        for (rawIndex, distance) in gpuResult.results(for: 0) {
            // Skip deleted vectors
            if deletionMask.isDeleted(rawIndex) { continue }

            // Get valid handle
            guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

            results.append(SearchResult(handle: handle, distance: distance))

            if results.count >= effectiveK { break }
        }

        return results
    }

    private func searchFiltered(
        query: [Float],
        k: Int,
        filter: @Sendable (VectorHandle, VectorMetadata?) -> Bool
    ) async throws -> [SearchResult] {
        guard let kernel = fusedL2TopKKernel,
              let datasetBuffer = storage.buffer else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "search")
        }

        let device = context.device.rawDevice

        // Create query buffer
        let queryBufferSize = configuration.dimension * MemoryLayout<Float>.size
        guard let queryBuffer = device.makeBuffer(
            bytes: query,
            length: queryBufferSize,
            options: .storageModeShared
        ) else {
            throw IndexAccelerationError.bufferError(
                operation: "search",
                reason: "Failed to create query buffer"
            )
        }

        // Iterative fetch strategy: start with 2x, double until we have enough
        var results: [SearchResult] = []
        var fetchK = min(k * 2, storage.allocatedSlots)
        var processedCount = 0

        while results.count < k && fetchK <= storage.allocatedSlots {
            let parameters = FusedL2TopKParameters(
                numQueries: 1,
                numDataset: storage.allocatedSlots,
                dimension: configuration.dimension,
                k: fetchK
            )

            let gpuResult = try await kernel.execute(
                queries: queryBuffer,
                dataset: datasetBuffer,
                parameters: parameters
            )

            // Process results starting from where we left off
            let gpuResults = gpuResult.results(for: 0)
            for i in processedCount..<gpuResults.count {
                let (rawIndex, distance) = gpuResults[i]

                // Skip deleted vectors
                if deletionMask.isDeleted(rawIndex) { continue }

                // Get valid handle
                guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

                let meta = metadataStore[handle.index]

                if filter(handle, meta) {
                    results.append(SearchResult(handle: handle, distance: distance))
                    if results.count >= k { break }
                }
            }

            if results.count >= k { break }

            // Double fetch size for next iteration
            processedCount = fetchK
            fetchK = min(fetchK * 2, storage.allocatedSlots)

            if processedCount >= storage.allocatedSlots { break }
        }

        return results
    }

    // MARK: - Private Implementation - IVF Search

    private func searchIVF(
        query: [Float],
        k: Int,
        filter: (@Sendable (VectorHandle, VectorMetadata?) -> Bool)?
    ) async throws -> [SearchResult] {
        guard let ivf = ivfStructure,
              let pipeline = ivfSearchPipeline else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "searchIVF")
        }

        // Prepare GPU structure
        let gpuStructure = try ivf.prepareGPUStructure(
            storage: storage,
            device: context.device.rawDevice
        )

        // Execute IVF search
        let ivfResult = try await pipeline.search(
            queries: [query],
            structure: gpuStructure,
            k: filter != nil ? k * 3 : k  // Over-fetch for filtering
        )

        // Convert to SearchResults
        var results: [SearchResult] = []
        results.reserveCapacity(k)

        for (rawIndex, distance) in ivfResult.results(for: 0) {
            // Skip deleted vectors
            if deletionMask.isDeleted(rawIndex) { continue }

            // Get valid handle
            guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

            // Apply filter if provided
            if let filter = filter {
                let meta = metadataStore[handle.index]
                if !filter(handle, meta) { continue }
            }

            results.append(SearchResult(handle: handle, distance: distance))

            if results.count >= k { break }
        }

        return results
    }

    private func searchIVFBatch(queries: [[Float]], k: Int) async throws -> [[SearchResult]] {
        guard let ivf = ivfStructure,
              let pipeline = ivfSearchPipeline else {
            throw IndexAccelerationError.gpuNotInitialized(operation: "searchIVFBatch")
        }

        // Prepare GPU structure
        let gpuStructure = try ivf.prepareGPUStructure(
            storage: storage,
            device: context.device.rawDevice
        )

        // Execute batch IVF search
        let ivfResult = try await pipeline.search(
            queries: queries,
            structure: gpuStructure,
            k: k
        )

        // Convert all results
        var allResults: [[SearchResult]] = []
        allResults.reserveCapacity(queries.count)

        for queryIdx in 0..<queries.count {
            var results: [SearchResult] = []
            results.reserveCapacity(k)

            for (rawIndex, distance) in ivfResult.results(for: queryIdx) {
                // Skip deleted vectors
                if deletionMask.isDeleted(rawIndex) { continue }

                // Get valid handle
                guard let handle = handleAllocator.handle(for: UInt32(rawIndex)) else { continue }

                results.append(SearchResult(handle: handle, distance: distance))

                if results.count >= k { break }
            }

            allResults.append(results)
        }

        return allResults
    }
}

// MARK: - Async Sequence Support

extension AcceleratedVectorIndex {
    /// Async iterator for vectors in the index.
    public struct VectorIterator: AsyncIteratorProtocol {
        let index: AcceleratedVectorIndex
        var slotIndices: [Int]
        var currentPosition: Int = 0

        init(index: AcceleratedVectorIndex, slotIndices: [Int]) {
            self.index = index
            self.slotIndices = slotIndices
        }

        public mutating func next() async throws -> (VectorHandle, [Float], VectorMetadata?)? {
            while currentPosition < slotIndices.count {
                let slotIndex = slotIndices[currentPosition]
                currentPosition += 1

                guard let handle = await index.handle(for: UInt32(slotIndex)) else { continue }
                let vector = try await index.vector(for: handle)
                guard let vector = vector else { continue }
                let metadata = await index.metadata(for: handle)

                return (handle, vector, metadata)
            }
            return nil
        }
    }

    /// Create an async iterator over all vectors.
    ///
    /// - Returns: Async iterator yielding (handle, vector, metadata) tuples
    public func makeAsyncIterator() -> VectorIterator {
        VectorIterator(index: self, slotIndices: deletionMask.activeIndices())
    }
}
