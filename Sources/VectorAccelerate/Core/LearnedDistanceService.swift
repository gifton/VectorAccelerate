// VectorAccelerate: Learned Distance Service
//
// Unified distance computation service with automatic fallback.
// Part of Phase 4: Experimental ML Integration.
//
// Provides learned distance computation when available, with automatic
// fallback to standard distance computation when:
// - ML features are disabled in configuration
// - MLTensor is not supported on the device
// - Projection weights are not loaded

import Metal
import VectorCore

/// Unified distance computation service with learned projection support.
///
/// This service provides a high-level API for distance computation that:
/// - Uses learned projections when available and enabled
/// - Falls back to standard distance when ML features are unavailable
/// - Manages kernel lifecycle and weight loading
///
/// ## Configuration
/// ```swift
/// var config = AccelerationConfiguration.default
/// config = AccelerationConfiguration(
///     enableExperimentalML: true  // Enable ML features
/// )
///
/// let service = try await LearnedDistanceService(
///     context: context,  // Metal4Context
///     configuration: config
/// )
///
/// // Load projection weights
/// try await service.loadProjection(from: weightsURL, inputDim: 768, outputDim: 128)
///
/// // Compute distances (automatically uses projection if loaded)
/// let distances = try await service.computeL2(queries: queries, database: database)
/// ```
///
/// ## Fallback Behavior
/// The service automatically falls back to standard L2 distance when:
/// - `enableExperimentalML` is false in configuration
/// - Device doesn't support MLTensor (Metal4Capabilities.supportsMLTensor == false)
/// - No projection weights are loaded
public actor LearnedDistanceService {
    // MARK: - Properties

    private let context: Metal4Context
    private let configuration: AccelerationConfiguration
    private let capabilities: Metal4Capabilities

    // Standard kernel (always available) - Metal4 version
    // Using nonisolated(unsafe) because the kernel is immutable after init
    // and its methods are thread-safe (they create their own command buffers)
    nonisolated private let standardKernel: L2DistanceKernel

    // Learned kernel (only when ML is enabled and supported)
    nonisolated private let learnedKernel: LearnedDistanceKernel?

    // Currently loaded projection
    private var currentProjection: TensorBuffer?
    private var projectionMetadata: ProjectionMetadata?

    /// Metadata about loaded projection
    public struct ProjectionMetadata: Sendable {
        public let name: String
        public let inputDimension: Int
        public let outputDimension: Int
        public let loadedAt: Date
    }

    /// Distance computation mode
    public enum ComputationMode: String, Sendable {
        case standard   // Standard L2 distance
        case learned    // Learned projected distance
    }

    // MARK: - Initialization

    /// Initialize the learned distance service
    ///
    /// - Parameters:
    ///   - context: Metal 4 context for computation
    ///   - configuration: Acceleration configuration
    /// - Throws: VectorError if kernel initialization fails
    public init(
        context: Metal4Context,
        configuration: AccelerationConfiguration = .default
    ) async throws {
        self.context = context
        self.configuration = configuration
        self.capabilities = Metal4Capabilities(device: context.device.rawDevice)

        // Always create standard kernel (Metal4 version)
        self.standardKernel = try await L2DistanceKernel(context: context)

        // Create learned kernel only if ML is enabled and supported
        if configuration.enableExperimentalML && capabilities.supportsMLTensor {
            do {
                self.learnedKernel = try LearnedDistanceKernel(device: context.device.rawDevice)
            } catch {
                // Log but don't fail - fallback will be used
                print("Warning: Failed to initialize learned distance kernel: \(error)")
                print("Falling back to standard distance computation.")
                self.learnedKernel = nil
            }
        } else {
            self.learnedKernel = nil
        }
    }

    // MARK: - Configuration Queries

    /// Whether learned distance computation is available
    public var isLearnedDistanceAvailable: Bool {
        learnedKernel != nil
    }

    /// Whether a projection is currently loaded
    public var hasProjectionLoaded: Bool {
        currentProjection != nil
    }

    /// Current computation mode based on state
    public var currentMode: ComputationMode {
        if learnedKernel != nil && currentProjection != nil {
            return .learned
        }
        return .standard
    }

    /// Get metadata about currently loaded projection
    public var loadedProjectionMetadata: ProjectionMetadata? {
        projectionMetadata
    }

    /// Check why learned distance might be unavailable
    public var fallbackReason: String? {
        if !configuration.enableExperimentalML {
            return "ML features disabled in configuration (enableExperimentalML = false)"
        }
        if !capabilities.supportsMLTensor {
            return "Device does not support MLTensor operations"
        }
        if learnedKernel == nil {
            return "Learned distance kernel failed to initialize"
        }
        if currentProjection == nil {
            return "No projection weights loaded"
        }
        return nil
    }

    // MARK: - Weight Loading

    /// Load projection weights from file
    ///
    /// - Parameters:
    ///   - url: URL to binary weight file (row-major float32)
    ///   - inputDim: Input vector dimension
    ///   - outputDim: Output dimension after projection
    ///   - name: Optional name for the projection
    /// - Throws: VectorError if loading fails or ML features unavailable
    public func loadProjection(
        from url: URL,
        inputDim: Int,
        outputDim: Int,
        name: String = "default"
    ) async throws {
        guard let kernel = learnedKernel else {
            throw VectorError.invalidOperation(
                "Cannot load projection: \(fallbackReason ?? "ML features unavailable")"
            )
        }

        let projection = try await kernel.loadProjection(
            from: url,
            inputDim: inputDim,
            outputDim: outputDim,
            name: name
        )

        currentProjection = projection
        projectionMetadata = ProjectionMetadata(
            name: name,
            inputDimension: inputDim,
            outputDimension: outputDim,
            loadedAt: Date()
        )
    }

    /// Load projection weights from Data
    public func loadProjection(
        from data: Data,
        inputDim: Int,
        outputDim: Int,
        name: String = "default"
    ) async throws {
        guard let kernel = learnedKernel else {
            throw VectorError.invalidOperation(
                "Cannot load projection: \(fallbackReason ?? "ML features unavailable")"
            )
        }

        let projection = try await kernel.loadProjection(
            from: data,
            inputDim: inputDim,
            outputDim: outputDim,
            name: name
        )

        currentProjection = projection
        projectionMetadata = ProjectionMetadata(
            name: name,
            inputDimension: inputDim,
            outputDimension: outputDim,
            loadedAt: Date()
        )
    }

    /// Create projection from Float array
    public func loadProjection(
        from weights: [Float],
        inputDim: Int,
        outputDim: Int,
        name: String = "default"
    ) async throws {
        guard let kernel = learnedKernel else {
            throw VectorError.invalidOperation(
                "Cannot load projection: \(fallbackReason ?? "ML features unavailable")"
            )
        }

        let projection = try await kernel.createProjection(
            from: weights,
            inputDim: inputDim,
            outputDim: outputDim,
            name: name
        )

        currentProjection = projection
        projectionMetadata = ProjectionMetadata(
            name: name,
            inputDimension: inputDim,
            outputDimension: outputDim,
            loadedAt: Date()
        )
    }

    /// Create random projection (for testing/baseline)
    public func loadRandomProjection(
        inputDim: Int,
        outputDim: Int,
        name: String = "random"
    ) async throws {
        guard let kernel = learnedKernel else {
            throw VectorError.invalidOperation(
                "Cannot create projection: \(fallbackReason ?? "ML features unavailable")"
            )
        }

        let projection = try await kernel.createRandomProjection(
            inputDim: inputDim,
            outputDim: outputDim,
            name: name
        )

        currentProjection = projection
        projectionMetadata = ProjectionMetadata(
            name: name,
            inputDimension: inputDim,
            outputDimension: outputDim,
            loadedAt: Date()
        )
    }

    /// Unload current projection to free memory
    public func unloadProjection() async {
        if let metadata = projectionMetadata, let kernel = learnedKernel {
            await kernel.unloadProjection(name: metadata.name)
        }
        currentProjection = nil
        projectionMetadata = nil
    }

    // MARK: - Distance Computation

    /// Compute L2 distances with automatic mode selection
    ///
    /// Uses learned projection if available and loaded, otherwise uses standard L2.
    ///
    /// - Parameters:
    ///   - queries: Query vectors
    ///   - database: Database vectors
    ///   - computeSqrt: Whether to compute square root
    ///   - forceFallback: Force standard distance even if projection available
    /// - Returns: Distance matrix and computation mode used
    public func computeL2(
        queries: [[Float]],
        database: [[Float]],
        computeSqrt: Bool = true,
        forceFallback: Bool = false
    ) async throws -> (distances: [[Float]], mode: ComputationMode) {
        // Use learned if available and not forcing fallback
        if !forceFallback,
           let kernel = learnedKernel,
           let projection = currentProjection {
            let distances = try await kernel.compute(
                queries: queries,
                database: database,
                projection: projection,
                computeSqrt: computeSqrt
            )
            return (distances, .learned)
        }

        // Fallback to standard (Metal4 kernel infers dimension from input)
        let distances = try await standardKernel.compute(
            queries: queries,
            database: database,
            computeSqrt: computeSqrt
        )
        return (distances, .standard)
    }

    /// Compute L2 distances using VectorCore types
    public func computeL2<V: VectorProtocol>(
        queries: [V],
        database: [V],
        computeSqrt: Bool = true,
        forceFallback: Bool = false
    ) async throws -> (distances: [[Float]], mode: ComputationMode) where V.Scalar == Float {
        if !forceFallback,
           let kernel = learnedKernel,
           let projection = currentProjection {
            let distances = try await kernel.compute(
                queries: queries,
                database: database,
                projection: projection,
                computeSqrt: computeSqrt
            )
            return (distances, .learned)
        }

        let distances = try await standardKernel.compute(
            queries: queries,
            database: database,
            computeSqrt: computeSqrt
        )
        return (distances, .standard)
    }

    /// Compute distances using compile-time dimensioned vectors
    public func computeL2<D: StaticDimension>(
        queries: [Vector<D>],
        database: [Vector<D>],
        computeSqrt: Bool = true,
        forceFallback: Bool = false
    ) async throws -> (distances: [[Float]], mode: ComputationMode) {
        if !forceFallback,
           let kernel = learnedKernel,
           let projection = currentProjection {
            let distances = try await kernel.compute(
                queries: queries,
                database: database,
                projection: projection,
                computeSqrt: computeSqrt
            )
            return (distances, .learned)
        }

        let distances = try await standardKernel.compute(
            queries: queries,
            database: database,
            computeSqrt: computeSqrt
        )
        return (distances, .standard)
    }

    // MARK: - Batch Projection

    /// Pre-project vectors for subsequent fast distance computation
    ///
    /// Useful for projecting a database once then computing many distances.
    ///
    /// - Parameters:
    ///   - vectors: Vectors to project
    ///   - normalize: Whether to L2 normalize output
    /// - Returns: Projected vectors
    public func projectBatch(
        vectors: [[Float]],
        normalize: Bool = false
    ) async throws -> [[Float]] {
        guard let kernel = learnedKernel,
              let projection = currentProjection else {
            throw VectorError.invalidOperation(
                "Cannot project: \(fallbackReason ?? "No projection loaded")"
            )
        }

        return try await kernel.projectBatch(
            vectors: vectors,
            projection: projection,
            normalize: normalize
        )
    }

    // MARK: - Statistics

    /// Get statistics about the service
    public struct ServiceStatistics: Sendable {
        public let mode: ComputationMode
        public let mlFeaturesEnabled: Bool
        public let mlTensorSupported: Bool
        public let projectionLoaded: Bool
        public let projectionMetadata: ProjectionMetadata?
        public let fallbackReason: String?
    }

    /// Get current service statistics
    public func getStatistics() -> ServiceStatistics {
        ServiceStatistics(
            mode: currentMode,
            mlFeaturesEnabled: configuration.enableExperimentalML,
            mlTensorSupported: capabilities.supportsMLTensor,
            projectionLoaded: currentProjection != nil,
            projectionMetadata: projectionMetadata,
            fallbackReason: fallbackReason
        )
    }
}

// MARK: - Factory Methods

extension LearnedDistanceService {
    /// Create service with ML features enabled
    public static func withMLEnabled(context: Metal4Context) async throws -> LearnedDistanceService {
        let config = AccelerationConfiguration(enableExperimentalML: true)
        return try await LearnedDistanceService(context: context, configuration: config)
    }

    /// Create service with standard distance only (no ML)
    public static func standardOnly(context: Metal4Context) async throws -> LearnedDistanceService {
        let config = AccelerationConfiguration(enableExperimentalML: false)
        return try await LearnedDistanceService(context: context, configuration: config)
    }
}
