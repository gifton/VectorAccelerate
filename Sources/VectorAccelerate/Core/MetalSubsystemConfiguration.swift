//
//  MetalSubsystemConfiguration.swift
//  VectorAccelerate
//
//  Configuration for Metal 4 lifecycle management
//

import Foundation

/// Configuration for MetalSubsystem behavior.
///
/// `MetalSubsystemConfiguration` controls how Metal initializes, what pipelines
/// are warmed, and how the warmup process respects user activity and system state.
///
/// ## Production vs Development
///
/// In production builds, runtime shader compilation should be disabled to ensure
/// predictable performance. Use `production(metallibURL:archiveURL:)` to create
/// a production configuration that requires precompiled shaders.
///
/// ## Example Usage
///
/// ```swift
/// // Development (default)
/// let devConfig = MetalSubsystemConfiguration.default
///
/// // Production with precompiled metallib
/// let prodConfig = MetalSubsystemConfiguration.production(
///     metallibURL: Bundle.main.url(forResource: "default", withExtension: "metallib")!,
///     archiveURL: cacheDir.appending(path: "pipelines.metalarchive")
/// )
///
/// // Custom configuration
/// let customConfig = MetalSubsystemConfiguration(
///     allowRuntimeCompilation: false,
///     metallibURL: metallibURL,
///     pipelineRegistry: .journalingApp,
///     backgroundQoS: .utility
/// )
/// ```
public struct MetalSubsystemConfiguration: Sendable {

    // MARK: - Compilation Mode

    /// Allow runtime shader compilation.
    ///
    /// - `true`: Development mode, shaders can be compiled at runtime from source
    /// - `false`: Production mode, requires precompiled metallib
    ///
    /// When `false`, any attempt to compile shaders from source will throw an error.
    /// This ensures predictable, jank-free performance in production.
    public let allowRuntimeCompilation: Bool

    // MARK: - Resource URLs

    /// URL to precompiled metallib.
    ///
    /// Required when `allowRuntimeCompilation` is `false`.
    /// In app bundles, this is typically `Bundle.main.url(forResource: "default", withExtension: "metallib")`.
    public let metallibURL: URL?

    /// URL to binary archive for PSO caching.
    ///
    /// If `nil`, uses default cache directory location:
    /// `~/Library/Caches/VectorAccelerate/pipelines.metalarchive`
    ///
    /// Binary archives provide near-instant PSO creation after first compilation.
    public let binaryArchiveURL: URL?

    // MARK: - Pipeline Configuration

    /// Pipeline registry defining tiers for warmup prioritization.
    ///
    /// The registry categorizes pipelines into critical, occasional, and rare tiers.
    /// Critical pipelines are warmed in Phase B, occasional in Phase C.
    public let pipelineRegistry: PipelineRegistry

    // MARK: - QoS Configuration

    /// Quality of service for background initialization.
    ///
    /// Controls the priority of Phase B (Metal device creation, context setup).
    /// Default is `.utility` to avoid competing with UI work.
    public let backgroundQoS: QualityOfService

    /// Quality of service for critical pipeline warmup.
    ///
    /// Controls the priority of critical pipeline compilation.
    /// Default is `.userInitiated` since it affects first user operation.
    public let criticalQoS: QualityOfService

    // MARK: - Warmup Configuration

    /// Idle timeout before resuming opportunistic warmup (seconds).
    ///
    /// After user interaction, warmup pauses for this duration before resuming.
    /// This prevents warmup from causing jank during active use.
    public let warmupIdleTimeout: TimeInterval

    /// Whether to pause warmup during thermal throttling.
    ///
    /// When `true`, opportunistic warmup will pause if the device reports
    /// serious or critical thermal state, resuming when cooled.
    public let respectThermalState: Bool

    // MARK: - Initialization

    /// Create a configuration with all options.
    ///
    /// - Parameters:
    ///   - allowRuntimeCompilation: Whether to allow shader compilation from source
    ///   - metallibURL: URL to precompiled metallib (required if runtime compilation disabled)
    ///   - binaryArchiveURL: Optional URL for PSO binary archive
    ///   - pipelineRegistry: Pipeline tier definitions
    ///   - backgroundQoS: QoS for background initialization
    ///   - criticalQoS: QoS for critical pipeline warmup
    ///   - warmupIdleTimeout: Seconds to wait after user activity before resuming warmup
    ///   - respectThermalState: Whether to pause warmup on thermal throttling
    public init(
        allowRuntimeCompilation: Bool = true,
        metallibURL: URL? = nil,
        binaryArchiveURL: URL? = nil,
        pipelineRegistry: PipelineRegistry = .default,
        backgroundQoS: QualityOfService = .utility,
        criticalQoS: QualityOfService = .userInitiated,
        warmupIdleTimeout: TimeInterval = 2.0,
        respectThermalState: Bool = true
    ) {
        self.allowRuntimeCompilation = allowRuntimeCompilation
        self.metallibURL = metallibURL
        self.binaryArchiveURL = binaryArchiveURL
        self.pipelineRegistry = pipelineRegistry
        self.backgroundQoS = backgroundQoS
        self.criticalQoS = criticalQoS
        self.warmupIdleTimeout = warmupIdleTimeout
        self.respectThermalState = respectThermalState
    }

    // MARK: - Presets

    /// Default configuration (development mode).
    ///
    /// - Runtime compilation: enabled
    /// - Pipeline registry: default (all common keys as critical)
    /// - Warmup idle timeout: 2 seconds
    /// - Thermal respect: enabled
    public static let `default` = MetalSubsystemConfiguration()

    /// Development configuration with runtime compilation enabled.
    ///
    /// Optimized for development workflows where shaders may be modified
    /// and recompiled frequently.
    public static let development = MetalSubsystemConfiguration(
        allowRuntimeCompilation: true,
        pipelineRegistry: .default,
        backgroundQoS: .utility,
        criticalQoS: .userInitiated,
        warmupIdleTimeout: 1.0,  // Faster warmup in dev
        respectThermalState: false  // Don't pause warmup in dev
    )

    /// Production configuration requiring precompiled metallib.
    ///
    /// - Parameter metallibURL: URL to precompiled metallib (required)
    /// - Parameter archiveURL: Optional URL for binary archive caching
    /// - Returns: Production-ready configuration
    ///
    /// This configuration:
    /// - Disables runtime shader compilation
    /// - Uses journaling app pipeline registry
    /// - Respects thermal state
    public static func production(
        metallibURL: URL,
        archiveURL: URL? = nil
    ) -> MetalSubsystemConfiguration {
        MetalSubsystemConfiguration(
            allowRuntimeCompilation: false,
            metallibURL: metallibURL,
            binaryArchiveURL: archiveURL,
            pipelineRegistry: .journalingApp,
            backgroundQoS: .utility,
            criticalQoS: .userInitiated,
            warmupIdleTimeout: 2.0,
            respectThermalState: true
        )
    }

    /// Configuration optimized for journaling app.
    ///
    /// Alias for `production()` with journaling-specific defaults.
    public static func journalingApp(
        metallibURL: URL,
        archiveURL: URL? = nil
    ) -> MetalSubsystemConfiguration {
        .production(metallibURL: metallibURL, archiveURL: archiveURL)
    }

    /// Test configuration for unit tests.
    ///
    /// - Allows runtime compilation
    /// - Uses minimal pipeline registry
    /// - Fast warmup timeout
    public static let testing = MetalSubsystemConfiguration(
        allowRuntimeCompilation: true,
        pipelineRegistry: .minimal,
        backgroundQoS: .userInitiated,  // Faster for tests
        criticalQoS: .userInitiated,
        warmupIdleTimeout: 0.1,
        respectThermalState: false
    )
}

// MARK: - QoS Extension

extension QualityOfService {
    /// Convert to Swift Concurrency TaskPriority.
    ///
    /// Maps Dispatch QoS levels to their closest TaskPriority equivalent.
    public var taskPriority: TaskPriority {
        switch self {
        case .userInteractive:
            return .high
        case .userInitiated:
            return .high
        case .utility:
            return .medium
        case .background:
            return .low
        case .default:
            return .medium
        @unknown default:
            return .medium
        }
    }
}

// MARK: - Validation

extension MetalSubsystemConfiguration {
    /// Validate configuration for production use.
    ///
    /// - Throws: `MetalSubsystemConfigurationError` if invalid
    public func validateForProduction() throws {
        guard !allowRuntimeCompilation else {
            return  // Development config, no validation needed
        }

        guard metallibURL != nil else {
            throw MetalSubsystemConfigurationError.metallibRequiredInProduction
        }
    }

    /// Check if this configuration is valid for the current environment.
    public var isValid: Bool {
        if !allowRuntimeCompilation && metallibURL == nil {
            return false
        }
        return true
    }
}

/// Errors from configuration validation.
public enum MetalSubsystemConfigurationError: Error, LocalizedError, Sendable {
    case metallibRequiredInProduction

    public var errorDescription: String? {
        switch self {
        case .metallibRequiredInProduction:
            return "Precompiled metallib URL required when runtime compilation is disabled"
        }
    }
}
