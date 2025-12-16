//
//  Metal4ShaderCompiler.swift
//  VectorAccelerate
//
//  Metal 4 shader compiler wrapper with QoS support and caching
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Compiler Configuration

/// Configuration for Metal 4 shader compiler
public struct Metal4CompilerConfiguration: Sendable {
    /// Quality of service for compilation operations
    public let qualityOfService: QualityOfService

    /// Enable fast math optimizations
    public let fastMathEnabled: Bool

    /// Metal language version (MSL 4.0)
    public let languageVersion: MTLLanguageVersion

    /// Enable compiler optimizations
    public let optimizationLevel: OptimizationLevel

    /// Maximum concurrent compilations
    public let maxConcurrentCompilations: Int

    /// Whether runtime shader compilation is allowed.
    ///
    /// When `false`, only precompiled metallibs can be used. Attempting to call
    /// `makeLibrary(source:)` will throw an error. This ensures predictable,
    /// jank-free performance in production builds.
    ///
    /// - `true`: Development mode, shaders can be compiled from source
    /// - `false`: Production mode, requires precompiled metallib
    public let allowRuntimeCompilation: Bool

    public enum OptimizationLevel: Sendable {
        case none
        case `default`
        case size
        case performance
    }

    public init(
        qualityOfService: QualityOfService = .userInteractive,
        fastMathEnabled: Bool = true,
        languageVersion: MTLLanguageVersion = .version3_1,  // Use highest available
        optimizationLevel: OptimizationLevel = .performance,
        maxConcurrentCompilations: Int = 4,
        allowRuntimeCompilation: Bool = true
    ) {
        self.qualityOfService = qualityOfService
        self.fastMathEnabled = fastMathEnabled
        self.languageVersion = languageVersion
        self.optimizationLevel = optimizationLevel
        self.maxConcurrentCompilations = maxConcurrentCompilations
        self.allowRuntimeCompilation = allowRuntimeCompilation
    }

    public static let `default` = Metal4CompilerConfiguration()

    /// Configuration for batch compilation
    public static let batch = Metal4CompilerConfiguration(
        qualityOfService: .utility,
        fastMathEnabled: true,
        optimizationLevel: .performance,
        maxConcurrentCompilations: 8,
        allowRuntimeCompilation: true
    )

    /// Configuration for real-time compilation
    public static let realTime = Metal4CompilerConfiguration(
        qualityOfService: .userInteractive,
        fastMathEnabled: true,
        optimizationLevel: .default,
        maxConcurrentCompilations: 2,
        allowRuntimeCompilation: true
    )

    /// Production configuration that disables runtime compilation.
    ///
    /// Use this in production builds where all shaders should be precompiled
    /// in a metallib file for predictable performance.
    public static let production = Metal4CompilerConfiguration(
        qualityOfService: .userInitiated,
        fastMathEnabled: true,
        optimizationLevel: .performance,
        maxConcurrentCompilations: 4,
        allowRuntimeCompilation: false
    )
}

// MARK: - Compilation Statistics

/// Statistics for shader compilation
public struct Metal4CompilationStatistics: Sendable {
    public let totalCompilations: Int
    public let successfulCompilations: Int
    public let failedCompilations: Int
    public let totalCompilationTime: TimeInterval
    public let averageCompilationTime: TimeInterval
    public let cachedLibraries: Int
    public let cachedPipelines: Int

    public var successRate: Double {
        totalCompilations > 0
            ? Double(successfulCompilations) / Double(totalCompilations)
            : 0
    }
}

// MARK: - Metal 4 Shader Compiler

/// Metal 4 shader compiler with enhanced caching and QoS support
///
/// Provides:
/// - MTL4Compiler wrapper (simulated for compatibility)
/// - Library and function caching
/// - QoS-aware compilation
/// - Function specialization support
/// - Pipeline harvesting preparation
///
/// Example:
/// ```swift
/// let compiler = try Metal4ShaderCompiler(device: device)
/// let pipeline = try await compiler.compilePipeline(for: .l2Distance(dimension: 384))
/// ```
public actor Metal4ShaderCompiler {
    // MARK: - Properties

    private let device: any MTLDevice
    private let configuration: Metal4CompilerConfiguration

    // Caches
    private var libraryCache: [String: any MTLLibrary] = [:]
    private var functionCache: [String: any MTLFunction] = [:]
    private var pipelineCache: [PipelineCacheKey: any MTLComputePipelineState] = [:]

    // Default library
    private var defaultLibrary: (any MTLLibrary)?

    // Statistics
    private var totalCompilations: Int = 0
    private var successfulCompilations: Int = 0
    private var failedCompilations: Int = 0
    private var totalCompilationTime: TimeInterval = 0

    // Compilation semaphore for controlling concurrency
    private let compilationQueue: OperationQueue

    // MARK: - Initialization

    /// Create a Metal 4 shader compiler
    ///
    /// - Parameters:
    ///   - device: Metal device for compilation
    ///   - configuration: Compiler configuration
    public init(
        device: any MTLDevice,
        configuration: Metal4CompilerConfiguration = .default
    ) throws {
        self.device = device
        self.configuration = configuration

        // Create compilation queue with QoS
        self.compilationQueue = OperationQueue()
        compilationQueue.maxConcurrentOperationCount = configuration.maxConcurrentCompilations
        compilationQueue.qualityOfService = configuration.qualityOfService

        // Note: In actual Metal 4, we would create MTL4Compiler here:
        // let descriptor = MTL4CompilerDescriptor()
        // descriptor.qualityOfService = configuration.qualityOfService
        // self.compiler = try device.makeCompiler(descriptor: descriptor)
    }

    /// Create compiler with default library pre-loaded
    public init(
        device: any MTLDevice,
        defaultLibrary: any MTLLibrary,
        configuration: Metal4CompilerConfiguration = .default
    ) throws {
        self.device = device
        self.configuration = configuration
        self.defaultLibrary = defaultLibrary

        self.compilationQueue = OperationQueue()
        compilationQueue.maxConcurrentOperationCount = configuration.maxConcurrentCompilations
        compilationQueue.qualityOfService = configuration.qualityOfService
    }

    // MARK: - Library Compilation

    /// Compile a Metal library from source
    ///
    /// - Parameters:
    ///   - source: Metal shader source code
    ///   - label: Optional label for debugging
    /// - Returns: Compiled Metal library
    /// - Throws: `VectorError.shaderCompilationFailed` if runtime compilation is disabled
    ///
    /// - Note: In production mode (when `allowRuntimeCompilation` is `false`),
    ///   this method throws an error. Use precompiled metallibs via `loadLibrary(url:)` instead.
    public func makeLibrary(source: String, label: String? = nil) async throws -> any MTLLibrary {
        // Check if runtime compilation is allowed
        guard configuration.allowRuntimeCompilation else {
            throw VectorError.shaderCompilationFailed(
                "Runtime compilation disabled in production mode. Use precompiled metallib via loadLibrary(url:)."
            )
        }

        let cacheKey = "src_\(source.hashValue)"

        // Check cache
        if let cached = libraryCache[cacheKey] {
            return cached
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        totalCompilations += 1

        do {
            // Configure compile options
            let options = MTLCompileOptions()
            options.fastMathEnabled = configuration.fastMathEnabled
            options.languageVersion = configuration.languageVersion

            // Compile library
            // In Metal 4: let library = try await compiler.makeLibrary(source: source, options: options)
            let library = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<any MTLLibrary, any Error>) in
                device.makeLibrary(source: source, options: options) { library, error in
                    if let library = library {
                        continuation.resume(returning: library)
                    } else {
                        continuation.resume(throwing: error ?? VectorError.shaderCompilationFailed(label ?? "unknown"))
                    }
                }
            }

            if let label = label {
                library.label = label
            }

            // Update statistics
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            totalCompilationTime += elapsed
            successfulCompilations += 1

            // Cache library
            libraryCache[cacheKey] = library

            return library
        } catch {
            failedCompilations += 1
            throw VectorError.shaderCompilationFailed(error.localizedDescription)
        }
    }

    /// Compile library from URL
    public func makeLibrary(url: URL) async throws -> any MTLLibrary {
        let source = try String(contentsOf: url, encoding: .utf8)
        return try await makeLibrary(source: source, label: url.lastPathComponent)
    }

    /// Load library from compiled metallib
    public func loadLibrary(url: URL) async throws -> any MTLLibrary {
        let cacheKey = "url_\(url.absoluteString.hashValue)"

        if let cached = libraryCache[cacheKey] {
            return cached
        }

        let library = try device.makeLibrary(URL: url)
        libraryCache[cacheKey] = library

        return library
    }

    // MARK: - Function Creation

    /// Get or create a function from a library
    public func makeFunction(
        name: String,
        library: (any MTLLibrary)? = nil
    ) async throws -> any MTLFunction {
        let lib = library ?? defaultLibrary

        guard let lib = lib else {
            throw VectorError.shaderNotFound(name: "No library available")
        }

        let cacheKey = "\(name)_\(ObjectIdentifier(lib).hashValue)"

        if let cached = functionCache[cacheKey] {
            return cached
        }

        guard let function = lib.makeFunction(name: name) else {
            throw VectorError.shaderNotFound(name: name)
        }

        functionCache[cacheKey] = function
        return function
    }

    /// Create function with specialization constants
    public func makeFunction(
        name: String,
        library: (any MTLLibrary)? = nil,
        constantValues: MTLFunctionConstantValues
    ) async throws -> any MTLFunction {
        let lib = library ?? defaultLibrary

        guard let lib = lib else {
            throw VectorError.shaderNotFound(name: "No library available")
        }

        // Specialized functions aren't cached as constants vary
        return try await lib.makeFunction(name: name, constantValues: constantValues)
    }

    // MARK: - Pipeline Compilation

    /// Compile a compute pipeline for a cache key
    public func compilePipeline(for key: PipelineCacheKey) async throws -> any MTLComputePipelineState {
        // Check cache
        if let cached = pipelineCache[key] {
            return cached
        }

        let functionName = key.functionName
        let function = try await makeFunction(name: functionName)

        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.label = key.cacheString

        // Apply optimization hints based on features
        if key.features.contains(.simdgroupMatrix) {
            // Enable SIMD group optimizations
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        totalCompilations += 1

        do {
            let pipeline = try await device.makeComputePipelineState(function: function)

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            totalCompilationTime += elapsed
            successfulCompilations += 1

            pipelineCache[key] = pipeline
            return pipeline
        } catch {
            failedCompilations += 1
            throw VectorError.shaderCompilationFailed("Pipeline creation failed for \(functionName): \(error.localizedDescription)")
        }
    }

    /// Compile pipeline from function directly
    public func compilePipeline(
        function: any MTLFunction,
        label: String? = nil
    ) async throws -> any MTLComputePipelineState {
        try await device.makeComputePipelineState(function: function)
    }

    /// Get pipeline state by function name (simple API)
    public func getPipelineState(functionName: String) async throws -> any MTLComputePipelineState {
        let key = PipelineCacheKey(operation: functionName)

        if let cached = pipelineCache[key] {
            return cached
        }

        let function = try await makeFunction(name: functionName)
        let pipeline = try await device.makeComputePipelineState(function: function)

        pipelineCache[key] = pipeline
        return pipeline
    }

    // MARK: - Batch Compilation

    /// Compile multiple pipelines concurrently
    public func compileMultiple(keys: [PipelineCacheKey]) async throws -> [PipelineCacheKey: any MTLComputePipelineState] {
        var results: [PipelineCacheKey: any MTLComputePipelineState] = [:]

        // Check which keys need compilation
        let uncached = keys.filter { pipelineCache[$0] == nil }

        // Add already cached
        for key in keys {
            if let cached = pipelineCache[key] {
                results[key] = cached
            }
        }

        // Compile uncached in parallel
        await withTaskGroup(of: (PipelineCacheKey, (any MTLComputePipelineState)?).self) { group in
            for key in uncached {
                group.addTask {
                    do {
                        let pipeline = try await self.compilePipeline(for: key)
                        return (key, pipeline)
                    } catch {
                        return (key, nil)
                    }
                }
            }

            for await (key, pipeline) in group {
                if let pipeline = pipeline {
                    results[key] = pipeline
                }
            }
        }

        return results
    }

    /// Pre-warm cache with common pipelines
    public func warmUp(keys: [PipelineCacheKey]) async {
        _ = try? await compileMultiple(keys: keys)
    }

    // MARK: - Default Library Management

    /// Get the default library, loading it if necessary.
    ///
    /// This method attempts to load the library in order:
    /// 1. Return cached default library if available
    /// 2. Try device's default library (app bundle)
    /// 3. Try KernelContext's shared library (SPM/runtime compiled)
    ///
    /// - Returns: The default Metal library
    /// - Throws: `VectorError.shaderNotFound` if no library is available
    public func getDefaultLibrary() async throws -> any MTLLibrary {
        // Return cached if available
        if let library = defaultLibrary {
            return library
        }

        // Try device's default library first
        if let library = device.makeDefaultLibrary() {
            self.defaultLibrary = library
            return library
        }

        // Fall back to KernelContext's shared library (handles SPM/runtime)
        let library = try KernelContext.getSharedLibrary(for: device)
        self.defaultLibrary = library
        return library
    }

    /// Set the default library for function lookups
    public func setDefaultLibrary(_ library: any MTLLibrary) {
        self.defaultLibrary = library
    }

    /// Load default library from bundle
    public func loadDefaultLibrary() async throws {
        // Try to get default library from device
        guard let library = device.makeDefaultLibrary() else {
            throw VectorError.shaderNotFound(name: "Default library not found")
        }
        self.defaultLibrary = library
    }

    /// Compile and set default library from embedded source
    public func compileDefaultLibrary(source: String) async throws {
        let library = try await makeLibrary(source: source, label: "VectorAccelerate.Default")
        self.defaultLibrary = library
    }

    // MARK: - Cache Management

    /// Clear all caches
    public func clearCache() {
        libraryCache.removeAll()
        functionCache.removeAll()
        pipelineCache.removeAll()
    }

    /// Clear only pipeline cache (keep libraries)
    public func clearPipelineCache() {
        pipelineCache.removeAll()
    }

    /// Get current cache sizes
    public var cacheInfo: (libraries: Int, functions: Int, pipelines: Int) {
        (libraryCache.count, functionCache.count, pipelineCache.count)
    }

    /// Check if a pipeline is cached
    public func isCached(_ key: PipelineCacheKey) -> Bool {
        pipelineCache[key] != nil
    }

    // MARK: - Statistics

    /// Get compilation statistics
    public func getStatistics() -> Metal4CompilationStatistics {
        Metal4CompilationStatistics(
            totalCompilations: totalCompilations,
            successfulCompilations: successfulCompilations,
            failedCompilations: failedCompilations,
            totalCompilationTime: totalCompilationTime,
            averageCompilationTime: successfulCompilations > 0
                ? totalCompilationTime / Double(successfulCompilations)
                : 0,
            cachedLibraries: libraryCache.count,
            cachedPipelines: pipelineCache.count
        )
    }

    /// Reset statistics
    public func resetStatistics() {
        totalCompilations = 0
        successfulCompilations = 0
        failedCompilations = 0
        totalCompilationTime = 0
    }

    // MARK: - Harvesting Support (Metal 4)

    /// Get all currently cached pipelines for harvesting
    public func getCachedPipelines() -> [any MTLComputePipelineState] {
        Array(pipelineCache.values)
    }

    /// Get pipeline cache keys
    public func getCachedKeys() -> [PipelineCacheKey] {
        Array(pipelineCache.keys)
    }
}

// MARK: - Convenience Extension

public extension Metal4ShaderCompiler {
    /// Compile pipeline with error context
    func compilePipelineWithContext(
        for key: PipelineCacheKey,
        context: String
    ) async throws -> any MTLComputePipelineState {
        do {
            return try await compilePipeline(for: key)
        } catch {
            throw VectorError.shaderCompilationFailed("\(context): \(error.localizedDescription)")
        }
    }
}
