// VectorAccelerate: Shader Library Management
//
// Centralized shader compilation and caching with variant support
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Common shader variants for pre-compilation
public struct ShaderVariant: Hashable, Sendable {
    public let dimension: Int
    public let precision: Precision
    public let batchSize: Int?
    
    public enum Precision: String, Sendable {
        case full = "float"
        case half = "half"
        case mixed = "mixed"
    }
    
    /// Common variants for ML workloads
    public static let commonVariants: [ShaderVariant] = [
        ShaderVariant(dimension: 128, precision: .full, batchSize: nil),
        ShaderVariant(dimension: 256, precision: .full, batchSize: nil),
        ShaderVariant(dimension: 384, precision: .full, batchSize: nil),
        ShaderVariant(dimension: 512, precision: .full, batchSize: nil),
        ShaderVariant(dimension: 768, precision: .full, batchSize: nil),  // BERT
        ShaderVariant(dimension: 1024, precision: .full, batchSize: nil),
        ShaderVariant(dimension: 1536, precision: .full, batchSize: nil), // GPT
        // Batch variants
        ShaderVariant(dimension: 512, precision: .full, batchSize: 32),
        ShaderVariant(dimension: 512, precision: .full, batchSize: 64),
        ShaderVariant(dimension: 512, precision: .full, batchSize: 128),
    ]
}

/// Enhanced shader library with variant support and pre-compilation
public actor ShaderLibrary {
    private let device: MetalDevice
    private let shaderManager: ShaderManager
    
    // Multi-level caching
    private var functionCache: [String: any MTLFunction] = [:]
    private var pipelineCache: [String: any MTLComputePipelineState] = [:]
    private var variantCache: [ShaderVariant: [String: any MTLComputePipelineState]] = [:]
    
    // Shader sources
    private var libraries: [String: any MTLLibrary] = [:]
    
    // Performance tracking
    private var compilationTime: TimeInterval = 0
    private var cacheHits: Int = 0
    private var cacheMisses: Int = 0
    
    public init(device: MetalDevice, shaderManager: ShaderManager) async throws {
        self.device = device
        self.shaderManager = shaderManager
        
        // Pre-compile common variants
        await precompileCommonVariants()
    }
    
    // MARK: - Shader Compilation
    
    /// Get or compile a shader function
    public func getFunction(
        name: String,
        library: String? = nil
    ) async throws -> any MTLFunction {
        let cacheKey = "\(library ?? "default").\(name)"
        
        // Check cache
        if let cached = functionCache[cacheKey] {
            cacheHits += 1
            return cached
        }
        
        cacheMisses += 1
        
        // Get the function from shader manager
        let function = try await shaderManager.getFunction(name: name)
        functionCache[cacheKey] = function
        
        return function
    }
    
    /// Get or create pipeline state
    public func getPipeline(
        functionName: String,
        variant: ShaderVariant? = nil
    ) async throws -> any MTLComputePipelineState {
        // Check variant cache first
        if let variant = variant,
           let variantPipelines = variantCache[variant],
           let cached = variantPipelines[functionName] {
            cacheHits += 1
            return cached
        }
        
        // Check regular cache
        let cacheKey = functionName
        if let cached = pipelineCache[cacheKey] {
            cacheHits += 1
            return cached
        }
        
        cacheMisses += 1
        
        // Create pipeline
        let pipeline = try await shaderManager.getPipelineState(functionName: functionName)
        
        // Cache it
        pipelineCache[cacheKey] = pipeline
        
        // Also cache in variant if provided
        if let variant = variant {
            if variantCache[variant] == nil {
                variantCache[variant] = [:]
            }
            variantCache[variant]?[functionName] = pipeline
        }
        
        return pipeline
    }
    
    /// Compile shader source with constants
    public func compileWithConstants(
        source: String,
        constants: [String: Any],
        label: String
    ) async throws -> any MTLLibrary {
        // Check if already compiled
        if let cached = libraries[label] {
            return cached
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Build constant values
        let constantValues = MTLFunctionConstantValues()
        for (key, value) in constants {
            var val = value
            switch value {
            case let intVal as Int:
                withUnsafeBytes(of: &val) { bytes in
                    constantValues.setConstantValue(bytes.baseAddress!, type: .int, withName: key)
                }
            case let floatVal as Float:
                withUnsafeBytes(of: &val) { bytes in
                    constantValues.setConstantValue(bytes.baseAddress!, type: .float, withName: key)
                }
            case let boolVal as Bool:
                withUnsafeBytes(of: &val) { bytes in
                    constantValues.setConstantValue(bytes.baseAddress!, type: .bool, withName: key)
                }
            default:
                continue
            }
        }
        
        // Compile with constants
        let library = try await shaderManager.compileLibrary(source: source, label: label)
        
        compilationTime += CFAbsoluteTimeGetCurrent() - startTime
        libraries[label] = library
        
        return library
    }
    
    // MARK: - Pre-compilation
    
    private func precompileCommonVariants() async {
        // Distance functions to precompile
        let distanceFunctions = [
            "euclideanDistance",
            "cosineDistance",
            "dotProduct",
            "manhattanDistance",
            "chebyshevDistance"
        ]
        
        // Batch operations to precompile  
        let batchFunctions = [
            "batchEuclideanDistance",
            "batchCosineDistance",
            "batchDotProduct"
        ]
        
        // Pre-compile distance functions
        for function in distanceFunctions {
            _ = try? await getPipeline(functionName: function)
        }
        
        // Pre-compile batch functions with common dimensions
        for variant in ShaderVariant.commonVariants.prefix(5) {
            for function in batchFunctions {
                _ = try? await getPipeline(functionName: function, variant: variant)
            }
        }
    }
    
    // MARK: - Specialized Pipelines
    
    /// Get optimal pipeline for vector dimension
    public func getOptimalPipeline(
        operation: String,
        dimension: Int,
        batchSize: Int? = nil
    ) async throws -> any MTLComputePipelineState {
        // Find best matching variant
        let precision: ShaderVariant.Precision = .full
        let variant = ShaderVariant(
            dimension: dimension,
            precision: precision,
            batchSize: batchSize
        )
        
        // Check if we have a pre-compiled variant
        if let variantPipelines = variantCache[variant],
           let pipeline = variantPipelines[operation] {
            cacheHits += 1
            return pipeline
        }
        
        // Fall back to generic pipeline
        return try await getPipeline(functionName: operation, variant: variant)
    }
    
    // MARK: - Statistics
    
    public struct Statistics: Sendable {
        public let cacheHitRate: Double
        public let totalCompilationTime: TimeInterval
        public let cachedFunctions: Int
        public let cachedPipelines: Int
        public let cachedVariants: Int
    }
    
    public func getStatistics() -> Statistics {
        let total = Double(cacheHits + cacheMisses)
        let hitRate = total > 0 ? Double(cacheHits) / total : 0
        
        return Statistics(
            cacheHitRate: hitRate,
            totalCompilationTime: compilationTime,
            cachedFunctions: functionCache.count,
            cachedPipelines: pipelineCache.count,
            cachedVariants: variantCache.count
        )
    }
    
    /// Clear all caches
    public func clearCaches() {
        functionCache.removeAll()
        pipelineCache.removeAll()
        variantCache.removeAll()
        libraries.removeAll()
        cacheHits = 0
        cacheMisses = 0
    }
}

// MARK: - Shader Constants

/// Constants for shader compilation
public struct ShaderConstants {
    /// Thread group sizes optimized for different architectures
    public static let threadGroupSizes: [String: Int] = [
        "apple_silicon": 32,
        "intel": 64,
        "amd": 64,
        "default": 32
    ]
    
    /// Maximum shared memory per thread group
    public static let maxSharedMemory = 32768 // 32KB
    
    /// Optimal vector widths for SIMD
    public static let simdWidths: [String: Int] = [
        "float4": 4,
        "float8": 8,
        "float16": 16
    ]
}