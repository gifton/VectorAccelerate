// KernelContext.swift
// Internal utility for Metal4ShaderCompiler shader loading fallback
//
// NOTE: After the Metal4-only migration, KernelContext is primarily used internally
// by Metal4ShaderCompiler for loading Metal libraries in SPM environments.
// Prefer using Metal4Context for all new kernel development.

@preconcurrency import Metal
import Foundation
import VectorCore

/// Internal utility for Metal library loading with SPM fallback support.
///
/// This class provides synchronous Metal library loading used by `Metal4ShaderCompiler`
/// as a fallback when the default library is unavailable (common in SPM packages).
///
/// **Note**: After the Metal4-only migration, prefer using `Metal4Context` and
/// `Metal4ShaderCompiler` for all kernel development. This class is retained for
/// its shader compilation fallback capabilities.
///
/// ## Primary Use Case
/// ```swift
/// // Used internally by Metal4ShaderCompiler
/// let library = try KernelContext.getSharedLibrary(for: device)
/// ```
public final class KernelContext: @unchecked Sendable {
    public let device: any MTLDevice
    public let commandQueue: any MTLCommandQueue
    public private(set) var library: (any MTLLibrary)?

    // Cache for shared instances
    // Using nonisolated(unsafe) as this is protected by explicit lock synchronization
    nonisolated(unsafe) private static var sharedInstances: [ObjectIdentifier: KernelContext] = [:]
    nonisolated(unsafe) private static var sharedLibrary: (any MTLLibrary)?
    private static let lock = NSLock()

    /// Create a kernel context with the given device
    public init(device: any MTLDevice) throws {
        self.device = device

        // Create command queue synchronously
        guard let queue = device.makeCommandQueue() else {
            throw VectorError.commandQueueCreationFailed()
        }
        self.commandQueue = queue
    }

    /// Get or create a shared context for the device
    public static func shared(for device: any MTLDevice) throws -> KernelContext {
        let deviceId = ObjectIdentifier(device)

        lock.lock()
        defer { lock.unlock() }

        if let existing = sharedInstances[deviceId] {
            return existing
        }

        let context = try KernelContext(device: device)

        // Load or reuse shared library
        if let library = sharedLibrary {
            context.library = library
        } else {
            context.library = Self.loadMetalLibrary(device: device)
            sharedLibrary = context.library
        }

        sharedInstances[deviceId] = context
        return context
    }

    /// Load Metal library with fallback support for different environments
    /// Tries multiple approaches: default library, package bundle, runtime compilation
    public static func loadMetalLibrary(device: any MTLDevice) -> (any MTLLibrary)? {
        // 1. Try default library (works in app bundles)
        if let library = device.makeDefaultLibrary() {
            return library
        }

        // 2. Try loading from package bundle (SPM resources)
        #if SWIFT_PACKAGE
        if let libraryURL = Bundle.module.url(forResource: "default", withExtension: "metallib"),
           let library = try? device.makeLibrary(URL: libraryURL) {
            return library
        }

        // 3. Try loading Metal source files from bundle and compile at runtime
        if let library = compileMetalSourcesFromBundle(device: device) {
            return library
        }
        #endif

        return nil
    }

    /// Compile Metal shader sources from bundle resources
    ///
    /// Note: This runtime compilation supports core shaders only. For full Metal4 kernel
    /// support (matrix ops, quantization, etc.), use a pre-compiled metallib or Xcode project.
    private static func compileMetalSourcesFromBundle(device: any MTLDevice) -> (any MTLLibrary)? {
        #if SWIFT_PACKAGE
        // Load core shader files that can be safely combined without symbol conflicts
        // Additional shaders (matrix ops, quantization) require pre-compiled metallib
        // due to complex dependencies and potential symbol collisions
        let shaderFiles = [
            // Core distance kernels
            "L2Distance",
            "DotProduct",
            "CosineSimilarity",
            "DistanceShaders",
            "HammingDistance",
            "MinkowskiDistance",
            // Utility operations
            "BasicOperations",
            "L2Normalization",
            "DataTransformations",
            "StatisticsShaders",
            // Selection and reduction
            "AdvancedTopK",
            "SearchAndRetrieval",
            // IVF index operations
            "IVFCandidateBuilder",
            // Quantization (all types)
            "QuantizationShaders",
            "ProductQuantization",
            // Matrix operations
            "OptimizedMatrixOps",
            // ML integration
            "LearnedDistance",
            "NeuralQuantization",
            "AttentionSimilarity"
            // NOTE: All shaders now use VA_* prefixed guards to avoid conflicts
            // NOTE: Histogram kernels are in StatisticsShaders.metal
        ]

        var combinedSource = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        #include <metal_atomic>
        using namespace metal;

        // Common constants
        #ifndef EPSILON
        #define EPSILON 1e-7f
        #endif

        // Atomic types for ProductQuantization
        #ifndef VA_ATOMIC_TYPES_DEFINED
        #define VA_ATOMIC_TYPES_DEFINED
        typedef atomic<float> atomic_float;
        typedef atomic<uint> atomic_uint;
        #endif

        """

        for fileName in shaderFiles {
            if let url = Bundle.module.url(forResource: fileName, withExtension: "metal"),
               let source = try? String(contentsOf: url, encoding: .utf8) {
                // Strip duplicate includes and namespace declarations
                var cleanedSource = source
                    .replacingOccurrences(of: "#include <metal_stdlib>", with: "")
                    .replacingOccurrences(of: "#include <metal_simdgroup>", with: "")
                    .replacingOccurrences(of: "#include <metal_math>", with: "")
                    .replacingOccurrences(of: "#include <metal_atomic>", with: "")
                    .replacingOccurrences(of: "#include \"Metal4Common.h\"", with: "")
                    .replacingOccurrences(of: "using namespace metal;", with: "")

                // Remove conflicting constant definitions
                cleanedSource = cleanedSource.replacingOccurrences(
                    of: "constant float EPSILON = 1e-7f;",
                    with: "// EPSILON defined globally"
                )
                cleanedSource = cleanedSource.replacingOccurrences(
                    of: "constant float EPSILON = 1e-8f;",
                    with: "// EPSILON defined globally"
                )
                cleanedSource = cleanedSource.replacingOccurrences(
                    of: "constexpr constant float EPSILON = 1e-7f;",
                    with: "// EPSILON defined globally"
                )
                // Remove Metal4Common.h aliases
                cleanedSource = cleanedSource.replacingOccurrences(
                    of: "constant float EPSILON = VA_EPSILON;",
                    with: "// EPSILON defined globally"
                )
                // Replace Metal4Common.h constants used in LearnedDistance.metal
                cleanedSource = cleanedSource.replacingOccurrences(
                    of: "VA_EPSILON",
                    with: "EPSILON"
                )
                cleanedSource = cleanedSource.replacingOccurrences(
                    of: "VA_INVALID_INDEX",
                    with: "0xFFFFFFFF"
                )

                combinedSource += "\n// === \(fileName).metal ===\n"
                combinedSource += cleanedSource
                combinedSource += "\n"
            }
        }

        // Compile the combined source
        do {
            let options = MTLCompileOptions()
            options.mathMode = .fast
            let library = try device.makeLibrary(source: combinedSource, options: options)
            return library
        } catch {
            // Compilation failed - log error details
            print("Warning: Failed to compile Metal shaders from bundle: \(error)")
            return nil
        }
        #else
        return nil
        #endif
    }

    /// Get the shared Metal library (loads if needed)
    public static func getSharedLibrary(for device: any MTLDevice) throws -> any MTLLibrary {
        lock.lock()
        defer { lock.unlock() }

        if let library = sharedLibrary {
            return library
        }

        guard let library = loadMetalLibrary(device: device) else {
            throw VectorError.libraryCreationFailed()
        }

        sharedLibrary = library
        return library
    }

    /// Create a buffer from data
    public func createBuffer<T>(from data: [T], options: MTLResourceOptions) -> (any MTLBuffer)? {
        let size = data.count * MemoryLayout<T>.stride
        return data.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: size, options: options)
        }
    }

    /// Create a buffer with guaranteed alignment for SIMD operations
    /// - Parameters:
    ///   - data: Input data array
    ///   - options: Metal resource options
    ///   - alignment: Required alignment in bytes (default 16 for SIMD)
    /// - Returns: Aligned buffer or nil if creation fails
    public func createAlignedBuffer<T>(
        from data: [T],
        options: MTLResourceOptions,
        alignment: Int = 16
    ) -> (any MTLBuffer)? {
        let size = data.count * MemoryLayout<T>.stride
        // Ensure size is aligned
        let alignedSize = (size + alignment - 1) & ~(alignment - 1)

        return data.withUnsafeBytes { bytes in
            // Metal buffers are already 256-byte aligned by default
            // But we add explicit alignment for documentation
            device.makeBuffer(bytes: bytes.baseAddress!, length: alignedSize, options: options)
        }
    }

    /// Validate buffer alignment for SIMD operations
    /// - Parameters:
    ///   - buffer: Buffer to validate
    ///   - requiredAlignment: Required alignment in bytes
    /// - Returns: True if buffer is properly aligned
    public static func isBufferAligned(_ buffer: any MTLBuffer, alignment: Int = 16) -> Bool {
        // Check if buffer address is aligned
        let address = buffer.contents()
        let addressInt = Int(bitPattern: address)
        return addressInt % alignment == 0
    }

    /// Validate that buffer size is suitable for SIMD operations
    /// - Parameters:
    ///   - buffer: Buffer to validate
    ///   - elementSize: Size of each element in bytes
    ///   - simdWidth: SIMD width (e.g., 4 for float4)
    /// - Returns: True if buffer can be processed with SIMD
    public static func isBufferSIMDCompatible(
        _ buffer: any MTLBuffer,
        elementSize: Int,
        simdWidth: Int = 4
    ) -> Bool {
        let elementCount = buffer.length / elementSize
        // Check if we have at least one full SIMD vector
        return elementCount >= simdWidth && isBufferAligned(buffer, alignment: elementSize * simdWidth)
    }

    // MARK: - Zero-Copy VectorProtocol Buffer Creation

    /// Create an aligned buffer directly from VectorProtocol types without intermediate allocations.
    ///
    /// This method avoids the `.toArray()` anti-pattern by using `withUnsafeBufferPointer`
    /// to copy vector data directly into the Metal buffer. This eliminates:
    /// - Intermediate array allocations from `.toArray()`
    /// - Additional copies from `.flatMap { $0 }`
    ///
    /// - Parameters:
    ///   - vectors: Array of VectorProtocol-conforming vectors to flatten into buffer
    ///   - options: Metal resource options (default: .storageModeShared for unified memory)
    ///   - alignment: Required alignment in bytes (default 16 for float4 SIMD)
    /// - Returns: Aligned buffer containing flattened vector data, or nil if creation fails
    ///
    /// - Complexity: O(n * d) where n is number of vectors and d is dimension
    /// - Note: All vectors must have the same dimension. First vector's count is used as dimension.
    @inlinable
    public func createAlignedBufferFromVectors<V: VectorProtocol>(
        _ vectors: [V],
        options: MTLResourceOptions = .storageModeShared,
        alignment: Int = 16
    ) -> (any MTLBuffer)? where V.Scalar == Float {
        guard !vectors.isEmpty else { return nil }

        let dimension = vectors[0].count
        let totalCount = vectors.count * dimension
        let byteSize = totalCount * MemoryLayout<Float>.stride
        let alignedSize = (byteSize + alignment - 1) & ~(alignment - 1)

        // Create buffer with aligned size
        guard let buffer = device.makeBuffer(length: alignedSize, options: options) else {
            return nil
        }

        // Get pointer to buffer contents
        let destination = buffer.contents().bindMemory(to: Float.self, capacity: totalCount)

        // Copy each vector directly using withUnsafeBufferPointer (zero intermediate allocation)
        for (i, vector) in vectors.enumerated() {
            let offset = i * dimension
            vector.withUnsafeBufferPointer { srcPtr in
                guard let srcBase = srcPtr.baseAddress else { return }
                let dst = destination.advanced(by: offset)
                // Direct memory copy from vector storage to Metal buffer
                dst.update(from: srcBase, count: min(srcPtr.count, dimension))
            }
        }

        return buffer
    }

    /// Create an aligned buffer from a single VectorProtocol without intermediate allocation.
    ///
    /// - Parameters:
    ///   - vector: Single VectorProtocol-conforming vector
    ///   - options: Metal resource options
    ///   - alignment: Required alignment in bytes
    /// - Returns: Aligned buffer containing vector data, or nil if creation fails
    @inlinable
    public func createAlignedBufferFromVector<V: VectorProtocol>(
        _ vector: V,
        options: MTLResourceOptions = .storageModeShared,
        alignment: Int = 16
    ) -> (any MTLBuffer)? where V.Scalar == Float {
        let count = vector.count
        let byteSize = count * MemoryLayout<Float>.stride
        let alignedSize = (byteSize + alignment - 1) & ~(alignment - 1)

        // Use withUnsafeBufferPointer to create buffer directly from vector storage
        return vector.withUnsafeBufferPointer { srcPtr in
            guard let srcBase = srcPtr.baseAddress else { return nil }
            return device.makeBuffer(bytes: srcBase, length: alignedSize, options: options)
        }
    }

    /// Create aligned buffers from two vector arrays efficiently.
    ///
    /// This is optimized for the common case of query/database vector pairs.
    /// Both buffers are created with a single iteration through the vectors.
    ///
    /// - Parameters:
    ///   - vectorsA: First array of vectors (e.g., queries)
    ///   - vectorsB: Second array of vectors (e.g., database)
    ///   - options: Metal resource options
    ///   - alignment: Required alignment in bytes
    /// - Returns: Tuple of buffers (A, B), or nil if creation fails
    @inlinable
    public func createAlignedBufferPair<V: VectorProtocol>(
        _ vectorsA: [V],
        _ vectorsB: [V],
        options: MTLResourceOptions = .storageModeShared,
        alignment: Int = 16
    ) -> (bufferA: any MTLBuffer, bufferB: any MTLBuffer)? where V.Scalar == Float {
        guard let bufferA = createAlignedBufferFromVectors(vectorsA, options: options, alignment: alignment),
              let bufferB = createAlignedBufferFromVectors(vectorsB, options: options, alignment: alignment) else {
            return nil
        }
        return (bufferA, bufferB)
    }
}
