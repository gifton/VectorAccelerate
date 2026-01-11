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

    // MARK: - Bundle Resolution

    /// Finds the VectorAccelerate resource bundle using multiple fallback strategies.
    ///
    /// This handles transitive dependency scenarios where `Bundle.module` may not
    /// resolve correctly (e.g., `App → PackageB → VectorAccelerate`).
    ///
    /// Strategies tried in order:
    /// 1. `Bundle.module` - works for direct SPM consumption
    /// 2. `Bundle(for: KernelContext.self)` - works for frameworks/transitive deps
    /// 3. Search all loaded bundles for one containing our resources
    /// 4. Known bundle identifiers as last resort
    private static func findVectorAccelerateBundle() -> Bundle? {
        // Strategy 1: Bundle.module (works for direct SPM consumption)
        #if SWIFT_PACKAGE
        let moduleBundle = Bundle.module
        if moduleBundle.url(forResource: "default", withExtension: "metallib") != nil ||
           moduleBundle.url(forResource: "L2Distance", withExtension: "metal") != nil {
            return moduleBundle
        }
        #endif

        // Strategy 2: Bundle containing this class (works for frameworks/transitive deps)
        let classBundle = Bundle(for: KernelContext.self)
        if classBundle.url(forResource: "default", withExtension: "metallib") != nil ||
           classBundle.url(forResource: "L2Distance", withExtension: "metal") != nil {
            return classBundle
        }

        // Strategy 3: Search all loaded bundles for one containing our resources
        for bundle in Bundle.allBundles + Bundle.allFrameworks {
            if bundle.url(forResource: "default", withExtension: "metallib") != nil ||
               bundle.url(forResource: "L2Distance", withExtension: "metal") != nil {
                return bundle
            }
        }

        // Strategy 4: Known bundle identifiers as last resort
        let knownIdentifiers = [
            "VectorAccelerate_VectorAccelerate",  // SPM-generated identifier
            "com.gifton.VectorAccelerate"         // Potential custom identifier
        ]
        for identifier in knownIdentifiers {
            if let bundle = Bundle(identifier: identifier),
               (bundle.url(forResource: "default", withExtension: "metallib") != nil ||
                bundle.url(forResource: "L2Distance", withExtension: "metal") != nil) {
                return bundle
            }
        }

        return nil
    }

    // MARK: - Library Validation

    /// Validates that a Metal library contains VectorAccelerate shader functions.
    ///
    /// This prevents accidentally using the host app's metallib when VectorAccelerate
    /// is consumed as a transitive SPM dependency. The host app's metallib would be
    /// returned by `device.makeDefaultLibrary()` but wouldn't contain our kernels.
    ///
    /// - Parameter library: The Metal library to validate
    /// - Returns: `true` if the library contains core VectorAccelerate kernel functions
    private static func isVectorAccelerateLibrary(_ library: any MTLLibrary) -> Bool {
        // Check for core VectorAccelerate kernel functions
        // If these exist, it's definitely our library
        return library.makeFunction(name: "l2_distance_kernel") != nil &&
               library.makeFunction(name: "dot_product_kernel") != nil
    }

    // MARK: - Metal Library Loading

    /// Load Metal library with fallback support for different environments.
    ///
    /// **Important**: This method prioritizes VectorAccelerate's bundled metallib to handle
    /// transitive SPM dependency scenarios correctly. When VectorAccelerate is used as a
    /// dependency (e.g., `App -> PackageB -> VectorAccelerate`), `device.makeDefaultLibrary()`
    /// would incorrectly return the host app's metallib instead of ours.
    ///
    /// Loading order:
    /// 1. VectorAccelerate's bundle (debug.metallib in DEBUG, default.metallib otherwise)
    /// 2. Runtime compilation from .metal source files (fallback)
    /// 3. Device's default library WITH VALIDATION (only when VectorAccelerate is the main app)
    ///
    /// - Parameter device: The Metal device to create the library for
    /// - Returns: The loaded Metal library, or nil if all approaches fail
    public static func loadMetalLibrary(device: any MTLDevice) -> (any MTLLibrary)? {
        // 1. Find VectorAccelerate's resource bundle FIRST
        //    This ensures we use our own metallib even when used as a dependency
        if let resourceBundle = findVectorAccelerateBundle() {
            #if DEBUG
            print("[VectorAccelerate] Found resource bundle: \(resourceBundle.bundlePath)")
            #endif

            // 2. In DEBUG builds, prefer debug.metallib for Xcode Metal Debugger support
            //    This library contains shader source via -frecord-sources flag
            #if DEBUG
            if let libraryURL = resourceBundle.url(forResource: "debug", withExtension: "metallib"),
               let library = try? device.makeLibrary(URL: libraryURL),
               isVectorAccelerateLibrary(library) {
                print("[VectorAccelerate] Loaded debug.metallib with shader debugging support")
                return library
            }
            #endif

            // 3. Try default.metallib from our bundle
            if let libraryURL = resourceBundle.url(forResource: "default", withExtension: "metallib"),
               let library = try? device.makeLibrary(URL: libraryURL),
               isVectorAccelerateLibrary(library) {
                #if DEBUG
                print("[VectorAccelerate] Loaded default.metallib from bundle")
                #endif
                return library
            }

            // 4. Fallback: Runtime compile from .metal sources
            if let library = compileMetalSourcesFromBundle(device: device, bundle: resourceBundle),
               isVectorAccelerateLibrary(library) {
                #if DEBUG
                print("[VectorAccelerate] Compiled Metal shaders at runtime from bundle resources")
                #endif
                return library
            }
        }

        // 5. Last resort: Try device's default library WITH VALIDATION
        //    Only succeeds if it actually contains VectorAccelerate functions
        //    (i.e., when VectorAccelerate IS the main app, like in tests)
        if let library = device.makeDefaultLibrary(),
           isVectorAccelerateLibrary(library) {
            #if DEBUG
            print("[VectorAccelerate] Using validated default library (VectorAccelerate is main app)")
            #endif
            return library
        }

        #if DEBUG
        print("[VectorAccelerate] Warning: Could not load VectorAccelerate Metal library")
        #endif
        return nil
    }

    /// Compile Metal shader sources from bundle resources.
    ///
    /// This is a fallback for edge cases where pre-compiled metallib is unavailable.
    /// Runtime compilation supports all shaders but has higher startup latency.
    ///
    /// - Parameters:
    ///   - device: The Metal device to create the library for
    ///   - bundle: The bundle containing .metal source files
    /// - Returns: The compiled Metal library, or nil if compilation fails
    private static func compileMetalSourcesFromBundle(
        device: any MTLDevice,
        bundle: Bundle
    ) -> (any MTLLibrary)? {
        // Load shader files that can be safely combined without symbol conflicts
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
            "BatchMax",
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
            "AttentionSimilarity",
            // IVF indexing
            "IVFListSearch",
            // Clustering operations (K-means, K-means++)
            "ClusteringShaders",
            // HDBSCAN / mutual reachability
            "MutualReachability",
            // HDBSCAN / MST computation (Boruvka's algorithm)
            "BoruvkaMST",
            // UMAP gradient computation
            "UMAPGradient",
            // Log-sum-exp and softmax for probability distributions
            "LogSumExp",
            // NLP / Topic Modeling
            "SparseLogTFIDF"
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

        // Metal4Common.h constants for ClusteringShaders
        #ifndef VA_SIMD_WIDTH
        #define VA_SIMD_WIDTH 32
        #endif
        #ifndef VA_INFINITY
        #define VA_INFINITY INFINITY
        #endif

        """

        for fileName in shaderFiles {
            if let url = bundle.url(forResource: fileName, withExtension: "metal"),
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
            #if DEBUG
            print("[VectorAccelerate] Warning: Failed to compile Metal shaders from bundle: \(error)")
            #endif
            return nil
        }
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
