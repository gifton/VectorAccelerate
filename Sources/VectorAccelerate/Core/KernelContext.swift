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
    ///
    /// - Note: `internal` rather than `private` so tests can drive the
    ///   runtime-compilation path directly.
    internal static func findVectorAccelerateBundle() -> Bundle? {
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
    /// This is a fallback for edge cases where pre-compiled metallib is unavailable
    /// — which in **release** builds is the primary path, since `debug.metallib` is
    /// only loaded under `#if DEBUG`. A single compile error here disables every
    /// kernel in the package, so `NormalizationParityTests` exercises this entry
    /// point directly as a regression guard.
    ///
    /// - Parameters:
    ///   - device: The Metal device to create the library for
    ///   - bundle: The bundle containing .metal source files
    /// - Returns: The compiled Metal library, or nil if compilation fails
    internal static func compileMetalSourcesFromBundle(
        device: any MTLDevice,
        bundle: Bundle
    ) -> (any MTLLibrary)? {
        do {
            return try makeLibraryFromBundleSources(device: device, bundle: bundle)
        } catch {
            // Compilation failed - log error details
            #if DEBUG
            print("[VectorAccelerate] Warning: Failed to compile Metal shaders from bundle: \(error)")
            #endif
            return nil
        }
    }

    /// Every `.metal` file compiled into the runtime combined-source library, by base name.
    ///
    /// !! This list MUST cover every shader file the package ships — on the runtime-compile
    /// load path (the primary path in release builds, where `debug.metallib` is `#if DEBUG`-
    /// gated and SPM produces no `default.metallib` alongside MetalCompilerPlugin) a file
    /// missing from this list silently strands every kernel it defines. `SoADistance` was
    /// missing here from 0.6.0 until the 2026-08 hardening audit, which made
    /// `MetalComputeProvider` init throw in every release-configuration process (AUDIT-2
    /// VA2-001). `ShaderLibraryCompletenessTests` now asserts directory ↔ list parity and
    /// per-kernel presence in the compiled library.
    internal static let runtimeCompileShaderFiles: [String] = [
        // Core distance kernels (CosineSimilarity.metal — the never-dispatched specialized
        // matrix-cosine family — was deleted in AUDIT-3 Group F; live cosine kernels are in
        // DistanceShaders/BasicOperations/SoADistance)
        "L2Distance",
        "DotProduct",
        "DistanceShaders",
        "HammingDistance",
        "MinkowskiDistance",
        // Zero-copy lane-major SoA scoring (0.6.0)
        "SoADistance",
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

    /// Shader files deliberately NOT in the runtime combined compile, with the reason.
    /// `ShaderLibraryCompletenessTests` enforces: directory = listed ∪ excluded (disjoint).
    ///
    /// Excluded kernels exist only in the prebuilt metallib (per-file compilation), i.e. they are
    /// UNAVAILABLE on the runtime-compile load path — acceptable solely for files nothing in the
    /// Swift target references. Currently empty: the dead ManhattanDistance/ChebyshevDistance
    /// shaders that briefly sat here were deleted outright (AUDIT-2 VA2-002).
    internal static let runtimeCompileExcludedShaderFiles: [String: String] = [:]

    /// The hand-maintained preamble prepended to the runtime combined-source compile in place
    /// of the stripped `#include "Metal4Common.h"`. Internal so `PreambleParityTests` can
    /// assert numeric identity against the header without parsing this Swift file.
    internal static let runtimeCompilePreamble: String = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        #include <metal_atomic>
        using namespace metal;

        // Numerical-stability floor — MUST STAY NUMERICALLY IDENTICAL to Metal4Common.h's
        // `VA_EPSILON` (PreambleParityTests.testPreambleConstantsMatchHeader enforces it).
        // Defined under the SAME name the shader sources use: no epsilon token rewriting
        // happens below, so a per-file macro shadow can no longer silently redefine the
        // shared value for every downstream file in this ONE combined TU (AUDIT-3 VA3-012;
        // EpsilonCompileParityTests guards both build paths against re-drift).
        #ifndef VA_EPSILON
        #define VA_EPSILON 1e-7f
        #endif

        // VA3-030: rooted L2 keeps its fast accumulation, then rescues range failures.
        // Squared outputs intentionally bypass this helper. EuclideanRangePolicyTests covers
        // AoS/SoA layouts, both libraries, overflow, underflow, and nonfinite input controls.
        inline float va_euclidean_finalize(float sum, device const float* a, device const float* b,
                                            uint dimension, ulong b_lane_stride = 4) {
            const uint sum_bits = as_type<uint>(sum);
            if (sum_bits >= 0x00800000u && sum_bits < 0x7F800000u) return sqrt(sum);

            float max_diff = 0.0f;
            bool has_nan = false;
            for (uint i = 0; i < dimension; ++i) {
                const ulong bi = (ulong)(i / 4) * b_lane_stride + (i & 3);
                const float diff = fabs(a[i] - b[bi]);
                const uint bits = as_type<uint>(diff) & 0x7FFFFFFFu;
                has_nan |= bits > 0x7F800000u;
                max_diff = max(max_diff, diff);
            }
            if (has_nan) return as_type<float>(0x7FC00000u);
            if (max_diff > FLT_MAX) return INFINITY; // Even one difference exceeds the output range.
            if (max_diff == 0.0f) return 0.0f;

            float scaled_sum = 0.0f;
            for (uint i = 0; i < dimension; ++i) {
                const ulong bi = (ulong)(i / 4) * b_lane_stride + (i & 3);
                const float normalized = precise::divide(a[i] - b[bi], max_diff);
                scaled_sum = fma(normalized, normalized, scaled_sum);
            }
            // Keep the large/tiny scale out of the squared arithmetic, including under fast-math.
            int exponent;
            const float mantissa = frexp(max_diff, exponent);
            return ldexp(mantissa * sqrt(scaled_sum), exponent);
        }

        // VA3-016: shared Top-K ordering. Integer NaN classification survives fast-math.
        // Numeric values precede NaNs in both directions; ties (including +/-0 and NaNs)
        // prefer the smaller original index. Invalid slots follow every real candidate.
        // TopKNaNPolicyTests exercises admission, sorting, merging, and padding in both libraries.
        inline bool va_topk_is_better(float a, uint ai, float b, uint bi, bool ascending) {
            if (ai == 0xFFFFFFFFu) return false;
            if (bi == 0xFFFFFFFFu) return true;
            const bool a_nan = (as_type<uint>(a) & 0x7FFFFFFFu) > 0x7F800000u;
            const bool b_nan = (as_type<uint>(b) & 0x7FFFFFFFu) > 0x7F800000u;
            if (a_nan != b_nan) return !a_nan;
            if (!a_nan) {
                // Order FP32 bit patterns without arithmetic: fast-math may flush subnormals.
                // Collapse signed zero, then reverse negatives and move positives above them.
                uint ab = as_type<uint>(a), bb = as_type<uint>(b);
                if ((ab & 0x7FFFFFFFu) == 0) ab = 0;
                if ((bb & 0x7FFFFFFFu) == 0) bb = 0;
                const uint ak = (ab & 0x80000000u) ? ~ab : (ab ^ 0x80000000u);
                const uint bk = (bb & 0x80000000u) ? ~bb : (bb ^ 0x80000000u);
                if (ak != bk) return ascending ? ak < bk : ak > bk;
            }
            return ai < bi;
        }

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

        // Normalization policy constants (BE3 §4.4) for BasicOperations.metal and
        // L2Normalization.metal.
        //
        // !! MUST STAY NUMERICALLY IDENTICAL TO Metal/Shaders/Metal4Common.h !!
        // The `#include "Metal4Common.h"` line is stripped below, so anything this
        // preamble omits is simply undeclared — and because a single compile error
        // fails the ONE combined source, that takes down every kernel in the
        // package, not just the file that used the constant. Metal4Common.h wraps
        // its own definitions in `#ifndef`, so these macros win if both are ever
        // seen. Any constant added there must be mirrored here.
        #ifndef VA_NORM_MIN_DENOM
        #define VA_NORM_MIN_DENOM 0x1p-126f
        #endif
        #ifndef VA_NORM_MAX_DENOM
        #define VA_NORM_MAX_DENOM 0x1p126f
        #endif
        #ifndef VA_NORM_MIN_SCALED
        #define VA_NORM_MIN_SCALED 0.5f
        #endif
        #ifndef VA_NORM_MAX_SCALED
        #define VA_NORM_MAX_SCALED 0x1p100f
        #endif

        // Cosine similarity overflow/underflow rescue (AUDIT-2 VA2-008/VA2-009).
        // !! BYTE-IDENTICAL COPY of the guarded block in Metal/Shaders/Metal4Common.h !!
        // The `#include "Metal4Common.h"` line is stripped from every file below, so the
        // cosine kernels' shared helpers must be provided here. PreambleParityTests
        // .testCosineRescueBlockIdentical compares the two blocks and fails on any drift.
        #ifndef VA_COSINE_RESCUE_DEFINED
        #define VA_COSINE_RESCUE_DEFINED

        // True when the naive accumulators cannot represent the pair correctly: an overflowed (Inf)
        // term, or a squared norm that collapsed to exactly 0 (zero vector — cheap to re-confirm — or
        // flushed subnormal squares).
        //
        // Expressed as magnitude comparisons, NOT isinf(): under fast math a toolchain may fold
        // isinf() to false (measured in the plugin-built metallib during the 2026-08 audit, while the
        // same source runtime-compiled kept it), silently disabling the rescue. `x > FLT_MAX` is true
        // exactly for +Inf, compiles as a dynamic compare, and is false for NaN — NaN deliberately
        // rides the primary path into the NaN-propagating finalization.
        inline bool va_cosine_accumulators_unreliable(float dotAB, float normSqA, float normSqB) {
            return fabs(dotAB) > FLT_MAX || normSqA > FLT_MAX || normSqB > FLT_MAX
                || normSqA == 0.0f || normSqB == 0.0f;
        }

        // Recompute (A·B, ‖A‖², ‖B‖²) in the pre-scaled domain. |a·aScale| ≤ 4 (≤ 1 when maxAbs ≤
        // 2^126), so every accumulator is bounded by 16·dimension — no overflow, no subnormal collapse.
        // max() drops a NaN operand, so maxAbs stays finite for NaN-poisoned vectors; the NaN itself
        // still propagates through the scaled products. Cold path: serial, correctness over speed.
        inline float3 va_cosine_rescaled_terms(
            device const float* a,
            device const float* b,
            uint dimension
        ) {
            float aMax = 0.0f;
            float bMax = 0.0f;
            for (uint i = 0; i < dimension; ++i) {
                aMax = max(aMax, fabs(a[i]));
                bMax = max(bMax, fabs(b[i]));
            }
            const float aScale = 1.0f / clamp(aMax, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);
            const float bScale = 1.0f / clamp(bMax, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);
            float dotAB = 0.0f;
            float aa = 0.0f;
            float bb = 0.0f;
            for (uint i = 0; i < dimension; ++i) {
                const float x = a[i] * aScale;
                const float y = b[i] * bScale;
                dotAB = fma(x, y, dotAB);
                aa = fma(x, x, aa);
                bb = fma(y, y, bb);
            }
            return float3(dotAB, aa, bb);
        }

        // Shared finalization: similarity = (dot/‖A‖)/‖B‖, NaN-propagating clamp to [-1, 1], FLT_MIN
        // degenerate floor per norm (BE3 4.5: absolute, not precision-relative), zero-vector policy
        // similarity = 0. Robust to fast-math comparison flips on NaN: both branches propagate NaN.
        //
        // The single-product denominator sqrt(aa)*sqrt(bb) is deliberately NOT formed: under fast math
        // (Metal's default) the compiler reassociates it into sqrt(aa*bb), whose argument overflows to
        // +Inf for |components| ≳ 1e18 and silently collapsed the similarity to 0 — measured on Apple
        // silicon during the 2026-08 audit (AUDIT-2 VA2-008). The divisions are precise::divide, also
        // measured necessary: plain `/` lets fast math rewrite (dot/normA)/normB into
        // dot·rcp(normA·normB), and that reciprocal is SUBNORMAL for norm products ≳ 8.5e37 — flushed
        // to zero, collapsing the similarity to 0 for |components| ≈ 1e19 even with every accumulator
        // finite. Same lesson as the normalize kernels (see VA_NORM_MIN_SCALED notes above). With the
        // two-stage precise divide every intermediate is exact-by-construction: |dotAB/normA| ≤ normB
        // by Cauchy-Schwarz, so the final quotient lands in [-1, 1] up to rounding.
        inline float va_cosine_similarity_finalize(float dotAB, float aa, float bb) {
            const float normA = sqrt(aa);
            const float normB = sqrt(bb);
            if (normA > FLT_MIN && normB > FLT_MIN) {
                const float raw = precise::divide(precise::divide(dotAB, normA), normB);
                return isnan(raw) ? raw : clamp(raw, -1.0f, 1.0f);
            }
            return (isnan(dotAB) || isnan(aa) || isnan(bb)) ? NAN : 0.0f;
        }

        #endif // VA_COSINE_RESCUE_DEFINED


        """

    /// Throwing core of ``compileMetalSourcesFromBundle(device:bundle:)``.
    ///
    /// Internal (not private) so tests can assert on the compiler diagnostic rather
    /// than on a silent `nil`.
    internal static func makeLibraryFromBundleSources(
        device: any MTLDevice,
        bundle: Bundle
    ) throws -> any MTLLibrary {
        let shaderFiles = runtimeCompileShaderFiles
        var combinedSource = runtimeCompilePreamble

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

                // NOTE: the EPSILON string surgery that used to live here (stripping four
                // exact spellings of `constant float EPSILON = …;` and rewriting every
                // VA_EPSILON token to EPSILON) is deliberately GONE (AUDIT-3 VA3-012): the
                // rewrite turned DistanceShaders' file-local macro shadow into a
                // redefinition of the shared value for every downstream file, and the
                // whitespace-fragile exact-string strips had no remaining targets.
                // VA_EPSILON is provided by the preamble under its real name; a file needing
                // a different floor names its own constant (EpsilonCompileParityTests
                // enforces this corpus-wide).
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
        let options = MTLCompileOptions()
        options.mathMode = .fast
        return try device.makeLibrary(source: combinedSource, options: options)
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
