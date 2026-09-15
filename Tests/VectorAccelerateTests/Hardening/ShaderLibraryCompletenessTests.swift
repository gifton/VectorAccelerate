//
//  ShaderLibraryCompletenessTests.swift
//  VectorAccelerateTests
//
//  Phase-1.2 hardening audit: the runtime combined-source compile
//  (`KernelContext.makeLibraryFromBundleSources`) is the primary shader load path in release
//  builds — `debug.metallib` is `#if DEBUG`-gated and SPM emits no `default.metallib` alongside
//  MetalCompilerPlugin. A `.metal` file missing from `runtimeCompileShaderFiles` silently strands
//  every kernel it defines (AUDIT-2 VA2-001: `SoADistance.metal` was missing from 0.6.0 until this
//  audit, so `MetalComputeProvider.init` threw in every release-configuration process).
//
//  These guards are GENERATED from the shader directory, not hand-maintained lists:
//    * every kernel in every non-excluded .metal file exists in the runtime-compiled library,
//    * the compiled/excluded lists exactly tile the shader directory,
//    * the runtime-compiled library's kernel set matches the plugin-built metallib's.
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class ShaderLibraryCompletenessTests: XCTestCase {

    /// Repo `Metal/Shaders` directory, derived from this file's compile-time path
    /// (`Tests/VectorAccelerateTests/Hardening/…` → repo root → `Sources/…`).
    private static let shadersDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // Hardening/
        .deletingLastPathComponent()   // VectorAccelerateTests/
        .deletingLastPathComponent()   // Tests/
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("Sources/VectorAccelerate/Metal/Shaders", isDirectory: true)

    private static func metalFiles() throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: shadersDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "metal" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func kernelNames(inMetalFile url: URL) throws -> [String] {
        let source = try String(contentsOf: url, encoding: .utf8)
        let regex = try NSRegularExpression(pattern: #"kernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)"#)
        let range = NSRange(source.startIndex..., in: source)
        return regex.matches(in: source, range: range).map {
            String(source[Range($0.range(at: 1), in: source)!])
        }
    }

    private func requireRepoShaders() throws -> [URL] {
        guard FileManager.default.fileExists(atPath: Self.shadersDir.path) else {
            throw XCTSkip("repo shader sources not present at \(Self.shadersDir.path) (running outside the repo)")
        }
        return try Self.metalFiles()
    }

    /// Every kernel in every non-excluded `.metal` file must exist in the runtime-compiled library.
    func testRuntimeCompiledLibraryContainsEveryKernel() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal not available") }
        let files = try requireRepoShaders()
        // 27 after AUDIT-3 Group F deleted CosineSimilarity.metal (the never-dispatched
        // specialized matrix-cosine family).
        XCTAssertGreaterThanOrEqual(files.count, 27, "expected the full shader corpus at \(Self.shadersDir.path)")

        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle(),
                                   "VectorAccelerate resource bundle not found")
        let library = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)
        let available = Set(library.functionNames)

        var missing: [String] = []
        for file in files {
            let base = file.deletingPathExtension().lastPathComponent
            if KernelContext.runtimeCompileExcludedShaderFiles[base] != nil { continue }
            for name in try Self.kernelNames(inMetalFile: file) where !available.contains(name) {
                missing.append("\(base).metal: \(name)")
            }
        }
        XCTAssertTrue(missing.isEmpty,
            "runtime-compiled library is missing \(missing.count) kernel(s):\n" + missing.joined(separator: "\n"))
    }

    // MARK: - VA3-031 class closure

    /// Known, currently-unarmed shadow residual (ledger-recorded): these two live kernels
    /// predate the `batch_` rewrite rule and are loaded via `makeFunction(name:)` directly,
    /// so the funnel path would mangle them if it were ever used. New entries here require a
    /// ledger record — shrink this set, never grow it silently.
    private static let knownShadowedKernelNames: Set<String> = [
        "batch_select_k_nearest_ascending",
        "batch_select_k_nearest_descending",
    ]

    /// VA3-031 class closure, part 1: `getPipeline(functionName:)` funnels literal kernel
    /// names through `PipelineCacheKey(operation:)`, so any rewriting case in
    /// `PipelineCacheKey.functionName` whose operation string equals a real kernel name
    /// silently hijacks that kernel's dispatch — exactly how the engine's single-pair dot
    /// product spent months running the batch kernel. Every kernel name in the corpus must
    /// round-trip the derivation unchanged.
    func testOperationDerivationNeverShadowsKernelNames() throws {
        let files = try requireRepoShaders()
        var shadowed: [String] = []
        for file in files {
            for name in try Self.kernelNames(inMetalFile: file)
            where !Self.knownShadowedKernelNames.contains(name) {
                let derived = PipelineCacheKey(operation: name).functionName
                if derived != name {
                    shadowed.append("\(name) → \(derived)")
                }
            }
        }
        XCTAssertTrue(shadowed.isEmpty,
            "operation-name derivation hijacks \(shadowed.count) literal kernel name(s):\n"
                + shadowed.joined(separator: "\n"))
    }

    /// VA3-031 class closure, part 2: every literal function name funneled through
    /// `getPipeline(functionName:)` / `getPipelineState(functionName:)` anywhere in Sources
    /// must name a real kernel. Four phantoms accumulated this way ("batchCosineDistance"
    /// pre-AUDIT-2, then "batchDotProduct", "batchManhattanDistance", "vectorMultiply") —
    /// each a throw that fired only when routing finally reached it. The allowlist holds the
    /// ledger's open items; shrink it, never grow it silently.
    func testSwiftLiteralPipelineRequestsResolveToRealKernels() throws {
        let files = try requireRepoShaders()
        var kernels = Set<String>()
        for file in files { kernels.formUnion(try Self.kernelNames(inMetalFile: file)) }

        let sourcesDir = Self.shadersDir
            .deletingLastPathComponent()   // Metal/
            .deletingLastPathComponent()   // → Sources/VectorAccelerate/
        let regex = try NSRegularExpression(
            pattern: #"getPipeline(?:State)?\(functionName:\s*"([A-Za-z_][A-Za-z0-9_]*)""#)
        // VA3-021 open item: VectorCoreIntegration's public multiply() has always thrown —
        // the "vectorMultiply" kernel never existed in any library.
        let openLedgerPhantoms: Set<String> = ["vectorMultiply"]

        var phantoms: [String] = []
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(
            at: sourcesDir, includingPropertiesForKeys: nil))
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let source = try String(contentsOf: url, encoding: .utf8)
            let range = NSRange(source.startIndex..., in: source)
            for match in regex.matches(in: source, range: range) {
                let name = String(source[Range(match.range(at: 1), in: source)!])
                if !kernels.contains(name) && !openLedgerPhantoms.contains(name) {
                    phantoms.append("\(url.lastPathComponent): \(name)")
                }
            }
        }
        XCTAssertTrue(phantoms.isEmpty,
            "\(phantoms.count) Swift literal pipeline request(s) name no existing kernel:\n"
                + phantoms.joined(separator: "\n"))
    }

    /// VA3-021 class closure, part 3: every literal funneled through `makeFunction(name:)`
    /// must also name a real kernel. This is the level where the OPTIONAL loads live —
    /// `if let f = library.makeFunction(name: "...") { ... } else { nil }` — and a phantom
    /// here is not even a throw: it is a permanently-nil pipeline behind a silent fallback
    /// (`tiledTransposeInPlace` quietly ran out-of-place; the specialized matmul trio and
    /// `batchMatrixVector` were dead weight; the neural tiled trio made three public APIs
    /// permanent throwers with forever-skipping tests). The audit knew of ONE such phantom;
    /// this test's first red run listed EIGHT. No allowlist: an optional load that never
    /// resolves is dead code by definition — delete it, don't exempt it.
    func testSwiftMakeFunctionLiteralsResolveToRealKernels() throws {
        let files = try requireRepoShaders()
        var kernels = Set<String>()
        for file in files { kernels.formUnion(try Self.kernelNames(inMetalFile: file)) }

        let sourcesDir = Self.shadersDir
            .deletingLastPathComponent()   // Metal/
            .deletingLastPathComponent()   // → Sources/VectorAccelerate/
        let regex = try NSRegularExpression(
            pattern: #"makeFunction\(name:\s*"([A-Za-z_][A-Za-z0-9_]*)""#)

        var phantoms: [String] = []
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(
            at: sourcesDir, includingPropertiesForKeys: nil))
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let source = try String(contentsOf: url, encoding: .utf8)
            let range = NSRange(source.startIndex..., in: source)
            for match in regex.matches(in: source, range: range) {
                let name = String(source[Range(match.range(at: 1), in: source)!])
                if !kernels.contains(name) {
                    phantoms.append("\(url.lastPathComponent): \(name)")
                }
            }
        }
        XCTAssertTrue(phantoms.isEmpty,
            "\(phantoms.count) makeFunction literal(s) name no existing kernel — "
                + "phantom pipelines behind silent fallbacks:\n"
                + phantoms.sorted().joined(separator: "\n"))
    }

    /// The compiled and excluded lists must exactly tile the shader directory, disjointly,
    /// and every exclusion must carry a reason. This is what turns "someone forgot to add the
    /// new file to the list" from a silent release-only breakage into a test failure.
    func testShaderFileListsTileTheDirectory() throws {
        let files = Set(try requireRepoShaders().map { $0.deletingPathExtension().lastPathComponent })
        let listed = Set(KernelContext.runtimeCompileShaderFiles)
        let excluded = Set(KernelContext.runtimeCompileExcludedShaderFiles.keys)

        for (name, reason) in KernelContext.runtimeCompileExcludedShaderFiles {
            XCTAssertFalse(reason.isEmpty, "exclusion for \(name).metal must state a reason")
        }
        XCTAssertTrue(listed.isDisjoint(with: excluded),
            "files both compiled and excluded: \(listed.intersection(excluded).sorted())")
        XCTAssertEqual(listed.union(excluded), files, """
            every .metal file must be either compiled or explicitly excluded with a reason.
            unaccounted: \(files.subtracting(listed.union(excluded)).sorted())
            stale list entries: \(listed.union(excluded).subtracting(files).sorted())
            """)
    }

    /// The runtime-compiled library and the plugin-built metallib must expose the same kernel set —
    /// they are two compilations of the same sources and MUST NOT drift.
    func testRuntimeLibraryMatchesMetallib() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal not available") }
        _ = try requireRepoShaders()
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        guard let metallibURL = bundle.url(forResource: "debug", withExtension: "metallib")
                ?? bundle.url(forResource: "default", withExtension: "metallib") else {
            throw XCTSkip("no prebuilt metallib in the resource bundle")
        }
        let metallib = try device.makeLibrary(URL: metallibURL)
        let runtime = try KernelContext.makeLibraryFromBundleSources(
            device: device, bundle: bundle)

        // Compare kernel names only for non-excluded files (excluded files are in the metallib
        // — per-file compiles can't conflict — but deliberately not in the combined compile).
        let excludedKernels = Set(try KernelContext.runtimeCompileExcludedShaderFiles.keys.flatMap {
            try Self.kernelNames(inMetalFile: Self.shadersDir.appendingPathComponent("\($0).metal"))
        })
        let fromMetallib = Set(metallib.functionNames).subtracting(excludedKernels)
        let fromRuntime = Set(runtime.functionNames)
        let missingInRuntime = fromMetallib.subtracting(fromRuntime).sorted()
        let extraInRuntime = fromRuntime.subtracting(Set(metallib.functionNames)).sorted()
        XCTAssertTrue(missingInRuntime.isEmpty, "metallib kernels absent from the runtime compile: \(missingInRuntime)")
        XCTAssertTrue(extraInRuntime.isEmpty, "runtime-compile kernels absent from the metallib: \(extraInRuntime)")
    }

    /// AUDIT-2 VA2-006 (FIXED): every `PipelineCacheKey.functionName` derivation for the
    /// pre-warm key sets must resolve to a function that actually exists. The pre-audit
    /// derivations produced four phantoms (`cosine_similarity_kernel`, `top_k_selection`,
    /// underscore-stripped `fusedl2topk`, and `.capitalized`-mangled `batchEuclideandistance`)
    /// that existed in no library, so requests through them could only throw — and warm-up
    /// swallowed the failures. Hard assertion now; a new phantom is a red test.
    func testCommonPipelineKeysResolveToRealFunctions() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal not available") }
        _ = try requireRepoShaders()
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        let runtime = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)
        var known = Set(runtime.functionNames)
        if let metallibURL = bundle.url(forResource: "debug", withExtension: "metallib"),
           let metallib = try? device.makeLibrary(URL: metallibURL) {
            known.formUnion(metallib.functionNames)
        }

        let unresolved = (PipelineCacheKey.commonKeys + PipelineCacheKey.embeddingModelKeys)
            .filter { !known.contains($0.functionName) }
            .map { "\($0.operation)(d\($0.dimension)) → \($0.functionName)" }
            .sorted()

        XCTAssertTrue(unresolved.isEmpty,
            "\(unresolved.count) cache keys resolve to nonexistent functions:\n" + unresolved.joined(separator: "\n"))
    }

    /// AUDIT-3 VA3-025-adjacent (Group D): every key in every *built-in PipelineRegistry* must
    /// also resolve to a real function. Before this test, the registries carried eight phantom
    /// operation names (`l2_normalize`, `compute_statistics`, `scalar_quantize_int8/int4`,
    /// `binary_quantize`, `matrix_multiply`, `matrix_transpose`, `attention_similarity`,
    /// `neural_quantization`) that existed in no library — registry-driven warm-up could only
    /// fail on them, silently before the AUDIT-2 loud-warm-up work.
    func testBuiltInRegistryKeysResolveToRealFunctions() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal not available") }
        _ = try requireRepoShaders()
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle())
        let runtime = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)
        var known = Set(runtime.functionNames)
        if let metallibURL = bundle.url(forResource: "debug", withExtension: "metallib"),
           let metallib = try? device.makeLibrary(URL: metallibURL) {
            known.formUnion(metallib.functionNames)
        }

        let registries: [(String, PipelineRegistry)] = [
            ("default", .default),
            ("journalingApp", .journalingApp),
            ("embeddingFocused", .embeddingFocused),
            ("minimal", .minimal),
        ]
        var unresolved: [String] = []
        for (name, registry) in registries {
            for tier in PipelineTier.allCases {
                for key in registry.keys(for: tier) where !known.contains(key.functionName) {
                    unresolved.append("\(name).\(tier): \(key.operation)(d\(key.dimension)) → \(key.functionName)")
                }
            }
        }
        XCTAssertTrue(unresolved.isEmpty,
            "\(unresolved.count) registry keys resolve to nonexistent functions:\n" + unresolved.sorted().joined(separator: "\n"))
    }
}
