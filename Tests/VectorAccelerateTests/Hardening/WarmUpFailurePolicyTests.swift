//
//  WarmUpFailurePolicyTests.swift
//  VectorAccelerateTests
//
//  `Metal4Context` warms pipelines in a detached task and asserts when a key names a kernel that
//  no library has (AUDIT-2 VA2-006). A pipeline compile error on a kernel that does exist is a
//  different thing: shared virtualized runners produce them nondeterministically, and asserting on
//  them aborted whichever unrelated test was running. Only the first kind may assert.
//

import XCTest
@preconcurrency import Metal
import VectorCore
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class WarmUpFailurePolicyTests: XCTestCase {

    private let phantomKey = PipelineCacheKey(operation: "va_warmup_policy_no_such_kernel")

    /// Uses the real load path, so this also pins that a missing kernel still surfaces in the shape
    /// `isMissingKernel` recognises.
    func testPhantomKeyIsAMissingKernelAndAsserts() async throws {
        let context = try await Metal4Context()
        let failures = await context.pipelineCache.warmUp(keys: [phantomKey])

        XCTAssertEqual(failures.count, 1)
        XCTAssertEqual(failures.first?.isMissingKernel, true, "\(failures)")

        let message = try XCTUnwrap(Metal4Context.warmUpAssertionMessage(for: failures))
        XCTAssertTrue(message.contains(phantomKey.functionName), message)
    }

    /// Same constructor `Metal4ShaderCompiler.compilePipeline` throws for a failed compile. A real
    /// compile failure can't be provoked on demand, so the error value is built directly.
    func testCompileFailureOnARealKernelDoesNotAssert() {
        let failure = PipelineWarmUpFailure(
            key: PipelineCacheKey(operation: "l2_distance_kernel"),
            error: VectorError.shaderCompilationFailed(
                "Pipeline creation failed for l2_distance_kernel: Compilation failed"
            )
        )

        XCTAssertFalse(failure.isMissingKernel)
        XCTAssertNil(Metal4Context.warmUpAssertionMessage(for: [failure]))
    }

    func testAssertionNamesOnlyMissingKernels() {
        let compileFailure = PipelineWarmUpFailure(
            key: PipelineCacheKey(operation: "l2_distance_kernel"),
            error: VectorError.shaderCompilationFailed("Compilation failed")
        )
        let missing = PipelineWarmUpFailure(
            key: phantomKey,
            error: VectorError.shaderNotFound(name: phantomKey.functionName)
        )

        let message = Metal4Context.warmUpAssertionMessage(for: [compileFailure, missing])

        XCTAssertEqual(message?.contains(phantomKey.functionName), true)
        XCTAssertEqual(message?.contains("l2_distance_kernel"), false)
    }

    func testNoFailuresDoNotAssert() {
        XCTAssertNil(Metal4Context.warmUpAssertionMessage(for: []))
    }

    /// The deterministic half of the original guard: no common key may name a missing kernel. Compile
    /// errors are tolerated here because this test also runs on shared runners.
    func testCommonKeysAllNameRealKernels() async throws {
        let context = try await Metal4Context()
        let failures = await context.pipelineCache.warmUp(keys: PipelineCacheKey.commonKeys)

        XCTAssertNil(Metal4Context.warmUpAssertionMessage(for: failures))
    }
}
