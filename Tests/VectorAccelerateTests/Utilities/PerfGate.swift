//
//  PerfGate.swift
//  VectorAccelerateTests
//
//  Gate for wall-clock, throughput, and speedup assertions.
//
//  Shared virtualized CI runners fail timing thresholds nondeterministically
//  (GPU command-buffer hangs, runtime pipeline-compile errors, VM scheduling
//  noise), so perf assertions run only when VECTORACCELERATE_STRICT_PERF=1 —
//  mirroring VectorCore's VECTORCORE_STRICT_PERF gate, introduced upstream in
//  0.3.2 for the same reason. The gated tests still execute their computation
//  everywhere (kernel smoke coverage is preserved); only the perf assertions
//  are conditional.
//
//  Local strict run:  VECTORACCELERATE_STRICT_PERF=1 swift test
//

import Foundation
import XCTest

enum PerfGate {
    /// True when perf assertions should be enforced (dedicated hardware).
    static let strict = ProcessInfo.processInfo.environment["VECTORACCELERATE_STRICT_PERF"] == "1"

    /// Skip gate for sustained-GPU-load benchmark suites on shared CI runners.
    ///
    /// The virtualized runner GPU wedges under sustained load: once one command
    /// buffer hangs (kIOGPUCommandBufferCallbackErrorHang), subsequent GPU work in
    /// the same VM — including unrelated suites running in parallel — fails with
    /// hangs and pipeline CompilerErrors. AttentionSimilarityBenchmarkTests was the
    /// epicenter in every observed failing run (2026-06 investigation + PR #36/#37
    /// reruns), so its suite skips where the CI env var is set (GitHub Actions sets
    /// CI=true) unless VECTORACCELERATE_GPU_STRESS=1 opts back in (dedicated
    /// hardware). Local runs are unaffected: CI is unset.
    static func skipUnlessGPUStressAllowed() throws {
        let env = ProcessInfo.processInfo.environment
        if env["CI"] != nil && env["VECTORACCELERATE_GPU_STRESS"] != "1" {
            throw XCTSkip("GPU-stress benchmark suite skipped on shared CI runner (set VECTORACCELERATE_GPU_STRESS=1 to run)")
        }
    }
}
