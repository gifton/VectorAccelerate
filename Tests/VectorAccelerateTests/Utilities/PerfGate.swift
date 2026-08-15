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

enum PerfGate {
    /// True when perf assertions should be enforced (dedicated hardware).
    static let strict = ProcessInfo.processInfo.environment["VECTORACCELERATE_STRICT_PERF"] == "1"
}
