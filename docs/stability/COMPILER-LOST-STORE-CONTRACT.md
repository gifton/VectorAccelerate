# Compiler lost-store contract (VC-C001)

**Decided:** 2026-10-30 (owner). The canary stays a green gate. It's an expected failure on the compilers where the miscompile is demonstrated, and a hard assert everywhere else.

## The miscompile

VectorCore's investigation found that Apple Swift 6.2 and 6.4.2 miscompile one shape at `-O`, `-Osize` and `-O` with WMO: an array literal held in a local, a `withUnsafeMutableBufferPointer` borrow whose closure hands the buffer to an out-of-line writer as `inout`, then an element read after or inside the borrow. RedundantLoadElimination forwards the literal across the write, so the read returns the stale value. By-value writers and non-literal initializers keep the store.

## Guards

- `CompilerLostStoreCanaryTests` holds the exact shape on purpose, with controls for the shapes VectorAccelerate relies on: a by-value writer, an out-pointer read inside the borrow, and a captured accumulator. In an optimized build on `6.2 <= compiler < 6.5`, a lost store is an expected, non-strict failure. On any other compiler, or in Debug, the store must be observed, so a newer toolchain that still loses the store turns the gate red.
- `LostStoreShapeGuardTests` scans `Sources` and `Tests` lexically for `inout` unsafe-pointer parameters, literal-initialized locals read after a mutable borrow, and uses of `x` after `withUnsafeMutablePointer(to: &x)` in its scope. Its allowlist needs reasons, and stale entries fail.

## Exposure review (2026-10-30)

Reviewed against the VectorCore 0.4.0 candidate on Swift 6.4.2.

- No `inout` unsafe-pointer parameters, apart from the canary's own writer.
- 8 mutable borrows in `Sources`. Every owner is `[T](repeating:count:)` or `Data(count:)`, with no literal owner and no `inout` hand-off.
- One post-borrow read, the `task_info` idiom in `PerformanceMonitor`. It passed the pointer by value, so it wasn't the miscompile, and it now reads inside the borrow.
- The full Release suite passed identically with `-sil-disable-pass=redundant-load-elimination`, so no VectorAccelerate test could tell the two compilers apart.

## Not enforced

The captured-variable form, which VectorCore flags by class but never reproduced, isn't enforced here. That's 20 sites, about half of them numerical accumulators in CPU fallbacks. The canary's accumulator control pins that this form keeps its writes on the current compiler.

`CPUTopK.search::heap`, `LearnedDistanceKernel.compute::result` x2, `UMAPGradientKernel` `result` x2, `QuantizationEngine` `result` x3, `HDBSCANDistanceModule.computeMST::result`, `MemoryMapManager.extractVector::result`, `SIMDFallback` `sum`/`sumSquared` x4, `SIMDOptimizer` `sum` x4, `StableNormalization.normalizedSIMD` `scale`/`sumSquares`.
