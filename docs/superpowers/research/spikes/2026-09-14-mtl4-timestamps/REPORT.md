> THROWAWAY SPIKE EVIDENCE. The source in this directory is not built by SwiftPM, is not product code, and is retained only so the numbers below can be re-run. Copied verbatim from the session scratchpad on 2026-09-14.

# Metal 4 counter-heap timestamp feasibility spike — REPORT

Throwaway spike. Nothing under the VectorAccelerate repo was modified. All files live in this directory:

- `mtl4_timestamp_spike.swift` — single-file Swift executable (source of every number below)
- `full-run-1.txt`, `full-run-2.txt`, `full-run-3-reverse.txt` — the three measured runs (20 warm-up + 100 measured per mode)
- `validation-run.txt` — short run under `MTL_DEBUG_LAYER=1` (Metal API validation), exit 0, no validation diagnostics

Machine: Apple M3 Max (40-core GPU, 48 GiB), macOS 26.5.2 (25F84), Xcode 26.6 (17F113), Swift 6.3.3, SDK MacOSX26.5.
`device.supportsFamily(.metal4) == true`.

Workload: `copy_f32x4` kernel compiled at runtime from an MSL string with `device.makeLibrary(source:options:)`; 16 M Float32 (64 MiB) copied as 4 M `float4` threads, 256 threads/threadgroup, `.storageModeShared` buffers; 128 MiB traffic per dispatch. One dispatch per `MTL4CommandBuffer`, `MTL4CommandQueue.commit([cb], options:)` + `signalEvent`, CPU waits with `MTLSharedEvent.wait(untilSignaledValue:timeoutMS:)`. Output verified equal to input with `memcmp` once before measuring. Process QoS set with `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`. No `CFAbsoluteTimeGetCurrent`/`CACurrentMediaTime` used anywhere.

Per iteration, four heap slots are written: `[cb0, enc1, enc2, cb3]` = CB-level before, encoder-level before dispatch, encoder-level after dispatch, CB-level after. Modes: `none` (no timestamps), `cbOnly` (cb0/cb3 only), `relaxed` (all four, encoder stamps `.relaxed`), `precise` (all four, encoder stamps `.precise`).

Time spent on API wrangling before first successful measured run: about 7 minutes (three compile iterations for Swift-side renames; one segfault that was my own `String(format: "%s", swiftString)` bug, not Metal).

---

## 1. Answers per numbered question

### Q1 — `MTL4CounterHeap(.timestamp)` + `MTL4ComputeCommandEncoder.writeTimestamp(granularity:counterHeap:index:)` before/after a dispatch: non-zero and monotonic? — **YES for both `.relaxed` and `.precise`.**

Evidence (3 runs × 100 measured samples per granularity, 600 encoder-stamp pairs total):

- Zero timestamps in written slots: **0 / 600**.
- Encoder pair `enc1 <= enc2`: held in **600 / 600** samples (violations "enc1>enc2: 0" in every run).
- Kernel time `(enc2 − enc1) / 24 MHz`: median 0.3599–0.3614 ms across all six (run × granularity) combinations.
- Example raw ticks (run 1, relaxed, first measured sample): `[31978070173683, 31978070173810, 31978070182450, 31978070182519]` → cb0→enc1 5.3 µs, enc1→enc2 360.0 µs, enc2→cb3 2.9 µs.

Swift signature that compiled: `encoder.writeTimestamp(granularity: MTL4TimestampGranularity, counterHeap: any MTL4CounterHeap, index: Int)` (the ObjC selector is `writeTimestampWithGranularity:intoHeap:atIndex:`; the Swift label is `counterHeap:`, not `into:`).

Caveat on "monotonic" across the *mixed* CB-level/encoder-level sequence: the CB-level "before" stamp `cb0` was later than the encoder "before" stamp `enc1` in 6–14 % of samples (cb0>enc1: relaxed 6/8/12, precise 10/14/12 per run), by 12–514 ticks (0.5–21.4 µs). `enc2 <= cb3` and `cb0 <= cb3` held in 100 % of samples. So each pair is internally monotonic, but a delta must not be formed by mixing a CB-level stamp with an encoder-level stamp.

### Q2 — `MTL4CommandBuffer.writeTimestamp(counterHeap:index:)` at command-buffer level: works? — **YES.**

- Non-zero in 300 / 300 measured samples (`cbOnly`, `relaxed`, `precise`, 3 runs), `cb0 <= cb3` in 300 / 300.
- CB-heap span median: 0.3672–0.3721 ms, within 0.7–1.0 % of the `MTL4CommitFeedback` GPU span for the same command buffers (cbHeap is consistently ~2.5 µs shorter than cbGPU).
- In `cbOnly` mode the never-written slots 1 and 2 resolve as 0, as the header documents for unwritten/invalidated entries. (Run 1's `cbOnly` line reports "any zero: 100/100" because my first check counted those never-written slots; runs 2–3 count written slots only and report 0/100.)

### Q3 — CPU resolve + `queryTimestampFrequency()`: frequency and resolution — **Works.**

- `device.queryTimestampFrequency()` = **24,000,000 ticks/s** → **41.667 ns per tick**. (This equals the Apple Silicon `mach_absolute_time` timebase rate, 24 MHz.)
- `device.size(ofCounterHeapEntry: .timestamp)` = 8 bytes; `resolveCounterRange(_:)` returned `Data` of exactly 8 × count bytes, tightly packed `UInt64`.
- Swift signature: `try heap.resolveCounterRange(Range<Int>) -> Data?` — it is `throws` in Swift and takes a `Range<Int>`, not `NSRange`. It never threw or returned nil in 900+ calls.
- Resolving per iteration (4 entries, immediately after the shared-event wait) and re-resolving the whole 480-entry heap at the end gave identical values: **0 / 300 mismatches**.
- Effective resolution vs the measured quantity: a 0.36 ms kernel is ≈ 8,650 ticks; one tick is 0.012 % of it. The run-to-run spread of the kernel median (≤ 1.5 µs ≈ 36 ticks) is far larger than one tick, so 41.7 ns is not the limiting factor for this workload.
- Epoch observation: heap ticks / 24 MHz (e.g. 31,980,091,731,261 / 24e6 = 1,332,503.82 s) numerically agree with the legacy nanosecond timestamps sampled seconds later (1,332,503,990,743,458 ns = 1,332,503.99 s) — consistent with both clocks sharing the system-uptime origin. Stated as an observation, not verified against documentation.

### Q4 — Comparison of the clocks for the same work — **See tables in §2.**

Command-buffer GPU time on the Metal 4 path: `MTL4CommandBuffer` itself has **no** `gpuStartTime`/`gpuEndTime`. The equivalent is `MTL4CommitFeedback.gpuStartTime` / `.gpuEndTime` (`CFTimeInterval`, host seconds), delivered asynchronously to a handler registered via `MTL4CommitOptions.addFeedbackHandler` and passed to `queue.commit([cb], options:)`. The repo's `Metal4Context.swift` (lines 436–437, 512–513) reads `MTLCommandBuffer.gpuStartTime/gpuEndTime` on a **legacy** `MTLCommandBuffer` — its `commandQueue` is `any MTLCommandQueue` (line 128) created via `device.rawDevice.makeCommandQueue()` (line 233); the "In Metal 4:" lines there are comments, not code.

Headline medians (run 1; runs 2–3 in §2):

| clock | what it brackets | median | vs kernel |
|---|---|---|---|
| wall (`ContinuousClock`) | allocator.reset + begin + encode + commit + signalEvent + shared-event wait | 0.5451–0.5668 ms | +0.18–0.21 ms |
| cbGPU (commit feedback) | GPU span of the command buffer | 0.3706–0.3722 ms (with timestamps), 0.3627 ms (`none`) | +10 µs / +2 µs |
| cbHeap (CB-level heap stamps) | cb0→cb3 | 0.3681–0.3696 ms | +8 µs |
| kernel (encoder heap stamps) | enc1→enc2 | 0.3605 ms (relaxed), 0.3614 ms (precise) | — |

### Q5 — Does `.precise` measurably change the kernel time vs `.relaxed`? — **No measurable difference on this workload.**

Kernel-time median, precise − relaxed: run 1 +0.8 µs (+0.23 %), run 2 −0.1 µs (−0.02 %), run 3 (reversed order) −1.4 µs (−0.40 %). Sign flips between runs; magnitude is below the run-to-run spread of either mode alone. The CB GPU span and the cb0→enc1 / enc2→cb3 gaps are also indistinguishable between the two. Caveat: with exactly one dispatch per encoder, a `.relaxed` stamp that "may sample at command encoder boundaries" (header wording) coincides with the dispatch boundary, so this test cannot distinguish the two; a multi-dispatch encoder was not tested.

What *is* measurable is the cost of having any heap timestamps in the command buffer at all: cbGPU median is 0.3617–0.3631 ms with no timestamps (`none`) and 0.3698–0.3750 ms with any (`cbOnly`, `relaxed`, `precise`), i.e. **+7 to +13 µs (~2–3 %)** on the command-buffer span, reproduced in all three runs including the reversed mode order. In `none` mode, cbGPU (0.362 ms) is essentially equal to the kernel time measured in the other modes (0.361 ms), so the extra CB span in timestamped modes is attributable to the timestamp writes themselves (inference from the data, not from documentation). The kernel span (enc1→enc2) is unaffected.

### Q6 — Legacy `MTLCounterSampleBuffer` all-zeros on macOS 26 (wgpu #9414)? — **Refuted for the paths tested here; one anomaly found.**

I did not fetch the wgpu issue (no web access used); the statement checked is the one given in the task: "returns all zeros on macOS 26".

- `supportsCounterSampling(.atStageBoundary)` = **true**; `.atDispatchBoundary` = **false**; `.atBlitBoundary` = **false**; `.atDrawBoundary` = **false**. `device.counterSets` = `["timestamp"]`.
- Stage-boundary sampling via `MTLComputePassDescriptor.sampleBufferAttachments[0]` (`startOfEncoderSampleIndex = 0`, `endOfEncoderSampleIndex = 1`) on a `.shared` sample buffer, resolved on the CPU with `resolveCounterRange(0..<4)`: **non-zero in 15 / 15 runs** (3 executions × 5 runs). Example: `[1332419849158833, 1332419849544166, 0, 0]`, stage delta 385,333.
- Units: the legacy values are **nanoseconds**. In run 1 (single encoder per CB) the stage delta (385,333) equals the legacy `MTLCommandBuffer.gpuEndTime − gpuStartTime` (0.3853 ms) to the printed precision. `device.sampleTimestamps()` returned **identical** CPU and GPU values (e.g. cpu = gpu = 1332419746790791) and a gpu/cpu delta ratio of exactly 1.000000 over a 100 ms sleep.
- Slots 2–3 are zero only because dispatch-boundary sampling is unsupported on this device, so `sampleCounters(sampleBuffer:sampleIndex:barrier:)` was never issued. A client that assumes `.atDispatchBoundary` and reads those slots would see zeros.
- GPU-side resolve via `MTLBlitCommandEncoder.resolveCounters(_:range:destinationBuffer:destinationOffset:)` into a shared `MTLBuffer`, `.shared` sample buffer: **non-zero, byte-identical to the CPU resolve** in 10 / 10 runs (executions 2–3).
- **Anomaly**: same blit resolve on a `.private` sample buffer written by a second compute encoder in the same command buffer: non-zero in 10 / 10 runs, but from the second run onward the resolved values were those of the **previous** run's encoder (one run stale). E.g. run 1 resolved `[1332504095525916, 1332504096212666]`, identical to run 0's result; run 2 resolved `[1332504097774458, …]`, which follows run 1's `.shared` end stamp (1332504097774000) by 458 ns, i.e. it is run 1's second encoder. Run 0 was current. Not investigated further (5-minute budget); could be a missing dependency between the stage-boundary write and the blit resolve for private storage, or a resolve-cache behaviour. It is not "all zeros".

---

## 2. Timing tables

All values in milliseconds. 20 warm-up + 100 measured dispatches per mode, one dispatch per command buffer, wait each time. p90 = nearest-rank. rsd = sample standard deviation / mean.

### Run 1 (mode order none → cbOnly → relaxed → precise) — full statistics

| mode | clock | median | p90 | min | max | rsd | implied GB/s (median) |
|---|---|---|---|---|---|---|---|
| none | wall | 0.5451 | 0.7683 | 0.5228 | 1.9682 | 48.07 % | — |
| none | cbGPU (feedback) | 0.3627 | 0.3696 | 0.3195 | 0.7415 | 12.15 % | 370.0 |
| cbOnly | wall | 0.5588 | 0.6139 | 0.5288 | 1.9643 | 43.39 % | — |
| cbOnly | cbGPU (feedback) | 0.3722 | 0.3758 | 0.3238 | 0.3921 | 3.26 % | 360.6 |
| cbOnly | cbHeap (cb0→cb3) | 0.3696 | 0.3734 | 0.3088 | 0.3898 | 3.61 % | 363.1 |
| relaxed | wall | 0.5600 | 0.6138 | 0.5398 | 2.0798 | 44.06 % | — |
| relaxed | cbGPU (feedback) | 0.3712 | 0.3758 | 0.3230 | 0.3872 | 3.07 % | 361.6 |
| relaxed | cbHeap (cb0→cb3) | 0.3688 | 0.3732 | 0.2975 | 0.3848 | 4.06 % | 363.9 |
| relaxed | **kernel (enc1→enc2)** | **0.3605** | 0.3646 | 0.3201 | 0.3771 | **2.80 %** | **372.3** |
| precise | wall | 0.5668 | 0.6393 | 0.5359 | 1.8558 | 45.93 % | — |
| precise | cbGPU (feedback) | 0.3706 | 0.3796 | 0.3222 | 0.4414 | 3.92 % | 362.2 |
| precise | cbHeap (cb0→cb3) | 0.3681 | 0.3756 | 0.2949 | 0.4391 | 4.53 % | 364.6 |
| precise | **kernel (enc1→enc2)** | **0.3614** | 0.3679 | 0.3188 | 0.4015 | **3.38 %** | **371.4** |

GB/s = 134,217,728 bytes / median seconds / 1e9.

### Medians across the three runs (run 3 executed modes in reverse order)

| clock | mode | run 1 | run 2 | run 3 (reversed) |
|---|---|---|---|---|
| kernel | relaxed | 0.3605 | 0.3610 | 0.3614 |
| kernel | precise | 0.3614 | 0.3609 | 0.3599 |
| cbGPU | none | 0.3627 | 0.3617 | 0.3631 |
| cbGPU | cbOnly | 0.3722 | 0.3744 | 0.3750 |
| cbGPU | relaxed | 0.3712 | 0.3737 | 0.3727 |
| cbGPU | precise | 0.3706 | 0.3705 | 0.3698 |
| cbHeap | cbOnly | 0.3696 | 0.3719 | 0.3721 |
| cbHeap | relaxed | 0.3688 | 0.3711 | 0.3703 |
| cbHeap | precise | 0.3681 | 0.3679 | 0.3672 |
| wall | none | 0.5451 | 0.5551 | 0.5425 |
| wall | cbOnly | 0.5588 | 0.5778 | 0.5738 |
| wall | relaxed | 0.5600 | 0.5815 | 0.5807 |
| wall | precise | 0.5668 | 0.5780 | 0.5825 |

### Monotonicity / validity counts (per 100 measured samples)

| run | mode | zero stamps (written slots) | cb0>enc1 | enc1>enc2 | enc2>cb3 | cb0>cb3 | per-iter vs full-heap resolve mismatches |
|---|---|---|---|---|---|---|---|
| 1 | relaxed | 0 | 6 | 0 | 0 | 0 | 0 |
| 1 | precise | 0 | 10 | 0 | 0 | 0 | 0 |
| 2 | relaxed | 0 | 8 | 0 | 0 | 0 | 0 |
| 2 | precise | 0 | 14 | 0 | 0 | 0 | 0 |
| 3 | relaxed | 0 | 12 | 0 | 0 | 0 | 0 |
| 3 | precise | 0 | 12 | 0 | 0 | 0 | 0 |
| 1–3 | cbOnly | 0 | n/a | n/a | n/a | 0 | 0 |

Runs 2 and 3 show larger kernel/cbGPU rsd (5–44 %) than run 1 because of a handful of outliers (kernel max 0.52–0.91 ms, cbGPU max up to 2.04 ms); medians and p90 are unaffected. The desktop session was live during all runs; no attempt was made to quiesce other GPU clients.

---

## 3. Exact build and run commands

```sh
cd /private/tmp/claude-501/-Users-goftin-dev-gsuite-VSK-future-VectorAccelerate/0ef5f4bb-eebb-4e08-b69c-66ef6f7d656f/scratchpad/mtl4-timestamp-spike

# build
swiftc -O -framework Metal -framework Foundation mtl4_timestamp_spike.swift -o mtl4_timestamp_spike

# API-validation smoke run (short)
MTL_DEBUG_LAYER=1 ./mtl4_timestamp_spike --warmup 3 --iters 5 > validation-run.txt 2>&1

# measured runs
./mtl4_timestamp_spike --warmup 20 --iters 100 > full-run-1.txt 2>&1
./mtl4_timestamp_spike --warmup 20 --iters 100 > full-run-2.txt 2>&1
./mtl4_timestamp_spike --warmup 20 --iters 100 --reverse > full-run-3-reverse.txt 2>&1
```

Flags: `--warmup N` (default 20), `--iters N` (default 100), `--reverse` (run modes precise → relaxed → cbOnly → none).

Note: `full-run-1.txt` was produced by the build immediately before the `--reverse` flag, the per-pair violation breakdown, and the legacy blit/`.private` additions were added; its Metal 4 measurement code path is otherwise identical. Runs 2 and 3 are from the final source.

---

## 4. Caveats and surprises

1. **Swift API names differ from the ObjC selectors** in ways that cost compile iterations: `writeTimestamp(granularity:counterHeap:index:)` and `writeTimestamp(counterHeap:index:)` (not `into:`); `dispatchThreads(threadsPerGrid:threadsPerThreadgroup:)`; `resolveCounterRange(_: Range<Int>) throws -> Data?`; `device.size(ofCounterHeapEntry:)`; `device.sampleTimestamps() -> (cpu:, gpu:)`; `MTL4ArgumentTable.setAddress(_:index:)`; `queue.commit([cb], options:)` array overload exists.
2. **`MTL4CommandBuffer` has no GPU start/end time.** Command-buffer-level timing on the Metal 4 path comes only from `MTL4CommitFeedback` (async handler). The repo's `Metal4Context` does not use `MTL4CommandBuffer` at all today; it uses a legacy `MTLCommandQueue`/`MTLCommandBuffer` and reads `gpuStartTime/gpuEndTime` there. Counter heaps cannot be written from legacy `MTLComputeCommandEncoder`, so heap instrumentation requires moving that code path to `MTL4CommandBuffer.makeComputeCommandEncoder()`.
3. **CB-level "before" stamp can post after the encoder's "before" stamp** (6–14 % of samples, up to ~21 µs). Deltas must pair like with like.
4. **Any heap timestamp in a command buffer adds ~7–13 µs to the command buffer's GPU span** (2–3 % here), but does not change the enc1→enc2 kernel span. For a 20 µs kernel this would be a ~50 % bias on CB-level time and a ~0 % bias on encoder-level time — untested extrapolation, stated as arithmetic only.
5. **`.precise` vs `.relaxed` is indistinguishable with one dispatch per encoder.** The header says relaxed "may sample at command encoder boundaries" and precise "may cause splitting of command encoders"; only a multi-dispatch encoder would expose that, and it was not tested.
6. **`sampleTimestamps()` returns identical CPU and GPU values** on this machine (ratio exactly 1.0), and the legacy sample-buffer timestamps are in nanoseconds while the MTL4 heap timestamps are 24 MHz ticks. Both appear to share the uptime origin (observation only).
7. **`.atDispatchBoundary` is unsupported** on M3 Max under macOS 26.5.2; only `.atStageBoundary` is. The legacy stage-boundary path returns valid non-zero nanosecond timestamps through CPU resolve and through blit resolve of a `.shared` sample buffer. The blit resolve of a **`.private`** sample buffer returned one-run-stale values from the second run on (see Q6) — a real oddity, but not "all zeros".
8. Wall-time rsd of 43–61 % is driven by a few 1.8–2.7 ms outliers; wall median sits ~0.18–0.21 ms above the GPU span (encode + commit + event signalling + wake-up).
9. The command buffer and allocator were reused every iteration (`allocator.reset()` after the shared-event wait, then `beginCommandBuffer` again); this passed API validation and produced no feedback errors in ~1,400 command buffers.
10. All buffers were `.storageModeShared`; `.private` buffers were not benchmarked, so the ~372 GB/s figure is for shared storage.
11. The implied bandwidth from the kernel span (371–373 GB/s) sits within the 300–400 GB/s window the task expected; the CB-level figures (360–363 GB/s with timestamps, 370 GB/s without) are lower purely because of the extra CB overhead, not the kernel.

---

## 5. Recommendation

Instrument per-encoder timestamps inside `Metal4Context` as an opt-in, not as a harness-only workaround, because the data shows the encoder-level heap span is the only clock that measures the kernel itself: it is stable to ≤ 0.4 % across runs and granularities (rsd 2.8–3.4 % in a clean run), unaffected by whether timestamps are present, and free of the constant ~10 µs command-buffer overhead and the ~0.2 ms, 45–60 %-rsd wall overhead that the two alternatives carry; for the library's small kernels those constants would dominate a CB-level measurement even with strict one-kernel-per-command-buffer isolation, and CB-level isolation cannot attribute time inside fused/multi-encoder pipelines at all. The design that the evidence supports is: one `MTL4CounterHeap(.timestamp)` per context sized as a ring of 2 slots per encoder, `writeTimestamp(granularity: .relaxed, …)` immediately before and after each encoder's dispatches (`.precise` bought nothing measurable here and the header warns it may split encoders), CPU `resolveCounterRange` after the existing shared-event wait, conversion via `queryTimestampFrequency()` (24 MHz on this machine), and the result exposed next to the existing `GPUTimingInfo` with the pairing rule "never subtract a CB-level stamp from an encoder-level stamp". The hard precondition is that `Metal4Context` today encodes on a legacy `MTLCommandQueue`/`MTLCommandBuffer` (lines 128/233), where the counter-heap API does not exist; until that path is moved to `MTL4CommandBuffer`/`MTL4ComputeCommandEncoder` (with `MTL4CommitFeedback` replacing `gpuStartTime/gpuEndTime`), the harness should keep isolating one kernel per command buffer and report CB GPU time with a documented, measured bias of roughly +2 µs (no timestamps) to +10 µs (with timestamps) relative to the kernel span, and treat wall time as a sanity check only.
