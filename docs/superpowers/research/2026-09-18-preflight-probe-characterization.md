# Preflight probe characterization (GPU copy probe and CPU copy probe)

Date: 2026-09-18. Branch `gifton/harness-sp0`, SP0 Task 15 review rounds 1 and 2.
This note is the evidence that the comments on `GPUProbe.copyGBps` and `PreflightProbe.cpuCopyGBps` cite. Every number is
as printed by the scripts described below; nothing is rounded further than the scripts rounded it.

## Why this exists

Spec §4.10 reads a CPU copy probe and a GPU copy probe before and after every run and calls the run `quiet` only if neither
moved by more than 5 %. Baseline capture refuses artifacts that are not quiet. Both probes turned out to have a repeatability
comparable to or worse than that gate, for the same reason: a post-idle ramp that the reading landed on.

## Machine and conditions

| | |
|---|---|
| Machine | Apple M3 Max, `Mac15,9`, arm64, 16 CPU cores (12 performance, 4 efficiency), 40 GPU cores, 51539607552 bytes memory |
| OS / toolchain | os26.5.2 build 25F84, Swift 6.3.3, Xcode 26.6 (17F113), Metal toolchain 32023.883 |
| Power / thermal | AC, low-power mode off, thermal state nominal at the start and end of every recorded run |
| Build | release (`swiftc -O` for the throwaway tools, `swift build -c release` for the bench executable) |

**The machine was never idle.** Throughout, WindowServer ran at about 40 % of a core and a desktop application at about
50–65 %; load average 4.2–7.7. During much of the CPU-probe work another session on the same machine was compiling and
running tests (`swift-frontend`, `dsymutil`, `xctest` at 60–100 % CPU, in bursts). Where a table depends on that, the
background was sampled every 0.5 s (`ps -A -o %cpu,comm`, processes at ≥ 60 % CPU other than the tool) and is reported.
No measurement here was taken on a verified idle machine, and none could be.

## Method

Two throwaway single-file tools, compiled with `swiftc -O` outside the repository, replicate the probes exactly.

**GPU tool.** Same kernel source, buffer sizes, storage mode and threadgroup size as `GPUProbe` (128 MiB `float4` copy,
`.storageModeShared`, 256 threads per threadgroup, one command buffer per dispatch, `commit` + `waitUntilCompleted`). One
sample is `gpuEndTime − gpuStartTime` of one dispatch, in µs. Recorded sets:
- 3 process launches × 3 series of 300 consecutive dispatches: from process start, after a 2.0 s sleep, after a 0.2 s sleep.
- Idle-gap sweep: gaps 0.1, 0.25, 0.5, 1, 2, 5 s × 6 repetitions, 100 dispatches after each gap.
- Ramp study with wall stamps: gaps 0.25, 1, 5 s × 12 repetitions, 200 dispatches each, `ContinuousClock` time at the end of each dispatch.
Total 81 series, 13,500 dispatches.

**CPU tool.** Same sizes, alignment, fill and clock as `PreflightProbe.cpuCopyGBps` (2 × 256 MiB, 16,384-byte alignment,
`memset` both buffers, `ContinuousClock.measure { memcpy }`). One *call* is a fresh allocation plus 40 timed copies (the old
probe timed 3 and returned the fastest). Scenarios, 12 calls each per recording, two recordings: back to back; after a 0.3 s
sleep; after a 2.0 s sleep; after a burst of the six subprocesses the fingerprint gather launches (`git rev-parse`,
`git status`, `pmset` ×2, `swift --version`, `xcrun metal --version`); after a 0.15 s semaphore block; after about 0.12 s
blocked in 100 × `waitUntilCompleted`. Process-start calls: 10 launches each of "first thing in the process" and "after the
subprocess burst", per recording. Total 184 calls. Also two continuous series of 3,000 copies from one allocation, and two
sets of readings under the instrument-check protocol (ten readings 0.3 s apart per trial, fresh allocation per reading,
warm-up as in the fixed probe, raw copy times dumped so that every estimator is evaluated on identical data): 8 trials × 150
copies and 20 trials × 200 copies.

**Definitions.**
- *Settled level* of a series: median of its dispatches 51–end (GPU, 100-dispatch series), 101–end (GPU, longer series), or copies 11–40 (CPU call).
- *GPU ramp end*: scanning from the start, the last rolling median-of-5 that exceeds 1.08 × the settled level, stopping after 10 consecutive clean windows; reported as the number of dispatches consumed.
- *CPU ramp end*: number of leading copies before the first 3 consecutive copies within 5 % (slower side) of the call's settled median.
- *Plateau ratio*: median of the first 5 dispatches over the settled level (GPU); old best-of-3 reading over the settled rate (CPU).
- *Window deviation*: for every window of m consecutive settled samples, |estimator(window) − estimator(whole steady state)| / the latter.
- *Instrument-check statistic*: (max − min) / median of ten readings.

## GPU copy probe

Settled level ≈ 761–771 µs per dispatch ≈ 350 GB/s counting read plus write. It moved between recordings: 761.2 µs in the
sweep, 767.4–781.5 µs across the 36 series of the ramp study.

### A ramp exists, and idle resets it

Medians in µs, three launches (launches 2 and 3 started within a second of the previous launch's last series):

| Series | 1–5 | 6–15 | 16–50 | 51–150 | 151–300 |
|---|---|---|---|---|---|
| launch 1, from process start | 1436.5 | 780.9 | 763.6 | 766.6 | 766.1 |
| launch 1, after 2.0 s idle | 1439.5 | 801.8 | 762.4 | 760.7 | 766.7 |
| launch 1, after 0.2 s idle | 768.4 | 769.5 | 761.2 | 767.1 | 764.6 |
| launch 2, from process start | 772.2 | 767.3 | 760.3 | 763.5 | 765.7 |
| launch 2, after 2.0 s idle | 1542.5 | 794.3 | 765.8 | 758.7 | 762.8 |
| launch 2, after 0.2 s idle | 763.2 | 765.8 | 762.0 | 761.2 | 766.9 |
| launch 3, from process start | 764.2 | 763.2 | 765.9 | 765.8 | 761.8 |
| launch 3, after 2.0 s idle | 887.4 | 853.8 | 760.3 | 759.6 | 761.3 |
| launch 3, after 0.2 s idle | 765.7 | 763.0 | 758.5 | 760.3 | 762.3 |

The old reading was the median of dispatches 6–15. Launch 3 after idle gives 853.8 µs = 314.4 GB/s; the run that first
exposed the problem read 313.8 GB/s at its start and 346.7 GB/s at its end.

### Idle-gap sensitivity, plateau ratio, ramp length

Sweep plus ramp study, 72 series:

| gap (s) | series | started slow | plateau ratio (median) | ramp end, dispatches (median / max) |
|---|---|---|---|---|
| 0.10 | 6 | 1 | 1.87 | 21.0 / 21 |
| 0.25 | 18 | 7 | 1.61 | 19.0 / 41 |
| 0.50 | 6 | 5 | 1.88 | 27.0 / 32 |
| 1.00 | 18 | 17 | 1.89 | 19.0 / 34 |
| 2.00 | 6 | 5 | 1.93 | 19.0 / 21 |
| 5.00 | 18 | 17 | 1.88 | 11.0 / 25 |

Over all 81 series: 56 started slow; plateau ratio min 1.08, median 1.88, max 2.40; ramp end median 10, p90 25, max 41
dispatches (series that did not ramp count as 0). In the 36 wall-stamped series, 25 ramped and the ramp ended after a
median of 32.0 ms and at most 55.5 ms of wall time. Round 1 quoted "14–62 ms" from a looser metric (rolling median-of-9
more than 5 % slow), which also caught two mid-series disturbances at dispatches 114 and 166; the median-of-5 metric above
is the one the code comment uses.

### Steady state

About 1 dispatch in 7 reads ≈ 1110 µs (1.45 ×), at a cadence that matches a 120 Hz display (≈ 8.3 ms against ≈ 1.2 ms of
wall time per dispatch); mean rate of dispatches above 1.2 × settled 4.8 %, 14 % in the worst series; occasional
≈ 1520–1620 µs. Non-spike spread p1–p99: 0.970–1.099 of settled. (These three figures are from the 45 series of the
launches and the sweep; the ramp study was not included in that pass.)

### Estimator and window

Every window over the steady state (index ≥ 50), all 81 series:

| estimator | window | max dev % | p99 dev % | windows |
|---|---|---|---|---|
| median | 10 | 4.66 | 2.38 | 8721 |
| median | 20 | 2.95 | 1.68 | 7911 |
| median | 30 | 2.82 | 1.34 | 7101 |
| median | 50 | 1.85 | 1.01 | 5481 |
| p25 | 10 | 3.94 | 2.28 | 8721 |
| p25 | 50 | 1.62 | 1.18 | 5481 |
| min | 10 | 5.19 | 3.64 | 8721 |
| min | 50 | 2.26 | 1.57 | 5481 |

The median is kept. Ten samples are too few: two readings that can each be 4.66 % off cannot sit under a 5 % gate on their
difference.

### `WarmupPolicy` refuted; candidates

Simulated on the 45 series with ≥ 200 dispatches; reference = median of dispatches 101–end, disjoint from every measured
window. Error of the reading in percent, positive = reads slow:

| candidate | min | median | max | spread | \|err\| > 2.5 % |
|---|---|---|---|---|---|
| warm 5, measure 10 (original) | −2.09 | +2.02 | +64.73 | 66.83 | 21/45 |
| `WarmupPolicy.standard`, measure 10 | −1.95 | −0.10 | +41.32 | 43.27 | 4/45 |
| `WarmupPolicy.standard`, measure 50 | −1.39 | −0.35 | +0.92 | 2.31 | 0/45 |
| fixed warm 30, measure 10 | −2.14 | −0.34 | +3.23 | 5.37 | 1/45 |
| fixed warm 30, measure 50 | −1.38 | −0.31 | +0.59 | 1.97 | 0/45 |
| fixed warm 50, measure 10 | −1.71 | −0.42 | +1.61 | 3.32 | 0/45 |
| fixed warm 50, measure 30 | −1.27 | −0.36 | +0.95 | 2.23 | 0/45 |
| fixed warm 50, measure 50 | −1.41 | −0.23 | +0.85 | 2.26 | 0/45 |

`WarmupPolicy.standard` (window 8, CoV < 0.05, minimum 3, maximum 50) applied to per-dispatch GPU times fails through its
CoV rule, not its maximum. **The n = 3 mechanism:** the slow plateau is steady in itself, so at its minimum of 3 samples the
rule sees a CoV under 5 % and stops while the GPU is still slow. Three of the 45 series: window `[1607, 1462, 1547]` with a
settled level of 780 µs (next-10 median 1103 µs, +41 %), `[1570, 1460, 1461]` (settled 769, next-10 median 822),
`[1605, 1531, 1459]` (settled 771, next-10 median 995). In the other direction, one 1.45 × spike inside a window of 8 puts
the CoV near 0.15, so in steady state the stop point says nothing about the ramp: stop points ranged from 3 to 37, fourteen
of 45 at n = 3. "Policy + measure 50" passes in this table only because a median of 50 absorbs a short ramp; a stop at 3
followed by a 41-dispatch ramp would put 38 ramp samples among the 50. The runner uses the same policy for its own warm-up
and has the same weakness; it now notes on the record when a warm-up ends at the policy maximum without the CoV rule met.

**Chosen:** warm-up until both 50 dispatches and 0.1 s of wall time have passed, then the median of 50. Fifty covers the
longest ramp seen (41). The time floor (1.8 × the longest ramp's 55.5 ms) is a hedge, not a finding: no series here varied
the dispatch duration, so whether the ramp is bound by dispatch count or by wall time is not established (see "What is not
established"). If it is time-bound, the floor is what carries to a GPU that finishes 50 dispatches sooner; if it is
count-bound, the floor costs at most ≈ 0.1 s per call and changes nothing else. A median of 50 stays on the settled level
while fewer than 25 of its samples are slow. Cost of one `copyGBps()` call, measured in the bench executable: 19.8–39.7 ms
originally, 118.9–127.6 ms with the count-only warm-up, 156.7–161.4 ms with the time floor.

## CPU copy probe

Settled rate ≈ 115–119 GB/s counting read plus write (≈ 4.5 ms per 256 MiB copy); fastest single copy 127.0 GB/s.

### A ramp exists here too, and the old reading sat on it

Old reading = fastest of the first 3 copies, no warm-up. Both recordings, GB/s:

| scenario | calls | best-of-3 min / median / max | settled min / median / max | lowest best-of-3 ÷ settled | calls with best-of-3 < 0.95 × settled |
|---|---|---|---|---|---|
| process start | 20 | 78.9 / 117.0 / 125.9 | 113.4 / 116.7 / 119.1 | 0.67 | 3 |
| process start after subprocess burst | 20 | 83.0 / 117.6 / 122.6 | 114.8 / 116.3 / 119.0 | 0.71 | 3 |
| back to back | 24 | 109.1 / 120.9 / 122.9 | 113.4 / 116.2 / 118.9 | 0.94 | 1 |
| after 0.3 s idle | 24 | 69.5 / 118.1 / 122.7 | 110.5 / 115.4 / 118.3 | 0.60 | 6 |
| after 2.0 s idle | 24 | 59.3 / 115.8 / 122.2 | 113.2 / 116.3 / 117.8 | 0.52 | 8 |
| after subprocess burst | 24 | 77.4 / 118.0 / 123.4 | 115.0 / 117.7 / 119.7 | 0.66 | 10 |
| after 0.15 s semaphore block | 24 | 76.1 / 118.0 / 123.4 | 113.6 / 117.8 / 119.3 | 0.65 | 7 |
| after GPU blocked wait | 24 | 108.3 / 120.2 / 122.9 | 108.9 / 116.6 / 118.7 | 0.98 | 0 |

Ramp length over the 184 calls: 94 calls had a slow start by the definition above; ramp end at most 17 copies, p90 12; in
wall time at most 111.5 ms, p90 75.4 ms. A typical ramped call after 2.0 s idle, copies 1–14 in GB/s:
`73 78 76 78 80 76 84 84 77 83 113 111 109 106`. The blocked-in-`waitUntilCompleted` scenario does not ramp: the thread
wakes about every 1.2 ms there. The reviewer's hypothesis (slow start reading after the subprocess-heavy gather, slow end
reading after a blocked wait) is half supported: the subprocess burst does it, the GPU wait does not; plain idle does it most.
The frequency of ramped calls differed between the two recordings (the second, taken while another session was compiling,
ramped far more often), so these counts describe this machine under this load, not a rate.

Under the instrument-check protocol the old probe read, in four trials of ten readings: 59.6–77.2, 62.6–117.4, 59.1–116.1
and 56.4–89.4 GB/s; (max − min)/median 25.50 %, 74.08 %, 88.05 %, 51.04 %.

### Steady state is bimodal and alternating

Two continuous series (2,950 settled copies each; medians 4558 µs and 4543 µs): copy times fall into two modes about 5 %
apart (histogram peaks near −4…−3 % and +1…+2 % of the median). The mode indicator has lag-1 autocorrelation −0.46 and
−0.47: a fast copy is almost never followed by a fast copy (mean fast run 1.1 copies), slow runs last 1.7–1.8 copies on
average and up to 39. The median therefore sits in the valley between the modes and moves with the mix. Window deviation,
both series:

| estimator | window | max dev % (series 1 / 2) | p99 dev % (series 1 / 2) |
|---|---|---|---|
| min | 3 (old probe) | 15.57 / 19.80 | 9.52 / 9.76 |
| median | 50 | 3.47 / 3.25 | 2.54 / 2.27 |
| p25 | 50 | 2.98 / 3.23 | 2.63 / 2.87 |
| min | 50 | 3.90 / 3.18 | 3.66 / 3.04 |

### Estimator comparison on identical data, split by background load

Twenty trials of ten readings (warm-up 20 copies and 0.2 s, then 200 timed copies per reading; estimators applied to the
first m). A trial is *calm* when no process other than the tool reached 60 % CPU in any 0.5 s sample, *build* when another
session's compiler, linker or test runner did, *other* otherwise: 10 calm, 7 build, 3 other. Instrument-check statistic in
percent:

| estimator of first m copies | calm: median / max / trials ≥ 2.5 % | other: max | build: median / max / trials ≥ 2.5 % |
|---|---|---|---|
| min of 100 | 1.59 / 1.89 / 0 of 10 | 2.21 | 3.48 / 10.10 / 6 of 7 |
| min of 150 | 1.39 / 1.78 / 0 of 10 | 2.16 | 3.49 / 10.11 / 6 of 7 |
| p02 of 100 | 1.19 / 1.74 / 0 of 10 | 2.13 | 3.99 / 10.96 / 4 of 7 |
| **p05 of 100** | **1.08 / 1.76 / 0 of 10** | 2.05 | 4.48 / 10.45 / 5 of 7 |
| p05 of 150 | 0.92 / 1.82 / 0 of 10 | 1.51 | 4.64 / 10.68 / 5 of 7 |
| p05 of 50 | 1.50 / 3.14 / 1 of 10 | 1.94 | 3.39 / 11.51 / 7 of 7 |
| p10 of 100 | 1.28 / 2.11 / 0 of 10 | 2.01 | 4.99 / 10.48 / 6 of 7 |
| p25 of 100 | 1.93 / 3.36 / 2 of 10 | 2.69 | 5.25 / 11.39 / 6 of 7 |
| median of 100 | 1.96 / 3.57 / 1 of 10 | 3.14 | 6.39 / 11.91 / 6 of 7 |
| median of 150 | 1.58 / 3.39 / 2 of 10 | 3.19 | 6.05 / 11.62 / 6 of 7 |

Reading: the floor of the copy-time distribution is the repeatable quantity. Among the estimators of 100 copies, the fast
5th percentile is the most repeatable in the typical calm trial (median 1.08 % against 1.19 % for p02, 1.28 % for p10 and
1.59 % for the minimum) and within 0.02 points of the best at the worst calm trial. More copies keep improving it — p05 of
150 reads 0.92 % in the table, and the archived analysis (`task-15-probe-characterization/round2/cpucand/raw2_calm.py`,
untracked, archived beside the Task 15 report) has p05 of 200 at 0.84 % and p10 of 200 at 0.90 % — but
each extra 50 copies adds about 0.23 s to every call (50 × ≈ 4.5 ms per 256 MiB copy, the settled rate above), twice per
run, and p05 of 100 already sits under the 2.5 % instrument-check threshold in every calm trial (worst 1.76 %), so the
count stops at 100. The minimum is an extreme order statistic and the median is the worst of all here. Fifty copies are
too few for any of them. While another session compiles, every estimator spreads 2–12 %: that is the machine not being
quiet, which the gate exists to report, not an instrument defect. In the first, 8-trial set the last three trials sat
about 3–8 % below the first five for every estimator, including the minimum (124–125 GB/s dropping to 118–123), for about
50 s; its background was not sampled.

**Chosen:** warm-up until both 20 copies and 0.2 s of wall time have passed (1.8 × the longest ramp's 111.5 ms), then the
fast 5th percentile (`Stats.percentile(sorted:p: 0.05)`) of 100 timed copies. The first attempt used the 10th percentile; it
failed the instrument check once in three invocations (2.78 %) and was replaced on the table above. Cost of one
`cpuCopyGBps()` call in the bench executable: 49.1–57.3 ms before, 702.8–717.6 ms after.

## Before and after, as the quiet gate sees it

`VectorAccelerateBench run --mode smoke`, release, consecutive invocations, nothing discarded.

| set | probes | quiet | CPU drift % | GPU drift % | background |
|---|---|---|---|---|---|
| original (5 warm-up + 10; best of 3), 7 runs | both original | 2 of 7 | about 0.1, 0.6, 1.1, 7.4 and 0.2 in the five runs whose CPU readings were kept | 4.5, 8.9, 10.5, 25.70, 7.66, 63.86 and one quiet run whose value was not kept | not sampled |
| round 1, 10 runs | GPU 50 + 50 by count; CPU original | 10 of 10 | 1.32–2.81 | 0.11–1.30 | not sampled |
| reviewer's run | same | not quiet | 8.4 (122.09 → 111.85) | 0.003 (350.07 → 350.08) | not sampled |
| round 2 set A, 10 runs | both fixed | 9 of 10 | 0.01–1.28, and 6.47 in run 10 | 0.40–4.70; 4.43 in run 10 | another session's `swift-frontend` at ≥ 60 % CPU in all ten runs, `xctest` in six (runs 03–08) |
| round 2 set B, 10 runs | both fixed | 10 of 10 | 0.11–0.98 | 0.10–4.81 | none or a desktop app in runs 1, 9, 10; `swift-frontend` / `xctest` in runs 2–8 |

In the one not-quiet run of set A both probes dropped together (CPU 119.4 → 111.6 GB/s, GPU 340.0 → 324.9 GB/s) while
another session was compiling. The GPU probe's drift reached 4.4–4.8 % in three of the twenty round-2 runs, all with
another session's compiler or test runner active. Round 1, with the count-only GPU warm-up, had a maximum of 1.30 % over
its ten runs. Three things differ between the rounds and this data separates none of them: the GPU time floor, the CPU probe
(original in round 1, fixed in round 2 — the `probes` column), and the background load, which round 1 never sampled and
which is therefore unknown rather than known to differ. None of the three is established as the cause of the larger
round-2 drifts. Whether the GPU probe holds under 5 % on a loaded machine is not established by this data.

Opt-in instrument check (`VA_INSTRUMENT_CHECK=1 swift test --filter PreflightInstrumentCheckTests`), six invocations with
the chosen estimator, none discarded, in order: 1.61 %, 1.08 %, 1.45 %, 0.48 % (pass), 4.87 % (fail; readings dipped to
116.2 and 118.5 GB/s against ≈ 121.5) and 0.68 % (pass). The first five ran while another session's `swift-frontend` was at
≥ 60 % CPU in most samples, the sixth with no other process above 60 %; four passes came under the same compiler load as
the failure, so the load is not shown to be its cause.

## What is not established

- Behaviour on any other machine, OS or GPU. Both warm-up floors are counts and wall times observed on one M3 Max.
- Whether the GPU ramp is bound by dispatch count or by wall time. Every series used the same dispatch (128 MiB, `loops` 1),
  so count and elapsed time rose together and nothing here tells them apart; the 0.1 s floor hedges the time-bound case.
- Behaviour on an idle machine. None of this was measured on one.
- The cause of the ramps. The shapes are consistent with GPU and CPU power-state or core-placement ramps after idle, and
  the 2 × CPU plateau is consistent with an efficiency core, but nothing here measures frequency or placement.
- The cause of the alternating fast/slow copies.

## Rerunning

GPU tool: build a Swift file that creates the `copy_f32x4` pipeline exactly as `GPUProbe.init` does, defines
`dispatch() -> Double` as in `GPUProbe.dispatch(loops: 1)` returning `(gpuEndTime − gpuStartTime) · 1e6`, and prints
comma-separated series for: 300 dispatches from start; `Thread.sleep(2.0)` then 300; `Thread.sleep(0.2)` then 300; for each
gap in `[0.1, 0.25, 0.5, 1, 2, 5]` × 6: sleep, 100 dispatches; for each gap in `[0.25, 1, 5]` × 12: sleep, 200 dispatches
with `ContinuousClock` stamps. CPU tool: `call()` = allocate, `memset`, 40 × `clock.measure { memcpy }`, print; drive it
through the scenarios listed under Method; `long` = one allocation, 3,000 copies; protocol mode = per reading sleep 0.3 s,
allocate, warm up (`while warmed < 20 || elapsed < 0.2 s`), 200 timed copies, print raw. Sampler: a shell loop printing
`ps -A -o %cpu,comm | sort -nr | head -5` every 0.5 s with a timestamp. Analyses are the definitions above applied with
Python's `statistics.median`; percentiles use linear interpolation at rank p·(n−1), as `Stats.percentile` does. The ten-run
tables come from the `preflight` block of each run's artifact.
