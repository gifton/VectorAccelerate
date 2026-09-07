# IVF candidate capacity and CSR ordering

`IVFGPUCandidateBuilderKernel.buildCandidates` returns query-ordered CSR ranges.
Within each query, candidates follow the first occurrence of each valid probed list,
then entry order within that list. Duplicate probes and sentinel/out-of-range list IDs
are ignored. The existing maximum of 64 probes remains enforced on the public host path.

The fused shader allocates whole query segments atomically. Atomic reservation order is
not query order: its raw per-query start/count descriptors are **not CSR offsets**.
The host validates the completed descriptors, builds prefix offsets in query order,
and uses GPU blits to reorder segments when necessary. Already ordered nonempty segments
retain their original buffers. Empty queries receive correct empty CSR ranges.
This fixes the previous conversion that treated unordered starts as CSR boundaries,
which could assign one query another query's candidates or return reversed ranges.

The fused shader's parameter at buffer(7) is now `IVFCandidateBuildParams`, four UInt32
words (16 bytes), with `total_candidates` holding output capacity in records. Raw callers
must populate this fourth word; the former count-parameter padding is no longer ignored.
Capacity applies to both the IVF-index and query-ID arrays. Buffer binding indices stay
unchanged. Host dispatch derives capacity from the smaller actual output allocation,
limited by the requested estimate and UInt32.max-1.

The counter must be initialized to zero. A CAS loop reserves a complete query segment
only when it fits. Failure publishes UInt32.max/0 as that query's start/count and marks
at least capacity+1 in the global counter, without writing candidates. Subsequent
reservations cannot wrap the counter or erase the overflow marker. Candidate counts use
wide arithmetic before reservation. Empty successful segments write no candidates.
Read counts/descriptors only after GPU completion; reservations precede their writes.

`maxCandidatesPerQuery` is an allocation hint, not a recall/truncation limit. On overflow,
the host discards the fused attempt and reruns exact count/prefix/build. A hint too large
for the count/device limit routes directly to that exact path without multiplying an
unbounded hint. No truncated fused prefix is returned as a complete result.

The three-pass prefix sum uses a wide accumulator and saturates output at UInt32.max;
the host rejects that marker before allocating/building candidates. Supported totals
are below UInt32.max and within device/allocation limits. Exact output and metadata
allocations verify actual buffer length before CPU or GPU writes. This matters because
the current pool can cap requests at its largest 64 MiB bucket. Such an undersized exact
allocation throws an allocation error; this slice does not repair the pool globally.

Public counts and hints must be nonnegative; query/list counts must fit below UInt32.max.
Input buffers must cover the declared probe and CSR-offset arrays. Zero queries, zero
probes, or zero lists return an empty result before dispatch. Raw callers still supply
valid, stable, monotonic CSR list offsets, enough storage for every bound array and the
declared capacity, valid count/layout relationships, and synchronization. This is not
a general validator for malformed raw CSR or concurrent mutation.

`Hardening/IVFCandidateBoundsTests.swift` covers both shader compilation paths with
logical capacity canaries, whole-query reservations, duplicate/sentinel probes,
zero/exact/short capacity, saturated counters, prefix overflow, and over-dispatch.
Public tests cover skewed lists with underestimated/zero/oversized hints, fused and
three-pass batches, empty/invalid inputs, and physical allocation limits. A deliberately
permuted segment fixture forces CSR reordering independently of GPU scheduling and
checks rejection of overlap, missing records, and out-of-range descriptors.

These changes make no speedup claim. Recovery adds work when estimates are too small,
and unordered fused segments require additional output storage and a blit submission.
UMAP's embedding race and floating-point atomic accumulation policy remain separate
parts of VA3-019; this IVF slice does not close that entire finding.
