# Borůvka candidate bounds and edge validity

Borůvka retains a 2N candidate-record allocation. For a valid undirected graph, each
active component emits one outgoing edge per round. After the CPU merges all selected
edges, every remaining active component contains at least two previously active
components. Isolated components emit nothing. The number of emitted candidates across
completed rounds is therefore bounded by a finite geometric sum, strictly below 2N.
Duplicate A→B/B→A candidates are included in that count and removed by CPU union-find.

This proof requires complete merging between rounds. Repeated fusion calls without
merging, or a reused counter that was not reset, can exhaust the allocation. The shader
now enforces capacity independently of the proof:

- `BoruvkaParams` remains four UInt32 words (16 bytes). The former padding word is now
  `candidateCapacity`, measured in complete 12-byte `MSTEdge` records. Both Swift paths
  derive it from the actual candidate buffer length. The internal initializer requires
  an explicit capacity; raw shader callers must populate the fourth word too.
- Atomic compare/exchange reserves unique indices strictly below capacity. An attempted
  append to a full buffer sets the counter to `capacity + 1` and performs no edge write.
  Subsequent attempts cannot increase this normal overflow marker or wrap it to zero.
- Effective capacity is capped at `UInt32.max - 1`, leaving room for an overflow marker.
  An already corrupted counter above capacity is never converted into an address or
  decreased into the valid range.
- Standalone execution checks the count before dispatch and after GPU completion, before
  indexing candidate records. Overflow throws `VectorError.computeFailed`; a truncated
  iteration is not merged into a plausible partial result.
- Fusion callers must initialize component IDs and zero `edgeCount` before starting.
  After GPU completion, call `workBuffers.readCandidateCount()`, which throws on
  overflow, then merge every new candidate and flatten component IDs before the next
  round. `workBuffers.candidateCapacity` exposes the usable record count. Reset/restart
  or otherwise recover explicitly from overflow; do not use its prefix as a full round.

Allocation requires positive N with `2N < UInt32.max` and a candidate byte size within
the device's maximum buffer length. The checks precede allocation and use a bounded
integer product. Standalone empty/single-point result handling remains unchanged.
The collector checks source and target IDs against N before indexing component IDs.

A missing edge is represented by **`UInt32.max` endpoints**, independently of its weight.
All five find-min kernels and component reduction retain genuine +infinity-weight edges.
Reduction clears the source and target of non-representatives, preventing reused-buffer
contents from being mistaken for an edge. Finite weights still outrank infinite weights;
existing strict-comparison/scan-order ties are retained. NaN candidate weights are not
selected, using a bit check that survives fast math. This does not establish new NaN
semantics for input embeddings/core distances or alter their existing `max` arithmetic.

A valid infinite edge can connect components, and an MST using one reports infinite total
weight. Borůvka's distance calculation still uses its existing FP32 squared accumulator
before the square root; overflow can therefore produce infinite weights even where a
more precise distance would be finite. This slice repairs edge validity, not that range
limit. Truly absent usable outgoing edges retain the existing forest/early-stop behavior.
No deterministic edge order is promised: atomic candidate order and equal-weight tree
choices remain outside this slice.

Raw callers still provide enough storage for every bound buffer and the declared
capacity, consistent N/D/layouts, initialized counters/component IDs, and synchronization.
Shaders cannot inspect actual allocation sizes. The new count reader requires completed
GPU work and CPU-readable storage; it is not a concurrent progress counter.

`Hardening/BoruvkaBoundsTests.swift` exercises both shader compilation paths, all dimension
variants, finite/infinite/no-edge selection, poisoned non-representative endpoints,
zero/short/exact capacities, concurrent reservations, saturation at UInt32.max, invalid
endpoints, and over-dispatch. Fusion tests exercise repeated unmerged rounds and a
32-point hierarchical fixture with complete CPU merging, verifying per-round halving,
cumulative candidates below 2N, and an independently known 1-D MST weight. Public tests
cover infinite core distances, squared-accumulator overflow, and early allocation-count
rejection. These correctness checks establish no performance improvement.
