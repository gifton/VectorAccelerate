# Sparse c-TF-IDF bounds

`SparseLogTFIDFKernel` retains its existing scoring formula and host routing: the
vectorized shader is selected only when `nnz >= 16` and `nnz % 4 == 0`. Other counts
use the scalar shader. This fix does not change ranking comparisons or numerical policy.

Direct dispatch of `sparse_ctfidf_vectorized_kernel` also handles a final one to three
entries safely. Complete groups use the original vector loads, gathers, and stores;
the final group reads term IDs, frequencies, and corpus frequencies and writes scores
only for live entries. Masking an output after gathering an unused ID would not suffice.
The group base is computed in `ulong` before multiplication, and extra threads return.

The vector-typed buffer ABI is retained: term-index, term-frequency, and score buffer
bases must be 16-byte aligned, and each binding must expose at least 16 bytes even for
`nnz < 4` (Metal API validation enforces the argument's minimum size). Above that minimum,
a partial tail does not require padding to the next multiple of four. The scalar shader
and ordinary host routing remain suitable for smaller scalar allocations.

Top-K accepts K=0 as an empty operation:

- `topKPerCluster` returns one empty list per input cluster, preserves `clusterCount`
  and `k`, and reports zero execution time and throughput without allocating buffers.
- `encodeTopK` returns zero dispatched threadgroups without touching the encoder when
  K or the number of clusters is zero.
- `ctfidf_topk_per_cluster_kernel` returns before reading cluster offsets or forming
  output pointers when K=0. It never evaluates `topK - 1` for that case.

The throwing API rejects K outside `0...UInt32.max` before empty-input handling.
The nonthrowing encoder requires both K and `numClusters` to fit UInt32 and enforces
that with preconditions. For positive K, clusters with fewer terms retain the existing
`UInt32.max` / negative-infinity padding; the public result omits those entries.

These changes close VA3-025's partial-tail and zero-K defects. They do not add general
validation of caller-owned sparse data. Live term IDs must index `corpusFreqs`; term-ID
and frequency arrays must have matching lengths; cluster offsets must be monotone,
have `numClusters + 1` entries, and stay within the score/term buffers. Count conversions,
allocation sizes, buffer capacities, and aliasing/synchronization remain caller/API
requirements. This slice does not widen count/ID limits or promise allocations for
all mathematically representable shapes. Nonfinite-score ordering remains unchanged.

`Hardening/SparseTFIDFBoundsTests.swift` exercises both plugin and runtime libraries,
partial tails, exact allocations above the vector ABI minimum, output canaries,
K=0/1/oversized K, empty clusters, extra dispatched threads, and public empty-result
behavior. Test-only shader instrumentation intercepts poisoned unused-ID gathers and
entry into zero-K cluster reads before an invalid dereference can occur. These tests
establish bounds behavior, not a throughput improvement.
