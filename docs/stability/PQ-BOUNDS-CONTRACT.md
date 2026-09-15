# Product quantization bounds

Product quantization retains UInt8 codes and `1 <= K <= 256` centroids per subspace.
Centroid 255 is valid when K=256. The configuration initializer retains its existing
K<=256 precondition; training, encoding, and ADC execution additionally reject
nonpositive K with `VectorError.invalidInput` before their allocations or dispatches.
The combined training/encoding API validates K before staging input data too.

ADC's complete `M × K` FP32 distance table must fit **32 KB (8192 entries)**. This is an
ADC-only limit: training and encoding can still use larger tables/models. There is no
silent reduction of M or K and no new fallback for an oversized ADC request.

`computeDistances` checks M against `8192 / K` before multiplying, allocating buffers,
converting parameters to UInt32, or opening an encoder. It rounds the dynamic shared
memory binding up to a multiple of 16 bytes and verifies that it fits the device budget
after subtracting the pipeline's static threadgroup usage. The logical table buffer
still has exactly `M × K` floats. Apple's
[`setThreadgroupMemoryLength` documentation](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:))
requires the 16-byte binding granularity. Small or odd table sizes remain valid.

Direct shader dispatch has defensive behavior for these bounds:

- `pq_assignment_or_encoding` fills each live code with `0xff` for unsupported K before
  reading vectors or codebooks. This marks an unsupported *configuration*; it does not
  reserve code 255 in a valid K=256 model. Raw callers must validate the configuration
  before treating those output bytes as a trained/encoded model.
- `pq_train_update_accumulate` leaves accumulators/counts unchanged for unsupported K
  and skips individual assignments `>= K` before forming accumulator pointers. This
  prevents an invalid code from updating a neighboring subspace or crossing the buffer.
- `pq_compute_distances_adc` writes NaN for every dispatched live vector when K is
  unsupported, M is zero, or the table exceeds 8192 entries. The rejection is uniform
  across the group and precedes all shared-memory loading and the barrier. Its bounds
  division occurs only after K is known positive, avoiding both divide-by-zero and
  overflow of an unchecked `M × K` product.
- With a valid ADC table, a code `>= K` produces NaN for that vector before the lookup.
  This return occurs after the shared-table barrier. Other vectors continue normally;
  extra dispatched threads do not write beyond N.

These checks close VA3-026's byte-code and shared-table bounds. Raw callers still must
bind enough device/shared storage, use the required binding alignment and dispatch
layout, supply consistent dimension/subspace strides, and synchronize resources. Shaders
cannot inspect the dynamic shared-memory binding length. The precompute/finalize kernels
retain their caller-owned buffer/configuration contracts. This slice does not add general
shape/count/capacity validation, validate model/encoded-buffer compatibility, repair CPU
decode index checks, change squared-distance range limits, or change training's atomic
accumulation/determinism policy.

`Hardening/PQBoundsTests.swift` covers both library compilation paths, K=0/1/255/256/257,
byte 255, invalid codes within/beyond another subspace, table sizes below/at/above the
cap, a UInt32-wrapping product, and ragged/extra threads. Source-derived instrumentation
intercepts invalid shared accesses before dereference in the failing baseline. Real
library tests verify NaN fills, output canaries, and at-cap results. Host tests verify
early rejection (including an Int.max M), small binding alignment, and retained
training/encoding above the ADC cap. Validation and correctness tests establish no
performance improvement or new reproducibility guarantee.
