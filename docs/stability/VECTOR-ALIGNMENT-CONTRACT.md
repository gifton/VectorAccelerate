# Vector storage alignment

Scalar-backed shader buffers use `packed_float4`, `packed_uint4`, and `packed_char4`
pointer views for four-component loads and stores. Register accumulators and arithmetic
continue to use ordinary vectors. This preserves the four-element storage size while
requiring only the underlying scalar's alignment: four bytes for float/uint and one byte
for char. See Apple's [Metal Shading Language Specification, §2.2.3](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

For example, a float row beginning at element 7 is valid scalar storage but does not meet
`float4`'s 16-byte alignment requirement. Its complete four-element blocks can be read
through a `device const packed_float4*`. The same rule applies to strided output rows,
projection/codebook weights, byte-packed neural codes, and scalar threadgroup arrays.

The conversion covers both general and specialized scalar-backed paths, including
existing dimension-guarded paths. The original buffer parameter types, element sizes,
row strides, dispatch geometry, register arithmetic, and shared-array allocations remain
unchanged. APIs explicitly accepting vector-typed buffers (such as SoA or vectorized
batch operations) retain their vector-storage alignment/layout requirements. Packed views
do not make arbitrary byte-misaligned float buffers valid.

Alignment and bounds are separate requirements. A packed four-component load still needs
four valid elements. The generic transposed neural decoder now handles its final one to
three outputs using scalar weight reads after the final barrier, avoiding the former
full-block read beyond the last weight row. Complete output blocks keep the vector path.
The scalar tail may round differently from the unrolled dual-accumulator path; it retains
the library's ordinary FP32 precision policy. Existing latent-capacity and routing limits
remain in force.

Three specialized neural output stores also received the missing VA3-018 offset promotion:
`vectorIdx * INPUT_DIM` is evaluated in `ulong` before adding it to the output pointer.
The address-width contract remains in [INDEX-WIDTH-CONTRACT.md](INDEX-WIDTH-CONTRACT.md).

`Hardening/VectorAlignmentTests.swift` checks:

- The shader corpus with Metal's `-Werror=cast-align`. Because that warning deliberately
  accepts explicit C++ reinterpret casts, temporary diagnostic copies express those
  conversions as equivalent C-style casts. A deliberately invalid cast verifies that the
  diagnostic catches alignment increases. Production shader source is not rewritten by
  this test.
- Real GPU results through both library compilation paths for strided distance rows,
  paired reductions, odd-stride normalization including bit-preserving subnormal copies,
  PQ subvectors, neural byte/weight/output rows, tiled Hamming, and tiled Minkowski.
- Every full-block weight load in the generic neural decoder, instrumented in a test-only
  library to check its footprint before dereferencing. This catches invalid requests in
  otherwise unused lanes that numerical parity and optimized shader validation can miss.

`IndexWidthTests` covers the three specialized store offsets and the new wide scalar-tail
output offset. Compiler diagnostics establish pointer conversion alignment; they do not
establish buffer capacity or arbitrary caller offsets. These tests and the normal debug/
release gates make no throughput guarantee, and no performance improvement is claimed.
