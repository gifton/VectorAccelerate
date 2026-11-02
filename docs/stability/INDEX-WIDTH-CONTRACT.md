# Shader address width

VA3-018 widens buffer address calculations while retaining the existing Swift/Metal
parameter layouts. Counts, dimensions, strides, dispatch coordinates, and stored IDs
keep their current types. A product of individually valid 32-bit values can require
more than 32 bits: row 3,000,000 at stride 1,536 starts at element 4,608,000,000.

For device-buffer addresses, promote an operand **before** multiplication and preserve
the wide result through every local variable and helper argument:

```metal
const ulong rowOffset = (ulong)row * stride;
const ulong index = rowOffset + column;
output[index] = value;
```

Casting a completed 32-bit product is too late. Assigning a wide product to a `uint`
local or passing it to a `uint` index argument truncates it again. Independent products
also each need promotion; a wide batch base does not widen the multiplication in
`batchOffset + row * rowStride + column * columnStride`.

The sweep covers general and specialized distance/normalization paths, matrix and
projection paths, SoA, quantization, clustering/Borůvka, UMAP, statistics/LSE, strided
elementwise transforms, search/IVF, and sparse TF-IDF output rows. The normalization
bit-copy helper accepts a wide index as well. Existing shader parameter structures
and host bindings are unchanged. The alignment follow-up also promoted three specialized
neural vector-store offsets missed by the original sweep; their boundary regression is
part of `IndexWidthTests`.

This is an address-arithmetic contract, not a promise that every API can process every
64-bit shape. Callers still must supply valid dimensions/strides, sufficient buffers,
and layouts within device allocation limits. Existing 32-bit counts, grid coordinates,
ID/sentinel representations, and near-limit grid/loop arithmetic remain constraints.
The deprecated streaming Top-K kernel's explicit conversion of a global `ulong` ID to
`uint` remains a separate legacy API limitation. PQ table/code capacities and UMAP races are tracked independently in the audit.
Scalar-backed vector alignment is addressed in [VECTOR-ALIGNMENT-CONTRACT.md](VECTOR-ALIGNMENT-CONTRACT.md);
vector-typed buffer arguments retain their alignment requirements.

`Hardening/IndexWidthTests.swift` extracts production expressions and declarations,
compiles them into a Metal 4 fast-math probe, and compares their results against UInt64
references below, at, and above 2^32. It also checks batch products, independent strided
products, and downstream narrowing. Probes calculate offsets only: they do not allocate
or dereference multi-gigabyte buffers. Ordinary GPU regression tests and full debug and
release suites verify integration, including the plugin and runtime shader builds.
No throughput improvement or absence of a performance cost is claimed by this fix.
