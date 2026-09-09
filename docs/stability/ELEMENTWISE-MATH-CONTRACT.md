# Elementwise intrinsic-selection contract

VA3-023 is resolved by documenting the existing behavior. `useFastMath` is an
intrinsic-selection flag, not a library compilation mode. Its public name, default
`false`, UInt8 shader representation, buffer layout and numerical behavior are retained.

Both `elementwise_operation_kernel` and `elementwise_inplace_kernel` call
`perform_elementwise_operation` in `DataTransformations.metal`:

| Operations | `false` / zero | `true` / nonzero |
|---|---|---|
| divide | `a / b` | `fast::divide(a, b)` |
| power, powerScalar | `pow(a, b)` | `fast::pow(a, b)` |
| sqrt | `sqrt(a)` | `fast::sqrt(a)` |
| reciprocal | `1.0f / a` | `fast::divide(1.0f, a)` |
| exp | `exp(a)` | `fast::exp(a)` |
| log | `log(a)` | `fast::log(a)` |
| All other elementwise operations | Same expression regardless of flag | Same expression regardless of flag |

Swift Bool arguments become zero or one in `ElementwiseParameters`. Array and
VectorProtocol overloads forward the flag to those parameters; raw encode and
standalone buffer APIs consume the supplied parameters. The choice occurs inside
the existing pipeline and does not create or select another compiled library.

The package's default shader library uses fast math in both build configurations:

- Debug loads the plugin-produced `debug.metallib`. The checked-in plugin flags
  do not disable fast math. A local Metal 32023.883 driver expansion with the
  configured flags confirms `-ffast-math`, `-ffinite-math-only` and
  `-fmetal-math-fp32-functions=fast`.
- Release compiles the bundled sources through
  `KernelContext.makeLibraryFromBundleSources`, which sets `mathMode = .fast`.
- On the stock path, `ElementwiseKernel` obtains this library through
  `Metal4ShaderCompiler.getDefaultLibrary`. The separate
  `Metal4CompilerConfiguration.fastMathEnabled` setting controls explicit
  `makeLibrary(source:)` calls; configuration alone does not rebuild the packaged
  elementwise library. Callers can override the compiler's default library using
  `setDefaultLibrary(_:)` or `compileDefaultLibrary(source:)`; the latter compiles
  through the configured `makeLibrary(source:)`. Such replacements have their own
  compilation settings. An existing `ElementwiseKernel` retains the pipelines it
  created at initialization; replacing the library does not rebuild those pipelines.

Consequently, on the stock library path, `useFastMath: false` does not establish strict IEEE semantics,
correct rounding, full FP32 intermediate range, nonfinite propagation, signed-zero
preservation, subnormal preservation, or bitwise agreement across devices and
toolchains. The compiler may produce identical instructions for both branches.
No relative accuracy or performance improvement is promised by either value.
Changing the flag does not validate operation domains, such as positive inputs for
logarithms. Existing distance-specific range contracts do not extend to this API.

This is a documentation correction, with no new precise pipeline or arithmetic
change. The operation table and compilation path are verified against the sources
and local compiler invocation. Existing `ElementwiseKernelTests` cover ordinary
finite arithmetic using the default flag; they do not prove a numerical difference
between flag values or strict semantics. Debug and release verification results
are recorded in audit remediation slice 25.
