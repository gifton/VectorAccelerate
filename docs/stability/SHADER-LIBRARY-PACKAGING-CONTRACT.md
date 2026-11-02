# Shader library packaging contract

**Decided:** 2026-10-31 (owner). The precompiled `default.metallib` is the shipped shader library. The runtime source compile is a fallback that fails loudly.

## Why this exists

Which library a build loads depends on the SwiftPM build system, because each one packages the `.process`ed `.metal` resources differently:

| Build system | Bundle holds | Debug loads | Release loads |
|---|---|---|---|
| Swift Build (default since Swift 6.4) | `debug.metallib`, `default.metallib`, `Metal4Common.h`, no sources | `debug.metallib` | `default.metallib` |
| Native (`--build-system native`, deprecated) | `debug.metallib`, every `.metal` source, no `default.metallib` | `debug.metallib` | runtime compile |

Before this contract, a Swift Build bundle reached the runtime compile with no sources. It compiled only its preamble and silently returned a library with no kernels, and 76 tests that asked for a "runtime" library failed.

## The contract

- `KernelContext.loadMetalLibrary` tries, in order: `debug.metallib` (Debug only), `default.metallib`, the runtime compile, then the device default library. In production a throw from the runtime compile is caught and the next step is tried.
- `KernelContext.makeLibraryFromBundleSources` throws `shaderCompilationFailed` when the bundle has no sources, or when any listed source is missing. Pinned by `RuntimeShaderCompilePackagingTests`.
- Hardening suites certify whatever libraries this build's bundle ships, through `ShippedMetalLibraries`, instead of assuming a runtime library exists. The utility fails if a bundle ships none.
- `ShaderLibraryCompletenessTests` asserts that every kernel of every non-excluded shader file is in every shipped library, and that all shipped libraries expose the same kernel set. Under Swift Build in Debug, that compares `debug.metallib` with `default.metallib`.
- `EpsilonCompileParityTests` skips, with the reason stated, when the bundle has no sources to compile.

## Costs and open questions

- Every consumer's build host needs the Metal Toolchain component (`xcodebuild -downloadComponent MetalToolchain`). Without it, `swift build` fails at `CompileMetalFile`.
- The compile flags of `default.metallib` come from Xcode's defaults, not from `KernelContext`'s runtime options.
- **Open:** the compiler plugin's `debug.metallib` still ships in Release bundles.

## Verification scope

CI runs Xcode 26, whose default is the native build system, so CI doesn't exercise the Swift Build row. That row is verified locally only, until the runners move to Xcode 27.
