# Repository Context: High-Performance Vector Mathematics Framework

## Architectural Target & Hardware Constraints
*   **Primary Target:** Apple Silicon (M-Series Architecture).
*   **Memory Model:** Unified Memory Architecture (UMA). Crucial to eliminate unnecessary copying between CPU and GPU via shared memory allocation (`MTLStorageModeShared`).
*   **Execution Vector:** Hybrid processing leveraging CPU SIMD (`SIMD4`, `SIMD8`, `SIMD16`), Accelerate framework (vDSP/vForce/BLAS), and GPU compute shaders (Metal Shading Language / MSL).

## Technical Pillars & Optimization Goals
1.  **Zero Allocation Workflows:** Critical paths must avoid heap allocations. Prioritize stack-allocated types, value types (Structs), pointer manipulation via `withUnsafeBitCast` or `withUnsafeBufferPointer`, and reuse of transient resources.
2.  **Generic Specialization:** Avoid runtime dynamic dispatch and existential containers. Ensure all generic vector types are designed for total compiler visibility and guaranteed specialization down to primitive floating-point layouts.
3.  **Compiler Directives:** Heavily leverage `@inlinable` and `@inline(__always)` optimizations across the public and internal vector API surfaces to facilitate cross-module auto-vectorization.
4.  **Memory Alignment & Struct Packing:** Enforce strict alignment between Swift-side types and MSL-side types. Avoid padding traps across the bridging headers.

## AI Instruction Guardrails
*   Always evaluate code with a hardware-centric perspective (cache lines, SIMD lane utilization, threadgroup memory).
*   Do not suggest high-level Cocoa or Foundation alternatives if they introduce abstraction overhead.
*   Prioritize explicit vectorization over implicit loops.
