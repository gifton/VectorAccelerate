# Kernel Migration Task Template

> **Instructions:** Copy this template and fill in the placeholders for each kernel migration.

---

# Task: Migrate {KERNEL_NAME} Kernel to Metal 4

## Metadata

| Field | Value |
|-------|-------|
| **Kernel Name** | {KERNEL_NAME} |
| **Priority** | {P1/P2/P3} |
| **Swift File** | `Sources/VectorAccelerate/Kernels/{KERNEL_NAME}Kernel.swift` |
| **Metal Shader** | `Sources/VectorAccelerate/Metal/Shaders/{SHADER_FILE}.metal` |
| **Shader Functions** | {function1, function2, ...} |
| **Fusible With** | {other kernels this can be fused with, or "None"} |
| **Metal 4 Features** | {ArgumentTable, SimdgroupMatrix, MLTensor, etc.} |

---

## Current Implementation

### Swift Binding Pattern (Metal 3)

```swift
// Current kernel binding in {KERNEL_NAME}Kernel.swift
try await context.executeAndWait { commandBuffer, encoder in
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer({INPUT_BUFFER}, offset: 0, index: 0)
    encoder.setBuffer({INPUT_BUFFER_2}, offset: 0, index: 1)
    encoder.setBuffer({OUTPUT_BUFFER}, offset: 0, index: 2)
    encoder.setBytes(&params, length: MemoryLayout<{PARAMS_TYPE}>.size, index: 3)

    let threadgroups = MTLSize(width: {X}, height: {Y}, depth: 1)
    let threadsPerGroup = MTLSize(width: {TPG}, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
}
```

### Metal Shader Signature

```metal
// Current signature in {SHADER_FILE}.metal
kernel void {FUNCTION_NAME}(
    device const float* {input1} [[buffer(0)]],
    device const float* {input2} [[buffer(1)]],
    device float* {output} [[buffer(2)]],
    constant {ParamsType}& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // ... implementation
}
```

---

## Target Implementation (Metal 4)

### ArgumentTable Layout

| Index | Resource | Type | Notes |
|-------|----------|------|-------|
| 0 | {input1} | `device const float*` | GPU address via `.gpuAddress` |
| 1 | {input2} | `device const float*` | GPU address |
| 2 | {output} | `device float*` | GPU address |
| 3 | params | `constant {ParamsType}&` | Small struct, inline or buffer |

### Swift Binding Pattern (Metal 4)

```swift
// Updated kernel binding for Metal 4
try await context.execute { commandBuffer, encoder in
    // Get argument table from pool
    let argTable = try await context.argumentTablePool.acquire()
    defer { Task { await context.argumentTablePool.release(argTable) } }

    // Bind resources via GPU addresses
    argTable.setAddress({inputBuffer1}.gpuAddress, index: 0)
    argTable.setAddress({inputBuffer2}.gpuAddress, index: 1)
    argTable.setAddress({outputBuffer}.gpuAddress, index: 2)
    argTable.setAddress({paramsBuffer}.gpuAddress, index: 3)

    // Set pipeline and argument table
    encoder.setComputePipelineState(pipeline)
    encoder.setArgumentTable(argTable, stages: .compute)

    // Dispatch
    let threadgroups = MTLSize(width: {X}, height: {Y}, depth: 1)
    let threadsPerGroup = MTLSize(width: {TPG}, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
}
```

### Metal Shader Updates

```metal
// Updated shader header
#include <metal_stdlib>
#include <metal_tensor>  // If using tensor operations
using namespace metal;

// Shader signature unchanged (argument tables are transparent to shader)
kernel void {FUNCTION_NAME}(
    device const float* {input1} [[buffer(0)]],
    device const float* {input2} [[buffer(1)]],
    device float* {output} [[buffer(2)]],
    constant {ParamsType}& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    // Implementation unchanged unless adding new features
    // ...
}
```

---

## Barriers Required

| Step | Before | After | Resources | Notes |
|------|--------|-------|-----------|-------|
| {Describe if barriers needed between operations} |

**Example:**
| Step | Before | After | Resources | Notes |
|------|--------|-------|-----------|-------|
| Distance → TopK | `.dispatch` | `.dispatch` | `distances` | Wait for distance write before selection read |

---

## Fusion Opportunities

{Describe if this kernel can be fused with others}

**Example:**
- Can fuse with TopKSelection in single command buffer
- Avoid encoder end/begin overhead
- Share argument table where inputs overlap

---

## Metal 4 Feature Adoption

### ArgumentTable (Required)
- [x] Replace `setBuffer()` with argument table binding
- [x] Use GPU addresses from buffer pool

### SimdgroupMatrix (Optional - if applicable)
- [ ] Consider `simdgroup_matrix` for matrix operations
- [ ] Evaluate performance vs current implementation

### MLTensor (Optional - if applicable)
- [ ] Consider inline tensor ops for projections
- [ ] Requires Phase 4 completion

---

## Residency Requirements

All input and output buffers must be registered with ResidencyManager before use:

```swift
// Ensure buffers are resident
try await context.residencyManager.registerBuffer(inputBuffer1)
try await context.residencyManager.registerBuffer(inputBuffer2)
try await context.residencyManager.registerBuffer(outputBuffer)
try await context.residencyManager.commit()
```

**Note:** If using BufferPool with residency integration, this is automatic.

---

## Testing

### Correctness Tests

```swift
func test{KERNEL_NAME}Metal4Correctness() async throws {
    let metal4Context = try await Metal4Context()
    let kernel = {KERNEL_NAME}Kernel(context: metal4Context)

    // Generate test data
    let input = generateTestData(...)

    // Compute with Metal 4
    let result = try await kernel.compute(input)

    // Compare to CPU reference
    let expected = cpuReference(input)
    XCTAssertEqual(result, expected, accuracy: 1e-5)
}
```

### Performance Tests

```swift
func test{KERNEL_NAME}Metal4Performance() async throws {
    let metal4Context = try await Metal4Context()
    let kernel = {KERNEL_NAME}Kernel(context: metal4Context)

    measure {
        // Benchmark code
    }

    // Compare to baseline
    // Result should be >= 90% of Metal 3 baseline
}
```

### Baseline Comparison

| Metric | Metal 3 Baseline | Metal 4 Target | Status |
|--------|------------------|----------------|--------|
| Throughput ({unit}/sec) | {baseline} | >= {baseline * 0.9} | |
| Latency (ms) | {baseline} | <= {baseline * 1.1} | |
| Memory (MB) | {baseline} | <= {baseline * 1.05} | |

---

## Migration Checklist

### Swift Changes
- [ ] Update to use `Metal4Context`
- [ ] Replace `setBuffer()` with ArgumentTable
- [ ] Add barrier calls if needed
- [ ] Handle argument table lifecycle
- [ ] Ensure buffers registered with ResidencyManager

### Metal Shader Changes
- [ ] Add MSL 4.0 headers (`#include <metal_tensor>` if needed)
- [ ] Verify `threadgroup_barrier` compatibility
- [ ] Add `simdgroup_matrix` if applicable
- [ ] Compile with `-std=metal4.0`

### Testing
- [ ] Correctness test passes
- [ ] Performance test shows no regression
- [ ] Memory usage acceptable
- [ ] Works with various input sizes/dimensions

### Documentation
- [ ] Update kernel documentation
- [ ] Note any API changes
- [ ] Document performance characteristics

---

## Files Modified

| File | Changes |
|------|---------|
| `Kernels/{KERNEL_NAME}Kernel.swift` | ArgumentTable binding, Metal4Context |
| `Metal/Shaders/{SHADER_FILE}.metal` | MSL 4.0 headers, optional optimizations |

---

## Notes

{Any additional notes, gotchas, or considerations specific to this kernel}

---

## Context Files for External Agent

When delegating this task, include:
1. `00-context/metal4-api-summary.md`
2. `00-context/patterns.md`
3. `02-command-encoding/PHASE_OVERVIEW.md` (for barrier reference)
4. This completed task file
5. Current source files:
   - `Sources/VectorAccelerate/Kernels/{KERNEL_NAME}Kernel.swift`
   - `Sources/VectorAccelerate/Metal/Shaders/{SHADER_FILE}.metal`
