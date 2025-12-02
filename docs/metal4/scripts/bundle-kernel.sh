#!/bin/bash
#
# bundle-kernel.sh - Bundle all context needed for kernel migration
#
# Usage: ./bundle-kernel.sh <KernelName>
#
# Example: ./bundle-kernel.sh L2Distance
#          ./bundle-kernel.sh CosineSimilarity
#
# Output: Prints a complete context bundle for migrating a specific kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$(dirname "$DOCS_DIR")")"
CONTEXT_DIR="$DOCS_DIR/00-context"
SOURCES_DIR="$PROJECT_ROOT/Sources/VectorAccelerate"

KERNEL_NAME="$1"

if [ -z "$KERNEL_NAME" ]; then
    echo "Usage: $0 <KernelName>" >&2
    echo "" >&2
    echo "Available kernels:" >&2
    ls "$SOURCES_DIR/Kernels/" | grep "Kernel.swift" | sed 's/Kernel.swift//' >&2
    exit 1
fi

# Find kernel files
KERNEL_SWIFT=$(find "$SOURCES_DIR/Kernels" -name "${KERNEL_NAME}Kernel.swift" -o -name "${KERNEL_NAME}*.swift" | head -1)
KERNEL_METAL=$(find "$SOURCES_DIR/Metal/Shaders" -iname "*${KERNEL_NAME}*.metal" | head -1)

if [ -z "$KERNEL_SWIFT" ]; then
    echo "Error: Could not find Swift file for kernel: $KERNEL_NAME" >&2
    exit 1
fi

cat << HEADER
# Kernel Migration Bundle: ${KERNEL_NAME}

This document contains all context needed to migrate the ${KERNEL_NAME} kernel
from Metal 3 to Metal 4.

Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

---

HEADER

# Include patterns (most relevant for kernels)
echo "# Part 1: Current Patterns"
echo ""
cat "$CONTEXT_DIR/patterns.md"
echo ""
echo "---"
echo ""

# Include Metal 4 API reference
echo "# Part 2: Metal 4 API Reference"
echo ""
cat "$CONTEXT_DIR/metal4-api-summary.md"
echo ""
echo "---"
echo ""

# Include Swift kernel source
echo "# Part 3: Current Swift Implementation"
echo ""
echo "**File:** \`$(echo "$KERNEL_SWIFT" | sed "s|$PROJECT_ROOT/||")\`"
echo ""
echo '```swift'
cat "$KERNEL_SWIFT"
echo '```'
echo ""
echo "---"
echo ""

# Include Metal shader if found
if [ -n "$KERNEL_METAL" ] && [ -f "$KERNEL_METAL" ]; then
    echo "# Part 4: Current Metal Shader"
    echo ""
    echo "**File:** \`$(echo "$KERNEL_METAL" | sed "s|$PROJECT_ROOT/||")\`"
    echo ""
    echo '```metal'
    cat "$KERNEL_METAL"
    echo '```'
    echo ""
    echo "---"
    echo ""
fi

# Include migration instructions
cat << 'INSTRUCTIONS'
# Part 5: Migration Instructions

## Required Changes for Metal 4

### Swift Kernel Changes

1. **Buffer Binding** - Replace individual `setBuffer()` calls with ArgumentTable:
   ```swift
   // Before
   encoder.setBuffer(buffer, offset: 0, index: 0)

   // After
   let argTable = argumentTablePool.acquire()
   argTable.setAddress(buffer.gpuAddress, index: 0)
   encoder.setArgumentTable(argTable, stages: .compute)
   ```

2. **Context Type** - Accept `Metal4Context` instead of `MetalContext`:
   ```swift
   // Consider protocol-based approach for compatibility
   public actor MyKernel<Context: MetalContextProtocol> {
       private let context: Context
   }
   ```

3. **Residency** - Buffers must be in residency set before use:
   - BufferPool handles this automatically
   - Direct buffer creation needs explicit registration

### Metal Shader Changes

1. **Header Update**:
   ```metal
   #include <metal_stdlib>
   #include <metal_tensor>  // Add if using tensor operations
   using namespace metal;
   ```

2. **SIMD Optimization** - Consider simd_sum for reductions:
   ```metal
   // Before: manual threadgroup reduction
   // After: simd_sum + threadgroup reduction
   float sum = simd_sum(localSum);
   ```

3. **Barriers** - Verify `threadgroup_barrier` compatibility (usually unchanged)

## Output Format

Provide:
1. Updated Swift kernel file (complete)
2. Updated Metal shader file (if changes needed)
3. Brief explanation of changes
4. Any new dependencies or types needed

## Quality Checklist

- [ ] ArgumentTable used for buffer binding
- [ ] Works with Metal4Context
- [ ] Metal shader compiles with MSL 4.0
- [ ] Maintains same public API
- [ ] Performance equivalent or better
- [ ] All existing tests pass
INSTRUCTIONS
