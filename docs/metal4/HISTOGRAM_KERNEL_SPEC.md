# Histogram Kernel Implementation Spec

## Overview

This document provides all necessary context to implement three Metal shader functions for histogram computation in the VectorAccelerate library. The Swift kernel wrapper (`Metal4HistogramKernel.swift`) is already complete - only the Metal shader functions are missing.

**Target File:** `Sources/VectorAccelerate/Metal/Shaders/StatisticsShaders.metal`

**Functions to Implement:**
1. `uniformHistogram` - Equal-width bin histogram
2. `adaptiveHistogram` - Variable-width bin histogram (pre-computed edges)
3. `logarithmicHistogram` - Logarithmic-scale bin histogram

---

## Project Context

### MSL Version & Headers
- **MSL Version:** 4.0 (Metal 4 SDK)
- **Target:** macOS 26.0+, iOS 26.0+, visionOS 3.0+
- **Required Header:** `#include "Metal4Common.h"`

### Common Constants (from Metal4Common.h)
```metal
constant float VA_EPSILON = 1e-7f;
constant uint VA_MAX_THREADGROUP_SIZE = 1024;
constant uint VA_SIMD_WIDTH = 32;
```

---

## Function 1: uniformHistogram

### Purpose
Compute histogram with equal-width bins across a data range.

### Signature
```metal
kernel void uniformHistogram(
    device const float* data [[buffer(0)]],      // Input data array
    device const float* binEdges [[buffer(1)]],  // Bin edges (numBins + 1 values)
    device atomic_uint* histogram [[buffer(2)]], // Output histogram (numBins bins)
    constant uint4& params [[buffer(3)]],        // Parameters (see below)
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
);
```

### Parameters (params)
The Swift code passes `SIMD4<UInt32>` which maps to `uint4`:
- `params[0]` = dataCount (number of input elements)
- `params[1]` = numBins (number of histogram bins)
- `params[2]` = includeOutliers (0 or 1)
- `params[3]` = reserved (0)

### Algorithm
1. Each thread processes elements at indices `gid`, `gid + gridSize`, etc.
2. For each data value:
   - Compute bin index: `binIndex = floor((value - minEdge) / binWidth)`
   - Clamp to valid range: `[0, numBins-1]`
   - If `includeOutliers == 0`, skip values outside range
3. Atomically increment `histogram[binIndex]`

### Optimization Notes
- Use `atomic_fetch_add_explicit` with `memory_order_relaxed` for performance
- For uniform bins, bin index can be computed directly without searching:
  ```metal
  float minEdge = binEdges[0];
  float maxEdge = binEdges[numBins];
  float binWidth = (maxEdge - minEdge) / (float)numBins;
  uint binIndex = (uint)floor((value - minEdge) / binWidth);
  binIndex = clamp(binIndex, 0u, numBins - 1);
  ```

---

## Function 2: adaptiveHistogram

### Purpose
Compute histogram with variable-width bins (edges pre-computed by Swift).

### Signature
```metal
kernel void adaptiveHistogram(
    device const float* data [[buffer(0)]],      // Input data array
    device const float* binEdges [[buffer(1)]],  // Bin edges (numBins + 1 values, sorted)
    device atomic_uint* histogram [[buffer(2)]], // Output histogram (numBins bins)
    constant uint4& params [[buffer(3)]],        // Parameters (same as uniformHistogram)
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
);
```

### Parameters
Same as `uniformHistogram`.

### Algorithm
1. Each thread processes elements at indices `gid`, `gid + gridSize`, etc.
2. For each data value:
   - Binary search to find bin: `binEdges[i] <= value < binEdges[i+1]`
   - Handle edge case: `value == binEdges[numBins]` goes to last bin
3. Atomically increment `histogram[binIndex]`

### Binary Search Implementation
```metal
inline uint findBin(float value, device const float* edges, uint numBins) {
    uint lo = 0;
    uint hi = numBins;

    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (value < edges[mid]) {
            hi = mid;
        } else if (value >= edges[mid + 1]) {
            lo = mid + 1;
        } else {
            return mid;
        }
    }

    // Clamp to valid range
    return clamp(lo, 0u, numBins - 1);
}
```

---

## Function 3: logarithmicHistogram

### Purpose
Compute histogram with logarithmic-scale bins (for positively-skewed data).

### Signature
```metal
kernel void logarithmicHistogram(
    device const float* data [[buffer(0)]],      // Input data array (must be > 0)
    device const float* binEdges [[buffer(1)]],  // Log-scale bin edges (numBins + 1 values)
    device atomic_uint* histogram [[buffer(2)]], // Output histogram (numBins bins)
    constant uint4& params [[buffer(3)]],        // Parameters (same as uniformHistogram)
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
);
```

### Parameters
Same as `uniformHistogram`.

### Algorithm
1. Each thread processes elements at indices `gid`, `gid + gridSize`, etc.
2. For each data value:
   - Skip if `value <= 0` (logarithm undefined)
   - Use binary search on log-scale edges (same as adaptive)
   - OR compute directly if base is known:
     ```metal
     float logVal = log(value) / log(base);
     float logMin = log(binEdges[0]) / log(base);
     float logMax = log(binEdges[numBins]) / log(base);
     float logWidth = (logMax - logMin) / (float)numBins;
     uint binIndex = (uint)floor((logVal - logMin) / logWidth);
     ```
3. Atomically increment `histogram[binIndex]`

### Note
The bin edges are already in linear space (computed by Swift using the log scale). The simplest implementation is identical to `adaptiveHistogram` - just use binary search on the pre-computed edges.

---

## Buffer Layout

### Input Data Buffer (buffer 0)
```
[float, float, float, ...] // dataCount elements
```

### Bin Edges Buffer (buffer 1)
```
[edge0, edge1, edge2, ..., edgeN] // numBins + 1 elements (sorted ascending)
```
- `edge0` = minimum (first bin starts here)
- `edgeN` = maximum (last bin ends here)

### Histogram Output Buffer (buffer 2)
```
[count0, count1, count2, ...] // numBins elements (atomic_uint)
```
**Important:** The Swift code pre-initializes this buffer to zeros before kernel execution.

---

## Swift Dispatch Configuration

The Swift kernel dispatches like this:
```swift
let threadgroupSize = MTLSize(
    width: min(data.count, kernel.maxTotalThreadsPerThreadgroup),
    height: 1,
    depth: 1
)
let threadgroups = MTLSize(
    width: (data.count + threadgroupSize.width - 1) / threadgroupSize.width,
    height: 1,
    depth: 1
)
encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
```

This means:
- Multiple threadgroups may be dispatched
- Each thread should process one element at `gid`, using grid-stride loop if needed
- Use atomics for histogram updates (multiple threads may update same bin)

---

## Reference Implementation Pattern

From `computeBasicStatistics` in the same file:
```metal
kernel void computeBasicStatistics(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint2& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    uint dimension = params[0];

    // Thread-stride loop for processing
    for (uint i = tid; i < dimension; i += tgSize) {
        float x = input[i];
        // Process x...
    }
}
```

---

## Test Expectations

The tests use data like:
```swift
let data: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
let config = Metal4HistogramConfig(binningStrategy: .uniform(bins: 5))
// Expected bins for range [1, 10] with 5 uniform bins:
// [1-2.8): 2 elements (1, 2)
// [2.8-4.6): 2 elements (3, 4)
// [4.6-6.4): 2 elements (5, 6)
// [6.4-8.2): 2 elements (7, 8)
// [8.2-10]: 2 elements (9, 10)
```

---

## Implementation Checklist

- [x] Add `uniformHistogram` kernel function
- [x] Add `adaptiveHistogram` kernel function
- [x] Add `logarithmicHistogram` kernel function
- [x] Use `atomic_uint` and `atomic_fetch_add_explicit` for thread-safe updates
- [x] Handle `includeOutliers` parameter (skip out-of-range values if 0)
- [x] Ensure proper bounds checking
- [x] Test with the existing `Metal4HistogramKernelTests`

## Implementation Notes (2025-11-30)

The histogram kernels were implemented by merging the best aspects of two independent implementations:

### Key Features
1. **Grid-stride loop**: Robust processing pattern for any dispatch configuration
2. **NaN/Infinity handling**: Uses `isfinite()` to skip invalid data
3. **VA_EPSILON zero-range handling**: Graceful degradation for degenerate bin configurations
4. **Reusable `va_findBin` helper**: Binary search extracted with `va_` namespace prefix
5. **Precomputed `invBinWidth`**: Avoids division in hot loop for uniform histogram
6. **Explicit unused parameter handling**: `(void)tid;` for clean compilation

### Tests Passing
All 9 `Metal4HistogramKernelTests` pass:
- testUniformBinning
- testBatchHistograms
- testBinCenters
- testEntropy
- testHistogramIntersection
- testNormalization
- testPeakBin
- testStatisticsComputation
- testSummary

---

## File Location

Add the three kernel functions to:
```
Sources/VectorAccelerate/Metal/Shaders/StatisticsShaders.metal
```

After the existing `computeCorrelation` kernel (around line 449).

---

## Validation

After implementation, run tests:
```bash
swift test --filter Metal4HistogramKernelTests
```

The tests should transition from "skipped" to passing.
