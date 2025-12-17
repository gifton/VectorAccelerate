# VectorAccelerate Performance Analysis

**Date:** December 16, 2025
**Test Suite:** 1,077 tests in 444.2 seconds
**Platform:** macOS 26 / Metal 4 / Apple Silicon

---

## Executive Summary

This document presents performance findings from the VectorAccelerate test suite output. Analysis covers kernel throughput, latency metrics, and identifies areas for optimization as well as components performing well.

---

## 1. Product Quantization Training

### Metrics

| Samples | Codebooks | Duration | Rate |
|---------|-----------|----------|------|
| 200 | 8 | 7,127.99 ms | 28 samples/sec |
| 30 | 8 | 1,246.36 ms | 24 samples/sec |
| 50 | 4 | 136.44 ms | 366 samples/sec |
| 50 | 8 | 243.23 ms | 206 samples/sec |
| 50 | 16 | 510.54 ms | 98 samples/sec |
| 500 | 4 | 1,135.16 ms | 440 samples/sec |

### Observations

- Training time scales roughly linearly with codebook count
- 8-codebook configuration at 200 samples takes 7+ seconds
- Rate degrades significantly as sample count increases beyond 50
- KMeans clustering appears to be the dominant cost

### Source Location

- `QuantizationEngine.swift` - `trainProductQuantization`
- KMeans implementation in clustering pipeline

---

## 2. IVF Index Insert Performance

### Metrics

| Index Type | Vector Count | Insert Time | Throughput |
|------------|--------------|-------------|------------|
| Flat | 10,000 (128D) | 0.41s | 24,169 vec/s |
| Flat | 10,000 (768D) | 2.41s | 4,149 vec/s |
| IVF | 10,000 (128D) | 10.03s | 997 vec/s |

### Observations

- IVF insert is 24x slower than flat insert for equivalent data
- The overhead comes from KMeans training during index construction
- Dimension has expected impact on flat insert (6x slower for 6x dimensions)

### Source Location

- `IVFIndex.swift` - insert and training logic
- `KMeansPipeline.swift` - clustering implementation

---

## 3. Multi-Head Attention Performance

### Metrics

| Configuration | Avg Time | Throughput | Per-Head Overhead |
|---------------|----------|------------|-------------------|
| Single-head 768→64 | 24.55 ms | 4,073,520 pairs/s | baseline |
| Single-head 384→64 | 17.83 ms | 5,609,083 pairs/s | baseline |
| Multi-head 6 heads (384D) | 50.04 ms | 1,998,562 pairs/s | 2.0x |
| Multi-head 12 heads (768D) | 163.02 ms | 613,416 pairs/s | 6.6x |

### Observations

- Multi-head attention shows disproportionate overhead relative to head count
- 12-head configuration is 6.6x slower per-pair than single-head
- Suggests per-head dispatch overhead or poor memory coalescing

### Source Location

- `AttentionSimilarityKernel.swift`
- Metal shaders in `AttentionSimilarity.metal`

---

## 4. Batch Quantization Scaling

### Metrics (Scalar 8-bit)

| Batch Size | Quantize Time | Throughput | Dequantize Time | Throughput |
|------------|---------------|------------|-----------------|------------|
| 10 | 50.03 ms | 200 vec/s | 0.34 ms | 29,321 vec/s |
| 50 | 7.96 ms | 6,281 vec/s | 1.53 ms | 32,594 vec/s |
| 100 | ~8.72 ms | 11,465 vec/s | 3.03 ms | 32,970 vec/s |

### Observations

- Small batch quantization (n=10) is 57x slower per-vector than larger batches
- Dequantization throughput is consistent across batch sizes
- GPU dispatch overhead dominates at small batch sizes
- Quantization benefits significantly from batching; dequantization does not

### Source Location

- `QuantizationEngine.swift` - `scalarQuantize`, `batchQuantize`

---

## 5. Neural Quantization Encode/Decode Asymmetry

### Metrics

| Operation | Throughput | Relative |
|-----------|------------|----------|
| Encoding | 83,291 vec/s | 1.0x |
| Decoding | 9,491 vec/s | 0.11x |

Additional encoding benchmarks:

| Configuration | Avg Time | Throughput |
|---------------|----------|------------|
| 384→64 (MiniLM) | 6.92 ms | 1,445,674 vec/s |
| 768→128 | 10.33 ms | 968,299 vec/s |
| High compression | 8.54 ms | 1,171,237 vec/s |

### Observations

- Decoding is 8.8x slower than encoding
- Encoding throughput scales reasonably with dimension
- Decode path may have inefficient matrix reconstruction

### Source Location

- `NeuralQuantizationKernel.swift`
- Metal shaders for neural quantization

---

## 6. IVF Recall vs Throughput Tradeoff

### Metrics (64D, 100 vectors)

| Configuration | Recall | Throughput | p50 Latency |
|---------------|--------|------------|-------------|
| Flat | 100.0% | 1,917 q/s | 0.523 ms |
| IVF nprobe=nlist | 100.0% | 1,036 q/s | 0.890 ms |
| IVF nprobe=50 | 78.0% | 1,181 q/s | 0.797 ms |
| IVF nprobe=25 | 56.7% | 1,266 q/s | 0.768 ms |
| IVF nprobe=12 | 35.7% | 1,273 q/s | 0.724 ms |

### Observations

- At 100 vectors, IVF provides no throughput benefit over flat search
- Full nprobe IVF is 46% slower than flat with identical recall
- IVF overhead (coarse quantizer + candidate gathering) exceeds brute-force cost at small scale
- Crossover point where IVF becomes beneficial is likely >10K vectors

### Source Location

- `IVFIndex.swift` - search implementation
- `GPUCandidateBuilder.swift`

---

## 7. Flat Index Search Performance

### Metrics

| Dimension | Vector Count | Insert Throughput | Search Avg | Search Median |
|-----------|--------------|-------------------|------------|---------------|
| 128 | 10,000 | 24,169 vec/s | 0.690 ms | 0.621 ms |
| 768 | 10,000 | 4,149 vec/s | 0.821 ms | 0.829 ms |

Benchmark summary output:

```
Dim 128: Insert 22,643 vec/s | Search 0.60 ms
Dim 768: Insert  3,965 vec/s | Search 0.85 ms
IVF (nlist=8, nprobe=2):     | Search 0.21 ms
```

### Observations

- Search latency scales sub-linearly with dimension (1.2x for 6x dimensions)
- Insert throughput scales inversely with dimension as expected
- Consistent median/average indicates stable latency distribution

### Source Location

- `FlatIndex.swift`
- Distance computation kernels

---

## 8. Matrix Operations Performance

### Metrics

| Matrix Size | Multiply Time | MFLOPS | Transpose Time |
|-------------|---------------|--------|----------------|
| 2×3 × 3×4 | 0.005 ms | - | - |
| 8×8 × 8×8 | 0.010 ms | - | - |
| 16×16 | 0.239 ms | 17.1 | 0.232 ms |
| 32×32 | 0.259 ms | 126.5 | 0.207 ms |
| 64×64 | 1.008 ms | 260.1 | 0.239 ms |
| 128×128 | 0.476 ms | 4,405.8 | 0.309 ms |

Large matrix performance:

| Size | Multiply Time |
|------|---------------|
| 64×64 | 0.794 ms |
| 200×200 transpose | 0.844 ms |

### Observations

- Excellent FLOPS scaling from 16×16 to 128×128 (257x improvement)
- 128×128 achieves 4.4 GFLOPS, indicating good GPU utilization
- Transpose performance is consistent and fast
- Small matrices show expected overhead from dispatch

### Source Location

- `MatrixEngine.swift`
- Metal matrix shaders

---

## 9. Distance Computation Performance

### Metrics

| Distance Type | Configuration | Throughput |
|---------------|---------------|------------|
| Standard L2 | 768D | 8,525,649 pairs/s |
| Learned L2 | 384→64 | 6,909,840 pairs/s |
| Learned L2 | 768→128 | 2,268,943 pairs/s |
| Attention Similarity | 768→64 | 4,073,520 pairs/s |
| Attention Similarity | 384→64 | 5,609,083 pairs/s |

### Observations

- Standard L2 at 768D achieves 8.5M pairs/sec
- Learned L2 with projection adds ~3.7x overhead at 768D
- Attention similarity throughput is competitive with learned distance
- Smaller dimensions (384D) show better throughput as expected

### Source Location

- `L2DistanceKernel.swift`
- `AttentionSimilarityKernel.swift`
- Metal distance shaders

---

## 10. Batch Processing Performance

### Metrics

| Batch Size | Time | Throughput |
|------------|------|------------|
| 1 | 0.010 ms | - |
| 100 | 0.038 ms | - |
| 1,000 | 0.117 ms | 8.5M items/s |
| 5,000 | 0.137 ms | 36.5M items/s |
| 10,000 | 0.681 ms | 14.7M items/s |
| 50,000 | 0.913 ms | 54.8M items/s |
| 100,000 | 2.042 ms | 49.0M items/s |

### Observations

- Excellent batch scaling up to 50K items
- Throughput peaks around 50K batch size
- Minimal overhead for small batches
- Good amortization of dispatch costs

### Source Location

- `BatchProcessor.swift`

---

## 11. Buffer Pool Allocation

### Metrics

| Operation | Time |
|-----------|------|
| Average allocation | 0.00681 ms |
| Reuse performance | 0.083 ms (test duration) |

### Observations

- Sub-millisecond allocation times
- Buffer pooling working effectively
- No allocation bottlenecks observed

### Source Location

- `BufferPool.swift`

---

## 12. Warp-Optimized Selection (TopK)

### Metrics

```
Metal4WarpOptimizedSelection performance: 0.927 ms per run
  - 100 queries
  - 1,000 candidates
  - k=10
```

Per-query: 9.27 μs

### Observations

- Sub-millisecond TopK for 100 queries
- Efficient warp-level operations
- Good scaling with candidate count

### Source Location

- `WarpOptimizedSelectionKernel.swift`
- `AdvancedTopK.metal`

---

## 13. Metal Context Initialization

### Metrics

| Phase | Time |
|-------|------|
| Init completion | < 10 ms |
| Time to criticalReady (cold) | 55.06 ms |
| Time to deviceReady | 55.05 ms |
| Warm start with context | 57 ms |
| Warmup after idle timeout | 457 ms |

### Observations

- Cold start to critical ready in ~55ms is acceptable
- Init phase completes quickly
- Idle timeout recovery takes ~0.5s (expected for re-warmup)

### Source Location

- `Metal4Context.swift`
- `WarmupManager.swift`

---

## 14. Memory-Mapped Operations

### Metrics

| Operation | Count | Time |
|-----------|-------|------|
| Read | 10 | 0.055 ms |
| Read | 2 | 0.025 ms |
| Distance | - | 0.088 ms |

### Observations

- Efficient memory-mapped access
- Consistent read performance
- Distance computation on mapped data works well

### Source Location

- `MemoryMappedVectors.swift`

---

## 15. Scalar Quantization by Bit Depth

### Metrics (128D vectors)

| Bits | Quantize Time | Compression | SNR |
|------|---------------|-------------|-----|
| 4-bit | 0.054 ms | 8.0x | 23.98 dB |
| 8-bit | 0.036 ms | 4.0x | 49.10 dB |
| 16-bit | 0.035 ms | 2.0x | 96.74 dB |

### Observations

- 4-bit quantization is slightly slower (more bit manipulation)
- 8-bit and 16-bit have similar performance
- SNR degrades gracefully with compression
- Good quality-compression tradeoff

### Source Location

- `QuantizationEngine.swift` - `scalarQuantize`

---

## 16. Quantization Performance by Vector Size

### Metrics (8-bit scalar)

| Vector Size | Quantize | Dequantize | Binary Quantize | Binary Dequantize |
|-------------|----------|------------|-----------------|-------------------|
| 64 | 0.033 ms | 0.010 ms | 0.065 ms | 0.009 ms |
| 128 | 0.060 ms | 0.015 ms | 0.040 ms | 0.015 ms |
| 256 | 0.085 ms | 0.028 ms | 0.045 ms | 0.029 ms |
| 512 | 0.131 ms | 0.056 ms | 0.074 ms | 0.055 ms |
| 1024 | 0.226 ms | 0.107 ms | 0.124 ms | 0.106 ms |

### Observations

- Linear scaling with vector size
- Binary quantization faster than scalar for larger vectors
- Dequantization consistently faster than quantization
- Good vectorization efficiency

### Source Location

- `QuantizationEngine.swift`

---

## 17. Batch Insert Performance

### Metrics

```
Batch insert 1,000 vectors:
  Time: 0.438 ms
  Throughput: 2,283,236 vectors/sec
```

### Observations

- Excellent batch insert throughput
- Well-optimized bulk operations
- No bottlenecks in insert path

### Source Location

- `FlatIndex.swift` - batch insert

---

## 18. Batch Search Advantage

### Metrics

```
Batch search (20 queries): 4.313 ms
Sequential search (20 queries): 9.396 ms
Speedup: 2.18x
```

### Observations

- Batch search provides >2x speedup over sequential
- GPU utilization improves with batching
- Amortization of kernel dispatch overhead

### Source Location

- `FlatIndex.swift` - batch search

---

## 19. GPU vs CPU Transpose Comparison

### Metrics

```
GPU transpose time: 0.441 ms
CPU transpose time: 0.023 ms
```

### Observations

- CPU is faster for small matrix transpose
- GPU dispatch overhead exceeds computation time
- Routing logic should prefer CPU for small operations

### Source Location

- `MatrixEngine.swift` - routing logic

---

## 20. Histogram Kernel

### Metrics

Test completed in 0.024 seconds for batch histograms.

### Observations

- Fast histogram computation
- Batch processing efficient
- No issues observed

### Source Location

- `HistogramKernel.swift`

---

## Summary Tables

### Operations Performing Well

| Component | Key Metric | Assessment |
|-----------|------------|------------|
| Matrix 128×128 multiply | 4,405 MFLOPS | Excellent |
| Standard L2 distance | 8.5M pairs/s | Excellent |
| Buffer allocation | 0.007 ms | Excellent |
| Batch insert | 2.3M vec/s | Excellent |
| Warp TopK | 9.27 μs/query | Good |
| Flat search 10K | 0.69 ms | Good |
| Batch processing 50K | 54.8M items/s | Good |
| Context init | 55 ms cold start | Good |

### Operations Needing Optimization

| Component | Key Metric | Issue |
|-----------|------------|-------|
| PQ training 200 samples | 7,128 ms | Very slow |
| IVF insert 10K | 10.03 s | 24x slower than flat |
| Multi-head attention 12 heads | 6.6x overhead | Poor scaling |
| Small batch quantize (n=10) | 200 vec/s | 57x slower than batched |
| Neural decode | 9,491 vec/s | 8.8x slower than encode |
| IVF at small scale | -46% throughput | Slower than flat |
| GPU small transpose | 19x slower than CPU | Wrong routing |

---

## Appendix: Test Environment

- **Total Tests:** 1,077
- **Total Duration:** 444.158 seconds
- **Platform:** macOS 26 / Darwin 25.1.0
- **Metal Version:** Metal 4.0
- **SDK:** iOS 26 / macOS 26 minimum deployment

### Slowest Individual Tests

| Test | Duration | Category |
|------|----------|----------|
| testIVFThroughputVsFlat | 174.7 s | Benchmark |
| testRecallVsNprobeWithRandomQueries | 75.5 s | Validation |
| testIVFQualityVsFAISSBenchmarks | 49.0 s | Benchmark |
| testRecallAtScale | 27.2 s | Benchmark |
| testForEmbedKitSharingWaitsForSubsystemInit | 10.1 s | Integration |
| testSimulatedEmbedKitUsage | 10.1 s | Integration |
| testIVFSearch10K_128D | 10.1 s | Performance |

---

## Additional Findings (Automated Log Parse)

This section supplements the above analysis with findings extracted from the latest test run log (1,077 tests, 444.16s). Long individual test durations are acceptable given the intent for exhaustive coverage; the focus here is on test coverage quality and signal strength.

### Zero-Duration Tests Summary

- Total zero-duration tests: 171
- Zero failures, zero skipped
- These are typically either trivially fast, assertion-light, or not exercising a measurable code path. They can still be valid, but warrant a quick audit to ensure they provide coverage and guard against regressions.

Per-suite concentrations (higher counts first):

| Suite | Zero-Duration Count |
|-------|--------------------:|
| FallbackProviderTests | 33 |
| PipelineCacheKeyTests | 19 |
| SIMDFallbackTests | 15 |
| ThermalStateMonitorTests | 13 |
| PipelineRegistryTests | 13 |
| VectorAccelerateTests | 12 |
| SIMDOptimizerTests | 12 |
| MetalSubsystemTests | 9 |
| KMeansConfigurationTests | 8 |
| IVFSearchConfigurationTests | 8 |
| AdaptiveNlistTests | 6 |
| OptimizedKernelsTests | 4 |
| IntrospectionTests | 4 |
| Metal4CompilerConfigurationTests | 3 |
| IVFShaderArgsTests | 3 |
| ClusteringShaderArgsTests | 3 |
| IVFResultTypesTests | 2 |
| IVFQuantizationTests | 2 |
| ValidationEdgeCaseTests | 1 |
| MemoryMapManagerTests | 1 |

Full list of zero-duration test cases is available at `test-analysis/candidates/zero_duration.tsv` (generated from the attached log). Consider a quick audit of these tests for:

- Assertions: Ensure each test has at least one meaningful assertion.
- Execution path: Verify the code under test is actually exercised (no early returns or stubbed paths only).
- Measurement: Where applicable, wrap critical paths in `measure {}` and/or add `XCTClockMetric`, `XCTCPUMetric`, `XCTMemoryMetric` to capture performance signal.
- Rounding artifacts: Very fast tests can legitimately show `0.000` due to timer resolution; if expected, add a brief comment or rename to reflect semantics (e.g., “construction default values”).

### Context on Long-Running Tests (for SUT performance)

While long durations are acceptable for exhaustiveness, they do indicate hotspots in the source under test and are good targets for optimization work:

- Product quantization training (7.1s) — KMeans/codebook training and memory traffic dominate.
- IVF index build/insert (10.0s for 10K 128D) — coarse quantizer training and candidate structure creation.
- Large-scale recall validations (27–175s) — heavy search workloads and data movement.
- Multi-head attention kernels (1.9–5.2s cases; 163 ms per run in your benchmarks) — per-head dispatch overhead and memory layout.

These align with the bottlenecks you’ve documented above. As you iterate on kernel and memory-layout improvements, keep these tests as sentinel workloads to detect improvements/regressions in the SUT.

### Additional Signals Worth a Quick Look

The log also reported several “100.0%” metrics (e.g., recall and consistency). These are not necessarily problematic, but are worth verifying that the datasets/parameters aren’t overly favorable:

- IVF auto/quantized recall, full nprobe recall, benchmark harness recall, batch vs single consistency.

If intended (e.g., sanity checks), add a short rationale in the tests or use slightly more challenging synthetic data to avoid overfitting to trivial cases.
