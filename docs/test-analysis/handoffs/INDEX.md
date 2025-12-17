# Kernel Optimization Handoffs

This directory contains self-contained handoff documents for an external kernel specialist to work on high-complexity GPU optimization tasks.

## Quick Start

1. Read `CONTEXT.md` first for project overview
2. Pick a handoff document based on priority
3. All file paths are relative to repository root

## Handoff Documents

| ID | File | Issue | Priority | Complexity | Est. Impact |
|----|------|-------|----------|------------|-------------|
| 1 | `HANDOFF_1_KMEANS_GPU_ACCELERATION.md` | PQ/IVF training 14-24x too slow | CRITICAL | HIGH | 10x+ speedup |
| 2 | `HANDOFF_2_NEURAL_DECODE_OPTIMIZATION.md` | Decode 8.8x slower than encode | HIGH | MEDIUM | 5x+ speedup |
| 3 | `HANDOFF_3_MULTIHEAD_ATTENTION.md` | Multi-head 6.6x overhead | MEDIUM | MEDIUM | 3x+ speedup |

## Priority Rationale

### Handoff 1 (CRITICAL)
- Blocks production use of PQ and IVF indexing
- 7+ second training times are unacceptable
- Most impactful fix for overall library usability

### Handoff 2 (HIGH)
- Asymmetric encode/decode breaks real-time retrieval
- Decode is on hot path for re-ranking pipelines
- Relatively straightforward fix (memory layout)

### Handoff 3 (MEDIUM)
- Affects transformer-based similarity models
- Single-head works fine; multi-head is bonus feature
- Complex fix but well-understood problem

## Repository Clone

```bash
git clone <repo-url>
cd VectorAccelerate
swift build  # Verify build works
swift test   # Run full test suite (~7 minutes)
```

## Verification Commands

Each handoff includes specific test commands. General commands:

```bash
# Build
swift build -c release

# Run specific benchmark
swift test --filter "PerformanceBenchmark"

# Run all tests
swift test --parallel

# Profile Metal shaders
xcrun metal -c Sources/VectorAccelerate/Metal/Shaders/*.metal -std=metal3.0
```

## Deliverable Format

For each handoff, please provide:

1. **Modified .metal file(s)** with optimized kernel(s)
2. **Modified .swift file(s)** with updated kernel wrapper
3. **Benchmark results** showing before/after comparison
4. **Brief write-up** explaining the optimization approach

## Contact

Questions about the codebase or requirements can be directed back to the primary developer via the same channel this handoff was delivered.
