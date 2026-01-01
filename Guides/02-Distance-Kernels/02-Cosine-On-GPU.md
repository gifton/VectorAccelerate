# 2.2 Cosine Similarity on GPU

> **Optimizing for normalized embeddings—the fast path and the general path.**

---

## The Concept

Cosine similarity measures the angle between vectors, ignoring magnitude:

```
cos(q, d) = (q · d) / (‖q‖ × ‖d‖)

Where:
  q · d = Σᵢ qᵢ × dᵢ       (dot product)
  ‖q‖ = √Σᵢ qᵢ²           (L2 norm)
```

Cosine distance (for ranking) is:
```
cosine_distance(q, d) = 1 - cos(q, d)
```

---

## Why It Matters

Many embedding models produce **pre-normalized** vectors (‖v‖ = 1). This is a huge optimization opportunity:

```
Pre-normalized vectors:
  cos(q, d) = q · d       (Just a dot product!)

Non-normalized vectors:
  cos(q, d) = (q · d) / (‖q‖ × ‖d‖)
  Requires: dot product + 2 norms + 2 sqrts + 1 division

Pre-normalized is 3-5× faster!
```

VectorAccelerate exploits this with two code paths.

---

## The Technique: Two Code Paths

### Fast Path: Pre-Normalized Inputs

When vectors are already normalized, cosine similarity is just a dot product:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/CosineSimilarity.metal:70-118

kernel void cosine_similarity_normalized_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint queryIdx = tid.x;
    const uint dbIdx = tid.y;

    if (queryIdx >= params.numQueries || dbIdx >= params.numDatabase) {
        return;
    }

    device const float* query = queryVectors + (queryIdx * params.strideQuery);
    device const float* database = databaseVectors + (dbIdx * params.strideDatabase);

    // Simple dot product - inputs are unit vectors
    float4 dot_acc = float4(0.0f);
    const uint simd_blocks = params.dimension / 4;

    device const float4* query4 = (device const float4*)query;
    device const float4* database4 = (device const float4*)database;

    for (uint i = 0; i < simd_blocks; ++i) {
        dot_acc = fma(query4[i], database4[i], dot_acc);
    }

    float dotProduct = dot_acc.x + dot_acc.y + dot_acc.z + dot_acc.w;

    // Handle remainder...

    // Clamp to [-1, 1] for numerical safety
    float similarity = clamp(dotProduct, -1.0f, 1.0f);

    // Output similarity or distance
    float result = params.outputDistance ? (1.0f - similarity) : similarity;
    similarities[queryIdx * params.strideOutput + dbIdx] = result;
}
```

### General Path: Non-Normalized Inputs

When vectors aren't normalized, we compute norms on-the-fly:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/CosineSimilarity.metal:121-182

kernel void cosine_similarity_general_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    // ... bounds check ...

    // Three accumulators: dot product and two norms
    float4 dot_acc = float4(0.0f);
    float4 qnorm_acc = float4(0.0f);
    float4 dnorm_acc = float4(0.0f);

    device const float4* query4 = (device const float4*)query;
    device const float4* database4 = (device const float4*)database;

    // Compute all three metrics in a single pass
    for (uint i = 0; i < simd_blocks; ++i) {
        float4 q = query4[i];
        float4 d = database4[i];

        dot_acc = fma(q, d, dot_acc);      // q · d
        qnorm_acc = fma(q, q, qnorm_acc);  // ‖q‖²
        dnorm_acc = fma(d, d, dnorm_acc);  // ‖d‖²
    }

    // Horizontal reductions
    float dotProduct = dot_acc.x + dot_acc.y + dot_acc.z + dot_acc.w;
    float queryNormSq = qnorm_acc.x + qnorm_acc.y + qnorm_acc.z + qnorm_acc.w;
    float databaseNormSq = dnorm_acc.x + dnorm_acc.y + dnorm_acc.z + dnorm_acc.w;

    // Compute cosine similarity with numerical stability
    float denominator = sqrt(queryNormSq * databaseNormSq);
    float similarity;

    if (denominator > EPSILON) {
        similarity = dotProduct / denominator;
        similarity = clamp(similarity, -1.0f, 1.0f);
    } else {
        similarity = 0.0f;  // Handle zero vectors
    }

    float result = params.outputDistance ? (1.0f - similarity) : similarity;
    similarities[queryIdx * params.strideOutput + dbIdx] = result;
}
```

### Single-Pass Optimization

Notice how the general kernel computes **three values in one loop**:

```
Loop iteration i:
  Load q[i], d[i]
  dot_acc   += q[i] * d[i]    ← FMA
  qnorm_acc += q[i] * q[i]    ← FMA
  dnorm_acc += d[i] * d[i]    ← FMA

All three FMAs use the same loaded data!
Memory reads: 2 (q and d)
Compute: 3 FMAs

Without single-pass:
  Pass 1: dot product  → 2 memory reads
  Pass 2: query norm   → 1 memory read
  Pass 3: database norm → 1 memory read
  Total: 4 memory reads

Single-pass saves 50% memory bandwidth!
```

---

## Dimension-Optimized Cosine Kernels

Like L2, cosine has specialized kernels for common dimensions:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/CosineSimilarity.metal:189-273

kernel void cosine_similarity_384_kernel(
    device const float* queryVectors [[buffer(0)]],
    device const float* databaseVectors [[buffer(1)]],
    device float* similarities [[buffer(2)]],
    constant CosineSimilarityParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    // ... bounds check ...

    device const float4* query4 = (device const float4*)(queryVectors + queryIdx * 384);
    device const float4* database4 = (device const float4*)(databaseVectors + dbIdx * 384);

    float4 dot_acc0 = float4(0.0f);
    float4 dot_acc1 = float4(0.0f);

    if (params.inputsNormalized) {
        // Fast Path: Just dot product with unrolling
        for (uint i = 0; i < 96; i += 8) {
            dot_acc0 = fma(query4[i+0], database4[i+0], dot_acc0);
            dot_acc1 = fma(query4[i+1], database4[i+1], dot_acc1);
            dot_acc0 = fma(query4[i+2], database4[i+2], dot_acc0);
            dot_acc1 = fma(query4[i+3], database4[i+3], dot_acc1);
            dot_acc0 = fma(query4[i+4], database4[i+4], dot_acc0);
            dot_acc1 = fma(query4[i+5], database4[i+5], dot_acc1);
            dot_acc0 = fma(query4[i+6], database4[i+6], dot_acc0);
            dot_acc1 = fma(query4[i+7], database4[i+7], dot_acc1);
        }

        float4 total_dot = dot_acc0 + dot_acc1;
        float dotProduct = total_dot.x + total_dot.y + total_dot.z + total_dot.w;
        // ...
    } else {
        // General Path: All three metrics with unrolling
        float4 qnorm_acc0 = 0, qnorm_acc1 = 0;
        float4 dnorm_acc0 = 0, dnorm_acc1 = 0;

        for (uint i = 0; i < 96; i += 8) {
            float4 q0=query4[i+0]; float4 d0=database4[i+0];
            // ... load q1-q7, d1-d7 ...

            // Interleaved accumulation for ILP
            dot_acc0 = fma(q0, d0, dot_acc0);
            qnorm_acc0 = fma(q0, q0, qnorm_acc0);
            dnorm_acc0 = fma(d0, d0, dnorm_acc0);
            // ... continue for all 8 ...
        }
        // ...
    }
}
```

---

## Numerical Stability

Cosine similarity can produce values outside [-1, 1] due to floating-point error:

```metal
// 📍 See: Sources/VectorAccelerate/Metal/Shaders/CosineSimilarity.metal:42-57

constant float EPSILON = 1e-8f;  // Tighter epsilon for cosine

inline float calculate_similarity(
    float dotProduct,
    float queryNormSq,
    float databaseNormSq,
    bool outputDistance
) {
    float denominator = sqrt(queryNormSq * databaseNormSq);

    float similarity;
    if (denominator > EPSILON) {
        similarity = dotProduct / denominator;
        // CRITICAL: Clamp to valid range
        similarity = clamp(similarity, -1.0f, 1.0f);
    } else {
        // Handle zero vectors gracefully
        similarity = 0.0f;
    }

    return outputDistance ? (1.0f - similarity) : similarity;
}
```

⚠️ **Without clamping**, you might get:
- `similarity = 1.0000001` → causes issues in downstream operations
- `similarity = NaN` → from zero vector division

---

## 🔗 VectorCore Connection

VectorCore's cosine kernels have the same normalized/unnormalized split:

```swift
// VectorCore: Cosine for pre-normalized vectors
public func cosineSimilarityNormalized<V: VectorProtocol>(
    _ a: V, _ b: V
) -> Float where V.Scalar == Float {
    // Just dot product!
    return dotProduct(a, b)
}

// VectorCore: General cosine similarity
public func cosineSimilarity<V: VectorProtocol>(
    _ a: V, _ b: V
) -> Float where V.Scalar == Float {
    let dot = dotProduct(a, b)
    let normA = sqrt(dotProduct(a, a))
    let normB = sqrt(dotProduct(b, b))

    guard normA > 0 && normB > 0 else { return 0 }
    return dot / (normA * normB)
}
```

The GPU version achieves the same optimization but with thousands of threads.

---

## 🔗 VectorIndex Connection

VectorIndex's indices use cosine for normalized embeddings:

```swift
// VectorIndex: Cosine distance for normalized embeddings
public enum DistanceMetric {
    case euclidean     // L2 distance
    case cosine        // 1 - cosine_similarity (for normalized)
    case dotProduct    // -dot (for MIPS)
}

// At index creation:
let index = HNSWIndex<D>(metric: .cosine)

// Distance is computed as:
// 1 - dot(q, d) for normalized vectors
```

VectorAccelerate's `CosineSimilarityKernel` can output either similarity or distance:

```swift
// Output similarity (higher = more similar)
let similarities = try await kernel.compute(
    queries: queries,
    database: database,
    outputDistance: false
)

// Output distance (lower = more similar)
let distances = try await kernel.compute(
    queries: queries,
    database: database,
    outputDistance: true  // Returns 1 - similarity
)
```

---

## Using CosineSimilarityKernel

### Basic Usage

```swift
import VectorAccelerate

let context = try await Metal4Context()
let kernel = try await CosineSimilarityKernel(context: context)

// For pre-normalized embeddings (fast path)
let similarities = try await kernel.compute(
    queries: normalizedQueries,
    database: normalizedDatabase,
    inputsNormalized: true,   // Enables fast path
    outputDistance: false     // Return similarity, not distance
)

// For non-normalized embeddings
let distances = try await kernel.compute(
    queries: rawQueries,
    database: rawDatabase,
    inputsNormalized: false,  // General path with norms
    outputDistance: true      // Return distance
)
```

### Pre-Normalization Strategy

If your data isn't pre-normalized, you have two options:

```swift
// Option 1: Normalize once at insert time (RECOMMENDED)
let normalizedVectors = vectors.map { v in
    let norm = sqrt(v.reduce(0) { $0 + $1 * $1 })
    return v.map { $0 / norm }
}
// Store normalizedVectors in index
// Use inputsNormalized: true for all searches

// Option 2: Normalize on-the-fly (slower per query)
// Use inputsNormalized: false
// Good for: data that changes frequently, or mixed normalized/unnormalized
```

Option 1 is almost always better—normalize once, search fast forever.

---

## Performance Comparison

```
100 queries × 100K database × 768D:

Pre-normalized (fast path):
  GPU: 4.2 ms
  Operations: Just dot product

Non-normalized (general path):
  GPU: 12.8 ms
  Operations: dot + 2 norms + sqrt + divide

Speedup from pre-normalization: 3×
```

This matches our theoretical analysis—pre-normalized skips ~70% of the compute.

---

## Key Takeaways

1. **Pre-normalize when possible**: 3-5× speedup for just a dot product

2. **Single-pass optimization**: Compute dot product and norms together to save memory bandwidth

3. **Numerical stability**: Always clamp cosine to [-1, 1] and handle zero vectors

4. **Output mode matters**: Use `outputDistance: true` when you need a distance metric for sorting

5. **Same dimension optimizations**: 384/512/768/1536 kernels provide additional speedups

---

## Next Up

How do we compute entire distance matrices efficiently?

**[→ 2.3 Batch Distance Matrix](./03-Batch-Distance-Matrix.md)**

---

*Guide 2.2 of 2.4 • Chapter 2: Distance Kernels*
