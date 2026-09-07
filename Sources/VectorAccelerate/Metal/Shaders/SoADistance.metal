// VectorAccelerate: Lane-major SoA distance kernels
//
// Reads a candidate buffer in VectorCore's frozen SoA layout (0.3.0):
//   element(lane ℓ, candidate j) == buffer[ℓ * count + j], each a float4 (4 dims).
// One thread per candidate; loop lanes. Adjacent threads (j, j+1) read adjacent float4s within a
// lane block ⇒ coalesced. See docs/zero-copy-soa-planning-brief.md §2 / Docs/SoA_Layout_Contract.md.

#include "Metal4Common.h"

struct SoAL2Params { uint count; uint lanes; uint computeSqrt; uint _pad; };
struct SoACosineParams { uint count; uint lanes; };

kernel void soa_l2_distance(
    device const float4* query      [[buffer(0)]],   // `lanes` elements
    device const float4* candidates [[buffer(1)]],   // lanes*count, lane-major
    device float*        distances  [[buffer(2)]],
    constant SoAL2Params& p         [[buffer(3)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= p.count) return;
    float4 acc = float4(0.0f);
    for (uint l = 0; l < p.lanes; ++l) {
        float4 d = query[l] - candidates[(ulong)l * p.count + j];
        acc = fma(d, d, acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;
    distances[j] = p.computeSqrt
        ? va_euclidean_finalize(sum, reinterpret_cast<device const float*>(query),
                               reinterpret_cast<device const float*>(candidates + j),
                               p.lanes * 4, (ulong)p.count * 4)
        : sum;
}

kernel void soa_cosine_distance(
    device const float4* query      [[buffer(0)]],
    device const float4* candidates [[buffer(1)]],
    device float*        distances  [[buffer(2)]],
    constant SoACosineParams& p     [[buffer(3)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= p.count) return;
    float4 dotAcc = float4(0.0f);
    float4 qNormAcc = float4(0.0f);
    float4 cNormAcc = float4(0.0f);
    for (uint l = 0; l < p.lanes; ++l) {
        float4 q = query[l];
        float4 c = candidates[(ulong)l * p.count + j];
        dotAcc = fma(q, c, dotAcc);
        qNormAcc = fma(q, q, qNormAcc);     // query norm computed in-kernel (self-contained; symmetric with cNorm)
        cNormAcc = fma(c, c, cNormAcc);
    }
    float dot = dotAcc.x + dotAcc.y + dotAcc.z + dotAcc.w;
    float qNormSq = qNormAcc.x + qNormAcc.y + qNormAcc.z + qNormAcc.w;
    float cNormSq = cNormAcc.x + cNormAcc.y + cNormAcc.z + cNormAcc.w;

    // Overflow/underflow rescue (AUDIT-2 VA2-008): lane-major variant of Metal4Common.h's
    // va_cosine_rescaled_terms — recompute in the pre-scaled normalization domain when the
    // naive accumulators overflowed (Inf) or collapsed to 0. The per-vector scales cancel in
    // the similarity quotient. Cold path only.
    if (va_cosine_accumulators_unreliable(dot, qNormSq, cNormSq)) {
        float qMax = 0.0f;
        float cMax = 0.0f;
        for (uint l = 0; l < p.lanes; ++l) {
            float4 aq = fabs(query[l]);
            float4 ac = fabs(candidates[(ulong)l * p.count + j]);
            qMax = max(qMax, max(max(aq.x, aq.y), max(aq.z, aq.w)));
            cMax = max(cMax, max(max(ac.x, ac.y), max(ac.z, ac.w)));
        }
        const float qs = 1.0f / clamp(qMax, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);
        const float cs = 1.0f / clamp(cMax, VA_NORM_MIN_DENOM, VA_NORM_MAX_DENOM);
        float4 dA = float4(0.0f);
        float4 qA = float4(0.0f);
        float4 cA = float4(0.0f);
        for (uint l = 0; l < p.lanes; ++l) {
            float4 q = query[l] * qs;
            float4 c = candidates[(ulong)l * p.count + j] * cs;
            dA = fma(q, c, dA);
            qA = fma(q, q, qA);
            cA = fma(c, c, cA);
        }
        dot = dA.x + dA.y + dA.z + dA.w;
        qNormSq = qA.x + qA.y + qA.z + qA.w;
        cNormSq = cA.x + cA.y + cA.z + cA.w;
    }

    // Shared finalization (Metal4Common.h): two-stage precise::divide (never the
    // reassociation-prone sqrt(a)*sqrt(b) product), FLT_MIN floor, NaN-preserving clamp.
    distances[j] = 1.0f - va_cosine_similarity_finalize(dot, qNormSq, cNormSq);
}
