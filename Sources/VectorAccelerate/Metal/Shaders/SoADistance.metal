// VectorAccelerate: Lane-major SoA distance kernels
//
// Reads a candidate buffer in VectorCore's frozen SoA layout (0.3.0):
//   element(lane ℓ, candidate j) == buffer[ℓ * count + j], each a float4 (4 dims).
// One thread per candidate; loop lanes. Adjacent threads (j, j+1) read adjacent float4s within a
// lane block ⇒ coalesced. See docs/zero-copy-soa-planning-brief.md §2 / Docs/SoA_Layout_Contract.md.

#include "Metal4Common.h"

struct SoAL2Params { uint count; uint lanes; uint computeSqrt; uint _pad; };
struct SoACosineParams { uint count; uint lanes; float queryNormSq; uint _pad; };

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
        float4 d = query[l] - candidates[l * p.count + j];
        acc = fma(d, d, acc);
    }
    float sum = acc.x + acc.y + acc.z + acc.w;
    distances[j] = p.computeSqrt ? sqrt(sum) : sum;
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
    float4 cNormAcc = float4(0.0f);
    for (uint l = 0; l < p.lanes; ++l) {
        float4 q = query[l];
        float4 c = candidates[l * p.count + j];
        dotAcc = fma(q, c, dotAcc);
        cNormAcc = fma(c, c, cNormAcc);
    }
    float dot = dotAcc.x + dotAcc.y + dotAcc.z + dotAcc.w;
    float cNormSq = cNormAcc.x + cNormAcc.y + cNormAcc.z + cNormAcc.w;
    // BE3 parity: sqrt(a)*sqrt(b) (overflow-safe), FLT_MIN floor, NaN-preserving clamp.
    float denom = sqrt(p.queryNormSq) * sqrt(cNormSq);
    float raw = (denom < FLT_MIN) ? 0.0f : (dot / denom);
    float sim = isnan(raw) ? raw : clamp(raw, -1.0f, 1.0f);
    distances[j] = 1.0f - sim;
}
