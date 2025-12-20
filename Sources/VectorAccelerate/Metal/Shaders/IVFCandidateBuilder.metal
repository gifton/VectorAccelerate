// VectorAccelerate: IVF Candidate Builder Shaders
//
// GPU kernels for building candidate lists in IVF search.
// This eliminates the CPU bottleneck in candidate list construction.
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Parameter Structures

struct IVFCandidateCountParams {
    uint32_t num_queries;    // Q
    uint32_t nprobe;         // Number of probes per query
    uint32_t num_lists;      // Total number of IVF lists (nlist)
    uint32_t padding;
};

struct IVFCandidateBuildParams {
    uint32_t num_queries;    // Q
    uint32_t nprobe;         // Number of probes per query
    uint32_t num_lists;      // Total number of IVF lists (nlist)
    uint32_t total_candidates; // Sum of all candidate counts
};

// MARK: - Candidate Counting Kernel

/// Counts the number of candidate vectors for each query based on probed lists.
///
/// For each query, iterates through its nprobe nearest centroid indices and
/// sums up the sizes of the corresponding IVF lists. De-duplicates list indices
/// to handle ties in coarse search.
///
/// Input:
///   - nearestCentroids: [Q × nprobe] - centroid indices from coarse search
///   - listOffsets: [nlist + 1] - CSR offsets for IVF lists
///
/// Output:
///   - candidateCounts: [Q] - number of candidates per query
kernel void ivf_count_candidates(
    device const uint* nearestCentroids [[buffer(0)]],
    device const uint* listOffsets [[buffer(1)]],
    device uint* candidateCounts [[buffer(2)]],
    constant IVFCandidateCountParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_queries) return;

    const uint q = tid;
    const uint nprobe = params.nprobe;
    const uint num_lists = params.num_lists;

    // Track which lists we've already counted (de-duplication)
    // For small nprobe (typical: 1-64), linear search is efficient
    uint seenLists[64];  // Max nprobe we support in this kernel
    uint numSeen = 0;

    uint count = 0;

    for (uint p = 0; p < nprobe && p < 64; ++p) {
        uint listIdx = nearestCentroids[q * nprobe + p];

        // Skip invalid/sentinel indices
        if (listIdx >= num_lists) continue;

        // Check if we've already seen this list (de-dup)
        bool duplicate = false;
        for (uint i = 0; i < numSeen; ++i) {
            if (seenLists[i] == listIdx) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        // Mark as seen
        seenLists[numSeen++] = listIdx;

        // Add list size to count
        uint listStart = listOffsets[listIdx];
        uint listEnd = listOffsets[listIdx + 1];
        count += (listEnd - listStart);
    }

    candidateCounts[q] = count;
}

// MARK: - Prefix Sum Kernel (Parallel Scan)

struct PrefixSumParams {
    uint32_t num_elements;   // Number of elements to scan
    uint32_t padding[3];
};

/// Computes inclusive prefix sum on candidateCounts to produce candidateOffsets.
///
/// This is a simple single-threadgroup implementation for small arrays (Q < 1024).
/// For larger arrays, a multi-pass Blelloch scan would be needed.
///
/// Input:
///   - candidateCounts: [Q] - count per query
///
/// Output:
///   - candidateOffsets: [Q + 1] - CSR offsets (first element is 0)
kernel void ivf_prefix_sum_candidates(
    device const uint* candidateCounts [[buffer(0)]],
    device uint* candidateOffsets [[buffer(1)]],
    constant PrefixSumParams& params [[buffer(2)]],
    threadgroup uint* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    const uint n = params.num_elements;

    // Load to shared memory
    if (tid < n) {
        shared[tid] = candidateCounts[tid];
    } else {
        shared[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase
    for (uint stride = 1; stride < tg_size; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < tg_size) {
            shared[index] += shared[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Set root to 0 for exclusive scan
    if (tid == 0) {
        shared[tg_size - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < tg_size) {
            uint temp = shared[index];
            shared[index] += shared[index - stride];
            shared[index - stride] = temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output (exclusive prefix sum)
    // candidateOffsets[0] = 0
    // candidateOffsets[i+1] = candidateOffsets[i] + candidateCounts[i]
    if (tid == 0) {
        candidateOffsets[0] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < n) {
        // For exclusive scan, shift by 1
        candidateOffsets[tid + 1] = shared[tid] + candidateCounts[tid];
    }
}

// MARK: - Simple CPU-style Prefix Sum (Sequential)

/// Simple sequential prefix sum for small arrays.
/// Dispatched with a single thread when Q is small.
kernel void ivf_prefix_sum_sequential(
    device const uint* candidateCounts [[buffer(0)]],
    device uint* candidateOffsets [[buffer(1)]],
    constant PrefixSumParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;  // Single thread

    const uint n = params.num_elements;
    uint sum = 0;

    candidateOffsets[0] = 0;
    for (uint i = 0; i < n; ++i) {
        sum += candidateCounts[i];
        candidateOffsets[i + 1] = sum;
    }
}

// MARK: - Candidate List Builder Kernel

/// Builds the flat candidate lists using pre-computed offsets.
///
/// For each query, writes IVF entry indices and query IDs to the candidate buffers.
/// Each thread handles one query and writes its candidates to the appropriate range.
///
/// Input:
///   - nearestCentroids: [Q × nprobe] - centroid indices from coarse search
///   - listOffsets: [nlist + 1] - CSR offsets for IVF lists
///   - candidateOffsets: [Q + 1] - output offsets per query (from prefix sum)
///
/// Output:
///   - candidateIVFIndices: [total_candidates] - flat list of IVF entry indices
///   - candidateQueryIds: [total_candidates] - query ID for each candidate
kernel void ivf_build_candidates(
    device const uint* nearestCentroids [[buffer(0)]],
    device const uint* listOffsets [[buffer(1)]],
    device const uint* candidateOffsets [[buffer(2)]],
    device uint* candidateIVFIndices [[buffer(3)]],
    device uint* candidateQueryIds [[buffer(4)]],
    constant IVFCandidateBuildParams& params [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_queries) return;

    const uint q = tid;
    const uint nprobe = params.nprobe;
    const uint num_lists = params.num_lists;

    // Get write position for this query
    uint writePos = candidateOffsets[q];
    const uint writeEnd = candidateOffsets[q + 1];

    // Track which lists we've already processed (de-duplication)
    uint seenLists[64];  // Max nprobe we support
    uint numSeen = 0;

    for (uint p = 0; p < nprobe && p < 64 && writePos < writeEnd; ++p) {
        uint listIdx = nearestCentroids[q * nprobe + p];

        // Skip invalid/sentinel indices
        if (listIdx >= num_lists) continue;

        // Check if we've already processed this list (de-dup)
        bool duplicate = false;
        for (uint i = 0; i < numSeen; ++i) {
            if (seenLists[i] == listIdx) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        // Mark as seen
        seenLists[numSeen++] = listIdx;

        // Write all entries from this list
        uint listStart = listOffsets[listIdx];
        uint listEnd = listOffsets[listIdx + 1];

        for (uint ivfEntry = listStart; ivfEntry < listEnd && writePos < writeEnd; ++ivfEntry) {
            candidateIVFIndices[writePos] = ivfEntry;
            candidateQueryIds[writePos] = q;
            writePos++;
        }
    }
}

// MARK: - Combined Candidate Builder (Fused Version)

/// Fused kernel that combines counting, offset computation, and list building.
///
/// This version uses atomic operations to dynamically allocate output positions,
/// avoiding the need for a separate prefix sum pass. Best for small to medium
/// query batches where atomics are not a bottleneck.
///
/// Input:
///   - nearestCentroids: [Q × nprobe] - centroid indices from coarse search
///   - listOffsets: [nlist + 1] - CSR offsets for IVF lists
///
/// Output:
///   - candidateIVFIndices: [max_candidates] - flat list of IVF entry indices
///   - candidateQueryIds: [max_candidates] - query ID for each candidate
///   - candidateOffsets: [Q + 1] - output offsets per query
///   - totalCandidateCount: [1] - atomic counter for total candidates
kernel void ivf_build_candidates_fused(
    device const uint* nearestCentroids [[buffer(0)]],
    device const uint* listOffsets [[buffer(1)]],
    device uint* candidateIVFIndices [[buffer(2)]],
    device uint* candidateQueryIds [[buffer(3)]],
    device atomic_uint* totalCandidateCount [[buffer(4)]],
    device uint* perQueryOffsets [[buffer(5)]],
    device uint* perQueryCounts [[buffer(6)]],
    constant IVFCandidateCountParams& params [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_queries) return;

    const uint q = tid;
    const uint nprobe = params.nprobe;
    const uint num_lists = params.num_lists;

    // First pass: count candidates for this query
    uint seenLists[64];
    uint numSeen = 0;
    uint count = 0;

    for (uint p = 0; p < nprobe && p < 64; ++p) {
        uint listIdx = nearestCentroids[q * nprobe + p];
        if (listIdx >= num_lists) continue;

        bool duplicate = false;
        for (uint i = 0; i < numSeen; ++i) {
            if (seenLists[i] == listIdx) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        seenLists[numSeen++] = listIdx;
        count += listOffsets[listIdx + 1] - listOffsets[listIdx];
    }

    // Atomically allocate space for this query's candidates
    uint writeStart = atomic_fetch_add_explicit(totalCandidateCount, count, memory_order_relaxed);
    perQueryOffsets[q] = writeStart;
    perQueryCounts[q] = count;

    // Second pass: write candidates
    numSeen = 0;
    uint writePos = writeStart;

    for (uint p = 0; p < nprobe && p < 64; ++p) {
        uint listIdx = nearestCentroids[q * nprobe + p];
        if (listIdx >= num_lists) continue;

        bool duplicate = false;
        for (uint i = 0; i < numSeen; ++i) {
            if (seenLists[i] == listIdx) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        seenLists[numSeen++] = listIdx;

        uint listStart = listOffsets[listIdx];
        uint listEnd = listOffsets[listIdx + 1];

        for (uint ivfEntry = listStart; ivfEntry < listEnd; ++ivfEntry) {
            candidateIVFIndices[writePos] = ivfEntry;
            candidateQueryIds[writePos] = q;
            writePos++;
        }
    }
}
