//
//  CPUTopK.swift
//  VectorAccelerate
//
//  CPU fallback for L2² top-K nearest neighbor search.
//  Used when GPUDecisionEngine votes against GPU for small-N queries.
//
//  Reads directly from GPUVectorStorage's shared-mode MTLBuffer (zero-copy)
//  and uses vDSP_distancesq for vectorized L2² computation.
//

import Foundation
import Accelerate
@preconcurrency import Metal
import VectorCore

/// CPU-based L2² top-K search reading from a shared MTLBuffer.
///
/// Uses a bounded max-heap to maintain the K nearest neighbors in O(N log K) time.
/// Suitable for small dataset sizes where GPU kernel launch overhead dominates.
struct CPUTopK {

    /// Perform L2² top-K search on CPU.
    ///
    /// - Parameters:
    ///   - query: Query vector as [Float]
    ///   - buffer: Shared-mode MTLBuffer containing dataset vectors
    ///   - allocatedSlots: Number of occupied slots in the buffer
    ///   - dimension: Vector dimension
    ///   - k: Number of results to return
    ///   - deletionMask: Mask indicating which slots are deleted
    ///   - handleAllocator: Maps slot indices to stable handles
    /// - Returns: Array of (handle, distance) pairs sorted by distance ascending
    static func search(
        query: [Float],
        buffer: any MTLBuffer,
        allocatedSlots: Int,
        dimension: Int,
        k: Int,
        deletionMask: DeletionMask,
        handleAllocator: HandleAllocator
    ) -> [IndexSearchResult] {
        guard allocatedSlots > 0, k > 0 else { return [] }

        let effectiveK = min(k, allocatedSlots)
        let basePtr = buffer.contents().bindMemory(to: Float.self, capacity: allocatedSlots * dimension)

        // Max-heap: keeps the K smallest distances. Heap element = (distance, slotIndex).
        var heap: [(distance: Float, slot: Int)] = []
        heap.reserveCapacity(effectiveK)

        query.withUnsafeBufferPointer { queryBuf in
            guard let queryPtr = queryBuf.baseAddress else { return }

            for slot in 0..<allocatedSlots {
                if deletionMask.isDeleted(slot) { continue }
                guard handleAllocator.handle(for: UInt32(slot)) != nil else { continue }

                let vecPtr = basePtr.advanced(by: slot * dimension)

                var distSq: Float = 0
                vDSP_distancesq(queryPtr, 1, vecPtr, 1, &distSq, vDSP_Length(dimension))

                if heap.count < effectiveK {
                    heap.append((distSq, slot))
                    if heap.count == effectiveK {
                        // Build max-heap
                        buildMaxHeap(&heap)
                    }
                } else if distSq < heap[0].distance {
                    // Replace max with this smaller distance, sift down
                    heap[0] = (distSq, slot)
                    siftDown(&heap, from: 0, count: effectiveK)
                }
            }
        }

        // Sort by distance ascending and convert to IndexSearchResult
        heap.sort { $0.distance < $1.distance }
        return heap.compactMap { entry in
            guard let handle = handleAllocator.handle(for: UInt32(entry.slot)) else { return nil }
            return IndexSearchResult(id: handle, distance: entry.distance)
        }
    }

    // MARK: - Max-Heap Utilities

    private static func buildMaxHeap(_ heap: inout [(distance: Float, slot: Int)]) {
        let count = heap.count
        for i in stride(from: count / 2 - 1, through: 0, by: -1) {
            siftDown(&heap, from: i, count: count)
        }
    }

    private static func siftDown(_ heap: inout [(distance: Float, slot: Int)], from index: Int, count: Int) {
        var parent = index
        while true {
            let left = 2 * parent + 1
            let right = 2 * parent + 2
            var largest = parent

            if left < count && heap[left].distance > heap[largest].distance {
                largest = left
            }
            if right < count && heap[right].distance > heap[largest].distance {
                largest = right
            }
            if largest == parent { break }
            heap.swapAt(parent, largest)
            parent = largest
        }
    }
}
