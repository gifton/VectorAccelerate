//
//  SoACandidateSet.swift
//  VectorAccelerate
//
//  An immutable, GPU-ready candidate set in VectorCore's page-aligned SoA layout, bridged into a
//  zero-copy MTLBuffer (borrow mode). Build once, score many times.
//
//  See docs/superpowers/plans/2026-06-08-zero-copy-soa-scoring.md and
//  docs/zero-copy-soa-planning-brief.md (SoA Layout Contract).
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// An immutable, GPU-ready candidate set in VectorCore's page-aligned SoA layout, bridged into a
/// zero-copy `MTLBuffer` (borrow mode). Build once, score many times against it.
///
/// Borrow mode (SoA Layout Contract §5): the `SoA` is held strongly and frees its allocation on
/// `deinit`, so it must outlive `buffer`. Holding both on this one object guarantees that — and
/// callers that hand `buffer` to the GPU must keep this object alive until the GPU completes.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class SoACandidateSet<V: SoACompatible>: @unchecked Sendable {
    /// Owns the SoA allocation; pinned for the buffer's lifetime (borrow mode).
    public let soa: SoA<V>
    /// The candidate buffer the kernel reads (zero-copy alias, or a staged copy — see `isZeroCopy`).
    ///
    /// - Important: read-only by contract. In zero-copy mode this aliases `soa`'s allocation, which is
    ///   freed on `soa`'s `deinit`. Do not submit `buffer` to the GPU on any path that does not keep
    ///   this `SoACandidateSet` alive until the GPU completes (borrow mode).
    public let buffer: any MTLBuffer
    /// Frozen layout descriptor (lanes, count, strides) — the kernel's source of truth.
    public let layout: SoALayout
    /// True when `buffer` is a zero-copy alias of the SoA allocation; false when a staged copy was made.
    public let isZeroCopy: Bool

    public init(candidates: [V], device: MetalDevice) throws {
        let built = SoA<V>.build(from: candidates, pageAligned: true)
        self.soa = built
        self.layout = built.layoutDescriptor

        if let (base, byteCount) = built.pageAlignedBytes,
           let noCopy = device.makeNoCopyBuffer(
               bytes: UnsafeMutableRawPointer(mutating: base),
               length: byteCount,                       // page-rounded allocatedByteCount
               options: .storageModeShared,
               deallocator: nil) {                      // borrow mode: SoA frees on deinit
            self.buffer = noCopy
            self.isZeroCopy = true
        } else {
            // Staged fallback (e.g. not page-aligned): copy the logical bytes into a fresh buffer.
            let logical = built.layoutDescriptor.logicalByteCount
            guard let staged = device.rawDevice.makeBuffer(length: logical, options: .storageModeShared) else {
                throw VectorError.bufferAllocationFailed(size: logical)
            }
            built.withUnsafeRawBuffer { raw in
                if let src = raw.baseAddress { memcpy(staged.contents(), src, logical) }
            }
            self.buffer = staged
            self.isZeroCopy = false
        }
        self.buffer.label = "SoACandidateSet<\(V.self)>"
    }
}
