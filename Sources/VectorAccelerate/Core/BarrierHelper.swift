//
//  BarrierHelper.swift
//  VectorAccelerate
//
//  Utilities for Metal 4 barrier placement in fused pipelines
//

import Foundation
@preconcurrency import Metal

// MARK: - Pipeline Stage

/// Metal pipeline stages for barrier specification
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct PipelineStage: OptionSet, Sendable {
    public let rawValue: UInt

    public init(rawValue: UInt) {
        self.rawValue = rawValue
    }

    /// Compute dispatch operations
    public static let dispatch = PipelineStage(rawValue: 1 << 0)

    /// Blit/copy operations
    public static let blit = PipelineStage(rawValue: 1 << 1)

    /// Render operations (not typically used in VectorAccelerate)
    public static let render = PipelineStage(rawValue: 1 << 2)

    /// All stages
    public static let all: PipelineStage = [.dispatch, .blit, .render]
}

// MARK: - Barrier Type

/// Types of barriers supported in Metal 4
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum BarrierType: Sendable {
    /// Resource-specific barrier - waits for specific buffers
    case resource([any MTLBuffer])

    /// Memory barrier - ensures all memory operations complete
    case memory

    /// Pass barrier - full barrier between encoder passes
    case pass

    /// Fence - explicit synchronization point
    case fence
}

// MARK: - Barrier Descriptor

/// Describes a barrier to be inserted in a pipeline
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct BarrierDescriptor: Sendable {
    /// Type of barrier
    public let type: BarrierType

    /// Stages that must complete before barrier
    public let beforeStages: PipelineStage

    /// Stages that wait on barrier
    public let afterStages: PipelineStage

    /// Optional label for debugging
    public let label: String?

    public init(
        type: BarrierType,
        beforeStages: PipelineStage = .dispatch,
        afterStages: PipelineStage = .dispatch,
        label: String? = nil
    ) {
        self.type = type
        self.beforeStages = beforeStages
        self.afterStages = afterStages
        self.label = label
    }

    /// Create a dispatch-to-dispatch barrier for specific resources
    public static func dispatchBarrier(resources: [any MTLBuffer], label: String? = nil) -> BarrierDescriptor {
        BarrierDescriptor(
            type: .resource(resources),
            beforeStages: .dispatch,
            afterStages: .dispatch,
            label: label
        )
    }

    /// Create a memory barrier (all resources)
    public static func memoryBarrier(label: String? = nil) -> BarrierDescriptor {
        BarrierDescriptor(
            type: .memory,
            beforeStages: .dispatch,
            afterStages: .dispatch,
            label: label
        )
    }

    /// Create a pass barrier (full synchronization)
    public static func passBarrier(label: String? = nil) -> BarrierDescriptor {
        BarrierDescriptor(
            type: .pass,
            beforeStages: .all,
            afterStages: .all,
            label: label
        )
    }
}

// MARK: - Hazard Tracking

/// Tracks resource hazards for automatic barrier insertion
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class HazardTracker: @unchecked Sendable {
    /// Access type for a resource
    public enum AccessType: Sendable {
        case read
        case write
        case readWrite
    }

    /// Record of a resource access
    private struct AccessRecord {
        let buffer: ObjectIdentifier
        let access: AccessType
        let stage: PipelineStage
        let operationIndex: Int
    }

    private var accessHistory: [AccessRecord] = []
    private var operationCount: Int = 0
    private let lock = NSLock()

    public init() {}

    /// Record a resource access
    public func recordAccess(
        buffer: any MTLBuffer,
        access: AccessType,
        stage: PipelineStage = .dispatch
    ) {
        lock.lock()
        defer { lock.unlock() }

        accessHistory.append(AccessRecord(
            buffer: ObjectIdentifier(buffer),
            access: access,
            stage: stage,
            operationIndex: operationCount
        ))
    }

    /// Mark the end of an operation
    public func nextOperation() {
        lock.lock()
        defer { lock.unlock() }
        operationCount += 1
    }

    /// Check if a barrier is needed before accessing a resource
    public func needsBarrier(buffer: any MTLBuffer, access: AccessType) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let bufferId = ObjectIdentifier(buffer)

        // Find the last access to this buffer
        guard let lastAccess = accessHistory.last(where: { $0.buffer == bufferId }) else {
            return false
        }

        // Barrier needed if:
        // - Last access was a write and we're reading/writing
        // - Last access was read and we're writing
        switch (lastAccess.access, access) {
        case (.write, _), (.readWrite, _):
            return lastAccess.operationIndex < operationCount
        case (.read, .write), (.read, .readWrite):
            return lastAccess.operationIndex < operationCount
        case (.read, .read):
            return false
        }
    }

    /// Get the buffers that need barriers before the next operation
    public func getBarrierResources(for buffers: [any MTLBuffer], access: AccessType) -> [any MTLBuffer] {
        buffers.filter { needsBarrier(buffer: $0, access: access) }
    }

    /// Clear all tracked history
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        accessHistory.removeAll()
        operationCount = 0
    }
}

// MARK: - Barrier Helper

/// Helper for inserting barriers in Metal 4 encoders
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public struct BarrierHelper: Sendable {

    // MARK: - Common Barrier Patterns

    /// Insert barrier after distance computation before selection
    public static func barrierAfterDistance(
        encoder: any MTLComputeCommandEncoder,
        distanceBuffer: any MTLBuffer
    ) {
        // In Metal 4, this would be:
        // encoder.barrier(resources: [distanceBuffer], beforeStages: .dispatch, afterStages: .dispatch)

        // For Metal 3 compatibility, we use memoryBarrier
        encoder.memoryBarrier(scope: .buffers)
    }

    /// Insert barrier for in-place updates (e.g., streaming top-k)
    public static func barrierForInPlace(
        encoder: any MTLComputeCommandEncoder,
        buffer: any MTLBuffer
    ) {
        // Full memory barrier for in-place updates
        encoder.memoryBarrier(scope: .buffers)
    }

    /// Insert barrier between quantization and distance computation
    public static func barrierAfterQuantization(
        encoder: any MTLComputeCommandEncoder,
        quantizedBuffer: any MTLBuffer
    ) {
        encoder.memoryBarrier(scope: .buffers)
    }

    /// Insert barrier between matrix operations
    public static func barrierAfterMatrixOp(
        encoder: any MTLComputeCommandEncoder,
        resultBuffer: any MTLBuffer
    ) {
        encoder.memoryBarrier(scope: .buffers)
    }

    // MARK: - Generic Barrier Insertion

    /// Insert a barrier described by a descriptor
    public static func insertBarrier(
        encoder: any MTLComputeCommandEncoder,
        descriptor: BarrierDescriptor
    ) {
        switch descriptor.type {
        case .resource:
            // Resource-specific barrier
            // In Metal 4: encoder.barrier(resources: buffers, beforeStages:, afterStages:)
            encoder.memoryBarrier(scope: .buffers)

        case .memory:
            // Full memory barrier
            encoder.memoryBarrier(scope: .buffers)

        case .pass:
            // Pass barrier - typically handled by encoder.endEncoding() in Metal 3
            // In Metal 4 unified encoder, this would be encoder.passBarrier()
            encoder.memoryBarrier(scope: .buffers)

        case .fence:
            // Fence synchronization
            encoder.memoryBarrier(scope: .buffers)
        }
    }

    // MARK: - Pipeline Pattern Helpers

    /// Barrier configuration for distance + selection pipeline
    public static func distanceSelectionPipeline() -> [BarrierDescriptor] {
        [
            .dispatchBarrier(resources: [], label: "Distance → Selection")
        ]
    }

    /// Barrier configuration for PQ search pipeline (3 stages)
    public static func pqSearchPipeline() -> [BarrierDescriptor] {
        [
            .dispatchBarrier(resources: [], label: "PQ Lookup → Distance"),
            .dispatchBarrier(resources: [], label: "Distance → Selection")
        ]
    }

    /// Barrier configuration for streaming top-k (per chunk)
    public static func streamingTopKPipeline() -> [BarrierDescriptor] {
        [
            .dispatchBarrier(resources: [], label: "Chunk Distance → Merge"),
            .dispatchBarrier(resources: [], label: "Merge → Next Chunk")
        ]
    }

    /// Barrier configuration for quantize + distance
    public static func quantizeDistancePipeline() -> [BarrierDescriptor] {
        [
            .dispatchBarrier(resources: [], label: "Dequantize → Distance")
        ]
    }
}

// MARK: - Encoder Extension for Barriers

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public extension MTLComputeCommandEncoder {
    /// Insert a dispatch barrier for specific resources
    ///
    /// In Metal 4, this uses the unified encoder's barrier() method.
    /// In Metal 3, this falls back to memoryBarrier.
    func dispatchBarrier(resources: [any MTLBuffer]) {
        // Metal 4 would use: barrier(resources: resources, beforeStages: .dispatch, afterStages: .dispatch)
        memoryBarrier(scope: .buffers)
    }

    /// Insert barriers for a fused pipeline
    func insertPipelineBarriers(_ descriptors: [BarrierDescriptor]) {
        for descriptor in descriptors {
            BarrierHelper.insertBarrier(encoder: self, descriptor: descriptor)
        }
    }
}

// MARK: - Fused Pipeline Builder

/// Builder for constructing fused pipelines with automatic barrier insertion
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public final class FusedPipelineBuilder: @unchecked Sendable {
    /// A stage in the pipeline
    public struct Stage {
        let name: String
        let pipeline: any MTLComputePipelineState
        let argumentTable: (any ArgumentTable)?
        let threadgroups: MTLSize
        let threadsPerThreadgroup: MTLSize
        let inputBuffers: [any MTLBuffer]
        let outputBuffers: [any MTLBuffer]
    }

    private var stages: [Stage] = []
    private let hazardTracker = HazardTracker()

    public init() {}

    /// Add a stage to the pipeline
    public func addStage(
        name: String,
        pipeline: any MTLComputePipelineState,
        argumentTable: (any ArgumentTable)? = nil,
        threadgroups: MTLSize,
        threadsPerThreadgroup: MTLSize,
        inputs: [any MTLBuffer],
        outputs: [any MTLBuffer]
    ) {
        stages.append(Stage(
            name: name,
            pipeline: pipeline,
            argumentTable: argumentTable,
            threadgroups: threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup,
            inputBuffers: inputs,
            outputBuffers: outputs
        ))

        // Track hazards
        for buffer in inputs {
            hazardTracker.recordAccess(buffer: buffer, access: .read)
        }
        for buffer in outputs {
            hazardTracker.recordAccess(buffer: buffer, access: .write)
        }
        hazardTracker.nextOperation()
    }

    /// Execute all stages on an encoder with automatic barrier insertion
    public func execute(on encoder: any MTLComputeCommandEncoder) {
        for (index, stage) in stages.enumerated() {
            // Check if barriers needed before this stage
            let needsBarrier = index > 0 && stages[index - 1].outputBuffers.contains(where: { output in
                stage.inputBuffers.contains(where: { ObjectIdentifier($0) == ObjectIdentifier(output) })
            })

            if needsBarrier {
                encoder.memoryBarrier(scope: .buffers)
            }

            // Set pipeline
            encoder.setComputePipelineState(stage.pipeline)

            // Apply argument table or individual buffers
            if let argTable = stage.argumentTable as? Metal4ArgumentTable {
                argTable.apply(to: encoder)
            }

            // Dispatch
            encoder.dispatchThreadgroups(stage.threadgroups, threadsPerThreadgroup: stage.threadsPerThreadgroup)
        }
    }

    /// Reset the builder for reuse
    public func reset() {
        stages.removeAll()
        hazardTracker.reset()
    }

    /// Get the number of stages
    public var stageCount: Int {
        stages.count
    }
}
