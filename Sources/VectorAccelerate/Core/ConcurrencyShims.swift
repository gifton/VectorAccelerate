//
//  ConcurrencyShims.swift
//  VectorAccelerate
//
//  Centralized shims for Swift 6 sendability with Metal/MPS/Accelerate.
//  - Applies @preconcurrency imports to tame strict checks for SDK modules
//  - Provides an UnsafeSendable wrapper for rare cross-actor handle passing
//  - Adds convenience properties to mark Metal handles as unchecked sendable
//

@preconcurrency import Metal
#if canImport(MetalKit)
@preconcurrency import MetalKit
#endif
#if canImport(MetalPerformanceShaders)
@preconcurrency import MetalPerformanceShaders
#endif
#if canImport(Accelerate)
@preconcurrency import Accelerate
#endif

// Generic wrapper to explicitly opt-in to cross-actor passing of non-Sendable values.
// Use sparingly and ensure no concurrent use from multiple tasks.
@frozen
public struct UnsafeSendable<T>: @unchecked Sendable {
    public let value: T
    public init(_ value: T) { self.value = value }
}

// Convenience properties for common Metal handles to opt-in when absolutely necessary.
// Prefer keeping Metal objects actor-confined instead of using these.
public extension MTLDevice {
    var uncheckedSendable: UnsafeSendable<any MTLDevice> { .init(self) }
}

public extension MTLCommandQueue {
    var uncheckedSendable: UnsafeSendable<any MTLCommandQueue> { .init(self) }
}

public extension MTLCommandBuffer {
    var uncheckedSendable: UnsafeSendable<any MTLCommandBuffer> { .init(self) }
}

public extension MTLBuffer {
    var uncheckedSendable: UnsafeSendable<any MTLBuffer> { .init(self) }
}

public extension MTLTexture {
    var uncheckedSendable: UnsafeSendable<any MTLTexture> { .init(self) }
}

public extension MTLComputePipelineState {
    var uncheckedSendable: UnsafeSendable<any MTLComputePipelineState> { .init(self) }
}

