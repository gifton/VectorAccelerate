//
//  MetalComputeProvider.swift
//  VectorAccelerate
//
//  The single canonical GPU compute façade. Composes Metal4Context + Metal4ComputeEngine +
//  GPUDecisionEngine + the no-copy kernel distance providers + AccelerateFallback, routing
//  GPU-vs-CPU per call. Conforms to VectorCore's ComputeProvider as an honest capability shim.
//
//  See docs/superpowers/specs/2026-06-07-metal-compute-provider-design.md
//

import Foundation
import VectorCore
@preconcurrency import Metal

/// GPU compute façade for VectorAccelerate.
///
/// Use the dedicated methods (`batchDistance`, `findNearest`, `distanceMatrix`, `distance`) for
/// GPU-accelerated work. The `ComputeProvider` conformance is a capability/identity shim: it reports
/// `device == .gpu` with real device info, but its `execute`/`parallel*` methods run their (CPU)
/// closures via Swift concurrency — installing this as `Operations.computeProvider` does **not** move
/// VectorCore's Operations onto the GPU (those closures are CPU kernels; it will not be slower than
/// `CPUComputeProvider`, but it will not accelerate). For transparent GPU dispatch through VectorCore's
/// own API, see VectorCore request R4 in `docs/VECTORCORE_INTEGRATION_REQUESTS.md`.
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public actor MetalComputeProvider: ComputeProvider {

    /// Routing/fallback configuration.
    public struct Configuration: Sendable {
        /// When false, every call takes the CPU path (the decision engine is not consulted).
        public var preferGPU: Bool
        /// When true, a GPU kernel error falls back to the CPU path instead of throwing.
        public var fallbackToCPU: Bool
        public init(preferGPU: Bool = true, fallbackToCPU: Bool = true) {
            self.preferGPU = preferGPU
            self.fallbackToCPU = fallbackToCPU
        }
    }

    // Composed collaborators.
    private let context: Metal4Context
    private let engine: Metal4ComputeEngine
    private let decisionEngine: GPUDecisionEngine
    private let l2Provider: L2KernelDistanceProvider
    private let cosineProvider: CosineKernelDistanceProvider
    private let configuration: Configuration

    // Nonisolated ComputeProvider shim state, captured at init (no actor hop on access).
    private nonisolated let _device: ComputeDevice
    private nonisolated let _maxConcurrency: Int
    private nonisolated let _deviceInfo: ComputeDeviceInfo

    // MARK: - ComputeProvider conformance (scheduling-only shim)

    public nonisolated var device: ComputeDevice { _device }
    public nonisolated var maxConcurrency: Int { _maxConcurrency }
    public nonisolated var deviceInfo: ComputeDeviceInfo { _deviceInfo }

    public func execute<T: Sendable>(_ work: @Sendable @escaping () async throws -> T) async throws -> T {
        try await work()
    }
    // parallelExecute / parallelForEach / parallelReduce: inherited from the ComputeProvider
    // protocol's default TaskGroup implementations (correct, never slower than CPUComputeProvider).

    // MARK: - Initialization

    /// Primary initializer over an existing Metal4Context.
    public init(
        context: Metal4Context,
        configuration: Configuration = .init(),
        decisionEngine: GPUDecisionEngine? = nil
    ) async throws {
        self.context = context
        self.configuration = configuration
        let resolvedDecision: GPUDecisionEngine
        if let decisionEngine {
            resolvedDecision = decisionEngine
        } else {
            resolvedDecision = await GPUDecisionEngine(context: context)
        }
        self.decisionEngine = resolvedDecision
        self.engine = try await Metal4ComputeEngine(context: context, decisionEngine: resolvedDecision)
        self.l2Provider = try await L2KernelDistanceProvider(context: context)
        self.cosineProvider = try await CosineKernelDistanceProvider(context: context)

        // `context.device` is nonisolated; `rawDevice` is a nonisolated `any MTLDevice`.
        let raw = context.device.rawDevice
        self._device = .gpu(index: 0)
        self._maxConcurrency = raw.maxThreadsPerThreadgroup.width
        self._deviceInfo = ComputeDeviceInfo(
            name: raw.name,
            availableMemory: Int(raw.recommendedMaxWorkingSetSize),
            maxThreads: raw.maxThreadsPerThreadgroup.width,
            preferredChunkSize: 1024
        )
    }

    /// Convenience initializer that builds a default Metal4Context.
    public init(configuration: Configuration = .init()) async throws {
        guard ComputeDevice.gpu().isAvailable else { throw VectorError.metalNotAvailable() }
        let ctx = try await Metal4Context()
        try await self.init(context: ctx, configuration: configuration)
    }

    /// Build a provider with all defaults.
    public static func makeDefault() async throws -> MetalComputeProvider {
        try await MetalComputeProvider()
    }
}
