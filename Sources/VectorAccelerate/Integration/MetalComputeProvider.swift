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

    // MARK: - Batch distance

    /// Distance from `query` to each candidate. Per-metric semantics match the existing engine
    /// (euclidean → L2 distance; cosine → 1−similarity; dotProduct → raw dot; manhattan → L1;
    /// chebyshev → L∞). Euclidean/cosine use the no-copy GPU kernel path on a GPU vote; everything
    /// else (and the CPU vote / GPU error with `fallbackToCPU`) uses Accelerate.
    public func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: SupportedDistanceMetric
    ) async throws -> [Float] where V.Scalar == Float {
        guard !candidates.isEmpty else { return [] }
        let dim = query.withUnsafeBufferPointer { $0.count }
        guard candidates.allSatisfy({ $0.withUnsafeBufferPointer { $0.count } == dim }) else {
            throw VectorError.invalidInput("All candidates must match the query dimension (\(dim))")
        }

        switch metric {
        case .euclidean:
            let cpu = { AccelerateFallback.batchEuclideanDistance(query: query.toArray(), candidates: candidates.map { $0.toArray() }) }
            guard await routeToGPU(.l2Distance, candidateCount: candidates.count, k: 0, dimension: dim) else { return cpu() }
            do { return try await l2Provider.batchDistance(from: query, to: candidates, metric: .euclidean) }
            catch { if configuration.fallbackToCPU { return cpu() } else { throw error } }

        case .cosine:
            let cpu = { AccelerateFallback.batchCosineSimilarity(query: query.toArray(), candidates: candidates.map { $0.toArray() }).map { 1.0 - $0 } }
            guard await routeToGPU(.cosineSimilarity, candidateCount: candidates.count, k: 0, dimension: dim) else { return cpu() }
            do { return try await cosineProvider.batchDistance(from: query, to: candidates, metric: .cosine) }
            catch { if configuration.fallbackToCPU { return cpu() } else { throw error } }

        case .dotProduct:
            return AccelerateFallback.batchDotProduct(query: query.toArray(), candidates: candidates.map { $0.toArray() })

        case .manhattan:
            let q = query.toArray()
            return try candidates.map { try AccelerateFallback.manhattanDistance(q, $0.toArray()) }

        case .chebyshev:
            let q = query.toArray()
            return candidates.map { Self.chebyshev(q, $0.toArray()) }
        }
    }

    // MARK: - Private routing / CPU helpers

    /// Ask the decision engine whether to use the GPU for an operation (honoring `preferGPU`).
    private func routeToGPU(_ op: GPUOperation, candidateCount: Int, k: Int, dimension: Int) async -> Bool {
        guard configuration.preferGPU else { return false }
        return await decisionEngine.shouldUseGPU(
            operation: op,
            vectorCount: candidateCount,
            candidateCount: candidateCount,
            k: k,
            dimension: dimension
        )
    }

    /// Chebyshev (L∞) distance — no Accelerate primitive; computed inline.
    static func chebyshev(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0
        for i in 0..<min(a.count, b.count) { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    // MARK: - k-Nearest Neighbors (top-K)

    /// Return the `k` nearest candidates as (index, distance), ordered nearest-first. Euclidean/cosine
    /// use the fused GPU distance+top-K kernel on a GPU vote; other metrics (and CPU vote / GPU error
    /// with `fallbackToCPU`) compute `batchDistance` then select top-K on the CPU.
    public func findNearest<V: VectorProtocol>(
        query: V, in candidates: [V], k: Int, metric: SupportedDistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        guard k > 0, !candidates.isEmpty else { return [] }
        let dim = query.withUnsafeBufferPointer { $0.count }
        guard candidates.allSatisfy({ $0.withUnsafeBufferPointer { $0.count } == dim }) else {
            throw VectorError.invalidInput("All candidates must match the query dimension (\(dim))")
        }
        let effectiveK = min(k, candidates.count)

        // GPU fused path (euclidean/cosine only).
        if metric == .euclidean || metric == .cosine,
           await routeToGPU(.topKSelection, candidateCount: candidates.count, k: effectiveK, dimension: dim) {
            let m: Metal4DistanceMetric = (metric == .euclidean) ? .euclidean : .cosine
            do {
                let result = try await engine.fusedDistanceTopK(
                    query: query.toArray(), database: candidates.map { $0.toArray() },
                    k: effectiveK, metric: m
                )
                if !result.isEmpty { return result }
            } catch {
                if !configuration.fallbackToCPU { throw error }
            }
        }

        // CPU path: batch distance then select top-K.
        let distances = try await batchDistance(query: query, candidates: candidates, metric: metric)
        return Self.selectTopK(distances, k: effectiveK, largerIsCloser: metric == .dotProduct)
    }

    /// Select the k nearest (index, distance) pairs. For similarity metrics (dotProduct) larger is
    /// nearer; for distance metrics smaller is nearer.
    static func selectTopK(_ distances: [Float], k: Int, largerIsCloser: Bool) -> [(index: Int, distance: Float)] {
        let pairs = distances.enumerated().map { (index: $0.offset, distance: $0.element) }
        let sorted = largerIsCloser ? pairs.sorted { $0.distance > $1.distance }
                                    : pairs.sorted { $0.distance < $1.distance }
        return Array(sorted.prefix(k))
    }
}
