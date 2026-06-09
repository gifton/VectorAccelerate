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
/// Conforms to VectorCore's **`BatchKernelProvider`** — the R4 dispatch hook shipped in VectorCore
/// 0.3.0. Installing this as `Operations.computeProvider` makes VectorCore's `Operations.findNearest`
/// / `findNearestBatch` dispatch transparently to the GPU: euclidean/cosine run on the fused
/// distance+top-K kernel (GPU vote), and every other metric falls back to that metric's own
/// `batchDistance` so results never diverge from the CPU path.
///
/// The inherited `ComputeProvider` scheduling members (`execute` / `parallel*`) still just *schedule*
/// their closures via Swift concurrency — they do not GPU-accelerate arbitrary CPU closures. GPU work
/// is reached through the `BatchKernelProvider` kernels and the dedicated methods (`batchDistance`,
/// `findNearest`, `distanceMatrix`, `distance`).
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public actor MetalComputeProvider: BatchKernelProvider {

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
    let context: Metal4Context          // internal: used by the SoA scoring extension (+SoA.swift)
    private let engine: Metal4ComputeEngine
    private let decisionEngine: GPUDecisionEngine
    private let l2Provider: L2KernelDistanceProvider
    private let cosineProvider: CosineKernelDistanceProvider
    let soaKernel: SoADistanceKernel    // internal: lane-major zero-copy SoA scoring (built once)
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
        self.soaKernel = try await SoADistanceKernel(context: context)

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

    // MARK: - BatchKernelProvider conformance (R4: transparent VectorCore dispatch)

    /// GPU-accelerate only the built-in metrics whose VA kernel semantics provably match VectorCore's
    /// metric (euclidean → L2 distance; cosine → 1 − similarity). Any other metric returns `nil` and is
    /// computed via that metric's own authoritative `batchDistance`, so results never diverge from the
    /// CPU path the caller would otherwise get (VectorCore maps `dotProduct` to −dot, which VA's raw-dot
    /// kernel does not — so it must not be silently GPU-routed).
    private static func gpuMetric(for metric: any DistanceMetric) -> SupportedDistanceMetric? {
        if metric is EuclideanDistance { return .euclidean }
        if metric is CosineDistance { return .cosine }
        return nil
    }

    /// `BatchKernelProvider`: distance from `query` to each candidate, in candidate order.
    public func batchDistance<V: VectorProtocol>(
        query: V, candidates: [V], metric: any DistanceMetric
    ) async throws -> [Float] where V.Scalar == Float {
        if let supported = Self.gpuMetric(for: metric) {
            return try await batchDistance(query: query, candidates: candidates, metric: supported)
        }
        // Non-accelerated metric: defer to its own authoritative batch path. Constrain the
        // existential to `DistanceMetric<Float>` so `batchDistance` returns `[Float]` — the open
        // `any DistanceMetric` erases `Scalar` to `any BinaryFloatingPoint`.
        guard let floatMetric = metric as? any DistanceMetric<Float> else {
            throw VectorError.invalidInput("MetalComputeProvider requires a Float-scalar DistanceMetric; got \(type(of: metric))")
        }
        return floatMetric.batchDistance(query: query, candidates: candidates)
    }

    /// `BatchKernelProvider`: up to `k` nearest candidates, sorted ascending by distance.
    public func findNearest<V: VectorProtocol>(
        query: V, candidates: [V], k: Int, metric: any DistanceMetric
    ) async throws -> [(index: Int, distance: Float)] where V.Scalar == Float {
        guard k > 0, !candidates.isEmpty else { return [] }
        if let supported = Self.gpuMetric(for: metric) {
            return try await findNearest(query: query, in: candidates, k: k, metric: supported)
        }
        // Non-accelerated metric: reuse the metric-aware batchDistance, then CPU top-K.
        let distances = try await batchDistance(query: query, candidates: candidates, metric: metric)
        return Self.selectTopK(distances, k: min(k, candidates.count), largerIsCloser: false)
    }

    // findNearestBatch(queries:candidates:k:metric:) uses the BatchKernelProvider default
    // (parallelExecute over findNearest); a single batched GPU dispatch is a fast-follow.

    // MARK: - Distance matrix

    /// Full `queries × candidates` distance matrix; row `i` is `batchDistance(queries[i], candidates)`.
    public func distanceMatrix<V: VectorProtocol>(
        queries: [V], candidates: [V], metric: SupportedDistanceMetric
    ) async throws -> [[Float]] where V.Scalar == Float {
        guard !queries.isEmpty, !candidates.isEmpty else { return [] }
        var matrix: [[Float]] = []
        matrix.reserveCapacity(queries.count)
        for q in queries {
            matrix.append(try await batchDistance(query: q, candidates: candidates, metric: metric))
        }
        return matrix
    }

    // MARK: - Single distance

    /// Distance between two vectors. Computed on CPU (single-pair GPU dispatch never wins); semantics
    /// match `batchDistance`.
    public func distance<V: VectorProtocol>(
        _ a: V, _ b: V, metric: SupportedDistanceMetric
    ) async throws -> Float where V.Scalar == Float {
        let av = a.toArray(); let bv = b.toArray()
        guard av.count == bv.count else {
            throw VectorError.dimensionMismatch(expected: av.count, actual: bv.count)
        }
        switch metric {
        case .euclidean:  return try AccelerateFallback.euclideanDistance(av, bv)
        case .cosine:     return 1.0 - (try AccelerateFallback.cosineSimilarity(av, bv))
        case .dotProduct: return try AccelerateFallback.dotProduct(av, bv)
        case .manhattan:  return try AccelerateFallback.manhattanDistance(av, bv)
        case .chebyshev:  return Self.chebyshev(av, bv)
        }
    }
}
