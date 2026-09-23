import Foundation
import Metal
import VectorTestKit

/// 128 MiB float4 copy on a legacy command queue; `.commandBuffer` timing from gpuStartTime/gpuEndTime.
/// Also the preflight GPU calibration and the M0 deliberate-slowdown vehicle (spec §5.7, §5.8).
/// `@unchecked Sendable`: every stored property is `let` and `MTLDevice`/`MTLCommandQueue`/`MTLComputePipelineState` are documented thread-safe; the buffer contents are unguarded, so this is an unverified contract (no test) that one probe is driven sequentially — `Runner.run` awaits each `runOnce` before the next.
final class GPUProbe: @unchecked Sendable {
    static let bytes = 134_217_728
    static let floatCount = bytes / 4
    static let float4Count = bytes / 16

    private static let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct ProbeParams { uint n4; uint loops; float zero; };
    kernel void copy_f32x4(device const float4* src [[buffer(0)]],
                           device float4* dst [[buffer(1)]],
                           constant ProbeParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= p.n4) { return; }
        float4 v = src[gid];
        float4 sink = float4(0.0f);
        for (uint i = 1; i < p.loops; ++i) { sink += src[(gid + i * 4099u) % p.n4]; }
        dst[gid] = v + sink * p.zero;   // p.zero is 0.0f from the CPU; the compiler cannot fold it (exercised by the M0 slowdown run,
                                        // evidence in docs/superpowers/research/2026-09-17-sp0-m0-acceptance.md)
    }
    """

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let pipeline: any MTLComputePipelineState
    private let src: any MTLBuffer
    private let dst: any MTLBuffer
    private let params: any MTLBuffer

    var staticThreadgroupBytes: Int { pipeline.staticThreadgroupMemoryLength }
    var maxThreadgroupBytes: Int { device.maxThreadgroupMemoryLength }

    init(device: any MTLDevice) throws {
        self.device = device
        let library = try device.makeLibrary(source: Self.source, options: MTLCompileOptions())
        guard let function = library.makeFunction(name: "copy_f32x4") else { throw BenchError.probeSetup("kernel function copy_f32x4 not found in the compiled probe library") }
        pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue(),
              let src = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let dst = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let params = device.makeBuffer(length: 16, options: .storageModeShared) else { throw BenchError.metalUnavailable }
        self.queue = queue; self.src = src; self.dst = dst; self.params = params
        src.label = "probe.src"; dst.label = "probe.dst"; params.label = "probe.params"
        let p = src.contents().bindMemory(to: Float32.self, capacity: Self.floatCount)
        // This fill pattern is the case identity: `GPUCopyProbeCase.identity` (ProbeCases.swift) digests the string "pattern:(i&0xFFFF)*0.5+1" and labels the data "sawtooth_f32_(i&0xFFFF)*0.5+1"; change all three together.
        for i in 0..<Self.floatCount { p[i] = Float32(i & 0xFFFF) * 0.5 + 1.0 }
        resetDestination()
    }

    /// Zeroes `dst` so a later sample's `outputMatchesInput()` proves that sample's dispatch wrote it, not an earlier one.
    func resetDestination() { memset(dst.contents(), 0, Self.bytes) }

    /// One command buffer, one dispatch, synchronous wait. Returns the buffer's GPU start/end in host seconds.
    func dispatch(loops: Int) throws -> (startSeconds: Double, endSeconds: Double) {
        struct ProbeParams { var n4: UInt32; var loops: UInt32; var zero: Float }
        var pp = ProbeParams(n4: UInt32(Self.float4Count), loops: UInt32(max(loops, 1)), zero: 0)
        precondition(MemoryLayout<ProbeParams>.size == 12, "ProbeParams must match the MSL struct")
        memcpy(params.contents(), &pp, MemoryLayout<ProbeParams>.size)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw BenchError.metalUnavailable }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src, offset: 0, index: 0); enc.setBuffer(dst, offset: 0, index: 1); enc.setBuffer(params, offset: 0, index: 2)
        let tpg = MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        enc.dispatchThreads(MTLSize(width: Self.float4Count, height: 1, depth: 1), threadsPerThreadgroup: tpg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if let e = cb.error { throw e }
        return (cb.gpuStartTime, cb.gpuEndTime)
    }

    /// GB/s counting read plus write, like `PreflightProbe.cpuCopyGBps`: 2 × bytes / median seconds / 1e9.
    /// Warm-up ends only when BOTH `warmup` dispatches AND `warmupSeconds` of wall time have passed; the reading is the median of
    /// `measured` dispatches, which stays on the settled level as long as fewer than half of them (25 of 50) are slow.
    ///
    /// Why, from docs/superpowers/research/2026-09-18-preflight-probe-characterization.md (81 series, 13,500 dispatches, one
    /// M3 Max): after GPU idle — seen at process start and after gaps from 0.1 s to 5 s, in 56 of the 81 series — the first
    /// dispatches run slow (median 1.88×, range 1.08–2.40× the settled time) and reach the settled level after at most 41
    /// dispatches, or 55.5 ms of wall time in the 36 wall-stamped series. `WarmupPolicy` cannot find that point: the slow plateau is itself steady (CoV < 5 %), so the
    /// policy stopped on it at n = 3 in 3 of 45 series. A median of 10 settled dispatches strayed up to 4.66 % from its series'
    /// level, a median of 50 up to 1.85 %, against a 5 % quiet gate on the difference of two readings. Whether the ramp is bound
    /// by dispatch count or by wall time is not established (no series varied the dispatch duration): the time floor is a hedge
    /// that covers a time-bound ramp on a GPU that finishes 50 dispatches much sooner, and costs at most ≈ 0.1 s per call if
    /// the ramp is count-bound. Explicit, unverified contract on any other machine: the executable has no test target, so
    /// nothing exercises this.
    func copyGBps(warmup: Int = 50, warmupSeconds: Double = 0.1, measured: Int = 50) throws -> Double {
        precondition(warmupSeconds.isFinite && warmupSeconds >= 0, "warmupSeconds must be finite and non-negative")
        precondition(measured >= 1, "copyGBps needs at least one measured dispatch")
        let clock = ContinuousClock()
        let warmupStart = clock.now
        var warmed = 0
        while warmed < warmup || clock.now - warmupStart < .seconds(warmupSeconds) { _ = try dispatch(loops: 1); warmed += 1 }
        var secs: [Double] = []
        for _ in 0..<measured { let t = try dispatch(loops: 1); secs.append(t.endSeconds - t.startSeconds) }
        let median = Stats.percentile(sorted: secs.sorted(), p: 0.5)
        return median > 0 ? 2.0 * Double(Self.bytes) / median / 1e9 : 0
    }

    func outputMatchesInput() -> Bool { memcmp(src.contents(), dst.contents(), Self.bytes) == 0 }
}
