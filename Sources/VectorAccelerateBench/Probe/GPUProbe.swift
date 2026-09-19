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
        dst[gid] = v + sink * p.zero;   // p.zero is 0.0f from the CPU; the compiler cannot fold it
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
        guard let function = library.makeFunction(name: "copy_f32x4") else { throw BenchError.usage("probe kernel missing") }
        pipeline = try device.makeComputePipelineState(function: function)
        guard let queue = device.makeCommandQueue(),
              let src = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let dst = device.makeBuffer(length: Self.bytes, options: .storageModeShared),
              let params = device.makeBuffer(length: 16, options: .storageModeShared) else { throw BenchError.metalUnavailable }
        self.queue = queue; self.src = src; self.dst = dst; self.params = params
        src.label = "probe.src"; dst.label = "probe.dst"; params.label = "probe.params"
        let p = src.contents().bindMemory(to: Float32.self, capacity: Self.floatCount)
        for i in 0..<Self.floatCount { p[i] = Float32(i & 0xFFFF) * 0.5 + 1.0 }
        memset(dst.contents(), 0, Self.bytes)
    }

    /// One command buffer, one dispatch, synchronous wait. Returns the buffer's GPU start/end in host seconds.
    func dispatch(loops: Int) throws -> (startSeconds: Double, endSeconds: Double) {
        struct ProbeParams { var n4: UInt32; var loops: UInt32; var zero: Float }
        var pp = ProbeParams(n4: UInt32(Self.float4Count), loops: UInt32(max(loops, 1)), zero: 0)
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

    func copyGBps(warmup: Int = 5, measured: Int = 10) throws -> Double {
        for _ in 0..<warmup { _ = try dispatch(loops: 1) }
        var secs: [Double] = []
        for _ in 0..<measured { let t = try dispatch(loops: 1); secs.append(t.endSeconds - t.startSeconds) }
        let median = Stats.percentile(sorted: secs.sorted(), p: 0.5)
        return median > 0 ? Double(Self.bytes) / median / 1e9 : 0
    }

    func outputMatchesInput() -> Bool { memcmp(src.contents(), dst.contents(), Self.bytes) == 0 }
}
