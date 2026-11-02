// mtl4_timestamp_spike.swift
//
// THROWAWAY FEASIBILITY SPIKE — Metal 4 counter-heap timestamps vs command-buffer timing.
//
// Build:  swiftc -O -framework Metal -framework Foundation mtl4_timestamp_spike.swift -o mtl4_timestamp_spike
// Run:    ./mtl4_timestamp_spike [--warmup 20] [--iters 100]
//
// Workload: memory-bound float4 copy over 64 MiB (16M Float32) -> 128 MiB traffic per dispatch.
// One dispatch per MTL4CommandBuffer, wait for completion each time (MTLSharedEvent).
//
// Clocks compared for the same work:
//   (a) wall   : ContinuousClock around begin-encode ... shared-event wait
//   (b) cbGPU  : MTL4CommitFeedback.gpuEndTime - gpuStartTime (host seconds)
//   (c) kernel : MTL4CounterHeap timestamps written by MTL4ComputeCommandEncoder before/after dispatch
//   (d) cbHeap : MTL4CounterHeap timestamps written by MTL4CommandBuffer.writeTimestamp at CB level
//
// No CFAbsoluteTimeGetCurrent / CACurrentMediaTime anywhere.

import Foundation
import Metal
import Darwin

// MARK: - Configuration

var warmupCount: Int = 20
var measuredCount: Int = 100
var reverseOrder: Bool = false
do {
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let a = args.next() {
        switch a {
        case "--warmup": if let v = args.next(), let n = Int(v) { warmupCount = n }
        case "--iters": if let v = args.next(), let n = Int(v) { measuredCount = n }
        case "--reverse": reverseOrder = true
        default: break
        }
    }
}

let floatCount: Int = 16 * 1024 * 1024               // 16M Float32 = 64 MiB
let byteCount: Int = floatCount * MemoryLayout<Float32>.stride
let float4Count: Int = floatCount / 4                 // threads
let trafficBytes: Double = Double(byteCount) * 2.0    // read + write = 128 MiB

// MARK: - QoS

setvbuf(stdout, nil, _IONBF, 0)
pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)

// MARK: - Stats

struct Summary {
    let n: Int
    let median: Double
    let p90: Double
    let min: Double
    let max: Double
    let mean: Double
    let rsdPercent: Double
}

func summarize(_ xs: [Double]) -> Summary? {
    guard !xs.isEmpty else { return nil }
    let s = xs.sorted()
    let n = s.count
    let median: Double = (n % 2 == 1) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2])
    let p90Index = Swift.max(0, Swift.min(n - 1, Int((0.9 * Double(n)).rounded(.up)) - 1))
    let p90 = s[p90Index]
    let mean = s.reduce(0, +) / Double(n)
    let variance: Double = n > 1 ? s.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(n - 1) : 0
    let sd = variance.squareRoot()
    let rsd = mean > 0 ? sd / mean * 100.0 : .nan
    return Summary(n: n, median: median, p90: p90, min: s[0], max: s[n - 1], mean: mean, rsdPercent: rsd)
}

func fmtMs(_ seconds: Double) -> String { String(format: "%8.4f", seconds * 1000.0) }

func printRow(_ label: String, _ s: Summary?) {
    let padded = label.padding(toLength: 10, withPad: " ", startingAt: 0)
    guard let s = s else { print("  \(padded)  (no samples)"); return }
    print("  \(padded)  " + String(format: "median %@ ms  p90 %@ ms  min %@ ms  max %@ ms  rsd %6.2f%%  n=%ld",
                 fmtMs(s.median), fmtMs(s.p90), fmtMs(s.min), fmtMs(s.max), s.rsdPercent, s.n))
}

// MARK: - Device

guard let device = MTLCreateSystemDefaultDevice() else {
    print("FATAL: no Metal device"); exit(1)
}

print("=== Device ===")
print("name: \(device.name)")
print("supportsFamily(.metal4): \(device.supportsFamily(.metal4))")
print("supportsFamily(.apple9): \(device.supportsFamily(.apple9))")
let tsFrequency: UInt64 = device.queryTimestampFrequency()
print("queryTimestampFrequency(): \(tsFrequency) ticks/s")
if tsFrequency > 0 {
    print(String(format: "  -> tick resolution: %.4f ns", 1.0e9 / Double(tsFrequency)))
}
print("sizeOfCounterHeapEntry(.timestamp): \(device.size(ofCounterHeapEntry: .timestamp)) bytes")
print("supportsCounterSampling(.atStageBoundary):    \(device.supportsCounterSampling(.atStageBoundary))")
print("supportsCounterSampling(.atDispatchBoundary): \(device.supportsCounterSampling(.atDispatchBoundary))")
print("supportsCounterSampling(.atBlitBoundary):     \(device.supportsCounterSampling(.atBlitBoundary))")
print("supportsCounterSampling(.atDrawBoundary):     \(device.supportsCounterSampling(.atDrawBoundary))")
print("")

// MARK: - Kernel (compiled from source at runtime)

let kernelSource: String = """
#include <metal_stdlib>
using namespace metal;

kernel void copy_f32x4(device const float4* src  [[buffer(0)]],
                       device float4*       dst  [[buffer(1)]],
                       constant uint&       n4   [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= n4) { return; }
    dst[gid] = src[gid];
}
"""

let compileOptions = MTLCompileOptions()
let library: any MTLLibrary
do {
    library = try device.makeLibrary(source: kernelSource, options: compileOptions)
} catch {
    print("FATAL: makeLibrary failed: \(error)"); exit(1)
}
guard let function = library.makeFunction(name: "copy_f32x4") else {
    print("FATAL: makeFunction failed"); exit(1)
}
let pso: any MTLComputePipelineState
do {
    pso = try device.makeComputePipelineState(function: function)
} catch {
    print("FATAL: makeComputePipelineState failed: \(error)"); exit(1)
}
let threadsPerGroup = MTLSize(width: Swift.min(256, pso.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
let gridThreads = MTLSize(width: float4Count, height: 1, depth: 1)
print("=== Kernel ===")
print("threadExecutionWidth=\(pso.threadExecutionWidth) maxTotalThreadsPerThreadgroup=\(pso.maxTotalThreadsPerThreadgroup)")
print("grid threads=\(float4Count) (float4)  threadsPerThreadgroup=\(threadsPerGroup.width)")
print("")

// MARK: - Buffers

guard let srcBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared),
      let dstBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared),
      let paramBuffer = device.makeBuffer(length: 16, options: .storageModeShared) else {
    print("FATAL: buffer allocation failed"); exit(1)
}
srcBuffer.label = "src"
dstBuffer.label = "dst"
paramBuffer.label = "params"

do {
    let p = srcBuffer.contents().bindMemory(to: Float32.self, capacity: floatCount)
    for i in 0..<floatCount { p[i] = Float32(i & 0xFFFF) * 0.5 + 1.0 }
    memset(dstBuffer.contents(), 0, byteCount)
    paramBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(float4Count)
}

// MARK: - Metal 4 objects

guard let queue = device.makeMTL4CommandQueue() else { print("FATAL: makeMTL4CommandQueue"); exit(1) }
guard let allocator = device.makeCommandAllocator() else { print("FATAL: makeCommandAllocator"); exit(1) }
guard let commandBuffer = device.makeCommandBuffer() else { print("FATAL: device.makeCommandBuffer (MTL4)"); exit(1) }
commandBuffer.label = "spike-cb"

let residencyDesc = MTLResidencySetDescriptor()
residencyDesc.initialCapacity = 4
let residencySet: any MTLResidencySet
do {
    residencySet = try device.makeResidencySet(descriptor: residencyDesc)
} catch {
    print("FATAL: makeResidencySet: \(error)"); exit(1)
}
residencySet.addAllocation(srcBuffer)
residencySet.addAllocation(dstBuffer)
residencySet.addAllocation(paramBuffer)
residencySet.commit()
queue.addResidencySet(residencySet)

let argTableDesc = MTL4ArgumentTableDescriptor()
argTableDesc.maxBufferBindCount = 3
let argTable: any MTL4ArgumentTable
do {
    argTable = try device.makeArgumentTable(descriptor: argTableDesc)
} catch {
    print("FATAL: makeArgumentTable: \(error)"); exit(1)
}
argTable.setAddress(srcBuffer.gpuAddress, index: 0)
argTable.setAddress(dstBuffer.gpuAddress, index: 1)
argTable.setAddress(paramBuffer.gpuAddress, index: 2)

guard let sharedEvent = device.makeSharedEvent() else { print("FATAL: makeSharedEvent"); exit(1) }
var eventValue: UInt64 = 0

// MARK: - Counter heap

enum Mode: CustomStringConvertible {
    case none              // no timestamps at all (baseline)
    case cbOnly            // MTL4CommandBuffer.writeTimestamp only
    case relaxed           // encoder .relaxed + CB-level
    case precise           // encoder .precise + CB-level

    var granularity: MTL4TimestampGranularity? {
        switch self {
        case .relaxed: return .relaxed
        case .precise: return .precise
        default: return nil
        }
    }
    var usesCBTimestamps: Bool { self != .none }
    var description: String {
        switch self {
        case .none: return "none"
        case .cbOnly: return "cbOnly"
        case .relaxed: return "relaxed"
        case .precise: return "precise"
        }
    }
}

let entriesPerIteration: Int = 4
let totalIterations: Int = warmupCount + measuredCount

func makeHeap(count: Int, label: String) -> any MTL4CounterHeap {
    let d = MTL4CounterHeapDescriptor()
    d.type = .timestamp
    d.count = count
    do {
        let h = try device.makeCounterHeap(descriptor: d)
        h.label = label
        return h
    } catch {
        print("FATAL: makeCounterHeap: \(error)"); exit(1)
    }
}

// MARK: - Feedback box

final class FeedbackBox: @unchecked Sendable {
    let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var _start: Double = 0
    private var _end: Double = 0
    private var _error: (any Error)?
    func set(start: Double, end: Double, error: (any Error)?) {
        lock.lock(); defer { lock.unlock() }
        _start = start; _end = end; _error = error
    }
    var values: (start: Double, end: Double, error: (any Error)?) {
        lock.lock(); defer { lock.unlock() }
        return (_start, _end, _error)
    }
}

// MARK: - Sample

struct Sample {
    var wall: Double = 0          // seconds
    var cbGPU: Double = 0         // seconds (feedback)
    var kernel: Double = .nan     // seconds (heap, encoder timestamps)
    var cbHeap: Double = .nan     // seconds (heap, CB-level timestamps)
    var raw: [UInt64] = []        // raw ticks [cb0, enc1, enc2, cb3]
    var monotonic: Bool = true
    var v01: Bool = false   // cb0 > enc1
    var v12: Bool = false   // enc1 > enc2
    var v23: Bool = false   // enc2 > cb3
    var v03: Bool = false   // cb0 > cb3
    var zeros: Int = 0
    var feedbackError: String? = nil
}

nonisolated(unsafe) var resolveErrors: [String] = []
func resolveTicks(_ heap: any MTL4CounterHeap, range: Range<Int>) -> [UInt64] {
    let data: Data?
    do {
        data = try heap.resolveCounterRange(range)
    } catch {
        resolveErrors.append("\(error)")
        return []
    }
    guard let data = data else { resolveErrors.append("nil Data for range \(range)"); return [] }
    let stride = MemoryLayout<UInt64>.stride
    let n = data.count / stride
    var out = [UInt64](repeating: 0, count: n)
    _ = out.withUnsafeMutableBytes { dst in data.copyBytes(to: dst) }
    return out
}

// MARK: - One iteration

func runIteration(mode: Mode, heap: (any MTL4CounterHeap)?, slot: Int) -> Sample {
    var sample = Sample()
    let base = slot * entriesPerIteration

    let t0 = ContinuousClock.now

    allocator.reset()
    commandBuffer.beginCommandBuffer(allocator: allocator)

    if mode.usesCBTimestamps, let heap = heap {
        commandBuffer.writeTimestamp(counterHeap: heap, index: base + 0)
    }

    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        print("FATAL: makeComputeCommandEncoder"); exit(1)
    }
    encoder.setComputePipelineState(pso)
    encoder.setArgumentTable(argTable)
    if let g = mode.granularity, let heap = heap {
        encoder.writeTimestamp(granularity: g, counterHeap: heap, index: base + 1)
    }
    encoder.dispatchThreads(threadsPerGrid: gridThreads, threadsPerThreadgroup: threadsPerGroup)
    if let g = mode.granularity, let heap = heap {
        encoder.writeTimestamp(granularity: g, counterHeap: heap, index: base + 2)
    }
    encoder.endEncoding()

    if mode.usesCBTimestamps, let heap = heap {
        commandBuffer.writeTimestamp(counterHeap: heap, index: base + 3)
    }

    commandBuffer.endCommandBuffer()

    let box = FeedbackBox()
    let options = MTL4CommitOptions()
    options.addFeedbackHandler { fb in
        box.set(start: fb.gpuStartTime, end: fb.gpuEndTime, error: fb.error)
        box.semaphore.signal()
    }

    queue.commit([commandBuffer], options: options)
    eventValue += 1
    queue.signalEvent(sharedEvent, value: eventValue)
    let signaled = sharedEvent.wait(untilSignaledValue: eventValue, timeoutMS: 10_000)
    let t1 = ContinuousClock.now
    if !signaled { print("FATAL: shared event wait timed out"); exit(1) }

    let elapsed = t1 - t0
    sample.wall = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18

    // Feedback (arrives asynchronously; not part of the wall-time window)
    if box.semaphore.wait(timeout: .now() + 10) == .timedOut {
        print("FATAL: commit feedback timed out"); exit(1)
    }
    let fb = box.values
    sample.cbGPU = fb.end - fb.start
    if let e = fb.error { sample.feedbackError = "\(e)" }

    // Counter heap resolve (CPU timeline, after shared-event wait)
    if mode.usesCBTimestamps, let heap = heap {
        let ticks = resolveTicks(heap, range: base..<(base + entriesPerIteration))
        sample.raw = ticks
        if ticks.count == entriesPerIteration {
            if mode.granularity != nil {
                sample.zeros = ticks.filter { $0 == 0 }.count
                sample.v01 = ticks[0] > ticks[1]
                sample.v12 = ticks[1] > ticks[2]
                sample.v23 = ticks[2] > ticks[3]
                sample.v03 = ticks[0] > ticks[3]
                sample.monotonic = !(sample.v01 || sample.v12 || sample.v23 || sample.v03)
                if ticks[2] >= ticks[1] { sample.kernel = Double(ticks[2] - ticks[1]) / Double(tsFrequency) }
            } else {
                // only indices 0 and 3 are written in cbOnly mode
                sample.zeros = (ticks[0] == 0 ? 1 : 0) + (ticks[3] == 0 ? 1 : 0)
                sample.v03 = ticks[0] > ticks[3]
                sample.monotonic = !sample.v03
            }
            if ticks[3] >= ticks[0] { sample.cbHeap = Double(ticks[3] - ticks[0]) / Double(tsFrequency) }
        }
    }
    return sample
}

// MARK: - Correctness check of the copy (once)

do {
    let s = runIteration(mode: .none, heap: nil, slot: 0)
    if let e = s.feedbackError { print("FATAL: feedback error on verification dispatch: \(e)"); exit(1) }
    let equal = memcmp(srcBuffer.contents(), dstBuffer.contents(), byteCount) == 0
    print("=== Copy verification ===")
    print("dst == src after one dispatch: \(equal)")
    if !equal { print("FATAL: kernel output mismatch"); exit(1) }
    print(String(format: "first dispatch: wall %@ ms  cbGPU(feedback) %@ ms", fmtMs(s.wall), fmtMs(s.cbGPU)))
    print("")
}

// MARK: - Run all modes

struct ModeResult {
    let mode: Mode
    let samples: [Sample]
    let heap: (any MTL4CounterHeap)?
}

var results: [ModeResult] = []
let modes: [Mode] = reverseOrder ? [.precise, .relaxed, .cbOnly, .none] : [.none, .cbOnly, .relaxed, .precise]

for mode in modes {
    let heap: (any MTL4CounterHeap)? = mode.usesCBTimestamps
        ? makeHeap(count: totalIterations * entriesPerIteration, label: "heap-\(mode)")
        : nil

    var samples: [Sample] = []
    samples.reserveCapacity(measuredCount)
    for i in 0..<totalIterations {
        let s = runIteration(mode: mode, heap: heap, slot: i)
        if let e = s.feedbackError { print("mode \(mode) iter \(i): feedback error: \(e)") }
        if i >= warmupCount { samples.append(s) }
    }
    results.append(ModeResult(mode: mode, samples: samples, heap: heap))
}

// MARK: - Report

print("=== Results (\(warmupCount) warm-up, \(measuredCount) measured, one dispatch per command buffer) ===")
print(String(format: "traffic per dispatch: %.1f MiB", trafficBytes / 1048576.0))
print("")

for r in results {
    print("--- mode: \(r.mode) ---")
    let wall = summarize(r.samples.map { $0.wall })
    let cbGPU = summarize(r.samples.map { $0.cbGPU })
    printRow("wall", wall)
    printRow("cbGPU(fb)", cbGPU)
    if r.mode.usesCBTimestamps {
        let cbHeap = summarize(r.samples.map { $0.cbHeap }.filter { !$0.isNaN })
        printRow("cbHeap", cbHeap)
        if r.mode.granularity != nil {
            let kernel = summarize(r.samples.map { $0.kernel }.filter { !$0.isNaN })
            printRow("kernel", kernel)
            if let k = kernel {
                print(String(format: "  implied bandwidth from kernel median: %.1f GB/s  (from cbGPU median: %.1f GB/s)",
                             trafficBytes / k.median / 1e9, cbGPU.map { trafficBytes / $0.median / 1e9 } ?? .nan))
            }
        } else if let c = cbHeap {
            print(String(format: "  implied bandwidth from cbHeap median: %.1f GB/s", trafficBytes / c.median / 1e9))
        }
        let zeroSamples = r.samples.filter { $0.zeros > 0 }.count
        let nonMonotonic = r.samples.filter { !$0.monotonic }.count
        let emptyResolves = r.samples.filter { $0.raw.count != entriesPerIteration }.count
        print("  samples with any zero timestamp (written slots only): \(zeroSamples)/\(r.samples.count)   non-monotonic: \(nonMonotonic)/\(r.samples.count)   resolve size mismatch: \(emptyResolves)/\(r.samples.count)")
        if r.mode.granularity != nil {
            let c01 = r.samples.filter { $0.v01 }.count, c12 = r.samples.filter { $0.v12 }.count
            let c23 = r.samples.filter { $0.v23 }.count, c03 = r.samples.filter { $0.v03 }.count
            print("  violations by pair: cb0>enc1: \(c01)   enc1>enc2: \(c12)   enc2>cb3: \(c23)   cb0>cb3: \(c03)")
            for s in r.samples.filter({ !$0.monotonic }).prefix(3) {
                let t = s.raw
                print("    example raw [cb0, enc1, enc2, cb3] = \(t)  deltas(ticks): cb0->enc1 \(Int64(t[1]) - Int64(t[0]))  enc1->enc2 \(Int64(t[2]) - Int64(t[1]))  enc2->cb3 \(Int64(t[3]) - Int64(t[2]))")
            }
        }
        if let first = r.samples.first {
            print("  first measured raw ticks [cb0, enc1, enc2, cb3]: \(first.raw)")
            if first.raw.count == 4 {
                let d = first.raw.map { Double($0) / Double(tsFrequency) }
                print(String(format: "  first measured deltas: cb0->enc1 %.4f ms, enc1->enc2 %.4f ms, enc2->cb3 %.4f ms, cb0->cb3 %.4f ms",
                             (d[1] - d[0]) * 1e3, (d[2] - d[1]) * 1e3, (d[3] - d[2]) * 1e3, (d[3] - d[0]) * 1e3))
            }
        }
    } else if let c = cbGPU {
        print(String(format: "  implied bandwidth from cbGPU median: %.1f GB/s", trafficBytes / c.median / 1e9))
    }
    print("")
}

// Full-range resolve cross-check (does resolving the whole heap at the end agree with per-iteration resolves?)
print("=== Full-heap resolve cross-check ===")
print("resolveCounterRange errors so far: \(resolveErrors.count)\(resolveErrors.isEmpty ? "" : " first: \(resolveErrors[0])")")
for r in results where r.heap != nil {
    let heap = r.heap!
    let all = resolveTicks(heap, range: 0..<heap.count)
    var mismatches = 0
    for (i, s) in r.samples.enumerated() {
        let base = (warmupCount + i) * entriesPerIteration
        guard base + entriesPerIteration <= all.count, s.raw.count == entriesPerIteration else { mismatches += 1; continue }
        if Array(all[base..<(base + entriesPerIteration)]) != s.raw { mismatches += 1 }
    }
    print("mode \(r.mode): full resolve returned \(all.count) entries (heap.count=\(heap.count)); per-iteration vs full mismatches: \(mismatches)/\(r.samples.count)")
}
print("")

// MARK: - Relaxed vs precise comparison

do {
    func med(_ m: Mode, _ key: (Sample) -> Double) -> Double? {
        results.first { $0.mode == m }.flatMap { summarize($0.samples.map(key).filter { !$0.isNaN })?.median }
    }
    print("=== relaxed vs precise (medians) ===")
    if let kr = med(.relaxed, { $0.kernel }), let kp = med(.precise, { $0.kernel }) {
        print(String(format: "kernel   relaxed %@ ms  precise %@ ms  delta %+.4f ms (%+.2f%%)", fmtMs(kr), fmtMs(kp), (kp - kr) * 1e3, (kp - kr) / kr * 100))
    }
    if let cr = med(.relaxed, { $0.cbGPU }), let cp = med(.precise, { $0.cbGPU }), let c0 = med(.none, { $0.cbGPU }), let c1 = med(.cbOnly, { $0.cbGPU }) {
        print(String(format: "cbGPU    none %@ ms  cbOnly %@ ms  relaxed %@ ms  precise %@ ms", fmtMs(c0), fmtMs(c1), fmtMs(cr), fmtMs(cp)))
    }
    if let wr = med(.relaxed, { $0.wall }), let wp = med(.precise, { $0.wall }), let w0 = med(.none, { $0.wall }), let w1 = med(.cbOnly, { $0.wall }) {
        print(String(format: "wall     none %@ ms  cbOnly %@ ms  relaxed %@ ms  precise %@ ms", fmtMs(w0), fmtMs(w1), fmtMs(wr), fmtMs(wp)))
    }
    if let hr = med(.relaxed, { $0.cbHeap }), let hp = med(.precise, { $0.cbHeap }), let h1 = med(.cbOnly, { $0.cbHeap }) {
        print(String(format: "cbHeap   cbOnly %@ ms  relaxed %@ ms  precise %@ ms", fmtMs(h1), fmtMs(hr), fmtMs(hp)))
    }
    print("")
}

// MARK: - Q6: legacy MTLCounterSampleBuffer path (macOS 26 zero-timestamp report check)

print("=== Legacy MTLCounterSampleBuffer path ===")
do {
    // sampleTimestamps: CPU vs GPU timestamp relationship (legacy timebase)
    let s0 = device.sampleTimestamps()
    usleep(100_000)
    let s1 = device.sampleTimestamps()
    let cpu0: MTLTimestamp = s0.cpu, gpu0: MTLTimestamp = s0.gpu
    let cpu1: MTLTimestamp = s1.cpu, gpu1: MTLTimestamp = s1.gpu
    print("sampleTimestamps: cpu0=\(cpu0) gpu0=\(gpu0) cpu1=\(cpu1) gpu1=\(gpu1)")
    if cpu1 > cpu0 {
        print(String(format: "  gpuDelta/cpuDelta over ~100 ms sleep = %.6f (cpu timestamps are mach_absolute_time-domain ns per MTLDevice docs)",
                     Double(gpu1 &- gpu0) / Double(cpu1 - cpu0)))
    }

    guard let legacyQueue = device.makeCommandQueue() else { print("legacy: makeCommandQueue failed"); exit(1) }
    let counterSets = device.counterSets ?? []
    print("counterSets: \(counterSets.map { $0.name })")
    guard let tsSet = counterSets.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
        print("legacy: no timestamp counter set available -> legacy path cannot be tested")
        exit(0)
    }
    let csbDesc = MTLCounterSampleBufferDescriptor()
    csbDesc.counterSet = tsSet
    csbDesc.storageMode = .shared
    csbDesc.sampleCount = 4
    csbDesc.label = "legacy-ts"
    let csb: any MTLCounterSampleBuffer
    do {
        csb = try device.makeCounterSampleBuffer(descriptor: csbDesc)
    } catch {
        print("legacy: makeCounterSampleBuffer failed: \(error)"); exit(1)
    }

    let csbPrivDesc = MTLCounterSampleBufferDescriptor()
    csbPrivDesc.counterSet = tsSet
    csbPrivDesc.storageMode = .private
    csbPrivDesc.sampleCount = 4
    csbPrivDesc.label = "legacy-ts-private"
    let csbPrivate: (any MTLCounterSampleBuffer)? = try? device.makeCounterSampleBuffer(descriptor: csbPrivDesc)
    print("private-storage counter sample buffer created: \(csbPrivate != nil)")
    guard let blitDstShared = device.makeBuffer(length: 64, options: .storageModeShared),
          let blitDstPrivate = device.makeBuffer(length: 64, options: .storageModeShared) else { print("legacy: blit dst alloc failed"); exit(1) }

    let stageSupported = device.supportsCounterSampling(.atStageBoundary)
    let dispatchSupported = device.supportsCounterSampling(.atDispatchBoundary)
    let runs = 5
    var allZeroRuns = 0
    var anyZeroRuns = 0
    var blitZeroRuns = 0
    var privZeroRuns = 0
    for run in 0..<runs {
        memset(blitDstShared.contents(), 0, 64); memset(blitDstPrivate.contents(), 0, 64)
        guard let lcb = legacyQueue.makeCommandBuffer() else { print("legacy: makeCommandBuffer failed"); exit(1) }
        let passDesc = MTLComputePassDescriptor()
        if stageSupported {
            let att = passDesc.sampleBufferAttachments[0]!
            att.sampleBuffer = csb
            att.startOfEncoderSampleIndex = 0
            att.endOfEncoderSampleIndex = 1
        }
        guard let lenc = lcb.makeComputeCommandEncoder(descriptor: passDesc) else { print("legacy: encoder failed"); exit(1) }
        lenc.setComputePipelineState(pso)
        lenc.setBuffer(srcBuffer, offset: 0, index: 0)
        lenc.setBuffer(dstBuffer, offset: 0, index: 1)
        lenc.setBuffer(paramBuffer, offset: 0, index: 2)
        if dispatchSupported { lenc.sampleCounters(sampleBuffer: csb, sampleIndex: 2, barrier: true) }
        lenc.dispatchThreads(gridThreads, threadsPerThreadgroup: threadsPerGroup)
        if dispatchSupported { lenc.sampleCounters(sampleBuffer: csb, sampleIndex: 3, barrier: true) }
        lenc.endEncoding()
        // second encoder sampling into the .private buffer (stage boundary), if it exists
        if stageSupported, let csbP = csbPrivate {
            let pd2 = MTLComputePassDescriptor()
            let a2 = pd2.sampleBufferAttachments[0]!
            a2.sampleBuffer = csbP; a2.startOfEncoderSampleIndex = 0; a2.endOfEncoderSampleIndex = 1
            guard let e2 = lcb.makeComputeCommandEncoder(descriptor: pd2) else { print("legacy: encoder2 failed"); exit(1) }
            e2.setComputePipelineState(pso)
            e2.setBuffer(srcBuffer, offset: 0, index: 0); e2.setBuffer(dstBuffer, offset: 0, index: 1); e2.setBuffer(paramBuffer, offset: 0, index: 2)
            e2.dispatchThreads(gridThreads, threadsPerThreadgroup: threadsPerGroup)
            e2.endEncoding()
        }
        // GPU-side resolve via blit encoder (the path some engines use instead of CPU resolve)
        if let blit = lcb.makeBlitCommandEncoder() {
            blit.resolveCounters(csb, range: 0..<4, destinationBuffer: blitDstShared, destinationOffset: 0)
            if let csbP = csbPrivate { blit.resolveCounters(csbP, range: 0..<4, destinationBuffer: blitDstPrivate, destinationOffset: 0) }
            blit.endEncoding()
        }
        lcb.commit()
        lcb.waitUntilCompleted()
        if let e = lcb.error { print("legacy: command buffer error: \(e)") }
        let blitShared = Array(UnsafeBufferPointer(start: blitDstShared.contents().bindMemory(to: UInt64.self, capacity: 4), count: 4))
        let blitPriv = Array(UnsafeBufferPointer(start: blitDstPrivate.contents().bindMemory(to: UInt64.self, capacity: 4), count: 4))
        if blitShared[0] == 0 && blitShared[1] == 0 { blitZeroRuns += 1 }
        if blitPriv[0] == 0 && blitPriv[1] == 0 { privZeroRuns += 1 }

        var data: Data? = nil
        do { data = try csb.resolveCounterRange(0..<4) } catch { print("legacy: resolveCounterRange threw: \(error)") }
        var ts: [UInt64] = []
        if let data = data {
            let n = data.count / MemoryLayout<UInt64>.stride
            ts = [UInt64](repeating: 0, count: n)
            _ = ts.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        }
        let zeros = ts.filter { $0 == 0 }.count
        if !ts.isEmpty && zeros == ts.count { allZeroRuns += 1 }
        if zeros > 0 { anyZeroRuns += 1 }
        let errVals = ts.filter { $0 == UInt64.max }.count
        var line = "run \(run): resolved \(ts.count) entries [stageStart, stageEnd, dispatchStart, dispatchEnd] = \(ts)  zeros=\(zeros) errorValues=\(errVals)"
        if ts.count == 4 {
            if ts[1] > ts[0] { line += String(format: "  stageDelta=%llu ticks", ts[1] - ts[0]) }
            if ts[3] > ts[2] { line += String(format: "  dispatchDelta=%llu ticks", ts[3] - ts[2]) }
        }
        line += String(format: "  legacy cb gpuEnd-gpuStart=%.4f ms", (lcb.gpuEndTime - lcb.gpuStartTime) * 1e3)
        print(line)
        print("      blit-resolved (.shared csb): \(blitShared)   blit-resolved (.private csb): \(blitPriv)")
    }
    print("legacy summary: runs=\(runs)  CPU-resolve(.shared): allZeroRuns=\(allZeroRuns) anyZeroRuns(any of 4 slots, incl. never-sampled dispatch slots)=\(anyZeroRuns)  blit-resolve(.shared) stage pair zero runs=\(blitZeroRuns)  blit-resolve(.private) stage pair zero runs=\(privZeroRuns)")
}
print("")
print("done.")
