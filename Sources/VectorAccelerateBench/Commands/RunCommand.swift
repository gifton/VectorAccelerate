import Foundation
import Metal
import VectorTestKit

enum BenchVersion { static let package = "unreleased" }   // set when a release is cut; until then gitSHA identifies the build

enum RunCommand {
    static func run(_ args: Arguments) async throws -> Int32 {
        // The five typed options go through `Arguments.parsed`: absent takes the default, present-but-unparseable throws `BenchError.usage`.
        let mode = try args.parsed("--mode", expecting: "smoke, quick or full", { BenchMode(rawValue: $0) }) ?? .quick
        let options = try RunOptions(
            mode: mode,
            samples: args.parsed("--samples", expecting: "an integer", { Int($0) }) ?? mode.samples,
            iterations: args.parsed("--iterations", expecting: "an integer", { Int($0) }) ?? mode.iterations,
            seed: args.parsed("--seed", expecting: "an unsigned 64-bit integer", { UInt64($0) }) ?? 42,
            validationLayers: args.has("--validation-layers"), requireGPU: args.has("--require-gpu"),
            probeSlowdown: args.parsed("--probe-slowdown", expecting: "a non-negative integer", { Int($0).flatMap { $0 >= 0 ? $0 : nil } }) ?? 0,
            label: args.value("--label"))
        guard options.samples >= 1, options.iterations >= 1 else { throw BenchError.usage("--samples and --iterations must be >= 1") }
        let include = args.value("--filter").map { [$0] } ?? [], exclude = args.value("--exclude").map { [$0] } ?? []
        let cases = try Registry.select(include: include, exclude: exclude)
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let device = MTLCreateSystemDefaultDevice()
        let host = HostFingerprint.gather(packageRoot: root)
        let gpu = MetalFingerprint.gather(device: device)
        let context = BenchContext(device: device, host: host, gpu: gpu, options: options)

        let thermalStart = HostFingerprint.thermalStateName()
        let cpuStart = PreflightProbe.cpuCopyGBps()
        let probe = try device.map { try GPUProbe(device: $0) }
        let gpuStart = try probe?.copyGBps()
        print("preflight: cpu \(String(format: "%.1f", cpuStart)) GB/s" + (gpuStart.map { String(format: ", gpu %.1f GB/s", $0) } ?? ", gpu n/a"))

        var records: [CaseRecord] = []
        for c in cases {
            print("running \(c.id.rawValue) (\(options.samples)×\(options.iterations)) ...")
            let r = await Runner.run(c, context: context)
            print("  \(r.status.rawValue)" + (r.timing.wall.map { String(format: "  wall median %.0f ns", $0.summary.median) } ?? "") + (r.failureReason.map { "  \($0)" } ?? ""))
            records.append(r)
        }

        let cpuEnd = PreflightProbe.cpuCopyGBps()
        let gpuEnd = try probe?.copyGBps()
        let thermalEnd = HostFingerprint.thermalStateName()
        let preflight = PreflightProbe.evaluate(
            cpu: StartEnd(start: cpuStart, end: cpuEnd),
            gpu: gpuStart.flatMap { s in gpuEnd.map { StartEnd(start: s, end: $0) } },
            thermalStart: thermalStart, thermalEnd: thermalEnd, gpuDeviceExists: device != nil)

        let iso = ISO8601DateFormatter()
        let metadata = Metadata(
            packageVersion: BenchVersion.package, gitSHA: host.gitSHA, gitDirty: host.gitDirty, date: iso.string(from: Date()),
            os: host.osVersion, osBuild: host.osBuild, arch: host.arch, cpuCores: host.cpuCores, performanceCores: host.performanceCores,
            efficiencyCores: host.efficiencyCores, memoryBytes: host.memoryBytes, deviceModel: host.chip, modelIdentifier: host.modelIdentifier,
            hostname: host.hostname, swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration,
            deviceTag: HostFingerprint.deviceTag(arch: host.arch, osVersion: host.osVersion, hostname: host.hostname),
            environmentKey: EnvironmentKey.make(chip: host.chip, gpuCores: gpu?.coreCount, osBuild: host.osBuild, metalToolchain: gpu?.metalToolchain,
                                                swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration, validationLayers: options.validationLayers),
            gpu: gpu, power: host.power, thermal: ThermalInfo(start: thermalStart, end: thermalEnd), preflight: preflight,
            validationLayers: options.validationLayers, runSeed: options.seed, runLabel: options.label, mode: mode.rawValue,
            filters: (include.isEmpty && exclude.isEmpty) ? nil : Filters(include: include.isEmpty ? nil : include, exclude: exclude.isEmpty ? nil : exclude),
            flags: [:])
        let artifact = BenchArtifact(package: "VectorAccelerate", metadata: metadata, cases: records)
        try artifact.validate()

        let url: URL
        if let out = args.value("--out") {
            url = URL(fileURLWithPath: out)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try RunStore.encode(artifact).write(to: url, options: [.withoutOverwriting])
        } else {
            url = try RunStore(root: root.appendingPathComponent(".bench")).writeRun(artifact)
        }
        print("artifact: \(url.path)")
        print("key: \(metadata.environmentKey)  quiet: \(preflight.quiet)" + (preflight.reasons.isEmpty ? "" : "  reasons: \(preflight.reasons)"))

        if records.contains(where: { $0.status == .failed }) { return 1 }
        if options.requireGPU, records.contains(where: { $0.status == .unavailable }) { return 1 }
        return 0
    }
}
