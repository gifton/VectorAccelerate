import Foundation
import Metal
import VectorTestKit

enum BenchVersion { static let package = "unreleased" }   // set when a release is cut; until then gitSHA identifies the build

enum RunCommand {
    /// What `run` reads: `--filter`, `--exclude` and `--out` in `run`, the rest in `parseOptions`. `--allow-dirty` is accepted
    /// because spec §5.5 lists it for `run`; nothing here reads it (`run` never refuses a dirty tree, it only records it).
    static let allowedOptions: Set<String> = ["--filter", "--exclude", "--mode", "--samples", "--iterations", "--seed", "--out", "--label",
                                              "--probe-slowdown"]
    static let allowedFlags: Set<String> = ["--require-gpu", "--validation-layers", "--allow-dirty"]

    static func run(_ args: Arguments) async throws -> Int32 {
        try args.validate(options: allowedOptions, flags: allowedFlags, operands: 0)
        let options = try parseOptions(args)
        let include = args.value("--filter").map { [$0] } ?? [], exclude = args.value("--exclude").map { [$0] } ?? []
        let cases = try Registry.select(include: include, exclude: exclude)
        guard !cases.isEmpty else {
            throw BenchError.usage("no case selected: --filter \(include.first.map { "'\($0)'" } ?? "(none)") --exclude \(exclude.first.map { "'\($0)'" } ?? "(none)") "
                                   + "matched 0 of \(Registry.all.count) registered case(s); see `list`")
        }
        let root = Workspace.root
        let out = args.value("--out").map { URL(fileURLWithPath: $0) }
        // Spec §7: an existing path is an error and is never overwritten. Checked before the run so the measurements are not lost;
        // `.withoutOverwriting` in `write` remains the backstop for a file that appears during the run.
        if let out, FileManager.default.fileExists(atPath: out.path) { throw BenchError.outputExists(out.path) }

        let device = MTLCreateSystemDefaultDevice()
        let host = HostFingerprint.gather(packageRoot: root)
        let gpu = MetalFingerprint.gather(device: device)
        let context = BenchContext(device: device, host: host, gpu: gpu, options: options)

        // Spec §4.10: a preflight GPU probe that cannot be set up or read makes the probe "unavailable" (quiet: false with the
        // reason recorded); it never aborts the run, so the cases still run and the artifact is still written.
        var gpuProbeFailures: [String] = []
        func attempt<T>(_ step: String, _ body: () throws -> T) -> T? {
            do { return try body() } catch {
                let message = "preflight GPU probe (\(step)): \(error)"
                gpuProbeFailures.append(message)
                Diagnostics.line("warning: \(message)")
                return nil
            }
        }

        let thermalStart = HostFingerprint.thermalStateName()
        let cpuStart = PreflightProbe.cpuCopyGBps()
        let probe = device.flatMap { d in attempt("setup") { try GPUProbe(device: d) } }
        let gpuStart = probe.flatMap { p in attempt("start reading") { try p.copyGBps() } }
        print("preflight: cpu \(String(format: "%.1f", cpuStart)) GB/s" + (gpuStart.map { String(format: ", gpu %.1f GB/s", $0) } ?? ", gpu n/a"))

        var records: [CaseRecord] = []
        for c in cases {
            print("running \(c.id.rawValue) (\(options.samples)×\(options.iterations)) ...")
            let r = await Runner.run(c, context: context)
            print("  \(r.status.rawValue)" + (r.timing.wall.map { String(format: "  wall median %.0f ns", $0.summary.median) } ?? "") + (r.failureReason.map { "  \($0)" } ?? ""))
            records.append(r)
        }

        let cpuEnd = PreflightProbe.cpuCopyGBps()
        let gpuEnd = gpuStart.flatMap { _ in probe.flatMap { p in attempt("end reading") { try p.copyGBps() } } }
        let thermalEnd = HostFingerprint.thermalStateName()
        var preflight = PreflightProbe.evaluate(
            cpu: StartEnd(start: cpuStart, end: cpuEnd),
            gpu: gpuStart.flatMap { s in gpuEnd.map { StartEnd(start: s, end: $0) } },
            thermalStart: thermalStart, thermalEnd: thermalEnd, gpuDeviceExists: device != nil)
        preflight.reasons += gpuProbeFailures   // `evaluate` has already made this not quiet ("GPU probe unavailable ..."); these say why

        let filters = (include.isEmpty && exclude.isEmpty) ? nil : Filters(include: include.isEmpty ? nil : include, exclude: exclude.isEmpty ? nil : exclude)
        let metadata = makeMetadata(host: host, gpu: gpu, options: options, thermal: ThermalInfo(start: thermalStart, end: thermalEnd),
                                    preflight: preflight, filters: filters)
        let artifact = BenchArtifact(package: "VectorAccelerate", metadata: metadata, cases: records)
        try artifact.validate()
        let url = try write(artifact, to: out)
        print("artifact: \(url.path)")
        print("key: \(metadata.environmentKey)  quiet: \(preflight.quiet)" + (preflight.reasons.isEmpty ? "" : "  reasons: \(preflight.reasons)"))

        if records.contains(where: { $0.status == .failed }) { return 1 }
        if options.requireGPU, records.contains(where: { $0.status == .unavailable }) { return 1 }
        return 0
    }

    /// The five typed options go through `Arguments.parsed`: absent takes the default, present-but-rejected throws `BenchError.usage`
    /// naming the option and the value.
    private static func parseOptions(_ args: Arguments) throws -> RunOptions {
        let positive: (String) -> Int? = { Int($0).flatMap { $0 >= 1 ? $0 : nil } }
        let mode = try args.parsed("--mode", expecting: "smoke, quick or full", { BenchMode(rawValue: $0) }) ?? .quick
        return try RunOptions(
            mode: mode,
            samples: args.parsed("--samples", expecting: "a positive integer", positive) ?? mode.samples,
            iterations: args.parsed("--iterations", expecting: "a positive integer", positive) ?? mode.iterations,
            seed: args.parsed("--seed", expecting: "an unsigned 64-bit integer", { UInt64($0) }) ?? RunOptions.defaultSeed,
            validationLayers: args.has("--validation-layers"), requireGPU: args.has("--require-gpu"),
            probeSlowdown: args.parsed("--probe-slowdown", expecting: "a non-negative integer", { Int($0).flatMap { $0 >= 0 ? $0 : nil } }) ?? 0,
            label: args.value("--label"))
    }

    private static func makeMetadata(host: HostFingerprint, gpu: GPUInfo?, options: RunOptions, thermal: ThermalInfo,
                                     preflight: PreflightRecord, filters: Filters?) -> Metadata {
        Metadata(
            packageVersion: BenchVersion.package, gitSHA: host.gitSHA, gitDirty: host.gitDirty, date: ISO8601DateFormatter().string(from: Date()),
            os: host.osVersion, osBuild: host.osBuild, arch: host.arch, cpuCores: host.cpuCores, performanceCores: host.performanceCores,
            efficiencyCores: host.efficiencyCores, memoryBytes: host.memoryBytes, deviceModel: host.chip, modelIdentifier: host.modelIdentifier,
            hostname: host.hostname, swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration,
            deviceTag: HostFingerprint.deviceTag(arch: host.arch, osVersion: host.osVersion, hostname: host.hostname),
            environmentKey: EnvironmentKey.make(chip: host.chip, gpuCores: gpu?.coreCount, osBuild: host.osBuild, metalToolchain: gpu?.metalToolchain,
                                                swiftVersion: host.swiftVersion, buildConfiguration: host.buildConfiguration,
                                                validationLayers: options.validationLayers),
            gpu: gpu, power: host.power, thermal: thermal, preflight: preflight,
            validationLayers: options.validationLayers, runSeed: options.seed, runLabel: options.label, mode: options.mode.rawValue,
            filters: filters, flags: [:])
    }

    private static func write(_ artifact: BenchArtifact, to out: URL?) throws -> URL {
        guard let out else { return try Workspace.store.writeRun(artifact) }
        try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
        try RunStore.encode(artifact).write(to: out, options: [.withoutOverwriting])
        return out
    }
}
