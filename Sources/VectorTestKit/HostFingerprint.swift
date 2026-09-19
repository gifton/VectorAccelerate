import Foundation
import Darwin

enum Sysctl {
    static func string(_ key: String) -> String? {
        var size = 0
        guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buf = [CChar](repeating: 0, count: size)
        guard sysctlbyname(key, &buf, &size, nil, 0) == 0 else { return nil }
        let s = String(decoding: buf.prefix(max(size - 1, 0)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
        return s.isEmpty ? nil : s
    }
    static func int(_ key: String) -> Int? {
        var v: Int64 = 0; var size = MemoryLayout<Int64>.size
        guard sysctlbyname(key, &v, &size, nil, 0) == 0 else { return nil }
        return Int(v)
    }
}

/// Runs a fixed executable with fixed arguments, no shell, bounded by `timeout`; nil on any failure.
public enum Subprocess {
    public static func run(_ executable: String, _ args: [String], cwd: URL? = nil, timeout: TimeInterval = 5) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: executable)
        p.arguments = args
        if let cwd { p.currentDirectoryURL = cwd }
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = FileHandle.nullDevice
        let done = DispatchSemaphore(value: 0)
        p.terminationHandler = { _ in done.signal() }
        do { try p.run() } catch { return nil }
        // Drain on a background queue so a child that never exits cannot block the timeout below.
        final class Box: @unchecked Sendable { var data = Data() }
        let box = Box(); let drained = DispatchSemaphore(value: 0)
        DispatchQueue.global(qos: .utility).async { box.data = pipe.fileHandleForReading.readDataToEndOfFile(); drained.signal() }
        if done.wait(timeout: .now() + timeout) == .timedOut { p.terminate(); return nil }
        drained.wait()
        guard p.terminationStatus == 0 else { return nil }
        return String(decoding: box.data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public enum PowerState {
    /// Parses `pmset -g batt` (power source) and `pmset -g` (lowpowermode). "unknown" if pmset is unavailable.
    public static func read() -> PowerInfo {
        var source = "unknown"
        if let batt = Subprocess.run("/usr/bin/pmset", ["-g", "batt"]) {
            if batt.contains("AC Power") { source = "ac" } else if batt.contains("Battery Power") { source = "battery" }
        }
        var low = false
        if let all = Subprocess.run("/usr/bin/pmset", ["-g"]) {
            for line in all.split(separator: "\n") where line.contains("lowpowermode") {
                low = line.trimmingCharacters(in: .whitespaces).hasSuffix("1")
            }
        }
        return PowerInfo(source: source, lowPowerMode: low)
    }
}

public enum EnvironmentKey {
    /// Lowercase alphanumerics only: "Apple M3 Max" → "applem3max".
    public static func slug(_ s: String) -> String {
        String(s.lowercased().unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(Character.init))
    }

    /// Hostname is deliberately excluded so identical hardware on different machines compares (spec §4.9).
    public static func make(chip: String, gpuCores: Int?, osBuild: String, metalToolchain: String?, swiftVersion: String,
                            buildConfiguration: String, validationLayers: Bool) -> String {
        let cores = gpuCores.map(String.init) ?? "?"
        return "\(slug(chip))-g\(cores)-\(osBuild)-mtl\(metalToolchain ?? "unknown")-swift\(swiftVersion)-\(buildConfiguration)-\(validationLayers ? "val" : "noval")"
    }
}

/// The Metal-free part of the environment fingerprint (spec §4.9). The executable adds `GPUInfo`.
public struct HostFingerprint: Sendable, Equatable {
    public var chip: String
    public var modelIdentifier: String
    public var cpuCores: Int
    public var performanceCores: Int?
    public var efficiencyCores: Int?
    public var memoryBytes: UInt64
    public var osVersion: String
    public var osBuild: String
    public var arch: String
    public var hostname: String
    public var swiftVersion: String
    public var buildConfiguration: String
    public var gitSHA: String?
    public var gitDirty: Bool
    public var power: PowerInfo

    public static var buildConfiguration: String {
        #if DEBUG
        return "debug"
        #else
        return "release"
        #endif
    }

    public static func detectSwiftVersion() -> String {
        guard let out = Subprocess.run("/usr/bin/swift", ["--version"]),
              let m = out.range(of: #"Swift version (\d+\.\d+(\.\d+)?)"#, options: .regularExpression) else { return "unknown" }
        return String(out[m]).replacingOccurrences(of: "Swift version ", with: "")
    }

    public static func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// VectorCore-compatible tag: `<arch>-<os>-<hostname>` (spaces → dashes) unless `VA_DEVICE_TAG` is set.
    public static func deviceTag(arch: String, osVersion: String, hostname: String) -> String {
        if let tag = ProcessInfo.processInfo.environment["VA_DEVICE_TAG"], !tag.isEmpty { return tag }
        return "\(arch)-\(osVersion)-\(hostname.replacingOccurrences(of: " ", with: "-"))"
    }

    public static func gather(packageRoot: URL?) -> HostFingerprint {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        #if arch(arm64)
        let arch = "arm64"
        #else
        let arch = "x86_64"
        #endif
        let sha = packageRoot.flatMap { Subprocess.run("/usr/bin/git", ["rev-parse", "HEAD"], cwd: $0) }
        let dirty = packageRoot.flatMap { Subprocess.run("/usr/bin/git", ["status", "--porcelain"], cwd: $0) }.map { !$0.isEmpty } ?? false
        return HostFingerprint(
            chip: Sysctl.string("machdep.cpu.brand_string") ?? "unknown",
            modelIdentifier: Sysctl.string("hw.model") ?? "unknown",
            cpuCores: Sysctl.int("hw.ncpu") ?? ProcessInfo.processInfo.activeProcessorCount,
            performanceCores: Sysctl.int("hw.perflevel0.physicalcpu"),
            efficiencyCores: Sysctl.int("hw.perflevel1.physicalcpu"),
            memoryBytes: UInt64(Sysctl.int("hw.memsize") ?? 0),
            osVersion: "os\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
            osBuild: Sysctl.string("kern.osversion") ?? "unknown",
            arch: arch,
            hostname: ProcessInfo.processInfo.hostName,
            swiftVersion: detectSwiftVersion(),
            buildConfiguration: buildConfiguration,
            gitSHA: sha, gitDirty: dirty,
            power: PowerState.read())
    }
}
