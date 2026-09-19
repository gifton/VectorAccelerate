import Foundation
import Metal
import IOKit
import VectorTestKit

enum MetalFingerprint {
    static func gather(device: (any MTLDevice)?) -> GPUInfo? {
        guard let device else { return nil }
        return GPUInfo(name: device.name, coreCount: gpuCoreCount(), unifiedMemory: device.hasUnifiedMemory,
                       maxThreadgroupBytes: device.maxThreadgroupMemoryLength, metal4: device.supportsFamily(.metal4),
                       metalToolchain: metalToolchainVersion())
    }

    /// `gpu-core-count` on the first AGXAccelerator registry entry; nil when absent (never guessed).
    static func gpuCoreCount() -> Int? {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        guard service != 0 else { return nil }
        defer { IOObjectRelease(service) }
        guard let value = IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() else { return nil }
        return (value as? NSNumber)?.intValue
    }

    /// First line of `xcrun -sdk macosx metal --version`, e.g. "Apple metal version 32023.883 (metalfe-32023.883)".
    static func metalToolchainVersion() -> String? {
        guard let out = Subprocess.run("/usr/bin/xcrun", ["-sdk", "macosx", "metal", "--version"], timeout: 10),
              let line = out.split(separator: "\n").first,
              let m = line.range(of: #"version (\S+)"#, options: .regularExpression) else { return nil }
        return String(line[m]).replacingOccurrences(of: "version ", with: "")
    }
}
