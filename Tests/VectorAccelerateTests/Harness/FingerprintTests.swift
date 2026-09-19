import XCTest
import VectorTestKit

/// Spec §4.9 (host fingerprint, environment key) and §4.10 (preflight probes): host facts populate on this machine,
/// the environment key excludes the hostname and tracks the validation flag, and `evaluate` names every quiet-breaker.
final class FingerprintTests: XCTestCase {
    func testGatherPopulatesHostFacts() {
        let f = HostFingerprint.gather(packageRoot: RepoPaths.root)
        XCTAssertFalse(f.chip.isEmpty); XCTAssertFalse(f.modelIdentifier.isEmpty)
        XCTAssertGreaterThan(f.cpuCores, 0); XCTAssertGreaterThan(f.memoryBytes, 0)
        XCTAssertFalse(f.osBuild.isEmpty); XCTAssertTrue(f.osVersion.hasPrefix("os"))
        XCTAssertEqual(f.arch, "arm64")
        XCTAssertFalse(f.hostname.isEmpty)
        XCTAssertNotNil(f.gitSHA, "tests run inside the repository; git rev-parse must succeed")
        XCTAssertTrue(["debug", "release"].contains(f.buildConfiguration))
        XCTAssertTrue(["ac", "battery", "unknown"].contains(f.power.source))
    }

    func testEnvironmentKeyExcludesHostnameAndTracksValidationFlag() {
        let a = EnvironmentKey.make(chip: "Apple M3 Max", gpuCores: 40, osBuild: "25F84", metalToolchain: "32023.883", swiftVersion: "6.3.3", buildConfiguration: "release", validationLayers: false)
        XCTAssertEqual(a, "applem3max-g40-25F84-mtl32023.883-swift6.3.3-release-noval")
        let b = EnvironmentKey.make(chip: "Apple M3 Max", gpuCores: 40, osBuild: "25F84", metalToolchain: "32023.883", swiftVersion: "6.3.3", buildConfiguration: "release", validationLayers: true)
        XCTAssertTrue(b.hasSuffix("-val")); XCTAssertNotEqual(a, b)
        let c = EnvironmentKey.make(chip: "Apple M3 Max", gpuCores: nil, osBuild: "25F84", metalToolchain: nil, swiftVersion: "6.3.3", buildConfiguration: "debug", validationLayers: false)
        XCTAssertEqual(c, "applem3max-g?-25F84-mtlunknown-swift6.3.3-debug-noval")
        XCTAssertFalse(a.contains(ProcessInfo.processInfo.hostName))
    }

    func testDeviceTagMatchesVectorCoreShape() {
        XCTAssertEqual(HostFingerprint.deviceTag(arch: "arm64", osVersion: "os26.5.2", hostname: "my mac"), "arm64-os26.5.2-my-mac")
    }

    func testSwiftVersionDetectionParses() {
        let v = HostFingerprint.detectSwiftVersion()
        XCTAssertTrue(v.range(of: #"^\d+\.\d+(\.\d+)?$"#, options: .regularExpression) != nil, "got \(v)")
    }

    func testThermalStateNameIsKnown() {
        XCTAssertTrue(["nominal", "fair", "serious", "critical", "unknown"].contains(HostFingerprint.thermalStateName()))
    }

    func testCPUCopyProbeIsPositive() {
        XCTAssertGreaterThan(PreflightProbe.cpuCopyGBps(bytes: 16 << 20, repeats: 2), 0)
    }

    func testPreflightEvaluate() {
        let quiet = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 49), gpu: StartEnd(start: 370, end: 372), thermalStart: "nominal", thermalEnd: "nominal", gpuDeviceExists: true)
        XCTAssertTrue(quiet.quiet); XCTAssertEqual(quiet.driftPercent, 2.0, accuracy: 1e-9); XCTAssertEqual(quiet.reasons, [])
        let drift = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 40), gpu: nil, thermalStart: "nominal", thermalEnd: "nominal", gpuDeviceExists: false)
        XCTAssertFalse(drift.quiet); XCTAssertEqual(drift.driftPercent, 20.0, accuracy: 1e-9)
        XCTAssertEqual(drift.reasons, ["drift 20.0% exceeds 5.0%"])
        let hot = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 50), gpu: StartEnd(start: 1, end: 1), thermalStart: "nominal", thermalEnd: "fair", gpuDeviceExists: true)
        XCTAssertEqual(hot.reasons, ["thermal state end=fair (not nominal)"])
        let noGPU = PreflightProbe.evaluate(cpu: StartEnd(start: 50, end: 50), gpu: nil, thermalStart: "nominal", thermalEnd: "nominal", gpuDeviceExists: true)
        XCTAssertEqual(noGPU.reasons, ["GPU probe unavailable although a Metal device exists"])
    }
}
