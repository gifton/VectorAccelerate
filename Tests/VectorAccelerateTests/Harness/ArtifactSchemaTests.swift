import XCTest
import VectorTestKit

final class ArtifactSchemaTests: XCTestCase {
    static func sampleArtifact(validation: Bool = false) -> BenchArtifact {
        let series = Series(unit: "ns", raw: [[1, 2, 3], [2, 3, 4]], summary: Stats.summarize([1, 2, 3, 2, 3, 4], bootstrapSeed: 1), bias: nil)
        let identity = CaseIdentity(id: "probe.gpu_copy.bytes=8", fixtureDigest: "abc", generatorVersion: "1", seed: 42, stream: 0,
                                    route: "gpu.probe_copy", variant: nil, dataDescriptor: "pattern")
        let c = CaseRecord(id: "probe.gpu_copy.bytes=8", identity: identity, status: .measured, failureReason: nil,
                           samples: 2, iterations: 3, warmupUsed: [3, 3], unitCount: 1,
                           timing: Timing(wall: series, commandBuffer: nil, encoder: nil),
                           correctness: Correctness(status: .pass, comparator: "memcmp", tolerance: nil, maxAbsError: nil, maxRelError: nil, recall: nil, recallPolicy: nil, notes: []),
                           resources: nil, notes: [])
        let md = Metadata(packageVersion: "0.7.0", gitSHA: "deadbeef", gitDirty: false, date: "2026-09-17T00:00:00Z",
                          os: "os26.5.2", osBuild: "25F84", arch: "arm64", cpuCores: 16, performanceCores: 12, efficiencyCores: 4,
                          memoryBytes: 1, deviceModel: "Apple M3 Max", modelIdentifier: "Mac15,9", hostname: "h",
                          swiftVersion: "6.3.3", buildConfiguration: "release", deviceTag: "arm64-os26.5.2-h",
                          environmentKey: "applem3max-g40-25F84-mtl32023.883-swift6.3.3-release-noval",
                          gpu: GPUInfo(name: "Apple M3 Max", coreCount: 40, unifiedMemory: true, maxThreadgroupBytes: 32768, metal4: true, metalToolchain: "32023.883"),
                          power: PowerInfo(source: "ac", lowPowerMode: false), thermal: ThermalInfo(start: "nominal", end: "nominal"),
                          preflight: PreflightRecord(cpuCopyGBps: StartEnd(start: 50, end: 50), gpuCopyGBps: StartEnd(start: 370, end: 370), driftPercent: 0, quiet: true, reasons: []),
                          validationLayers: validation, runSeed: 42, runLabel: nil, mode: "quick", filters: nil, flags: [:])
        return BenchArtifact(schemaVersion: 1, package: "VectorAccelerate", metadata: md, cases: [c])
    }

    func testRoundTrip() throws {
        let a = Self.sampleArtifact()
        let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys]
        let data = try enc.encode(a)
        let back = try JSONDecoder().decode(BenchArtifact.self, from: data)
        XCTAssertEqual(back, a)
        try back.validate()
        let json = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(json.contains("\"preflight\"") && json.contains("\"environmentKey\"") && json.contains("\"commandBuffer\":null") == false)
    }

    func testValidationLayersArtifactIsValid() throws {
        try Self.sampleArtifact(validation: true).validate()
    }

    func testUnknownSchemaVersionRejected() {
        var a = Self.sampleArtifact(); a.schemaVersion = 2
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .unsupportedSchemaVersion(2)) }
    }

    func testRawShapeMustMatchSamplesAndIterations() {
        var a = Self.sampleArtifact(); a.cases[0].iterations = 4
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: timing.wall raw shape must be samples×iterations (2×4)")) }
    }

    func testWarmupUsedCountMustEqualSamples() {
        var a = Self.sampleArtifact(); a.cases[0].warmupUsed = [3]
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: warmupUsed.count must equal samples")) }
    }

    func testMeasuredRequiresWall() {
        var a = Self.sampleArtifact(); a.cases[0].timing.wall = nil
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .missing("case probe.gpu_copy.bytes=8: timing.wall")) }
    }

    func testRecallRange() {
        var a = Self.sampleArtifact(); a.cases[0].correctness?.recall = 1.5
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: correctness.recall must be in [0, 1]")) }
    }

    func testMissingRequiredMetadataFieldFailsDecoding() throws {
        let data = try JSONEncoder().encode(Self.sampleArtifact())
        var obj = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        var md = obj["metadata"] as! [String: Any]; md.removeValue(forKey: "environmentKey"); obj["metadata"] = md
        let stripped = try JSONSerialization.data(withJSONObject: obj)
        XCTAssertThrowsError(try JSONDecoder().decode(BenchArtifact.self, from: stripped)) { error in
            guard let decodingError = error as? DecodingError, case .keyNotFound(let key, _) = decodingError else {
                return XCTFail("expected DecodingError.keyNotFound, got \(error)")
            }
            XCTAssertEqual(key.stringValue, "environmentKey")
        }
    }

    func testTimingSubscript() {
        let a = Self.sampleArtifact()
        XCTAssertNotNil(a.cases[0].timing[.wall]); XCTAssertNil(a.cases[0].timing[.encoder])
    }

    func testMeasuredRequiresPositiveCounts() {
        var a = Self.sampleArtifact()
        a.cases[0].samples = 0; a.cases[0].iterations = 0; a.cases[0].warmupUsed = []
        a.cases[0].timing.wall = Series(raw: [], summary: Stats.summarize([], bootstrapSeed: 0), bias: nil)
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: samples/iterations/unitCount must be >= 1 for measured cases")) }
    }

    func testUnknownSchemaVersionRejectedAtDecode() throws {
        var a = Self.sampleArtifact(); a.schemaVersion = 2
        let data = try JSONEncoder().encode(a)
        XCTAssertThrowsError(try JSONDecoder().decode(BenchArtifact.self, from: data)) { XCTAssertEqual($0 as? ArtifactValidationError, .unsupportedSchemaVersion(2)) }
    }

    func testCaseIdsMustBeUniqueAndMatchIdentity() {
        var dup = Self.sampleArtifact(); dup.cases.append(dup.cases[0])
        XCTAssertThrowsError(try dup.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("duplicate case id: probe.gpu_copy.bytes=8")) }
        var mismatch = Self.sampleArtifact(); mismatch.cases[0].identity.id = "other"
        XCTAssertThrowsError(try mismatch.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .invalid("case probe.gpu_copy.bytes=8: identity.id differs (other)")) }
    }

    func testEmptyRequiredMetadataFieldsRejected() {
        var a = Self.sampleArtifact(); a.metadata.environmentKey = ""
        XCTAssertThrowsError(try a.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .missing("metadata.environmentKey")) }
        var b = Self.sampleArtifact(); b.metadata.deviceModel = ""
        XCTAssertThrowsError(try b.validate()) { XCTAssertEqual($0 as? ArtifactValidationError, .missing("metadata.deviceModel")) }
    }

    func testUInt64FieldsRoundTripAtFullRange() throws {
        var a = Self.sampleArtifact()
        a.cases[0].identity.seed = 0x9E3779B97F4A7C15
        a.metadata.runSeed = UInt64.max
        let back = try JSONDecoder().decode(BenchArtifact.self, from: JSONEncoder().encode(a))
        XCTAssertEqual(back, a)
        XCTAssertEqual(back.cases[0].identity.seed, 0x9E3779B97F4A7C15)
        XCTAssertEqual(back.metadata.runSeed, UInt64.max)
    }
}
