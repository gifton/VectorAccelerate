import Foundation

// Artifact schema v1 (spec §4.8). Property names are the JSON keys.

public enum TimingKind: String, Codable, Sendable, CaseIterable { case wall, commandBuffer, encoder }
public enum CaseStatus: String, Codable, Sendable { case measured, unavailable, failed }
public enum CorrectnessStatus: String, Codable, Sendable { case pass, fail, notMeasured }

public enum ArtifactValidationError: Error, Equatable {
    case unsupportedSchemaVersion(Int)
    case missing(String)
    case invalid(String)
}

public struct StartEnd: Codable, Sendable, Equatable {
    public var start: Double
    public var end: Double
    public init(start: Double, end: Double) { self.start = start; self.end = end }
}

public struct GPUInfo: Codable, Sendable, Equatable {
    public var name: String
    public var coreCount: Int?
    public var unifiedMemory: Bool
    public var maxThreadgroupBytes: Int
    public var metal4: Bool
    public var metalToolchain: String?
    public init(name: String, coreCount: Int?, unifiedMemory: Bool, maxThreadgroupBytes: Int, metal4: Bool, metalToolchain: String?) {
        self.name = name; self.coreCount = coreCount; self.unifiedMemory = unifiedMemory
        self.maxThreadgroupBytes = maxThreadgroupBytes; self.metal4 = metal4; self.metalToolchain = metalToolchain
    }
}

public struct PowerInfo: Codable, Sendable, Equatable {
    /// "ac" | "battery" | "unknown" (producer convention, not validated)
    public var source: String
    public var lowPowerMode: Bool
    public init(source: String, lowPowerMode: Bool) { self.source = source; self.lowPowerMode = lowPowerMode }
}

public struct ThermalInfo: Codable, Sendable, Equatable {
    /// "nominal" | "fair" | "serious" | "critical" | "unknown" (producer convention, not validated)
    public var start: String
    public var end: String
    public init(start: String, end: String) { self.start = start; self.end = end }
}

public struct PreflightRecord: Codable, Sendable, Equatable {
    public var cpuCopyGBps: StartEnd
    public var gpuCopyGBps: StartEnd?
    public var driftPercent: Double
    public var quiet: Bool
    public var reasons: [String]
    public init(cpuCopyGBps: StartEnd, gpuCopyGBps: StartEnd?, driftPercent: Double, quiet: Bool, reasons: [String]) {
        self.cpuCopyGBps = cpuCopyGBps; self.gpuCopyGBps = gpuCopyGBps; self.driftPercent = driftPercent; self.quiet = quiet; self.reasons = reasons
    }
}

public struct Filters: Codable, Sendable, Equatable {
    public var mode: String
    public var include: [String]?
    public var exclude: [String]?
    public init(mode: String = "glob", include: [String]?, exclude: [String]?) { self.mode = mode; self.include = include; self.exclude = exclude }
}

public struct Metadata: Codable, Sendable, Equatable {
    public var packageVersion: String
    public var gitSHA: String?
    public var gitDirty: Bool
    public var date: String
    public var os: String
    public var osBuild: String
    public var arch: String
    public var cpuCores: Int
    public var performanceCores: Int?
    public var efficiencyCores: Int?
    public var memoryBytes: UInt64
    public var deviceModel: String
    public var modelIdentifier: String
    public var hostname: String
    public var swiftVersion: String
    public var buildConfiguration: String
    public var deviceTag: String
    public var environmentKey: String
    public var gpu: GPUInfo?
    public var power: PowerInfo
    public var thermal: ThermalInfo
    public var preflight: PreflightRecord
    public var validationLayers: Bool
    public var runSeed: UInt64
    public var runLabel: String?
    public var mode: String
    public var filters: Filters?
    public var flags: [String: String]

    public init(packageVersion: String, gitSHA: String?, gitDirty: Bool, date: String, os: String, osBuild: String, arch: String,
                cpuCores: Int, performanceCores: Int?, efficiencyCores: Int?, memoryBytes: UInt64, deviceModel: String,
                modelIdentifier: String, hostname: String, swiftVersion: String, buildConfiguration: String, deviceTag: String,
                environmentKey: String, gpu: GPUInfo?, power: PowerInfo, thermal: ThermalInfo, preflight: PreflightRecord,
                validationLayers: Bool, runSeed: UInt64, runLabel: String?, mode: String, filters: Filters?, flags: [String: String]) {
        self.packageVersion = packageVersion; self.gitSHA = gitSHA; self.gitDirty = gitDirty; self.date = date; self.os = os
        self.osBuild = osBuild; self.arch = arch; self.cpuCores = cpuCores; self.performanceCores = performanceCores
        self.efficiencyCores = efficiencyCores; self.memoryBytes = memoryBytes; self.deviceModel = deviceModel
        self.modelIdentifier = modelIdentifier; self.hostname = hostname; self.swiftVersion = swiftVersion
        self.buildConfiguration = buildConfiguration; self.deviceTag = deviceTag; self.environmentKey = environmentKey
        self.gpu = gpu; self.power = power; self.thermal = thermal; self.preflight = preflight
        self.validationLayers = validationLayers; self.runSeed = runSeed; self.runLabel = runLabel; self.mode = mode
        self.filters = filters; self.flags = flags
    }
}

public struct CaseIdentity: Codable, Sendable, Equatable {
    public var id: String
    public var fixtureDigest: String
    public var generatorVersion: String
    public var seed: UInt64
    public var stream: UInt64
    public var route: String
    public var variant: String?
    public var dataDescriptor: String
    public init(id: String, fixtureDigest: String, generatorVersion: String, seed: UInt64, stream: UInt64, route: String, variant: String?, dataDescriptor: String) {
        self.id = id; self.fixtureDigest = fixtureDigest; self.generatorVersion = generatorVersion; self.seed = seed
        self.stream = stream; self.route = route; self.variant = variant; self.dataDescriptor = dataDescriptor
    }
}

public struct Series: Codable, Sendable, Equatable {
    public var unit: String
    /// raw[sample][iteration]
    public var raw: [[Double]]
    public var summary: Summary
    public var bias: String?
    public init(unit: String = "ns", raw: [[Double]], summary: Summary, bias: String?) { self.unit = unit; self.raw = raw; self.summary = summary; self.bias = bias }
}

public struct Timing: Codable, Sendable, Equatable {
    public var wall: Series?
    public var commandBuffer: Series?
    public var encoder: Series?
    public init(wall: Series?, commandBuffer: Series?, encoder: Series?) { self.wall = wall; self.commandBuffer = commandBuffer; self.encoder = encoder }

    public subscript(kind: TimingKind) -> Series? {
        switch kind { case .wall: return wall; case .commandBuffer: return commandBuffer; case .encoder: return encoder }
    }
}

public struct Correctness: Codable, Sendable, Equatable {
    public var status: CorrectnessStatus
    public var comparator: String
    public var tolerance: Tolerance?
    public var maxAbsError: Double?
    public var maxRelError: Double?
    public var recall: Double?
    public var recallPolicy: RecallPolicy?
    public var notes: [String]
    public init(status: CorrectnessStatus, comparator: String, tolerance: Tolerance?, maxAbsError: Double?, maxRelError: Double?, recall: Double?, recallPolicy: RecallPolicy?, notes: [String]) {
        self.status = status; self.comparator = comparator; self.tolerance = tolerance; self.maxAbsError = maxAbsError
        self.maxRelError = maxRelError; self.recall = recall; self.recallPolicy = recallPolicy; self.notes = notes
    }
}

public struct Resources: Codable, Sendable, Equatable {
    public var staticThreadgroupBytes: Int?
    public var maxThreadgroupBytes: Int?
    public var poolHighWaterBytes: Int?
    public var notes: [String]
    public init(staticThreadgroupBytes: Int?, maxThreadgroupBytes: Int?, poolHighWaterBytes: Int?, notes: [String]) {
        self.staticThreadgroupBytes = staticThreadgroupBytes; self.maxThreadgroupBytes = maxThreadgroupBytes
        self.poolHighWaterBytes = poolHighWaterBytes; self.notes = notes
    }
}

public struct CaseRecord: Codable, Sendable, Equatable {
    public var id: String
    public var identity: CaseIdentity
    public var status: CaseStatus
    public var failureReason: String?
    public var samples: Int
    public var iterations: Int
    public var warmupUsed: [Int]
    public var unitCount: Int
    public var timing: Timing
    public var correctness: Correctness?
    public var resources: Resources?
    public var notes: [String]
    public init(id: String, identity: CaseIdentity, status: CaseStatus, failureReason: String?, samples: Int, iterations: Int, warmupUsed: [Int],
                unitCount: Int, timing: Timing, correctness: Correctness?, resources: Resources?, notes: [String]) {
        self.id = id; self.identity = identity; self.status = status; self.failureReason = failureReason; self.samples = samples
        self.iterations = iterations; self.warmupUsed = warmupUsed; self.unitCount = unitCount; self.timing = timing
        self.correctness = correctness; self.resources = resources; self.notes = notes
    }
}

public struct BenchArtifact: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var package: String
    public var metadata: Metadata
    public var cases: [CaseRecord]
    public init(schemaVersion: Int = 1, package: String, metadata: Metadata, cases: [CaseRecord]) {
        self.schemaVersion = schemaVersion; self.package = package; self.metadata = metadata; self.cases = cases
    }

    private enum CodingKeys: String, CodingKey { case schemaVersion, package, metadata, cases }

    /// Decoding an unknown `schemaVersion` fails with a typed error before the payload is read (spec §4.8);
    /// `encode(to:)` stays synthesized. Pinned by `ArtifactSchemaTests.testUnknownSchemaVersionRejectedAtDecode`.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let version = try container.decode(Int.self, forKey: .schemaVersion)
        guard version == 1 else { throw ArtifactValidationError.unsupportedSchemaVersion(version) }
        schemaVersion = version
        package = try container.decode(String.self, forKey: .package)
        metadata = try container.decode(Metadata.self, forKey: .metadata)
        cases = try container.decode([CaseRecord].self, forKey: .cases)
    }

    /// Structural validation beyond what `Codable` enforces (spec §4.8). The version check is repeated here
    /// because in-memory records can be mutated after decoding.
    public func validate() throws {
        guard schemaVersion == 1 else { throw ArtifactValidationError.unsupportedSchemaVersion(schemaVersion) }
        if metadata.environmentKey.isEmpty { throw ArtifactValidationError.missing("metadata.environmentKey") }
        if metadata.deviceModel.isEmpty { throw ArtifactValidationError.missing("metadata.deviceModel") }
        var seen = Set<String>()
        for c in cases {
            guard seen.insert(c.id).inserted else { throw ArtifactValidationError.invalid("duplicate case id: \(c.id)") }
            if c.id != c.identity.id { throw ArtifactValidationError.invalid("case \(c.id): identity.id differs (\(c.identity.id))") }
        }
        for c in cases { try c.validate() }
    }
}

public extension CaseRecord {
    /// Per-case structural checks; intended for reuse by the compare step on baseline records.
    func validate() throws {
        // Non-measured cases may legitimately carry zero counts.
        if status == .measured && (samples < 1 || iterations < 1 || unitCount < 1) {
            throw ArtifactValidationError.invalid("case \(id): samples/iterations/unitCount must be >= 1 for measured cases")
        }
        if warmupUsed.count != samples { throw ArtifactValidationError.invalid("case \(id): warmupUsed.count must equal samples") }
        if status == .measured && timing.wall == nil { throw ArtifactValidationError.missing("case \(id): timing.wall") }
        for kind in TimingKind.allCases {
            guard let s = timing[kind] else { continue }
            let shapeOK = s.raw.count == samples && s.raw.allSatisfy { $0.count == iterations }
            if !shapeOK { throw ArtifactValidationError.invalid("case \(id): timing.\(kind.rawValue) raw shape must be samples×iterations (\(samples)×\(iterations))") }
        }
        if let r = correctness?.recall, !(0...1).contains(r) { throw ArtifactValidationError.invalid("case \(id): correctness.recall must be in [0, 1]") }
    }
}
