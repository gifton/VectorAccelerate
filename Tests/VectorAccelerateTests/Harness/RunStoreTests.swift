import XCTest
import VectorTestKit

final class RunStoreTests: XCTestCase {
    private var root: URL!
    override func setUp() { root = FileManager.default.temporaryDirectory.appendingPathComponent("runstore-\(UUID().uuidString)") }
    override func tearDown() { try? FileManager.default.removeItem(at: root) }

    private func artifact(dirty: Bool = false, quiet: Bool = true, validation: Bool = false) -> BenchArtifact {
        var a = ArtifactSchemaTests.sampleArtifact(validation: validation)
        a.metadata.gitDirty = dirty
        a.metadata.preflight.quiet = quiet
        if !quiet { a.metadata.preflight.reasons = ["drift 9.0% exceeds 5.0%"] }
        return a
    }

    func testRunPathLayoutAndCreateOnly() throws {
        let store = RunStore(root: root)
        let date = Date(timeIntervalSince1970: 1_789_000_000)   // 2026-09-10T00:26:40Z
        let url = try store.writeRun(artifact(), at: date)
        XCTAssertEqual(url.deletingLastPathComponent().lastPathComponent, "applem3max-g40-25F84-mtl32023.883-swift6.3.3-release-noval")
        XCTAssertEqual(url.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent, "runs")
        XCTAssertEqual(url.lastPathComponent, "20260910T002640Z_deadbee.json")
        XCTAssertThrowsError(try store.writeRun(artifact(), at: date)) { XCTAssertEqual($0 as? RunStoreError, .runExists(url.path)) }
        let dirtyURL = store.runURL(for: artifact(dirty: true), at: date)
        XCTAssertEqual(dirtyURL.lastPathComponent, "20260910T002640Z_deadbee_dirty.json")
        XCTAssertEqual(try RunStore.load(url), artifact())
    }

    func testCaptureAppendsAndApprovePointer() throws {
        let store = RunStore(root: root)
        let key = artifact().metadata.environmentKey, id = "probe.gpu_copy.bytes=8"
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false), [id: 1])
        XCTAssertNil(try store.approvedVersion(environmentKey: key, caseID: id))
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: true, allowDirty: false), [id: 2])
        XCTAssertEqual(try store.approvedVersion(environmentKey: key, caseID: id), 2)
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false), [id: 3])
        XCTAssertEqual(try store.approvedVersion(environmentKey: key, caseID: id), 2, "capture without --approve never moves the pointer")
        XCTAssertNotNil(try store.baseline(environmentKey: key, caseID: id, version: 1))
        XCTAssertNil(try store.baseline(environmentKey: key, caseID: id, version: 9))
        XCTAssertEqual(try store.approvedBaseline(environmentKey: key, caseID: id), artifact().cases[0])
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("baselines/\(key)/\(id)/1.json").path))
    }

    func testCaptureRefusals() {
        let store = RunStore(root: root)
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(validation: true), approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .refusedValidationLayers) }
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(quiet: false), approve: false, allowDirty: false)) {
            XCTAssertEqual($0 as? RunStoreError, .refusedNotQuiet(["drift 9.0% exceeds 5.0%"]))
        }
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(dirty: true), approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .refusedDirty) }
        XCTAssertNoThrow(try store.captureBaseline(from: artifact(dirty: true), approve: false, allowDirty: true))
        var empty = artifact(); empty.cases[0].status = .unavailable
        XCTAssertThrowsError(try store.captureBaseline(from: empty, approve: false, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .noMeasuredCases) }
    }

    func testUnsafePathComponentsAreRefusedBeforeAnyWrite() {
        let store = RunStore(root: root)
        var escape = artifact(); escape.cases[0].id = "../../escape"; escape.cases[0].identity.id = "../../escape"
        XCTAssertThrowsError(try store.captureBaseline(from: escape, approve: true, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("caseID: ../../escape")) }
        var dotdot = artifact(); dotdot.metadata.environmentKey = ".."
        XCTAssertThrowsError(try store.writeRun(dotdot)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("environmentKey: ..")) }
        var sha = artifact(); sha.metadata.gitSHA = "../../x"
        XCTAssertThrowsError(try store.writeRun(sha)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("gitSHA: ../../x")) }
        var nulKey = artifact(); nulKey.metadata.environmentKey = "..\u{0}"; nulKey.cases[0].id = "..\u{0}"; nulKey.cases[0].identity.id = "..\u{0}"
        XCTAssertThrowsError(try store.captureBaseline(from: nulKey, approve: true, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("environmentKey: ..\u{0}")) }
        XCTAssertThrowsError(try store.writeRun(nulKey)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("environmentKey: ..\u{0}")) }
        var nulID = artifact(); nulID.cases[0].id = "a\u{0}b"; nulID.cases[0].identity.id = "a\u{0}b"
        XCTAssertThrowsError(try store.captureBaseline(from: nulID, approve: true, allowDirty: false)) { XCTAssertEqual($0 as? RunStoreError, .unsafePathComponent("caseID: a\u{0}b")) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path), "an unsafe artifact must never create the store root")
    }

    func testBadApprovedPointerIsSurfaced() throws {
        let store = RunStore(root: root)
        let key = artifact().metadata.environmentKey, id = "probe.gpu_copy.bytes=8"
        let dir = root.appendingPathComponent("baselines/\(key)/\(id)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let pointer = dir.appendingPathComponent("approved")
        try "abc\n".write(to: pointer, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try store.approvedVersion(environmentKey: key, caseID: id)) { XCTAssertEqual($0 as? RunStoreError, .badApprovedPointer("abc")) }
        try "0\n".write(to: pointer, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try store.approvedVersion(environmentKey: key, caseID: id)) { XCTAssertEqual($0 as? RunStoreError, .badApprovedPointer("0")) }
        try "9\n".write(to: pointer, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try store.approvedBaseline(environmentKey: key, caseID: id)) { XCTAssertEqual($0 as? RunStoreError, .badApprovedPointer("version 9 missing")) }
    }

    /// A file named `Int.max`.json in a baseline directory once made `last + 1` trap on overflow. It is not a version now, and
    /// a directory that is genuinely full ends in a thrown create-only write error rather than a trap.
    func testVersionNumberingNeverOverflows() throws {
        let store = RunStore(root: root)
        let key = artifact().metadata.environmentKey, id = "probe.gpu_copy.bytes=8"
        let dir = root.appendingPathComponent("baselines/\(key)/\(id)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: dir.appendingPathComponent("\(Int.max).json"))
        XCTAssertEqual(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false), [id: 1])
        try Data("{}".utf8).write(to: dir.appendingPathComponent("\(Int.max - 1).json"))
        XCTAssertThrowsError(try store.captureBaseline(from: artifact(), approve: false, allowDirty: false))
    }

    /// `main.swift` prints a caught error by interpolating it as `any Error`; the same interpolation is checked here. A type
    /// without a description interpolates as its reflected case (`refusedDirty`, `runExists("…")`), which starts with the case
    /// name, hence the `hasPrefix` check. Payloads aside (the reflected rendering carries those too), every expected phrase
    /// contains a space or punctuation, which a case name cannot. The table is maintained by hand: a new `RunStoreError` case
    /// needs a row here.
    func testErrorDescriptionsSayWhatFailedAndCarryThePayload() {
        let table: [(error: RunStoreError, caseName: String, mustContain: [String])] = [
            (.runExists("/tmp/store/runs/key/run.json"), "runExists", ["/tmp/store/runs/key/run.json", "never overwritten"]),
            (.refusedValidationLayers, "refusedValidationLayers", ["baseline capture refused:", "validation layers"]),
            (.refusedNotQuiet(["drift 9.0% exceeds 5.0%", "thermal state serious"]), "refusedNotQuiet",
             ["baseline capture refused:", "not quiet", "drift 9.0% exceeds 5.0%; thermal state serious", "quiet runs"]),
            (.refusedNotQuiet([]), "refusedNotQuiet", ["baseline capture refused:", "not quiet", "no reason recorded"]),
            (.refusedDirty, "refusedDirty", ["baseline capture refused:", "clean tree", "--allow-dirty"]),
            (.noMeasuredCases, "noMeasuredCases", ["baseline capture refused:", "no measured case"]),
            (.badApprovedPointer("version 9 missing"), "badApprovedPointer", ["version 9 missing", "`approved` file"]),
            (.unsafePathComponent("caseID: ../../escape"), "unsafePathComponent", ["caseID: ../../escape", "path component"])
        ]
        for row in table {
            let message = "\(row.error as any Error)"
            XCTAssertFalse(message.isEmpty, row.caseName)
            XCTAssertFalse(message.hasPrefix(row.caseName), "\(row.caseName) renders as the raw case: \(message)")
            for piece in row.mustContain { XCTAssertTrue(message.contains(piece), "\(row.caseName): '\(piece)' missing from: \(message)") }
        }
    }
}
