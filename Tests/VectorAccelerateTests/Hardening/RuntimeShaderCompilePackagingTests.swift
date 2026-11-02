//
//  RuntimeShaderCompilePackagingTests.swift
//  VectorAccelerateTests
//
//  The runtime shader compile is a fallback that must fail loudly when it can't do its job
//  (docs/stability/SHADER-LIBRARY-PACKAGING-CONTRACT.md). It used to skip missing files silently and
//  return a library holding only its preamble, or strand every kernel of the skipped file
//  (AUDIT-2 VA2-001).
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
final class RuntimeShaderCompilePackagingTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("va-packaging-\(UUID().uuidString).bundle", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    private func compileError() throws -> String {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let bundle = try XCTUnwrap(Bundle(url: directory))
        do {
            _ = try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)
        } catch {
            return String(describing: error)
        }
        XCTFail("A bundle without its full set of sources must not produce a library")
        return ""
    }

    func testBundleWithoutSourcesThrows() throws {
        let message = try compileError()
        XCTAssertTrue(message.contains("carries no .metal sources"), message)
    }

    func testBundleMissingSomeSourcesThrowsAndNamesThem() throws {
        let listed = KernelContext.runtimeCompileShaderFiles
        let present = try XCTUnwrap(listed.first)
        let absent = try XCTUnwrap(listed.last)
        XCTAssertNotEqual(present, absent)
        try Data("// empty".utf8).write(to: directory.appendingPathComponent("\(present).metal"))

        let message = try compileError()

        XCTAssertTrue(message.contains("missing \(listed.count - 1) of \(listed.count)"), message)
        XCTAssertTrue(message.contains(absent), message)
    }
}
