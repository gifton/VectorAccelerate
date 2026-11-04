//
//  ShippedMetalLibraries.swift
//  VectorAccelerateTests
//
//  The Metal libraries a test can certify in this build, in `KernelContext.loadMetalLibrary` order:
//    "plugin"  — `debug.metallib` from MetalCompilerPlugin, loaded in Debug builds only;
//    "spm"     — `default.metallib`, which Swift Build (the default build system since Swift 6.4) compiles from
//                the `.process`ed `.metal` resources; loaded in Release under that build system;
//    "runtime" — `KernelContext.makeLibraryFromBundleSources`, the combined-source compile, loaded in Release
//                under SwiftPM's native build system; available only when the bundle carries the sources.
//
//  Owner decision 2026-10-31 (docs/stability/SHADER-LIBRARY-PACKAGING-CONTRACT.md): the precompiled metallib is
//  the shipped library and the runtime compile is a fallback that fails loudly without sources. The hardening
//  suites therefore certify whatever this build's bundle ships instead of assuming a "runtime" leg exists;
//  `ShaderLibraryCompletenessTests` asserts that every shipped library exposes the same kernel set.
//

import XCTest
@preconcurrency import Metal
@testable import VectorAccelerate

enum ShippedMetalLibraries {
    struct Entry {
        let label: String
        let library: any MTLLibrary
    }

    /// Every library this build's bundle can load, labelled. Fails the test (rather than skipping) when the
    /// bundle holds none: a bundle without any Metal library is a packaging defect.
    static func entries(device: any MTLDevice) throws -> [Entry] {
        let bundle = try XCTUnwrap(KernelContext.findVectorAccelerateBundle(), "VectorAccelerate resource bundle not found")
        var entries: [Entry] = []
        #if DEBUG
        if let url = bundle.url(forResource: "debug", withExtension: "metallib") {
            entries.append(Entry(label: "plugin", library: try device.makeLibrary(URL: url)))
        }
        #endif
        if let url = bundle.url(forResource: "default", withExtension: "metallib") {
            entries.append(Entry(label: "spm", library: try device.makeLibrary(URL: url)))
        }
        if !KernelContext.bundledShaderSourceURLs(in: bundle).isEmpty {
            entries.append(Entry(label: "runtime", library: try KernelContext.makeLibraryFromBundleSources(device: device, bundle: bundle)))
        }
        XCTAssertFalse(entries.isEmpty, "the resource bundle at \(bundle.bundlePath) ships no Metal library")
        return entries
    }

    /// Runs `body` once per shipped library. Skips when no Metal device exists.
    static func forEach(_ body: (any MTLDevice, any MTLLibrary, String) throws -> Void) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal unavailable") }
        for entry in try entries(device: device) {
            try body(device, entry.library, entry.label)
        }
    }

    /// Runs `body` once per shipped library on an already-chosen device.
    static func forEach(device: any MTLDevice, _ body: (any MTLLibrary, String) throws -> Void) throws {
        for entry in try entries(device: device) {
            try body(entry.library, entry.label)
        }
    }

    /// The union of every shipped library's function names.
    static func allFunctionNames(device: any MTLDevice) throws -> Set<String> {
        var names = Set<String>()
        for entry in try entries(device: device) { names.formUnion(entry.library.functionNames) }
        return names
    }
}
