// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
// ============================================================================
// METAL 4 ONLY - BREAKING CHANGE
// ============================================================================
// VectorAccelerate 0.4.2+ is Metal 4 ONLY and requires:
//   - macOS 26.0+ (Tahoe)
//   - iOS 26.0+
//   - tvOS 26.0+
//   - visionOS 3.0+
//
// There is NO backwards compatibility with older OS versions.
// For Metal 3 support, use VectorAccelerate 0.2.x or earlier.
//
// 0.4.4 — VectorCore 0.2.1 integration + internal refactor:
//   - Requires VectorCore 0.2.1+ (NormalizationHint<V> now conforms to IndexableVector)
//   - Refactored normalization hint storage into handle-keyed VectorHintStore
//   - Extracted CPU top-K fallback into Sources/VectorAccelerate/CPU/CPUTopK.swift
//   - Split gap tests into FactoryWALTests, DecisionEngineIndexTests, IndexableVectorInsertTests
//   - Removed: walConfiguration param from .ivfAuto() (use .ivf(...) directly for WAL on IVF)
//
// 0.4.3 — Integration surface gaps on AcceleratedVectorIndex:
//   - Added: walConfiguration param on IndexConfiguration.flat() and .ivf() factories
//   - Added: decisionEngine param on AcceleratedVectorIndex.init (CPU fallback for flat search)
//   - Added: insert<V: IndexableVector>(_:metadata:) single + batch overloads
//   - Added: vectorHints(for:) accessor for captured normalization hints
//
// 0.4.2 REMOVED (deprecated since 0.4.0, unused by all known consumers):
//   - Removed: AccelerationConfiguration.{cpuThreshold, gpuThreshold, hybridThreshold}
//   - Removed: AccelerationConfiguration.{performance, balanced, cpuOnly} factories
//   - Removed: IndexAccelerationConfiguration.{minimumCandidatesForGPU, minimumOperationsForGPU}
//   - Removed: IndexAccelerationConfiguration.{aggressive, conservative, benchmarking} factories
//   - Removed: BatchConfiguration.gpuThreshold (decisionEngine is now always supplied)
//   - Removed: AdaptiveThresholdManager (use GPUDecisionEngine directly)
//   - Removed: StreamingKernel typealias (use StreamingTopKKernel directly)
//   - Removed: toGPUActivationThresholds() / createDecisionEngine() migration helpers
//   Use GPUDecisionEngine for all adaptive GPU/CPU routing decisions.
//
// NOTE: The platform versions below are SPM placeholders because Swift Package
// Manager does not yet support .v26 enum cases. Runtime availability is enforced
// via @available attributes. These will be updated when SPM adds support.
// ============================================================================

import PackageDescription

let package = Package(
    name: "VectorAccelerate",
    platforms: [
        // PLACEHOLDER VALUES - Actual requirement is macOS 26.0+ / iOS 26.0+
        // These will be updated to .macOS(.v26), .iOS(.v26), etc. when SPM supports them
        .macOS(.v26),
        .iOS(.v26),
        .tvOS(.v26),
        .visionOS(.v26)
        // watchOS removed - no Metal 4 support (no GPU)
        // Linux not supported - Metal dependency
    ],
    products: [
        .library(
            name: "VectorAccelerate",
            targets: ["VectorAccelerate"]),
        .executable(
            name: "VectorAccelerateBenchmarks",
            targets: ["VectorAccelerateBenchmarks"]),
    ],
    dependencies: [
        // VectorCore for base protocols and types
        .package(url: "https://github.com/gifton/VectorCore", from: "0.2.1"),
        // MetalCompilerPlugin for debuggable Metal shaders (enables Xcode Metal Debugger)
        .package(url: "https://github.com/schwa/MetalCompilerPlugin", from: "0.1.5")
    ],
    targets: [
        // MARK: - Core GPU Acceleration
        // Includes GPU-first vector index (AcceleratedVectorIndex) and all Metal kernels
        .target(
            name: "VectorAccelerate",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore"),
            ],
            exclude: [
                "metal-compiler-plugin.json"  // Build-time config for MetalCompilerPlugin, not a runtime resource
            ],
            resources: [
                .process("Metal/Shaders")  // Metal shader files
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ],
            plugins: [
                // Compiles Metal shaders with debug symbols for Xcode Metal Debugger
                // Creates debug.metallib alongside SPM's default.metallib
                .plugin(name: "MetalCompilerPlugin", package: "MetalCompilerPlugin")
            ]
        ),

        // MARK: - Benchmarks
        .executableTarget(
            name: "VectorAccelerateBenchmarks",
            dependencies: [
                "VectorAccelerate",
                .product(name: "VectorCore", package: "VectorCore")
            ]
        ),

        // MARK: - Tests
        .testTarget(
            name: "VectorAccelerateTests",
            dependencies: [
                "VectorAccelerate",
                .product(name: "VectorCore", package: "VectorCore")
            ]
        )
    ]
)
