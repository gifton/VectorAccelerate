// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
// ============================================================================
// METAL 4 ONLY - BREAKING CHANGE
// ============================================================================
// VectorAccelerate 0.4.0+ is Metal 4 ONLY and requires:
//   - macOS 26.0+ (Tahoe)
//   - iOS 26.0+
//   - tvOS 26.0+
//   - visionOS 3.0+
//
// There is NO backwards compatibility with older OS versions.
// For Metal 3 support, use VectorAccelerate 0.2.x or earlier.
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
        .package(url: "https://github.com/gifton/VectorCore", from: "0.1.6")
    ],
    targets: [
        // MARK: - Core GPU Acceleration
        // Includes GPU-first vector index (AcceleratedVectorIndex) and all Metal kernels
        .target(
            name: "VectorAccelerate",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore"),
            ],
            resources: [
                .process("Metal/Shaders")  // Metal shader files
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
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
