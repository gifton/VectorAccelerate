// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
// IMPORTANT: VectorAccelerate 0.2.0+ requires Metal 4 (macOS 26.0+, iOS 26.0+)
// The platform versions below are SPM placeholders; runtime availability is enforced
// via @available(macOS 26.0, iOS 26.0, ...) attributes on all public APIs.
// For older OS support, use VectorAccelerate 0.1.x.

import PackageDescription

let package = Package(
    name: "VectorAccelerate",
    platforms: [
        // Note: Metal 4 requires macOS 26.0+ / iOS 26.0+ at runtime
        // SPM doesn't yet support these platform versions, so we use placeholders
        .macOS(.v15),
        .iOS(.v18),
        .tvOS(.v18),
        .watchOS(.v11),
        .visionOS(.v2)
        // Note: No Linux support due to Metal dependency
    ],
    products: [
        .library(
            name: "VectorAccelerate",
            targets: ["VectorAccelerate"]),
        // VectorIndexAcceleration: GPU-accelerated index operations
        // Provides Metal 4 acceleration for VectorIndex types (HNSW, IVF, Flat)
        .library(
            name: "VectorIndexAcceleration",
            targets: ["VectorIndexAcceleration"]),
        .executable(
            name: "VectorAccelerateBenchmarks",
            targets: ["VectorAccelerateBenchmarks"]),
    ],
    dependencies: [
        // VectorCore for base protocols and types
        .package(url: "https://github.com/gifton/VectorCore", from: "0.1.6"),
        // VectorIndex for index protocols and types (HNSW, IVF, Flat)
        .package(url: "https://github.com/gifton/VectorIndex", from: "0.1.3")
    ],
    targets: [
        // MARK: - Core GPU Acceleration (no VectorIndex dependency)
        .target(
            name: "VectorAccelerate",
            dependencies: [
                .product(name: "VectorCore", package: "VectorCore")
            ],
            resources: [
                .process("Metal/Shaders")  // Metal shader files
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("ExistentialAny")
            ]
        ),

        // MARK: - Index Acceleration (depends on VectorIndex)
        // Provides GPU-accelerated implementations of VectorIndex types
        .target(
            name: "VectorIndexAcceleration",
            dependencies: [
                "VectorAccelerate",
                .product(name: "VectorCore", package: "VectorCore"),
                .product(name: "VectorIndex", package: "VectorIndex")
            ],
            path: "Sources/VectorIndexAcceleration",
            resources: [
                .process("Shaders")  // Index-specific Metal shaders (HNSW, IVF, Clustering)
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
            ],
            exclude: [
                "ShaderManagerTests.swift.disabled",
                "VectorCoreIntegrationEnhancedTests.swift.disabled"
            ]
        ),
        .testTarget(
            name: "VectorIndexAccelerationTests",
            dependencies: [
                "VectorIndexAcceleration",
                "VectorAccelerate",
                .product(name: "VectorCore", package: "VectorCore"),
                .product(name: "VectorIndex", package: "VectorIndex")
            ]
        )
    ]
)
