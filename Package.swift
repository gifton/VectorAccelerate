// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "VectorAccelerate",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1)
        // Note: No Linux support due to Metal dependency
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
        .package(url: "https://github.com/gifton/VectorCore", from: "0.1.4")
    ], 
    targets: [
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
        .executableTarget(
            name: "VectorAccelerateBenchmarks",
            dependencies: [
                "VectorAccelerate",
                .product(name: "VectorCore", package: "VectorCore")
            ]
        ),
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
        )
    ]
)
