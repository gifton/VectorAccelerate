//
//  PipelineHarvester.swift
//  VectorAccelerate
//
//  Pipeline harvesting for ahead-of-time (AOT) compilation
//

import Foundation
@preconcurrency import Metal
import VectorCore

// MARK: - Harvest Manifest

/// Manifest describing harvested pipeline data
public struct HarvestManifest: Codable, Sendable {
    /// Manifest version
    public let version: String

    /// VectorAccelerate version that created this harvest
    public let vectorAccelerateVersion: String

    /// Metal SDK version used for compilation
    public let metalSDKVersion: String

    /// Target GPU family
    public let targetGPUFamily: String

    /// Device name used for compilation
    public let deviceName: String

    /// When the harvest was created
    public let generatedAt: Date

    /// List of harvested pipelines
    public let pipelines: [HarvestedPipelineInfo]

    /// Total size of all harvest files
    public let totalSize: Int

    public init(
        version: String = "1.0.0",
        vectorAccelerateVersion: String,
        metalSDKVersion: String,
        targetGPUFamily: String,
        deviceName: String,
        pipelines: [HarvestedPipelineInfo],
        totalSize: Int = 0
    ) {
        self.version = version
        self.vectorAccelerateVersion = vectorAccelerateVersion
        self.metalSDKVersion = metalSDKVersion
        self.targetGPUFamily = targetGPUFamily
        self.deviceName = deviceName
        self.generatedAt = Date()
        self.pipelines = pipelines
        self.totalSize = totalSize
    }
}

/// Information about a single harvested pipeline
public struct HarvestedPipelineInfo: Codable, Sendable {
    /// Cache key for the pipeline
    public let key: PipelineCacheKey

    /// Filename of the harvest data
    public let fileName: String

    /// Size of the harvest data in bytes
    public let size: Int

    /// Checksum for validation
    public let checksum: String

    public init(key: PipelineCacheKey, fileName: String, size: Int, checksum: String = "") {
        self.key = key
        self.fileName = fileName
        self.size = size
        self.checksum = checksum
    }
}

// MARK: - Harvest Result

/// Result of a harvest operation
public struct HarvestResult: Sendable {
    public let manifest: HarvestManifest
    public let successCount: Int
    public let failureCount: Int
    public let totalTime: TimeInterval
    public let outputDirectory: URL

    public var wasSuccessful: Bool {
        failureCount == 0 && successCount > 0
    }
}

// MARK: - Pipeline Harvester

/// Harvests compiled pipelines for ahead-of-time (AOT) loading
///
/// Pipeline harvesting captures compiled GPU binaries for faster startup:
/// 1. Compile all needed pipelines during development/build
/// 2. Harvest binaries to disk
/// 3. Load harvested binaries at runtime (skip compilation)
///
/// Note: Actual Metal 4 harvesting APIs (MTL4PipelineHarvestDescriptor) are
/// simulated here for compatibility. The structure supports the full Metal 4
/// harvesting workflow.
///
/// Example:
/// ```swift
/// let harvester = try PipelineHarvester(device: device, outputDirectory: harvestDir)
/// let result = try await harvester.harvest(keys: PipelineCacheKey.commonKeys)
/// ```
public actor PipelineHarvester {
    // MARK: - Properties

    private let device: any MTLDevice
    private let compiler: Metal4ShaderCompiler
    private let outputDirectory: URL
    private let vectorAccelerateVersion: String

    // MARK: - Initialization

    /// Create a pipeline harvester
    ///
    /// - Parameters:
    ///   - device: Metal device for compilation
    ///   - compiler: Shader compiler
    ///   - outputDirectory: Directory to write harvest files
    ///   - vectorAccelerateVersion: Version string for manifest
    public init(
        device: any MTLDevice,
        compiler: Metal4ShaderCompiler,
        outputDirectory: URL,
        vectorAccelerateVersion: String = "0.4.0"
    ) throws {
        self.device = device
        self.compiler = compiler
        self.outputDirectory = outputDirectory
        self.vectorAccelerateVersion = vectorAccelerateVersion

        // Create output directory
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
    }

    // MARK: - Harvesting

    /// Harvest pipelines for the given cache keys
    ///
    /// - Parameter keys: Pipeline cache keys to harvest
    /// - Returns: Harvest result with manifest and statistics
    public func harvest(keys: [PipelineCacheKey]) async throws -> HarvestResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        var successCount = 0
        var failureCount = 0
        var pipelineInfos: [HarvestedPipelineInfo] = []
        var totalSize = 0

        // Compile all pipelines first
        for key in keys {
            do {
                let pipeline = try await compiler.compilePipeline(for: key)

                // Generate harvest file
                let fileName = "\(key.cacheString).harvest"
                let fileURL = outputDirectory.appendingPathComponent(fileName)

                // In Metal 4, we would use:
                // let harvestDescriptor = MTL4PipelineHarvestDescriptor()
                // harvestDescriptor.pipelines = [pipeline]
                // let harvestData = try compiler.harvest(descriptor: harvestDescriptor)
                // try harvestData.write(to: fileURL)

                // For now, create a placeholder that stores the cache key
                let harvestData = try createHarvestPlaceholder(key: key, pipeline: pipeline)
                try harvestData.write(to: fileURL)

                let size = harvestData.count
                totalSize += size

                let checksum = harvestData.sha256Hash

                pipelineInfos.append(HarvestedPipelineInfo(
                    key: key,
                    fileName: fileName,
                    size: size,
                    checksum: checksum
                ))

                successCount += 1
            } catch {
                failureCount += 1
                // Continue harvesting other pipelines
            }
        }

        // Create manifest
        let manifest = HarvestManifest(
            vectorAccelerateVersion: vectorAccelerateVersion,
            metalSDKVersion: getMetalSDKVersion(),
            targetGPUFamily: getGPUFamily(),
            deviceName: device.name,
            pipelines: pipelineInfos,
            totalSize: totalSize
        )

        // Save manifest
        let manifestURL = outputDirectory.appendingPathComponent("manifest.json")
        let manifestData = try JSONEncoder().encode(manifest)
        try manifestData.write(to: manifestURL)

        let totalTime = CFAbsoluteTimeGetCurrent() - startTime

        return HarvestResult(
            manifest: manifest,
            successCount: successCount,
            failureCount: failureCount,
            totalTime: totalTime,
            outputDirectory: outputDirectory
        )
    }

    /// Harvest standard pipelines
    public func harvestStandard() async throws -> HarvestResult {
        try await harvest(keys: PipelineCacheKey.commonKeys)
    }

    /// Harvest embedding model pipelines
    public func harvestEmbeddings() async throws -> HarvestResult {
        try await harvest(keys: PipelineCacheKey.embeddingModelKeys)
    }

    // MARK: - Loading

    /// Load harvested pipelines into cache
    ///
    /// - Parameter cache: Pipeline cache to populate
    /// - Returns: Number of pipelines loaded
    public func loadHarvested(into cache: PipelineCache) async throws -> Int {
        let manifestURL = outputDirectory.appendingPathComponent("manifest.json")

        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw VectorError.harvestManifestNotFound()
        }

        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(HarvestManifest.self, from: manifestData)

        // Validate manifest compatibility
        guard isManifestCompatible(manifest) else {
            throw VectorError.harvestIncompatible(reason: "GPU family or SDK version mismatch")
        }

        var loadedCount = 0

        for info in manifest.pipelines {
            do {
                // In Metal 4, we would load from harvest data:
                // let harvestURL = outputDirectory.appendingPathComponent(info.fileName)
                // let harvestData = try Data(contentsOf: harvestURL)
                // let pipelines = try compiler.loadHarvested(data: harvestData)
                // cache.preCache(key: info.key, pipeline: pipelines[0])

                // For now, compile the pipeline (fallback)
                let pipeline = try await compiler.compilePipeline(for: info.key)
                await cache.preCache(key: info.key, pipeline: pipeline)
                loadedCount += 1
            } catch {
                // Continue loading other pipelines
            }
        }

        return loadedCount
    }

    // MARK: - Validation

    /// Check if a harvest manifest is compatible with current environment
    public func isManifestCompatible(_ manifest: HarvestManifest) -> Bool {
        // Check Metal SDK version
        let currentSDK = getMetalSDKVersion()
        guard manifest.metalSDKVersion == currentSDK else {
            return false
        }

        // Check GPU family compatibility
        let currentFamily = getGPUFamily()
        guard manifest.targetGPUFamily == currentFamily else {
            return false
        }

        return true
    }

    /// Get the harvest manifest if it exists
    public func getManifest() throws -> HarvestManifest? {
        let manifestURL = outputDirectory.appendingPathComponent("manifest.json")

        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return nil
        }

        let manifestData = try Data(contentsOf: manifestURL)
        return try JSONDecoder().decode(HarvestManifest.self, from: manifestData)
    }

    // MARK: - Cleanup

    /// Remove all harvested files
    public func clearHarvest() throws {
        let contents = try FileManager.default.contentsOfDirectory(
            at: outputDirectory,
            includingPropertiesForKeys: nil
        )

        for url in contents {
            try FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Private Helpers

    /// Create harvest placeholder data (simulates Metal 4 harvest binary)
    private func createHarvestPlaceholder(key: PipelineCacheKey, pipeline: any MTLComputePipelineState) throws -> Data {
        // In actual Metal 4, this would be binary pipeline data
        // For now, store the key info for validation
        struct HarvestPlaceholder: Codable {
            let key: PipelineCacheKey
            let deviceName: String
            let timestamp: Date
        }

        let placeholder = HarvestPlaceholder(
            key: key,
            deviceName: device.name,
            timestamp: Date()
        )

        return try JSONEncoder().encode(placeholder)
    }

    /// Get Metal SDK version string
    private func getMetalSDKVersion() -> String {
        // In practice, this would be determined at compile time or from Metal API
        "4.0"
    }

    /// Get GPU family string
    private func getGPUFamily() -> String {
        // Detect GPU family
        if device.supportsFamily(.apple9) {
            return "apple9"
        } else if device.supportsFamily(.apple8) {
            return "apple8"
        } else if device.supportsFamily(.apple7) {
            return "apple7"
        } else if device.supportsFamily(.apple6) {
            return "apple6"
        }
        #if os(macOS)
        if device.supportsFamily(.mac2) {
            return "mac2"
        }
        #endif
        return "unknown"
    }
}

// MARK: - Data Extension for Checksum

private extension Data {
    /// Simple hash for checksum (placeholder for actual SHA256)
    var sha256Hash: String {
        // In production, use CryptoKit or CommonCrypto for SHA256
        var hash: UInt64 = 0
        for byte in self {
            hash = hash &* 31 &+ UInt64(byte)
        }
        return String(format: "%016llx", hash)
    }
}

// MARK: - Harvester Factory

public enum PipelineHarvesterFactory {
    /// Create harvester with default output directory
    public static func create(
        device: any MTLDevice,
        compiler: Metal4ShaderCompiler
    ) throws -> PipelineHarvester {
        let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let harvestDir = cachesDir.appendingPathComponent("VectorAccelerate/HarvestedPipelines", isDirectory: true)

        return try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: harvestDir
        )
    }

    /// Create harvester with custom output directory
    public static func create(
        device: any MTLDevice,
        compiler: Metal4ShaderCompiler,
        outputDirectory: URL
    ) throws -> PipelineHarvester {
        try PipelineHarvester(
            device: device,
            compiler: compiler,
            outputDirectory: outputDirectory
        )
    }
}

// MARK: - Additional VectorError Extensions

public extension VectorError {
    /// Harvest manifest not found
    static func harvestManifestNotFound(
        file: StaticString = #file,
        function: StaticString = #function,
        line: UInt = #line
    ) -> VectorError {
        VectorError.invalidOperation("Harvest manifest not found at expected location")
    }

    /// Harvest is incompatible with current environment
    static func harvestIncompatible(
        reason: String,
        file: StaticString = #file,
        function: StaticString = #function,
        line: UInt = #line
    ) -> VectorError {
        VectorError.invalidOperation("Harvest is incompatible: \(reason)")
    }
}
