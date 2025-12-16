//
//  BinaryArchiveManager.swift
//  VectorAccelerate
//
//  Binary archive management for pipeline caching across app launches
//

import Foundation
@preconcurrency import Metal

/// State of the binary archive.
public enum BinaryArchiveState: Sendable, Equatable {
    /// Archive not loaded
    case unloaded

    /// Archive is being loaded or created
    case loading

    /// Archive is ready with the specified number of pipelines
    case ready(pipelineCount: Int)

    /// Archive operation failed
    case failed(BinaryArchiveError)

    public static func == (lhs: BinaryArchiveState, rhs: BinaryArchiveState) -> Bool {
        switch (lhs, rhs) {
        case (.unloaded, .unloaded):
            return true
        case (.loading, .loading):
            return true
        case (.ready(let lhsCount), .ready(let rhsCount)):
            return lhsCount == rhsCount
        case (.failed(let lhsError), .failed(let rhsError)):
            return lhsError.localizedDescription == rhsError.localizedDescription
        default:
            return false
        }
    }
}

/// Errors from binary archive operations.
public enum BinaryArchiveError: Error, LocalizedError, Sendable {
    /// Failed to create archive
    case creationFailed(String)

    /// Failed to load archive from disk
    case loadFailed(String)

    /// Failed to save archive to disk
    case saveFailed(String)

    /// Archive is corrupted
    case corrupted(String)

    /// Failed to add pipeline to archive
    case addPipelineFailed(String)

    /// Archive not in ready state
    case notReady

    /// Archive URL not configured
    case urlNotConfigured

    public var errorDescription: String? {
        switch self {
        case .creationFailed(let reason):
            return "Failed to create binary archive: \(reason)"
        case .loadFailed(let reason):
            return "Failed to load binary archive: \(reason)"
        case .saveFailed(let reason):
            return "Failed to save binary archive: \(reason)"
        case .corrupted(let reason):
            return "Binary archive is corrupted: \(reason)"
        case .addPipelineFailed(let reason):
            return "Failed to add pipeline to archive: \(reason)"
        case .notReady:
            return "Binary archive not in ready state"
        case .urlNotConfigured:
            return "Binary archive URL not configured"
        }
    }
}

/// Manifest tracking which pipeline keys are stored in the archive.
///
/// MTLBinaryArchive doesn't support enumeration, so we maintain a separate
/// manifest file alongside the archive to track stored pipelines.
public struct BinaryArchiveManifest: Codable, Sendable {
    /// Manifest version for compatibility checks
    public let version: String

    /// Hash of shader source for invalidation
    public let shaderSourceHash: String

    /// Keys stored in the archive
    public var keys: Set<PipelineCacheKey>

    /// When the manifest was created
    public let createdAt: Date

    /// When the manifest was last modified
    public var modifiedAt: Date

    /// Device name this archive was created for
    public let deviceName: String

    public init(
        shaderSourceHash: String,
        deviceName: String
    ) {
        self.version = "1.0.0"
        self.shaderSourceHash = shaderSourceHash
        self.keys = []
        self.createdAt = Date()
        self.modifiedAt = Date()
        self.deviceName = deviceName
    }

    /// Add a key to the manifest
    public mutating func addKey(_ key: PipelineCacheKey) {
        keys.insert(key)
        modifiedAt = Date()
    }

    /// Check if a key exists in the manifest
    public func containsKey(_ key: PipelineCacheKey) -> Bool {
        keys.contains(key)
    }
}

/// Manages MTLBinaryArchive for pipeline caching across app launches.
///
/// `BinaryArchiveManager` provides persistent storage for compiled pipeline state objects,
/// enabling near-instant PSO creation on subsequent launches. Key behaviors:
///
/// - **First launch:** Pipelines are compiled and saved to the archive
/// - **Subsequent launches:** Pipelines load from archive in ~1ms instead of ~50-200ms
/// - **Graceful degradation:** Corrupted archives are deleted and recreated
/// - **Manifest tracking:** Maintains a manifest of stored pipeline keys
///
/// ## Usage
///
/// ```swift
/// let manager = BinaryArchiveManager(device: device, archiveURL: archiveURL)
/// try await manager.loadOrCreate()
///
/// // Check if pipeline is in archive
/// if manager.containsPipeline(for: key) {
///     // Use archive in pipeline descriptor
/// }
///
/// // Add new pipeline
/// try await manager.addPipeline(pipeline, for: key)
///
/// // Save to disk
/// try await manager.save()
/// ```
public actor BinaryArchiveManager {

    // MARK: - Properties

    /// Metal device for archive creation
    private let device: any MTLDevice

    /// URL for archive storage (nil means use default location)
    public let archiveURL: URL?

    /// Current archive state
    public private(set) var state: BinaryArchiveState = .unloaded

    /// The underlying Metal binary archive
    private var archive: (any MTLBinaryArchive)?

    /// Manifest tracking stored pipeline keys
    private var manifest: BinaryArchiveManifest?

    /// Whether the archive has unsaved changes
    private var isDirty: Bool = false

    /// Hash of current shader source for invalidation
    private let shaderSourceHash: String

    // MARK: - Initialization

    /// Create a binary archive manager.
    ///
    /// - Parameters:
    ///   - device: Metal device for archive operations
    ///   - archiveURL: Optional URL for archive storage. If nil, uses default location:
    ///     `~/Library/Caches/VectorAccelerate/pipelines.metalarchive`
    ///   - shaderSourceHash: Hash of shader source for invalidation (default: fixed value)
    public init(
        device: any MTLDevice,
        archiveURL: URL? = nil,
        shaderSourceHash: String = "1.0.0"  // Should be actual hash in production
    ) {
        self.device = device
        self.archiveURL = archiveURL ?? Self.defaultArchiveURL
        self.shaderSourceHash = shaderSourceHash
    }

    /// Default archive URL in caches directory
    public static var defaultArchiveURL: URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir
            .appendingPathComponent("VectorAccelerate", isDirectory: true)
            .appendingPathComponent("pipelines.metalarchive")
    }

    /// Default manifest URL (alongside archive)
    private var manifestURL: URL? {
        archiveURL?.deletingPathExtension().appendingPathExtension("manifest.json")
    }

    // MARK: - Archive Lifecycle

    /// Load an existing archive or create a new one.
    ///
    /// This method:
    /// 1. Attempts to load existing archive and manifest
    /// 2. Validates the manifest against current shader source hash
    /// 3. If validation fails or no archive exists, creates a new empty archive
    ///
    /// - Throws: `BinaryArchiveError` if creation fails
    public func loadOrCreate() async throws {
        state = .loading

        guard let url = archiveURL else {
            throw BinaryArchiveError.urlNotConfigured
        }

        // Ensure parent directory exists
        try createArchiveDirectory()

        // Try to load existing archive
        if FileManager.default.fileExists(atPath: url.path) {
            do {
                try await loadExistingArchive(url: url)
                return
            } catch {
                // Archive corrupted or incompatible - delete and recreate
                try? deleteArchive()
            }
        }

        // Create new archive
        try await createNewArchive()
    }

    /// Save the archive to disk.
    ///
    /// Only performs I/O if the archive has unsaved changes.
    /// Note: Empty archives cannot be serialized - this is a Metal limitation.
    /// The method saves successfully if there are no pipelines (only manifest is saved).
    ///
    /// - Throws: `BinaryArchiveError.saveFailed` if serialization fails
    public func save() async throws {
        guard isDirty else { return }
        guard let archive = archive, let url = archiveURL else {
            throw BinaryArchiveError.notReady
        }

        // Save manifest (always succeeds)
        try saveManifest()

        // Only serialize archive if it contains pipelines
        // Empty archives throw "Nothing to serialize" error from Metal
        guard let manifest = manifest, !manifest.keys.isEmpty else {
            isDirty = false
            return
        }

        do {
            try archive.serialize(to: url)
            isDirty = false
        } catch {
            throw BinaryArchiveError.saveFailed(error.localizedDescription)
        }
    }

    // MARK: - Pipeline Operations

    /// Add a compiled pipeline to the archive.
    ///
    /// The pipeline is added to both the in-memory archive and the manifest.
    /// Call `save()` to persist changes to disk.
    ///
    /// - Parameters:
    ///   - descriptor: The compute pipeline descriptor used to create the pipeline
    ///   - key: Cache key for this pipeline
    /// - Throws: `BinaryArchiveError.addPipelineFailed` if addition fails
    public func addPipeline(
        descriptor: MTLComputePipelineDescriptor,
        for key: PipelineCacheKey
    ) async throws {
        guard let archive = archive else {
            throw BinaryArchiveError.notReady
        }

        // Skip if already in archive
        if manifest?.containsKey(key) == true {
            return
        }

        do {
            try archive.addComputePipelineFunctions(descriptor: descriptor)
            manifest?.addKey(key)
            isDirty = true
            updateStateCount()
        } catch {
            throw BinaryArchiveError.addPipelineFailed(error.localizedDescription)
        }
    }

    /// Check if a pipeline is stored in the archive.
    ///
    /// - Parameter key: The pipeline cache key to check
    /// - Returns: true if the pipeline is in the archive
    public func containsPipeline(for key: PipelineCacheKey) -> Bool {
        manifest?.containsKey(key) ?? false
    }

    /// Get the underlying MTLBinaryArchive for use in pipeline descriptors.
    ///
    /// - Returns: The archive, or nil if not ready
    public var underlyingArchive: (any MTLBinaryArchive)? {
        archive
    }

    /// Number of pipelines in the archive.
    public var pipelineCount: Int {
        manifest?.keys.count ?? 0
    }

    /// Archive file size in bytes (nil if not saved or file doesn't exist).
    public var serializedSize: Int? {
        guard let url = archiveURL,
              FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }

        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            return attributes[.size] as? Int
        } catch {
            return nil
        }
    }

    /// All pipeline keys stored in the archive.
    public var storedKeys: [PipelineCacheKey] {
        guard let manifest = manifest else { return [] }
        return Array(manifest.keys)
    }

    // MARK: - Private Implementation

    /// Create parent directory for archive
    private func createArchiveDirectory() throws {
        guard let url = archiveURL else { return }

        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
    }

    /// Load existing archive from disk
    private func loadExistingArchive(url: URL) async throws {
        // Load manifest first
        guard let manifestURL = manifestURL,
              FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw BinaryArchiveError.corrupted("Manifest missing")
        }

        let manifestData = try Data(contentsOf: manifestURL)
        let loadedManifest = try JSONDecoder().decode(BinaryArchiveManifest.self, from: manifestData)

        // Validate shader hash
        if loadedManifest.shaderSourceHash != shaderSourceHash {
            throw BinaryArchiveError.corrupted("Shader source changed")
        }

        // Validate device
        if loadedManifest.deviceName != device.name {
            throw BinaryArchiveError.corrupted("Different device")
        }

        // Load archive
        let descriptor = MTLBinaryArchiveDescriptor()
        descriptor.url = url

        do {
            let loadedArchive = try device.makeBinaryArchive(descriptor: descriptor)
            self.archive = loadedArchive
            self.manifest = loadedManifest
            self.state = .ready(pipelineCount: loadedManifest.keys.count)
        } catch {
            throw BinaryArchiveError.loadFailed(error.localizedDescription)
        }
    }

    /// Create a new empty archive
    private func createNewArchive() async throws {
        let descriptor = MTLBinaryArchiveDescriptor()
        // url = nil for new empty archive

        do {
            let newArchive = try device.makeBinaryArchive(descriptor: descriptor)
            self.archive = newArchive
            self.manifest = BinaryArchiveManifest(
                shaderSourceHash: shaderSourceHash,
                deviceName: device.name
            )
            self.isDirty = true  // Needs to be saved
            self.state = .ready(pipelineCount: 0)
        } catch {
            state = .failed(.creationFailed(error.localizedDescription))
            throw BinaryArchiveError.creationFailed(error.localizedDescription)
        }
    }

    /// Delete archive and manifest files
    private func deleteArchive() throws {
        if let url = archiveURL {
            try? FileManager.default.removeItem(at: url)
        }
        if let manifestURL = manifestURL {
            try? FileManager.default.removeItem(at: manifestURL)
        }
    }

    /// Save manifest to disk
    private func saveManifest() throws {
        guard let manifest = manifest, let manifestURL = manifestURL else { return }

        let data = try JSONEncoder().encode(manifest)
        try data.write(to: manifestURL)
    }

    /// Update state with current pipeline count
    private func updateStateCount() {
        if let count = manifest?.keys.count {
            state = .ready(pipelineCount: count)
        }
    }
}
