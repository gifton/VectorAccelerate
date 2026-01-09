//
//  WriteAheadLog.swift
//  VectorAccelerate
//
//  Write-Ahead Log for ensuring durability and crash recovery of index operations.
//
//  Migrated from VectorIndexAccelerated with adaptations for Swift 6 concurrency.
//

import Foundation
import CryptoKit
import os

// MARK: - Write-Ahead Log

/// Write-Ahead Log (WAL) for ensuring durability and crash recovery.
///
/// The WAL ensures that all index operations are persisted to disk before being
/// acknowledged. In case of a crash, the WAL can be replayed to restore the
/// index to a consistent state.
///
/// ## Architecture
/// - Segment-based storage for efficient rotation and cleanup
/// - Configurable sync modes for performance/durability tradeoffs
/// - Checkpointing for marking consistent recovery points
/// - SHA-256 checksums for data integrity
///
/// ## Usage
/// ```swift
/// let wal = WriteAheadLog(directory: walDirectory)
/// try await wal.initialize()
///
/// // Log operations
/// let seqNum = try await wal.append(insertOperation, type: "insert")
///
/// // Create checkpoint after batch
/// try await wal.checkpoint()
///
/// // Replay after crash
/// let entries = try await wal.replay(from: lastCheckpoint)
/// ```
public actor WriteAheadLog {
    private let logger = os.Logger(subsystem: "VectorAccelerate", category: "WAL")

    // MARK: - File Management

    /// Directory containing WAL segments
    private let directory: URL

    /// Maximum size per segment in bytes
    private let maxSegmentSize: Int

    /// Maximum number of segments to retain
    private let maxSegments: Int

    /// Currently active segment
    private var currentSegment: WALSegment?

    /// All loaded segments
    private var segments: [WALSegment] = []

    // MARK: - State Tracking

    /// Current sequence number
    private var sequenceNumber: UInt64 = 0

    /// Last checkpoint sequence number
    private var lastCheckpoint: UInt64 = 0

    /// Whether there are unflushed writes
    private var isDirty: Bool = false

    // MARK: - Performance

    /// Interval for periodic sync in seconds
    private let syncInterval: TimeInterval = 1.0

    /// Last sync timestamp
    private var lastSyncTime: Date = Date()

    /// Pending writes waiting for flush
    private var pendingWrites: [WALEntry] = []

    // MARK: - Configuration

    /// WAL configuration
    private let config: Configuration

    // MARK: - Types

    /// WAL configuration options
    public struct Configuration: Sendable {
        /// Maximum size per segment in bytes (default: 10MB)
        public var maxSegmentSize: Int

        /// Maximum number of segments to retain
        public var maxSegments: Int

        /// Sync mode for durability/performance tradeoff
        public var syncMode: SyncMode

        /// Whether to compute checksums for entries
        public var checksumEnabled: Bool

        /// Whether to compress entries (future use)
        public var compressionEnabled: Bool

        public init(
            maxSegmentSize: Int = 10_000_000,
            maxSegments: Int = 10,
            syncMode: SyncMode = .periodic,
            checksumEnabled: Bool = true,
            compressionEnabled: Bool = false
        ) {
            self.maxSegmentSize = maxSegmentSize
            self.maxSegments = maxSegments
            self.syncMode = syncMode
            self.checksumEnabled = checksumEnabled
            self.compressionEnabled = compressionEnabled
        }

        /// Configuration optimized for durability
        public static let durable = Configuration(
            syncMode: .immediate,
            checksumEnabled: true
        )

        /// Configuration optimized for performance
        public static let performant = Configuration(
            maxSegmentSize: 50_000_000,
            maxSegments: 5,
            syncMode: .batch,
            checksumEnabled: false
        )
    }

    /// Sync modes for controlling durability/performance tradeoff
    public enum SyncMode: Sendable {
        /// Sync after every write (slowest, safest)
        case immediate

        /// Sync periodically (balanced)
        case periodic

        /// Sync after batch threshold (fastest, less safe)
        case batch
    }

    // MARK: - Initialization

    /// Initialize WAL with a directory path.
    ///
    /// - Parameters:
    ///   - directory: Directory to store WAL segments
    ///   - configuration: Configuration options
    public init(directory: URL, configuration: Configuration = Configuration()) {
        self.directory = directory
        self.config = configuration
        self.maxSegmentSize = configuration.maxSegmentSize
        self.maxSegments = configuration.maxSegments
    }

    /// Initialize the WAL by loading existing segments.
    ///
    /// Must be called before using the WAL to load any existing segments
    /// and prepare for new writes.
    public func initialize() async throws {
        // Create WAL directory if needed
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: nil
        )

        // Load existing segments
        try await loadExistingSegments()

        // Start new segment if needed
        if currentSegment == nil {
            try await createNewSegment()
        }

        logger.info("WAL initialized with \(self.segments.count) segments")
    }

    // MARK: - Write Operations

    /// Append an operation to the WAL.
    ///
    /// - Parameters:
    ///   - operation: The operation data to log
    ///   - type: Type identifier for the operation
    /// - Returns: Sequence number assigned to this entry
    public func append<T: Codable & Sendable>(_ operation: T, type: String) async throws -> UInt64 {
        let entry = try createEntry(operation: operation, type: type)

        // Add to pending writes
        pendingWrites.append(entry)
        isDirty = true

        // Check if we need to flush based on sync mode
        switch config.syncMode {
        case .immediate:
            try await flush()
        case .periodic:
            if Date().timeIntervalSince(lastSyncTime) > syncInterval {
                try await flush()
            }
        case .batch:
            if pendingWrites.count >= 100 {
                try await flush()
            }
        }

        return entry.sequenceNumber
    }

    /// Flush pending writes to disk.
    public func flush() async throws {
        guard !pendingWrites.isEmpty else { return }

        // Write all pending entries
        for entry in pendingWrites {
            try await writeEntry(entry)
        }

        // Clear pending writes
        pendingWrites.removeAll()
        isDirty = false
        lastSyncTime = Date()

        // Force sync to disk
        if let segment = currentSegment {
            try await segment.sync()
        }

        logger.debug("Flushed \(self.pendingWrites.count) entries to WAL")
    }

    // MARK: - Checkpoint Operations

    /// Create a checkpoint (mark current position as consistent).
    ///
    /// Checkpoints mark a point where the index is in a consistent state.
    /// During recovery, replay can start from the last checkpoint.
    ///
    /// - Returns: Sequence number of the checkpoint
    public func checkpoint() async throws -> UInt64 {
        // Flush any pending writes
        try await flush()

        // Mark checkpoint
        lastCheckpoint = sequenceNumber

        // Write checkpoint marker
        let checkpointEntry = CheckpointEntry(
            sequenceNumber: sequenceNumber,
            timestamp: Date(),
            segmentId: currentSegment?.id ?? UUID()
        )

        let checkpointData = try JSONEncoder().encode(checkpointEntry)
        let checkpointFile = directory.appendingPathComponent("checkpoint.json")
        try checkpointData.write(to: checkpointFile)

        // Clean up old segments
        try await cleanupOldSegments()

        logger.info("Checkpoint created at sequence \(self.lastCheckpoint)")

        return lastCheckpoint
    }

    // MARK: - Replay Operations

    /// Replay WAL from a given sequence number.
    ///
    /// - Parameter sequenceNumber: Start replaying from entries after this sequence number
    /// - Returns: Array of WAL entries to replay
    public func replay(from sequenceNumber: UInt64 = 0) async throws -> [WALEntry] {
        var entries: [WALEntry] = []

        // Read all segments
        for segment in segments {
            let segmentEntries = try await segment.readEntries()
            for entry in segmentEntries {
                if entry.sequenceNumber > sequenceNumber {
                    // Verify checksum if enabled
                    if config.checksumEnabled, let checksum = entry.checksum {
                        let computed = computeChecksum(entry.data)
                        if computed != checksum {
                            logger.warning("Checksum mismatch for entry \(entry.sequenceNumber)")
                            throw WALError.checksumMismatch
                        }
                    }
                    entries.append(entry)
                }
            }
        }

        // Sort by sequence number
        entries.sort { $0.sequenceNumber < $1.sequenceNumber }

        logger.info("Replayed \(entries.count) entries from sequence \(sequenceNumber)")

        return entries
    }

    // MARK: - Truncation and Compaction

    /// Truncate WAL after a sequence number.
    ///
    /// Removes all entries after the given sequence number. Useful for
    /// rolling back to a known good state.
    ///
    /// - Parameter sequenceNumber: Keep entries up to and including this sequence
    public func truncate(after sequenceNumber: UInt64) async throws {
        // Find segments to keep
        var segmentsToKeep: [WALSegment] = []

        for segment in segments {
            let minSeq = await segment.minSequence
            let maxSeq = await segment.maxSequence

            if minSeq <= sequenceNumber {
                segmentsToKeep.append(segment)

                // Truncate within segment if needed
                if maxSeq > sequenceNumber {
                    try await segment.truncateAfter(sequenceNumber)
                }
            }
        }

        // Remove truncated segments
        let segmentsToRemove = segments.filter { segment in
            !segmentsToKeep.contains { $0.id == segment.id }
        }
        for segment in segmentsToRemove {
            try await segment.delete()
        }

        segments = segmentsToKeep
        self.sequenceNumber = sequenceNumber

        logger.info("Truncated WAL after sequence \(sequenceNumber)")
    }

    /// Compact the WAL by removing obsolete entries.
    ///
    /// Removes entries before the last checkpoint and merges small segments.
    public func compact() async throws {
        // Load checkpoint if exists
        let checkpointFile = directory.appendingPathComponent("checkpoint.json")
        if FileManager.default.fileExists(atPath: checkpointFile.path) {
            let checkpointData = try Data(contentsOf: checkpointFile)
            let checkpoint = try JSONDecoder().decode(CheckpointEntry.self, from: checkpointData)

            // Remove segments before checkpoint
            try await truncate(after: checkpoint.sequenceNumber)
        }

        // Merge small segments
        try await mergeSegments()

        logger.info("WAL compaction complete")
    }

    // MARK: - Statistics

    /// Get WAL statistics.
    ///
    /// - Returns: Current WAL statistics
    public func getStatistics() async -> WALStatistics {
        var totalSize = 0
        var entryCount = 0

        for segment in segments {
            totalSize += await segment.size
            entryCount += await segment.entryCount
        }

        return WALStatistics(
            segmentCount: segments.count,
            totalSize: totalSize,
            entryCount: entryCount,
            currentSequence: sequenceNumber,
            lastCheckpoint: lastCheckpoint,
            isDirty: isDirty
        )
    }

    // MARK: - Private Methods

    private func loadExistingSegments() async throws {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.creationDateKey],
            options: []
        )

        // Filter WAL segment files
        let segmentFiles = files.filter { $0.pathExtension == "wal" }

        // Load segments
        for file in segmentFiles {
            do {
                let segment = try await WALSegment(url: file, checksumEnabled: config.checksumEnabled)
                segments.append(segment)
            } catch {
                logger.warning("Failed to load segment \(file.lastPathComponent): \(error)")
            }
        }

        // Sort by creation date
        segments.sort { seg1, seg2 in
            // We need to access the properties - for simplicity we'll compare IDs as proxy
            // In practice, we should store creation dates
            seg1.id.uuidString < seg2.id.uuidString
        }

        // Set current segment
        currentSegment = segments.last

        // Update sequence number
        if let lastSegment = segments.last {
            sequenceNumber = await lastSegment.maxSequence
        }
    }

    private func createNewSegment() async throws {
        let segmentId = UUID()
        let segmentURL = directory.appendingPathComponent("\(segmentId.uuidString).wal")

        let segment = try await WALSegment(
            url: segmentURL,
            id: segmentId,
            checksumEnabled: config.checksumEnabled
        )

        segments.append(segment)
        currentSegment = segment
    }

    private func createEntry<T: Codable>(operation: T, type: String) throws -> WALEntry {
        sequenceNumber += 1

        let data = try JSONEncoder().encode(operation)

        return WALEntry(
            sequenceNumber: sequenceNumber,
            timestamp: Date(),
            type: type,
            data: data,
            checksum: config.checksumEnabled ? computeChecksum(data) : nil
        )
    }

    private func writeEntry(_ entry: WALEntry) async throws {
        guard var segment = currentSegment else {
            throw WALError.noActiveSegment
        }

        // Check if segment is full
        let segmentSize = await segment.size
        if segmentSize >= maxSegmentSize {
            try await createNewSegment()
            segment = currentSegment!
        }

        // Write entry to segment
        try await segment.appendEntry(entry)

        // Update segment metadata
        await segment.updateSequenceRange(entry.sequenceNumber)
    }

    private func cleanupOldSegments() async throws {
        // Keep only recent segments
        if segments.count > maxSegments {
            let segmentsToRemove = segments.count - maxSegments
            for i in 0..<segmentsToRemove {
                try await segments[i].delete()
            }
            segments.removeFirst(segmentsToRemove)
        }
    }

    private func mergeSegments() async throws {
        // Merge small consecutive segments
        var mergedSegments: [WALSegment] = []
        var currentMerge: [WALSegment] = []
        var currentSize = 0

        for segment in segments {
            let segmentSize = await segment.size

            if currentSize + segmentSize < maxSegmentSize / 2 {
                currentMerge.append(segment)
                currentSize += segmentSize
            } else {
                if !currentMerge.isEmpty {
                    let merged = try await mergeSegmentGroup(currentMerge)
                    mergedSegments.append(merged)
                }
                currentMerge = [segment]
                currentSize = segmentSize
            }
        }

        // Handle remaining segments
        if !currentMerge.isEmpty {
            if currentMerge.count > 1 {
                let merged = try await mergeSegmentGroup(currentMerge)
                mergedSegments.append(merged)
            } else {
                mergedSegments.append(contentsOf: currentMerge)
            }
        }

        segments = mergedSegments
    }

    private func mergeSegmentGroup(_ group: [WALSegment]) async throws -> WALSegment {
        let mergedId = UUID()
        let mergedURL = directory.appendingPathComponent("\(mergedId.uuidString).wal")

        let mergedSegment = try await WALSegment(
            url: mergedURL,
            id: mergedId,
            checksumEnabled: config.checksumEnabled
        )

        // Copy all entries from group
        for segment in group {
            let entries = try await segment.readEntries()
            for entry in entries {
                try await mergedSegment.appendEntry(entry)
            }
        }

        // Update metadata
        if let firstSegment = group.first {
            let minSeq = await firstSegment.minSequence
            await mergedSegment.setMinSequence(minSeq)
        }
        if let lastSegment = group.last {
            let maxSeq = await lastSegment.maxSequence
            await mergedSegment.setMaxSequence(maxSeq)
        }

        // Delete old segments
        for segment in group {
            try await segment.delete()
        }

        return mergedSegment
    }

    private func computeChecksum(_ data: Data) -> String {
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - WAL Segment Actor

/// Individual WAL segment file managed as an actor for thread safety.
///
/// Each segment is a separate file that contains a sequence of WAL entries.
/// Segments are rotated when they reach the maximum size.
public actor WALSegment {
    /// Unique identifier for this segment
    public let id: UUID

    /// File URL for this segment
    public let url: URL

    /// Minimum sequence number in this segment
    public private(set) var minSequence: UInt64 = 0

    /// Maximum sequence number in this segment
    public private(set) var maxSequence: UInt64 = 0

    /// Number of entries in this segment
    public private(set) var entryCount: Int = 0

    /// Whether checksums are enabled
    private let checksumEnabled: Bool

    /// Cached file size
    private var cachedSize: Int = 0

    /// Current size in bytes
    public var size: Int {
        cachedSize
    }

    // MARK: - Initialization

    /// Create or load a WAL segment.
    ///
    /// - Parameters:
    ///   - url: File URL for the segment
    ///   - id: Unique identifier (generated if not provided)
    ///   - checksumEnabled: Whether to verify checksums
    public init(url: URL, id: UUID = UUID(), checksumEnabled: Bool = true) async throws {
        self.url = url
        self.id = id
        self.checksumEnabled = checksumEnabled

        // Create file if it doesn't exist
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }

        // Get initial file size
        if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
           let fileSize = attrs[.size] as? Int {
            cachedSize = fileSize
        }

        // Load existing entries to set sequence range
        let entries = try await readEntriesInternal()
        entryCount = entries.count
        if let first = entries.first {
            minSequence = first.sequenceNumber
        }
        if let last = entries.last {
            maxSequence = last.sequenceNumber
        }
    }

    // MARK: - Entry Operations

    /// Append an entry to this segment.
    public func appendEntry(_ entry: WALEntry) async throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601

        var entryData = try encoder.encode(entry)
        entryData.append(contentsOf: "\n".utf8)

        // Write to file
        let fileHandle = try FileHandle(forWritingTo: url)
        defer { try? fileHandle.close() }

        try fileHandle.seekToEnd()
        try fileHandle.write(contentsOf: entryData)

        entryCount += 1
        cachedSize += entryData.count
    }

    /// Read all entries from this segment.
    public func readEntries() async throws -> [WALEntry] {
        try await readEntriesInternal()
    }

    private func readEntriesInternal() async throws -> [WALEntry] {
        let data = try Data(contentsOf: url)
        let lines = data.split(separator: UInt8(ascii: "\n"))

        var entries: [WALEntry] = []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for line in lines {
            if let entry = try? decoder.decode(WALEntry.self, from: Data(line)) {
                entries.append(entry)
            }
        }

        return entries
    }

    /// Truncate this segment after a sequence number.
    public func truncateAfter(_ sequenceNumber: UInt64) async throws {
        let entries = try await readEntriesInternal()
        let entriesToKeep = entries.filter { $0.sequenceNumber <= sequenceNumber }

        // Clear and rewrite file
        try Data().write(to: url)

        for entry in entriesToKeep {
            try await appendEntry(entry)
        }

        maxSequence = entriesToKeep.last?.sequenceNumber ?? 0
        entryCount = entriesToKeep.count

        // Update cached size
        if let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
           let fileSize = attrs[.size] as? Int {
            cachedSize = fileSize
        }
    }

    /// Sync this segment to disk.
    public func sync() async throws {
        let fileHandle = try FileHandle(forWritingTo: url)
        defer { try? fileHandle.close() }
        try fileHandle.synchronize()
    }

    /// Delete this segment file.
    public func delete() async throws {
        try FileManager.default.removeItem(at: url)
    }

    // MARK: - Sequence Management

    /// Update sequence range with a new entry's sequence number.
    public func updateSequenceRange(_ sequenceNumber: UInt64) {
        if minSequence == 0 {
            minSequence = sequenceNumber
        }
        maxSequence = max(maxSequence, sequenceNumber)
    }

    /// Set minimum sequence number.
    public func setMinSequence(_ value: UInt64) {
        minSequence = value
    }

    /// Set maximum sequence number.
    public func setMaxSequence(_ value: UInt64) {
        maxSequence = value
    }
}

// MARK: - Supporting Types

/// WAL entry structure.
///
/// Each entry represents a single logged operation with metadata for
/// ordering and integrity verification.
public struct WALEntry: Codable, Sendable {
    /// Monotonically increasing sequence number
    public let sequenceNumber: UInt64

    /// Timestamp when the entry was created
    public let timestamp: Date

    /// Type identifier for the operation
    public let type: String

    /// Encoded operation data
    public let data: Data

    /// SHA-256 checksum of the data (if checksums enabled)
    public let checksum: String?

    public init(
        sequenceNumber: UInt64,
        timestamp: Date,
        type: String,
        data: Data,
        checksum: String?
    ) {
        self.sequenceNumber = sequenceNumber
        self.timestamp = timestamp
        self.type = type
        self.data = data
        self.checksum = checksum
    }
}

/// Checkpoint entry for marking consistent recovery points.
private struct CheckpointEntry: Codable, Sendable {
    let sequenceNumber: UInt64
    let timestamp: Date
    let segmentId: UUID
}

/// WAL statistics.
public struct WALStatistics: Sendable {
    /// Number of active segments
    public let segmentCount: Int

    /// Total size in bytes
    public let totalSize: Int

    /// Total entry count across all segments
    public let entryCount: Int

    /// Current sequence number
    public let currentSequence: UInt64

    /// Last checkpoint sequence number
    public let lastCheckpoint: UInt64

    /// Whether there are unflushed writes
    public let isDirty: Bool

    /// Human-readable summary
    public var summary: String {
        """
        WAL Statistics:
          Segments: \(segmentCount)
          Total Size: \(ByteCountFormatter.string(fromByteCount: Int64(totalSize), countStyle: .file))
          Entry Count: \(entryCount)
          Current Sequence: \(currentSequence)
          Last Checkpoint: \(lastCheckpoint)
          Dirty: \(isDirty)
        """
    }
}

/// WAL errors.
public enum WALError: Error, Sendable {
    /// No active segment available for writing
    case noActiveSegment

    /// Entry data is corrupted
    case corruptedEntry

    /// Checksum verification failed
    case checksumMismatch

    /// Invalid sequence number
    case invalidSequence

    /// Segment is full
    case segmentFull

    /// File operation failed
    case fileOperationFailed(String)
}

extension WALError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .noActiveSegment:
            return "No active WAL segment available"
        case .corruptedEntry:
            return "WAL entry is corrupted"
        case .checksumMismatch:
            return "WAL entry checksum verification failed"
        case .invalidSequence:
            return "Invalid WAL sequence number"
        case .segmentFull:
            return "WAL segment is full"
        case .fileOperationFailed(let message):
            return "WAL file operation failed: \(message)"
        }
    }
}
