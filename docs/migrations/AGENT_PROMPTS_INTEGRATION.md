# Agent Prompts: GPUDecisionEngine & WAL Integration

Use these prompts to migrate VectorAccelerate components to use the newly added `GPUDecisionEngine` and `WriteAheadLog`.

---

## Phase 1A: GPU Decision Engine - Easy Consolidation

**Estimated LOC:** ~150-250 modified
**Files:** 4
**Complexity:** Low
**One-shot:** Yes

### Prompt

```
I need you to integrate GPUDecisionEngine into several VectorAccelerate components that currently have hardcoded GPU/CPU routing thresholds.

## Context

GPUDecisionEngine was recently added at:
- `Sources/VectorAccelerate/Core/GPUDecisionEngine.swift`

It provides adaptive GPU/CPU routing with:
- `shouldUseGPU(operation:vectorCount:candidateCount:k:queryCount:dimension:)` method
- Performance history tracking via `recordPerformance()`
- Configurable thresholds via `GPUActivationThresholds`

## Task: Replace Hardcoded Thresholds

### File 1: BatchDistanceOperations.swift
Location: `Sources/VectorAccelerate/Operations/BatchDistanceOperations.swift`

Current state (lines 16-17):
```swift
private let gpuThreshold = 1000  // Use GPU for batches > 1000 vectors
private let simdThreshold = 100  // Use SIMD for batches > 100 vectors
```

Changes needed:
1. Add `GPUDecisionEngine` as an optional property (lazy initialization)
2. Replace hardcoded checks on lines 43, 144, 257, 330 with calls to `shouldUseGPU()`
3. Map operations to appropriate `GPUOperation` cases:
   - `batchEuclideanDistance` → `.l2Distance` or `.distanceMatrix`
   - `batchCosineSimilarity` → `.cosineSimilarity`
   - `batchDotProduct` → `.dotProduct`
   - `batchManhattanDistance` → `.manhattanDistance`
4. Keep `simdThreshold` for now (SIMD vs CPU decision is separate)
5. Optionally call `recordPerformance()` after operations complete

### File 2: BatchProcessor.swift
Location: `Sources/VectorAccelerate/Operations/BatchProcessor.swift`

Current state: Has `gpuThreshold: Int = 100` in configuration.

Changes needed:
1. Add optional `GPUDecisionEngine` to `BatchProcessorConfiguration`
2. When engine is provided, delegate GPU decisions to it
3. Keep existing threshold as fallback when engine is nil (backward compatibility)

### File 3: AccelerationConfiguration.swift
Location: `Sources/VectorAccelerate/Configuration/AccelerationConfiguration.swift`

Current state: Has `cpuThreshold`, `gpuThreshold`, `hybridThreshold` properties.

Changes needed:
1. Add `@available(*, deprecated, message: "Use GPUDecisionEngine for adaptive routing")` to threshold properties
2. Add convenience factory method to create `GPUActivationThresholds` from this config
3. Keep existing API working for backward compatibility

### File 4: IndexAccelerationConfiguration.swift
Location: `Sources/VectorAccelerate/Index/Core/IndexAccelerationConfiguration.swift`

Current state: Has `minimumCandidatesForGPU` and `minimumOperationsForGPU`.

Changes needed:
1. Add deprecation notices pointing to GPUDecisionEngine
2. Add migration helper to convert to GPUActivationThresholds

## Requirements

1. **Backward Compatibility**: All existing code must continue to work
2. **Optional Adoption**: GPUDecisionEngine should be opt-in, not required
3. **No Breaking Changes**: Deprecate, don't remove
4. **Test**: Run `swift build` and `swift test` to verify no regressions

## Verification

After changes:
1. `swift build` succeeds with no errors
2. Deprecation warnings appear for old threshold usage
3. Existing tests pass
4. New code paths use GPUDecisionEngine when provided
```

---

## Phase 1B: AdaptiveThresholds Consolidation

**Estimated LOC:** ~100-150 modified
**Files:** 1-2
**Complexity:** Medium-High
**One-shot:** Yes (but careful)

### Prompt

```
I need you to consolidate AdaptiveThresholdManager with GPUDecisionEngine to eliminate duplicate adaptive routing logic.

## Context

Two components now provide adaptive GPU/CPU routing:

1. **GPUDecisionEngine** (new, preferred):
   - Location: `Sources/VectorAccelerate/Core/GPUDecisionEngine.swift`
   - Actor-based, Swift 6 compliant
   - Has `shouldUseGPU()`, `recordPerformance()`, adaptive ratios

2. **AdaptiveThresholdManager** (existing, to consolidate):
   - Location: `Sources/VectorAccelerate/Configuration/AdaptiveThresholds.swift`
   - Actor-based
   - Has `recommendExecutionPath()`, `updatePerformance()`, similar adaptive logic

## Task: Consolidate Without Breaking Changes

### Option A: Wrapper Approach (Recommended)

Make AdaptiveThresholdManager delegate to GPUDecisionEngine internally:

1. Add private `GPUDecisionEngine` instance to AdaptiveThresholdManager
2. Implement `recommendExecutionPath()` by calling `shouldUseGPU()`
3. Implement `updatePerformance()` by calling `recordPerformance()`
4. Map `OperationType` to `GPUOperation`:
   - `.distanceComputation` → `.l2Distance`
   - `.batchDistanceComputation` → `.distanceMatrix`
   - `.matrixMultiplication` → `.batchMatrixMultiply`
   - `.quantization` → `.pqEncode`
5. Add deprecation notice to AdaptiveThresholdManager class

### Option B: Direct Migration

Mark AdaptiveThresholdManager as deprecated and provide migration guide:

1. Add `@available(*, deprecated, renamed: "GPUDecisionEngine")`
2. Document migration path in deprecation message
3. Keep existing implementation working

## Key Mappings

| AdaptiveThresholdManager | GPUDecisionEngine |
|-------------------------|-------------------|
| `OperationType` | `GPUOperation` |
| `ExecutionPath` | `Bool` (shouldUseGPU) |
| `recommendExecutionPath()` | `shouldUseGPU()` |
| `updatePerformance()` | `recordPerformance()` |
| `OperationThreshold` | `GPUActivationThresholds` |

## Requirements

1. **No Breaking Changes**: Existing AdaptiveThresholdManager users must work
2. **Deprecation Path**: Clear migration guidance
3. **Single Source of Truth**: Performance history should be shared

## Verification

1. `swift build` succeeds
2. Deprecation warnings guide users to GPUDecisionEngine
3. Existing tests pass
4. `AdaptiveThresholdManager` behavior unchanged
```

---

## Phase 2A: WAL Integration - Foundation

**Estimated LOC:** ~200-300 added
**Files:** 2-3
**Complexity:** Medium
**One-shot:** Yes

### Prompt

```
I need you to add Write-Ahead Log integration to AcceleratedVectorIndex for crash recovery. This phase focuses on the foundation: operation types, WAL instance, and simple logging.

## Context

WriteAheadLog was recently added at:
- `Sources/VectorAccelerate/Persistence/WriteAheadLog.swift`

AcceleratedVectorIndex is at:
- `Sources/VectorAccelerate/Index/AcceleratedVectorIndex.swift` (1185 lines)

## Task 1: Create WAL Operation Types

Create new file: `Sources/VectorAccelerate/Index/Internal/WALOperations.swift`

Define Codable operation types for WAL logging:

```swift
/// WAL operation types for AcceleratedVectorIndex
public enum IndexWALOperationType: String, Codable, Sendable {
    case insert
    case batchInsert
    case remove
    case batchRemove
    case compact
    case trainStart
    case trainComplete
    case checkpoint
}

/// Insert operation for WAL replay
public struct InsertOperation: Codable, Sendable {
    public let handle: UInt32  // stableID
    public let vector: [Float]
    public let metadata: [String: String]?
    public let timestamp: Date
}

/// Remove operation for WAL replay
public struct RemoveOperation: Codable, Sendable {
    public let handle: UInt32  // stableID
    public let timestamp: Date
}

/// Batch insert operation
public struct BatchInsertOperation: Codable, Sendable {
    public let handles: [UInt32]
    public let vectorCount: Int
    public let dimension: Int
    // Note: vectors stored separately due to size
    public let timestamp: Date
}

/// Batch remove operation
public struct BatchRemoveOperation: Codable, Sendable {
    public let handles: [UInt32]
    public let timestamp: Date
}
```

## Task 2: Add WAL to AcceleratedVectorIndex

Modify `AcceleratedVectorIndex.swift`:

1. Add WAL configuration to IndexConfiguration:
```swift
public struct WALConfiguration: Sendable {
    public var enabled: Bool = false
    public var directory: URL?
    public var syncMode: WriteAheadLog.SyncMode = .periodic

    public static let disabled = WALConfiguration(enabled: false)
    public static func enabled(directory: URL) -> WALConfiguration {
        WALConfiguration(enabled: true, directory: directory, syncMode: .periodic)
    }
}
```

2. Add WAL property to AcceleratedVectorIndex:
```swift
/// Write-ahead log for crash recovery (nil if disabled)
private var wal: WriteAheadLog?
```

3. Initialize WAL in init if configured:
```swift
if let walConfig = configuration.walConfiguration, walConfig.enabled {
    guard let directory = walConfig.directory else {
        throw IndexError.configurationError("WAL enabled but no directory specified")
    }
    self.wal = WriteAheadLog(
        directory: directory,
        configuration: .init(syncMode: walConfig.syncMode)
    )
    try await self.wal?.initialize()
}
```

## Task 3: Add Logging to insert()

In the `insert(_ vector:metadata:)` method (~line 370):

1. BEFORE the actual insert, log the operation:
```swift
if let wal = wal {
    let op = InsertOperation(
        handle: stableID,
        vector: vector,
        metadata: metadata,
        timestamp: Date()
    )
    _ = try await wal.append(op, type: IndexWALOperationType.insert.rawValue)
}
```

2. Similarly for batch insert (~line 436)

## Task 4: Add Logging to remove()

In the `remove(_ handle:)` method (~line 574):

1. Log before marking deleted:
```swift
if let wal = wal {
    let op = RemoveOperation(handle: handle.stableID, timestamp: Date())
    _ = try await wal.append(op, type: IndexWALOperationType.remove.rawValue)
}
```

## DO NOT implement yet:
- `compact()` logging (Phase 2B)
- `train()` logging (Phase 2B)
- `recover()` method (Phase 2C)

## Requirements

1. WAL is **optional** - default is disabled
2. Logging happens BEFORE mutations (write-ahead)
3. Operations must be Codable for WAL serialization
4. No changes to public API signatures

## Verification

1. `swift build` succeeds
2. Index works normally with WAL disabled (default)
3. With WAL enabled, operations are logged to disk
4. Existing tests pass
```

---

## Phase 2B: WAL Integration - Transaction Logging

**Estimated LOC:** ~100-150 added
**Files:** 1-2
**Complexity:** High
**One-shot:** Yes (focused)

### Prompt

```
I need you to add WAL logging for complex operations (compact, train) in AcceleratedVectorIndex. These require transaction semantics.

## Prerequisites

Phase 2A must be complete:
- WALOperations.swift exists with operation types
- WAL instance added to AcceleratedVectorIndex
- insert/remove logging implemented

## Context

- `compact()` modifies 4 structures atomically: storage, handleAllocator, ivfStructure, deletionMask
- `train()` is long-running and modifies IVF structure

## Task 1: Add Compact Operation Types

Add to `WALOperations.swift`:

```swift
/// Compact operation - captures state for recovery
public struct CompactStartOperation: Codable, Sendable {
    public let deletedHandles: [UInt32]
    public let vectorCountBefore: Int
    public let timestamp: Date
}

/// Compact completion marker
public struct CompactCompleteOperation: Codable, Sendable {
    public let slotMapping: [UInt32: UInt32]  // oldSlot -> newSlot
    public let vectorCountAfter: Int
    public let timestamp: Date
}
```

## Task 2: Add Train Operation Types

```swift
/// Training start marker
public struct TrainStartOperation: Codable, Sendable {
    public let vectorCount: Int
    public let numClusters: Int
    public let timestamp: Date
}

/// Training complete with centroids
public struct TrainCompleteOperation: Codable, Sendable {
    public let centroids: [[Float]]
    public let clusterAssignments: [UInt32]
    public let timestamp: Date
}
```

## Task 3: Add Logging to compact()

In `compact()` method (~line 617):

```swift
public func compact() async throws {
    // 1. Log compact START with current state
    if let wal = wal {
        let deletedHandles = deletionMask.allDeletedHandles()
        let op = CompactStartOperation(
            deletedHandles: deletedHandles,
            vectorCountBefore: handleAllocator.occupiedCount,
            timestamp: Date()
        )
        _ = try await wal.append(op, type: IndexWALOperationType.compact.rawValue)
    }

    // 2. Perform compaction (existing code)
    let slotMapping = try storage.compact(deletionMask: deletionMask)
    handleAllocator.applyCompaction(slotMapping: slotMapping)
    ivfStructure?.compact(using: slotMapping)
    deletionMask.reset()

    // 3. Log compact COMPLETE
    if let wal = wal {
        let op = CompactCompleteOperation(
            slotMapping: slotMapping,
            vectorCountAfter: handleAllocator.occupiedCount,
            timestamp: Date()
        )
        _ = try await wal.append(op, type: "compact_complete")
        // Checkpoint after successful compaction
        _ = try await wal.checkpoint()
    }
}
```

## Task 4: Add Logging to train()

In `train()` method (~line 319):

```swift
public func train() async throws {
    guard let ivf = ivfStructure else { return }

    // 1. Log training START
    if let wal = wal {
        let op = TrainStartOperation(
            vectorCount: handleAllocator.occupiedCount,
            numClusters: ivf.numClusters,
            timestamp: Date()
        )
        _ = try await wal.append(op, type: IndexWALOperationType.trainStart.rawValue)
    }

    // 2. Perform training (existing code)
    // ... existing training logic ...

    // 3. Log training COMPLETE with results
    if let wal = wal {
        let op = TrainCompleteOperation(
            centroids: ivf.centroids,
            clusterAssignments: ivf.assignments,
            timestamp: Date()
        )
        _ = try await wal.append(op, type: IndexWALOperationType.trainComplete.rawValue)
        _ = try await wal.checkpoint()
    }
}
```

## Requirements

1. **Atomic Transactions**: compact must log both start AND complete
2. **Checkpoint After**: Both operations should checkpoint on success
3. **Recovery Info**: Log enough state to replay or rollback

## Verification

1. `swift build` succeeds
2. Existing tests pass
3. compact() and train() create appropriate WAL entries
4. Checkpoints created after successful operations
```

---

## Phase 2C: WAL Integration - Recovery

**Estimated LOC:** ~300-450 added
**Files:** 2-3
**Complexity:** High
**One-shot:** Yes (but careful testing needed)

### Prompt

```
I need you to implement WAL recovery for AcceleratedVectorIndex - the ability to replay the WAL after a crash to restore index state.

## Prerequisites

Phases 2A and 2B must be complete:
- All WAL operation types defined
- WAL instance in AcceleratedVectorIndex
- Logging for insert, remove, compact, train

## Task 1: Add Recovery Method

Add to `AcceleratedVectorIndex.swift`:

```swift
/// Recover index state from Write-Ahead Log.
///
/// Call this after creating an index to replay any operations
/// that were logged but not yet persisted to the main index.
///
/// - Parameter fromCheckpoint: If true, replay only from last checkpoint
/// - Returns: Number of operations replayed
/// - Throws: If WAL is corrupted or replay fails
public func recover(fromCheckpoint: Bool = true) async throws -> Int {
    guard let wal = wal else {
        return 0  // WAL not enabled
    }

    // Get checkpoint sequence if recovering from checkpoint
    let startSequence: UInt64 = fromCheckpoint ? await getLastCheckpointSequence() : 0

    // Replay entries
    let entries = try await wal.replay(from: startSequence)
    var replayedCount = 0

    for entry in entries {
        try await replayEntry(entry)
        replayedCount += 1
    }

    return replayedCount
}

private func replayEntry(_ entry: WALEntry) async throws {
    guard let opType = IndexWALOperationType(rawValue: entry.type) else {
        // Unknown operation type - skip or throw based on config
        return
    }

    switch opType {
    case .insert:
        let op = try JSONDecoder().decode(InsertOperation.self, from: entry.data)
        try await replayInsert(op)

    case .batchInsert:
        let op = try JSONDecoder().decode(BatchInsertOperation.self, from: entry.data)
        try await replayBatchInsert(op)

    case .remove:
        let op = try JSONDecoder().decode(RemoveOperation.self, from: entry.data)
        try await replayRemove(op)

    case .batchRemove:
        let op = try JSONDecoder().decode(BatchRemoveOperation.self, from: entry.data)
        try await replayBatchRemove(op)

    case .compact:
        // Compact start - check if complete exists
        break

    case .trainStart:
        // Training start - check if complete exists
        break

    case .trainComplete:
        let op = try JSONDecoder().decode(TrainCompleteOperation.self, from: entry.data)
        try await replayTrainComplete(op)

    case .checkpoint:
        // Checkpoint markers are informational
        break
    }
}
```

## Task 2: Implement Replay Methods

```swift
private func replayInsert(_ op: InsertOperation) async throws {
    // Check if already exists (idempotency)
    if handleAllocator.hasHandle(stableID: op.handle) {
        return  // Already inserted
    }

    // Re-insert without WAL logging (we're replaying)
    let slot = handleAllocator.allocateWithID(op.handle)
    try storage.writeVector(op.vector, at: slot)
    if let metadata = op.metadata {
        metadataStore[op.handle] = metadata
    }
    if let ivf = ivfStructure, ivf.isTrained {
        try await ivf.assignToCluster(slot: slot, vector: op.vector)
    }
}

private func replayRemove(_ op: RemoveOperation) async throws {
    // Check if exists
    guard handleAllocator.hasHandle(stableID: op.handle) else {
        return  // Already removed or never existed
    }

    // Mark as deleted without WAL logging
    deletionMask.markDeleted(stableID: op.handle)
}

private func replayTrainComplete(_ op: TrainCompleteOperation) async throws {
    guard let ivf = ivfStructure else { return }

    // Restore trained state
    ivf.restoreFromTraining(
        centroids: op.centroids,
        assignments: op.clusterAssignments
    )
}
```

## Task 3: Add Static Recovery Factory

```swift
/// Create an index and recover from WAL if available.
///
/// This is the recommended way to open an index that may have
/// crashed during operation.
public static func open(
    configuration: IndexConfiguration,
    walDirectory: URL
) async throws -> AcceleratedVectorIndex {
    var config = configuration
    config.walConfiguration = .enabled(directory: walDirectory)

    let index = try await AcceleratedVectorIndex(configuration: config)
    let recovered = try await index.recover()

    if recovered > 0 {
        // Checkpoint after successful recovery
        _ = try await index.wal?.checkpoint()
    }

    return index
}
```

## Task 4: Add Tests

Create `Tests/VectorAccelerateTests/WALRecoveryTests.swift`:

```swift
final class WALRecoveryTests: XCTestCase {
    var tempDirectory: URL!

    override func setUp() async throws {
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: tempDirectory)
    }

    func testRecoveryAfterInserts() async throws {
        // 1. Create index with WAL, insert vectors
        var config = IndexConfiguration.flat(dimension: 128, capacity: 1000)
        config.walConfiguration = .enabled(directory: tempDirectory)

        let index1 = try await AcceleratedVectorIndex(configuration: config)
        let handle1 = try await index1.insert(randomVector(128))
        let handle2 = try await index1.insert(randomVector(128))

        // 2. Simulate crash (don't call any cleanup)
        // Just create new index pointing to same WAL

        // 3. Create new index and recover
        let index2 = try await AcceleratedVectorIndex.open(
            configuration: config,
            walDirectory: tempDirectory
        )

        // 4. Verify vectors exist
        XCTAssertEqual(await index2.count, 2)
    }

    func testRecoveryAfterRemoves() async throws {
        // Similar test for remove recovery
    }

    func testIdempotentReplay() async throws {
        // Test that replaying twice doesn't duplicate
    }
}
```

## Requirements

1. **Idempotency**: Replaying the same entry twice must be safe
2. **Order Independence**: Handle out-of-order replay where possible
3. **Incomplete Transactions**: Detect and skip incomplete compact/train
4. **Test Coverage**: Tests for insert, remove, and edge cases

## Verification

1. `swift build` succeeds
2. All existing tests pass
3. New WAL recovery tests pass
4. Manual test: insert vectors, "crash", recover, verify vectors exist
```

---

## Execution Order

| Phase | Prerequisites | Estimated Time |
|-------|---------------|----------------|
| **1A** | GPUDecisionEngine exists | 1 session |
| **1B** | 1A complete | 1 session |
| **2A** | WriteAheadLog exists | 1 session |
| **2B** | 2A complete | 1 session |
| **2C** | 2B complete | 1 session |

**Recommended parallel execution:**
- 1A and 2A can run in parallel (no dependencies)
- 1B depends on 1A
- 2B depends on 2A
- 2C depends on 2B
