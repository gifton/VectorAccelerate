# Task: ResidencyManager

## Objective

Create a `ResidencyManager` actor that handles explicit Metal 4 resource residency. This component is critical for ensuring GPU memory access validity in the Metal 4 model.

## Background

Metal 4 moves from implicit resource retention to explicit residency management. Every buffer accessed by the GPU must be registered in a `MTLResidencySet` that is attached to the command queue or command buffer. Failure to do so results in undefined behavior or GPU page faults.

### Key Concepts

- **MTLResidencySet**: Container for resources that should be GPU-resident
- **Residency Commit**: Makes resources actually resident in GPU memory
- **Allocation Tracking**: Adding/removing resources from the set
- **Capacity Management**: Sets have limits, must manage size

## Current State (Metal 3)

**No residency management required** - Metal 3 automatically retains resources referenced by command buffers.

```swift
// Metal 3 - implicit retention
encoder.setBuffer(buffer, offset: 0, index: 0)
// buffer automatically retained until command buffer completes
```

## Target Implementation (Metal 4)

```swift
import Metal

/// Manages explicit resource residency for Metal 4
///
/// Metal 4 requires all GPU-accessed resources to be in a residency set.
/// This actor handles:
/// - Creating and managing MTLResidencySet
/// - Tracking buffer additions/removals
/// - Integrating with BufferPool for automatic registration
/// - Handling memory pressure and eviction
public actor ResidencyManager {
    // MARK: - Types

    /// Residency registration mode
    public enum RegistrationMode: Sendable {
        /// Resource is long-lived, rarely changes
        case static_

        /// Resource is transient, frequently allocated/freed
        case ephemeral

        /// Resource is "hot" - frequently accessed, should stay resident
        case hot
    }

    /// Residency statistics for monitoring
    public struct Statistics: Sendable, Equatable {
        public let totalAllocations: Int
        public let staticAllocations: Int
        public let ephemeralAllocations: Int
        public let hotAllocations: Int
        public let totalBytesRegistered: Int
        public let commitCount: Int
        public let lastCommitTime: Date?
    }

    // MARK: - Properties

    /// The underlying Metal residency set
    public let residencySet: MTLResidencySet

    /// Device reference for capacity checks
    private let device: MTLDevice

    /// Track registered allocations for debugging and removal
    private var registeredAllocations: [ObjectIdentifier: AllocationInfo] = [:]

    /// Commit batch tracking
    private var pendingAdds: [MTLAllocation] = []
    private var pendingRemoves: [MTLAllocation] = []
    private var needsCommit: Bool = false

    /// Statistics
    private var stats: Statistics

    // MARK: - Private Types

    private struct AllocationInfo {
        let allocation: MTLAllocation
        let mode: RegistrationMode
        let byteSize: Int
        let label: String?
        let registeredAt: Date
    }

    // MARK: - Initialization

    /// Create a ResidencyManager with default configuration
    public init(device: MTLDevice) throws {
        self.device = device

        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = 256

        guard let set = try? device.makeResidencySet(descriptor: descriptor) else {
            throw Metal4Error.residencySetCreationFailed
        }

        self.residencySet = set
        self.stats = Statistics(
            totalAllocations: 0,
            staticAllocations: 0,
            ephemeralAllocations: 0,
            hotAllocations: 0,
            totalBytesRegistered: 0,
            commitCount: 0,
            lastCommitTime: nil
        )
    }

    /// Create with custom configuration
    public init(device: MTLDevice, initialCapacity: Int) throws {
        self.device = device

        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = initialCapacity

        guard let set = try? device.makeResidencySet(descriptor: descriptor) else {
            throw Metal4Error.residencySetCreationFailed
        }

        self.residencySet = set
        self.stats = Statistics(
            totalAllocations: 0,
            staticAllocations: 0,
            ephemeralAllocations: 0,
            hotAllocations: 0,
            totalBytesRegistered: 0,
            commitCount: 0,
            lastCommitTime: nil
        )
    }

    // MARK: - Registration

    /// Register a buffer for GPU residency
    ///
    /// - Parameters:
    ///   - buffer: The MTLBuffer to register
    ///   - mode: How the buffer will be used (affects eviction priority)
    /// - Throws: Metal4Error if registration fails or set is full
    public func registerBuffer(_ buffer: any MTLBuffer, mode: RegistrationMode = .ephemeral) throws {
        let id = ObjectIdentifier(buffer)

        // Check if already registered
        guard registeredAllocations[id] == nil else {
            return // Already registered, no-op
        }

        // Track allocation
        let info = AllocationInfo(
            allocation: buffer,
            mode: mode,
            byteSize: buffer.length,
            label: buffer.label,
            registeredAt: Date()
        )
        registeredAllocations[id] = info

        // Add to pending batch
        pendingAdds.append(buffer)
        needsCommit = true

        // Update stats
        updateStats(adding: info)
    }

    /// Register multiple buffers at once (more efficient)
    public func registerBuffers(_ buffers: [any MTLBuffer], mode: RegistrationMode = .ephemeral) throws {
        for buffer in buffers {
            try registerBuffer(buffer, mode: mode)
        }
    }

    /// Unregister a buffer (will be removed from residency)
    public func unregisterBuffer(_ buffer: any MTLBuffer) {
        let id = ObjectIdentifier(buffer)

        guard let info = registeredAllocations.removeValue(forKey: id) else {
            return // Not registered, no-op
        }

        pendingRemoves.append(buffer)
        needsCommit = true

        // Update stats
        updateStats(removing: info)
    }

    // MARK: - Commit

    /// Commit pending changes to the residency set
    ///
    /// This makes newly added resources resident and removes unregistered ones.
    /// Call after a batch of registrations for efficiency.
    public func commit() throws {
        guard needsCommit else { return }

        // Add new allocations
        for allocation in pendingAdds {
            residencySet.addAllocation(allocation)
        }

        // Remove old allocations
        for allocation in pendingRemoves {
            residencySet.removeAllocation(allocation)
        }

        // Commit the changes
        do {
            try residencySet.commit()
        } catch {
            throw Metal4Error.residencyCommitFailed(underlying: error)
        }

        // Clear pending
        pendingAdds.removeAll()
        pendingRemoves.removeAll()
        needsCommit = false

        // Update stats
        stats = Statistics(
            totalAllocations: stats.totalAllocations,
            staticAllocations: stats.staticAllocations,
            ephemeralAllocations: stats.ephemeralAllocations,
            hotAllocations: stats.hotAllocations,
            totalBytesRegistered: stats.totalBytesRegistered,
            commitCount: stats.commitCount + 1,
            lastCommitTime: Date()
        )
    }

    // MARK: - Query

    /// Check if a buffer is registered
    public func contains(_ buffer: any MTLBuffer) -> Bool {
        registeredAllocations[ObjectIdentifier(buffer)] != nil
    }

    /// Get current statistics
    public func getStatistics() -> Statistics {
        stats
    }

    /// Get count of registered allocations
    public var allocationCount: Int {
        registeredAllocations.count
    }

    // MARK: - Memory Management

    /// Clear all registrations (for cleanup/reset)
    public func clear() {
        for (_, info) in registeredAllocations {
            residencySet.removeAllocation(info.allocation)
        }

        try? residencySet.commit()

        registeredAllocations.removeAll()
        pendingAdds.removeAll()
        pendingRemoves.removeAll()
        needsCommit = false

        stats = Statistics(
            totalAllocations: 0,
            staticAllocations: 0,
            ephemeralAllocations: 0,
            hotAllocations: 0,
            totalBytesRegistered: 0,
            commitCount: stats.commitCount,
            lastCommitTime: stats.lastCommitTime
        )
    }

    /// Evict ephemeral allocations to reduce memory pressure
    ///
    /// Keeps static and hot allocations, removes ephemeral ones.
    /// Call when experiencing memory pressure.
    public func evictEphemeral() throws {
        var toRemove: [ObjectIdentifier] = []

        for (id, info) in registeredAllocations where info.mode == .ephemeral {
            residencySet.removeAllocation(info.allocation)
            toRemove.append(id)
        }

        for id in toRemove {
            registeredAllocations.removeValue(forKey: id)
        }

        if !toRemove.isEmpty {
            try commit()
        }
    }

    // MARK: - Private Helpers

    private func updateStats(adding info: AllocationInfo) {
        var newStatic = stats.staticAllocations
        var newEphemeral = stats.ephemeralAllocations
        var newHot = stats.hotAllocations

        switch info.mode {
        case .static_: newStatic += 1
        case .ephemeral: newEphemeral += 1
        case .hot: newHot += 1
        }

        stats = Statistics(
            totalAllocations: stats.totalAllocations + 1,
            staticAllocations: newStatic,
            ephemeralAllocations: newEphemeral,
            hotAllocations: newHot,
            totalBytesRegistered: stats.totalBytesRegistered + info.byteSize,
            commitCount: stats.commitCount,
            lastCommitTime: stats.lastCommitTime
        )
    }

    private func updateStats(removing info: AllocationInfo) {
        var newStatic = stats.staticAllocations
        var newEphemeral = stats.ephemeralAllocations
        var newHot = stats.hotAllocations

        switch info.mode {
        case .static_: newStatic -= 1
        case .ephemeral: newEphemeral -= 1
        case .hot: newHot -= 1
        }

        stats = Statistics(
            totalAllocations: stats.totalAllocations - 1,
            staticAllocations: newStatic,
            ephemeralAllocations: newEphemeral,
            hotAllocations: newHot,
            totalBytesRegistered: stats.totalBytesRegistered - info.byteSize,
            commitCount: stats.commitCount,
            lastCommitTime: stats.lastCommitTime
        )
    }
}

// MARK: - BufferPool Integration

extension BufferPool {
    /// Initialize BufferPool with ResidencyManager integration
    public init(
        device: MetalDevice,
        factory: MetalBufferFactory,
        residencyManager: ResidencyManager,
        maxBuffersPerBucket: Int,
        maxTotalMemory: Int
    ) {
        // Implementation: register buffers with residencyManager
        // when acquired, optionally unregister when returned
    }
}
```

## Files to Modify

- `Sources/VectorAccelerate/Core/ResidencyManager.swift` - NEW FILE (implementation above)
- `Sources/VectorAccelerate/Core/BufferPool.swift` - Add residency integration
- `Sources/VectorAccelerate/Core/Metal4Context.swift` - Use ResidencyManager
- `Sources/VectorAccelerate/Core/Metal4Error.swift` - Add residency errors

## Dependencies

- `Metal4Context` (creates and owns ResidencyManager)
- `Metal4Error` (for error types)

## Error Handling

### Recoverable Errors

| Error | Recovery |
|-------|----------|
| Duplicate registration | No-op, return silently |
| Remove unregistered buffer | No-op, return silently |
| Commit with no changes | No-op, return silently |

### Non-Recoverable Errors

| Error | Action |
|-------|--------|
| ResidencySet creation fails | Throw immediately, cannot continue |
| Commit fails (memory pressure) | Throw `residencyCommitFailed`, caller should evict/retry |
| Capacity exceeded | Throw `residencySetFull`, caller should evict |

## Threading Model

- **Actor isolation**: All state is actor-isolated
- **Batch commits**: Accumulate changes, commit in batches for efficiency
- **No locks needed**: Actor model handles concurrency
- **Integration**: BufferPool calls are async through actor

## Lifetime Model

```
Buffer Creation → Register → Commit → GPU Access → Unregister → Commit → Buffer Dealloc

Timeline:
1. BufferPool.getBuffer() creates buffer
2. ResidencyManager.registerBuffer() called (batched)
3. ResidencyManager.commit() makes buffer GPU-resident
4. Kernel uses buffer
5. BufferPool.returnBuffer() optionally unregisters
6. ResidencyManager.commit() removes from residency
7. Buffer can be reused or deallocated
```

## Multiple Residency Sets

**Decision Point**: Should we use one global set or multiple sets?

### Option A: Single Global Set (Recommended for v1)
- Simpler implementation
- All resources in one set attached to queue
- Good for most workloads

### Option B: Multiple Sets (Future)
- Per-operation or per-shard sets
- Finer-grained control
- Useful for very large datasets with hot/cold regions

**Recommendation**: Start with single set, add multi-set support if needed.

## Acceptance Criteria

- [ ] `ResidencyManager` compiles without errors
- [ ] Can create `MTLResidencySet` successfully
- [ ] Buffers can be registered and committed
- [ ] Buffers can be unregistered
- [ ] Batched commits work correctly
- [ ] Statistics tracking is accurate
- [ ] Memory pressure eviction works
- [ ] Integration with BufferPool works
- [ ] All tests pass

## Test Cases

```swift
func testResidencyManagerCreation() async throws {
    let device = MTLCreateSystemDefaultDevice()!
    let manager = try ResidencyManager(device: device)
    XCTAssertNotNil(manager)
}

func testBufferRegistration() async throws {
    let device = MTLCreateSystemDefaultDevice()!
    let manager = try ResidencyManager(device: device)

    let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
    try await manager.registerBuffer(buffer, mode: .ephemeral)
    try await manager.commit()

    XCTAssertTrue(await manager.contains(buffer))
}

func testBufferUnregistration() async throws {
    let device = MTLCreateSystemDefaultDevice()!
    let manager = try ResidencyManager(device: device)

    let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
    try await manager.registerBuffer(buffer)
    try await manager.commit()

    await manager.unregisterBuffer(buffer)
    try await manager.commit()

    XCTAssertFalse(await manager.contains(buffer))
}

func testBatchRegistration() async throws {
    let device = MTLCreateSystemDefaultDevice()!
    let manager = try ResidencyManager(device: device)

    var buffers: [MTLBuffer] = []
    for _ in 0..<100 {
        buffers.append(device.makeBuffer(length: 1024, options: .storageModeShared)!)
    }

    try await manager.registerBuffers(buffers)
    try await manager.commit()

    XCTAssertEqual(await manager.allocationCount, 100)
}

func testStatistics() async throws {
    let device = MTLCreateSystemDefaultDevice()!
    let manager = try ResidencyManager(device: device)

    let buffer1 = device.makeBuffer(length: 1024, options: .storageModeShared)!
    let buffer2 = device.makeBuffer(length: 2048, options: .storageModeShared)!

    try await manager.registerBuffer(buffer1, mode: .static_)
    try await manager.registerBuffer(buffer2, mode: .ephemeral)
    try await manager.commit()

    let stats = await manager.getStatistics()
    XCTAssertEqual(stats.totalAllocations, 2)
    XCTAssertEqual(stats.staticAllocations, 1)
    XCTAssertEqual(stats.ephemeralAllocations, 1)
    XCTAssertEqual(stats.totalBytesRegistered, 3072)
    XCTAssertEqual(stats.commitCount, 1)
}

func testEvictEphemeral() async throws {
    let device = MTLCreateSystemDefaultDevice()!
    let manager = try ResidencyManager(device: device)

    let staticBuffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
    let ephemeralBuffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

    try await manager.registerBuffer(staticBuffer, mode: .static_)
    try await manager.registerBuffer(ephemeralBuffer, mode: .ephemeral)
    try await manager.commit()

    try await manager.evictEphemeral()

    XCTAssertTrue(await manager.contains(staticBuffer))
    XCTAssertFalse(await manager.contains(ephemeralBuffer))
}

func testDuplicateRegistration() async throws {
    let device = MTLCreateSystemDefaultDevice()!
    let manager = try ResidencyManager(device: device)

    let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!

    try await manager.registerBuffer(buffer)
    try await manager.registerBuffer(buffer) // Should not throw or duplicate
    try await manager.commit()

    XCTAssertEqual(await manager.allocationCount, 1)
}
```

## Context Files to Include

When sharing with external agent, include:
1. `00-context/metal4-api-summary.md` (residency set API)
2. `01-foundation/task-metal4context.md` (how it's used)
3. This task file
4. Current `BufferPool.swift` source

## Notes

- Actual `MTLResidencySet` API may differ slightly - verify against Xcode 17 headers
- Consider memory pressure notifications from system
- Future: add per-shard residency sets for large databases
- `MTLAllocation` protocol adopted by `MTLBuffer`, `MTLTexture`, `MTLHeap`
