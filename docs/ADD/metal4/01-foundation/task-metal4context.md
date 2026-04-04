# Task: Metal4Context

## Objective

Create a new `Metal4Context` actor that uses Metal 4's MTL4CommandQueue and MTL4CommandBuffer APIs, with proper residency management.

## Background

Metal 4 decouples command buffers from command queues and requires explicit resource residency management. This is a fundamental change from Metal 3's implicit resource retention model.

## Current Implementation

**File:** `Sources/VectorAccelerate/Core/MetalContext.swift`

```swift
public actor MetalContext: AccelerationProvider {
    public let device: MetalDevice
    public let bufferPool: BufferPool
    internal let commandQueue: any MTLCommandQueue
    public nonisolated let bufferFactory: MetalBufferFactory

    nonisolated public init(configuration: MetalConfiguration = .default) async throws {
        // ...
        self.commandQueue = try await device.makeCommandQueue(label: configuration.commandQueueLabel)
        // ...
    }

    public func makeCommandBuffer() -> (any MTLCommandBuffer)? {
        commandQueue.makeCommandBuffer()
    }

    public func execute<T: Sendable>(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T
    ) async throws -> T {
        guard let commandBuffer = makeCommandBuffer() else {
            throw VectorError.deviceInitializationFailed("Failed to create command buffer")
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw VectorError.deviceInitializationFailed("Failed to create compute encoder")
        }

        let result = try await operation(commandBuffer, encoder)
        encoder.endEncoding()
        commandBuffer.commit()

        return result
    }
}
```

## Target Implementation (Metal 4)

```swift
import Metal

/// Metal 4 context with explicit residency management
public actor Metal4Context {
    // MARK: - Properties

    public let device: MetalDevice
    public let bufferPool: BufferPool
    public nonisolated let bufferFactory: MetalBufferFactory

    /// Metal 4 command queue with residency support
    internal let commandQueue: MTL4CommandQueue

    /// Manages resource residency for GPU access
    internal let residencyManager: ResidencyManager

    /// Shared event for synchronization
    private let completionEvent: MTLSharedEvent
    private var eventCounter: UInt64 = 0

    // MARK: - Configuration

    public let configuration: MetalConfiguration

    // MARK: - Initialization

    public init(configuration: MetalConfiguration = .default) async throws {
        self.configuration = configuration

        // Select device
        if configuration.preferHighPerformanceDevice {
            self.device = try await MetalDevice.selectBestDevice()
        } else {
            self.device = try MetalDevice()
        }

        // Verify Metal 4 support
        guard device.supportsMetal4 else {
            throw VectorError.deviceInitializationFailed("Metal 4 not supported on this device")
        }

        // Create buffer factory
        self.bufferFactory = device.makeBufferFactory()

        // Create Metal 4 command queue
        let queueDescriptor = MTL4CommandQueueDescriptor()
        queueDescriptor.maxCommandBufferCount = 3
        guard let queue = device.device.makeCommandQueue(descriptor: queueDescriptor) as? MTL4CommandQueue else {
            throw VectorError.deviceInitializationFailed("Failed to create MTL4CommandQueue")
        }
        self.commandQueue = queue

        // Create residency manager
        self.residencyManager = try ResidencyManager(device: device.device)

        // Attach residency set to queue
        commandQueue.addResidencySet(residencyManager.residencySet)

        // Create completion event
        guard let event = device.device.makeSharedEvent() else {
            throw VectorError.deviceInitializationFailed("Failed to create shared event")
        }
        self.completionEvent = event

        // Initialize buffer pool with residency integration
        self.bufferPool = BufferPool(
            device: device,
            factory: bufferFactory,
            residencyManager: residencyManager,
            maxBuffersPerBucket: configuration.maxBuffersPerSize,
            maxTotalMemory: configuration.maxBufferPoolMemory
        )
    }

    // MARK: - Command Buffer Creation

    /// Create a new Metal 4 command buffer (from device, not queue)
    public func makeCommandBuffer() -> MTL4CommandBuffer? {
        device.device.makeCommandBuffer() as? MTL4CommandBuffer
    }

    // MARK: - Execution

    /// Execute a compute operation with Metal 4 APIs
    public func execute<T: Sendable>(
        _ operation: @Sendable (MTL4CommandBuffer, MTL4ComputeCommandEncoder) async throws -> T
    ) async throws -> T {
        guard let commandBuffer = makeCommandBuffer() else {
            throw VectorError.deviceInitializationFailed("Failed to create command buffer")
        }

        // Attach residency set to this command buffer
        commandBuffer.useResidencySet(residencyManager.residencySet)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() as? MTL4ComputeCommandEncoder else {
            throw VectorError.deviceInitializationFailed("Failed to create compute encoder")
        }

        let result = try await operation(commandBuffer, encoder)

        encoder.endEncoding()

        // Submit via queue (Metal 4 pattern)
        commandQueue.commit([commandBuffer])

        return result
    }

    /// Execute and wait for completion using shared event
    public func executeAndWait(
        _ operation: @Sendable (MTL4CommandBuffer, MTL4ComputeCommandEncoder) async throws -> Void
    ) async throws {
        guard let commandBuffer = makeCommandBuffer() else {
            throw VectorError.deviceInitializationFailed("Failed to create command buffer")
        }

        commandBuffer.useResidencySet(residencyManager.residencySet)

        guard let encoder = commandBuffer.makeComputeCommandEncoder() as? MTL4ComputeCommandEncoder else {
            throw VectorError.deviceInitializationFailed("Failed to create compute encoder")
        }

        try await operation(commandBuffer, encoder)

        encoder.endEncoding()

        // Increment event counter
        eventCounter += 1
        let targetValue = eventCounter

        // Submit and signal
        commandQueue.commit([commandBuffer])
        commandQueue.signalEvent(completionEvent, value: targetValue)

        // Wait for completion
        await withCheckedContinuation { continuation in
            completionEvent.notifyListener(at: targetValue) { _, _ in
                continuation.resume()
            }
        }
    }

    // MARK: - Resource Management

    /// Get a buffer from the pool (with automatic residency)
    public func getBuffer(size: Int) async throws -> BufferToken {
        try await bufferPool.getBuffer(size: size)
    }

    /// Get a buffer for typed data
    public func getBuffer<T: Sendable>(for data: [T]) async throws -> BufferToken {
        try await bufferPool.getBuffer(for: data)
    }

    // MARK: - Cleanup

    public func cleanup() async {
        await bufferPool.reset()
        residencyManager.clear()
    }
}
```

## Files to Modify

- `Sources/VectorAccelerate/Core/MetalContext.swift` - Keep existing, add deprecation notice
- `Sources/VectorAccelerate/Core/Metal4Context.swift` - NEW FILE (implementation above)
- `Sources/VectorAccelerate/Core/MetalDevice.swift` - Add `supportsMetal4` property
- `Sources/VectorAccelerate/Core/BufferPool.swift` - Add residency manager parameter

## Dependencies

- None (this is a foundational task)

## Acceptance Criteria

- [ ] `Metal4Context` compiles without errors
- [ ] Can create `MTL4CommandQueue` successfully
- [ ] Can create `MTL4CommandBuffer` from device
- [ ] Command buffers are submitted via queue (not buffer.commit())
- [ ] Residency set is attached to queue
- [ ] Completion uses `MTLSharedEvent` instead of `addCompletedHandler`
- [ ] Unit tests pass for basic operations

## Test Cases

```swift
func testMetal4ContextCreation() async throws {
    let context = try await Metal4Context()
    XCTAssertNotNil(context)
}

func testCommandBufferCreation() async throws {
    let context = try await Metal4Context()
    let buffer = context.makeCommandBuffer()
    XCTAssertNotNil(buffer)
}

func testBasicExecution() async throws {
    let context = try await Metal4Context()
    var executed = false

    try await context.executeAndWait { commandBuffer, encoder in
        // Simple operation
        executed = true
    }

    XCTAssertTrue(executed)
}

func testResidencySetAttached() async throws {
    let context = try await Metal4Context()
    let buffer = try await context.getBuffer(size: 1024)

    // Buffer should be in residency set
    XCTAssertTrue(context.residencyManager.contains(buffer.buffer))
}
```

## Context Files to Include

When sharing with external agent, include:
1. `00-context/architecture.md`
2. `00-context/metal4-api-summary.md`
3. `00-context/patterns.md`
4. This task file
5. Current `MetalContext.swift` source
6. Current `MetalDevice.swift` source

## Concurrency Semantics

This section defines the threading and concurrency model for `Metal4Context`.

### Command Buffer Lifecycle

**Policy: One command buffer per logical operation**

Each call to `execute()` or `executeAndWait()` creates a fresh `MTL4CommandBuffer`. Command buffers are:
- Created from device (not reused)
- Submitted via queue after encoding completes
- Automatically released after GPU completion

```swift
// Each execute call = new command buffer
try await context.executeAndWait { cb, encoder in
    // This is a fresh command buffer
}

try await context.executeAndWait { cb, encoder in
    // This is a different command buffer
}
```

**Rationale:** Simplicity over optimization. Command buffer creation is cheap in Metal 4. Reuse adds complexity with minimal benefit for typical workloads.

**Future optimization:** Consider command buffer pooling if profiling shows creation overhead.

### Parallel Operation Support

**Policy: Multiple overlapping operations are allowed**

Callers may issue many overlapping operations on one context:

```swift
// Legal: Multiple concurrent operations
async let result1 = context.execute { cb, encoder in
    // Operation 1
}
async let result2 = context.execute { cb, encoder in
    // Operation 2
}
let (r1, r2) = try await (result1, result2)
```

**Ordering guarantees:**
- Operations within a single `execute()` block are ordered
- Operations across `execute()` blocks may execute in any order
- Use `executeAndWait()` when ordering matters
- Use barriers within an encoder for intra-operation ordering

### Actor Isolation

`Metal4Context` is an actor, so:
- All mutable state is protected
- Callers must `await` all method calls
- No explicit locking needed

```swift
// All calls are async due to actor isolation
let buffer = try await context.getBuffer(size: 1024)
try await context.execute { ... }
```

### AccelerationProvider Contract

Both `MetalContext` and `Metal4Context` should conform to a protocol:

```swift
/// Protocol for GPU acceleration contexts
public protocol AccelerationContext: Actor {
    /// Execute a compute operation
    func execute<T: Sendable>(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> T
    ) async throws -> T

    /// Execute and wait for GPU completion
    func executeAndWait(
        _ operation: @Sendable (any MTLCommandBuffer, any MTLComputeCommandEncoder) async throws -> Void
    ) async throws

    /// Get a buffer from the pool
    func getBuffer(size: Int) async throws -> BufferToken
}
```

**Caller guarantees:**
- All operations are `async` - GPU work is off the calling actor
- Operations may throw on GPU errors
- Buffers obtained via `getBuffer()` are valid until returned to pool

### Task Cancellation

**Policy: Cancellation is best-effort**

```swift
let task = Task {
    try await context.executeAndWait { cb, encoder in
        // Long-running operation
    }
}

task.cancel()
// Cancellation will be checked at next await point
// GPU work in progress will complete
```

**Behavior:**
- Cancellation is checked before starting new operations
- In-flight GPU work is NOT cancelled (Metal doesn't support this)
- Results from cancelled operations should be discarded

```swift
public func execute<T: Sendable>(...) async throws -> T {
    // Check cancellation before starting
    try Task.checkCancellation()

    guard let commandBuffer = makeCommandBuffer() else { ... }
    // ... encoding ...

    // Check again before waiting
    try Task.checkCancellation()

    // Wait for GPU
    await waitForCompletion()

    return result
}
```

### Timeout Support (Optional)

Consider adding timeout support for long operations:

```swift
public func executeAndWait(
    timeout: Duration = .seconds(30),
    _ operation: @Sendable (...) async throws -> Void
) async throws {
    // ... encoding ...

    // Wait with timeout
    let completed = try await withTimeout(timeout) {
        await waitForCompletion()
    }

    if !completed {
        throw Metal4Error.timeout(duration: timeout)
    }
}
```

### Thread Safety Summary

| Component | Thread Safety | Notes |
|-----------|---------------|-------|
| Metal4Context | Actor-isolated | All access is serialized |
| ResidencyManager | Actor-isolated | Owned by context |
| BufferPool | Actor-isolated | Owned by context |
| MTL4CommandBuffer | Thread-safe | Metal provides this |
| ArgumentTable | Per-operation | Not shared across operations |
| MTLSharedEvent | Thread-safe | Used for synchronization |

### Multi-Operation Patterns

**Pattern 1: Sequential operations (simplest)**
```swift
let distResult = try await context.executeAndWait { cb, encoder in
    // Distance computation
}
let topKResult = try await context.executeAndWait { cb, encoder in
    // TopK selection using distResult
}
```

**Pattern 2: Fused operations (better performance)**
```swift
try await context.executeAndWait { cb, encoder in
    // Distance computation
    encoder.setComputePipelineState(distancePipeline)
    encoder.dispatch(...)

    // Barrier
    encoder.barrier(resources: [distanceBuffer], beforeStages: .dispatch, afterStages: .dispatch)

    // TopK selection
    encoder.setComputePipelineState(topKPipeline)
    encoder.dispatch(...)
}
```

**Pattern 3: Parallel independent operations**
```swift
// Process multiple queries in parallel
let results = try await withThrowingTaskGroup(of: [Float].self) { group in
    for query in queries {
        group.addTask {
            try await context.executeAndWait { cb, encoder in
                // Process single query
            }
        }
    }
    return try await group.reduce(into: []) { $0.append(contentsOf: $1) }
}
```

---

## Notes

- The `MTL4CommandQueue`, `MTL4CommandBuffer`, and `MTL4ComputeCommandEncoder` types are Metal 4 specific
- Actual API names may differ slightly from documentation - verify against Xcode 17 headers
- Consider adding a protocol that both `MetalContext` and `Metal4Context` conform to for easier migration
- Command buffer creation per-operation is deliberate - profile before optimizing
