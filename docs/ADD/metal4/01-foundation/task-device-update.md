# Task: MetalDevice Updates for Metal 4

## Objective

Update `MetalDevice` to support Metal 4 capabilities detection, backend selection, and expose `Metal4Capabilities` for feature gating throughout the codebase.

## Background

Metal 4 features are not uniformly available across all Apple Silicon devices. Some features require specific GPU families (e.g., M3+ for placement sparse). VectorAccelerate needs granular capability detection to:

1. Choose appropriate backend (Metal 3 vs Metal 4)
2. Gate advanced features per-device
3. Provide clear errors when features unavailable
4. Optimize code paths for specific hardware

## Current Implementation

**File:** `Sources/VectorAccelerate/Core/MetalDevice.swift`

```swift
public actor MetalDevice {
    public let device: any MTLDevice
    public let capabilities: MetalDeviceCapabilities

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw VectorError.deviceInitializationFailed("No Metal device available")
        }
        self.device = device
        self.capabilities = MetalDeviceCapabilities(device: device)
    }

    // Factory methods
    public func makeCommandQueue(label: String) async throws -> any MTLCommandQueue
    public func makeBuffer(length: Int, options: MTLResourceOptions) -> (any MTLBuffer)?
    public func makeLibrary(source: String) async throws -> any MTLLibrary
}

public struct MetalDeviceCapabilities: Sendable {
    public let maxThreadsPerThreadgroup: Int
    public let maxBufferLength: Int
    public let supportsFamily: (MTLGPUFamily) -> Bool
    // ... current capabilities
}
```

## Target Implementation

### 1. Metal4Capabilities Struct

```swift
/// Granular Metal 4 feature detection
///
/// Use this to check individual Metal 4 features rather than a blanket
/// "supports Metal 4" check. Features may vary by GPU family.
public struct Metal4Capabilities: Sendable, Equatable, Codable {
    // MARK: - Core Support

    /// Core Metal 4 support (MTL4CommandQueue, MTL4CommandBuffer, etc.)
    /// Requires Metal 4 GPU family
    public let supportsMetal4Core: Bool

    /// Can create MTL4Compiler for shader compilation
    public let supportsMTL4Compiler: Bool

    /// Can use MTL4ArgumentTable for resource binding
    public let supportsArgumentTables: Bool

    /// Can use MTLResidencySet for explicit residency
    public let supportsResidencySets: Bool

    // MARK: - Advanced Memory Features

    /// Placement heaps for sub-allocation control
    public let supportsPlacementHeaps: Bool

    /// Sparse buffers for large datasets
    public let supportsPlacementSparseBuffers: Bool

    // MARK: - ML/Tensor Features

    /// MTLTensor type for inline ML in shaders
    public let supportsMLTensor: Bool

    /// Machine Learning command encoder
    public let supportsMachineLearningEncoder: Bool

    // MARK: - Compute Features

    /// Enhanced barrier system
    public let supportsAdvancedBarriers: Bool

    /// Simdgroup matrix operations (8x8, 16x16)
    public let supportsSimdgroupMatrix: Bool

    /// Maximum simdgroup matrix dimensions
    public let maxSimdgroupMatrixDimension: Int

    // MARK: - Limits

    /// Maximum buffers in an argument table
    public let maxArgumentTableBuffers: Int

    /// Maximum textures in an argument table
    public let maxArgumentTableTextures: Int

    /// Maximum residency set capacity
    public let maxResidencySetCapacity: Int

    // MARK: - Initialization

    /// Initialize from MTLDevice
    public init(device: MTLDevice) {
        // Core Metal 4 - requires metal4 family
        self.supportsMetal4Core = device.supportsFamily(.metal4)
        self.supportsMTL4Compiler = device.supportsFamily(.metal4)
        self.supportsArgumentTables = device.supportsFamily(.metal4)
        self.supportsResidencySets = device.supportsFamily(.metal4)

        // Advanced memory - M3+ (apple9)
        self.supportsPlacementHeaps = device.supportsFamily(.apple9)
        self.supportsPlacementSparseBuffers = device.supportsFamily(.apple9)

        // ML features
        self.supportsMLTensor = device.supportsFamily(.metal4)
        self.supportsMachineLearningEncoder = device.supportsFamily(.apple9)

        // Compute features
        self.supportsAdvancedBarriers = device.supportsFamily(.metal4)
        self.supportsSimdgroupMatrix = device.supportsFamily(.apple8)  // M2+

        // Limits - these may need runtime query
        if device.supportsFamily(.apple9) {
            self.maxSimdgroupMatrixDimension = 16
            self.maxArgumentTableBuffers = 64
            self.maxArgumentTableTextures = 128
            self.maxResidencySetCapacity = 4096
        } else if device.supportsFamily(.apple8) {
            self.maxSimdgroupMatrixDimension = 8
            self.maxArgumentTableBuffers = 31
            self.maxArgumentTableTextures = 64
            self.maxResidencySetCapacity = 2048
        } else {
            self.maxSimdgroupMatrixDimension = 8
            self.maxArgumentTableBuffers = 31
            self.maxArgumentTableTextures = 31
            self.maxResidencySetCapacity = 1024
        }
    }

    /// Create capabilities indicating no Metal 4 support
    public static var unsupported: Metal4Capabilities {
        Metal4Capabilities(
            supportsMetal4Core: false,
            supportsMTL4Compiler: false,
            supportsArgumentTables: false,
            supportsResidencySets: false,
            supportsPlacementHeaps: false,
            supportsPlacementSparseBuffers: false,
            supportsMLTensor: false,
            supportsMachineLearningEncoder: false,
            supportsAdvancedBarriers: false,
            supportsSimdgroupMatrix: false,
            maxSimdgroupMatrixDimension: 0,
            maxArgumentTableBuffers: 0,
            maxArgumentTableTextures: 0,
            maxResidencySetCapacity: 0
        )
    }

    // Private memberwise init for static factory
    private init(
        supportsMetal4Core: Bool,
        supportsMTL4Compiler: Bool,
        supportsArgumentTables: Bool,
        supportsResidencySets: Bool,
        supportsPlacementHeaps: Bool,
        supportsPlacementSparseBuffers: Bool,
        supportsMLTensor: Bool,
        supportsMachineLearningEncoder: Bool,
        supportsAdvancedBarriers: Bool,
        supportsSimdgroupMatrix: Bool,
        maxSimdgroupMatrixDimension: Int,
        maxArgumentTableBuffers: Int,
        maxArgumentTableTextures: Int,
        maxResidencySetCapacity: Int
    ) {
        self.supportsMetal4Core = supportsMetal4Core
        self.supportsMTL4Compiler = supportsMTL4Compiler
        self.supportsArgumentTables = supportsArgumentTables
        self.supportsResidencySets = supportsResidencySets
        self.supportsPlacementHeaps = supportsPlacementHeaps
        self.supportsPlacementSparseBuffers = supportsPlacementSparseBuffers
        self.supportsMLTensor = supportsMLTensor
        self.supportsMachineLearningEncoder = supportsMachineLearningEncoder
        self.supportsAdvancedBarriers = supportsAdvancedBarriers
        self.supportsSimdgroupMatrix = supportsSimdgroupMatrix
        self.maxSimdgroupMatrixDimension = maxSimdgroupMatrixDimension
        self.maxArgumentTableBuffers = maxArgumentTableBuffers
        self.maxArgumentTableTextures = maxArgumentTableTextures
        self.maxResidencySetCapacity = maxResidencySetCapacity
    }
}

extension Metal4Capabilities: CustomStringConvertible {
    public var description: String {
        """
        Metal4Capabilities:
          Core: \(supportsMetal4Core)
          ArgumentTables: \(supportsArgumentTables)
          ResidencySets: \(supportsResidencySets)
          PlacementSparse: \(supportsPlacementSparseBuffers)
          MLTensor: \(supportsMLTensor)
          SimdgroupMatrix: \(supportsSimdgroupMatrix) (max: \(maxSimdgroupMatrixDimension))
        """
    }
}
```

### 2. GPUBackend Enum

```swift
/// Backend selection for VectorAccelerate operations
public enum GPUBackend: Sendable {
    /// Classic Metal 3 backend
    case metal3(MTLDevice)

    /// Metal 4 backend with specific capabilities
    case metal4(MTLDevice, Metal4Capabilities)

    /// The underlying device
    public var device: MTLDevice {
        switch self {
        case .metal3(let device): return device
        case .metal4(let device, _): return device
        }
    }

    /// Metal 4 capabilities (nil for Metal 3)
    public var metal4Capabilities: Metal4Capabilities? {
        switch self {
        case .metal3: return nil
        case .metal4(_, let caps): return caps
        }
    }

    /// Whether this is a Metal 4 backend
    public var isMetal4: Bool {
        switch self {
        case .metal3: return false
        case .metal4: return true
        }
    }

    /// Select recommended backend for device
    public static func recommended(for device: MTLDevice) -> GPUBackend {
        let caps = Metal4Capabilities(device: device)
        if caps.supportsMetal4Core {
            return .metal4(device, caps)
        }
        return .metal3(device)
    }

    /// Force specific backend (throws if not supported)
    public static func force(_ preference: BackendPreference, device: MTLDevice) throws -> GPUBackend {
        let caps = Metal4Capabilities(device: device)

        switch preference {
        case .metal3Only:
            return .metal3(device)

        case .metal4IfAvailable:
            if caps.supportsMetal4Core {
                return .metal4(device, caps)
            }
            return .metal3(device)

        case .metal4Required:
            guard caps.supportsMetal4Core else {
                throw Metal4Error.metal4NotSupported(
                    reason: "Device \(device.name) does not support Metal 4"
                )
            }
            return .metal4(device, caps)

        case .automatic:
            return recommended(for: device)
        }
    }
}

/// User preference for backend selection
public enum BackendPreference: Sendable, Codable {
    /// Use Metal 3 always
    case metal3Only

    /// Use Metal 4 if available, fall back to Metal 3
    case metal4IfAvailable

    /// Require Metal 4, fail if unavailable
    case metal4Required

    /// Auto-select based on device and rollout phase
    case automatic
}
```

### 3. Updated MetalDevice

```swift
public actor MetalDevice {
    // MARK: - Properties

    public let device: any MTLDevice

    /// Legacy capabilities (Metal 3)
    public let capabilities: MetalDeviceCapabilities

    /// Metal 4 specific capabilities
    public let metal4Capabilities: Metal4Capabilities

    /// Selected backend
    public let backend: GPUBackend

    // MARK: - Initialization

    public init(preference: BackendPreference = .automatic) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw VectorError.deviceInitializationFailed("No Metal device available")
        }

        self.device = device
        self.capabilities = MetalDeviceCapabilities(device: device)
        self.metal4Capabilities = Metal4Capabilities(device: device)
        self.backend = try GPUBackend.force(preference, device: device)
    }

    public init(device: MTLDevice, preference: BackendPreference = .automatic) throws {
        self.device = device
        self.capabilities = MetalDeviceCapabilities(device: device)
        self.metal4Capabilities = Metal4Capabilities(device: device)
        self.backend = try GPUBackend.force(preference, device: device)
    }

    // MARK: - Convenience

    /// Whether Metal 4 is active
    public var isMetal4Active: Bool {
        backend.isMetal4
    }

    /// Check if a specific feature is available
    public func supports(_ feature: Metal4Feature) -> Bool {
        guard let caps = backend.metal4Capabilities else {
            return false
        }

        switch feature {
        case .argumentTables: return caps.supportsArgumentTables
        case .residencySets: return caps.supportsResidencySets
        case .placementSparse: return caps.supportsPlacementSparseBuffers
        case .mlTensor: return caps.supportsMLTensor
        case .simdgroupMatrix: return caps.supportsSimdgroupMatrix
        case .advancedBarriers: return caps.supportsAdvancedBarriers
        }
    }

    // MARK: - Factory Methods (Metal 4)

    /// Create MTL4CommandQueue
    public func makeMetal4CommandQueue(
        maxCommandBufferCount: Int = 3
    ) async throws -> MTL4CommandQueue {
        guard isMetal4Active else {
            throw Metal4Error.metal4NotSupported(reason: "Metal 4 backend not active")
        }

        let descriptor = MTL4CommandQueueDescriptor()
        descriptor.maxCommandBufferCount = maxCommandBufferCount

        guard let queue = device.makeCommandQueue(descriptor: descriptor) as? MTL4CommandQueue else {
            throw Metal4Error.commandQueueCreationFailed(underlying: nil)
        }

        return queue
    }

    /// Create MTL4Compiler
    public func makeMetal4Compiler(
        qualityOfService: QualityOfService = .userInteractive
    ) async throws -> MTL4Compiler {
        guard metal4Capabilities.supportsMTL4Compiler else {
            throw Metal4Error.metal4NotSupported(reason: "MTL4Compiler not supported")
        }

        let descriptor = MTL4CompilerDescriptor()
        descriptor.qualityOfService = qualityOfService

        guard let compiler = try? device.makeCompiler(descriptor: descriptor) else {
            throw Metal4Error.compilerCreationFailed(underlying: nil)
        }

        return compiler
    }

    /// Create residency set
    public func makeResidencySet(initialCapacity: Int = 256) async throws -> MTLResidencySet {
        guard metal4Capabilities.supportsResidencySets else {
            throw Metal4Error.metal4NotSupported(reason: "ResidencySets not supported")
        }

        let descriptor = MTLResidencySetDescriptor()
        descriptor.initialCapacity = initialCapacity

        return try device.makeResidencySet(descriptor: descriptor)
    }

    /// Create argument table
    public func makeArgumentTable(
        maxBuffers: Int? = nil,
        maxTextures: Int? = nil
    ) async throws -> MTL4ArgumentTable {
        guard metal4Capabilities.supportsArgumentTables else {
            throw Metal4Error.metal4NotSupported(reason: "ArgumentTables not supported")
        }

        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = maxBuffers ?? metal4Capabilities.maxArgumentTableBuffers
        descriptor.maxTextureBindCount = maxTextures ?? metal4Capabilities.maxArgumentTableTextures

        return try device.makeArgumentTable(descriptor: descriptor)
    }
}

/// Checkable Metal 4 features
public enum Metal4Feature: Sendable, CaseIterable {
    case argumentTables
    case residencySets
    case placementSparse
    case mlTensor
    case simdgroupMatrix
    case advancedBarriers
}
```

### 4. Environment Variable Support

```swift
extension MetalDevice {
    /// Check environment for backend override
    static func environmentPreference() -> BackendPreference? {
        guard let value = ProcessInfo.processInfo.environment["VECTORACCELERATE_BACKEND"] else {
            return nil
        }

        switch value.lowercased() {
        case "metal3": return .metal3Only
        case "metal4": return .metal4Required
        case "auto": return .automatic
        default:
            print("Warning: Unknown VECTORACCELERATE_BACKEND value '\(value)', using automatic")
            return .automatic
        }
    }

    /// Create device respecting environment override
    public static func createWithEnvironment(
        defaultPreference: BackendPreference = .automatic
    ) throws -> MetalDevice {
        let preference = environmentPreference() ?? defaultPreference
        return try MetalDevice(preference: preference)
    }
}
```

## Files to Modify

| File | Changes |
|------|---------|
| `Core/MetalDevice.swift` | Add Metal4Capabilities, GPUBackend, factory methods |
| `Core/Metal4Capabilities.swift` | NEW FILE - capability struct |
| `Core/GPUBackend.swift` | NEW FILE - backend enum |
| `Core/Metal4Error.swift` | Add capability-related errors |
| `Core/MetalContext.swift` | Use backend selection |
| `Core/Metal4Context.swift` | Require Metal 4 backend |

## Dependencies

- None (this is foundational)

## Acceptance Criteria

- [ ] `Metal4Capabilities` correctly detects features per device
- [ ] `GPUBackend` correctly selects backend based on preference
- [ ] Environment variable override works
- [ ] Factory methods create Metal 4 objects when supported
- [ ] Clear errors when features unavailable
- [ ] Compiles without warnings
- [ ] All existing tests pass

## Test Cases

```swift
func testMetal4CapabilitiesDetection() {
    let device = MTLCreateSystemDefaultDevice()!
    let caps = Metal4Capabilities(device: device)

    // On modern hardware, these should be true
    // Test validates the detection logic runs without crash
    print("Metal 4 Core: \(caps.supportsMetal4Core)")
    print("Argument Tables: \(caps.supportsArgumentTables)")
    print("ML Tensor: \(caps.supportsMLTensor)")
}

func testBackendSelection() throws {
    let device = MTLCreateSystemDefaultDevice()!

    // Automatic should work
    let auto = GPUBackend.recommended(for: device)
    XCTAssertNotNil(auto.device)

    // Metal 3 should always work
    let m3 = try GPUBackend.force(.metal3Only, device: device)
    XCTAssertFalse(m3.isMetal4)

    // Metal 4 required may throw on older hardware
    do {
        let m4 = try GPUBackend.force(.metal4Required, device: device)
        XCTAssertTrue(m4.isMetal4)
    } catch Metal4Error.metal4NotSupported {
        // Expected on older hardware
    }
}

func testEnvironmentOverride() {
    // Test with environment variable set
    setenv("VECTORACCELERATE_BACKEND", "metal3", 1)
    XCTAssertEqual(MetalDevice.environmentPreference(), .metal3Only)

    setenv("VECTORACCELERATE_BACKEND", "metal4", 1)
    XCTAssertEqual(MetalDevice.environmentPreference(), .metal4Required)

    unsetenv("VECTORACCELERATE_BACKEND")
    XCTAssertNil(MetalDevice.environmentPreference())
}

func testFeatureCheck() async throws {
    let device = try MetalDevice(preference: .automatic)

    // Check all features without crash
    for feature in Metal4Feature.allCases {
        let supported = await device.supports(feature)
        print("\(feature): \(supported)")
    }
}

func testMetal4FactoryMethods() async throws {
    let device = try MetalDevice(preference: .metal4IfAvailable)

    guard await device.isMetal4Active else {
        throw XCTSkip("Metal 4 not available")
    }

    let queue = try await device.makeMetal4CommandQueue()
    XCTAssertNotNil(queue)

    let compiler = try await device.makeMetal4Compiler()
    XCTAssertNotNil(compiler)

    let residencySet = try await device.makeResidencySet()
    XCTAssertNotNil(residencySet)
}
```

## GPU Family Reference

| GPU Family | Devices | Key Features |
|------------|---------|--------------|
| `.apple7` | M1, A14 | Base Apple Silicon |
| `.apple8` | M2, A15, A16 | Simdgroup matrix 8x8 |
| `.apple9` | M3, A17 | Placement sparse, ML encoder, simdgroup 16x16 |
| `.metal4` | All iOS 26 | Core Metal 4 features |

## Notes

- GPU family checks may need adjustment based on Xcode 17 headers
- Some limits are estimates - verify with actual hardware
- Consider adding runtime capability queries for dynamic limits
- Environment override is debug-only, don't ship with it forced

## Context Files

When sharing with external agent, include:
1. `00-context/metal4-api-summary.md`
2. `00-context/architecture.md`
3. This task file
4. Current `MetalDevice.swift` source
