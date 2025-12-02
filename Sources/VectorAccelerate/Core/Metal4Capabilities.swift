//
//  Metal4Capabilities.swift
//  VectorAccelerate
//
//  Granular Metal 4 feature detection for capability-based feature gating
//

import Foundation
@preconcurrency import Metal

/// Granular Metal 4 feature detection
///
/// Use this to check individual Metal 4 features rather than a blanket
/// "supports Metal 4" check. Features may vary by GPU family.
///
/// Example:
/// ```swift
/// let caps = Metal4Capabilities(device: device)
/// if caps.supportsMetal4Core {
///     // Use Metal 4 APIs
/// }
/// if caps.supportsSimdgroupMatrix {
///     // Use simdgroup_matrix operations
/// }
/// ```
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
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

    // MARK: - Device Info

    /// GPU family identifier for logging/debugging
    public let gpuFamilyDescription: String

    // MARK: - Initialization

    /// Initialize from MTLDevice
    public init(device: any MTLDevice) {
        // Core Metal 4 - requires metal4 family
        // Note: .metal4 family check - adjust based on actual Xcode 17 headers
        // For now, detect based on OS version since we target Metal 4 exclusively
        _ = device.supportsFamily(.apple7) // Placeholder for future metal4 family check
        self.supportsMetal4Core = true  // We're only running on Metal 4+ OSes
        self.supportsMTL4Compiler = true
        self.supportsArgumentTables = true
        self.supportsResidencySets = true

        // Advanced memory - M3+ (apple9)
        self.supportsPlacementHeaps = device.supportsFamily(.apple9)
        self.supportsPlacementSparseBuffers = device.supportsFamily(.apple9)

        // ML features - available on Metal 4
        self.supportsMLTensor = true
        self.supportsMachineLearningEncoder = device.supportsFamily(.apple9)

        // Compute features
        self.supportsAdvancedBarriers = true

        // Simdgroup matrix support
        if device.supportsFamily(.apple9) {
            self.supportsSimdgroupMatrix = true
            self.maxSimdgroupMatrixDimension = 16
        } else if device.supportsFamily(.apple8) {
            self.supportsSimdgroupMatrix = true
            self.maxSimdgroupMatrixDimension = 8
        } else if device.supportsFamily(.apple7) {
            self.supportsSimdgroupMatrix = true
            self.maxSimdgroupMatrixDimension = 8
        } else {
            self.supportsSimdgroupMatrix = false
            self.maxSimdgroupMatrixDimension = 0
        }

        // Limits - these may need runtime query from actual Metal 4 APIs
        if device.supportsFamily(.apple9) {
            self.maxArgumentTableBuffers = 64
            self.maxArgumentTableTextures = 128
            self.maxResidencySetCapacity = 4096
        } else if device.supportsFamily(.apple8) {
            self.maxArgumentTableBuffers = 31
            self.maxArgumentTableTextures = 64
            self.maxResidencySetCapacity = 2048
        } else {
            self.maxArgumentTableBuffers = 31
            self.maxArgumentTableTextures = 31
            self.maxResidencySetCapacity = 1024
        }

        // GPU family description for debugging
        if device.supportsFamily(.apple9) {
            self.gpuFamilyDescription = "Apple9 (M3/A17)"
        } else if device.supportsFamily(.apple8) {
            self.gpuFamilyDescription = "Apple8 (M2/A15/A16)"
        } else if device.supportsFamily(.apple7) {
            self.gpuFamilyDescription = "Apple7 (M1/A14)"
        } else {
            self.gpuFamilyDescription = "Unknown"
        }
    }

    /// Create capabilities indicating no Metal 4 support
    /// Used for fallback scenarios or testing
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
            maxResidencySetCapacity: 0,
            gpuFamilyDescription: "Unsupported"
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
        maxResidencySetCapacity: Int,
        gpuFamilyDescription: String
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
        self.gpuFamilyDescription = gpuFamilyDescription
    }
}

// MARK: - CustomStringConvertible

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4Capabilities: CustomStringConvertible {
    public var description: String {
        """
        Metal4Capabilities(\(gpuFamilyDescription)):
          Core: \(supportsMetal4Core)
          ArgumentTables: \(supportsArgumentTables) (max: \(maxArgumentTableBuffers) buffers)
          ResidencySets: \(supportsResidencySets) (max: \(maxResidencySetCapacity))
          PlacementSparse: \(supportsPlacementSparseBuffers)
          MLTensor: \(supportsMLTensor)
          SimdgroupMatrix: \(supportsSimdgroupMatrix) (max: \(maxSimdgroupMatrixDimension)x\(maxSimdgroupMatrixDimension))
        """
    }
}

// MARK: - Feature Enum

/// Checkable Metal 4 features for runtime queries
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum Metal4Feature: String, Sendable, CaseIterable, Codable {
    case argumentTables
    case residencySets
    case placementSparse
    case mlTensor
    case simdgroupMatrix
    case advancedBarriers
    case machineLearningEncoder
}

// MARK: - Feature Checking

@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
extension Metal4Capabilities {
    /// Check if a specific feature is available
    public func supports(_ feature: Metal4Feature) -> Bool {
        switch feature {
        case .argumentTables: return supportsArgumentTables
        case .residencySets: return supportsResidencySets
        case .placementSparse: return supportsPlacementSparseBuffers
        case .mlTensor: return supportsMLTensor
        case .simdgroupMatrix: return supportsSimdgroupMatrix
        case .advancedBarriers: return supportsAdvancedBarriers
        case .machineLearningEncoder: return supportsMachineLearningEncoder
        }
    }

    /// Get list of all supported features
    public var supportedFeatures: [Metal4Feature] {
        Metal4Feature.allCases.filter { supports($0) }
    }

    /// Get list of unsupported features
    public var unsupportedFeatures: [Metal4Feature] {
        Metal4Feature.allCases.filter { !supports($0) }
    }
}

// MARK: - Backend Selection

/// Backend selection for VectorAccelerate operations
@available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 3.0, *)
public enum GPUBackend: Sendable {
    /// Metal 4 backend with specific capabilities
    case metal4(any MTLDevice, Metal4Capabilities)

    /// The underlying device
    public var device: any MTLDevice {
        switch self {
        case .metal4(let device, _): return device
        }
    }

    /// Metal 4 capabilities
    public var capabilities: Metal4Capabilities {
        switch self {
        case .metal4(_, let caps): return caps
        }
    }

    /// Select recommended backend for device
    public static func create(for device: any MTLDevice) -> GPUBackend {
        let caps = Metal4Capabilities(device: device)
        return .metal4(device, caps)
    }
}

// MARK: - Backend Preference

/// User preference for backend selection
public enum BackendPreference: String, Sendable, Codable, CaseIterable {
    /// Use Metal 4 (only option for iOS 26+)
    case metal4

    /// Auto-select (same as metal4 on supported platforms)
    case automatic
}
