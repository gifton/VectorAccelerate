//
//  SIMDOptimizer.swift
//  VectorAccelerate
//
//  Adaptive SIMD width detection and optimization
//

import Foundation
import simd

/// Platform-specific SIMD capabilities
public struct SIMDCapabilities: Sendable {
    public let maxVectorWidth: Int
    public let supportsAVX: Bool
    public let supportsAVX512: Bool
    public let supportsNEON: Bool
    public let supportsAMX: Bool
    public let optimalWidth: Int
    public let l1CacheSize: Int
    public let l2CacheSize: Int
    
    /// Detect current platform capabilities
    public static func detect() -> SIMDCapabilities {
        #if arch(x86_64)
        return detectX86Capabilities()
        #elseif arch(arm64)
        return detectARMCapabilities()
        #else
        return SIMDCapabilities(
            maxVectorWidth: 4,
            supportsAVX: false,
            supportsAVX512: false,
            supportsNEON: false,
            supportsAMX: false,
            optimalWidth: 4,
            l1CacheSize: 32 * 1024,
            l2CacheSize: 256 * 1024
        )
        #endif
    }
    
    #if arch(x86_64)
    private static func detectX86Capabilities() -> SIMDCapabilities {
        // Check CPU features using sysctl
        var hasAVX = false
        var hasAVX512 = false
        var hasAMX = false
        
        var size = 0
        sysctlbyname("hw.optional.avx1_0", nil, &size, nil, 0)
        if size > 0 {
            var value: Int32 = 0
            sysctlbyname("hw.optional.avx1_0", &value, &size, nil, 0)
            hasAVX = value != 0
        }
        
        sysctlbyname("hw.optional.avx512f", nil, &size, nil, 0)
        if size > 0 {
            var value: Int32 = 0
            sysctlbyname("hw.optional.avx512f", &value, &size, nil, 0)
            hasAVX512 = value != 0
        }
        
        // Determine optimal width
        let optimalWidth: Int
        let maxWidth: Int
        if hasAVX512 {
            maxWidth = 16  // 512 bits / 32 bits per float
            optimalWidth = 16
        } else if hasAVX {
            maxWidth = 8   // 256 bits / 32 bits per float
            optimalWidth = 8
        } else {
            maxWidth = 4   // 128 bits / 32 bits per float (SSE)
            optimalWidth = 4
        }
        
        // Get cache sizes
        var l1Size: Int64 = 0
        var l2Size: Int64 = 0
        size = MemoryLayout<Int64>.size
        sysctlbyname("hw.l1dcachesize", &l1Size, &size, nil, 0)
        sysctlbyname("hw.l2cachesize", &l2Size, &size, nil, 0)
        
        return SIMDCapabilities(
            maxVectorWidth: maxWidth,
            supportsAVX: hasAVX,
            supportsAVX512: hasAVX512,
            supportsNEON: false,
            supportsAMX: hasAMX,
            optimalWidth: optimalWidth,
            l1CacheSize: Int(l1Size),
            l2CacheSize: Int(l2Size)
        )
    }
    #endif
    
    #if arch(arm64)
    private static func detectARMCapabilities() -> SIMDCapabilities {
        // ARM64 always has NEON
        var cacheLineSize: Int64 = 0
        var size = MemoryLayout<Int64>.size
        sysctlbyname("hw.cachelinesize", &cacheLineSize, &size, nil, 0)
        
        // Get cache sizes
        var l1Size: Int64 = 0
        var l2Size: Int64 = 0
        sysctlbyname("hw.l1dcachesize", &l1Size, &size, nil, 0)
        sysctlbyname("hw.l2cachesize", &l2Size, &size, nil, 0)
        
        // Check for AMX on Apple Silicon
        // Note: macOS 26+ is required (see Package.swift platform requirements)
        var hasAMX = false
        #if os(macOS)
        // Apple Silicon M1 and later have AMX
        var cpuFamily: Int32 = 0
        size = MemoryLayout<Int32>.size
        sysctlbyname("hw.cpufamily", &cpuFamily, &size, nil, 0)

        // Apple Silicon CPU families (using UInt32 to avoid overflow)
        let appleSiliconFamilies: Set<UInt32> = [
            0x1b588bb3,  // M1
            0xda33d83d,  // M2
            0xfa33415e,  // M3
        ]
        hasAMX = appleSiliconFamilies.contains(UInt32(bitPattern: cpuFamily))
        #endif
        
        return SIMDCapabilities(
            maxVectorWidth: 4,  // 128 bits / 32 bits per float (NEON)
            supportsAVX: false,
            supportsAVX512: false,
            supportsNEON: true,
            supportsAMX: hasAMX,
            optimalWidth: 4,
            l1CacheSize: l1Size > 0 ? Int(l1Size) : 64 * 1024,
            l2CacheSize: l2Size > 0 ? Int(l2Size) : 4 * 1024 * 1024
        )
    }
    #endif
}

/// Adaptive SIMD optimizer that selects optimal vector width
public final class SIMDOptimizer: @unchecked Sendable {
    public let capabilities: SIMDCapabilities
    private let lock = NSLock()
    
    // Performance measurements for adaptive tuning
    private var performanceHistory: [Int: Double] = [:]  // width -> avg time
    
    public init() {
        self.capabilities = SIMDCapabilities.detect()
    }
    
    /// Select optimal SIMD width for given data size
    public func selectOptimalWidth(for dataSize: Int) -> Int {
        // For small data, use smaller width to avoid overhead
        if dataSize < 64 {
            return min(4, capabilities.optimalWidth)
        }
        
        // For medium data, use optimal width
        if dataSize < 10000 {
            return capabilities.optimalWidth
        }
        
        // For large data, consider cache effects
        let bytesPerVector = dataSize * MemoryLayout<Float>.stride
        if bytesPerVector > capabilities.l2CacheSize {
            // Use smaller width for better cache locality
            return min(8, capabilities.optimalWidth)
        }
        
        return capabilities.optimalWidth
    }
    
    /// Generic SIMD operation with adaptive width
    public func adaptiveDotProduct(_ a: [Float], _ b: [Float]) -> Float {
        let width = selectOptimalWidth(for: a.count)
        
        switch width {
        case 16:
            return simd16DotProduct(a, b)
        case 8:
            return simd8DotProduct(a, b)
        case 4:
            return simd4DotProduct(a, b)
        default:
            return scalarDotProduct(a, b)
        }
    }
    
    private func simd16DotProduct(_ a: [Float], _ b: [Float]) -> Float {
        // Would use SIMD16 if available (AVX-512)
        // For now, fall back to SIMD8
        return simd8DotProduct(a, b)
    }
    
    private func simd8DotProduct(_ a: [Float], _ b: [Float]) -> Float {
        let count = a.count
        let vectorCount = count / 8
        var sum: Float = 0
        
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                var vectorSum = SIMD8<Float>.zero
                
                for i in 0..<vectorCount {
                    let offset = i * 8
                    var va = SIMD8<Float>.zero
                    var vb = SIMD8<Float>.zero
                    
                    for j in 0..<8 {
                        va[j] = aPtr[offset + j]
                        vb[j] = bPtr[offset + j]
                    }
                    
                    vectorSum += va * vb
                }
                
                // Reduce vector sum
                for i in 0..<8 {
                    sum += vectorSum[i]
                }
                
                // Handle remainder
                for i in (vectorCount * 8)..<count {
                    sum += aPtr[i] * bPtr[i]
                }
            }
        }
        
        return sum
    }
    
    private func simd4DotProduct(_ a: [Float], _ b: [Float]) -> Float {
        let count = a.count
        let vectorCount = count / 4
        var sum: Float = 0
        
        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                var vectorSum = SIMD4<Float>.zero
                
                for i in 0..<vectorCount {
                    let offset = i * 4
                    var va = SIMD4<Float>.zero
                    var vb = SIMD4<Float>.zero
                    
                    for j in 0..<4 {
                        va[j] = aPtr[offset + j]
                        vb[j] = bPtr[offset + j]
                    }
                    
                    vectorSum += va * vb
                }
                
                // Reduce vector sum
                for i in 0..<4 {
                    sum += vectorSum[i]
                }
                
                // Handle remainder
                for i in (vectorCount * 4)..<count {
                    sum += aPtr[i] * bPtr[i]
                }
            }
        }
        
        return sum
    }
    
    private func scalarDotProduct(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            sum += a[i] * b[i]
        }
        return sum
    }
    
    /// Benchmark different widths and select optimal
    public func benchmarkAndOptimize(dataSize: Int) -> Int {
        let testData1 = [Float](repeating: 1.0, count: dataSize)
        let testData2 = [Float](repeating: 2.0, count: dataSize)
        
        var results: [(width: Int, time: Double)] = []
        
        // Test different widths
        for width in [4, 8, capabilities.optimalWidth] {
            let start = CFAbsoluteTimeGetCurrent()
            
            for _ in 0..<100 {
                switch width {
                case 4:
                    _ = simd4DotProduct(testData1, testData2)
                case 8:
                    _ = simd8DotProduct(testData1, testData2)
                default:
                    _ = adaptiveDotProduct(testData1, testData2)
                }
            }
            
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            results.append((width, elapsed))
        }
        
        // Find optimal width
        let optimal = results.min { $0.time < $1.time }!
        
        // Update performance history
        lock.lock()
        performanceHistory[dataSize] = optimal.time
        lock.unlock()
        
        return optimal.width
    }
    
    /// Get performance statistics
    public func getPerformanceStats() -> String {
        """
        SIMD Capabilities:
        - Max Vector Width: \(capabilities.maxVectorWidth) floats
        - Optimal Width: \(capabilities.optimalWidth) floats
        - AVX: \(capabilities.supportsAVX)
        - AVX-512: \(capabilities.supportsAVX512)
        - NEON: \(capabilities.supportsNEON)
        - AMX: \(capabilities.supportsAMX)
        - L1 Cache: \(capabilities.l1CacheSize / 1024) KB
        - L2 Cache: \(capabilities.l2CacheSize / 1024) KB
        """
    }
}