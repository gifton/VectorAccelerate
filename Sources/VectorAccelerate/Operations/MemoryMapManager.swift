//
//  MemoryMapManager.swift
//  VectorAccelerate
//
//  Memory-mapped file support for large dataset operations
//

import Foundation
import Metal
import VectorCore

/// Configuration for memory-mapped operations
public struct MemoryMapConfiguration: Sendable {
    public let pageSize: Int
    public let maxCacheSize: Int
    public let prefetchDistance: Int
    public let enableAsyncIO: Bool
    public let compressionEnabled: Bool
    
    public init(
        pageSize: Int = 4096 * 1024,  // 4MB pages
        maxCacheSize: Int = 256 * 1024 * 1024,  // 256MB cache
        prefetchDistance: Int = 2,  // Prefetch 2 pages ahead
        enableAsyncIO: Bool = true,
        compressionEnabled: Bool = false
    ) {
        self.pageSize = pageSize
        self.maxCacheSize = maxCacheSize
        self.prefetchDistance = prefetchDistance
        self.enableAsyncIO = enableAsyncIO
        self.compressionEnabled = compressionEnabled
    }
    
    public static let `default` = MemoryMapConfiguration()
    public static let largeDataset = MemoryMapConfiguration(
        pageSize: 16 * 1024 * 1024,  // 16MB pages
        maxCacheSize: 1024 * 1024 * 1024,  // 1GB cache
        prefetchDistance: 4
    )
}

/// Represents a memory-mapped vector dataset
public struct MappedVectorDataset: Sendable {
    public let fileURL: URL
    public let vectorCount: Int
    public let dimension: Int
    public let dataType: DataType
    
    public enum DataType: String, Sendable {
        case float32 = "f32"
        case float16 = "f16"
        case int8 = "i8"
        case uint8 = "u8"
    }
    
    public var vectorByteSize: Int {
        dimension * dataType.byteSize
    }
    
    public var totalByteSize: Int {
        vectorCount * vectorByteSize
    }
}

extension MappedVectorDataset.DataType {
    var byteSize: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .int8, .uint8: return 1
        }
    }
}

/// Cache entry for memory-mapped pages
private struct CacheEntry {
    let pageIndex: Int
    let data: Data
    let lastAccessTime: Date
}

/// Manager for memory-mapped large dataset operations
public actor MemoryMapManager {
    private let configuration: MemoryMapConfiguration
    private let logger: Logger
    
    // Cache management
    private var cache: [Int: CacheEntry] = [:]
    private var cacheSize: Int = 0
    private let cacheQueue = DispatchQueue(label: "com.vectoraccelerate.mmap.cache", attributes: .concurrent)
    
    // File handles
    private var fileHandles: [URL: FileHandle] = [:]
    
    // Prefetch management
    private var prefetchTasks: [Int: Task<Void, Never>] = [:]
    
    // Performance metrics
    private var cacheHits: Int = 0
    private var cacheMisses: Int = 0
    private var totalReads: Int = 0
    
    // MARK: - Initialization
    
    public init(configuration: MemoryMapConfiguration = .default) {
        self.configuration = configuration
        self.logger = Logger.shared
    }
    
    // MARK: - Dataset Operations
    
    /// Map a vector dataset file for efficient access
    public func mapDataset(at url: URL) async throws -> MappedVectorDataset {
        // Verify file exists
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AccelerationError.fileNotFound(url.path)
        }
        
        // Read metadata from file header
        let handle = try FileHandle(forReadingFrom: url)
        fileHandles[url] = handle
        
        // Read header (assuming simple format: vectorCount, dimension, dataType)
        let headerData = try handle.read(upToCount: 16) ?? Data()
        guard headerData.count == 16 else {
            throw AccelerationError.invalidDataFormat("Invalid header size")
        }
        
        let vectorCount = headerData.withUnsafeBytes { $0.load(as: Int32.self) }
        let dimension = headerData.withUnsafeBytes { $0.load(fromByteOffset: 4, as: Int32.self) }
        let dataTypeRaw = headerData.withUnsafeBytes { $0.load(fromByteOffset: 8, as: UInt32.self) }
        
        let dataType: MappedVectorDataset.DataType
        switch dataTypeRaw {
        case 0: dataType = .float32
        case 1: dataType = .float16
        case 2: dataType = .int8
        case 3: dataType = .uint8
        default:
            throw AccelerationError.invalidDataFormat("Unknown data type: \(dataTypeRaw)")
        }
        
        await logger.info("Mapped dataset: \(vectorCount) vectors of dimension \(dimension)", category: "MemoryMap")
        
        return MappedVectorDataset(
            fileURL: url,
            vectorCount: Int(vectorCount),
            dimension: Int(dimension),
            dataType: dataType
        )
    }
    
    /// Read vectors from memory-mapped dataset
    public func readVectors(
        from dataset: MappedVectorDataset,
        range: Range<Int>
    ) async throws -> [[Float]] {
        let measureToken = await logger.startMeasure("memoryMappedRead")
        measureToken.addMetadata("count", value: "\(range.count)")
        defer { measureToken.end() }
        
        var vectors: [[Float]] = []
        vectors.reserveCapacity(range.count)
        
        // Calculate byte ranges
        let headerSize = 16  // Header size in bytes
        let startByte = headerSize + range.lowerBound * dataset.vectorByteSize
        let endByte = headerSize + range.upperBound * dataset.vectorByteSize
        
        // Process in pages
        let _ = startByte / configuration.pageSize
        let endPage = (endByte - 1) / configuration.pageSize
        
        // Prefetch upcoming pages
        if configuration.enableAsyncIO {
            for i in 1...configuration.prefetchDistance {
                let pageToFetch = endPage + i
                await prefetchPage(pageToFetch, from: dataset)
            }
        }
        
        // Read vectors
        for vectorIndex in range {
            let vector = try await readVector(at: vectorIndex, from: dataset)
            vectors.append(vector)
        }
        
        return vectors
    }
    
    /// Read a single vector from the dataset
    private func readVector(at index: Int, from dataset: MappedVectorDataset) async throws -> [Float] {
        totalReads += 1
        
        let headerSize = 16
        let byteOffset = headerSize + index * dataset.vectorByteSize
        let pageIndex = byteOffset / configuration.pageSize
        
        // Try cache first
        if let cached = cache[pageIndex] {
            cacheHits += 1
            cache[pageIndex] = CacheEntry(
                pageIndex: pageIndex,
                data: cached.data,
                lastAccessTime: Date()
            )
            
            return extractVector(
                from: cached.data,
                offset: byteOffset % configuration.pageSize,
                dimension: dataset.dimension,
                dataType: dataset.dataType
            )
        }
        
        // Cache miss - load page
        cacheMisses += 1
        let pageData = try await loadPage(pageIndex, from: dataset)
        
        return extractVector(
            from: pageData,
            offset: byteOffset % configuration.pageSize,
            dimension: dataset.dimension,
            dataType: dataset.dataType
        )
    }
    
    /// Load a page from disk
    private func loadPage(_ pageIndex: Int, from dataset: MappedVectorDataset) async throws -> Data {
        guard let handle = fileHandles[dataset.fileURL] else {
            throw AccelerationError.fileNotFound(dataset.fileURL.path)
        }
        
        let pageOffset = UInt64(pageIndex * configuration.pageSize)
        let pageSize = min(configuration.pageSize, Int(handle.seekToEndOfFile()) - Int(pageOffset))
        
        try handle.seek(toOffset: pageOffset)
        guard let data = try handle.read(upToCount: pageSize) else {
            throw AccelerationError.invalidDataFormat("Failed to read page \(pageIndex)")
        }
        
        // Update cache
        await updateCache(pageIndex: pageIndex, data: data)
        
        return data
    }
    
    /// Update cache with new page
    private func updateCache(pageIndex: Int, data: Data) async {
        // Check if we need to evict
        if cacheSize + data.count > configuration.maxCacheSize {
            await evictLRUPages(targetSize: configuration.maxCacheSize - data.count)
        }
        
        // Add to cache
        cache[pageIndex] = CacheEntry(
            pageIndex: pageIndex,
            data: data,
            lastAccessTime: Date()
        )
        cacheSize += data.count
    }
    
    /// Evict least recently used pages
    private func evictLRUPages(targetSize: Int) async {
        // Sort by last access time
        let sortedEntries = cache.values.sorted { $0.lastAccessTime < $1.lastAccessTime }
        
        for entry in sortedEntries {
            if cacheSize <= targetSize { break }
            
            cache.removeValue(forKey: entry.pageIndex)
            cacheSize -= entry.data.count
            
            await logger.debug("Evicted page \(entry.pageIndex) from cache", category: "MemoryMap")
        }
    }
    
    /// Prefetch a page asynchronously
    private func prefetchPage(_ pageIndex: Int, from dataset: MappedVectorDataset) async {
        // Skip if already cached or being fetched
        if cache[pageIndex] != nil || prefetchTasks[pageIndex] != nil {
            return
        }
        
        // Start prefetch task
        let task = Task {
            do {
                _ = try await loadPage(pageIndex, from: dataset)
                await logger.verbose("Prefetched page \(pageIndex)", category: "MemoryMap")
            } catch {
                await logger.warning("Prefetch failed for page \(pageIndex): \(error)", category: "MemoryMap")
            }
        }
        
        prefetchTasks[pageIndex] = task
        
        // Clean up when done
        Task {
            await task.value
            prefetchTasks.removeValue(forKey: pageIndex)
        }
    }
    
    /// Extract vector from raw data
    private func extractVector(
        from data: Data,
        offset: Int,
        dimension: Int,
        dataType: MappedVectorDataset.DataType
    ) -> [Float] {
        var result = [Float](repeating: 0, count: dimension)
        
        data.withUnsafeBytes { bytes in
            let basePtr = bytes.baseAddress!.advanced(by: offset)
            
            switch dataType {
            case .float32:
                let floatPtr = basePtr.bindMemory(to: Float.self, capacity: dimension)
                for i in 0..<dimension {
                    result[i] = floatPtr[i]
                }
                
            case .float16:
                // Convert half-precision to full precision
                let halfPtr = basePtr.bindMemory(to: UInt16.self, capacity: dimension)
                for i in 0..<dimension {
                    result[i] = Float(bitPattern: halfToFloat(halfPtr[i]))
                }
                
            case .int8:
                let intPtr = basePtr.bindMemory(to: Int8.self, capacity: dimension)
                for i in 0..<dimension {
                    result[i] = Float(intPtr[i]) / 127.0  // Normalize to [-1, 1]
                }
                
            case .uint8:
                let uintPtr = basePtr.bindMemory(to: UInt8.self, capacity: dimension)
                for i in 0..<dimension {
                    result[i] = Float(uintPtr[i]) / 255.0  // Normalize to [0, 1]
                }
            }
        }
        
        return result
    }
    
    // MARK: - Streaming Operations
    
    /// Stream process vectors from a large dataset
    public func streamProcess<T: Sendable>(
        dataset: MappedVectorDataset,
        batchSize: Int,
        operation: @escaping ([[Float]]) async throws -> T
    ) async throws -> AsyncStream<T> {
        AsyncStream { continuation in
            Task {
                do {
                    for i in stride(from: 0, to: dataset.vectorCount, by: batchSize) {
                        let endIndex = min(i + batchSize, dataset.vectorCount)
                        let vectors = try await readVectors(from: dataset, range: i..<endIndex)
                        let result = try await operation(vectors)
                        continuation.yield(result)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }
    
    /// Compute distances for memory-mapped dataset
    public func computeDistances(
        dataset: MappedVectorDataset,
        query: [Float],
        metric: DistanceMetric = .euclidean,
        topK: Int? = nil
    ) async throws -> [(index: Int, distance: Float)] {
        let measureToken = await logger.startMeasure("memoryMappedDistance")
        defer { measureToken.end() }
        
        var results: [(index: Int, distance: Float)] = []
        let batchSize = 1000  // Process in batches
        
        // Process dataset in chunks
        for i in stride(from: 0, to: dataset.vectorCount, by: batchSize) {
            let endIndex = min(i + batchSize, dataset.vectorCount)
            let vectors = try await readVectors(from: dataset, range: i..<endIndex)
            
            // Compute distances for batch
            for (offset, vector) in vectors.enumerated() {
                let distance = computeDistance(vector, query, metric: metric)
                results.append((index: i + offset, distance: distance))
            }
            
            // Keep only top-K if specified
            if let k = topK {
                results.sort { $0.distance < $1.distance }
                results = Array(results.prefix(k))
            }
        }
        
        // Final sort
        results.sort { $0.distance < $1.distance }
        
        if let k = topK {
            return Array(results.prefix(k))
        }
        
        return results
    }
    
    private func computeDistance(_ a: [Float], _ b: [Float], metric: DistanceMetric) -> Float {
        switch metric {
        case .euclidean:
            var sum: Float = 0
            for i in 0..<a.count {
                let diff = a[i] - b[i]
                sum += diff * diff
            }
            return sqrt(sum)
            
        case .cosine:
            var dot: Float = 0
            var normA: Float = 0
            var normB: Float = 0
            for i in 0..<a.count {
                dot += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }
            return 1.0 - (dot / (sqrt(normA) * sqrt(normB)))
            
        case .manhattan:
            var sum: Float = 0
            for i in 0..<a.count {
                sum += abs(a[i] - b[i])
            }
            return sum
        }
    }
    
    // MARK: - Performance Metrics
    
    public func getCacheStatistics() -> (
        hitRate: Float,
        cacheSize: Int,
        pagesCached: Int,
        totalReads: Int
    ) {
        let hitRate = totalReads > 0 ? Float(cacheHits) / Float(totalReads) : 0
        return (hitRate, cacheSize, cache.count, totalReads)
    }
    
    public func clearCache() async {
        cache.removeAll()
        cacheSize = 0
        await logger.info("Cache cleared", category: "MemoryMap")
    }
    
    public func closeDataset(_ dataset: MappedVectorDataset) async throws {
        if let handle = fileHandles.removeValue(forKey: dataset.fileURL) {
            try handle.close()
        }
        
        // Remove cached pages for this dataset
        cache.removeAll()
        cacheSize = 0
    }
    
    // MARK: - Helper Functions
    
    /// Convert half-precision float to full precision
    private func halfToFloat(_ half: UInt16) -> UInt32 {
        let sign = UInt32((half & 0x8000) >> 15)
        let exponent = UInt32((half & 0x7C00) >> 10)
        let mantissa = UInt32(half & 0x03FF)
        
        if exponent == 0 {
            // Subnormal or zero
            if mantissa == 0 {
                return sign << 31
            } else {
                // Subnormal - normalize
                var e: UInt32 = 0
                var m = mantissa
                while (m & 0x400) == 0 {
                    m <<= 1
                    e += 1
                }
                m &= ~0x400
                return (sign << 31) | ((127 - 15 - e) << 23) | (m << 13)
            }
        } else if exponent == 31 {
            // Infinity or NaN
            if mantissa == 0 {
                return (sign << 31) | 0x7F800000
            } else {
                return (sign << 31) | 0x7F800000 | (mantissa << 13)
            }
        } else {
            // Normalized
            return (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13)
        }
    }
}

// MARK: - Dataset Creation Utilities

public extension MemoryMapManager {
    
    /// Create a memory-mapped dataset file from vectors
    static func createDatasetFile(
        vectors: [[Float]],
        at url: URL,
        dataType: MappedVectorDataset.DataType = .float32
    ) throws {
        guard !vectors.isEmpty else {
            throw AccelerationError.invalidDataFormat("Empty vector set")
        }
        
        let dimension = vectors[0].count
        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw AccelerationError.dimensionMismatch(expected: dimension, actual: -1)
        }
        
        // Create file handle
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        
        // Write header
        var header = Data()
        var vectorCount = Int32(vectors.count)
        var dim = Int32(dimension)
        var dtype = UInt32(dataType == .float32 ? 0 : dataType == .float16 ? 1 : dataType == .int8 ? 2 : 3)
        var reserved = UInt32(0)
        
        header.append(Data(bytes: &vectorCount, count: 4))
        header.append(Data(bytes: &dim, count: 4))
        header.append(Data(bytes: &dtype, count: 4))
        header.append(Data(bytes: &reserved, count: 4))
        
        handle.write(header)
        
        // Write vectors
        for vector in vectors {
            let data: Data
            
            switch dataType {
            case .float32:
                data = vector.withUnsafeBytes { Data($0) }
                
            case .float16:
                var halfData = Data()
                for value in vector {
                    let half = floatToHalf(value.bitPattern)
                    var halfValue = half
                    halfData.append(Data(bytes: &halfValue, count: 2))
                }
                data = halfData
                
            case .int8:
                let intValues = vector.map { Int8(max(-127, min(127, $0 * 127))) }
                data = intValues.withUnsafeBytes { Data($0) }
                
            case .uint8:
                let uintValues = vector.map { UInt8(max(0, min(255, $0 * 255))) }
                data = uintValues.withUnsafeBytes { Data($0) }
            }
            
            handle.write(data)
        }
    }
    
    /// Convert full precision float to half precision
    private static func floatToHalf(_ float: UInt32) -> UInt16 {
        let sign = (float >> 31) & 0x1
        let exponent = (float >> 23) & 0xFF
        let mantissa = float & 0x7FFFFF
        
        if exponent == 0 {
            // Zero or subnormal
            return UInt16(sign << 15)
        } else if exponent == 0xFF {
            // Infinity or NaN
            if mantissa == 0 {
                return UInt16((sign << 15) | 0x7C00)
            } else {
                return UInt16((sign << 15) | 0x7C00 | (mantissa >> 13))
            }
        } else {
            // Normalized
            let newExp = Int(exponent) - 127 + 15
            if newExp <= 0 {
                // Underflow to zero
                return UInt16(sign << 15)
            } else if newExp >= 31 {
                // Overflow to infinity
                return UInt16((sign << 15) | 0x7C00)
            } else {
                let signBits = UInt16(sign << 15)
                let expBits = UInt16(newExp) << 10
                let mantissaBits = UInt16(mantissa >> 13)
                return signBits | expBits | mantissaBits
            }
        }
    }
}

// MARK: - Error Extensions

extension AccelerationError {
    static func fileNotFound(_ path: String) -> AccelerationError {
        .unsupportedOperation("File not found: \(path)")
    }
    
    static func invalidDataFormat(_ reason: String) -> AccelerationError {
        .unsupportedOperation("Invalid data format: \(reason)")
    }
}
