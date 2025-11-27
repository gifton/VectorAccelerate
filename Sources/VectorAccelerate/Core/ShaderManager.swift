//
//  ShaderManager.swift
//  VectorAccelerate
//
//  Manages shader compilation, caching, and pipeline state creation
//

import Foundation
@preconcurrency import Metal
import VectorCore

/// Manages Metal shader compilation and pipeline state caching
public actor ShaderManager {
    private let device: MetalDevice
    private var pipelineCache: [String: any MTLComputePipelineState] = [:]
    private var functionCache: [String: any MTLFunction] = [:]
    private var libraryCache: [String: any MTLLibrary] = [:]
    
    // Shader source management
    private var embeddedShaderSource: String?
    private var defaultLibrary: (any MTLLibrary)?
    
    // Performance tracking
    private var compilationTime: TimeInterval = 0
    private var compilationCount: Int = 0
    
    // MARK: - Initialization
    
    public init(device: MetalDevice) async throws {
        self.device = device
        
        // Try to get default library from device (includes compiled Metal files)
        do {
            self.defaultLibrary = try await device.getDefaultLibrary()
            self.embeddedShaderSource = nil
        } catch {
            // For tests, compile a minimal set of shaders
            // These are ONLY the ones not in the Metal files to avoid duplicates
            self.embeddedShaderSource = Self.getMinimalTestShaders()
            if !embeddedShaderSource!.isEmpty {
                self.defaultLibrary = try await compileLibrary(source: embeddedShaderSource!, label: "VectorAccelerate.Test")
            } else {
                throw VectorError.shaderCompilationFailed("Could not load Metal shaders")
            }
        }
    }
    
    // MARK: - Shader Compilation
    
    /// Compile shader library from source
    @preconcurrency
    public func compileLibrary(source: String, label: String? = nil) async throws -> any MTLLibrary {
        // Check cache
        let cacheKey = "lib_\(source.hashValue)"
        if let cached = libraryCache[cacheKey] {
            return cached
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Compile library
        let library = try await device.makeLibrary(source: source)
        
        if let label = label {
            library.label = label
        }
        
        // Update metrics
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        compilationTime += elapsed
        compilationCount += 1
        
        // Cache library
        libraryCache[cacheKey] = library
        
        return library
    }
    
    /// Get or compile function from library
    @preconcurrency
    public func getFunction(name: String, library: (any MTLLibrary)? = nil) async throws -> any MTLFunction {
        // Check cache
        let lib = library ?? defaultLibrary
        let cacheKey = "\(name)_\(ObjectIdentifier(lib ?? defaultLibrary!).hashValue)"
        
        if let cached = functionCache[cacheKey] {
            return cached
        }
        
        guard let lib = lib else {
            throw VectorError.shaderNotFound(name: "No library available")
        }
        
        guard let function = lib.makeFunction(name: name) else {
            throw VectorError.shaderNotFound(name: name)
        }
        
        // Cache function
        functionCache[cacheKey] = function
        
        return function
    }
    
    /// Get or create compute pipeline state
    @preconcurrency
    public func getPipelineState(functionName: String, library: (any MTLLibrary)? = nil) async throws -> any MTLComputePipelineState {
        // Check cache
        let cacheKey = functionName
        if let cached = pipelineCache[cacheKey] {
            return cached
        }
        
        // Get function
        let function = try await getFunction(name: functionName, library: library)
        
        // Create pipeline state
        let pipelineState = try await device.makeComputePipelineState(function: function)
        
        // Cache pipeline state
        pipelineCache[cacheKey] = pipelineState
        
        return pipelineState
    }
    
    /// Create pipeline with custom configuration
    @preconcurrency
    public func createPipeline(
        functionName: String,
        constantValues: MTLFunctionConstantValues? = nil,
        library: (any MTLLibrary)? = nil
    ) async throws -> any MTLComputePipelineState {
        let lib = library ?? defaultLibrary
        
        guard let lib = lib else {
            throw VectorError.shaderNotFound(name: "No library available")
        }
        
        // Create function with constants if provided
        let function: any MTLFunction
        if let constantValues = constantValues {
            function = try await lib.makeFunction(
                name: functionName,
                constantValues: constantValues
            )
        } else {
            guard let fn = lib.makeFunction(name: functionName) else {
                throw VectorError.shaderNotFound(name: functionName)
            }
            function = fn
        }
        
        // Create pipeline state
        return try await device.makeComputePipelineState(function: function)
    }
    
    // MARK: - Shader Source Management
    
    /// Get minimal test shaders embedded directly for test execution
    private static func getMinimalTestShaders() -> String {
        // For tests, we embed minimal shader implementations directly
        // This ensures tests can run without file system dependencies
        return """
        #include <metal_stdlib>
        using namespace metal;
        
        constant float EPSILON = 1e-7f;
        
        // === Basic Distance Operations ===
        
        kernel void euclideanDistance(
            device const float* vectorA [[buffer(0)]],
            device const float* vectorB [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float partialSums[256];
            
            float sum = 0.0f;
            for (uint i = tid; i < dimension; i += tgSize) {
                float diff = vectorA[i] - vectorB[i];
                sum += diff * diff;
            }
            
            partialSums[tid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    partialSums[tid] += partialSums[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            if (tid == 0) {
                result[0] = sqrt(partialSums[0]);
            }
        }
        
        kernel void cosineDistance(
            device const float* vectorA [[buffer(0)]],
            device const float* vectorB [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float dotProducts[256];
            threadgroup float normA[256];
            threadgroup float normB[256];
            
            float localDot = 0.0f;
            float localNormA = 0.0f;
            float localNormB = 0.0f;
            
            for (uint i = tid; i < dimension; i += tgSize) {
                float a = vectorA[i];
                float b = vectorB[i];
                localDot += a * b;
                localNormA += a * a;
                localNormB += b * b;
            }
            
            dotProducts[tid] = localDot;
            normA[tid] = localNormA;
            normB[tid] = localNormB;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    dotProducts[tid] += dotProducts[tid + stride];
                    normA[tid] += normA[tid + stride];
                    normB[tid] += normB[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            if (tid == 0) {
                float dot = dotProducts[0];
                float magA = sqrt(normA[0]);
                float magB = sqrt(normB[0]);
                
                if (magA > EPSILON && magB > EPSILON) {
                    float cosineSim = dot / (magA * magB);
                    result[0] = 1.0f - cosineSim;
                } else {
                    result[0] = 1.0f;
                }
            }
        }
        
        kernel void dotProduct(
            device const float* vectorA [[buffer(0)]],
            device const float* vectorB [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float partialSums[256];
            
            float sum = 0.0f;
            for (uint i = tid; i < dimension; i += tgSize) {
                sum += vectorA[i] * vectorB[i];
            }
            
            partialSums[tid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
                if (tid < stride) {
                    partialSums[tid] += partialSums[tid + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            if (tid == 0) {
                result[0] = partialSums[0];
            }
        }
        
        kernel void normalizeVectors(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& dimension [[buffer(2)]],
            uint tid [[thread_position_in_grid]],
            uint threadId [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float partialSums[256];
            
            float localSum = 0.0f;
            for (uint i = threadId; i < dimension; i += tgSize) {
                float val = input[i];
                localSum += val * val;
            }
            
            partialSums[threadId] = localSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
                if (threadId < stride) {
                    partialSums[threadId] += partialSums[threadId + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            threadgroup float magnitude;
            if (threadId == 0) {
                magnitude = sqrt(partialSums[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (tid < dimension) {
                if (magnitude > EPSILON) {
                    output[tid] = input[tid] / magnitude;
                } else {
                    output[tid] = input[tid];
                }
            }
        }
        
        kernel void vectorNormalize(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& dimension [[buffer(2)]],
            uint tid [[thread_position_in_grid]],
            uint threadId [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float partialSums[256];
            
            float localSum = 0.0f;
            for (uint i = threadId; i < dimension; i += tgSize) {
                float val = input[i];
                localSum += val * val;
            }
            
            partialSums[threadId] = localSum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
                if (threadId < stride) {
                    partialSums[threadId] += partialSums[threadId + stride];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            threadgroup float magnitude;
            if (threadId == 0) {
                magnitude = sqrt(partialSums[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (tid < dimension) {
                if (magnitude > EPSILON) {
                    output[tid] = input[tid] / magnitude;
                } else {
                    output[tid] = input[tid];
                }
            }
        }
        
        kernel void vectorScale(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant float& scalar [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= dimension) return;
            output[tid] = input[tid] * scalar;
        }
        
        kernel void matrixVectorMultiply(
            device const float* matrix [[buffer(0)]],
            device const float* vector [[buffer(1)]],
            device float* output [[buffer(2)]],
            constant uint& rows [[buffer(3)]],
            constant uint& cols [[buffer(4)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= rows) return;
            
            float sum = 0.0f;
            uint rowOffset = tid * cols;
            
            for (uint i = 0; i < cols; i++) {
                sum += matrix[rowOffset + i] * vector[i];
            }
            
            output[tid] = sum;
        }
        
        kernel void batchEuclideanDistance(
            device const float* query [[buffer(0)]],
            device const float* database [[buffer(1)]],
            device float* distances [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            constant uint& numDatabase [[buffer(4)]],
            uint2 id [[thread_position_in_grid]]
        ) {
            uint dbIdx = id.x;
            if (dbIdx >= numDatabase) return;
            
            float sum = 0.0f;
            uint dbOffset = dbIdx * dimension;
            
            for (uint i = 0; i < dimension; i++) {
                float diff = query[i] - database[dbOffset + i];
                sum += diff * diff;
            }
            
            distances[dbIdx] = sqrt(sum);
        }
        
        // === Matrix Operations ===
        
        kernel void matrixMatrixMultiply(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float* C [[buffer(2)]],
            constant uint3& dims [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            const uint M = dims.x;
            const uint K = dims.y;
            const uint N = dims.z;
            
            const uint row = gid.y;
            const uint col = gid.x;
            
            if (row >= M || col >= N) return;
            
            float sum = 0.0f;
            for (uint k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            
            C[row * N + col] = sum;
        }
        
        kernel void transposeMatrix(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint2& dims [[buffer(2)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            const uint rows = dims.x;
            const uint cols = dims.y;
            
            const uint row = gid.y;
            const uint col = gid.x;
            
            if (row >= rows || col >= cols) return;
            
            output[col * rows + row] = input[row * cols + col];
        }
        
        // === Quantization Operations ===
        
        kernel void quantizeFloat32ToInt8(
            device const float* input [[buffer(0)]],
            device int8_t* output [[buffer(1)]],
            constant float& scale [[buffer(2)]],
            constant float& zeroPoint [[buffer(3)]],
            constant uint& count [[buffer(4)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= count) return;
            
            float value = input[tid];
            float quantized = round(value / scale + zeroPoint);
            
            quantized = clamp(quantized, -128.0f, 127.0f);
            output[tid] = int8_t(quantized);
        }
        
        kernel void dequantizeInt8ToFloat32(
            device const int8_t* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant float& scale [[buffer(2)]],
            constant float& zeroPoint [[buffer(3)]],
            constant uint& count [[buffer(4)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= count) return;
            
            float value = float(input[tid]);
            output[tid] = (value - zeroPoint) * scale;
        }
        
        // === Batch Operations ===
        
        kernel void batchNormalize(
            device const float* input [[buffer(0)]],
            device float* output [[buffer(1)]],
            constant uint& num_vectors [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint gid [[thread_position_in_grid]])
        {
            const uint vector_idx = gid;
            if (vector_idx >= num_vectors) return;
            
            const uint vector_offset = vector_idx * dimension;
            
            // Compute magnitude
            float sum = 0.0f;
            for (uint d = 0; d < dimension; d++) {
                float val = input[vector_offset + d];
                sum += val * val;
            }
            float magnitude = sqrt(sum);
            
            // Normalize
            for (uint d = 0; d < dimension; d++) {
                if (magnitude > EPSILON) {
                    output[vector_offset + d] = input[vector_offset + d] / magnitude;
                } else {
                    output[vector_offset + d] = input[vector_offset + d];
                }
            }
        }
        
        // === Clustering Operations ===
        
        kernel void assignToCentroids(
            device const float* points [[buffer(0)]],
            device const float* centroids [[buffer(1)]],
            device uint* assignments [[buffer(2)]],
            device float* minDistances [[buffer(3)]],
            constant uint& numPoints [[buffer(4)]],
            constant uint& numCentroids [[buffer(5)]],
            constant uint& dimension [[buffer(6)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= numPoints) return;
            
            uint pointOffset = tid * dimension;
            float minDist = INFINITY;
            uint closestCentroid = 0;
            
            for (uint c = 0; c < numCentroids; c++) {
                uint centroidOffset = c * dimension;
                float dist = 0.0f;
                
                for (uint d = 0; d < dimension; d++) {
                    float diff = points[pointOffset + d] - centroids[centroidOffset + d];
                    dist += diff * diff;
                }
                
                if (dist < minDist) {
                    minDist = dist;
                    closestCentroid = c;
                }
            }
            
            assignments[tid] = closestCentroid;
            minDistances[tid] = sqrt(minDist);
        }
        
        kernel void updateCentroids(
            device const float* points [[buffer(0)]],
            device const uint* assignments [[buffer(1)]],
            device float* centroids [[buffer(2)]],
            device uint* counts [[buffer(3)]],
            constant uint& numPoints [[buffer(4)]],
            constant uint& numCentroids [[buffer(5)]],
            constant uint& dimension [[buffer(6)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint centroidIdx = gid.x;
            uint dimIdx = gid.y;
            
            if (centroidIdx >= numCentroids || dimIdx >= dimension) return;
            
            float sum = 0.0f;
            uint count = 0;
            
            for (uint p = 0; p < numPoints; p++) {
                if (assignments[p] == centroidIdx) {
                    sum += points[p * dimension + dimIdx];
                    count++;
                }
            }
            
            if (dimIdx == 0) {
                counts[centroidIdx] = count;
            }
            
            if (count > 0) {
                centroids[centroidIdx * dimension + dimIdx] = sum / float(count);
            }
        }
        
        // === Additional Distance Metrics ===
        
        kernel void manhattanDistance(
            device const float* vectorA [[buffer(0)]],
            device const float* vectorB [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float partialSums[256];
            
            // Each thread accumulates multiple elements with strided access
            float sum = 0.0f;
            for (uint i = tid; i < dimension; i += tgSize) {
                sum += abs(vectorA[i] - vectorB[i]);
            }
            
            // Store in threadgroup memory (safe because tid < tgSize <= 256)
            if (tid < 256) {
                partialSums[tid] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Parallel reduction - handle non-power-of-2 sizes
            uint activeThreads = min(tgSize, 256u);
            if (activeThreads > 1) {
                // Round up to next power of 2 for reduction
                uint powerOf2 = 1;
                while (powerOf2 < activeThreads) powerOf2 *= 2;
                powerOf2 /= 2;
                
                for (uint stride = powerOf2; stride > 0; stride /= 2) {
                    if (tid < stride) {
                        uint partner = tid + stride;
                        if (partner < activeThreads) {
                            partialSums[tid] += partialSums[partner];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            
            // Thread 0 writes the final result
            if (tid == 0) {
                result[0] = partialSums[0];
            }
        }
        
        kernel void chebyshevDistance(
            device const float* vectorA [[buffer(0)]],
            device const float* vectorB [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float partialMax[256];
            
            // Each thread finds maximum across multiple elements
            float localMax = 0.0f;
            for (uint i = tid; i < dimension; i += tgSize) {
                localMax = max(localMax, abs(vectorA[i] - vectorB[i]));
            }
            
            // Store in threadgroup memory
            if (tid < 256) {
                partialMax[tid] = localMax;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Parallel reduction to find maximum
            uint activeThreads = min(tgSize, 256u);
            if (activeThreads > 1) {
                uint powerOf2 = 1;
                while (powerOf2 < activeThreads) powerOf2 *= 2;
                powerOf2 /= 2;
                
                for (uint stride = powerOf2; stride > 0; stride /= 2) {
                    if (tid < stride) {
                        uint partner = tid + stride;
                        if (partner < activeThreads) {
                            partialMax[tid] = max(partialMax[tid], partialMax[partner]);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            
            // Thread 0 writes the final result
            if (tid == 0) {
                result[0] = partialMax[0];
            }
        }
        
        kernel void minkowskiDistance(
            device const float* vectorA [[buffer(0)]],
            device const float* vectorB [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            constant float& p [[buffer(4)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup float partialSums[256];
            
            // Each thread accumulates powered differences
            float sum = 0.0f;
            for (uint i = tid; i < dimension; i += tgSize) {
                float diff = abs(vectorA[i] - vectorB[i]);
                sum += pow(diff, p);
            }
            
            // Store in threadgroup memory
            if (tid < 256) {
                partialSums[tid] = sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Parallel reduction
            uint activeThreads = min(tgSize, 256u);
            if (activeThreads > 1) {
                uint powerOf2 = 1;
                while (powerOf2 < activeThreads) powerOf2 *= 2;
                powerOf2 /= 2;
                
                for (uint stride = powerOf2; stride > 0; stride /= 2) {
                    if (tid < stride) {
                        uint partner = tid + stride;
                        if (partner < activeThreads) {
                            partialSums[tid] += partialSums[partner];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            
            // Thread 0 computes final result with inverse power
            if (tid == 0) {
                result[0] = pow(partialSums[0], 1.0f / p);
            }
        }
        
        kernel void hammingDistance(
            device const float* vectorA [[buffer(0)]],
            device const float* vectorB [[buffer(1)]],
            device float* result [[buffer(2)]],
            constant uint& dimension [[buffer(3)]],
            uint tid [[thread_position_in_threadgroup]],
            uint tgSize [[threads_per_threadgroup]]
        ) {
            threadgroup uint partialCounts[256];
            
            // Each thread counts differences
            uint count = 0;
            for (uint i = tid; i < dimension; i += tgSize) {
                bool aBit = (vectorA[i] != 0.0f);
                bool bBit = (vectorB[i] != 0.0f);
                if (aBit != bBit) {
                    count++;
                }
            }
            
            // Store in threadgroup memory
            if (tid < 256) {
                partialCounts[tid] = count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Parallel reduction
            uint activeThreads = min(tgSize, 256u);
            if (activeThreads > 1) {
                uint powerOf2 = 1;
                while (powerOf2 < activeThreads) powerOf2 *= 2;
                powerOf2 /= 2;
                
                for (uint stride = powerOf2; stride > 0; stride /= 2) {
                    if (tid < stride) {
                        uint partner = tid + stride;
                        if (partner < activeThreads) {
                            partialCounts[tid] += partialCounts[partner];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            
            // Thread 0 writes the final result
            if (tid == 0) {
                result[0] = float(partialCounts[0]);
            }
        }
        
        """
    }
    
    // MARK: - Performance Metrics
    
    /// Get compilation statistics
    public func getStatistics() -> CompilationStats {
        CompilationStats(
            totalCompilationTime: compilationTime,
            compilationCount: compilationCount,
            averageCompilationTime: compilationCount > 0 ? compilationTime / Double(compilationCount) : 0,
            cachedPipelines: pipelineCache.count,
            cachedFunctions: functionCache.count,
            cachedLibraries: libraryCache.count
        )
    }
    
    /// Clear all caches
    public func clearCache() {
        pipelineCache.removeAll()
        functionCache.removeAll()
        libraryCache = libraryCache.filter { key, _ in key == "VectorAccelerate.Default" }
    }
}

/// Statistics for shader compilation
public struct CompilationStats: Sendable {
    public let totalCompilationTime: TimeInterval
    public let compilationCount: Int
    public let averageCompilationTime: TimeInterval
    public let cachedPipelines: Int
    public let cachedFunctions: Int
    public let cachedLibraries: Int
}

// MARK: - Specialized Shader Configurations

public extension ShaderManager {
    
    /// Pre-compile commonly used shaders
    func precompileCommonShaders() async throws {
        let commonShaders = [
            "euclideanDistance",
            "cosineDistance",
            "dotProduct",
            "vectorNormalize",
            "vectorScale",
            "matrixVectorMultiply",
            "batchEuclideanDistance"
        ]
        
        for shaderName in commonShaders {
            do {
                _ = try await getPipelineState(functionName: shaderName)
            } catch {
                // Continue even if some shaders fail
                print("Failed to precompile shader '\(shaderName)': \(error)")
            }
        }
    }
    
    /// Get optimal shader variant for dimension size
    func getOptimalShader(
        operation: String,
        dimension: Int,
        vectorCount: Int = 1
    ) async throws -> any MTLComputePipelineState {
        // Select shader variant based on dimension and vector count
        let shaderName: String
        
        switch operation {
        case "euclideanDistance":
            if dimension <= 128 {
                shaderName = "euclideanDistanceSmall"
            } else if dimension <= 1024 {
                shaderName = "euclideanDistance"
            } else {
                shaderName = "euclideanDistanceLarge"
            }
            
        case "dotProduct":
            if vectorCount > 1 {
                shaderName = "batchDotProduct"
            } else {
                shaderName = "dotProduct"
            }
            
        default:
            shaderName = operation
        }
        
        // Try to get optimized variant, fall back to base version
        do {
            return try await getPipelineState(functionName: shaderName)
        } catch {
            return try await getPipelineState(functionName: operation)
        }
    }
}

// MARK: - Shader Function Registry

/// Registry of available shader functions and their metadata
public struct ShaderFunctionInfo: Sendable {
    public let name: String
    public let description: String
    public let inputBuffers: Int
    public let outputBuffers: Int
    public let requiresParams: Bool
    public let supportsVectorized: Bool
}

public extension ShaderManager {
    
    static let availableShaders: [ShaderFunctionInfo] = [
        ShaderFunctionInfo(
            name: "euclideanDistance",
            description: "Compute Euclidean distance between vectors",
            inputBuffers: 2,
            outputBuffers: 1,
            requiresParams: true,
            supportsVectorized: true
        ),
        ShaderFunctionInfo(
            name: "cosineDistance",
            description: "Compute cosine distance between vectors",
            inputBuffers: 2,
            outputBuffers: 1,
            requiresParams: true,
            supportsVectorized: true
        ),
        ShaderFunctionInfo(
            name: "dotProduct",
            description: "Compute dot product of vectors",
            inputBuffers: 2,
            outputBuffers: 1,
            requiresParams: true,
            supportsVectorized: true
        ),
        ShaderFunctionInfo(
            name: "normalizeVectors",
            description: "Normalize vectors to unit length",
            inputBuffers: 1,
            outputBuffers: 1,
            requiresParams: true,
            supportsVectorized: true
        ),
        ShaderFunctionInfo(
            name: "matrixVectorMultiply",
            description: "Matrix-vector multiplication",
            inputBuffers: 2,
            outputBuffers: 1,
            requiresParams: true,
            supportsVectorized: false
        ),
        ShaderFunctionInfo(
            name: "batchEuclideanDistance",
            description: "Batch Euclidean distance computation",
            inputBuffers: 2,
            outputBuffers: 1,
            requiresParams: true,
            supportsVectorized: true
        )
    ]
    
    /// Check if a shader function is available
    func isShaderAvailable(_ name: String) async -> Bool {
        do {
            _ = try await getFunction(name: name)
            return true
        } catch {
            return false
        }
    }
}
