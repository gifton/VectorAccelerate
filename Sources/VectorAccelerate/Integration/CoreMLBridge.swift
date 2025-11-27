//
//  CoreMLBridge.swift
//  VectorAccelerate
//
//  Core ML integration for accelerated model inference
//

import Foundation
@preconcurrency import CoreML
import Metal
import Accelerate
import VectorCore

/// Core ML model configuration
public struct CoreMLConfiguration: Sendable {
    public let computeUnits: MLComputeUnits
    public let batchSize: Int
    public let shareMemory: Bool  // Share Metal buffers between Core ML and VectorAccelerate
    public let enableProfiling: Bool
    
    public init(
        computeUnits: MLComputeUnits = .all,
        batchSize: Int = 32,
        shareMemory: Bool = true,
        enableProfiling: Bool = false
    ) {
        self.computeUnits = computeUnits
        self.batchSize = batchSize
        self.shareMemory = shareMemory
        self.enableProfiling = enableProfiling
    }
}

/// Wrapper for Core ML embedding models
public struct EmbeddingModel {
    public let model: MLModel
    public let inputName: String
    public let outputName: String
    public let dimension: Int
    public let maxSequenceLength: Int?
    
    public init(
        model: MLModel,
        inputName: String,
        outputName: String,
        dimension: Int,
        maxSequenceLength: Int? = nil
    ) {
        self.model = model
        self.inputName = inputName
        self.outputName = outputName
        self.dimension = dimension
        self.maxSequenceLength = maxSequenceLength
    }
}

/// Bridge between Core ML and VectorAccelerate
public actor CoreMLBridge {
    private let context: MetalContext?
    private let configuration: CoreMLConfiguration
    private let logger: Logger
    
    // Cached models
    private var loadedModels: [String: EmbeddingModel] = [:]
    
    // Performance tracking
    private var inferenceCount: Int = 0
    private var totalInferenceTime: TimeInterval = 0
    
    // MARK: - Initialization
    
    public init(
        configuration: CoreMLConfiguration = CoreMLConfiguration(),
        context: MetalContext? = nil
    ) async {
        self.configuration = configuration
        if let ctx = context {
            self.context = ctx
        } else {
            self.context = await MetalContext.createDefault()
        }
        self.logger = Logger.shared
        
        await logger.info("Initialized CoreMLBridge with compute units: \(configuration.computeUnits)",
                         category: "CoreMLBridge")
    }
    
    // MARK: - Model Loading
    
    /// Load a Core ML embedding model
    public func loadModel(at url: URL, identifier: String? = nil) async throws -> EmbeddingModel {
        let measureToken = await logger.startMeasure("loadCoreMLModel")
        defer { measureToken.end() }
        
        // Configure model
        let config = MLModelConfiguration()
        config.computeUnits = configuration.computeUnits
        
        // Load model
        let model = try MLModel(contentsOf: url, configuration: config)
        let modelId = identifier ?? url.lastPathComponent
        
        // Analyze model structure
        let modelDescription = model.modelDescription
        let (inputName, outputName, dimension) = try analyzeModelStructure(modelDescription)
        
        let embeddingModel = EmbeddingModel(
            model: model,
            inputName: inputName,
            outputName: outputName,
            dimension: dimension,
            maxSequenceLength: nil
        )
        
        loadedModels[modelId] = embeddingModel
        
        await logger.info("Loaded Core ML model: \(modelId) with dimension: \(dimension)",
                         category: "CoreMLBridge")
        
        return embeddingModel
    }
    
    /// Analyze model to extract input/output structure
    private func analyzeModelStructure(_ description: MLModelDescription) throws -> (String, String, Int) {
        // Find text/token input
        guard let inputDesc = description.inputDescriptionsByName.first else {
            throw VectorError.invalidOperation("No inputs found in model")
        }
        let inputName = inputDesc.key
        
        // Find embedding output
        guard let outputDesc = description.outputDescriptionsByName.first else {
            throw VectorError.invalidOperation("No outputs found in model")
        }
        let outputName = outputDesc.key
        
        // Determine embedding dimension
        var dimension = 0
        if let multiArray = outputDesc.value.multiArrayConstraint {
            let shape = multiArray.shape
            dimension = shape.last?.intValue ?? 0
        }
        
        guard dimension > 0 else {
            throw VectorError.invalidOperation("Could not determine embedding dimension")
        }
        
        return (inputName, outputName, dimension)
    }
    
    // MARK: - Inference
    
    /// Run inference on text input
    public func embedText(
        _ text: String,
        modelId: String
    ) async throws -> [Float] {
        guard let embeddingModel = loadedModels[modelId] else {
            throw VectorError.invalidOperation("Model not loaded: \(modelId)")
        }
        
        let measureToken = await logger.startMeasure("coreMLInference")
        defer { measureToken.end() }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Create input
        let input = try createTextInput(text, for: embeddingModel)
        
        // Run inference
        let output = try await Task.detached {
            try embeddingModel.model.prediction(from: input)
        }.value
        
        // Extract embeddings
        let embeddings = try extractEmbeddings(from: output, model: embeddingModel)
        
        // Update metrics
        inferenceCount += 1
        totalInferenceTime += CFAbsoluteTimeGetCurrent() - startTime
        
        return embeddings
    }
    
    /// Batch inference on multiple texts
    public func embedBatch(
        _ texts: [String],
        modelId: String
    ) async throws -> [[Float]] {
        guard let embeddingModel = loadedModels[modelId] else {
            throw VectorError.invalidOperation("Model not loaded: \(modelId)")
        }
        
        let measureToken = await logger.startMeasure("batchCoreMLInference")
        measureToken.addMetadata("batchSize", value: "\(texts.count)")
        defer { measureToken.end() }
        
        var allEmbeddings: [[Float]] = []
        
        // Process in batches
        for i in stride(from: 0, to: texts.count, by: configuration.batchSize) {
            let batchEnd = min(i + configuration.batchSize, texts.count)
            let batch = Array(texts[i..<batchEnd])
            
            // Create batch input
            let batchInput = try createBatchInput(batch, for: embeddingModel)
            
            // Run batch inference
            let outputs = try embeddingModel.model.predictions(from: batchInput, options: MLPredictionOptions())
            
            // Extract embeddings from batch
            for i in 0..<outputs.count {
                let output = outputs.features(at: i)
                let embeddings = try extractEmbeddings(from: output, model: embeddingModel)
                allEmbeddings.append(embeddings)
            }
        }
        
        return allEmbeddings
    }
    
    // MARK: - Tensor Operations
    
    /// Convert Core ML multi-array to Metal buffer for zero-copy processing
    public func multiArrayToMetalBuffer(_ multiArray: MLMultiArray) async throws -> BufferToken? {
        guard configuration.shareMemory,
              let ctx = context,
              multiArray.dataType == .float32 else {
            return nil
        }
        
        // Check if we can share memory
        let dataPointer = multiArray.dataPointer.bindMemory(
            to: Float.self,
            capacity: multiArray.count
        )
        
        // Create Metal buffer from existing memory (if possible)
        // Note: This is simplified - actual implementation would need careful memory management
        let size = multiArray.count * MemoryLayout<Float>.stride
        let buffer = try await ctx.getBuffer(size: size)
        
        // Copy data (in real implementation, could share memory directly)
        buffer.write(data: Array(UnsafeBufferPointer(start: dataPointer, count: multiArray.count)))
        
        return buffer
    }
    
    /// Process Core ML embeddings with VectorAccelerate operations
    public func processEmbeddings(
        _ embeddings: MLMultiArray,
        operation: EmbeddingOperation
    ) async throws -> [Float] {
        let measureToken = await logger.startMeasure("processMLEmbeddings")
        defer { measureToken.end() }
        
        // Convert to Metal buffer if possible
        if let metalBuffer = try await multiArrayToMetalBuffer(embeddings) {
            // Use GPU acceleration
            return try await performGPUOperation(metalBuffer, operation: operation)
        } else {
            // Fallback to CPU
            let floatArray = multiArrayToFloatArray(embeddings)
            return try await performCPUOperation(floatArray, operation: operation)
        }
    }
    
    // MARK: - Helper Functions
    
    /// Create text input for Core ML model
    private func createTextInput(_ text: String, for model: EmbeddingModel) throws -> any MLFeatureProvider {
        // This is simplified - actual implementation would handle tokenization
        let input = try MLDictionaryFeatureProvider(dictionary: [
            model.inputName: text
        ])
        return input
    }
    
    /// Create batch input
    private func createBatchInput(_ texts: [String], for model: EmbeddingModel) throws -> any MLBatchProvider {
        var providers: [any MLFeatureProvider] = []
        
        for text in texts {
            let input = try createTextInput(text, for: model)
            providers.append(input)
        }
        
        return MLArrayBatchProvider(array: providers)
    }
    
    /// Extract embeddings from model output
    private func extractEmbeddings(from output: any MLFeatureProvider, model: EmbeddingModel) throws -> [Float] {
        guard let multiArray = output.featureValue(for: model.outputName)?.multiArrayValue else {
            throw VectorError.invalidOperation("Could not extract embeddings from output")
        }
        
        return multiArrayToFloatArray(multiArray)
    }
    
    /// Convert MLMultiArray to Float array
    private func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var result = [Float](repeating: 0, count: count)
        
        let dataPointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            result[i] = dataPointer[i]
        }
        
        return result
    }
    
    /// Perform GPU operation on embeddings
    private func performGPUOperation(_ buffer: BufferToken, operation: EmbeddingOperation) async throws -> [Float] {
        guard context != nil else {
            throw VectorError.metalNotAvailable()
        }
        
        // For now, all operations fall back to CPU
        // GPU shaders could be added later for better performance
        let cpuData = buffer.copyData(as: Float.self)
        
        switch operation {
        case .normalize:
            var normalized = cpuData
            
            var norm: Float = 0
            vDSP_svesq(cpuData, 1, &norm, vDSP_Length(cpuData.count))
            norm = sqrt(norm)
            
            if norm > 0 {
                var invNorm = 1.0 / norm
                vDSP_vsmul(cpuData, 1, &invNorm, &normalized, 1, vDSP_Length(cpuData.count))
            }
            
            return normalized
            
        case .reduce(let method):
            // Fall back to CPU reduction operations
            switch method {
            case .mean:
                var result: Float = 0
                vDSP_meanv(cpuData, 1, &result, vDSP_Length(cpuData.count))
                return [result]
            case .max:
                var result: Float = 0
                vDSP_maxv(cpuData, 1, &result, vDSP_Length(cpuData.count))
                return [result]
            case .sum:
                var result: Float = 0
                vDSP_sve(cpuData, 1, &result, vDSP_Length(cpuData.count))
                return [result]
            }
        }
    }
    
    /// Perform CPU operation on embeddings
    private func performCPUOperation(_ data: [Float], operation: EmbeddingOperation) async throws -> [Float] {
        switch operation {
        case .normalize:
            let simd = SIMDFallback()
            return await simd.normalize(data)
            
        case .reduce(let method):
            switch method {
            case .mean:
                var result: Float = 0
                vDSP_meanv(data, 1, &result, vDSP_Length(data.count))
                return [result]
            case .max:
                var result: Float = 0
                vDSP_maxv(data, 1, &result, vDSP_Length(data.count))
                return [result]
            case .sum:
                var result: Float = 0
                vDSP_sve(data, 1, &result, vDSP_Length(data.count))
                return [result]
            }
        }
    }
    
    // MARK: - Performance Metrics
    
    public func getPerformanceMetrics() -> (
        modelsLoaded: Int,
        inferenceCount: Int,
        averageInferenceTime: TimeInterval
    ) {
        let avgTime = inferenceCount > 0 ? totalInferenceTime / Double(inferenceCount) : 0
        return (loadedModels.count, inferenceCount, avgTime)
    }
}

/// Embedding operation types
public enum EmbeddingOperation: Sendable {
    case normalize
    case reduce(ReductionMethod)
}

/// Reduction methods
public enum ReductionMethod: Sendable {
    case mean
    case max
    case sum
}