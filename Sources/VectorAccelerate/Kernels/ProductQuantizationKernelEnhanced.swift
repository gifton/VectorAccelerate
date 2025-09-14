import Metal
import QuartzCore
import Foundation
import QuartzCore

public enum PQError: Error, LocalizedError {
    case libraryCreationFailed, kernelNotFound, encoderCreationFailed, commandQueueCreationFailed
    case invalidInput(String), bufferCreationFailed, configurationError(String)
    case executionError(Error), ioError(String)

    public var errorDescription: String? {
        switch self {
        case .configurationError(let msg): return "Invalid PQ Configuration: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        case .executionError(let error): return "Execution error: \(error.localizedDescription)"
        case .ioError(let msg): return "IO error: \(msg)"
        case .libraryCreationFailed: return "Failed to create Metal library"
        case .kernelNotFound: return "Required kernel not found"
        case .encoderCreationFailed: return "Failed to create command encoder"
        case .commandQueueCreationFailed: return "Failed to create command queue"
        case .bufferCreationFailed: return "Failed to create Metal buffer"
        }
    }
}

// MARK: - Training Metrics

public struct TrainingMetrics {
    public let iterations: Int
    public let finalConvergence: Float
    public let trainingTime: TimeInterval
    public let convergedEarly: Bool
    
    public var averageIterationTime: TimeInterval {
        return trainingTime / Double(iterations)
    }
}

// MARK: - Enhanced Product Quantization Kernel

public final class ProductQuantizationKernel {
    private let device: any MTLDevice
    private let commandQueue: any MTLCommandQueue
    
    // Pipelines
    private let pipelineAssignmentEncoding: any MTLComputePipelineState
    private let pipelineTrainAccumulate: any MTLComputePipelineState
    private let pipelineTrainFinalize: any MTLComputePipelineState
    private let pipelinePrecomputeDist: any MTLComputePipelineState
    private let pipelineADC: any MTLComputePipelineState

    // MARK: - Configuration and Parameters

    public struct Configuration {
        public let D: Int; public let M: Int; public let K: Int
        public var D_sub: Int { return D / M }

        public init(D: Int, M: Int, K: Int = 256) throws {
            self.D = D; self.M = M; self.K = K
            if D % M != 0 {
                throw PQError.configurationError("D (\(D)) must be divisible by M (\(M)).")
            }
            if K > 256 {
                throw PQError.configurationError("K > 256 not supported (kernels use uint8_t codes).")
            }
        }
    }
    
    // Matches Metal PQConfig
    private struct PQConfig {
        var N: UInt32; var D: UInt32; var M: UInt32; var K: UInt32; var D_sub: UInt32;
        init(_ config: Configuration, N: Int) {
            self.N = UInt32(N); self.D = UInt32(config.D); self.M = UInt32(config.M);
            self.K = UInt32(config.K); self.D_sub = UInt32(config.D_sub);
        }
    }
    
    // MARK: - Model Persistence
    
    public struct PQModel {
        public let codebooks: any MTLBuffer
        public let configuration: Configuration
        public let trainingMetrics: TrainingMetrics?
        
        public init(codebooks: any MTLBuffer, configuration: Configuration, metrics: TrainingMetrics? = nil) {
            self.codebooks = codebooks
            self.configuration = configuration
            self.trainingMetrics = metrics
        }
        
        /// Save codebooks to file
        public func save(to url: URL) throws {
            let codebookSize = configuration.M * configuration.K * configuration.D_sub * MemoryLayout<Float>.stride
            let codebookData = Data(bytesNoCopy: codebooks.contents(), count: codebookSize, deallocator: .none)
            
            let modelDict: [String: Any] = [
                "version": 1,
                "D": configuration.D,
                "M": configuration.M,
                "K": configuration.K,
                "codebooks": codebookData.base64EncodedString(),
                "trainingIterations": trainingMetrics?.iterations ?? 0,
                "finalConvergence": trainingMetrics?.finalConvergence ?? 0,
                "trainingTime": trainingMetrics?.trainingTime ?? 0
            ]
            
            let jsonData = try JSONSerialization.data(withJSONObject: modelDict, options: .prettyPrinted)
            try jsonData.write(to: url)
        }
        
        /// Load codebooks from file
        public static func load(from url: URL, device: any MTLDevice) throws -> PQModel {
            let jsonData = try Data(contentsOf: url)
            guard let modelDict = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
                throw PQError.ioError("Invalid model file format")
            }
            
            guard let D = modelDict["D"] as? Int,
                  let M = modelDict["M"] as? Int,
                  let K = modelDict["K"] as? Int,
                  let codebooksBase64 = modelDict["codebooks"] as? String,
                  let codebookData = Data(base64Encoded: codebooksBase64) else {
                throw PQError.ioError("Missing or invalid model parameters")
            }
            
            let config = try Configuration(D: D, M: M, K: K)
            
            guard let codebooks = device.makeBuffer(bytes: Array(codebookData), 
                                                   length: codebookData.count,
                                                   options: MTLResourceOptions.storageModeShared) else {
                throw PQError.bufferCreationFailed
            }
            
            // Load training metrics if available
            var metrics: TrainingMetrics? = nil
            if let iterations = modelDict["trainingIterations"] as? Int,
               let convergence = modelDict["finalConvergence"] as? Float,
               let time = modelDict["trainingTime"] as? TimeInterval {
                metrics = TrainingMetrics(
                    iterations: iterations,
                    finalConvergence: convergence,
                    trainingTime: time,
                    convergedEarly: iterations < 25 // Assuming max 25 iterations
                )
            }
            
            return PQModel(codebooks: codebooks, configuration: config, metrics: metrics)
        }
    }

    // MARK: - Initialization

    public init(device: any MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw PQError.commandQueueCreationFailed }
        self.commandQueue = queue
        
        guard let library = device.makeDefaultLibrary() else { throw PQError.libraryCreationFailed }
        
        // Load kernels
        guard let kAssignEncode = library.makeFunction(name: "pq_assignment_or_encoding"),
              let kTrainAcc = library.makeFunction(name: "pq_train_update_accumulate"),
              let kTrainFinal = library.makeFunction(name: "pq_train_update_finalize"),
              let kPrecompute = library.makeFunction(name: "pq_precompute_distance_table"),
              let kADC = library.makeFunction(name: "pq_compute_distances_adc")
        else { throw PQError.kernelNotFound }
        
        // Create pipelines (Requires device support for float atomics for training pipelines)
        self.pipelineAssignmentEncoding = try device.makeComputePipelineState(function: kAssignEncode)
        self.pipelineTrainAccumulate = try device.makeComputePipelineState(function: kTrainAcc)
        self.pipelineTrainFinalize = try device.makeComputePipelineState(function: kTrainFinal)
        self.pipelinePrecomputeDist = try device.makeComputePipelineState(function: kPrecompute)
        self.pipelineADC = try device.makeComputePipelineState(function: kADC)
    }

    // MARK: - Phase 2: Encoding

    public func encode(
        vectors: any MTLBuffer,
        codebooks: any MTLBuffer,
        outputCodes: any MTLBuffer,
        config: Configuration,
        N: Int,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        if N == 0 { return }
        let pqConfig = PQConfig(config, N: N)
        // The encoding logic is identical to the training assignment step.
        try encodeAssignmentOrEncoding(commandBuffer: commandBuffer, pipeline: pipelineAssignmentEncoding, vectors: vectors, codebooks: codebooks, output: outputCodes, config: pqConfig)
    }

    // MARK: - Phase 3: Distance Computation (ADC)

    public func computeDistances(
        query: any MTLBuffer,
        codes: any MTLBuffer,
        codebooks: any MTLBuffer,
        outputDistances: any MTLBuffer,
        config: Configuration,
        N: Int,
        commandBuffer: any MTLCommandBuffer
    ) throws {
        if N == 0 { return }
        
        // 1. Allocate intermediate Distance Table [M x K].
        // Use GPU Private memory as it's only used between the two stages.
        let distanceTableLength = config.M * config.K * MemoryLayout<Float>.stride
        guard let distanceTable = device.makeBuffer(length: distanceTableLength, options: MTLResourceOptions.storageModePrivate) else {
            throw PQError.bufferCreationFailed
        }

        // 2. Stage 1: Precompute Distance Table
        try encodePrecompute(commandBuffer: commandBuffer, query: query, codebooks: codebooks, distanceTable: distanceTable, config: config)
        
        // 3. Stage 2: ADC Lookup (Scan)
        // A barrier is implicitly enforced between the two compute encoders.
        try encodeADC(commandBuffer: commandBuffer, codes: codes, distanceTable: distanceTable, outputDistances: outputDistances, config: config, N: N)
    }
    
    // MARK: - Convenience Method: Find Top-K
    
    /// Find top-K nearest neighbors for a query
    public func findTopK(
        query: [Float],
        codes: any MTLBuffer,
        codebooks: any MTLBuffer,
        config: Configuration,
        N: Int,
        k: Int
    ) async throws -> [(index: Int, distance: Float)] {
        // Validate input
        guard query.count == config.D else {
            throw PQError.invalidInput("Query dimension \(query.count) doesn't match configuration \(config.D)")
        }
        guard k > 0 && k <= N else {
            throw PQError.invalidInput("K must be between 1 and \(N)")
        }
        
        // Create query buffer
        guard let queryBuffer = device.makeBuffer(bytes: query, length: query.count * MemoryLayout<Float>.stride, options: MTLResourceOptions.storageModeShared) else {
            throw PQError.bufferCreationFailed
        }
        
        // Create output buffer for distances
        guard let distancesBuffer = device.makeBuffer(length: N * MemoryLayout<Float>.stride, options: MTLResourceOptions.storageModeShared) else {
            throw PQError.bufferCreationFailed
        }
        
        // Compute distances
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw PQError.encoderCreationFailed
        }
        
        try computeDistances(
            query: queryBuffer,
            codes: codes,
            codebooks: codebooks,
            outputDistances: distancesBuffer,
            config: config,
            N: N,
            commandBuffer: commandBuffer
        )
        
        try await commit(commandBuffer)
        
        // Extract distances and find top-k
        let distancesPtr = distancesBuffer.contents().bindMemory(to: Float.self, capacity: N)
        var indexedDistances: [(index: Int, distance: Float)] = []
        
        for i in 0..<N {
            indexedDistances.append((index: i, distance: sqrt(distancesPtr[i]))) // Convert from squared distance
        }
        
        // Partial sort for efficiency (only sort top-k)
        indexedDistances.sort { $0.distance < $1.distance }
        return Array(indexedDistances.prefix(k))
    }

    // MARK: - Phase 1: Training (K-Means Orchestration)
    
    /// Trains the PQ codebooks using K-means clustering on the training data.
    /// Note: Initialization (e.g., K-means++ or random sampling) must be performed on the initialCodebooks buffer before calling this function.
    public func trainCodebooks(
        trainingData: any MTLBuffer,
        initialCodebooks: any MTLBuffer,
        N_train: Int,
        config: Configuration,
        maxIterations: Int = 25,
        convergenceThreshold: Float = 0.001,
        progressHandler: ((Int, Float) -> Void)? = nil
    ) async throws -> PQModel {
        
        let startTime = CACurrentMediaTime()
        
        // 1. Buffer Allocation
        let M = config.M; let K = config.K; let D_sub = config.D_sub
        let codebookSize = M * K * D_sub * MemoryLayout<Float>.stride
        let assignmentSize = N_train * M * MemoryLayout<UInt8>.stride
        let accumulatorSize = codebookSize // Float atomics
        let countsSize = M * K * MemoryLayout<UInt32>.stride
        // Convergence stores movement per centroid [M * K]
        let convergenceSize = M * K * MemoryLayout<Float>.stride
        
        // Use Private storage for efficiency during training iterations
        guard let codebooks = device.makeBuffer(length: codebookSize, options: MTLResourceOptions.storageModePrivate),
              let assignments = device.makeBuffer(length: assignmentSize, options: MTLResourceOptions.storageModePrivate),
              let accumulators = device.makeBuffer(length: accumulatorSize, options: MTLResourceOptions.storageModePrivate),
              let counts = device.makeBuffer(length: countsSize, options: MTLResourceOptions.storageModePrivate),
              // Convergence must be Shared to read back on CPU
              let convergenceBuffer = device.makeBuffer(length: convergenceSize, options: MTLResourceOptions.storageModeShared)
        else { throw PQError.bufferCreationFailed }

        // Copy initial codebooks into the working buffer
        try await copyBuffer(from: initialCodebooks, to: codebooks, size: codebookSize)

        // 2. Iterative K-Means
        let pqConfig = PQConfig(config, N: N_train)
        
        var finalConvergence: Float = Float.infinity
        var actualIterations = 0
        
        for iteration in 0..<maxIterations {
            actualIterations = iteration + 1
            
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw PQError.encoderCreationFailed }
            commandBuffer.label = "PQ Iteration \(iteration)"

            // Step 1: Assignment (E-step) - 2D Dispatch (N_train x M)
            try encodeAssignmentOrEncoding(commandBuffer: commandBuffer, pipeline: pipelineAssignmentEncoding, vectors: trainingData, codebooks: codebooks, output: assignments, config: pqConfig)
            
            // Step 2: Update (M-step) - Initialization (Zeroing buffers)
            // Use BlitEncoder for efficient zeroing.
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { throw PQError.encoderCreationFailed }
            blitEncoder.fill(buffer: accumulators, range: 0..<accumulatorSize, value: 0)
            blitEncoder.fill(buffer: counts, range: 0..<countsSize, value: 0)
            blitEncoder.endEncoding()

            // Step 3: Update (M-step) - Accumulation - 2D Dispatch (N_train x M)
            try encodeAccumulation(commandBuffer: commandBuffer, trainingData: trainingData, assignments: assignments, accumulators: accumulators, counts: counts, config: pqConfig)

            // Step 4: Update (M-step) - Finalization - 2D Dispatch (K x M)
            try encodeFinalization(commandBuffer: commandBuffer, codebooks: codebooks, accumulators: accumulators, counts: counts, convergence: convergenceBuffer, config: pqConfig)
            
            // Commit and wait (necessary for CPU convergence check)
            try await commit(commandBuffer)
            
            // Step 5: Convergence Check (CPU side)
            let convergencePtr = convergenceBuffer.contents().bindMemory(to: Float.self, capacity: M * K)
            // CPU reduction of the movement array
            let totalMovement = Array(UnsafeBufferPointer(start: convergencePtr, count: M * K)).reduce(0, +)
            finalConvergence = totalMovement
            
            // Report progress
            progressHandler?(iteration, totalMovement)
            
            if totalMovement < convergenceThreshold {
                break
            }
        }

        // 3. Copy final codebooks from Private to Shared buffer for external access.
        let finalCodebooks = try await copyBufferToShared(buffer: codebooks, size: codebookSize)
        
        let trainingTime = CACurrentMediaTime() - startTime
        let metrics = TrainingMetrics(
            iterations: actualIterations,
            finalConvergence: finalConvergence,
            trainingTime: trainingTime,
            convergedEarly: actualIterations < maxIterations
        )
        
        return PQModel(codebooks: finalCodebooks, configuration: config, metrics: metrics)
    }
    
    // MARK: - Performance Metrics
    
    public struct PerformanceMetrics {
        public let encodingThroughput: Double // vectors per second
        public let queryThroughput: Double    // queries per second
        public let compressionRatio: Float
        public let memoryUsage: Int          // bytes
        
        public var description: String {
            return """
            PQ Performance Metrics:
            - Encoding: \(Int(encodingThroughput)) vectors/sec
            - Query: \(Int(queryThroughput)) queries/sec
            - Compression: \(compressionRatio)x
            - Memory: \(memoryUsage / 1024 / 1024) MB
            """
        }
    }
    
    /// Benchmark PQ performance
    public func benchmark(
        dataSize: Int = 10000,
        queryCount: Int = 100,
        config: Configuration
    ) async throws -> PerformanceMetrics {
        // Generate random test data
        let dimension = config.D
        let trainingData = (0..<dataSize).flatMap { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        guard let trainingBuffer = device.makeBuffer(
            bytes: trainingData,
            length: trainingData.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw PQError.bufferCreationFailed
        }
        
        // Initialize codebooks randomly
        let codebookSize = config.M * config.K * config.D_sub
        let initialCodebooks = (0..<codebookSize).map { _ in Float.random(in: -1...1) }
        guard let initialBuffer = device.makeBuffer(
            bytes: initialCodebooks,
            length: initialCodebooks.count * MemoryLayout<Float>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw PQError.bufferCreationFailed
        }
        
        // Train
        let model = try await trainCodebooks(
            trainingData: trainingBuffer,
            initialCodebooks: initialBuffer,
            N_train: dataSize,
            config: config,
            maxIterations: 10 // Reduced for benchmarking
        )
        
        // Encoding benchmark
        guard let codes = device.makeBuffer(
            length: dataSize * config.M * MemoryLayout<UInt8>.stride,
            options: MTLResourceOptions.storageModeShared
        ) else {
            throw PQError.bufferCreationFailed
        }
        
        let encodeStart = CACurrentMediaTime()
        guard let encodeCommand = commandQueue.makeCommandBuffer() else {
            throw PQError.encoderCreationFailed
        }
        try encode(
            vectors: trainingBuffer,
            codebooks: model.codebooks,
            outputCodes: codes,
            config: config,
            N: dataSize,
            commandBuffer: encodeCommand
        )
        try await commit(encodeCommand)
        let encodeTime = CACurrentMediaTime() - encodeStart
        
        // Query benchmark
        let queryStart = CACurrentMediaTime()
        for _ in 0..<queryCount {
            let query = (0..<dimension).map { _ in Float.random(in: -1...1) }
            _ = try await findTopK(
                query: query,
                codes: codes,
                codebooks: model.codebooks,
                config: config,
                N: dataSize,
                k: 10
            )
        }
        let queryTime = CACurrentMediaTime() - queryStart
        
        // Calculate metrics
        let originalSize = dataSize * dimension * MemoryLayout<Float>.stride
        let compressedSize = dataSize * config.M * MemoryLayout<UInt8>.stride
        let codebookMemory = config.M * config.K * config.D_sub * MemoryLayout<Float>.stride
        
        return PerformanceMetrics(
            encodingThroughput: Double(dataSize) / encodeTime,
            queryThroughput: Double(queryCount) / queryTime,
            compressionRatio: Float(originalSize) / Float(compressedSize),
            memoryUsage: compressedSize + codebookMemory
        )
    }

    // MARK: - Encoding Helpers (Internal)

    // Handles 2D dispatch for Assignment/Encoding
    private func encodeAssignmentOrEncoding(commandBuffer: any MTLCommandBuffer, pipeline: any MTLComputePipelineState, vectors: any MTLBuffer, codebooks: any MTLBuffer, output: any MTLBuffer, config: PQConfig) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw PQError.encoderCreationFailed }
        encoder.setComputePipelineState(pipeline)
        
        var cfg = config
        // Signature: vectors (0), codebooks (1), assignments/codes (2), config (3)
        encoder.setBuffer(vectors, offset: 0, index: 0)
        encoder.setBuffer(codebooks, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        encoder.setBytes(&cfg, length: MemoryLayout<PQConfig>.stride, index: 3)

        // Dispatch 2D (N x M)
        let gridSize = MTLSize(width: Int(cfg.N), height: Int(cfg.M), depth: 1)
        // Calculate efficient Threadgroup Size (TGS)
        let w = pipeline.threadExecutionWidth
        let h = max(1, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threadgroupSize = MTLSize(width: w, height: h, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }
    
    // Handles 2D dispatch for Accumulation
    private func encodeAccumulation(commandBuffer: any MTLCommandBuffer, trainingData: any MTLBuffer, assignments: any MTLBuffer, accumulators: any MTLBuffer, counts: any MTLBuffer, config: PQConfig) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw PQError.encoderCreationFailed }
        encoder.setComputePipelineState(pipelineTrainAccumulate)
        
        var cfg = config
        // Signature: training_data (0), assignments (1), centroids_accum (2), centroid_counts (3), config (4)
        encoder.setBuffer(trainingData, offset: 0, index: 0)
        encoder.setBuffer(assignments, offset: 0, index: 1)
        encoder.setBuffer(accumulators, offset: 0, index: 2)
        encoder.setBuffer(counts, offset: 0, index: 3)
        encoder.setBytes(&cfg, length: MemoryLayout<PQConfig>.stride, index: 4)
        
        // Dispatch 2D (N_train x M)
        let gridSize = MTLSize(width: Int(cfg.N), height: Int(cfg.M), depth: 1)
        let w = pipelineTrainAccumulate.threadExecutionWidth
        let h = max(1, pipelineTrainAccumulate.maxTotalThreadsPerThreadgroup / w)
        let threadgroupSize = MTLSize(width: w, height: h, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
    }

    // Handles 2D dispatch for Finalization
    private func encodeFinalization(commandBuffer: any MTLCommandBuffer, codebooks: any MTLBuffer, accumulators: any MTLBuffer, counts: any MTLBuffer, convergence: any MTLBuffer, config: PQConfig) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw PQError.encoderCreationFailed }
        encoder.setComputePipelineState(pipelineTrainFinalize)

        var cfg = config
        // Signature: codebooks (0), centroids_accum (1), centroid_counts (2), convergence_out (3), config (4)
        encoder.setBuffer(codebooks, offset: 0, index: 0)
        encoder.setBuffer(accumulators, offset: 0, index: 1)
        encoder.setBuffer(counts, offset: 0, index: 2)
        encoder.setBuffer(convergence, offset: 0, index: 3)
        encoder.setBytes(&cfg, length: MemoryLayout<PQConfig>.stride, index: 4)

        // Dispatch 2D (K x M)
        let gridSize = MTLSize(width: Int(cfg.K), height: Int(cfg.M), depth: 1)
        // Optimize TGS for the smaller grid (KxM)
        let w = pipelineTrainFinalize.threadExecutionWidth
        let tgs = MTLSize(width: w, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }
    
    // Handles 2D dispatch for Precomputation
    private func encodePrecompute(commandBuffer: any MTLCommandBuffer, query: any MTLBuffer, codebooks: any MTLBuffer, distanceTable: any MTLBuffer, config: Configuration) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw PQError.encoderCreationFailed }
        encoder.setComputePipelineState(pipelinePrecomputeDist)

        // N=0 as it's irrelevant for precomputation
        var cfg = PQConfig(config, N: 0)
        // Signature: query (0), codebooks (1), distance_table (2), config (3)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(codebooks, offset: 0, index: 1)
        encoder.setBuffer(distanceTable, offset: 0, index: 2)
        encoder.setBytes(&cfg, length: MemoryLayout<PQConfig>.stride, index: 3)

        // Dispatch 2D (K x M)
        let gridSize = MTLSize(width: config.K, height: config.M, depth: 1)
        let w = pipelinePrecomputeDist.threadExecutionWidth
        let tgs = MTLSize(width: w, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }

    // Handles 1D dispatch for ADC Scan with Threadgroup Memory
    private func encodeADC(commandBuffer: any MTLCommandBuffer, codes: any MTLBuffer, distanceTable: any MTLBuffer, outputDistances: any MTLBuffer, config: Configuration, N: Int) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { throw PQError.encoderCreationFailed }
        encoder.setComputePipelineState(pipelineADC)

        var cfg = PQConfig(config, N: N)
        // Signature: codes (0), distance_table (1), distances (2), config (3)
        encoder.setBuffer(codes, offset: 0, index: 0)
        encoder.setBuffer(distanceTable, offset: 0, index: 1)
        encoder.setBuffer(outputDistances, offset: 0, index: 2)
        encoder.setBytes(&cfg, length: MemoryLayout<PQConfig>.stride, index: 3)
        
        // Set Threadgroup Memory (Shared Memory) length for the distance table
        // Size = M * K * sizeof(float)
        let sharedMemorySize = config.M * config.K * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(sharedMemorySize, index: 0)

        // Dispatch 1D (N)
        let gridSize = MTLSize(width: N, height: 1, depth: 1)
        let w = pipelineADC.threadExecutionWidth
        let maxT = pipelineADC.maxTotalThreadsPerThreadgroup
        let tgs_width = min(N, min(maxT, w * (maxT / w)))
        let tgs = MTLSize(width: tgs_width, height: 1, depth: 1)
        
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgs)
        encoder.endEncoding()
    }

    // MARK: - Utility Helpers (Async/Await and Buffer Management)
    
    private func commit(_ commandBuffer: any MTLCommandBuffer) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { completedCommandBuffer in
                if let error = completedCommandBuffer.error {
                    continuation.resume(throwing: PQError.executionError(error))
                } else {
                    continuation.resume()
                }
            }
            commandBuffer.commit()
        }
    }
    
    private func copyBuffer(from source: any MTLBuffer, to destination: any MTLBuffer, size: Int) async throws {
        guard let blitCmd = commandQueue.makeCommandBuffer(), let blitEnc = blitCmd.makeBlitCommandEncoder() else {
             throw PQError.encoderCreationFailed
        }
        blitEnc.copy(from: source, sourceOffset: 0, to: destination, destinationOffset: 0, size: size)
        blitEnc.endEncoding()
        try await commit(blitCmd)
    }
    
    private func copyBufferToShared(buffer: any MTLBuffer, size: Int) async throws -> any MTLBuffer {
        guard let sharedBuffer = device.makeBuffer(length: size, options: MTLResourceOptions.storageModeShared) else {
             throw PQError.bufferCreationFailed
        }
        try await copyBuffer(from: buffer, to: sharedBuffer, size: size)
        return sharedBuffer
    }
}