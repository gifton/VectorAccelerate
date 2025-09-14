// VectorAccelerate: Quantization Shaders
//
// GPU kernels for vector quantization operations

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Scalar Quantization

/// Quantize floating-point values to 8-bit unsigned integers
kernel void scalarQuantize(
    constant float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& offset [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float value = input[id];
    
    // Apply affine transformation to map to quantization range
    float quantized = round((value - offset) * scale);
    
    // Clamp to 8-bit range [0, 255]
    quantized = clamp(quantized, 0.0f, 255.0f);
    
    // Cast to unsigned char for storage
    output[id] = static_cast<uchar>(quantized);
}

/// Dequantize 8-bit values back to floating-point
kernel void scalarDequantize(
    constant uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant float& offset [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    // Convert 8-bit value to float
    float value = static_cast<float>(input[id]);
    
    // Apply inverse transformation to reconstruct original scale
    output[id] = (value / scale) + offset;
}

// MARK: - Product Quantization

/// Product Quantization: Advanced vector compression technique
kernel void productQuantize(
    constant float* vectors [[buffer(0)]],
    constant float* codebook [[buffer(1)]],
    device uchar* codes [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numSubspaces [[buffer(4)]],
    constant uint& subspaceDimension [[buffer(5)]],
    constant uint& codebookSize [[buffer(6)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint vectorIdx = id.x;
    uint subspaceIdx = id.y;
    
    if (subspaceIdx >= numSubspaces) return;
    
    // Calculate offsets for this vector's subspace
    uint vectorOffset = vectorIdx * vectorDimension + subspaceIdx * subspaceDimension;
    uint codebookOffset = subspaceIdx * codebookSize * subspaceDimension;
    
    float minDistance = INFINITY;
    uint bestCode = 0;
    
    // Exhaustive search for nearest codebook entry
    for (uint code = 0; code < codebookSize; ++code) {
        float distance = 0.0;
        
        // Compute squared Euclidean distance to codebook entry
        for (uint dim = 0; dim < subspaceDimension; ++dim) {
            float diff = vectors[vectorOffset + dim] - 
                        codebook[codebookOffset + code * subspaceDimension + dim];
            distance += diff * diff;
        }
        
        // Track nearest codebook entry
        if (distance < minDistance) {
            minDistance = distance;
            bestCode = code;
        }
    }
    
    // Store index of nearest codebook entry
    codes[vectorIdx * numSubspaces + subspaceIdx] = static_cast<uchar>(bestCode);
}

/// Reconstruct vectors from Product Quantization codes
kernel void productDequantize(
    constant uchar* codes [[buffer(0)]],
    constant float* codebook [[buffer(1)]],
    device float* vectors [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numSubspaces [[buffer(4)]],
    constant uint& subspaceDimension [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint vectorIdx = id.x;
    uint dimIdx = id.y;
    
    if (dimIdx >= vectorDimension) return;
    
    // Determine which subspace this dimension belongs to
    uint subspaceIdx = dimIdx / subspaceDimension;
    uint subspaceDimIdx = dimIdx % subspaceDimension;
    
    // Look up the code for this subspace
    uint code = codes[vectorIdx * numSubspaces + subspaceIdx];
    
    // Calculate offset into codebook (assumes 256 codes per subspace)
    uint codebookOffset = subspaceIdx * 256 * subspaceDimension + 
                         code * subspaceDimension + subspaceDimIdx;
    
    // Copy value from codebook
    vectors[vectorIdx * vectorDimension + dimIdx] = codebook[codebookOffset];
}

// MARK: - Binary Quantization

/// Binary quantization: Extreme compression to 1 bit per dimension
kernel void binaryQuantize(
    constant float* vectors [[buffer(0)]],
    device uint* binaryVectors [[buffer(1)]],
    constant uint& vectorDimension [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint vectorIdx = id;
    // Calculate number of 32-bit words needed
    uint numWords = (vectorDimension + 31) / 32;
    
    // Process each 32-bit word
    for (uint wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        uint word = 0;
        
        // Pack 32 dimensions into one word
        for (uint bit = 0; bit < 32 && wordIdx * 32 + bit < vectorDimension; ++bit) {
            uint dimIdx = wordIdx * 32 + bit;
            float value = vectors[vectorIdx * vectorDimension + dimIdx];
            
            // Set bit if value is positive
            if (value > 0.0) {
                word |= (1u << bit);
            }
        }
        
        // Store packed bits
        binaryVectors[vectorIdx * numWords + wordIdx] = word;
    }
}

/// Compute Hamming distance between binary vectors
kernel void binaryHammingDistance(
    constant uint* queryBinary [[buffer(0)]],
    constant uint* candidateBinary [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& numWords [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint candidateIdx = id;
    uint distance = 0;
    
    // Process each 32-bit word
    for (uint wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        uint queryWord = queryBinary[wordIdx];
        uint candidateWord = candidateBinary[candidateIdx * numWords + wordIdx];
        
        // XOR gives 1 where bits differ
        uint xor_result = queryWord ^ candidateWord;
        
        // Count set bits (Hamming distance)
        distance += popcount(xor_result);
    }
    
    // Convert to float for consistency with other distance functions
    distances[candidateIdx] = static_cast<float>(distance);
}

// MARK: - Quantization Statistics

/// Compute quantization quality metrics: MSE and PSNR
kernel void computeQuantizationStats(
    constant float* original [[buffer(0)]],
    constant float* quantized [[buffer(1)]],
    device float* mse [[buffer(2)]],
    device float* psnr [[buffer(3)]],
    constant uint& vectorDimension [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint vectorIdx = id;
    uint offset = vectorIdx * vectorDimension;
    
    float sumSquaredError = 0.0;
    float maxValue = 0.0;
    
    // Compute error statistics for this vector
    for (uint dim = 0; dim < vectorDimension; ++dim) {
        float orig = original[offset + dim];
        float quant = quantized[offset + dim];
        float error = orig - quant;
        
        // Accumulate squared error
        sumSquaredError += error * error;
        
        // Track maximum absolute value (for PSNR)
        maxValue = max(maxValue, abs(orig));
    }
    
    // Mean Squared Error
    float meanSquaredError = sumSquaredError / float(vectorDimension);
    mse[vectorIdx] = meanSquaredError;
    
    // Peak Signal-to-Noise Ratio (in dB)
    if (meanSquaredError > 0.0) {
        psnr[vectorIdx] = 20.0 * log10(maxValue / sqrt(meanSquaredError));
    } else {
        // Perfect reconstruction
        psnr[vectorIdx] = INFINITY;
    }
}