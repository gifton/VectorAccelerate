// VectorAccelerate: Data Transformation Shaders
//
// GPU kernels for element-wise operations and data transformations
//
// MSL Version: 4.0 (Metal 4 SDK)
// Target: macOS 26.0+, iOS 26.0+, visionOS 3.0+

#include "Metal4Common.h"

// MARK: - Part 1: Element-wise Operations

// (Spec Section: Parameters Structures - Enum)
enum ElementwiseOp : uint8_t {
    // Binary
    OP_ADD = 0, OP_SUBTRACT = 1, OP_MULTIPLY = 2, OP_DIVIDE = 3,
    OP_POWER = 4, OP_MAXIMUM = 5, OP_MINIMUM = 6,
    // Scalar
    OP_ADD_SCALAR = 10, OP_MULTIPLY_SCALAR = 11, OP_POWER_SCALAR = 12, OP_CLAMP = 13,
    // Unary
    OP_ABSOLUTE = 20, OP_SQUARE = 21, OP_SQRT = 22, OP_RECIPROCAL = 23,
    OP_NEGATE = 24, OP_EXP = 25, OP_LOG = 26
};

// (Spec Section: Parameters Structures - ElementwiseParams)
struct ElementwiseParams {
    uint32_t num_elements;
    uint32_t stride_a;
    uint32_t stride_b;
    uint32_t stride_output;
    float scalar_value;
    float scalar_value2;     // clamp (min)
    float scalar_value3;     // clamp (max)
    uint8_t operation;
    uint8_t use_fast_math;
    uint8_t padding[2];
};

// Helper function to execute the operation
float perform_elementwise_operation(float a, float b, constant ElementwiseParams& params) {
    // 'b' is pre-loaded with either the second vector element or the scalar value.
    const bool use_fast = params.use_fast_math;

    switch (params.operation) {
        case OP_ADD: case OP_ADD_SCALAR: return a + b;
        case OP_SUBTRACT: return a - b;
        case OP_MULTIPLY: case OP_MULTIPLY_SCALAR: return a * b;
        case OP_DIVIDE:
            return use_fast ? fast::divide(a, b) : (a / b);
        case OP_POWER: case OP_POWER_SCALAR:
            return use_fast ? fast::pow(a, b) : pow(a, b);
        case OP_MAXIMUM: return max(a, b);
        case OP_MINIMUM: return min(a, b);
        case OP_CLAMP:
            return clamp(a, params.scalar_value2, params.scalar_value3);
        case OP_ABSOLUTE: return abs(a);
        case OP_SQUARE: return a * a;
        case OP_SQRT:
            return use_fast ? fast::sqrt(a) : sqrt(a);
        case OP_RECIPROCAL:
            return use_fast ? fast::divide(1.0f, a) : (1.0f / a);
        case OP_NEGATE: return -a;
        case OP_EXP:
            return use_fast ? fast::exp(a) : exp(a);
        case OP_LOG:
            return use_fast ? fast::log(a) : log(a);
        default: return a;
    }
}

// (Spec Section: Metal Kernel Signatures)
kernel void elementwise_operation_kernel(
    device const float* input_a [[buffer(0)]],
    device const float* input_b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ElementwiseParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    // Calculate indices based on strides
    const uint idx_a = tid * params.stride_a;
    const uint idx_out = tid * params.stride_output;

    float a = input_a[idx_a];
    float b;

    // Determine 'b'. If input_b is provided (binary), use it; otherwise use scalar (scalar/unary).
    if (input_b != nullptr) {
        const uint idx_b = tid * params.stride_b;
        b = input_b[idx_b];
    } else {
        b = params.scalar_value;
    }

    output[idx_out] = perform_elementwise_operation(a, b, params);
}

kernel void elementwise_inplace_kernel(
    device float* data [[buffer(0)]],
    device const float* operand [[buffer(1)]],
    constant ElementwiseParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    const uint idx_a = tid * params.stride_a;
    float a = data[idx_a];
    float b;

    if (operand != nullptr) {
        const uint idx_b = tid * params.stride_b;
        b = operand[idx_b];
    } else {
        b = params.scalar_value;
    }

    data[idx_a] = perform_elementwise_operation(a, b, params);
}

// MARK: - Part 2: Scalar Quantization

// (Spec Section: Parameters Structures)
enum QuantizationType : uint8_t {
    QUANT_SYMMETRIC = 0,
    QUANT_ASYMMETRIC = 1,
    QUANT_PER_CHANNEL = 2,
    QUANT_DYNAMIC = 3 // Treated as PER_CHANNEL if scales are pre-computed
};

struct QuantizationParams {
    uint32_t num_elements;
    uint32_t num_channels;
    uint32_t elements_per_channel;
    float global_scale;
    int8_t global_zero_point;
    uint8_t quantization_type;
    uint8_t bit_width;
    uint8_t padding;
};

// Helper to determine scale and zero point (Global vs Per-Channel)
void get_quant_params(
    constant QuantizationParams& params,
    device const float* scales,
    device const int8_t* zero_points,
    uint tid,
    thread float& scale,
    thread int8_t& zero_point
) {
    bool use_per_channel = (params.quantization_type == QUANT_PER_CHANNEL || params.quantization_type == QUANT_DYNAMIC);

    if (use_per_channel && scales != nullptr && params.elements_per_channel > 0) {
        uint channel = tid / params.elements_per_channel;
        // Ensure channel index is valid
        if (channel < params.num_channels) {
            scale = scales[channel];
            // Retrieve zero point if available, otherwise assume 0 (symmetric)
            zero_point = (zero_points != nullptr) ? zero_points[channel] : 0;
            return;
        }
    }

    // Fallback to Global/Static parameters
    scale = params.global_scale;
    // Ensure symmetric type uses ZP=0 unless overridden by global_zero_point (which shouldn't happen if spec followed)
    if (params.quantization_type == QUANT_SYMMETRIC) {
        zero_point = 0;
    } else {
        zero_point = params.global_zero_point;
    }
}

// (Spec Section: Metal Kernel Signatures)
kernel void quantize_int8_kernel(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const int8_t* zero_points [[buffer(3)]],
    constant QuantizationParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    float scale;
    int8_t zero_point;
    get_quant_params(params, scales, zero_points, tid, scale, zero_point);

    // Handle near-zero scale defensively
    if (abs(scale) < 1e-9f) {
        output[tid] = zero_point;
        return;
    }

    float value = input[tid];
    // Formula: q = round(x / scale + zero_point)
    float scaled_value = value / scale + float(zero_point);

    // Rounding (rint: round-to-nearest-even minimizes bias) and Clamping to [-128, 127]
    float clamped_value = clamp(rint(scaled_value), -128.0f, 127.0f);
    output[tid] = int8_t(clamped_value);
}

kernel void dequantize_int8_kernel(
    device const int8_t* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const int8_t* zero_points [[buffer(3)]],
    constant QuantizationParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.num_elements) return;

    float scale;
    int8_t zero_point;
    get_quant_params(params, scales, zero_points, tid, scale, zero_point);

    int8_t quantized_value = input[tid];

    // Dequantization: x = (q - zero_point) * scale
    output[tid] = (float(quantized_value) - float(zero_point)) * scale;
}

// INT4 Quantization (Supports Per-Channel)
kernel void quantize_int4_kernel(
    device const float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const int8_t* zero_points [[buffer(3)]],
    constant QuantizationParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]] // tid is the output byte index
) {
    uint idx0 = tid * 2;
    if (idx0 >= params.num_elements) return;
    uint idx1 = idx0 + 1;

    // 1. Retrieve parameters for both elements
    float scale0, scale1;
    int8_t zp0, zp1;

    get_quant_params(params, scales, zero_points, idx0, scale0, zp0);

    if (idx1 < params.num_elements) {
        get_quant_params(params, scales, zero_points, idx1, scale1, zp1);
    } else {
        // Padding case: use params from idx0
        scale1 = scale0; zp1 = zp0;
    }

    // 2. Load Values
    float val0 = input[idx0];
    float val1 = (idx1 < params.num_elements) ? input[idx1] : 0.0f;

    // 3. Quantize
    // Define range for Signed INT4: [-8, 7]
    const float q_min = -8.0f;
    const float q_max = 7.0f;

    // Apply quantization formula, handling near-zero scale
    float scaled0 = (abs(scale0) < 1e-9f) ? float(zp0) : (val0 / scale0 + float(zp0));
    float scaled1 = (abs(scale1) < 1e-9f) ? float(zp1) : (val1 / scale1 + float(zp1));

    // Round (rint) and Clamp. Use int8_t (char) as proxy for INT4.
    int8_t q0 = int8_t(clamp(rint(scaled0), q_min, q_max));
    int8_t q1 = int8_t(clamp(rint(scaled1), q_min, q_max));

    // 4. Pack: q0 in lower nibble, q1 in upper nibble.
    // Mask with 0xF to handle two's complement correctly.
    output[tid] = ((uchar(q1) & 0xF) << 4) | (uchar(q0) & 0xF);
}

// INT4 Dequantization (Supports Per-Channel)
kernel void dequantize_int4_kernel(
    device const uchar* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const int8_t* zero_points [[buffer(3)]],
    constant QuantizationParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]] // tid is the input byte index
) {
    uint idx0 = tid * 2;
    if (idx0 >= params.num_elements) return;
    uint idx1 = idx0 + 1;

    // 1. Unpack
    uchar packed_byte = input[tid];
    int8_t q0 = packed_byte & 0xF;
    int8_t q1 = (packed_byte >> 4) & 0xF;

    // 2. Sign extension for Signed INT4 (values >= 8 are negative)
    // If the 4th bit (0x08) is set, set the upper bits of the int8_t to 1.
    if (q0 & 0x08) q0 |= 0xF0;
    if (q1 & 0x08) q1 |= 0xF0;

    // 3. Retrieve parameters and Dequantize
    float scale0, scale1;
    int8_t zp0, zp1;

    get_quant_params(params, scales, zero_points, idx0, scale0, zp0);
    output[idx0] = (float(q0) - float(zp0)) * scale0;

    // Handle second element boundary
    if (idx1 < params.num_elements) {
        get_quant_params(params, scales, zero_points, idx1, scale1, zp1);
        output[idx1] = (float(q1) - float(zp1)) * scale1;
    }
}
