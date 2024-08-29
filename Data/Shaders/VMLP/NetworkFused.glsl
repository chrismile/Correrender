/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

-- Compute

#version 450 core

layout(local_size_x = SUBGROUP_SIZE, local_size_y = NUM_ROWS, local_size_z = 1) in;

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_memory_scope_semantics : enable
//#extension GL_EXT_debug_printf : enable

/**
 * Vulkan GLSL implementation of a fully-fused MLP as described in:
 * Müller, T., F. Rousselle, J. Novák, and A. Keller (2021). "Real-Time Neural Radiance Caching for Path Tracing".
 * In: ACM Trans. Graph. 40.4. ISSN: 0730-0301. DOI: https://doi.org/10.1145/3450626.3459812.
 *
 * Performs a batch-wise (B) multiplication of the weight matrices of each MLP layer by the inputs of the layer.
 * - Layers: X' = W * X, R^{N_out x B} = R^{N_out x N_in} * R^{N_in x B} + R^{N_out x B}
 * - Subgroup operations:  R^{M x N} = R^{M x K} * R^{K x N} + R^{M x N}
 *
 * Global defines:
 * - M, K, N: The matrix block sizes (usually M = K = N = 16 on NVIDIA hardware).
 * - SUBGROUP_SIZE: The subgroup size.
 * - NUM_ROWS: The number of subgroup blocks (NUM_CHANNELS_HIDDEN / M) in the weight/output row direction.
 * - NUM_BATCHES: The number of batch blocks (multiples of N).
 * - SHARED_MEMORY_SIZE: The number of half-precision float elements in shared memory (NUM_BATCHES * N * NUM_CHANNELS).
 * - NUM_LAYERS: Total number of layers.
 * - NUM_CHANNELS_IN, NUM_CHANNELS_OUT: The number of input/output channels.
 * - NUM_CHANNELS_IN_PADDED, NUM_CHANNELS_OUT_PADDED: The number of input/output channels (padded).
 * - NUM_CHANNELS_HIDDEN: The number of channels in the hidden layers.
 * - ACTIVATION_FUNCTION: Name of the activation function.
 * - NO_OUTPUT_ACTIVATION: Defined if no activation function should be used after the last layer.
 */

/*
 * There are some resources online suggesting that using a type != float16_t for shared memory has advantages:
 * - https://github.com/jeffbolznv/vk_cooperative_matrix_perf/blob/master/shaders/shmem.comp
 * - https://github.com/KhronosGroup/glslang/blob/4605e2ed2b2b1acbe157d365c3c528367b8b168f/Test/spv.coopmat.comp
 */
//#define SMEM_FACTOR 8 // sizeof(uvec4) / sizeof(float16_t)
//#define STORAGE_TYPE uvec4
//#define SMEM_FACTOR 1 // sizeof(float16_t) / sizeof(float16_t)
//#define STORAGE_TYPE float16_t
shared STORAGE_TYPE sharedMemory[SHARED_MEMORY_SIZE / SMEM_FACTOR];

//#define WEIGHT_IDX(channelOutIdx, channelInIdx) (WEIGHT_OFFSET + (channelInIdx) + (channelOutIdx) * NUM_CHANNELS_IN_PADDED)
#define IDX_IN(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_IN)
#ifdef FLOAT16_NO_PADDING
#define IDX_OUT(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_OUT)
#else
#define IDX_OUT(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_OUT_PADDED)
#endif

// Row major.
layout(binding = 0, scalar) readonly buffer ParametersBuffer {
    float16_t parametersBuffer[];
};

// Column major.
layout(binding = 1, scalar) readonly buffer InputBuffer {
    STORAGE_TYPE inputBuffer[];
};

// Column major.
layout(binding = 2, scalar) writeonly buffer OutputBuffer {
    STORAGE_TYPE outputBuffer[];
};

layout(push_constant) uniform PushConstants {
    uint batchSize, inputBufferSizeTyped, outputBufferSizeTyped;
};

#define real float16_t
#include "ActivationFunctions.glsl"

void main() {
    const uint localThreadIdx = gl_LocalInvocationID.x;
    const uint subgroupIdx = gl_WorkGroupID.x;
    const uint blockRowIdx = gl_LocalInvocationID.y;
    const uint batchOffset = subgroupIdx * (N * NUM_BATCHES / SMEM_FACTOR);

#ifndef USE_CUSTOM_LAYERS_CODE
    // Load inputs into shared memory (layout: NUM_CHANNELS_IN_PADDED rows, NUM_BATCHES cols).
#ifdef FLOAT16_NO_PADDING
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_IN * N * NUM_BATCHES; i += SUBGROUP_SIZE * NUM_ROWS) {
        const uint channelIdx = i % NUM_CHANNELS_IN;
        const uint batchIdxLocal = i / NUM_CHANNELS_IN;
        const uint batchIdxGlobal = batchOffset + batchIdxLocal;
        if (batchIdxGlobal < batchSize) {
#ifdef BANK_SKEW
            sharedMemory[channelIdx + batchIdxLocal * (NUM_CHANNELS_IN_PADDED + BANK_SKEW)] = inputBuffer[IDX_IN(channelIdx, batchIdxGlobal)];
#else
            sharedMemory[channelIdx + batchIdxLocal * NUM_CHANNELS_IN_PADDED] = inputBuffer[IDX_IN(channelIdx, batchIdxGlobal)];
#endif
        }
    }
#else
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_IN_PADDED * N * NUM_BATCHES / SMEM_FACTOR; i += SUBGROUP_SIZE * NUM_ROWS) {
        if (i < inputBufferSizeTyped) {
#ifdef BANK_SKEW
            const uint channelIdx = i % NUM_CHANNELS_IN_PADDED;
            const uint batchIdxLocal = i / NUM_CHANNELS_IN_PADDED;
            sharedMemory[channelIdx + batchIdxLocal * (NUM_CHANNELS_IN_PADDED + BANK_SKEW)] = inputBuffer[i + batchOffset * NUM_CHANNELS_IN_PADDED];
#else
            sharedMemory[i] = inputBuffer[i + batchOffset * NUM_CHANNELS_IN_PADDED];
#endif
        }
    }
#endif
    memoryBarrierShared();
    barrier();
#endif

    CoopMatA weightsMat; // row major
    CoopMatB inputMat; // column major
    CoopMatAcc outputMat[NUM_BATCHES];

    uint weightOffsetBase = 0;
    uint weightStride = 0;
    uint inputOffset = 0;
    uint inputStride = 0;
    uint outputOffset = 0;
    uint outputStride = 0;

#ifdef USE_CUSTOM_LAYERS_CODE
#codefrag "CUSTOM_LAYERS_CODE"
#else
#define LAYER_USE_SHARED_MEMORY_INPUT
#define LAYER_USE_SHARED_MEMORY_OUTPUT
    [[unroll]] for (uint layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++)
#include "NetworkFusedLayer.glsl"
#endif

#ifndef USE_CUSTOM_LAYERS_CODE
    // Write outputs into global memory
#ifdef FLOAT16_NO_PADDING
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_OUT * N * NUM_BATCHES; i += SUBGROUP_SIZE * NUM_ROWS) {
        const uint channelIdx = i % NUM_CHANNELS_OUT;
        const uint batchIdxLocal = i / NUM_CHANNELS_OUT;
        const uint batchIdxGlobal = batchOffset + batchIdxLocal;
        if (batchIdxGlobal < batchSize) {
#ifdef BANK_SKEW
            outputBuffer[IDX_OUT(channelIdx, batchIdxGlobal)] = sharedMemory[channelIdx + batchIdxLocal * (NUM_CHANNELS_OUT_PADDED + BANK_SKEW)];
#else
            outputBuffer[IDX_OUT(channelIdx, batchIdxGlobal)] = sharedMemory[channelIdx + batchIdxLocal * NUM_CHANNELS_OUT_PADDED];
#endif
        }
    }
#else
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_OUT_PADDED * N * NUM_BATCHES / SMEM_FACTOR; i += SUBGROUP_SIZE * NUM_ROWS) {
        //if (i < outputBufferSizeTyped) {
#ifdef BANK_SKEW
        const uint channelIdx = i % NUM_CHANNELS_OUT_PADDED;
        const uint batchIdxLocal = i / NUM_CHANNELS_OUT_PADDED;
        outputBuffer[i + batchOffset * NUM_CHANNELS_OUT_PADDED] = sharedMemory[channelIdx + batchIdxLocal * (NUM_CHANNELS_OUT_PADDED + BANK_SKEW)];
#else
        outputBuffer[i + batchOffset * NUM_CHANNELS_OUT_PADDED] = sharedMemory[i];
#endif
        //}
    }
#endif
#endif
}


-- Compute.SingleSource

#version 450 core

layout(local_size_x = SUBGROUP_SIZE, local_size_y = NUM_ROWS, local_size_z = 1) in;

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_memory_scope_semantics : enable
//#extension GL_EXT_debug_printf : enable

/**
 * Vulkan GLSL implementation of a fully-fused MLP as described in:
 * Müller, T., F. Rousselle, J. Novák, and A. Keller (2021). "Real-Time Neural Radiance Caching for Path Tracing".
 * In: ACM Trans. Graph. 40.4. ISSN: 0730-0301. DOI: https://doi.org/10.1145/3450626.3459812.
 *
 * Performs a batch-wise (B) multiplication of the weight matrices of each MLP layer by the inputs of the layer.
 * - Layers: X' = W * X, R^{N_out x B} = R^{N_out x N_in} * R^{N_in x B} + R^{N_out x B}
 * - Subgroup operations:  R^{M x N} = R^{M x K} * R^{K x N} + R^{M x N}
 *
 * Global defines:
 * - M, K, N: The matrix block sizes (usually M = K = N = 16 on NVIDIA hardware).
 * - SUBGROUP_SIZE: The subgroup size.
 * - NUM_ROWS: The number of subgroup blocks (NUM_CHANNELS_HIDDEN / M) in the weight/output row direction.
 * - NUM_BATCHES: The number of batch blocks (multiples of N).
 * - SHARED_MEMORY_SIZE: The number of half-precision float elements in shared memory (NUM_BATCHES * N * NUM_CHANNELS).
 * - NUM_LAYERS: Total number of layers.
 * - NUM_CHANNELS_IN, NUM_CHANNELS_OUT: The number of input/output channels.
 * - NUM_CHANNELS_IN_PADDED, NUM_CHANNELS_OUT_PADDED: The number of input/output channels (padded).
 * - NUM_CHANNELS_HIDDEN: The number of channels in the hidden layers.
 * - ACTIVATION_FUNCTION: Name of the activation function.
 * - NO_OUTPUT_ACTIVATION: Defined if no activation function should be used after the last layer.
 */

/*
 * There are some resources online suggesting that using a type != float16_t for shared memory has advantages:
 * - https://github.com/jeffbolznv/vk_cooperative_matrix_perf/blob/master/shaders/shmem.comp
 * - https://github.com/KhronosGroup/glslang/blob/4605e2ed2b2b1acbe157d365c3c528367b8b168f/Test/spv.coopmat.comp
 */
//#define SMEM_FACTOR 8 // sizeof(uvec4) / sizeof(float16_t)
//#define STORAGE_TYPE uvec4
//#define SMEM_FACTOR 1 // sizeof(float16_t) / sizeof(float16_t)
//#define STORAGE_TYPE float16_t
shared STORAGE_TYPE sharedMemory[SHARED_MEMORY_SIZE / SMEM_FACTOR];

//#define WEIGHT_IDX(channelOutIdx, channelInIdx) (WEIGHT_OFFSET + (channelInIdx) + (channelOutIdx) * NUM_CHANNELS_IN_PADDED)
#define IDX_IN(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_IN)
#ifdef FLOAT16_NO_PADDING
#define IDX_OUT(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_OUT)
#else
#define IDX_OUT(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_OUT_PADDED)
#endif

// Row major.
layout(binding = 0, scalar) readonly buffer ParametersBuffer {
    float16_t parametersBuffer[];
};

// Column major.
layout(binding = 1, scalar) readonly buffer InputBuffer {
    STORAGE_TYPE inputBuffer[];
};

// Column major.
layout(binding = 2, scalar) writeonly buffer OutputBuffer {
    STORAGE_TYPE outputBuffer[];
};

layout(push_constant) uniform PushConstants {
    uint batchSize, inputBufferSizeTyped, outputBufferSizeTyped;
};

#define real float16_t
#include "ActivationFunctions.glsl"

void main() {
    const uint localThreadIdx = gl_LocalInvocationID.x;
    const uint subgroupIdx = gl_WorkGroupID.x;
    const uint blockRowIdx = gl_LocalInvocationID.y;
    const uint batchOffset = subgroupIdx * (N * NUM_BATCHES / SMEM_FACTOR);

    // Load inputs into shared memory (layout: NUM_CHANNELS_IN_PADDED rows, NUM_BATCHES cols).
#ifdef FLOAT16_NO_PADDING
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_IN * N * NUM_BATCHES; i += SUBGROUP_SIZE * NUM_ROWS) {
        const uint channelIdx = i % NUM_CHANNELS_IN;
        const uint batchIdxLocal = i / NUM_CHANNELS_IN;
        const uint batchIdxGlobal = batchOffset + batchIdxLocal;
        if (batchIdxGlobal < batchSize) {
            sharedMemory[channelIdx + batchIdxLocal * NUM_CHANNELS_IN_PADDED] = inputBuffer[IDX_IN(channelIdx, batchIdxGlobal)];
        }
    }
#else
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_IN_PADDED * N * NUM_BATCHES / SMEM_FACTOR; i += SUBGROUP_SIZE * NUM_ROWS) {
        if (i < inputBufferSizeTyped) {
            sharedMemory[i] = inputBuffer[i + batchOffset * NUM_CHANNELS_IN_PADDED];
        }
    }
#endif
    memoryBarrierShared();
    barrier();

    //if (localThreadIdx == 0 && blockRowIdx == 0) {
    //    debugPrintfEXT("%f", float(sharedMemory[0]));
    //}

    CoopMatA weightsMat; // row major
    CoopMatB inputMat; // column major
    CoopMatAcc outputMat[NUM_BATCHES];

    uint weightOffsetBase = 0;
    uint weightStride = 0;
    uint inputOffset = 0;
    uint inputStride = 0;
    uint outputOffset = 0;
    uint outputStride = 0;

    [[unroll]] for (uint layerIdx = 0; layerIdx < NUM_LAYERS; layerIdx++) {
        const uint numChannelsInPadded = layerIdx == 0 ? NUM_CHANNELS_IN_PADDED : NUM_CHANNELS_HIDDEN;
        const uint numChannelsOutPadded = layerIdx == NUM_LAYERS - 1 ? NUM_CHANNELS_OUT_PADDED : NUM_CHANNELS_HIDDEN;
        const uint numCols = numChannelsInPadded / K;
        const uint numRows = numChannelsOutPadded / M; // May be less than NUM_ROWS for the last layer.
        weightStride = numChannelsInPadded;
        inputStride = numChannelsInPadded / SMEM_FACTOR;
        outputStride = numChannelsOutPadded / SMEM_FACTOR;

        if (blockRowIdx < numRows) {
            // Clear the output matrices.
            [[unroll]] for (uint b = 0; b < NUM_BATCHES; b++) {
                outputMat[b] = CoopMatAcc(0.0);
            }

            for (uint c = 0; c < numCols; c++) {
                const uint weightOffset = weightOffsetBase + c * K + blockRowIdx * M * weightStride;
                matLoad(weightsMat, parametersBuffer, weightOffset, weightStride, ROW_MAJOR);
                [[unroll]] for (uint b = 0; b < NUM_BATCHES; b++) {
                    inputOffset = c * (K / SMEM_FACTOR) + b * N * inputStride;
                    matLoad(inputMat, sharedMemory, inputOffset, inputStride, COL_MAJOR);
                    outputMat[b] = matMulAdd(weightsMat, inputMat, outputMat[b]);
                }
            }

            // Apply activation function.
#ifdef NO_OUTPUT_ACTIVATION
            if (layerIdx != NUM_LAYERS - 1) {
#endif
                [[unroll]] for (uint b = 0; b < NUM_BATCHES; b++) {
                    for (uint i = 0; i < outputMat[b].length(); ++i) {
                        outputMat[b][i] = ACTIVATION_FUNCTION(outputMat[b][i]);
                    }
                }
#ifdef NO_OUTPUT_ACTIVATION
            }
#endif
        }

        barrier();

        if (blockRowIdx < numRows) {
            // Store to shared memory.
            [[unroll]] for (uint b = 0; b < NUM_BATCHES; b++) {
                outputOffset = blockRowIdx * (M / SMEM_FACTOR) + b * N * outputStride;
                matStore(outputMat[b], sharedMemory, outputOffset, outputStride, COL_MAJOR);
            }
        }

        memoryBarrierShared();
        barrier();

        // Update offsets.
        weightOffsetBase += numChannelsOutPadded * numChannelsInPadded;
    }

    // Write outputs into global memory
#ifdef FLOAT16_NO_PADDING
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_OUT * N * NUM_BATCHES; i += SUBGROUP_SIZE * NUM_ROWS) {
        const uint channelIdx = i % NUM_CHANNELS_OUT;
        const uint batchIdxLocal = i / NUM_CHANNELS_OUT;
        const uint batchIdxGlobal = batchOffset + batchIdxLocal;
        if (batchIdxGlobal < batchSize) {
            outputBuffer[IDX_OUT(channelIdx, batchIdxGlobal)] = sharedMemory[channelIdx + batchIdxLocal * NUM_CHANNELS_OUT_PADDED];
        }
    }
#else
    for (uint i = localThreadIdx + blockRowIdx * SUBGROUP_SIZE; i < NUM_CHANNELS_OUT_PADDED * N * NUM_BATCHES / SMEM_FACTOR; i += SUBGROUP_SIZE * NUM_ROWS) {
        //if (i < outputBufferSizeTyped) {
        outputBuffer[i + batchOffset * NUM_CHANNELS_OUT_PADDED] = sharedMemory[i];
        //}
    }
#endif
}
