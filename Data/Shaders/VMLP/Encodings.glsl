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

-- Header

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

/**
 * Global defines:
 * - OFFSET_IN, OFFSET_OUT: The read/write channel offset.
 * - NUM_CHANNELS_IN, NUM_CHANNELS_OUT: The number of input/output channels.
 * - NUM_CHANNELS_TO_ENCODE: The number of channels to encode.
 */

// Analogous to tiny-cuda-nn with column major format.
#define IDX_IN(channelIdx, batchIdx) (OFFSET_IN + (channelIdx) + (batchIdx) * NUM_CHANNELS_IN)
#define IDX_OUT(channelIdx, batchIdx) (OFFSET_OUT + (channelIdx) + (batchIdx) * NUM_CHANNELS_OUT)

layout(binding = 0, std430) readonly buffer InputBuffer {
    float inputBuffer[];
};

layout(binding = 1, std430) writeonly buffer OutputBuffer {
    real outputBuffer[];
};


-- Padding.Compute

#version 450 core

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

/**
 * Global defines:
 * - OFFSET_OUT: The write channel offset.
 * - NUM_CHANNELS_OUT: The number of output channels.
 * - NUM_CHANNELS_TO_ENCODE: The number of channels to encode.
 */

// Analogous to tiny-cuda-nn with column major format.
#define IDX_OUT(channelIdx, batchIdx) (OFFSET_OUT + (channelIdx) + (batchIdx) * NUM_CHANNELS_OUT)

layout(binding = 0, std430) writeonly buffer OutputBuffer {
    real outputBuffer[];
};

layout(push_constant) uniform PushConstants {
    uint numOutputs; // The number of outputs to be written in total.
};

void main() {
    const uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numOutputs) {
        return;
    }
    const uint batchIdx = threadIdx / NUM_CHANNELS_TO_ENCODE;
    const uint channelIdx = threadIdx % NUM_CHANNELS_TO_ENCODE;
    outputBuffer[IDX_OUT(channelIdx, batchIdx)] = real(1.0);
}


-- Identity.Compute

#version 450 core

#import ".Header"

layout(push_constant) uniform PushConstants {
    uint numOutputs; // The number of outputs to be written in total.
};

void main() {
    const uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numOutputs) {
        return;
    }
    const uint batchIdx = threadIdx / NUM_CHANNELS_TO_ENCODE;
    const uint channelIdx = threadIdx % NUM_CHANNELS_TO_ENCODE;
#if defined(OUTPUT_OP_SUM)
    outputBuffer[IDX_OUT(channelIdx, batchIdx)] += real(inputBuffer[IDX_IN(channelIdx, batchIdx)]);
#elif defined(OUTPUT_OP_PRODUCT)
    outputBuffer[IDX_OUT(channelIdx, batchIdx)] *= real(inputBuffer[IDX_IN(channelIdx, batchIdx)]);
#else
    outputBuffer[IDX_OUT(channelIdx, batchIdx)] = real(inputBuffer[IDX_IN(channelIdx, batchIdx)]);
#endif
}


-- Frequency.Compute

#version 450 core

#import ".Header"

layout(push_constant) uniform PushConstants {
    uint numOutputs; // The number of outputs to be written in total.
};

/**
 * Additional defines:
 * - PI, PI_HALF
 * - NUM_FREQUENCIES
 */
void main() {
    const uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numOutputs) {
        return;
    }
    const uint batchIdx = threadIdx / (NUM_CHANNELS_TO_ENCODE * NUM_FREQUENCIES * 2);
    const uint channelOutIdx = threadIdx % (NUM_CHANNELS_TO_ENCODE * NUM_FREQUENCIES * 2);
    const uint channelInIdx = channelOutIdx / (NUM_FREQUENCIES * 2);

    const uint powerOfTwo = (channelOutIdx / 2) % NUM_FREQUENCIES;
    const float phaseShift = (channelOutIdx % 2) * PI_HALF; // sin vs. cos

    float val = inputBuffer[IDX_IN(channelInIdx, batchIdx)] * float(1u << powerOfTwo);
    val = fma(val, PI, phaseShift); // val * PI + phaseShift
#if defined(OUTPUT_OP_SUM)
    outputBuffer[IDX_OUT(channelOutIdx, batchIdx)] += real(sin(val));
#elif defined(OUTPUT_OP_PRODUCT)
    outputBuffer[IDX_OUT(channelOutIdx, batchIdx)] *= real(sin(val));
#else
    outputBuffer[IDX_OUT(channelOutIdx, batchIdx)] = real(sin(val));
#endif
}


-- GridTranspose.Compute

#version 450 core

#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

/**
 * Global defines:
 * - OFFSET_OUT: The write channel offset.
 * - NUM_CHANNELS_OUT: The number of output channels.
 * - NUM_FEATURES: The number of channels to transpose.
 */

// Analogous to tiny-cuda-nn with column major format.
#define IDX_IN(channelIdx, batchIdx) ((batchIdx) + (channelIdx) * batchSize)
#define IDX_OUT(channelIdx, batchIdx) (OFFSET_OUT + (channelIdx) + (batchIdx) * NUM_CHANNELS_OUT)

layout(binding = 0, scalar) readonly buffer InputBuffer {
    real inputBuffer[];
};

layout(binding = 1, scalar) writeonly buffer OutputBuffer {
    real outputBuffer[];
};

layout(push_constant) uniform PushConstants {
    uint batchSize;
};

void main() {
    const uint batchIdx = gl_GlobalInvocationID.x / NUM_FEATURES;
    if (batchIdx >= batchSize) {
        return;
    }
    const uint channelIdx = gl_LocalInvocationID.x % NUM_FEATURES;
#if defined(OUTPUT_OP_SUM)
    outputBuffer[IDX_OUT(channelIdx, batchIdx)] += inputBuffer[IDX_IN(channelIdx, batchIdx)];
#elif defined(OUTPUT_OP_PRODUCT)
    outputBuffer[IDX_OUT(channelIdx, batchIdx)] *= inputBuffer[IDX_IN(channelIdx, batchIdx)];
#else
    outputBuffer[IDX_OUT(channelIdx, batchIdx)] = inputBuffer[IDX_IN(channelIdx, batchIdx)];
#endif
}


-- Grid.Compute

/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION, Christoph Neuhauser.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#version 450 core

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_control_flow_attributes : require
//#extension GL_EXT_debug_printf : enable

/**
 * Global defines:
 * - NUM_CHANNELS_IN: The number of input channels.
 * - NUM_CHANNELS_TO_ENCODE: The number of channels to encode (hard-coded to 3 for now).
 * - NUM_FEATURES_PER_LEVEL
 * - GRID_TYPE_HASH, GRID_TYPE_DENSE
 * - INTERPOLATION_TYPE_NEAREST, INTERPOLATION_TYPE_LINEAR
 * - PRIME_HASH, COHERENT_PRIME_HASH, REVERSED_PRIME_HASH
 */

// Analogous to tiny-cuda-nn with column major format.
#define IDX_IN(channelIdx, batchIdx) (OFFSET_IN + (channelIdx) + (batchIdx) * NUM_CHANNELS_IN)

layout(push_constant) uniform PushConstants {
    uint batchSize; // The number of valid threads in x direction.
};

layout(binding = 0) uniform UniformBuffer {
    uint base_resolution;
    float log2_per_level_scale;
};

layout(binding = 1, std430) readonly buffer InputBuffer {
    float positions_in[];
};

#if NUM_FEATURES_PER_LEVEL == 2
#define VALUE_TYPE real2
#elif NUM_FEATURES_PER_LEVEL == 3
#define VALUE_TYPE real3
#elif NUM_FEATURES_PER_LEVEL == 4
#define VALUE_TYPE real4
#else
#define VALUE_TYPE real
#endif

layout(binding = 2, scalar) readonly buffer GridBuffer {
    VALUE_TYPE grid[];
};

layout(binding = 3, std430) readonly buffer OffsetTableBuffer {
    uint offset_table[];
};

layout(binding = 4, scalar) writeonly buffer OutputBuffer {
    real encoded_positions[];
};

#if defined(PRIME_HASH)
const uint primes[7] = { 1958374283u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
#elif defined(COHERENT_PRIME_HASH)
const uint primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };
#elif defined(REVERSED_PRIME_HASH)
const uint primes[7] = { 2165219737u, 1434869437u, 2097192037u, 3674653429u, 805459861u, 2654435761u, 1958374283u };
#endif

uint grid_hash(uvec3 pos_grid) {
    uint result = 0;
    [[unroll]] for (uint i = 0; i < 3; ++i) {
        result ^= pos_grid[i] * primes[i];
    }
    return result;
}

uint grid_index(uint hashmap_size, uint grid_resolution, uvec3 pos_grid) {
    uint stride = 1;
    uint index = 0;

    // The second part of the loop condition is needed to avoid integer overflows in finer levels.
    [[unroll]] for (uint dim = 0; dim < 3 && stride <= hashmap_size; ++dim) {
        index += pos_grid[dim] * stride;
        stride *= grid_resolution;
    }

#ifdef GRID_TYPE_HASH
    if (hashmap_size < stride) {
        index = grid_hash(pos_grid);
    }
#endif

    return index % hashmap_size;
}

float grid_scale(uint level, float log2_per_level_scale, uint base_resolution) {
    // The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
    // than the number of cells. This is slightly different from the notation in the paper,
    // but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
    return exp2(level * log2_per_level_scale) * base_resolution - 1.0f;
}

uint grid_resolution(float scale) {
    return uint(ceil(scale)) + 1u;
}

#if NUM_FEATURES_PER_LEVEL <= 4
void grid_val(
        uvec3 local_pos, out VALUE_TYPE value,
        const uint grid_offset, const uint hashmap_size, const uint resolution) {
    const uint index = grid_index(hashmap_size, resolution, local_pos);
    value = grid[grid_offset + index];
}
#else
void grid_val(
        uvec3 local_pos, out real value[NUM_FEATURES_PER_LEVEL],
        const uint grid_offset, const uint hashmap_size, const uint resolution) {
    const uint index = grid_index(hashmap_size, resolution, local_pos) * NUM_FEATURES_PER_LEVEL;
    [[unroll]] for (uint i = 0; i < NUM_FEATURES_PER_LEVEL; i++) {
        value[i] = grid[(grid_offset + index) * NUM_FEATURES_PER_LEVEL + i];
    }
}
#endif

void main() {
    const uint i = gl_GlobalInvocationID.x;
    if (i >= batchSize) {
        return;
    }

    const uint level = gl_WorkGroupID.y; // <- the level is the same for all threads

    /*max_level = (max_level * num_grid_features) / NUM_FEATURES_PER_LEVEL;
    if (level >= max_level + 1e-3) {
        [[unroll]] for (uint f = 0; f < NUM_FEATURES_PER_LEVEL; ++f) {
            encoded_positions[i + (level * NUM_FEATURES_PER_LEVEL + f) * batchSize] = real(0.0);
        }
        return;
    }*/

    const uint grid_offset = offset_table[level];
    const uint hashmap_size = offset_table[level + 1] - offset_table[level];

    const float scale = grid_scale(level, log2_per_level_scale, base_resolution);
    const uint resolution = grid_resolution(scale);

    vec3 pos;
    vec3 pos_derivative;
    uvec3 pos_grid;

    [[unroll]] for (uint dim = 0; dim < 3; ++dim) {
        // The offset of 0.5 causes different scales to be staggered with respect to each other, thus
        // preventing spurious alignment of fractional coordinates upon integer scales (or powers thereof).
        // This is mentioned in Appendix A of the "Instant Neural Graphics Primitives" paper.
        // The offset can cause wraparound indexing in dense grids, which didn't negatively impact
        // the approximation quality in any of our tests.
        pos[dim] = fma(scale, positions_in[IDX_IN(dim, i)], 0.5);
        float tmp = floor(pos[dim]);
        pos_grid[dim] = uint(int(tmp));
        pos[dim] -= tmp;
    }

#if NUM_FEATURES_PER_LEVEL <= 4
    VALUE_TYPE result;
#else
    real result[NUM_FEATURES_PER_LEVEL];
#endif

#ifdef INTERPOLATION_TYPE_NEAREST
    grid_val(pos_grid, result, grid_offset, hashmap_size, resolution);
#else
#if NUM_FEATURES_PER_LEVEL <= 4
    result = VALUE_TYPE(0.0);
#else
    [[unroll]] for (uint f = 0; f < NUM_FEATURES_PER_LEVEL; ++f) {
        result[f] = real(0.0);
    }
#endif
#endif

#ifdef INTERPOLATION_TYPE_LINEAR
#if NUM_FEATURES_PER_LEVEL <= 4
    VALUE_TYPE tmp;
#else
    real tmp[NUM_FEATURES_PER_LEVEL];
#endif
    // N-linear interpolation
    [[unroll]] for (uint idx = 0; idx < (1 << 3); ++idx) {
        float weight = 1;
        uvec3 pos_grid_local;

        [[unroll]] for (uint dim = 0; dim < 3; ++dim) {
            if ((idx & (1u << dim)) == 0u) {
                weight *= 1 - pos[dim];
                pos_grid_local[dim] = pos_grid[dim];
            } else {
                weight *= pos[dim];
                pos_grid_local[dim] = pos_grid[dim] + 1;
            }
        }

        grid_val(pos_grid_local, tmp, grid_offset, hashmap_size, resolution);
#if NUM_FEATURES_PER_LEVEL <= 4
        result = fma(VALUE_TYPE(weight), tmp, result);
#else
        [[unroll]] for (uint f = 0; f < NUM_FEATURES_PER_LEVEL; ++f) {
            result[f] += real(weight) * tmp[f];
        }
#endif
    }
#endif

#if NUM_FEATURES_PER_LEVEL > 1
    [[unroll]] for (uint f = 0; f < NUM_FEATURES_PER_LEVEL; ++f) {
        encoded_positions[i + (level * NUM_FEATURES_PER_LEVEL + f) * batchSize] = result[f];
    }
#else
    encoded_positions[i + level * batchSize] = result;
#endif

    //debugPrintfEXT("%u %u %f %f", i, level, result.x, result.y);
}


-- Dictionary.Compute

#version 450 core

#import ".Header"

layout(push_constant) uniform PushConstants {
    uint numOutputs; // The number of outputs to be written in total.
};

layout(binding = 2, scalar) readonly buffer ParametersBuffer {
    real dictionaryBuffer[];
};

/**
 * Additional defines:
 * - NUM_EMBEDDINGS, NUM_FEATURES
 */
void main() {
    const uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numOutputs) {
        return;
    }
    const uint batchIdx = threadIdx / (NUM_CHANNELS_TO_ENCODE * NUM_FEATURES);
    const uint channelOutIdx = threadIdx % (NUM_CHANNELS_TO_ENCODE * NUM_FEATURES);
    const uint featureIdx = channelOutIdx % NUM_FEATURES;
    const uint channelInIdx = channelOutIdx / NUM_FEATURES;

    uint dictionaryEntryIdx = uint(inputBuffer[IDX_IN(channelInIdx, batchIdx)]);
    real val = dictionaryBuffer[dictionaryEntryIdx * NUM_FEATURES + featureIdx];
#if defined(OUTPUT_OP_SUM)
    outputBuffer[IDX_OUT(channelOutIdx, batchIdx)] += val;
#elif defined(OUTPUT_OP_PRODUCT)
    outputBuffer[IDX_OUT(channelOutIdx, batchIdx)] *= val;
#else
    outputBuffer[IDX_OUT(channelOutIdx, batchIdx)] = val;
#endif
}
