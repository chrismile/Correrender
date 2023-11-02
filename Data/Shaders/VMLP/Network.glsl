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
 * - WEIGHT_OFFSET: The offset into the weights buffer.
 * - NUM_CHANNELS_IN, NUM_CHANNELS_OUT: The number of input/output channels.
 * - NUM_CHANNELS_IN_PADDED, NUM_CHANNELS_OUT_PADDED: The number of input/output channels (padded).
 * - ACTIVATION_FUNCTION: Name of the activation function.
 */

// Analogous to tiny-cuda-nn with column major format.
#define WEIGHT_IDX(channelOutIdx, channelInIdx) (WEIGHT_OFFSET + (channelInIdx) + (channelOutIdx) * NUM_CHANNELS_IN_PADDED)
#define IDX_IN(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_IN)
#define IDX_OUT(channelIdx, batchIdx) ((channelIdx) + (batchIdx) * NUM_CHANNELS_OUT)

layout(binding = 0, std430) readonly buffer ParametersBuffer {
    real parametersBufferGlobal[];
};

layout(binding = 1, std430) readonly buffer InputBuffer {
    real inputBuffer[];
};

layout(binding = 2, std430) writeonly buffer OutputBuffer {
    real outputBuffer[];
};


-- GlobalMemory.Compute

#version 450 core

#import ".Header"
#include "ActivationFunctions.glsl"

layout(push_constant) uniform PushConstants {
    uint numOutputs; // The number of outputs to be written in total.
};

void main() {
    const uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numOutputs) {
        return;
    }
    const uint batchIdx = threadIdx / NUM_CHANNELS_OUT;
    const uint channelOutIdx = threadIdx % NUM_CHANNELS_OUT;

    real valueOut = real(0.0);
    for (uint channelInIdx = 0; channelInIdx < NUM_CHANNELS_IN; channelInIdx++) {
        real weight = parametersBufferGlobal[WEIGHT_IDX(channelOutIdx, channelInIdx)];
        real valueIn = inputBuffer[IDX_IN(channelInIdx, batchIdx)];
        valueOut = fma(weight, valueIn, valueOut);
    }

#ifdef ACTIVATION_FUNCTION
    valueOut = ACTIVATION_FUNCTION(valueOut);
#endif

    outputBuffer[IDX_OUT(channelOutIdx, batchIdx)] = valueOut;
}
