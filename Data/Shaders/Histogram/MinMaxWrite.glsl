/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

layout (local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint sizeOfInput; ///< Number of entries in MinMaxInBuffer.
};

// Size: sizeOfInput
layout (std430, binding = 0) readonly buffer DataBuffer {
    float dataBuffer[];
};

// Size: iceil(sizeOfInput, BLOCK_SIZE)
layout (std430, binding = 1) writeonly buffer MinMaxOutBuffer {
    vec2 minMaxOutBuffer[];
};

shared vec2 sharedMemoryMinMax[BLOCK_SIZE];

void main() {
    uint localIdx = gl_LocalInvocationID.x;
    uint localDim = gl_WorkGroupSize.x; // == BLOCK_SIZE
    uint reductionIdx = gl_WorkGroupID.x;

    vec2 minMaxVal = vec2(FLT_MAX, FLT_LOWEST);
    if (gl_GlobalInvocationID.x < sizeOfInput) {
        float value = dataBuffer[gl_GlobalInvocationID.x];
        minMaxVal = vec2(value, value);
    }

    sharedMemoryMinMax[localIdx] = minMaxVal;
    memoryBarrierShared();
    barrier();

    for (uint stride = localDim / 2; stride > 0; stride >>= 1) {
        if (localIdx < stride) {
            vec2 minMaxDepth0 = sharedMemoryMinMax[localIdx];
            vec2 minMaxDepth1 = sharedMemoryMinMax[localIdx + stride];
            sharedMemoryMinMax[localIdx] = vec2(
                    min(minMaxDepth0.x, minMaxDepth1.x),
                    max(minMaxDepth0.y, minMaxDepth1.y)
            );
        }
        memoryBarrierShared();
        barrier();
    }

    if (localIdx == 0) {
        minMaxOutBuffer[reductionIdx] = sharedMemoryMinMax[0];
    }
}
