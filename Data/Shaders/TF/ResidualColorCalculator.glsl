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

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, es;
    uint fieldIndex0, fieldIndex1;
};

layout (binding = 1, INPUT_IMAGE_0_FORMAT) uniform readonly image3D inputImage0;
layout (binding = 2, INPUT_IMAGE_1_FORMAT) uniform readonly image3D inputImage1;
layout (binding = 3, r32f) uniform writeonly image3D outputImage;

#include "UniformData.glsl"
#include "TransferFunction.glsl"

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }

    float val0 = imageLoad(inputImage0, currentPointIdx).x;
    float val1 = imageLoad(inputImage1, currentPointIdx).x;
    float outputValue = 0.0;
    if (!isnan(val0) && !isnan(val1)) {
        vec4 volumeColor0 = transferFunction(val0, fieldIndex0);
        vec4 volumeColor1 = transferFunction(val1, fieldIndex1);
        volumeColor0.rgb *= volumeColor0.a;
        volumeColor1.rgb *= volumeColor1.a;
        vec4 vdiff = volumeColor0 - volumeColor1;
        outputValue = length(vdiff);
    }

    imageStore(outputImage, currentPointIdx, vec4(outputValue));
}
