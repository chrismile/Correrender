/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout (binding = 1, r32f) uniform writeonly image3D outputImage;

#define SET_PREDICATE_CALCULATOR
#include "ScalarFields.glsl"

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }

#ifndef USE_SCALAR_FIELD_IMAGES
    uint linearPointIdx = IDXS(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

    int count = 0;
    for (int c = 0; c < cs; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
        float val = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
#else
        float val = scalarFields[nonuniformEXT(c)].values[linearPointIdx];
#endif
        if (comparator(val, comparisonValue)) {
            count++;
        }
    }

    float outputValue;
    if (countLower == countUpper) {
        outputValue = clamp(float(count) - float(countLower), 0.0, 1.0);
    } else {
        outputValue = clamp((float(count) - float(countLower)) / (float(countUpper) - float(countLower)), 0.0, 1.0);
    }
    imageStore(outputImage, currentPointIdx, vec4(outputValue));
}
