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

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, cs;
};
layout (binding = 1, r32f) uniform writeonly image3D outputImage;
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFields[MEMBER_COUNT];

layout(push_constant) uniform PushConstants {
    ivec3 referencePointIdx;
    int padding0;
    uvec3 batchOffset;
    uint padding1;
};

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz + batchOffset);
    if (currentPointIdx.x >= xs || currentPointIdx.y >= ys || currentPointIdx.z >= zs) {
        return;
    }

    float n = float(cs);
    float meanX = 0;
    float meanY = 0;
    float invN = float(1) / n;
    for (uint c = 0; c < cs; c++) {
        float x = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
        float y = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
        meanX += invN * x;
        meanY += invN * y;
    }
    float varX = 0;
    float varY = 0;
    float invNm1 = float(1) / (n - float(1));
    for (uint c = 0; c < cs; c++) {
        float x = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
        float y = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
        float diffX = x - meanX;
        float diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    float stdDevX = sqrt(varX);
    float stdDevY = sqrt(varY);
    float pearsonCorrelation = 0;
    for (uint c = 0; c < cs; c++) {
        float x = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
        float y = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }

    imageStore(outputImage, currentPointIdx, vec4(pearsonCorrelation));
}
