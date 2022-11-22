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

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, es;
    vec3 boundingBoxMin;
    float minEnsembleVal;
    vec3 boundingBoxMax;
    float maxEnsembleVal;
};
layout (binding = 1) writeonly buffer OutputBuffer {
    vec4 outputBuffer[];
};
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFieldEnsembles[ENSEMBLE_MEMBER_COUNT];

layout(push_constant) uniform PushConstants {
    uint batchOffset;
    uint batchSize;
};

void main() {
    uint globalThreadIdx = gl_GlobalInvocationID.x;
    uint pointIdxWriteOffset = globalThreadIdx * es;
    uint pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint x = pointIdxReadOffset % xs;
    uint y = (pointIdxReadOffset / xs) % ys;
    uint z = pointIdxReadOffset / (xs * ys);
    if (globalThreadIdx.x >= batchSize) {
        return;
    }
    vec3 pointCoords = vec3(x, y, z) / vec3(xs - 1, ys - 1, zs - 1) * 2.0 - vec3(1.0);
    for (uint e = 0; e < es; e++) {
        float ensembleValue = texelFetch(sampler3D(
                scalarFieldEnsembles[nonuniformEXT(e)], scalarFieldSampler), ivec3(x, y, z), 0).r;
        ensembleValue = (ensembleValue - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
        outputBuffer[pointIdxWriteOffset + e] = vec4(ensembleValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}


-- Compute.Reference

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, es;
    vec3 boundingBoxMin;
    float minEnsembleVal;
    vec3 boundingBoxMax;
    float maxEnsembleVal;
};
layout (binding = 1) writeonly buffer OutputBuffer {
    vec4 outputBuffer[];
};
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFieldEnsembles[ENSEMBLE_MEMBER_COUNT];

layout(push_constant) uniform PushConstants {
    uvec3 referencePointIdx;
};

void main() {
    uint e = gl_GlobalInvocationID.x;
    if (e >= es) {
        return;
    }
    vec3 pointCoords = vec3(referencePointIdx) / vec3(xs - 1, ys - 1, zs - 1) * 2.0 - vec3(1.0);
    float ensembleValue = texelFetch(sampler3D(
            scalarFieldEnsembles[nonuniformEXT(e)], scalarFieldSampler), ivec3(referencePointIdx), 0).r;
    ensembleValue = (ensembleValue - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
    outputBuffer[e] = vec4(ensembleValue, pointCoords.x, pointCoords.y, pointCoords.z);
}
