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

layout (binding = 1) writeonly buffer OutputBuffer {
    vec4 outputBuffer[];
};

#define COMBINE_CORRELATION_MEMBERS
#include "ScalarFields.glsl"

layout(push_constant) uniform PushConstants {
    uint batchOffset;
    uint batchSize;
};

void main() {
    uint globalThreadIdx = gl_GlobalInvocationID.x;
    uint pointIdxWriteOffset = globalThreadIdx * cs;
    uint pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint x = pointIdxReadOffset % xs;
    uint y = (pointIdxReadOffset / xs) % ys;
    uint z = pointIdxReadOffset / (xs * ys);
    if (globalThreadIdx >= batchSize) {
        return;
    }
    vec3 pointCoords = vec3(x, y, z) / vec3(xs - 1, ys - 1, zs - 1) * 2.0 - vec3(1.0);
    for (uint c = 0; c < cs; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
        float fieldValue = texelFetch(sampler3D(
                scalarFields[nonuniformEXT(c)], scalarFieldSampler), ivec3(x, y, z), 0).r;
#else
        float fieldValue = scalarFields[nonuniformEXT(c)].values[IDXS(x, y, z)];
#endif
        fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
        outputBuffer[pointIdxWriteOffset + c] = vec4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}


-- Compute.Reference

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

layout (binding = 1) writeonly buffer OutputBuffer {
    vec4 outputBuffer[];
};

#define COMBINE_CORRELATION_MEMBERS
#include "ScalarFields.glsl"

layout(push_constant) uniform PushConstants {
    uvec3 referencePointIdx;
};

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= cs) {
        return;
    }
    vec3 pointCoords = vec3(referencePointIdx) / vec3(xs - 1, ys - 1, zs - 1) * 2.0 - vec3(1.0);
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
    float fieldValue = referenceValues[c];
#else
#ifdef USE_SCALAR_FIELD_IMAGES
    float fieldValue = texelFetch(sampler3D(
            scalarFields[nonuniformEXT(c)], scalarFieldSampler), ivec3(referencePointIdx), 0).r;
#else
    float fieldValue = scalarFields[nonuniformEXT(c)].values[IDXS(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z)];
#endif
#endif
    fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
    outputBuffer[c] = vec4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
}
