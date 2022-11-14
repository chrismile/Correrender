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

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, es;
    vec3 boundingBoxMin;
    float padding0;
    vec3 boundingBoxMax;
    float padding1;
};
layout (binding = 1) writeonly buffer OutputBuffer {
    vec4 outputBuffer[];
};
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFieldEnsembles[];

void main() {
    uvec3 pointIdx = gl_GlobalInvocationID;
    if (pointIdx.x >= xs || pointIdx.y >= ys || pointIdx.z >= zs) {
        return;
    }
    vec3 pointCoords =
            vec3(pointIdx) / vec3(xs - 1, ys - 1, zs - 1) * (boundingBoxMax - boundingBoxMin) + boundingBoxMin;
    pointCoords = pointCoords * 2.0 - vec3(1.0);
    uint pointOffset = (pointIdx.x + (pointIdx.y + pointIdx.z * ys) * xs) * es;
    for (uint e = 0; e < es; e++) {
        float ensembleValue = texelFetch(sampler3D(scalarFieldEnsembles[e], scalarFieldSampler), ivec3(pointIdx), 0).r;
        outputBuffer[pointOffset + e] = vec4(pointCoords, ensembleValue);
    }
}
