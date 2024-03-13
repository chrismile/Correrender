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
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, es;
};

layout (binding = 1, r32f) uniform writeonly image3D outputImage;
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFieldEnsembles[ENSEMBLE_MEMBER_COUNT];

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }

    int numValid = 0;
    float ensembleMean = 0.0f;
    float nanValue = 0.0f;
    for (int e = 0; e < es; e++) {
        float val = texelFetch(sampler3D(scalarFieldEnsembles[nonuniformEXT(e)], scalarFieldSampler), currentPointIdx, 0).r;
        if (!isnan(val)) {
            ensembleMean += val;
            numValid++;
        } else {
            nanValue = val;
        }
    }
    float stdDev;
    if (numValid > 1) {
        ensembleMean = ensembleMean / float(numValid);
        float ensembleVarianceSum = 0.0f;
        for (int e = 0; e < es; e++) {
            float val = texelFetch(sampler3D(scalarFieldEnsembles[nonuniformEXT(e)], scalarFieldSampler), currentPointIdx, 0).r;
            if (!isnan(val)) {
                float diff = ensembleMean - val;
                ensembleVarianceSum += diff * diff;
            }
        }
        stdDev = sqrt(ensembleVarianceSum / float(numValid - 1));
    } else {
        stdDev = nanValue;
    }

    imageStore(outputImage, currentPointIdx, vec4(stdDev));
}
