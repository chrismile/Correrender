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

-- Compute.Voxels

#version 450 core

#extension GL_EXT_shader_atomic_float : require
//#extension GL_EXT_debug_printf : enable

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout(binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, tfNumEntries;
    float minGT, maxGT, minOpt, maxOpt;
};

layout(push_constant) uniform PushConstants {
    float Nj;
};

layout(binding = 1, INPUT_IMAGE_0_FORMAT) uniform readonly image3D inputImageGT;
layout(binding = 2, INPUT_IMAGE_1_FORMAT) uniform readonly image3D inputImageOpt;

layout(binding = 3) readonly buffer TfGTBuffer {
    vec4 tfGT[];
};

layout(binding = 4, std430) readonly buffer TfOptBuffer {
    vec4 tfOpt[];
};

layout(binding = 5, std430) buffer TfOptGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float g[];
#else
    uint g[];
#endif
};

void atomicAddGrad(uint idx, float value) {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    atomicAdd(g[idx], value);
#else
    uint oldValue = g[idx];
    uint expectedValue, newValue;
    do {
        expectedValue = oldValue;
        newValue = floatBitsToUint(uintBitsToFloat(oldValue) + value);
        oldValue = atomicCompSwap(g[idx], expectedValue, newValue);
    } while (oldValue != expectedValue);
#endif
}

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (currentPointIdx.x >= xs || currentPointIdx.y >= ys || currentPointIdx.z >= zs) {
        return;
    }
    float scalarGT = imageLoad(inputImageGT, currentPointIdx).x;
    float scalarOpt = imageLoad(inputImageOpt, currentPointIdx).x;
    if (isnan(scalarGT) || isnan(scalarOpt)) {
        return;
    }

    float tGT = (scalarGT - minGT) / (maxGT - minGT);
    float tGT0 = clamp(floor(tGT * Nj), 0.0f, Nj);
    float tGT1 = clamp(ceil(tGT * Nj), 0.0f, Nj);
    float fGT = tGT * Nj - tGT0;
    int jGT0 = int(tGT0);
    int jGT1 = int(tGT1);
    vec4 cGT0 = tfGT[jGT0];
    vec4 cGT1 = tfGT[jGT1];
    vec4 colorGT = mix(cGT0, cGT1, fGT);

    float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
    float tOpt0 = clamp(floor(tOpt * Nj), 0.0f, Nj);
    float tOpt1 = clamp(ceil(tOpt * Nj), 0.0f, Nj);
    float fOpt = tOpt * Nj - tOpt0;
    uint jOpt0 = uint(tOpt0);
    uint jOpt1 = uint(tOpt1);
    vec4 cOpt0 = tfOpt[jOpt0];
    vec4 cOpt1 = tfOpt[jOpt1];
    vec4 colorOpt = mix(cOpt0, cOpt1, fOpt);

    vec4 colorDiff = colorOpt - colorGT;

    const float invN = 1.0 / float(xs * ys * zs);
#if defined(L1_LOSS)
    vec4 dColorOpt = invN * (2.0 * vec4(greaterThanEqual(colorDiff, vec4(0.0))) - vec4(1.0));
#elif defined(L2_LOSS)
    vec4 dColorOpt = (invN * 2.0) * colorDiff;
#endif

    //if (currentPointIdx.x == 1 && currentPointIdx.y == 1 && currentPointIdx.z == 1) {
    //    debugPrintfEXT("%f, %f, %u, %u", colorDiff.y, tOpt, jOpt0, jOpt1);
    //}

    if (jOpt0 == jOpt1) {
        //fOpt = 1.0;
    }
    for (int c = 0; c < 4; c++) {
        uint i = jOpt0 * 4 + c;
        atomicAddGrad(i, (1.0 - fOpt) * dColorOpt[c]);
        if (jOpt0 != jOpt1) {
            uint j = jOpt1 * 4 + c;
            float fOpt1 = fOpt;
            atomicAddGrad(j, fOpt1 * dColorOpt[c]);
        }
    }
}


-- Compute.Image

#version 450 core

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform UniformBuffer {
    uint imageWidth, imageHeight, batchSize;
};

layout(binding = 1, std430) readonly buffer FinalColorsGTBuffer {
    vec4 finalColorsGT[];
};

layout(binding = 2, std430) readonly buffer FinalColorsOptBuffer {
    vec4 finalColorsOpt[];
};

layout(binding = 3, std430) writeonly buffer AdjointColorsBuffer {
    vec4 adjointColors[];
};

void main() {
    const uint workSizeLinear = imageWidth * imageHeight * batchSize;
    const float invN = 1.0 / float(workSizeLinear);
    const uint workStep = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    for (uint workIdx = gl_GlobalInvocationID.x; workIdx < workSizeLinear; workIdx += workStep) {
        vec4 colorDiff = finalColorsOpt[workIdx] - finalColorsGT[workIdx];
#if defined(L1_LOSS)
        adjointColors[workIdx] = invN * (2.0 * vec4(greaterThanEqual(colorDiff, vec4(0.0))) - vec4(1.0));
#elif defined(L2_LOSS)
        adjointColors[workIdx] = (invN * 2.0) * colorDiff;
#endif
    }
}
