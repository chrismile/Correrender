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

#extension GL_EXT_shader_atomic_float : require
#extension GL_KHR_shader_subgroup_arithmetic : enable

//#extension GL_EXT_debug_printf : enable

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform DvrSettingsBuffer {
    mat4 inverseProjectionMatrix;
    vec3 minBoundingBox;
    float attenuationCoefficient;
    vec3 maxBoundingBox;
    float stepSize;
    uint imageWidth, imageHeight, batchSize;
};

layout(push_constant) uniform PushConstants {
    float minFieldValue, maxFieldValue, Nj;
};

struct BatchSettings {
    mat4 inverseViewMatrix;
};

layout(binding = 1) readonly buffer BatchSettingsBuffer {
    BatchSettings batchSettingsArray[];
};

layout(binding = 2) uniform sampler3D scalarField;

layout(binding = 3, std430) readonly buffer TfOptBuffer {
    float tfOpt[];
};

layout(binding = 4, std430) readonly buffer FinalColorsOptBuffer {
    vec4 finalColorsOpt[];
};

layout(binding = 5, std430) readonly buffer TerminationIndexBuffer {
    int terminationIndices[];
};

layout(binding = 6, std430) readonly buffer AdjointColorsBuffer {
    vec4 adjointColors[];
};

layout(binding = 7, std430) buffer TfOptGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float g[];
#else
    uint g[];
#endif
};

shared float sharedMemory[NUM_TF_ENTRIES * BLOCK_SIZE];

void atomicAddGradient(uint tfEntryIdx, float value) {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    atomicAdd(g[tfEntryIdx], value);
#else
    uint oldValue = g[tfEntryIdx];
    uint expectedValue, newValue;
    do {
        expectedValue = oldValue;
        newValue = floatBitsToUint(uintBitsToFloat(oldValue) + value);
        oldValue = atomicCompSwap(g[tfEntryIdx], expectedValue, newValue);
    } while (oldValue != expectedValue);
#endif
}

#include "RayIntersectionTests.glsl"

void renderAdjoint(uint workIdx, uint x, uint y, uint b, uint threadSharedMemoryOffset) {
    mat4 inverseViewMatrix = batchSettingsArray[b].inverseViewMatrix;
    vec3 rayOrigin = inverseViewMatrix[3].xyz;
    vec2 fragNdc = 2.0 * ((vec2(x, y) + vec2(0.5)) / vec2(imageWidth, imageHeight)) - 1.0;
    vec3 rayTarget = (inverseProjectionMatrix * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 normalizedTarget = normalize(rayTarget.xyz);
    vec3 rayDirection = (inverseViewMatrix * vec4(normalizedTarget, 0.0)).xyz;

    float tNear, tFar;
    if (rayBoxIntersectionRayCoords(rayOrigin, rayDirection, minBoundingBox, maxBoundingBox, tNear, tFar)) {
        vec3 entrancePoint = rayOrigin + rayDirection * tNear;
        vec3 exitPoint = rayOrigin + rayDirection * tFar;
        vec3 currentPoint;

        vec4 colorCurr = finalColorsOpt[workIdx];
        vec4 colorCurrAdjoint = adjointColors[workIdx];
        int terminationIndexMax = terminationIndices[workIdx] - 1;
        //if (x == 256 && y == 256) {
        //    debugPrintfEXT("a: %u", terminationIndexMax);
        //}
        for (int terminationIndex = terminationIndexMax; terminationIndex >= 0; terminationIndex--) {
            currentPoint = entrancePoint + rayDirection * (float(terminationIndex) * stepSize);
            vec3 texCoords = (currentPoint - minBoundingBox) / (maxBoundingBox - minBoundingBox);

            float scalarValue = texture(scalarField, texCoords).r;
            bool isValueNan = isnan(scalarValue);
            scalarValue = (scalarValue - minFieldValue) / (maxFieldValue - minFieldValue);
            if (isValueNan) {
                scalarValue = 0.0;
            }

            // Query the transfer function.
            float t0 = clamp(floor(scalarValue * Nj), 0.0f, Nj);
            float t1 = clamp(ceil(scalarValue * Nj), 0.0f, Nj);
            float f = scalarValue * Nj - t0;
            uint j0 = uint(t0) * 4u;
            uint j1 = uint(t1) * 4u;
            vec4 c0 = vec4(tfOpt[j0], tfOpt[j0 + 1], tfOpt[j0 + 2], tfOpt[j0 + 3]);
            vec4 c1 = vec4(tfOpt[j0], tfOpt[j0 + 1], tfOpt[j0 + 2], tfOpt[j0 + 3]);
            vec4 volumeColor = mix(c0, c1, f);
            if (isValueNan) {
                volumeColor = vec4(0.0);
            }

            // Use Beer-Lambert law for computing the blending alpha.
            float attStepSize = stepSize * attenuationCoefficient;
            float alphaVolume = 1.0 - exp(-volumeColor.a * attStepSize);
            //alphaVolume = clamp(alphaVolume, 1e-6, 0.9999);
            vec4 colorVolume = vec4(volumeColor.rgb * alphaVolume, alphaVolume);

            // Inversion trick from "Differentiable Direct Volume Rendering", Weiß et al. 2021.
            float alphaIn = (alphaVolume - colorCurr.a) / (alphaVolume - 1.0);
            vec3 colorIn = colorCurr.rgb - (1.0 - alphaIn) * colorVolume.rgb;

            // Compute the for the volume color/alpha.
            vec4 colorVolumeAdjoint;
            colorVolumeAdjoint.rgb = alphaVolume * (1.0 - alphaIn) * colorCurrAdjoint.rgb;
            float alphaAdjoint = colorCurrAdjoint.a * (1.0 - alphaIn) + dot(colorCurrAdjoint.rgb, volumeColor.rgb - volumeColor.rgb * alphaIn);
            colorVolumeAdjoint.a = alphaAdjoint * attStepSize * exp(-volumeColor.a * attStepSize);

            // Backpropagation for the accumulated color.
            // colorCurrAdjoint.rgb stays the same (see paper cited above, Chat^(i) = Chat^(i+1)).
            float alphaNewAdjoint = colorCurrAdjoint.a * (1.0 - alphaVolume) - dot(colorCurrAdjoint.rgb, colorVolume.rgb);
            colorCurrAdjoint.a = alphaNewAdjoint;
            colorCurr = vec4(colorIn, alphaIn);

            // Compute adjoint for the transfer function entries.
            //float fAdjoint = dot(colorVolumeAdjoint, c1 - c0);
            vec4 c0adj = (1.0 - f) * colorVolumeAdjoint;
            vec4 c1adj = f * colorVolumeAdjoint;
            if (isValueNan) {
                c0adj = vec4(0.0);
                c1adj = vec4(0.0);
            }

#ifdef ADJOINT_DELAYED
            sharedMemory[threadSharedMemoryOffset + j0] += c0adj.r;
            sharedMemory[threadSharedMemoryOffset + j0 + 1] += c0adj.g;
            sharedMemory[threadSharedMemoryOffset + j0 + 2] += c0adj.b;
            sharedMemory[threadSharedMemoryOffset + j0 + 3] += c0adj.a;
            if (j0 != j1) {
                sharedMemory[threadSharedMemoryOffset + j1] += c1adj.r;
                sharedMemory[threadSharedMemoryOffset + j1 + 1] += c1adj.g;
                sharedMemory[threadSharedMemoryOffset + j1 + 2] += c1adj.b;
                sharedMemory[threadSharedMemoryOffset + j1 + 3] += c1adj.a;
            }
#else
            atomicAddGradient(j0, c0adj.r);
            atomicAddGradient(j0 + 1, c0adj.g);
            atomicAddGradient(j0 + 2, c0adj.b);
            atomicAddGradient(j0 + 3, c0adj.a);
            if (j0 != j1) {
                atomicAddGradient(j1, c1adj.r);
                atomicAddGradient(j1 + 1, c1adj.g);
                atomicAddGradient(j1 + 2, c1adj.b);
                atomicAddGradient(j1 + 3, c1adj.a);
            }
#endif
        }
    }
}

void main() {
    uint threadSharedMemoryOffset = gl_LocalInvocationID.x * NUM_TF_ENTRIES;

#ifdef ADJOINT_DELAYED
    for (uint tfEntryIdx = 0; tfEntryIdx < NUM_TF_ENTRIES; tfEntryIdx++) {
        sharedMemory[threadSharedMemoryOffset + tfEntryIdx] = 0.0;
    }
#endif

    const uint workSizeLinear = imageWidth * imageHeight * batchSize;
    const uint workStep = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    for (uint workIdx = gl_GlobalInvocationID.x; workIdx < workSizeLinear; workIdx += workStep) {
        uint x = workIdx % imageWidth;
        uint y = (workIdx / imageWidth) % imageHeight;
        uint b = workIdx / (imageWidth * imageHeight);
        renderAdjoint(workIdx, x, y, b, threadSharedMemoryOffset);
    }

#ifdef ADJOINT_DELAYED
    for (uint tfEntryIdx = 0; tfEntryIdx < NUM_TF_ENTRIES; tfEntryIdx++) {
        float value = subgroupAdd(sharedMemory[threadSharedMemoryOffset + tfEntryIdx]);
        if (subgroupElect()) {
            atomicAddGradient(tfEntryIdx, value);
        }
    }
#endif
}
