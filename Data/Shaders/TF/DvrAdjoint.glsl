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

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform DvrSettingsBuffer {
    mat4 inverseProjectionMatrix;
    vec3 minBoundingBox;
    float attenuationCoefficient;
    vec3 maxBoundingBox;
    float stepSize;
};

layout(push_constant) uniform PushConstants {
    vec2 minMaxFieldValues;
};

struct BatchSettings {
    mat4 inverseViewMatrix;
};

layout(binding = 1) readonly buffer BatchSettingsBuffer {
    BatchSettings batchSettingsArray[];
};

layout(binding = 2) uniform sampler3D scalarField;

layout(binding = 3, std430) readonly buffer TfGTBuffer {
    float tfGT[];
};

layout(binding = 4, std430) readonly buffer TfOptBuffer {
    float tfOpt[];
};

layout(binding = 5, std430) readonly buffer FinalColorsBuffer {
    vec4 finalColors[];
};

layout(binding = 6, std430) readonly buffer TerminationIndexBuffer {
    uint terminationIndices[];
};

layout(binding = 7, std430) readonly buffer TfOptGradientBuffer {
#ifdef SUPPORT_BUFFER_FLOAT_ATOMIC_ADD
    float g[];
#else
    uint g[];
#endif
};

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

void renderAdjoint(uint workIdx, uint x, uint y, uint b, uint threadSharedMemoryOffset) {
    mat4 inverseViewMatrix = batchSettingsArray[b];
    vec3 rayOrigin = (inverseViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec2 fragNdc = 2.0 * ((vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) / vec2(outputImageSize)) - 1.0;
    vec3 rayTarget = (inverseProjectionMatrix * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 normalizedTarget = normalize(rayTarget.xyz);
    vec3 rayDirection = (inverseViewMatrix * vec4(normalizedTarget, 0.0)).xyz;

    float tNear, tFar;
    if (rayBoxIntersectionRayCoords(rayOrigin, rayDirection, minBoundingBox, maxBoundingBox, tNear, tFar)) {
        vec3 entrancePoint = rayOrigin + rayDirection * tNear;
        vec3 exitPoint = rayOrigin + rayDirection * tFar;

        int terminationIndexMax = terminationIndices[workIdx] - 1;
        for (int terminationIndex = terminationIndexMax; terminationIndex >= 0; terminationIndex--) {
            ;
        }
    }
}

shared float sharedMemory[NUM_TF_ENTRIES * BLOCK_SIZE];

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
        int x = workIdx % imageWidth;
        int y = (workIdx / imageWidth) % imageHeight;
        int b = workIdx / (imageWidth * imageHeight);
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
