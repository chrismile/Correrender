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

layout(binding = 3, std430) readonly buffer TfBuffer {
    vec4 tf[];
};

layout(binding = 4, std430) writeonly buffer FinalColorsBuffer {
    vec4 finalColors[];
};

#ifdef FORWARD_PASS_OPT
layout(binding = 5, std430) writeonly buffer TerminationIndexBuffer {
    int terminationIndices[];
};
#endif

#include "RayIntersectionTests.glsl"
#include "Blending.glsl"

void renderForward(uint workIdx, uint x, uint y, uint b) {
    mat4 inverseViewMatrix = batchSettingsArray[b].inverseViewMatrix;
    vec3 rayOrigin = inverseViewMatrix[3].xyz;
    vec2 fragNdc = 2.0 * ((vec2(x, y) + vec2(0.5)) / vec2(imageWidth, imageHeight)) - 1.0;
    vec3 rayTarget = (inverseProjectionMatrix * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 normalizedTarget = normalize(rayTarget.xyz);
    vec3 rayDirection = (inverseViewMatrix * vec4(normalizedTarget, 0.0)).xyz;

    vec4 outputColor = vec4(0.0);
    int terminationIndex = 0;

    float tNear, tFar;
    if (rayBoxIntersectionRayCoords(rayOrigin, rayDirection, minBoundingBox, maxBoundingBox, tNear, tFar)) {
        vec3 entrancePoint = rayOrigin + rayDirection * tNear;
        if (tNear < 0) {
            entrancePoint = rayOrigin;
        }
        vec3 exitPoint = rayOrigin + rayDirection * tFar;

        float volumeDepth = length(exitPoint - entrancePoint);
        vec3 currentPoint = entrancePoint;

        while (length(currentPoint - entrancePoint) < volumeDepth) {
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
            uint j0 = uint(t0);
            uint j1 = uint(t1);
            vec4 c0 = tf[j0];
            vec4 c1 = tf[j1];
            vec4 volumeColor = mix(c0, c1, f);
            if (isValueNan) {
                volumeColor = vec4(0.0);
            }

            float alpha = 1.0 - exp(-volumeColor.a * stepSize * attenuationCoefficient);
            vec4 color = vec4(volumeColor.rgb, alpha);

            terminationIndex++;
            if (blend(color, outputColor)) {
                break;
            }
            currentPoint = entrancePoint + rayDirection * (float(terminationIndex) * stepSize);
        }
    }
    //if (x == 256 && y == 256) {
        //debugPrintfEXT("a: %f", attenuationCoefficient);
        //debugPrintfEXT("t256 %f, %f, o %f, %f, %f, d %f, %f, %f", tNear, tFar, rayOrigin.x, rayOrigin.y, rayOrigin.z, rayDirection.x, rayDirection.y, rayDirection.z);
        //debugPrintfEXT("%f %f %f", rayTarget[0], rayTarget[1], rayTarget[2]);
        //for (int i = 0; i < 4; i++) {
        //    debugPrintfEXT("%f %f %f %f", inverseViewMatrix[i][0], inverseViewMatrix[i][1], inverseViewMatrix[i][2], inverseViewMatrix[i][3]);
        //}
    //}

    finalColors[workIdx] = outputColor;
#ifdef FORWARD_PASS_OPT
    terminationIndices[workIdx] = terminationIndex;
#endif
}

void main() {
    const uint workSizeLinear = imageWidth * imageHeight * batchSize;
    const uint workStep = gl_WorkGroupSize.x * gl_NumWorkGroups.x;
    for (uint workIdx = gl_GlobalInvocationID.x; workIdx < workSizeLinear; workIdx += workStep) {
        uint x = workIdx % imageWidth;
        uint y = (workIdx / imageWidth) % imageHeight;
        uint b = workIdx / (imageWidth * imageHeight);
        renderForward(workIdx, x, y, b);
    }
}
