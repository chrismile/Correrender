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

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform RendererUniformDataBuffer {
    mat4 inverseViewMatrix;
    mat4 inverseProjectionMatrix;
    float zNear;
    float zFar;
    uint fieldIndex;
    float padding1;
    vec3 minBoundingBox;
    float attenuationCoefficient;
    vec3 maxBoundingBox;
    float stepSize;
};

layout(binding = 1, rgba32f) uniform image2D outputImage;
layout(binding = 2) uniform sampler3D scalarField;

#ifdef SUPPORT_DEPTH_BUFFER
layout(binding = 3, r32f) uniform image2D depthBuffer;
float closestDepth;
#include "DepthHelper.glsl"
#endif

#include "RayIntersectionTests.glsl"
#include "Blending.glsl"
#include "UniformData.glsl"
#include "TransferFunction.glsl"

void main() {
    ivec2 outputImageSize = imageSize(outputImage);
    ivec2 imageCoords = ivec2(gl_GlobalInvocationID.xy);
    if (imageCoords.x >= outputImageSize.x || imageCoords.y >= outputImageSize.y) {
        return;
    }

    vec3 rayOrigin = (inverseViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec2 fragNdc = 2.0 * ((vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) / vec2(outputImageSize)) - 1.0;
    vec3 rayTarget = (inverseProjectionMatrix * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 normalizedTarget = normalize(rayTarget.xyz);
    vec3 rayDirection = (inverseViewMatrix * vec4(normalizedTarget, 0.0)).xyz;
    float zFactor = abs(normalizedTarget.z);

    float tNear, tFar;
    if (rayBoxIntersectionRayCoords(rayOrigin, rayDirection, minBoundingBox, maxBoundingBox, tNear, tFar)) {
        vec3 entrancePoint = rayOrigin + rayDirection * tNear;
        vec3 exitPoint = rayOrigin + rayDirection * tFar;
        float volumeDepth = length(exitPoint - entrancePoint);

        vec3 currentPoint = entrancePoint;
        if (tNear < 0) {
            currentPoint = rayOrigin;
        }

#ifdef SUPPORT_DEPTH_BUFFER
        closestDepth = convertDepthBufferValueToLinearDepth(imageLoad(depthBuffer, imageCoords).x);
        // Convert depth to distance.
        closestDepth = closestDepth / zFactor;
#endif

        vec4 outputColor = vec4(0.0);
        while (length(currentPoint - entrancePoint) < volumeDepth) {
#ifdef SUPPORT_DEPTH_BUFFER
            if (length(currentPoint - rayOrigin) >= closestDepth) {
                break;
            }
#endif
            vec3 texCoords = (currentPoint - minBoundingBox) / (maxBoundingBox - minBoundingBox);

            float scalarValue = texture(scalarField, texCoords).r;
            vec4 volumeColor = transferFunction(scalarValue, fieldIndex);
            float alpha = 1 - exp(-volumeColor.a * stepSize * attenuationCoefficient);
            vec4 color = vec4(volumeColor.rgb, alpha);

            if (blend(color, outputColor)) {
                break;
            }
            currentPoint += rayDirection * stepSize;
        }

        vec4 backgroundColor = imageLoad(outputImage, imageCoords);
        blend(backgroundColor, outputColor);

        outputColor = vec4(outputColor.rgb / outputColor.a, outputColor.a);
        imageStore(outputImage, imageCoords, outputColor);
#ifdef SUPPORT_DEPTH_BUFFER
         // Convert depth to distance.
        //closestDepth = closestDepth * zFactor;
        //imageStore(depthBuffer, imageCoords, vec4(convertLinearDepthToDepthBufferValue(closestDepth)));
#endif
    }
}
