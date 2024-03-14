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

-- Uniform

layout(binding = 0) uniform RendererUniformDataBuffer {
    vec3 cameraPosition;
    uint fieldIndex;
    vec3 minBoundingBox;
    float lightingFactor;
    vec3 maxBoundingBox;
    uint fixOnGround;
};


-- Vertex

#version 450 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 0) out vec3 fragmentPositionWorld;
layout(location = 1) out vec3 fragmentPositionWorldReal;
layout(location = 2) out vec3 fragmentNormal;

#import ".Uniform"

void main() {
    fragmentPositionWorld = vertexPosition;
    fragmentNormal = vertexNormal;
    vec3 positionOut = vertexPosition;
    if (fixOnGround != 0u) {
        positionOut.z = minBoundingBox.z;
    }
    fragmentPositionWorldReal = positionOut;
    gl_Position = mvpMatrix * vec4(positionOut, 1.0);
}


-- Fragment

#version 450 core

layout(location = 0) in vec3 fragmentPositionWorld;
layout(location = 1) in vec3 fragmentPositionWorldReal;
layout(location = 2) in vec3 fragmentNormal;
layout(location = 0) out vec4 fragColor;

#import ".Uniform"
layout (binding = 1) uniform sampler3D scalarField;

#include "UniformData.glsl"
#include "TransferFunction.glsl"
#include "Lighting.glsl"

void main() {
    vec3 texCoords = (fragmentPositionWorld - minBoundingBox) / (maxBoundingBox - minBoundingBox);
    float scalarValue = texture(scalarField, texCoords).r;
    vec4 volumeColor = transferFunction(scalarValue, fieldIndex);
    volumeColor.a = 1.0;
    vec3 n = normalize(fragmentNormal);
    vec4 color = blinnPhongShadingSurface(volumeColor, fragmentPositionWorldReal, n);
    color = mix(volumeColor, color, lightingFactor);
    fragColor = color;
}
