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

-- Vertex

#version 450 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 2) in vec2 vertexTexCoord;
layout(location = 0) out vec3 fragmentPositionWorld;
layout(location = 1) out vec3 fragmentNormal;
layout(location = 2) out vec2 fragmentTexCoord;

void main() {
    fragmentPositionWorld = vertexPosition;
    fragmentNormal = vertexNormal;
    fragmentTexCoord = vertexTexCoord;
    gl_Position = mvpMatrix * vec4(vertexPosition, 1.0);
}


-- Fragment

#version 450 core

layout(binding = 0) uniform RendererUniformDataBuffer {
    vec3 cameraPosition;
    uint padding0;
    vec3 minBoundingBox;
    float lightingFactor;
    vec3 maxBoundingBox;
    float padding1;
};

layout (binding = 1) uniform sampler2D worldMapTexture;

layout(location = 0) in vec3 fragmentPositionWorld;
layout(location = 1) in vec3 fragmentNormal;
layout(location = 2) in vec2 fragmentTexCoord;
layout(location = 0) out vec4 fragColor;

#include "UniformData.glsl"
#include "Lighting.glsl"

void main() {
    //vec2 texCoords = (fragmentPositionWorld.xy - minBoundingBox.xy) / (maxBoundingBox.xy - minBoundingBox.xy);
    vec4 worldMapColor = texture(worldMapTexture, fragmentTexCoord);
    worldMapColor.a = 1.0;
    vec3 n = normalize(fragmentNormal);
    vec4 color = blinnPhongShadingSurface(worldMapColor, fragmentPositionWorld, n);
    color = mix(worldMapColor, color, lightingFactor);
    fragColor = color;
}
