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

layout(binding = 0) uniform UniformDataBuffer {
    vec4 c0;
    vec4 c1;
    vec3 p0;
    float lineWidth;
    vec3 p1;
    float padding;
};


-- Vertex

#version 450 core

#import ".Uniform"

layout(location = 0) out float pct;

#define M_PI 3.14159265358979323846

void main() {
    uint linePointIdx = gl_VertexIndex / NUM_TUBE_SUBDIVISIONS;
    uint circleIdx = gl_VertexIndex % NUM_TUBE_SUBDIVISIONS;
    vec3 linePoint = linePointIdx == 0 ? p0 : p1;
    vec3 lineCenterPosition = (mMatrix * vec4(linePoint, 1.0)).xyz;

    vec3 tangent = normalize(p1 - p0);
    vec3 helperAxis = vec3(1.0, 0.0, 0.0);
    if (length(cross(helperAxis, tangent)) < 0.01) {
        helperAxis = vec3(0.0, 1.0, 0.0);
    }
    vec3 normal = normalize(helperAxis - tangent * dot(helperAxis, tangent)); // Gram-Schmidt
    vec3 binormal = cross(tangent, normal);
    mat3 tangentFrameMatrix = mat3(normal, binormal, tangent);

    float t = float(circleIdx) / float(NUM_TUBE_SUBDIVISIONS) * 2.0 * M_PI;
    float cosAngle = cos(t);
    float sinAngle = sin(t);

    const float lineRadius = lineWidth * 0.5;
    vec3 localPosition = vec3(cosAngle, sinAngle, 0.0);
    vec3 vertexPosition = lineRadius * (tangentFrameMatrix * localPosition) + lineCenterPosition;

    pct = linePointIdx == 0 ? 0.0 : 1.0;
    gl_Position = mvpMatrix * vec4(vertexPosition, 1.0);
}


-- Fragment

#version 450 core

#import ".Uniform"

layout(location = 0) in float pct;
layout(location = 0) out vec4 fragColor;

void main() {
    fragColor = mix(c0, c1, pct);
}
