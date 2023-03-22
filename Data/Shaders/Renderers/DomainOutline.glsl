/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2018 - 2021, Christoph Neuhauser
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

#ifdef USE_DEPTH_CUES
layout(location = 0) out vec3 screenSpacePosition;
#endif

void main() {
#ifdef USE_DEPTH_CUES
    screenSpacePosition = (vMatrix * vec4(vertexPosition, 1.0)).xyz;
#endif
    gl_Position = mvpMatrix * vec4(vertexPosition, 1.0);
}


-- Fragment

#version 450 core

layout(binding = 0) uniform UniformDataBuffer {
    vec4 objectColor;
    float minDepth;
    float maxDepth;
};

#ifdef USE_DEPTH_CUES
layout(location = 0) in vec3 screenSpacePosition;
#endif

layout(location = 0) out vec4 fragColor;

void main() {
    vec4 color = objectColor;

#ifdef USE_DEPTH_CUES
    const float depthCueStrength = 0.8;
    float depthCueFactor = clamp((-screenSpacePosition.z - minDepth) / (maxDepth - minDepth), 0.0, 1.0);
    depthCueFactor = depthCueFactor * depthCueFactor * depthCueStrength;
    color.rgb = mix(color.rgb, vec3(0.5, 0.5, 0.5), depthCueFactor);
#endif

    fragColor = color;
}


-- Compute

#version 450 core

#extension GL_EXT_scalar_block_layout : require

layout(local_size_x = 16) in;

layout(push_constant) uniform PushConstants {
    vec3 aabbMin;
    float lineWidth;
    vec3 aabbMax;
    float offset; //< To avoid z-fighting of multiple outlines.
};

layout(binding = 0, std430) writeonly buffer IndexBuffer {
    uint indexBuffer[];
};

layout(binding = 1, scalar) writeonly buffer VertexBuffer {
    vec3 vertexBuffer[];
};

const uint indexData[36] = {
    0, 1, 2, 1, 3, 2, // front
    4, 6, 5, 5, 6, 7, // back
    0, 2, 4, 4, 2, 6, // left
    1, 5, 3, 5, 7, 3, // right
    0, 4, 1, 1, 4, 5, // bottom
    2, 3, 6, 3, 7, 6, // top
};
const vec3 vertexData[8] = {
    vec3(0.0f, 0.0f, 0.0f),
    vec3(1.0f, 0.0f, 0.0f),
    vec3(0.0f, 1.0f, 0.0f),
    vec3(1.0f, 1.0f, 0.0f),
    vec3(0.0f, 0.0f, 1.0f),
    vec3(1.0f, 0.0f, 1.0f),
    vec3(0.0f, 1.0f, 1.0f),
    vec3(1.0f, 1.0f, 1.0f),
};

void addEdge(vec3 lower, vec3 upper, uint vertexOffset, uint indexOffset) {
    for (uint idx = 0; idx < 36; idx++) {
        indexBuffer[indexOffset + idx] = indexData[idx] + vertexOffset;
    }
    for (uint vidx = 0; vidx < 8; vidx++) {
        vec3 pos = vertexData[vidx];
        pos = pos * (upper - lower) + lower;
        vertexBuffer[vertexOffset + vidx] = pos;
    }
}

void main() {
    uint threadIdx = gl_GlobalInvocationID.x;
    vec3 min0 = aabbMin - vec3(lineWidth / 2.0f + offset);
    vec3 min1 = aabbMin + vec3(lineWidth / 2.0f + offset);
    vec3 max0 = aabbMax - vec3(lineWidth / 2.0f + offset);
    vec3 max1 = aabbMax + vec3(lineWidth / 2.0f + offset);
    vec3 lower, upper;
    if (threadIdx == 0) {
        lower = vec3(min0.x, min0.y, min0.z); upper = vec3(max1.x, min1.y, min1.z);
    } else if (threadIdx == 1) {
        lower = vec3(min0.x, min0.y, max0.z); upper = vec3(max1.x, min1.y, max1.z);
    } else if (threadIdx == 2) {
        lower = vec3(min0.x, min0.y, min1.z); upper = vec3(min1.x, min1.y, max0.z);
    } else if (threadIdx == 3) {
        lower = vec3(max0.x, min0.y, min1.z); upper = vec3(max1.x, min1.y, max0.z);
    } else if (threadIdx == 4) {
        lower = vec3(min0.x, max0.y, min0.z); upper = vec3(max1.x, max1.y, min1.z);
    } else if (threadIdx == 5) {
        lower = vec3(min0.x, max0.y, max0.z); upper = vec3(max1.x, max1.y, max1.z);
    } else if (threadIdx == 6) {
        lower = vec3(min0.x, max0.y, min1.z); upper = vec3(min1.x, max1.y, max0.z);
    } else if (threadIdx == 7) {
        lower = vec3(max0.x, max0.y, min1.z); upper = vec3(max1.x, max1.y, max0.z);
    } else if (threadIdx == 8) {
        lower = vec3(min0.x, min1.y, min0.z); upper = vec3(min1.x, max0.y, min1.z);
    } else if (threadIdx == 9) {
        lower = vec3(max0.x, min1.y, min0.z); upper = vec3(max1.x, max0.y, min1.z);
    } else if (threadIdx == 10) {
        lower = vec3(min0.x, min1.y, max0.z); upper = vec3(min1.x, max0.y, max1.z);
    } else if (threadIdx == 11) {
        lower = vec3(max0.x, min1.y, max0.z); upper = vec3(max1.x, max0.y, max1.z);
    }
    addEdge(lower, upper, threadIdx * 8, threadIdx * 36);
}
