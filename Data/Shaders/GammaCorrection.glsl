/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020 - 2021, Christoph Neuhauser
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
layout(location = 1) in vec2 vertexTexCoord;
layout(location = 0) out vec2 fragTexCoord;

void main() {
    fragTexCoord = vertexTexCoord;
    gl_Position = vec4(vertexPosition, 1.0);
}

-- Fragment

#version 450 core

layout(binding = 0) uniform sampler2D inputTexture;
layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 fragColor;

/**
 * Converts linear RGB to sRGB.
 * For more details see: https://en.wikipedia.org/wiki/SRGB
 */
vec3 toSRGB(vec3 u) {
    return mix(1.055 * pow(u, vec3(1.0 / 2.4)) - 0.055, u * 12.92, lessThanEqual(u, vec3(0.0031308)));
}

void main() {
    vec4 linearColor = texture(inputTexture, fragTexCoord);
    fragColor = vec4(toSRGB(linearColor.rgb), linearColor.a);
}

-- FragmentDownscale

#version 450 core

layout(binding = 0) uniform sampler2D inputTexture;
layout(push_constant) uniform PushConstants {
    int supersamplingFactor;
};
layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 fragColor;

/**
 * Converts linear RGB to sRGB.
 * For more details see: https://en.wikipedia.org/wiki/SRGB
 */
vec3 toSRGB(vec3 u) {
    return mix(1.055 * pow(u, vec3(1.0 / 2.4)) - 0.055, u * 12.92, lessThanEqual(u, vec3(0.0031308)));
}

void main() {
    ivec2 inputSize = textureSize(inputTexture, 0);
    ivec2 outputSize = inputSize / supersamplingFactor;
    ivec2 outputLocation = ivec2(int(fragTexCoord.x * outputSize.x), int(fragTexCoord.y * outputSize.y));
    vec4 color = vec4(0.0);
    for (int sampleIdxY = 0; sampleIdxY < supersamplingFactor; sampleIdxY++) {
        for (int sampleIdxX = 0; sampleIdxX < supersamplingFactor; sampleIdxX++) {
            ivec2 inputLocation = outputLocation * supersamplingFactor + ivec2(sampleIdxX, sampleIdxY);
            vec4 sampleColor = texelFetch(inputTexture, inputLocation, 0);
            color += sampleColor;
        }
    }

    int totalNumSamples = supersamplingFactor * supersamplingFactor;
    color /= float(totalNumSamples);
    fragColor = toSRGB(toSRGB(linearColor.rgb), color.a);
}
