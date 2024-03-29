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

-- Vertex

#version 450 core

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 vertexNormal;
layout(location = 0) out vec3 fragmentPositionWorld;
layout(location = 1) out vec3 fragmentNormal;
#ifdef MULTI_COLOR_ISOSURFACE
layout(location = 2) in vec4 vertexColor;
layout(location = 2) out vec4 isoSurfaceColor;
#endif

void main() {
    fragmentPositionWorld = vertexPosition;
    fragmentNormal = vertexNormal;
#ifdef MULTI_COLOR_ISOSURFACE
    isoSurfaceColor = vertexColor;
#endif
    gl_Position = mvpMatrix * vec4(vertexPosition, 1.0);
}


-- Fragment

#version 450 core

layout(binding = 0) uniform RendererUniformDataBuffer {
    vec3 cameraPosition;
    float padding;
#ifndef MULTI_COLOR_ISOSURFACE
    vec4 isoSurfaceColor;
#endif
};

layout(location = 0) in vec3 fragmentPositionWorld;
layout(location = 1) in vec3 fragmentNormal;
#ifdef MULTI_COLOR_ISOSURFACE
layout(location = 2) in vec4 isoSurfaceColor;
#endif

#include "Lighting.glsl"

#ifdef USE_RENDER_RESTRICTION
#include "UniformData.glsl"
#include "RenderRestriction.glsl"
#endif

#ifdef USE_OIT
#include "LinkedListGather.glsl"
#else
layout(location = 0) out vec4 fragColor;
#endif

void main() {
#ifdef USE_RENDER_RESTRICTION
    if (!getShouldRender(fragmentPositionWorld)) {
        discard;
    }
#endif

    //vec3 tangentX = dFdx(fragmentPositionWorld);
    //vec3 tangentY = dFdy(fragmentPositionWorld);
    //vec3 n = normalize(cross(tangentX, tangentY));
    vec3 n = normalize(fragmentNormal);
    vec4 color = blinnPhongShadingSurface(isoSurfaceColor, fragmentPositionWorld, n);

#ifdef USE_OIT
    gatherFragment(color);
#else
    fragColor = color;
#endif
}
