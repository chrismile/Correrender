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

#ifndef CORRERENDER_RENDERINGMODES_HPP
#define CORRERENDER_RENDERINGMODES_HPP

#include <cstdint>

enum RenderingMode : int32_t {
    RENDERING_MODE_NONE = -1,
    RENDERING_MODE_DIRECT_VOLUME_RENDERING = 0,
    RENDERING_MODE_ISOSURFACE_RAYCASTER = 1,
    RENDERING_MODE_ISOSURFACE_RASTERIZER = 2,
    RENDERING_MODE_DOMAIN_OUTLINE_RENDERER = 3,
    RENDERING_MODE_SLICE_RENDERER = 4,
    RENDERING_MODE_WORLD_MAP_RENDERER = 5,
    RENDERING_MODE_DIAGRAM_RENDERER = 6,
    RENDERING_MODE_SCATTER_PLOT = 7,
    RENDERING_MODE_CUSTOM = 8 // e.g., reference point selection renderer; cannot be chosen in UI.
};
const char* const RENDERING_MODE_NAMES[] = {
        "Direct Volume Renderer",
        "Iso Surface Raycaster",
        "Iso Surface Rasterizer",
        "Domain Outline Renderer",
        "Slice Renderer",
        "World Map Renderer",
        "Diagram Renderer",
        "Scatter Plot"
};
const char* const RENDERING_MODE_NAMES_ID[] = {
        "dvr",
        "iso_ray",
        "iso_raster",
        "domain_outline",
        "slice",
        "world_map",
        "diagram",
        "scatter_plot"
};
const int NUM_RENDERING_MODES = ((int)(sizeof(RENDERING_MODE_NAMES) / sizeof(*RENDERING_MODE_NAMES)));

const uint32_t ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT = 4052753091u;

#endif //CORRERENDER_RENDERINGMODES_HPP
