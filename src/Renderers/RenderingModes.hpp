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
};
const char* const RENDERING_MODE_NAMES[] = {
        "Direct Volume Renderer",
        "Iso Surface Raycaster",
        "Iso Surface Rasterizer"
};
const int NUM_RENDERING_MODES = ((int)(sizeof(RENDERING_MODE_NAMES) / sizeof(*RENDERING_MODE_NAMES)));

const uint32_t ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT = 4052753091u;

#endif //CORRERENDER_RENDERINGMODES_HPP
