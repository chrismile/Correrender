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

#ifndef CORRERENDER_DIAGRAMCOLORMAP_HPP
#define CORRERENDER_DIAGRAMCOLORMAP_HPP

#include <vector>
#include <glm/vec3.hpp>

#include <Graphics/Color.hpp>

enum class DiagramColorMap {
    VIRIDIS, HEATMAP, CIVIDIS, GRAY,
    SPRING, SUMMER, AUTUMN, WINTER, COOL, WISTIA,
    COOL_TO_WARM, //< Diverging.
    PIYG, PRGN, BRBG, PUOR, RDGY, RDBU, RDYLBU, RDYLGN, SPECTRAL, COOLWARM, BWR, SEISMIC, //< Diverging from Matplotlib.
    NEON_GREENS, NEON_GREEN, NEON_RED, NEON_BLUE, NEON_ORANGE,
    BLACK_YELLOW, BLACK_ORANGE, BLACK_GREEN, BLACK_BLUE, BLACK_CYAN, BLACK_PURPLE,
    BLACK_NEON_GREEN, BLACK_NEON_RED, BLACK_NEON_BLUE, BLACK_NEON_ORANGE
};
const char* const DIAGRAM_COLOR_MAP_NAMES[] = {
        "Viridis", "Heatmap", "Cividis", "Gray",
        "Spring", "Summer", "Autumn", "Winter", "Cool", "Wistia",
        "Cool to Warm",
        "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdYlBu", "RdYlGn", "Spectral", "Coolwarm", "bwr", "Seismic",
        "Neon Greens", "Neon Green", "Neon Red", "Neon Blue", "Neon Orange",
        "Black-Yellow", "Black-Orange", "Black-Green", "Black-Blue", "Black-Cyan", "Black-Purple",
        "Black-Neon Green", "Black-Neon Red", "Black-Neon Blue", "Black-Neon Orange",
};
const int NUM_COLOR_MAPS = ((int)(sizeof(DIAGRAM_COLOR_MAP_NAMES) / sizeof(*(DIAGRAM_COLOR_MAP_NAMES))));
std::vector<glm::vec3> getColorPoints(DiagramColorMap colorMap);
inline bool getIsGrayscaleColorMap(DiagramColorMap colorMap) { return colorMap == DiagramColorMap::GRAY; }

extern std::vector<sgl::Color> defaultColors;

#endif //CORRERENDER_DIAGRAMCOLORMAP_HPP
