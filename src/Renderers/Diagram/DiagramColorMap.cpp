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

#include "DiagramColorMap.hpp"

std::vector<sgl::Color> defaultColors = {
        sgl::Color(100, 255, 100),
        sgl::Color(255, 60, 50),
        sgl::Color(0, 170, 255),
        sgl::Color(255, 148, 60),
};

std::vector<glm::vec3> getColorPoints(DiagramColorMap colorMap) {
    std::vector<glm::vec3> colorPoints;
    if (colorMap == DiagramColorMap::VIRIDIS) {
        colorPoints = {
                { 52.0f / 255.0f, 0.0f / 255.0f, 66.0f / 255.0f },
                { 45.0f / 255.0f, 62.0f / 255.0f, 120.0f / 255.0f },
                { 31.0f / 255.0f, 129.0f / 255.0f, 121.0f / 255.0f },
                { 81.0f / 255.0f, 195.0f / 255.0f, 78.0f / 255.0f },
                { 252.0f / 255.0f, 229.0f / 255.0f, 120.0f / 255.0f },
        };
    } else if (colorMap == DiagramColorMap::HEATMAP) {
        colorPoints = {
                { 0.0f / 255.0f, 0.0f / 255.0f, 0.0f / 255.0f },
                { 57.0f / 255.0f, 15.0f / 255.0f, 110.0f / 255.0f },
                { 139.0f / 255.0f, 41.0f / 255.0f, 129.0f / 255.0f },
                { 220.0f / 255.0f, 72.0f / 255.0f, 105.0f / 255.0f },
                { 254.0f / 255.0f, 159.0f / 255.0f, 109.0f / 255.0f },
                { 252.0f / 255.0f, 253.0f / 255.0f, 191.0f / 255.0f },
        };
    } else if (colorMap == DiagramColorMap::CIVIDIS) {
        // Cividis (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.0f / 255.0f, 34.0f / 255.0f, 77.0f / 255.0f },
                { 124.0f / 255.0f, 123.0f / 255.0f, 120.0f / 255.0f },
                { 253.0f / 255.0f, 231.0f / 255.0f, 55.0f / 255.0f },
        };
    } else if (colorMap == DiagramColorMap::SPRING) {
        // Spring (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 1.0000f, 0.0000f, 1.0000f },
                { 1.0000f, 0.2510f, 0.7490f },
                { 1.0000f, 0.5020f, 0.4980f },
                { 1.0000f, 0.7529f, 0.2471f },
                { 1.0000f, 1.0000f, 0.0000f },
        };
    } else if (colorMap == DiagramColorMap::SUMMER) {
        // Summer (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.0000f, 0.5000f, 0.4000f },
                { 0.2510f, 0.6255f, 0.4000f },
                { 0.5020f, 0.7510f, 0.4000f },
                { 0.7529f, 0.8765f, 0.4000f },
                { 1.0000f, 1.0000f, 0.4000f },
        };
    } else if (colorMap == DiagramColorMap::AUTUMN) {
        // Autumn (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 1.0000f, 0.0000f, 0.0000f },
                { 1.0000f, 0.2510f, 0.0000f },
                { 1.0000f, 0.5020f, 0.0000f },
                { 1.0000f, 0.7529f, 0.0000f },
                { 1.0000f, 1.0000f, 0.0000f },
        };
    } else if (colorMap == DiagramColorMap::WINTER) {
        // Winter (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.0000f, 0.0000f, 1.0000f },
                { 0.0000f, 0.2510f, 0.8745f },
                { 0.0000f, 0.5020f, 0.7490f },
                { 0.0000f, 0.7529f, 0.6235f },
                { 0.0000f, 1.0000f, 0.5000f },
        };
    } else if (colorMap == DiagramColorMap::COOL) {
        // Cool (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.0000f, 1.0000f, 1.0000f },
                { 0.2510f, 0.7490f, 1.0000f },
                { 0.5020f, 0.4980f, 1.0000f },
                { 0.7529f, 0.2471f, 1.0000f },
                { 1.0000f, 0.0000f, 1.0000f },
        };
    } else if (colorMap == DiagramColorMap::WISTIA) {
        // Wistia (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.8941f, 1.0000f, 0.4784f },
                { 1.0000f, 0.9091f, 0.1016f },
                { 1.0000f, 0.7403f, 0.0000f },
                { 0.9999f, 0.6259f, 0.0000f },
                { 0.9882f, 0.4980f, 0.0000f },
        };
    } else if (colorMap == DiagramColorMap::NEON_GREENS) {
        colorPoints = {
                { 208.0f / 255.0f, 231.0f / 255.0f, 208.0f / 255.0f },
                { 100.0f / 255.0f, 255.0f / 255.0f, 100.0f / 255.0f },
        };
    } else {
        sgl::Color color = defaultColors.at(int(colorMap) - int(DiagramColorMap::NEON_GREEN));
        colorPoints = {
                { color.getFloatR(), color.getFloatG(), color.getFloatB() },
                { color.getFloatR(), color.getFloatG(), color.getFloatB() },
        };
    }
    return colorPoints;
}
