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

#include <ImGui/Widgets/TransferFunctionWindow.hpp>

#include "DiagramColorMap.hpp"

std::vector<sgl::Color> defaultColors = {
        sgl::Color(100, 255, 100), // NEON_GREEN
        sgl::Color(255, 60, 50),   // NEON_RED
        sgl::Color(0, 170, 255),   // NEON_BLUE
        sgl::Color(255, 148, 60),  // NEON_ORANGE
};

std::vector<sgl::Color> blackColorMapColors = {
        sgl::Color(255, 255, 100), // Yellow
        sgl::Color(252, 127,   0), // Orange
        sgl::Color(0,   255, 127), // Green
        sgl::Color(0,   127, 255), // Blue
        sgl::Color(0,   255, 255), // Cyan
        sgl::Color(255, 0,   255), // Purple
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
    } else if (colorMap == DiagramColorMap::GRAY) {
        // Gray (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.0000f, 0.0000f, 0.0000f },
                { 0.2510f, 0.2510f, 0.2510f },
                { 0.5020f, 0.5020f, 0.5020f },
                { 0.7529f, 0.7529f, 0.7529f },
                { 1.0000f, 1.0000f, 1.0000f },
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
    } else if (colorMap == DiagramColorMap::COOL_TO_WARM) {
        colorPoints = {
                {  59.0f / 255.0f,  76.0f / 255.0f, 192.0f / 255.0f },
                { 144.0f / 255.0f, 178.0f / 255.0f, 254.0f / 255.0f },
                { 220.0f / 255.0f, 220.0f / 255.0f, 220.0f / 255.0f },
                { 245.0f / 255.0f, 156.0f / 255.0f, 125.0f / 255.0f },
                { 180.0f / 255.0f,   4.0f / 255.0f,  38.0f / 255.0f },
        };
    } else if (colorMap == DiagramColorMap::PIYG) {
        // PiYG (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.5569f, 0.0039f, 0.3216f },
                { 0.9086f, 0.5926f, 0.7703f },
                { 0.9673f, 0.9685f, 0.9656f },
                { 0.6032f, 0.8055f, 0.3822f },
                { 0.1529f, 0.3922f, 0.0980f },
        };
    } else if (colorMap == DiagramColorMap::PRGN) {
        // PRGn (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.2510f, 0.0000f, 0.2941f },
                { 0.6820f, 0.5452f, 0.7426f },
                { 0.9663f, 0.9681f, 0.9659f },
                { 0.4932f, 0.7654f, 0.4967f },
                { 0.0000f, 0.2667f, 0.1059f },
        };
    } else if (colorMap == DiagramColorMap::BRBG) {
        // BrBG (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.3294f, 0.1882f, 0.0196f },
                { 0.8130f, 0.6358f, 0.3364f },
                { 0.9572f, 0.9599f, 0.9596f },
                { 0.3463f, 0.6918f, 0.6531f },
                { 0.0000f, 0.2353f, 0.1882f },
        };
    } else if (colorMap == DiagramColorMap::PUOR) {
        // PuOr (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.4980f, 0.2314f, 0.0314f },
                { 0.9364f, 0.6178f, 0.2364f },
                { 0.9662f, 0.9664f, 0.9677f },
                { 0.5942f, 0.5543f, 0.7446f },
                { 0.1765f, 0.0000f, 0.2941f },
        };
    } else if (colorMap == DiagramColorMap::RDGY) {
        // RdGy (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.4039f, 0.0000f, 0.1216f },
                { 0.8992f, 0.5144f, 0.4079f },
                { 0.9976f, 0.9976f, 0.9976f },
                { 0.6235f, 0.6235f, 0.6235f },
                { 0.1020f, 0.1020f, 0.1020f },
        };
    } else if (colorMap == DiagramColorMap::RDBU) {
        // RdBu (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.4039f, 0.0000f, 0.1216f },
                { 0.8992f, 0.5144f, 0.4079f },
                { 0.9657f, 0.9672f, 0.9681f },
                { 0.4085f, 0.6687f, 0.8145f },
                { 0.0196f, 0.1882f, 0.3804f },
        };
    } else if (colorMap == DiagramColorMap::RDYLBU) {
        // RdYlBu (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.6471f, 0.0000f, 0.1490f },
                { 0.9749f, 0.5574f, 0.3227f },
                { 0.9976f, 0.9991f, 0.7534f },
                { 0.5564f, 0.7596f, 0.8639f },
                { 0.1922f, 0.2118f, 0.5843f },
        };
    } else if (colorMap == DiagramColorMap::RDYLGN) {
        // RdYlGn (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.6471f, 0.0000f, 0.1490f },
                { 0.9749f, 0.5574f, 0.3227f },
                { 0.9971f, 0.9988f, 0.7450f },
                { 0.5181f, 0.7928f, 0.4012f },
                { 0.0000f, 0.4078f, 0.2157f },
        };
    } else if (colorMap == DiagramColorMap::SPECTRAL) {
        // Spectral (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.6196f, 0.0039f, 0.2588f },
                { 0.9749f, 0.5574f, 0.3227f },
                { 0.9981f, 0.9992f, 0.7460f },
                { 0.5273f, 0.8106f, 0.6452f },
                { 0.3686f, 0.3098f, 0.6353f },
        };
    } else if (colorMap == DiagramColorMap::COOLWARM) {
        // coolwarm (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.2298f, 0.2987f, 0.7537f },
                { 0.5543f, 0.6901f, 0.9955f },
                { 0.8674f, 0.8644f, 0.8626f },
                { 0.9567f, 0.5980f, 0.4773f },
                { 0.7057f, 0.0156f, 0.1502f },
        };
    } else if (colorMap == DiagramColorMap::BWR) {
        // bwr (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.0000f, 0.0000f, 1.0000f },
                { 0.5020f, 0.5020f, 1.0000f },
                { 1.0000f, 0.9961f, 0.9961f },
                { 1.0000f, 0.4941f, 0.4941f },
                { 1.0000f, 0.0000f, 0.0000f },
        };
    } else if (colorMap == DiagramColorMap::SEISMIC) {
        // seismic (https://matplotlib.org/stable/tutorials/colors/colormaps.html).
        colorPoints = {
                { 0.0000f, 0.0000f, 0.3000f },
                { 0.0039f, 0.0039f, 1.0000f },
                { 1.0000f, 0.9922f, 0.9922f },
                { 0.9941f, 0.0000f, 0.0000f },
                { 0.5000f, 0.0000f, 0.0000f },
        };
    } else if (colorMap == DiagramColorMap::NEON_GREENS) {
        colorPoints = {
                { 208.0f / 255.0f, 231.0f / 255.0f, 208.0f / 255.0f },
                { 100.0f / 255.0f, 255.0f / 255.0f, 100.0f / 255.0f },
        };
    } else if (int(colorMap) >= int(DiagramColorMap::NEON_GREEN) && int(colorMap) <= int(DiagramColorMap::NEON_ORANGE)) {
        sgl::Color color = defaultColors.at(int(colorMap) - int(DiagramColorMap::NEON_GREEN));
        colorPoints = {
                { color.getFloatR(), color.getFloatG(), color.getFloatB() },
                { color.getFloatR(), color.getFloatG(), color.getFloatB() },
        };
    } else if (int(colorMap) >= int(DiagramColorMap::BLACK_YELLOW) && int(colorMap) <= int(DiagramColorMap::BLACK_PURPLE)) {
        const float start = 0.3f;
        const float stop = 1.0f;
        const int numSteps = 5;
        sgl::Color baseColor = blackColorMapColors.at(int(colorMap) - int(DiagramColorMap::BLACK_YELLOW));
        //glm::vec3 baseColorLinear = sgl::TransferFunctionWindow::sRGBToLinearRGB(baseColor.getFloatColorRGB());
        for (int i = 0; i < numSteps; i++) {
            float t = float(i) / float(numSteps - 1) * (stop - start) + start;
            //colorPoints.push_back(sgl::TransferFunctionWindow::linearRGBTosRGB(t * baseColorLinear));
            colorPoints.push_back(t * baseColor.getFloatColorRGB());
        }
    } else if (int(colorMap) >= int(DiagramColorMap::BLACK_NEON_GREEN) && int(colorMap) <= int(DiagramColorMap::BLACK_NEON_ORANGE)) {
        const float start = 0.25f;
        const float stop = 1.0f;
        const int numSteps = 5;
        sgl::Color baseColor = defaultColors.at(int(colorMap) - int(DiagramColorMap::BLACK_NEON_GREEN));
        //glm::vec3 baseColorLinear = sgl::TransferFunctionWindow::sRGBToLinearRGB(baseColor.getFloatColorRGB());
        for (int i = 0; i < numSteps; i++) {
            float t = float(i) / float(numSteps - 1) * (stop - start) + start;
            //colorPoints.push_back(sgl::TransferFunctionWindow::linearRGBTosRGB(t * baseColorLinear));
            colorPoints.push_back(t * baseColor.getFloatColorRGB());
        }
    }
    return colorPoints;
}
