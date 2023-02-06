/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Christoph Neuhauser
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

#include <random>

#include <Utils/AppSettings.hpp>
#include <Input/Mouse.hpp>
#include <Math/Math.hpp>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>
#include <Graphics/Vulkan/libs/nanovg/nanovg.h>
#include <ImGui/ImGuiWrapper.hpp>

#include "DiagramBase.hpp"

DiagramBase::DiagramBase() {
    sgl::NanoVGSettings nanoVgSettings{};
    if (sgl::AppSettings::get()->getOffscreenContext()) {
        nanoVgSettings.renderBackend = sgl::RenderSystem::OPENGL;
    } else {
        nanoVgSettings.renderBackend = sgl::RenderSystem::VULKAN;
    }

    registerRenderBackendIfSupported<sgl::VectorBackendNanoVG>([this]() { this->renderBaseNanoVG(); }, nanoVgSettings);
}

void DiagramBase::initialize() {
    _initialize();
}

void DiagramBase::getNanoVGContext() {
    vg = static_cast<sgl::VectorBackendNanoVG*>(vectorBackend)->getContext();
}

void DiagramBase::renderBaseNanoVG() {
    getNanoVGContext();

    //NVGcolor backgroundFillColor = nvgRGBA(230, 230, 230, std::clamp(
    //        int(backgroundOpacity * 255), 0, 255));
    //NVGcolor backgroundStrokeColor = nvgRGBA(190, 190, 190, std::clamp(
    //        int(backgroundOpacity * 255), 0, 255));
    NVGcolor backgroundFillColor = nvgRGBA(20, 20, 20, std::clamp(
            int(backgroundOpacity * 255), 0, 255));
    NVGcolor backgroundStrokeColor = nvgRGBA(60, 60, 60, std::clamp(
            int(backgroundOpacity * 255), 0, 255));

    // Render the render target-filling widget rectangle.
    nvgBeginPath(vg);
    nvgRoundedRect(
            vg, borderWidth, borderWidth, windowWidth - 2.0f * borderWidth, windowHeight - 2.0f * borderWidth,
            borderRoundingRadius);
    nvgFillColor(vg, backgroundFillColor);
    nvgFill(vg);
    nvgStrokeColor(vg, backgroundStrokeColor);
    nvgStroke(vg);
}

/// Removes trailing zeros and unnecessary decimal points.
std::string removeTrailingZeros(const std::string& numberString) {
    size_t lastPos = numberString.size();
    for (int i = int(numberString.size()) - 1; i > 0; i--) {
        char c = numberString.at(i);
        if (c == '.') {
            lastPos--;
            break;
        }
        if (c != '0') {
            break;
        }
        lastPos--;
    }
    return numberString.substr(0, lastPos);
}

/// Removes decimal points if more than maxDigits digits are used.
std::string DiagramBase::getNiceNumberString(float number, int digits) {
    int maxDigits = digits + 2; // Add 2 digits for '.' and one digit afterwards.
    std::string outString = removeTrailingZeros(toString(number, digits, true));

    // Can we remove digits after the decimal point?
    size_t dotPos = outString.find('.');
    if (int(outString.size()) > maxDigits && dotPos != std::string::npos) {
        size_t substrSize = dotPos;
        if (int(dotPos) < maxDigits - 1) {
            substrSize = maxDigits;
        }
        outString = outString.substr(0, substrSize);
    }

    // Still too large?
    if (int(outString.size()) > maxDigits) {
        outString = toString(number, std::max(digits - 2, 1), false, false, true);
    }
    return outString;
}

void DiagramBase::drawColorLegend(
        const NVGcolor& textColor, float x, float y, float w, float h, int numLabels, size_t numTicks,
        const std::function<std::string(float)>& labelMap, const std::function<NVGcolor(float)>& colorMap,
        const std::string& textTop) {
    const int numPoints = 9;
    const int numSubdivisions = numPoints - 1;

    // Draw color bar.
    for (int i = 0; i < numSubdivisions; i++) {
        float t0 = 1.0f - float(i) / float(numSubdivisions);
        float t1 = 1.0f - float(i + 1) / float(numSubdivisions);
        NVGcolor fillColor0 = colorMap(t0);
        NVGcolor fillColor1 = colorMap(t1);
        nvgBeginPath(vg);
        nvgRect(vg, x, y + h * float(i) / float(numSubdivisions), w, h / float(numSubdivisions));
        NVGpaint paint = nvgLinearGradient(
                vg, x, y + h * float(i) / float(numSubdivisions), x, y + h * float(i+1) / float(numSubdivisions),
                fillColor0, fillColor1);
        nvgFillPaint(vg, paint);
        nvgFill(vg);
    }

    // Draw ticks.
    const float tickWidth = 4.0f;
    const float tickHeight = 1.0f;
    nvgBeginPath(vg);
    for (size_t tickIdx = 0; tickIdx < numTicks; tickIdx++) {
        //float t = 1.0f - float(tickIdx) / float(int(numTicks) - 1);
        float centerY = y + float(tickIdx) / float(int(numTicks) - 1) * h;
        nvgRect(vg, x + w, centerY - tickHeight / 2.0f, tickWidth, tickHeight);
    }
    nvgFillColor(vg, textColor);
    nvgFill(vg);

    // Draw on the right.
    nvgFontSize(vg, textSizeLegend);
    nvgFontFace(vg, "sans");
    for (size_t tickIdx = 0; tickIdx < numTicks; tickIdx++) {
        float t = 1.0f - float(tickIdx) / float(int(numTicks) - 1);
        float centerY = y + float(tickIdx) / float(int(numTicks) - 1) * h;
        std::string labelText = labelMap(t);
        nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
        nvgFillColor(vg, textColor);
        nvgText(vg, x + w + 2.0f * tickWidth, centerY, labelText.c_str(), nullptr);
    }
    nvgFillColor(vg, textColor);
    nvgFill(vg);

    // Draw text on the top.
    nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_BOTTOM);
    nvgFillColor(vg, textColor);
    nvgText(vg, x + w / 2.0f, y - 4, textTop.c_str(), nullptr);

    // Draw box outline.
    nvgBeginPath(vg);
    nvgRect(vg, x, y, w, h);
    nvgStrokeWidth(vg, 0.75f);
    nvgStrokeColor(vg, textColor);
    nvgStroke(vg);
}
