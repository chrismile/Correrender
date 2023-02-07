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

#include <Utils/AppSettings.hpp>
#include <Input/Mouse.hpp>
#include <Input/Keyboard.hpp>
#include <Math/Math.hpp>
#include <Math/Geometry/AABB2.hpp>
#include <Math/Geometry/MatrixUtil.hpp>
#include <Graphics/Color.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Vector/nanovg/nanovg.h>

#include "RadarBarChart.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const std::vector<sgl::Color> predefinedColors = {
        // RED
        sgl::Color(228, 26, 28),
        // BLUE
        sgl::Color(55, 126, 184),
        // GREEN
        sgl::Color(5, 139, 69),
        // PURPLE
        sgl::Color(129, 15, 124),
        // ORANGE
        sgl::Color(217, 72, 1),
        // PINK
        sgl::Color(231, 41, 138),
        // GOLD
        sgl::Color(254, 178, 76),
        // DARK BLUE
        sgl::Color(0, 7, 255)
};

RadarBarChart::RadarBarChart(bool equalArea) : equalArea(equalArea) {}

void RadarBarChart::initialize() {
    borderSizeX = 90;
    borderSizeY = textMode == TextMode::HORIZONTAL ? 30 + float(numVariables) / 2.0f : 110;
    chartRadius = 200;
    chartHoleRadius = 50;
    windowWidth = (chartRadius + borderSizeX) * 2.0f;
    windowHeight = (chartRadius + borderSizeY) * 2.0f;

    DiagramBase::initialize();
}

void RadarBarChart::setDataTimeIndependent(
        const std::vector<std::string>& variableNames,
        const std::vector<float>& variableValues) {
    useTimeDependentData = false;
    this->variableNames = variableNames;
    this->variableValues = variableValues;
    numVariables = variableNames.size();
    onWindowSizeChanged();
}

void RadarBarChart::setDataTimeDependent(
        const std::vector<std::string>& variableNames,
        const std::vector<std::vector<float>>& variableValuesTimeDependent) {
    useTimeDependentData = true;
    this->variableNames = variableNames;
    this->variableValuesTimeDependent = variableValuesTimeDependent;
    numVariables = variableNames.size();
    onWindowSizeChanged();
}

glm::vec3 RadarBarChart::transferFunction(float value) {
    std::vector<sgl::Color> colorPoints = {
            sgl::Color(59, 76, 192),
            sgl::Color(144, 178, 254),
            sgl::Color(220, 220, 220),
            sgl::Color(245, 156, 125),
            sgl::Color(180, 4, 38)
    };
    int stepLast = glm::clamp(int(std::floor(value / 0.25f)), 0, 4);
    int stepNext = glm::clamp(int(std::ceil(value / 0.25f)), 0, 4);
    float t = glm::fract(value / 0.25f);
    glm::vec3 colorLast = colorPoints.at(stepLast).getFloatColorRGB();
    glm::vec3 colorNext = colorPoints.at(stepNext).getFloatColorRGB();
    return glm::mix(colorLast, colorNext, t);
}

void RadarBarChart::drawPieSlice(const glm::vec2& center, int varIdx) {
    float varValue = variableValues.at(varIdx);
    if (varValue <= std::numeric_limits<float>::epsilon()) {
        return;
    }
    float radius = varValue * (chartRadius - chartHoleRadius) + chartHoleRadius;

    sgl::Color circleFillColorSgl = predefinedColors.at(varIdx % predefinedColors.size());
    //glm::vec3 hsvColor = rgbToHSV(circleFillColorSgl.getFloatColorRGB());
    //hsvColor.g *= 0.5f;
    //glm::vec3 rgbColor = hsvToRGB(hsvColor);
    glm::vec3 rgbColor = glm::mix(glm::vec3(1.0f), circleFillColorSgl.getFloatColorRGB(), 0.7f);
    NVGcolor circleFillColor = nvgRGBf(rgbColor.r, rgbColor.g, rgbColor.b);
    NVGcolor circleStrokeColor = nvgRGBA(0, 0, 0, 255);

    nvgBeginPath(vg);
    if (numVariables == 1) {
        nvgCircle(vg, center.x, center.y, radius);
    } else {
        float angleStart = mapVarIdxToAngle(float(varIdx));
        float angleEnd = mapVarIdxToAngle(float(varIdx + 1));

        if (chartHoleRadius > 0.0f) {
            nvgArc(vg, center.x, center.y, chartHoleRadius, angleEnd, angleStart, NVG_CCW);
            nvgLineTo(vg, center.x + std::cos(angleStart) * radius, center.y + std::sin(angleStart) * radius);
            nvgArc(vg, center.x, center.y, radius, angleStart, angleEnd, NVG_CW);
            nvgLineTo(
                    vg, center.x + std::cos(angleEnd) * chartHoleRadius,
                    center.y + std::sin(angleEnd) * chartHoleRadius);
        } else {
            nvgMoveTo(vg, center.x, center.y);
            nvgLineTo(vg, center.x + std::cos(angleStart) * radius, center.y + std::sin(angleStart) * radius);
            nvgArc(vg, center.x, center.y, radius, angleStart, angleEnd, NVG_CW);
            nvgLineTo(vg, center.x, center.y);
        }
    }

    nvgFillColor(vg, circleFillColor);
    nvgFill(vg);
    nvgStrokeWidth(vg, 0.75f);
    nvgStrokeColor(vg, circleStrokeColor);
    nvgStroke(vg);
}

void RadarBarChart::drawEqualAreaPieSlices(const glm::vec2& center, int varIdx) {
    const size_t numTimesteps = variableValuesTimeDependent.size();
    float radiusInner = chartHoleRadius;
    for (size_t timeStepIdx = 0; timeStepIdx < numTimesteps; timeStepIdx++) {
        float varValue = variableValuesTimeDependent.at(timeStepIdx).at(varIdx);
        float radiusOuter;
        if (equalArea) {
            radiusOuter = std::sqrt((chartRadius*chartRadius - chartHoleRadius*chartHoleRadius)
                                    / float(numTimesteps) + radiusInner*radiusInner);
        } else {
            radiusOuter =
                    chartHoleRadius + (chartRadius - chartHoleRadius) * float(timeStepIdx + 1) / float(numTimesteps);
        }

        glm::vec3 rgbColor = transferFunction(varValue);
        NVGcolor fillColor = nvgRGBf(rgbColor.r, rgbColor.g, rgbColor.b);
        NVGcolor circleStrokeColor = nvgRGBA(0, 0, 0, 255);

        nvgBeginPath(vg);
        if (numVariables == 1) {
            nvgCircle(vg, center.x, center.y, radiusOuter);
            if (chartHoleRadius > 0.0f) {
                nvgCircle(vg, center.x, center.y, radiusInner);
                nvgPathWinding(vg, NVG_HOLE);
            }
        } else {
            float angleStart = mapVarIdxToAngle(float(varIdx));
            float angleEnd = mapVarIdxToAngle(float(varIdx + 1));

            if (radiusInner > 0.0f) {
                nvgArc(vg, center.x, center.y, radiusInner, angleEnd, angleStart, NVG_CCW);
                nvgLineTo(vg, center.x + std::cos(angleStart) * radiusOuter, center.y + std::sin(angleStart) * radiusOuter);
                nvgArc(vg, center.x, center.y, radiusOuter, angleStart, angleEnd, NVG_CW);
                nvgLineTo(
                        vg, center.x + std::cos(angleEnd) * radiusInner,
                        center.y + std::sin(angleEnd) * radiusInner);
            } else {
                nvgMoveTo(vg, center.x, center.y);
                nvgLineTo(vg, center.x + std::cos(angleStart) * radiusOuter, center.y + std::sin(angleStart) * radiusOuter);
                nvgArc(vg, center.x, center.y, radiusOuter, angleStart, angleEnd, NVG_CW);
                nvgLineTo(vg, center.x, center.y);
            }
        }

        nvgFillColor(vg, fillColor);
        nvgFill(vg);
        nvgStrokeWidth(vg, 0.75f);
        nvgStrokeColor(vg, circleStrokeColor);
        nvgStroke(vg);

        radiusInner = radiusOuter;
    }
}

void RadarBarChart::drawEqualAreaPieSlicesWithLabels(const glm::vec2& center) {
    const size_t numTimesteps = variableValuesTimeDependent.size();
    const NVGcolor circleStrokeColor = nvgRGBA(0, 0, 0, 255);
    float radiusInner = chartHoleRadius;
    for (size_t timeStepIdx = 0; timeStepIdx < numTimesteps; timeStepIdx++) {
        float radiusOuter;
        if (equalArea) {
            radiusOuter = std::sqrt((chartRadius*chartRadius - chartHoleRadius*chartHoleRadius)
                                    / float(numTimesteps) + radiusInner*radiusInner);
        } else {
            radiusOuter =
                    chartHoleRadius + (chartRadius - chartHoleRadius) * float(timeStepIdx + 1) / float(numTimesteps);
        }

        sgl::Color timeStepColorSgl = predefinedColors.at(timeStepIdx % predefinedColors.size());
        //glm::vec3 hsvColor = rgbToHSV(circleFillColorSgl.getFloatColorRGB());
        //hsvColor.g *= 0.5f;
        //glm::vec3 rgbColor = hsvToRGB(hsvColor);
        glm::vec3 timeStepRgbColor = glm::mix(glm::vec3(1.0f), timeStepColorSgl.getFloatColorRGB(), 0.9f);
        NVGcolor timeStepFillColor = nvgRGBf(timeStepRgbColor.r, timeStepRgbColor.g, timeStepRgbColor.b);

        // Draw label.
        float angleStartLabel = mapVarIdxToAngle(float(numVariables) + 0.3f);
        float angleEndLabel = mapVarIdxToAngle(float(-0.3f));
        float radiusInnerLabel = radiusInner + 0.2f * (radiusOuter - radiusInner);
        float radiusOuterLabel = radiusInner + 0.8f * (radiusOuter - radiusInner);
        nvgBeginPath(vg);
        if (radiusInner > 0.0f) {
            nvgArc(vg, center.x, center.y, radiusInnerLabel, angleEndLabel, angleStartLabel, NVG_CCW);
            nvgLineTo(
                    vg, center.x + std::cos(angleStartLabel) * radiusOuterLabel,
                    center.y + std::sin(angleStartLabel) * radiusOuterLabel);
            nvgArc(vg, center.x, center.y, radiusOuterLabel, angleStartLabel, angleEndLabel, NVG_CW);
            nvgLineTo(
                    vg, center.x + std::cos(angleEndLabel) * radiusInnerLabel,
                    center.y + std::sin(angleEndLabel) * radiusInnerLabel);
        } else {
            nvgMoveTo(vg, center.x, center.y);
            nvgLineTo(
                    vg, center.x + std::cos(angleStartLabel) * radiusOuterLabel,
                    center.y + std::sin(angleStartLabel) * radiusOuterLabel);
            nvgArc(vg, center.x, center.y, radiusOuterLabel, angleStartLabel, angleEndLabel, NVG_CW);
            nvgLineTo(vg, center.x, center.y);
        }
        nvgFillColor(vg, timeStepFillColor);
        nvgFill(vg);

        for (size_t varIdx = 0; varIdx < numVariables; varIdx++) {
            float varValue = variableValuesTimeDependent.at(timeStepIdx).at(varIdx);
            glm::vec3 rgbColor = transferFunction(varValue);
            NVGcolor circleFillColor = nvgRGBf(rgbColor.r, rgbColor.g, rgbColor.b);

            nvgBeginPath(vg);
            if (numVariables == 1) {
                nvgCircle(vg, center.x, center.y, radiusOuter);
                if (chartHoleRadius > 0.0f) {
                    nvgCircle(vg, center.x, center.y, radiusInner);
                    nvgPathWinding(vg, NVG_HOLE);
                }
            } else {
                float angleStart = mapVarIdxToAngle(float(varIdx));
                float angleEnd = mapVarIdxToAngle(float(varIdx + 1));

                if (radiusInner > 0.0f) {
                    nvgArc(vg, center.x, center.y, radiusInner, angleEnd, angleStart, NVG_CCW);
                    nvgLineTo(vg, center.x + std::cos(angleStart) * radiusOuter, center.y + std::sin(angleStart) * radiusOuter);
                    nvgArc(vg, center.x, center.y, radiusOuter, angleStart, angleEnd, NVG_CW);
                    nvgLineTo(
                            vg, center.x + std::cos(angleEnd) * radiusInner,
                            center.y + std::sin(angleEnd) * radiusInner);
                } else {
                    nvgMoveTo(vg, center.x, center.y);
                    nvgLineTo(vg, center.x + std::cos(angleStart) * radiusOuter, center.y + std::sin(angleStart) * radiusOuter);
                    nvgArc(vg, center.x, center.y, radiusOuter, angleStart, angleEnd, NVG_CW);
                    nvgLineTo(vg, center.x, center.y);
                }
            }

            nvgFillColor(vg, circleFillColor);
            nvgFill(vg);
            nvgStrokeWidth(vg, 0.75f);
            nvgStrokeColor(vg, circleStrokeColor);
            nvgStroke(vg);
        }

        radiusInner = radiusOuter;
    }
}

float RadarBarChart::mapVarIdxToAngle(float varIdxFloat) {
    if (timeStepColorMode) {
        float minAngle = -float(M_PI) / 2.0f + float(M_PI) / 32.0f;
        float maxAngle = 2.0f * float(M_PI) - float(M_PI) / 2.0f - float(M_PI) / 32.0f;
        float t = varIdxFloat / float(numVariables);
        float angle = minAngle + t * (maxAngle - minAngle);
        return angle;
    } else {
        //float minAngle = -float(M_PI) / 2.0f;
        //float maxAngle = 2.0f * float(M_PI) - float(M_PI) / 2.0f;
        //float t = varIdxFloat / float(numVariables);
        //float angle = minAngle + t * (maxAngle - minAngle);
        //return angle;
        return varIdxFloat / float(numVariables) * 2.0f * float(M_PI) - float(M_PI) / 2.0f;
    }
}

void RadarBarChart::drawPieSliceTextHorizontal(const NVGcolor& textColor, const glm::vec2& center, int varIdx) {
    float radius = 1.0f * chartRadius + 10;
    float angleCenter = mapVarIdxToAngle(float(varIdx) + 0.5f);
    glm::vec2 circlePoint(center.x + std::cos(angleCenter) * radius, center.y + std::sin(angleCenter) * radius);

    float dirX = glm::clamp(std::cos(angleCenter) * 2.0f, -1.0f, 1.0f);
    float dirY = glm::clamp(std::sin(angleCenter) * 2.0f, -1.0f, 1.0f);
    float scaleFactorText = 0.0f;

    const char* text = variableNames.at(varIdx).c_str();
    glm::vec2 bounds[2];
    nvgFontSize(vg, numVariables > 50 ? 7.0f : 12.0f);
    nvgFontFace(vg, "sans");
    nvgTextBounds(vg, 0, 0, text, nullptr, &bounds[0].x);
    glm::vec2 textSize = bounds[1] - bounds[0];

    glm::vec2 textPosition(circlePoint.x, circlePoint.y);
    textPosition.x += textSize.x * (dirX - 1) * 0.5f;
    textPosition.y += textSize.y * ((dirY - 1) * 0.5f + scaleFactorText);

    nvgTextAlign(vg,NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
    if (selectedVariableIndices.find(varIdx) != selectedVariableIndices.end()) {
        nvgFontBlur(vg, 1);
        nvgFillColor(vg, nvgRGBA(255, 0, 0, 255));
        nvgText(vg, textPosition.x, textPosition.y, text, nullptr);
        nvgFontBlur(vg, 0);
    }
    nvgFillColor(vg, textColor);
    nvgText(vg, textPosition.x, textPosition.y, text, nullptr);
}

void RadarBarChart::drawPieSliceTextRotated(const NVGcolor& textColor, const glm::vec2& center, int varIdx) {
    nvgSave(vg);

    float radius = 1.0f * chartRadius + 10;
    float angleCenter = mapVarIdxToAngle(float(varIdx) + 0.5f);
    glm::vec2 circlePoint(center.x + std::cos(angleCenter) * radius, center.y + std::sin(angleCenter) * radius);

    const char* text = variableNames.at(varIdx).c_str();
    nvgFontSize(vg, numVariables > 50 ? 8.0f : 12.0f);
    nvgFontFace(vg, "sans");

    glm::vec2 textPosition(circlePoint.x, circlePoint.y);

    nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
    glm::vec2 bounds[2];
    nvgTextBounds(vg, textPosition.x, textPosition.y, text, nullptr, &bounds[0].x);

    nvgTranslate(vg, textPosition.x, textPosition.y);
    nvgRotate(vg, angleCenter);
    nvgTranslate(vg, -textPosition.x, -textPosition.y);
    if (std::cos(angleCenter) < -1e-5f) {
        nvgTranslate(vg, (bounds[0].x + bounds[1].x) / 2.0f, (bounds[0].y + bounds[1].y) / 2.0f);
        nvgRotate(vg, sgl::PI);
        nvgTranslate(vg, -(bounds[0].x + bounds[1].x) / 2.0f, -(bounds[0].y + bounds[1].y) / 2.0f);
    }

    if (selectedVariableIndices.find(varIdx) != selectedVariableIndices.end()) {
        nvgFontBlur(vg, 1);
        nvgFillColor(vg, nvgRGBA(255, 0, 0, 255));
        nvgText(vg, textPosition.x, textPosition.y, text, nullptr);
        nvgFontBlur(vg, 0);
    }
    nvgFillColor(vg, textColor);
    nvgText(vg, textPosition.x, textPosition.y, text, nullptr);

    nvgRestore(vg);
}

void RadarBarChart::drawDashedCircle(
        const NVGcolor& circleColor, const glm::vec2& center, float radius,
        int numDashes, float dashSpaceRatio, float thickness) {
    const float radiusLower = radius - thickness / 2.0f;
    const float radiusUpper = radius + thickness / 2.0f;
    const float dashSize = 2.0f * float(M_PI) * dashSpaceRatio / float(numDashes);

    nvgBeginPath(vg);
    for (int i = 0; i < numDashes; i++) {
        float angleStart = 2.0f * float(M_PI) * float(i) / float(numDashes);
        float angleEnd = angleStart + dashSize;
        glm::vec2 startPointLower(
                center.x + std::cos(angleStart) * radiusLower,
                center.y + std::sin(angleStart) * radiusLower);
        glm::vec2 endPointUpper(
                center.x + std::cos(angleEnd) * radiusUpper,
                center.y + std::sin(angleEnd) * radiusUpper);
        nvgMoveTo(vg, startPointLower.x, startPointLower.y);
        nvgArc(vg, center.x, center.y, radiusLower, angleStart, angleEnd, NVG_CW);
        nvgLineTo(vg, endPointUpper.x, endPointUpper.y);
        nvgArc(vg, center.x, center.y, radiusUpper, angleEnd, angleStart, NVG_CCW);
        nvgLineTo(vg, startPointLower.x, startPointLower.y);
    }
    nvgFillColor(vg, circleColor);
    nvgFill(vg);
}


void RadarBarChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    NVGcolor textColor = nvgRGBA(0, 0, 0, 255);
    NVGcolor circleFillColor = nvgRGBA(180, 180, 180, 70);
    NVGcolor circleStrokeColor = nvgRGBA(120, 120, 120, 120);
    NVGcolor dashedCircleStrokeColor = nvgRGBA(120, 120, 120, 120);

    // Render the central radial chart area.
    glm::vec2 center(windowWidth / 2.0f, windowHeight / 2.0f);
    nvgBeginPath(vg);
    nvgCircle(vg, center.x, center.y, chartRadius);
    if (chartHoleRadius > 0.0f) {
        nvgCircle(vg, center.x, center.y, chartHoleRadius);
        nvgPathWinding(vg, NVG_HOLE);
    }
    nvgFillColor(vg, circleFillColor);
    nvgFill(vg);
    nvgStrokeColor(vg, circleStrokeColor);
    nvgStroke(vg);

    if (!useTimeDependentData) {
        // Dotted lines at 0.25, 0.5 and 0.75.
        drawDashedCircle(
                dashedCircleStrokeColor, center, chartHoleRadius + (chartRadius - chartHoleRadius) * 0.25f,
                75, 0.5f, 0.25f);
        drawDashedCircle(
                dashedCircleStrokeColor, center, chartHoleRadius + (chartRadius - chartHoleRadius) * 0.50f,
                75, 0.5f, 0.75f);
        drawDashedCircle(
                dashedCircleStrokeColor, center, chartHoleRadius + (chartRadius - chartHoleRadius) * 0.75f,
                75, 0.5f, 0.25f);
    }


    if (useTimeDependentData) {
        if (timeStepColorMode) {
            drawEqualAreaPieSlicesWithLabels(center);
        } else {
            for (size_t varIdx = 0; varIdx < numVariables; varIdx++) {
                drawEqualAreaPieSlices(center, int(varIdx));
            }
        }
    } else {
        for (size_t varIdx = 0; varIdx < numVariables; varIdx++) {
            drawPieSlice(center, int(varIdx));
        }
    }

    if (textMode == TextMode::HORIZONTAL) {
        for (size_t varIdx = 0; varIdx < numVariables; varIdx++) {
            drawPieSliceTextHorizontal(textColor, center, int(varIdx));
        }
    } else {
        for (size_t varIdx = 0; varIdx < numVariables; varIdx++) {
            drawPieSliceTextRotated(textColor, center, int(varIdx));
        }
    }

    // Draw color legend.
    if (useTimeDependentData) {
        auto labelMap = [](float t) {
            const float EPS = 1e-5f;
            if (std::abs(t - 0.0f) < EPS) {
                return "min";
            } else if (std::abs(t - 1.0f) < EPS) {
                return "max";
            } else {
                return "";
            }
        };
        auto colorMap = [this](float t) {
            glm::vec3 color = transferFunction(t);
            return nvgRGBf(color.r, color.g, color.b);
        };
        drawColorLegend(
                textColor,
                windowWidth - colorLegendWidth - textWidthMax - 10, windowHeight - colorLegendHeight - 10,
                colorLegendWidth, colorLegendHeight, 2, 5, labelMap, colorMap);
    }
}

void RadarBarChart::update(float dt) {
    glm::vec2 mousePosition(sgl::Mouse->getX(), sgl::Mouse->getY());
    mousePosition.y = float(sgl::AppSettings::get()->getMainWindow()->getHeight()) - mousePosition.y - 1;
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();
    mousePosition.y = windowHeight - mousePosition.y;

    // Let the user click on variables to select different variables to show in linked views.
    sgl::AABB2 windowAabb(
            glm::vec2(borderWidth, borderWidth),
            glm::vec2(windowWidth - 2.0f * borderWidth, windowHeight - 2.0f * borderWidth));
    if (windowAabb.contains(mousePosition) && sgl::Mouse->buttonReleased(1)) {
        glm::vec2 center(windowWidth / 2.0f, windowHeight / 2.0f);

        nvgFontSize(vg, numVariables > 50 ? 8.0f : 12.0f);
        nvgFontFace(vg, "sans");
        if (textMode == TextMode::HORIZONTAL) {
            nvgTextAlign(vg,NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
        } else {
            nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
        }

        for (size_t varIdx = 0; varIdx < numVariables; varIdx++) {
            glm::vec2 bounds[2];
            glm::vec2 transformedMousePosition;
            if (textMode == TextMode::HORIZONTAL) {
                float radius = 1.0f * chartRadius + 10;
                float angleCenter = mapVarIdxToAngle(float(varIdx) + 0.5f);
                glm::vec2 circlePoint(center.x + std::cos(angleCenter) * radius, center.y + std::sin(angleCenter) * radius);

                float dirX = glm::clamp(std::cos(angleCenter) * 2.0f, -1.0f, 1.0f);
                float dirY = glm::clamp(std::sin(angleCenter) * 2.0f, -1.0f, 1.0f);
                float scaleFactorText = 0.0f;

                const char* text = variableNames.at(varIdx).c_str();
                glm::vec2 boundsLocal[2];
                nvgFontSize(vg, numVariables > 50 ? 7.0f : 12.0f);
                nvgFontFace(vg, "sans");
                nvgTextBounds(vg, 0, 0, text, nullptr, &boundsLocal[0].x);
                glm::vec2 textSize = boundsLocal[1] - boundsLocal[0];

                glm::vec2 textPosition(circlePoint.x, circlePoint.y);
                textPosition.x += textSize.x * (dirX - 1) * 0.5f;
                textPosition.y += textSize.y * ((dirY - 1) * 0.5f + scaleFactorText);

                nvgTextBounds(
                        vg, textPosition.x, textPosition.y,
                        variableNames.at(varIdx).c_str(), nullptr, &bounds[0].x);
                transformedMousePosition = mousePosition;
            } else {
                float radius = 1.0f * chartRadius + 10;
                float angleCenter = mapVarIdxToAngle(float(varIdx) + 0.5f);
                glm::vec2 circlePoint(
                        center.x + std::cos(angleCenter) * radius,
                        center.y + std::sin(angleCenter) * radius);
                glm::vec2 textPosition(circlePoint.x, circlePoint.y);

                nvgSave(vg);

                glm::vec2 boundsLocal[2];
                nvgTextBounds(
                        vg, textPosition.x, textPosition.y,
                        variableNames.at(varIdx).c_str(), nullptr, &boundsLocal[0].x);

                nvgTranslate(vg, textPosition.x, textPosition.y);
                nvgRotate(vg, angleCenter);
                nvgTranslate(vg, -textPosition.x, -textPosition.y);
                if (std::cos(angleCenter) < -1e-5f) {
                    nvgTranslate(
                            vg, (boundsLocal[0].x + boundsLocal[1].x) / 2.0f,
                            (boundsLocal[0].y + boundsLocal[1].y) / 2.0f);
                    nvgRotate(vg, sgl::PI);
                    nvgTranslate(
                            vg, -(boundsLocal[0].x + boundsLocal[1].x) / 2.0f,
                            -(boundsLocal[0].y + boundsLocal[1].y) / 2.0f);
                }
                nvgTextBounds(
                        vg, textPosition.x, textPosition.y,
                        variableNames.at(varIdx).c_str(), nullptr, &bounds[0].x);

                float xform[6] = {};
                float xformInv[6] = {};
                nvgCurrentTransform(vg, xform);
                nvgTransformInverse(xformInv, xform);
                nvgTransformPoint(
                        &transformedMousePosition[0], &transformedMousePosition[1],
                        xformInv, mousePosition[0], mousePosition[1]);
                nvgRestore(vg);

                glm::mat4 trafo = sgl::matrixIdentity();
                trafo *= sgl::matrixTranslation(glm::vec2(textPosition.x, textPosition.y));
                trafo = glm::rotate(trafo, angleCenter, glm::vec3(0.0f, 0.0f, 1.0f));
                trafo *= sgl::matrixTranslation(glm::vec2(-textPosition.x, -textPosition.y));
                if (std::cos(angleCenter) < -1e-5f) {
                    trafo *= sgl::matrixTranslation(
                            glm::vec2((bounds[0].x + bounds[1].x) / 2.0f, (bounds[0].y + bounds[1].y) / 2.0f));
                    trafo = glm::rotate(trafo, sgl::PI, glm::vec3(0.0f, 0.0f, 1.0f));
                    trafo *= sgl::matrixTranslation(
                            glm::vec2(-(bounds[0].x + bounds[1].x) / 2.0f, -(bounds[0].y + bounds[1].y) / 2.0f));
                }
                glm::vec4 trafoPt = glm::inverse(trafo) * glm::vec4(mousePosition.x, mousePosition.y, 0.0f, 1.0f);
                transformedMousePosition = glm::vec2(trafoPt.x, trafoPt.y);
            }

            sgl::AABB2 textAabb(bounds[0], bounds[1]);
            if (textAabb.contains(transformedMousePosition)) {
                if (selectedVariableIndices.find(varIdx) == selectedVariableIndices.end()) {
                    selectedVariableIndices.insert(varIdx);
                } else {
                    selectedVariableIndices.erase(varIdx);
                }
            }
        }
    }
}
