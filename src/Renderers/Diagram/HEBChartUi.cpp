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

#include <iostream>
#include <chrono>

#ifdef SUPPORT_SKIA
#include <core/SkCanvas.h>
#include <core/SkPaint.h>
#include <core/SkPath.h>
#include <core/SkShader.h>
#include <core/SkFont.h>
#include <core/SkFontMetrics.h>
#include <effects/SkGradientShader.h>
#endif
#ifdef SUPPORT_VKVG
#include <vkvg.h>
#endif

#include <Math/Math.hpp>
#include <Math/Geometry/Circle.hpp>
#include <Graphics/Vector/nanovg/nanovg.h>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>
#include <Input/Mouse.hpp>
#include <Input/Keyboard.hpp>

#include "Volume/VolumeData.hpp"
#ifdef SUPPORT_SKIA
#include "VectorBackendSkia.hpp"
#endif
#ifdef SUPPORT_VKVG
#include "VectorBackendVkvg.hpp"
#endif
#include "ColorSpace.hpp"
#include "HEBChart.hpp"

void HEBChartFieldData::setColorMap(DiagramColorMap _colorMap) {
    colorMapLines = _colorMap;
    initializeColorPoints();
}

void HEBChartFieldData::setColorMapVariance(DiagramColorMap _colorMap) {
    colorMapVariance = _colorMap;
    initializeColorPointsVariance();
}

void HEBChartFieldData::initializeColorPoints() {
    colorPointsLines = getColorPoints(colorMapLines);
}

void HEBChartFieldData::initializeColorPointsVariance() {
    colorPointsVariance = getColorPoints(colorMapVariance);
}

glm::vec4 HEBChartFieldData::evalColorMapVec4(float t) {
    if (std::isnan(t)) {
        if (colorMapLines == DiagramColorMap::VIRIDIS) {
            return glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        } else {
            return glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
        }
    }
    t = glm::clamp(t, 0.0f, 1.0f);
    float opacity = parent->curveOpacity;
    if (parent->opacityByValue) {
        opacity *= t * 0.75f + 0.25f;
    }
    auto N = int(colorPointsLines.size());
    float arrayPosFlt = t * float(N - 1);
    int lastIdx = std::min(int(arrayPosFlt), N - 1);
    int nextIdx = std::min(lastIdx + 1, N - 1);
    float f1 = arrayPosFlt - float(lastIdx);
    const glm::vec3& c0 = colorPointsLines.at(lastIdx);
    const glm::vec3& c1 = colorPointsLines.at(nextIdx);
    return glm::vec4(glm::mix(c0, c1, f1), opacity);
}

sgl::Color HEBChartFieldData::evalColorMap(float t) {
    return sgl::colorFromVec4(evalColorMapVec4(t));
}

glm::vec4 HEBChartFieldData::evalColorMapVec4Variance(float t, bool saturated) {
    if (!separateColorVarianceAndCorrelation) {
        return evalColorMapVec4(t);
    }
    if (std::isnan(t)) {
        if (colorMapLines == DiagramColorMap::VIRIDIS) {
            return glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        } else {
            return glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
        }
    }
    t = glm::clamp(t, 0.0f, 1.0f);
    auto N = int(colorPointsVariance.size());
    float arrayPosFlt = t * float(N - 1);
    int lastIdx = std::min(int(arrayPosFlt), N - 1);
    int nextIdx = std::min(lastIdx + 1, N - 1);
    float f1 = arrayPosFlt - float(lastIdx);
    const glm::vec3& c0 = colorPointsVariance.at(lastIdx);
    const glm::vec3& c1 = colorPointsVariance.at(nextIdx);
    glm::vec3 colorResult = glm::mix(c0, c1, f1);
    if (!saturated && *desaturateUnselectedRing) {
        colorResult = rgbToHsl(colorResult);
        colorResult.g = 0.0f;
        colorResult = hslToRgb(colorResult);
    }
    return glm::vec4(colorResult, 1.0f);
}

sgl::Color HEBChartFieldData::evalColorMapVariance(float t, bool saturated) {
    return sgl::colorFromVec4(evalColorMapVec4Variance(t, saturated));
}



HEBChart::HEBChart() {
#ifdef SUPPORT_SKIA
    registerRenderBackendIfSupported<VectorBackendSkia>([this]() { this->renderBaseSkia(); });
#endif
#ifdef SUPPORT_VKVG
    registerRenderBackendIfSupported<VectorBackendVkvg>([this]() { this->renderBaseVkvg(); });
#endif

    std::string defaultBackendId = sgl::VectorBackendNanoVG::getClassID();
#if defined(SUPPORT_SKIA) || defined(SUPPORT_VKVG)
    // NanoVG Vulkan port is broken at the moment, so use Skia or VKVG if OpenGL NanoVG cannot be used.
    if (!sgl::AppSettings::get()->getOffscreenContext()) {
#if defined(SUPPORT_SKIA)
        defaultBackendId = VectorBackendSkia::getClassID();
#elif defined(SUPPORT_VKVG)
        defaultBackendId = VectorBackendVkvg::getClassID();
#endif
    }
#endif
#if defined(__linux__) && defined(SUPPORT_VKVG)
    // OpenGL interop seems to results in kernel soft lockups as of 2023-07-06 on NVIDIA hardware.
    //sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    //if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
    //    defaultBackendId = VectorBackendVkvg::getClassID();
    //}
#endif
    setDefaultBackendId(defaultBackendId);
}

void HEBChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    totalRadius = 160;
    if (showRing) {
        chartRadius = totalRadius * (1.0f - outerRingSizePct);
    } else {
        chartRadius = totalRadius;
    }
    outerRingWidth = totalRadius - chartRadius - outerRingOffset;
    computeColorLegendHeight();

    windowWidth = (totalRadius + borderSizeX) * 2.0f + colorLegendWidth * 4.0f;
    windowHeight = (totalRadius + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onWindowSizeChanged();
}

void HEBChart::onUpdatedWindowSize() {
    if (windowWidth < 360.0f || windowHeight < 360.0f) {
        borderSizeX = borderSizeY = 10.0f;
    } else {
        borderSizeX = borderSizeY = std::min(windowWidth, windowHeight) / 36.0f;
    }
    float minDim = std::min(windowWidth - 2.0f * borderSizeX, windowHeight - 2.0f * borderSizeY);
    totalRadius = std::round(0.5f * minDim);
    if (showRing) {
        chartRadius = totalRadius * (1.0f - outerRingSizePct);
    } else {
        chartRadius = totalRadius;
    }
    outerRingWidth = totalRadius - chartRadius - outerRingOffset;
    computeColorLegendHeight();
}

void HEBChart::setDiagramMode(DiagramMode _diagramMode) {
    if (diagramMode != _diagramMode) {
        diagramMode = _diagramMode;
        dataDirty = true;
    }
}

void HEBChart::setIsFocusView(int focusViewLevel) {
    isFocusView = true;
    for (int i = 0; i < focusViewLevel; i++) {
        if (i == focusViewLevel - 1) {
            buttons.emplace_back(ButtonType::BACK);
        } else if (i == focusViewLevel - 2) {
            buttons.emplace_back(ButtonType::BACK_TWO);
        } else if (i == focusViewLevel - 3) {
            buttons.emplace_back(ButtonType::BACK_THREE);
        } else {
            buttons.emplace_back(ButtonType::BACK_FOUR);
        }
    }
    buttons.emplace_back(ButtonType::CLOSE);
}

void HEBChart::setBeta(float _beta) {
    beta = _beta;
    dataDirty = true;
}

void HEBChart::setLineCountFactor(int _factor) {
    MAX_NUM_LINES = _factor;
    dataDirty = true;
}

void HEBChart::setCurveThickness(float _curveThickness) {
    curveThickness = _curveThickness;
}

void HEBChart::setCurveOpacity(float _alpha) {
    curveOpacity = _alpha;
}

void HEBChart::setDiagramRadius(int radius) {
    totalRadius = float(radius);
    if (showRing) {
        chartRadius = totalRadius * (1.0f - outerRingSizePct);
    } else {
        chartRadius = totalRadius;
    }
    outerRingWidth = totalRadius - chartRadius - outerRingOffset;
    computeColorLegendHeight();

    windowWidth = (totalRadius + borderSizeX) * 2.0f + colorLegendWidth * 4.0f;
    windowHeight = (totalRadius + borderSizeY) * 2.0f;
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void HEBChart::setOuterRingSizePercentage(float pct) {
    outerRingSizePct = pct;
    showRing = pct > 0.0f;
    if (showRing) {
        chartRadius = totalRadius * (1.0f - outerRingSizePct);
    } else {
        chartRadius = totalRadius;
    }
    outerRingWidth = totalRadius - chartRadius - outerRingOffset;
    needsReRender = true;
}

void HEBChart::setAlignWithParentWindow(bool _align) {
    alignWithParentWindow = _align;
    renderBackgroundStroke = !_align;
    isWindowFixed = alignWithParentWindow;
    if (alignWithParentWindow) {
        updateSizeByParent();
    }
}

void HEBChart::setOpacityByValue(bool _opacityByValue) {
    opacityByValue = _opacityByValue;
    needsReRender = true;
}

void HEBChart::setColorByValue(bool _colorByValue) {
    colorByValue = _colorByValue;
    needsReRender = true;
}

void HEBChart::setUseSeparateColorVarianceAndCorrelation(bool _separate) {
    separateColorVarianceAndCorrelation = _separate;
    for (auto& fieldData : fieldDataArray) {
        fieldData->separateColorVarianceAndCorrelation = _separate;
        if (separateColorVarianceAndCorrelation) {
            fieldData->initializeColorPointsVariance();
        }
    }
    computeColorLegendHeight();
    needsReRender = true;
}

void HEBChart::setDesaturateUnselectedRing(bool _desaturate) {
    desaturateUnselectedRing = _desaturate;
    needsReRender = true;
}

void HEBChart::setUseNeonSelectionColors(bool _useNeonSelectionColors) {
    useNeonSelectionColors = _useNeonSelectionColors;
    if (useNeonSelectionColors) {
        //sgl::Color(100, 255, 100), // NEON_GREEN
        //sgl::Color(255, 60, 50),   // NEON_RED
        //sgl::Color(0, 170, 255),   // NEON_BLUE
        //sgl::Color(255, 148, 60),  // NEON_ORANGE
        circleFillColorSelected0 = sgl::Color(255, 60, 50, 255);
        circleFillColorSelected1 = sgl::Color(0, 170, 255, 255);
    } else {
        circleFillColorSelected0 = sgl::Color(180, 80, 80, 255);
        circleFillColorSelected1 = sgl::Color(50, 100, 180, 255);
    }
    needsReRender = true;
}

void HEBChart::setUseGlobalStdDevRange(bool _useGlobalStdDevRange) {
    useGlobalStdDevRange = _useGlobalStdDevRange;
    needsReRender = true;
}

void HEBChart::setShowSelectedRegionsByColor(bool _show) {
    showSelectedRegionsByColor = _show;
    needsReRender = true;
}

void HEBChart::setColorMap(int fieldIdx, DiagramColorMap _colorMap) {
    fieldDataArray.at(fieldIdx)->setColorMap(_colorMap);
    needsReRender = true;
}

void HEBChart::setColorMapVariance(DiagramColorMap _colorMap) {
    colorMapVariance = _colorMap;
    for (auto& fieldData : fieldDataArray) {
        fieldData->setColorMapVariance(_colorMap);
    }
    needsReRender = true;
}

/**
 * Computes the distance of a point to a line segment.
 * See: http://geomalgorithms.com/a02-_lines.html
 *
 * @param p The position of the point.
 * @param l0 The first line point.
 * @param l1 The second line point.
 * @return The distance of p to the line segment.
 */
inline float getDistanceToLineSegment(glm::vec2 p, glm::vec2 l0, glm::vec2 l1) {
    glm::vec2 v = l1 - l0;
    glm::vec2 w = p - l0;
    float c1 = glm::dot(v, w);
    if (c1 <= 0.0) {
        return glm::length(p - l0);
    }

    float c2 = glm::dot(v, v);
    if (c2 <= c1) {
        return glm::length(p - l1);
    }

    float b = c1 / c2;
    glm::vec2 pb = l0 + b * v;
    return glm::length(p - pb);
}

inline int sign(float x) { return x > 0.0f ? 1 : (x < 0.0f ? -1 : 0); }

bool isInsidePolygon(const std::vector<glm::vec2>& polygon, const glm::vec2& pt) {
    int firstSide = 0;
    auto n = int(polygon.size());
    for (int i = 0; i < n; i++) {
        const glm::vec2& p0 = polygon.at(i);
        const glm::vec2& p1 = polygon.at((i + 1) % n);
        int side = sign((p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0]));
        if (i == 0) {
            firstSide = side;
        } else if (firstSide != side) {
            return false;
        }
    }
    return true;
}

void HEBChart::updateSizeByParent() {
    auto [parentWidth, parentHeight] = getBlitTargetSize();
    auto ssf = float(blitTargetSupersamplingFactor);
    windowOffsetX = 0;
    windowOffsetY = 0;
    windowWidth = float(parentWidth) / (scaleFactor * float(ssf));
    windowHeight = float(parentHeight) / (scaleFactor * float(ssf));
    onUpdatedWindowSize();
    onWindowSizeChanged();
    updateButtonsLayout();
}

void HEBChart::updateButtonsLayout() {
    const float buttonSize = std::clamp(totalRadius * 0.125f, 10.0f, 30.0f);
    for (size_t buttonIdx = 0; buttonIdx < buttons.size(); buttonIdx++) {
        auto& button = buttons.at(buttonIdx);
        if (button.getButtonType() == ButtonType::CLOSE) {
            // The close button is on the right.
            button.setPosition(windowWidth - borderSizeX - buttonSize, borderSizeY);
        } else {
            // Order the other buttons from left to right with spacing.
            button.setPosition(borderSizeX + float(buttonIdx) * (buttonSize * 1.25f), borderSizeY);
        }
        button.setSize(buttonSize);
    }
}

void HEBChart::update(float dt) {
    // Check whether the mouse is hovering a button. In this case, window move and resize events should be disabled.
    glm::vec2 mousePosition(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePosition -= glm::vec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();
    bool isAnyButtonHovered = false;
    for (auto& button : buttons) {
        isAnyButtonHovered = isAnyButtonHovered || button.getIsHovered(mousePosition);
    }
    isMouseGrabbedByParent = isMouseGrabbedByParent || isAnyButtonHovered;

    DiagramBase::update(dt);

    mousePosition = glm::vec2(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePosition -= glm::vec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    //else {
    //    mousePosition.y = float(sgl::AppSettings::get()->getMainWindow()->getHeight()) - mousePosition.y - 1;
    //}
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();
    //mousePosition.y = windowHeight - mousePosition.y;

    sgl::AABB2 windowAabb(
            glm::vec2(borderWidth, borderWidth),
            glm::vec2(windowWidth - 2.0f * borderWidth, windowHeight - 2.0f * borderWidth));
    bool isMouseInWindow = windowAabb.contains(mousePosition) && !isMouseGrabbedByParent;

    // Update the buttons.
    updateButtonsLayout();
    isAnyButtonHovered = false;
    for (size_t buttonIdx = 0; buttonIdx < buttons.size(); buttonIdx++) {
        auto& button = buttons.at(buttonIdx);
        if (button.update(dt, mousePosition)) {
            needsReRender = true;
        }
        if (button.getButtonTriggered()) {
            if (button.getButtonType() == ButtonType::CLOSE) {
                returnToViewIdx = 0;
            } else if (isAnyBackButton(button.getButtonType())) {
                returnToViewIdx = int(buttonIdx);
            }
        }
        isAnyButtonHovered = isAnyButtonHovered || button.getIsHovered(mousePosition);
    }
    isMouseInWindow = isMouseInWindow && !isAnyButtonHovered;

    if (!isMouseInWindow) {
        if (hoveredPointIdx != -1) {
            hoveredPointIdx = -1;
            needsReRender = true;
        }
        if (hoveredLineIdx != -1) {
            hoveredLineIdx = -1;
            needsReRender = true;
        }
        if (hoveredGridIdx.has_value()) {
            hoveredGridIdx = {};
            needsReRender = true;
        }
    } else if (diagramMode == DiagramMode::CHORD) {
        //auto startTime = std::chrono::system_clock::now();
        glm::vec2 centeredMousePos = mousePosition - glm::vec2(windowWidth / 2.0f, windowHeight / 2.0f);
        float radiusMouse = std::sqrt(centeredMousePos.x * centeredMousePos.x + centeredMousePos.y * centeredMousePos.y);
        float phiMouse = std::fmod(std::atan2(centeredMousePos.y, centeredMousePos.x) + sgl::TWO_PI, sgl::TWO_PI);

        // Factor 4 is used so the user does not need to exactly hit the (potentially very small) points.
        float minRadius = chartRadius - pointRadiusBase * 4.0f;
        float maxRadius = chartRadius + pointRadiusBase * 4.0f;

        if (radiusMouse >= minRadius && radiusMouse <= maxRadius && !nodesList.empty()) {
            if (regionsEqual) {
                auto numLeaves = int(pointToNodeIndexMap0.size());
                int sectorIdx = int(std::round(phiMouse / sgl::TWO_PI * float(numLeaves))) % numLeaves;
                float sectorCenterAngle = float(sectorIdx) / float(numLeaves) * sgl::TWO_PI;
                sgl::Circle circle(
                        chartRadius * glm::vec2(std::cos(sectorCenterAngle), std::sin(sectorCenterAngle)),
                        pointRadiusBase * 4.0f);
                if (circle.contains(centeredMousePos)) {
                    hoveredPointIdx = sectorIdx;
                } else {
                    hoveredPointIdx = -1;
                }
            } else {
                float minAngleGroup0 = nodesList.at(leafIdxOffset).angle;
                float maxAngleGroup0 = nodesList.at(regionsEqual ? nodesList.size() - 1 : leafIdxOffset1 - 1).angle;
                float minAngleGroup1 = nodesList.at(leafIdxOffset1).angle;
                float maxAngleGroup1 = nodesList.at(nodesList.size() - 1).angle;

                auto numLeaves0 = int(pointToNodeIndexMap0.size());
                auto numLeaves1 = int(pointToNodeIndexMap1.size());
                int sectorIdx0 = int(std::round((phiMouse - minAngleGroup0) / (maxAngleGroup0 - minAngleGroup0) * float(numLeaves0 - 1)));
                int sectorIdx1 = int(std::round((phiMouse - minAngleGroup1) / (maxAngleGroup1 - minAngleGroup1) * float(numLeaves1 - 1)));
                int distanceSector0 = sectorIdx0 < 0 ? -sectorIdx0 : (sectorIdx0 > numLeaves0 - 1 ? sectorIdx0 - numLeaves0 + 1 : 0);
                int distanceSector1 = sectorIdx1 < 0 ? -sectorIdx1 : (sectorIdx1 > numLeaves1 - 1 ? sectorIdx1 - numLeaves1 + 1 : 0);

                int closestGroupIdx = distanceSector0 <= distanceSector1 ? 0 : 1;
                int numLeaves = closestGroupIdx == 0 ? numLeaves0 : numLeaves1;
                float minAngleGroup = closestGroupIdx == 0 ? minAngleGroup0 : minAngleGroup1;
                float maxAngleGroup = closestGroupIdx == 0 ? maxAngleGroup0 : maxAngleGroup1;
                int sectorIdx = std::clamp(closestGroupIdx == 0 ? sectorIdx0 : sectorIdx1, 0, numLeaves - 1);

                float sectorCenterAngle = float(sectorIdx) / float(numLeaves - 1) * (maxAngleGroup - minAngleGroup) + minAngleGroup;
                sgl::Circle circle(
                        chartRadius * glm::vec2(std::cos(sectorCenterAngle), std::sin(sectorCenterAngle)),
                        pointRadiusBase * 4.0f);
                if (circle.contains(centeredMousePos)) {
                    hoveredPointIdx = sectorIdx + (closestGroupIdx == 0 ? 0 : int(leafIdxOffset1 - leafIdxOffset));
                } else {
                    hoveredPointIdx = -1;
                }
            }
        } else {
            hoveredPointIdx = -1;
        }

        // Select a line.
        const float minDist = 4.0f;
        const float minDistHalf = 2.0f;
        if (!curvePoints.empty() && hoveredPointIdx < 0) {
            // TODO: Test if point lies in convex hull of control points first (using @see isInsidePolygon).
            int closestLineIdx = -1;
            float closestLineDist = std::numeric_limits<float>::max();
            float closestLineCorrelationValue = std::numeric_limits<float>::lowest();
            for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
                float correlationValue = std::abs(correlationValuesArray.at(lineIdx));
                for (int ptIdx = 0; ptIdx < NUM_SUBDIVISIONS - 1; ptIdx++) {
                    glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                    pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
                    pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
                    glm::vec2 pt1 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx + 1);
                    pt1.x = windowWidth / 2.0f + pt1.x * chartRadius;
                    pt1.y = windowHeight / 2.0f + pt1.y * chartRadius;
                    float dist = getDistanceToLineSegment(mousePosition, pt0, pt1);
                    if ((dist <= minDist && dist < closestLineDist && (closestLineDist > minDistHalf || correlationValue > closestLineCorrelationValue))
                            || (dist <= minDistHalf && correlationValue > closestLineCorrelationValue)) {
                        closestLineIdx = lineIdx;
                        closestLineDist = dist;
                        closestLineCorrelationValue = correlationValue;
                    }
                }
            }
            if (closestLineIdx >= 0 && closestLineDist <= minDist) {
                hoveredLineIdx = closestLineIdx;
            } else {
                hoveredLineIdx = -1;
            }
        } else {
            hoveredLineIdx = -1;
        }
        //auto endTime = std::chrono::system_clock::now();
        //auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        //std::cout << "Elapsed time update: " << elapsedTime.count() << "ms" << std::endl;
    } else if (diagramMode == DiagramMode::MATRIX && correlationMatrix) {
        float startX = windowWidth / 2.0f - chartRadius;
        float startY = windowHeight / 2.0f - chartRadius;
        float matrixWidth = 2.0f * chartRadius;
        float matrixHeight = 2.0f * chartRadius;
        glm::vec2 pctMousePos = (mousePosition - glm::vec2(startX, startY)) / glm::vec2(matrixWidth, matrixHeight);
        glm::vec2 gridMousePos = pctMousePos * glm::vec2(correlationMatrix->getNumRows(), correlationMatrix->getNumColumns());
        auto gridPosition = glm::ivec2(gridMousePos);
        int i = gridPosition.x;
        int j = gridPosition.y;

        if (pctMousePos.x >= 0.0f && pctMousePos.y >= 0.0f && pctMousePos.x < 1.0f && pctMousePos.y < 1.0f
                && (!correlationMatrix->getIsSymmetric() || i > j)) {
            if (!regionsEqual) {
                gridPosition.y += int(leafIdxOffset1 - leafIdxOffset);
            }
            hoveredGridIdx = gridPosition;
        } else {
            hoveredGridIdx = {};
        }

        hoveredPointIdx = -1;
        hoveredLineIdx = -1;
        selectedLineIdx = -1;
    }

    if (isMouseInWindow && (sgl::Mouse->buttonReleased(1) || sgl::Mouse->buttonReleased(3))
            && !windowMoveOrResizeJustFinished) {
        bool showCorrelationForClickedPointNew = false;
        clickedLineIdx = -1;
        clickedGridIdx = {};
        clickedPointIdx = -1;
        bool showCorrelationForClickedPointChanged = false;
        if (diagramMode == DiagramMode::CHORD && hoveredLineIdx >= 0) {
            clickedLineIdx = hoveredLineIdx;
            // Don't reset lines for clicked point if one of those lines was selected.
            showCorrelationForClickedPointNew = showCorrelationForClickedPoint;
        } else if (diagramMode == DiagramMode::MATRIX && hoveredGridIdx.has_value()) {
            clickedGridIdx = hoveredGridIdx;
            showCorrelationForClickedPointNew = showCorrelationForClickedPoint;
        } else if (hoveredPointIdx >= 0) {
            clickedPointIdx = hoveredPointIdx;
            if (regionsEqual && (sgl::Keyboard->getModifier() & KMOD_CTRL) != 0) {
                showCorrelationForClickedPointNew = true;
                showCorrelationForClickedPointChanged = true;
                clickedPointGridIdx = uint32_t(std::find(
                        pointToNodeIndexMap0.begin(), pointToNodeIndexMap0.end(),
                        int(leafIdxOffset) + clickedPointIdx) - pointToNodeIndexMap0.begin());
            }
        }
        if (showCorrelationForClickedPoint != showCorrelationForClickedPointNew) {
            showCorrelationForClickedPoint = showCorrelationForClickedPointNew;
            dataDirty = true;
        }
        // Left click opens focus view, right click only selects the line.
        isFocusSelectionReset = !sgl::Mouse->buttonReleased(1) || showCorrelationForClickedPointChanged;
        if (isFocusSelectionReset) {
            clickedLineIdxOld = -1;
            clickedGridIdxOld = {};
            clickedPointIdxOld = -1;
        }
    }

    int newSelectedLineIdx = -1;
    int newSelectedPointIndices[2] = { -1, -1 };
    std::optional<glm::ivec2> newSelectedGridIdx{};
    if (hoveredLineIdx >= 0) {
        newSelectedLineIdx = hoveredLineIdx;
    } else if (hoveredGridIdx.has_value()) {
        newSelectedGridIdx = hoveredGridIdx;
    } else if (clickedLineIdx >= 0 && hoveredPointIdx < 0) {
        newSelectedLineIdx = clickedLineIdx;
    } else if (clickedGridIdx.has_value() && hoveredPointIdx < 0) {
        newSelectedGridIdx = clickedGridIdx;
    }

    if (newSelectedLineIdx >= 0) {
        const auto& points = connectedPointsArray.at(newSelectedLineIdx);
        newSelectedPointIndices[0] = points.first;
        newSelectedPointIndices[1] = points.second;
    } else if (newSelectedGridIdx.has_value()) {
        newSelectedPointIndices[0] = newSelectedGridIdx->x;
        newSelectedPointIndices[1] = newSelectedGridIdx->y;
    } else if (hoveredPointIdx >= 0) {
        newSelectedPointIndices[0] = hoveredPointIdx;
    } else if (clickedPointIdx >= 0) {
        newSelectedPointIndices[0] = clickedPointIdx;
    }

    if (selectedPointIndices[0] != newSelectedPointIndices[0] || selectedPointIndices[1] != newSelectedPointIndices[1]
            || selectedLineIdx != newSelectedLineIdx || selectedGridIdx != newSelectedGridIdx) {
        needsReRender = true;
    }
    selectedLineIdx = newSelectedLineIdx;
    selectedGridIdx = newSelectedGridIdx;
    selectedPointIndices[0] = newSelectedPointIndices[0];
    selectedPointIndices[1] = newSelectedPointIndices[1];
}

std::pair<float, float> HEBChart::getMinMaxCorrelationValue() {
    if (diagramMode == DiagramMode::CHORD) {
        float minValue = correlationValuesArray.empty() ? 0.0f : correlationValuesArray.front();
        float maxValue = correlationValuesArray.empty() ? 0.0f : correlationValuesArray.back();
        if (!useAbsoluteCorrelationMeasure && isMeasureMI(correlationMeasureType)) {
            maxValue = std::abs(maxValue);
            minValue = -maxValue;
        }
        return std::make_pair(minValue, maxValue);
    } else {
        return std::make_pair(minCorrelationValueGlobal, maxCorrelationValueGlobal);
    }
}

void HEBChart::renderCorrelationMatrix() {
#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr, *gradientPaint = nullptr;
    if (canvas) {
        paint = new SkPaint;
        gradientPaint = new SkPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(paint);
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(gradientPaint);
    }
#endif

    auto [minMi, maxMi] = getMinMaxCorrelationValue();
    auto numFields = int(fieldDataArray.size());
    //for (int fieldIdx = 0; fieldIdx < numFields; fieldIdx++) {
    if (numFields > 0) {
        int fieldIdx = 0;
        float startX = windowWidth / 2.0f - chartRadius;
        float startY = windowHeight / 2.0f - chartRadius;
        float matrixWidth = 2.0f * chartRadius;
        float matrixHeight = 2.0f * chartRadius;

        auto* fieldData = fieldDataArray.at(fieldIdx).get();
        bool isSymmetric = correlationMatrix->getIsSymmetric();
        int numRows = correlationMatrix->getNumRows();
        int numColumns = correlationMatrix->getNumRows();
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numColumns; j++) {
                if (isSymmetric && i <= j) {
                    continue;
                }
                float correlationValue = correlationMatrix->get(i, j);
                if (std::isnan(correlationValue)) {
                    continue;
                }
                float factor = 1.0f;
                if (minMi < maxMi) {
                    factor = (correlationValue - minMi) / (maxMi - minMi);
                }

                float w = matrixWidth * float(1) / float(numRows);
                float h = matrixHeight * float(1) / float(numColumns);
                //float x = startX + float(numRows - i - 1) * w;
                float x = startX + float(i) * w;
                float y = startY + float(j) * h;
                w += 1e-1f;
                h += 1e-1f;

                if (vg) {
                    glm::vec4 color = fieldData->evalColorMapVec4(factor);
                    nvgBeginPath(vg);
                    nvgRect(vg, x, y, w, h);
                    nvgFillColor(vg, nvgRGBAf(color.x, color.y, color.z, 1.0f));
                    nvgFill(vg);
                }
#ifdef SUPPORT_SKIA
                else if (canvas) {
                    sgl::Color color = fieldData->evalColorMap(factor);
                    color.setA(255);
                    paint->setStroke(false);
                    paint->setColor(toSkColor(color));
                    canvas->drawRect(SkRect{x * s, y * s, (x + w) * s, (y + h) * s}, *paint);
                }
#endif
#ifdef SUPPORT_VKVG
                else if (context) {
                    sgl::Color color = fieldData->evalColorMap(factor);
                    color.setA(255);
                    vkvg_rectangle(context, x * s, y * s, w * s, h * s);
                    vkvg_set_source_color(context, color.getColorRGBA());
                    vkvg_fill(context);
                }
#endif
            }
        }

        // Draw the current selection.
        if (selectedGridIdx.has_value()) {
            int offset = regionsEqual ? 0 : int(leafIdxOffset1 - leafIdxOffset);
            int i = selectedPointIndices[0];
            int j = selectedPointIndices[1] - offset;
            float w = matrixWidth * float(1) / float(numRows);
            float h = matrixHeight * float(1) / float(numColumns);
            float x = startX + float(i) * w;
            float y = startY + float(j) * h;

            sgl::Color strokeColor = isDarkMode ? circleStrokeColorDark : circleStrokeColorBright;
            if (vg) {
                nvgBeginPath(vg);
                nvgRect(vg, x, y, w, h);
                NVGcolor strokeColorNvg = nvgRGBA(
                        strokeColor.getR(), strokeColor.getG(), strokeColor.getB(), strokeColor.getA());
                nvgStrokeColor(vg, strokeColorNvg);
                nvgStroke(vg);
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                paint->setStroke(true);
                paint->setStrokeWidth(1.0f * s);
                paint->setColor(toSkColor(strokeColor));
                canvas->drawRect(SkRect{x * s, y * s, (x + w) * s, (y + h) * s}, *paint);
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                vkvg_rectangle(context, x * s, y * s, w * s, h * s);
                vkvg_set_source_color(context, strokeColor.getColorRGBA());
                vkvg_set_line_width(context, 1.0f * s);
                vkvg_stroke(context);
            }
#endif
        }

        if (showRing) {
            for (int dim = 0; dim < 2; dim++) {
                std::pair<float, float> stdDevRange;
                if (useGlobalStdDevRange) {
                    if (!fieldData->useTwoFields) {
                        stdDevRange = getGlobalStdDevRange(fieldData->selectedFieldIdx, 0);
                    } else {
                        stdDevRange = getGlobalStdDevRange(fieldData->selectedFieldIdx, dim);
                    }
                } else {
                    stdDevRange = std::make_pair(fieldData->minStdDev, fieldData->maxStdDev);
                }

                std::vector<float>* leafStdDevArray = nullptr;
                if (fieldData->useTwoFields) {
                    leafStdDevArray = dim == 0 ? &fieldData->leafStdDevArray : &fieldData->leafStdDevArray2;
                } else {
                    leafStdDevArray = &fieldData->leafStdDevArray;
                }

                auto lower = int(leafIdxOffset);
                auto upper = int(nodesList.size());
                if (!regionsEqual) {
                    if (dim == 0) {
                        upper = int(leafIdxOffset1);
                    } else {
                        lower = int(leafIdxOffset1);
                    }
                }

                float barW = dim == 0 ? matrixWidth : outerRingWidth;
                float barH = dim == 1 ? matrixHeight : outerRingWidth;
                float barX = dim == 0 ? startX : startX - outerRingWidth;
                float barY = dim == 1 ? startY : startY - outerRingWidth;

                for (int leafIdx = lower; leafIdx < upper; leafIdx++) {
                    float stdev0 = leafStdDevArray->at(leafIdx - int(leafIdxOffset));
                    float t0 = stdDevRange.first == stdDevRange.second ? 0.0f : (stdev0 - stdDevRange.first) / (stdDevRange.second - stdDevRange.first);
                    //float stdev1 = leafStdDevArray->at(leafIdx + 1 - int(leafIdxOffset));
                    //float t1 = stdDevRange.first == stdDevRange.second ? 0.0f : (stdev1 - stdDevRange.first) / (stdDevRange.second - stdDevRange.first);

                    /*float rlo = chartRadius + outerRingOffset + 0.0f * outerRingWidth;
                    float rmi = chartRadius + outerRingOffset + 0.5f * outerRingWidth;
                    float rhi = chartRadius + outerRingOffset + 1.0f * outerRingWidth;

                    glm::vec2 mi0, mi1;
                    float x, y, w, h;
                    if (dim == 0) {
                        x = barX + barW * float(leafIdx - lower) / float(upper - lower - 1);
                        w = ;
                        y = barY;
                        h = barH;
                        mi0.x = x;
                        mi0.y = y;
                        mi1.x = x + w;
                        mi1.y = y;
                    }*/

                    float x, y, w, h;
                    if (dim == 0) {
                        w = barW * float(1) / float(upper - lower);
                        h = barH;
                        x = barX + float(leafIdx - lower) * w;
                        y = barY;
                    } else {
                        w = barW;
                        h = barH * float(1) / float(upper - lower);
                        x = barX;
                        y = barY + float(leafIdx - lower) * h;
                    }

                    if (vg) {
                        glm::vec4 rgbColor0 = fieldData->evalColorMapVec4Variance(t0, true);
                        rgbColor0.w = 1.0f;
                        //glm::vec4 rgbColor1 = fieldData->evalColorMapVec4Variance(t1, true);
                        //rgbColor1.w = 1.0f;
                        NVGcolor fillColor0 = nvgRGBAf(rgbColor0.x, rgbColor0.y, rgbColor0.z, rgbColor0.w);
                        //NVGcolor fillColor1 = nvgRGBAf(rgbColor1.x, rgbColor1.y, rgbColor1.z, rgbColor1.w);

                        nvgBeginPath(vg);
                        nvgRect(vg, x, y, w, h);

                        //NVGpaint paint = nvgLinearGradient(vg, mi0.x, mi0.y, mi1.x, mi1.y, fillColor0, fillColor1);
                        //nvgFillPaint(vg, paint);
                        nvgFillColor(vg, fillColor0);
                        nvgFill(vg);
                    }
#ifdef SUPPORT_SKIA
                    else if (canvas) {
                        sgl::Color rgbColor0 = fieldData->evalColorMapVariance(t0, true);
                        rgbColor0.setA(255);

                        paint->setStroke(false);
                        paint->setColor(toSkColor(rgbColor0));
                        canvas->drawRect(SkRect{x * s, y * s, (x + w) * s, (y + h) * s}, *paint);
                    }
#endif
#ifdef SUPPORT_VKVG
                    else if (context) {
                        sgl::Color rgbColor0 = fieldData->evalColorMapVariance(t0, true);
                        rgbColor0.setA(255);
                        //sgl::Color rgbColor1 = fieldData->evalColorMapVariance(t1, true);
                        //rgbColor1.setA(255);

                        vkvg_rectangle(context, x * s, y * s, w * s, h * s);

                        //auto pattern = vkvg_pattern_create_linear(mi0.x * s, mi0.y * s, mi1.x * s, mi1.y * s);
                        //vkvg_pattern_add_color_stop(pattern, 0.0f, rgbColor0.x, rgbColor0.y, rgbColor0.z, 1.0f);
                        //vkvg_pattern_add_color_stop(pattern, 1.0f, rgbColor1.x, rgbColor1.y, rgbColor1.z, 1.0f);
                        //vkvg_set_source(context, pattern);
                        //vkvg_pattern_destroy(pattern);
                        vkvg_set_source_color(context, rgbColor0.getColorRGBA());
                        vkvg_fill(context);
                    }
#endif
                }

                sgl::Color strokeColor = isDarkMode ? circleStrokeColorDark : circleStrokeColorBright;
                if (vg) {
                    nvgBeginPath(vg);
                    nvgRect(vg, barX, barY, barW, barH);
                    NVGcolor strokeColorNvg = nvgRGBA(
                            strokeColor.getR(), strokeColor.getG(), strokeColor.getB(), strokeColor.getA());
                    nvgStrokeColor(vg, strokeColorNvg);
                    nvgStroke(vg);
                }
#ifdef SUPPORT_SKIA
                else if (canvas) {
                    paint->setStroke(true);
                    paint->setStrokeWidth(1.0f * s);
                    paint->setColor(toSkColor(strokeColor));
                    canvas->drawRect(SkRect{barX * s, barY * s, (barX + barW) * s, (barY + barH) * s}, *paint);
                }
#endif
#ifdef SUPPORT_VKVG
                else if (context) {
                    vkvg_rectangle(context, barX * s, barY * s, barW * s, barH * s);
                    vkvg_set_source_color(context, strokeColor.getColorRGBA());
                    vkvg_set_line_width(context, 1.0f * s);
                    vkvg_stroke(context);
                }
#endif
            }
        }
    }

    if (selectedPointIndices[0] >= 0) {
        drawSelectionArrows();
    } else if (!regionsEqual && showSelectedRegionsByColor && isFocusView && !fieldDataArray.empty()) {
        for (int dim = 0; dim < 2; dim++) {
            float x0, y0, x1, y1;
            if (dim == 0) {
                x0 = windowWidth / 2.0f - chartRadius;
                y0 = windowHeight / 2.0f - totalRadius;
                x1 = windowWidth / 2.0f + chartRadius;
                y1 = y0;
            } else {
                x0 = windowWidth / 2.0f - totalRadius;
                y0 = windowHeight / 2.0f - chartRadius;
                x1 = x0;
                y1 = windowHeight / 2.0f + chartRadius;
            }
            const sgl::Color& circleFillColorSelected = getColorSelected(dim);
            if (vg) {
                auto circleFillColorSelectedNvg = nvgRGBA(
                        circleFillColorSelected.getR(), circleFillColorSelected.getG(),
                        circleFillColorSelected.getB(), circleFillColorSelected.getA());
                nvgBeginPath(vg);
                nvgMoveTo(vg, x0, y0);
                nvgLineTo(vg, x1, y1);
                nvgStrokeWidth(vg, 3.0f);
                nvgStrokeColor(vg, circleFillColorSelectedNvg);
                nvgStroke(vg);
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                paint->setStroke(true);
                paint->setStrokeWidth(3.0f * s);
                paint->setColor(toSkColor(circleFillColorSelected));
                canvas->drawLine(x0 * s, y0 * s, x1 * s, y1 * s, *paint);
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                vkvg_move_to(context, x0 * s, y0 * s);
                vkvg_line_to(context, x1 * s, y1 * s);
                vkvg_set_line_width(context, 3.0f * s);
                vkvg_set_source_color(context, circleFillColorSelected.getColorRGBA());
                vkvg_stroke(context);
            }
#endif
        }
    }


    // Render buttons.
    for (auto& button : buttons) {
        button.render(
                vg,
#ifdef SUPPORT_SKIA
                canvas, paint,
#endif
#ifdef SUPPORT_VKVG
                context,
#endif
                isDarkMode, s);
    }

    // Draw color legend.
    if (shallDrawColorLegend) {
        drawColorLegends();
    }

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
        delete gradientPaint;
    }
#endif
}

void HEBChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    if (diagramMode == DiagramMode::CHORD) {
        renderChordDiagramNanoVG();
    } else {
        renderCorrelationMatrix();
    }
}

void HEBChart::renderChordDiagramNanoVG() {
    auto [minMi, maxMi] = getMinMaxCorrelationValue();

    // Draw the B-spline curves.
    NVGcolor curveStrokeColor = nvgRGBA(
            100, 255, 100, uint8_t(std::clamp(int(std::ceil(curveOpacity * 255.0f)), 0, 255)));
    if (!curvePoints.empty()) {
        nvgStrokeWidth(vg, curveThickness);
        for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
            if (lineIdx == selectedLineIdx) {
                continue;
            }
            nvgBeginPath(vg);
            glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            nvgMoveTo(vg, pt0.x, pt0.y);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                nvgLineTo(vg, pt.x, pt.y);
            }

            if (colorByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(lineIdx)).get();
                glm::vec4 color = fieldData->evalColorMapVec4(factor);
                curveStrokeColor = nvgRGBAf(color.x, color.y, color.z, color.w);
            } else if (opacityByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                }
                curveStrokeColor.a = curveOpacity * factor;
            } else {
                curveStrokeColor.a = curveOpacity;
            }
            nvgStrokeColor(vg, curveStrokeColor);
            nvgStroke(vg);
        }

        if (selectedLineIdx >= 0) {
            // Background color outline.
            sgl::Color outlineColor = isDarkMode ? backgroundFillColorDark : backgroundFillColorBright;
            nvgStrokeWidth(vg, curveThickness * 3.0f);
            nvgBeginPath(vg);
            glm::vec2 pt0 = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            nvgMoveTo(vg, pt0.x, pt0.y);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                nvgLineTo(vg, pt.x, pt.y);
            }
            nvgStrokeColor(vg, nvgRGBA(
                    outlineColor.getR(), outlineColor.getG(), outlineColor.getB(), outlineColor.getA()));
            nvgStroke(vg);

            // Line itself.
            nvgStrokeWidth(vg, curveThickness * 2.0f);
            nvgBeginPath(vg);
            nvgMoveTo(vg, pt0.x, pt0.y);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                nvgLineTo(vg, pt.x, pt.y);
            }
            if (colorByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(selectedLineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(selectedLineIdx)).get();
                glm::vec4 color = fieldData->evalColorMapVec4(factor);
                curveStrokeColor = nvgRGBAf(color.x, color.y, color.z, color.w);
            }
            curveStrokeColor.a = 1.0f;
            nvgStrokeColor(vg, curveStrokeColor);
            nvgStroke(vg);
        }
    }

    // Draw the point circles.
    float pointRadius = curveThickness * pointRadiusBase;
    //sgl::Color colors[8] = {
    //        sgl::Color(179,226,205), sgl::Color(253,205,172), sgl::Color(203,213,232), sgl::Color(244,202,228),
    //        sgl::Color(230,245,201), sgl::Color(255,242,174), sgl::Color(241,226,204), sgl::Color(204,204,204)
    //};
    /*sgl::Color colors[8] = {
            sgl::Color(228,26,28), sgl::Color(55,126,184), sgl::Color(77,175,74), sgl::Color(152,78,163),
            sgl::Color(255,127,0), sgl::Color(255,255,51), sgl::Color(166,86,40), sgl::Color(247,129,191),
    };
    for (int i = 0; i < int(nodesList.size()); i++) {
        const auto& leaf = nodesList.at(i);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgBeginPath(vg);
        nvgCircle(vg, pointX, pointY, pointRadius);
        sgl::Color col = colors[std::hash<int>{}(int(leaf.parentIdx)) % 8];
        NVGcolor circleFillColorNvg = nvgRGBA(col.getR(), col.getG(), col.getB(), 255);
        nvgFillColor(vg, circleFillColorNvg);
        nvgFill(vg);
    }*/
    nvgBeginPath(vg);
    for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
        const auto& leaf = nodesList.at(leafIdx);
        int pointIdx = leafIdx - int(leafIdxOffset);
        if (pointIdx == selectedPointIndices[0] || pointIdx == selectedPointIndices[1]) {
            continue;
        }
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgCircle(vg, pointX, pointY, pointRadius);
    }
    NVGcolor circleFillColorNvg = nvgRGBA(
            circleFillColor.getR(), circleFillColor.getG(),
            circleFillColor.getB(), circleFillColor.getA());
    nvgFillColor(vg, circleFillColorNvg);
    nvgFill(vg);

    int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
    NVGcolor circleFillColorSelectedNvg = nvgRGBA(
            circleFillColorSelected0.getR(), circleFillColorSelected0.getG(),
            circleFillColorSelected0.getB(), circleFillColorSelected0.getA());
    for (int idx = 0; idx < numPointsSelected; idx++) {
        if (showSelectedRegionsByColor && idx == 1) {
            circleFillColorSelectedNvg = nvgRGBA(
                    circleFillColorSelected1.getR(), circleFillColorSelected1.getG(),
                    circleFillColorSelected1.getB(), circleFillColorSelected1.getA());
        }
        const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgBeginPath(vg);
        nvgCircle(vg, pointX, pointY, pointRadius * 1.5f);
        nvgFillColor(vg, circleFillColorSelectedNvg);
        nvgFill(vg);
    }

    if (showRing) {
        renderRings();
    }

    if (selectedPointIndices[0] >= 0) {
        drawSelectionArrows();
    } else if (!regionsEqual && showSelectedRegionsByColor && isFocusView && !fieldDataArray.empty()) {
        glm::vec2 center(windowWidth / 2.0f, windowHeight / 2.0f);
        HEBChartFieldData* fieldData = nullptr;
        if (limitedFieldIdx >= 0) {
            for (auto& fieldDataIt : fieldDataArray) {
                if (limitedFieldIdx == fieldDataIt->selectedFieldIdx) {
                    fieldData = fieldDataIt.get();
                    break;
                }
            }
        } else {
            fieldData = fieldDataArray.front().get();
        }
        float rhi = totalRadius + std::max(totalRadius * 0.015f, 4.0f);
        for (int i = 0; i < 2; i++) {
            float angle0 = i == 0 ? fieldData->a00 : fieldData->a10;
            float angle1 = i == 0 ? fieldData->a01 : fieldData->a11;
            bool isAnglePositive = angle1 - angle0 > 0.0f;
            const sgl::Color& circleFillColorSelected = getColorSelected(i);
            circleFillColorSelectedNvg = nvgRGBA(
                    circleFillColorSelected.getR(), circleFillColorSelected.getG(),
                    circleFillColorSelected.getB(), circleFillColorSelected.getA());
            nvgBeginPath(vg);
            nvgArc(vg, center.x, center.y, rhi, angle0, angle1, isAnglePositive ? NVG_CW : NVG_CCW);
            nvgStrokeWidth(vg, 3.0f);
            nvgStrokeColor(vg, circleFillColorSelectedNvg);
            nvgStroke(vg);
        }
    }


    // Render buttons.
    for (auto& button : buttons) {
        button.render(
                vg,
#ifdef SUPPORT_SKIA
                canvas, nullptr,
#endif
#ifdef SUPPORT_VKVG
                context,
#endif
                isDarkMode, s);
    }

    // Draw color legend.
    if (shallDrawColorLegend) {
        drawColorLegends();
    }
}

#ifdef SUPPORT_SKIA
void HEBChart::renderBaseSkia() {
    DiagramBase::renderBaseSkia();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    if (diagramMode == DiagramMode::CHORD) {
        renderChordDiagramSkia();
    } else {
        renderCorrelationMatrix();
    }
}

void HEBChart::renderChordDiagramSkia() {
#ifdef __APPLE__
    auto minMaxMi = getMinMaxCorrelationValue();
    float minMi = minMaxMi.first;
    float maxMi = minMaxMi.second;
#else
    auto [minMi, maxMi] = getMinMaxCorrelationValue();
#endif

    SkPaint paint;
    static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(&paint);

    // Draw the B-spline curves.
    sgl::Color curveStrokeColor = sgl::Color(
            100, 255, 100, uint8_t(std::clamp(int(std::ceil(curveOpacity * 255.0f)), 0, 255)));
    if (!curvePoints.empty()) {
        paint.setStroke(true);
        paint.setStrokeWidth(curveThickness * s);
        paint.setColor(toSkColor(curveStrokeColor));
        SkPath path;
        for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
            if (lineIdx == selectedLineIdx) {
                continue;
            }
            path.reset();
            glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            path.moveTo(pt0.x * s, pt0.y * s);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                path.lineTo(pt.x * s, pt.y * s);
            }

            if (colorByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(lineIdx)).get();
                curveStrokeColor = fieldData->evalColorMap(factor);
                paint.setColor(toSkColor(curveStrokeColor));
            } else if (opacityByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                }
                curveStrokeColor.setFloatA(curveOpacity * factor);
                paint.setColor(toSkColor(curveStrokeColor));
            }
            canvas->drawPath(path, paint);
        }

        if (selectedLineIdx >= 0) {
            // Background color outline.
            sgl::Color outlineColor = isDarkMode ? backgroundFillColorDark : backgroundFillColorBright;
            path.reset();
            glm::vec2 pt0 = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            path.moveTo(pt0.x * s, pt0.y * s);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                path.lineTo(pt.x * s, pt.y * s);
            }
            paint.setColor(toSkColor(outlineColor));
            paint.setStrokeWidth(curveThickness * 3.0f * s);
            canvas->drawPath(path, paint);

            // Line itself.
            path.reset();
            path.moveTo(pt0.x * s, pt0.y * s);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                path.lineTo(pt.x * s, pt.y * s);
            }
            if (colorByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(selectedLineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(selectedLineIdx)).get();
                curveStrokeColor = fieldData->evalColorMap(factor);
                paint.setColor(toSkColor(curveStrokeColor));
            }
            curveStrokeColor.setA(255);
            paint.setColor(toSkColor(curveStrokeColor));
            paint.setStrokeWidth(curveThickness * 2.0f * s);
            canvas->drawPath(path, paint);
        }
    }

    // Draw the point circles.
    float pointRadius = curveThickness * pointRadiusBase * s;
    paint.setColor(toSkColor(circleFillColor));
    paint.setStroke(false);
    /*for (int i = 0; i < int(nodesList.size()); i++) {
        const auto& leaf = nodesList.at(i);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        canvas->drawCircle(pointX * s, pointY * s, pointRadius, paint);
    }*/
    for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
        const auto& leaf = nodesList.at(leafIdx);
        int pointIdx = leafIdx - int(leafIdxOffset);
        if (pointIdx == selectedPointIndices[0] || pointIdx == selectedPointIndices[1]) {
            continue;
        }
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        canvas->drawCircle(pointX * s, pointY * s, pointRadius, paint);
    }

    int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
    for (int idx = 0; idx < numPointsSelected; idx++) {
        const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        paint.setColor(toSkColor(
                showSelectedRegionsByColor && idx == 1 ? circleFillColorSelected1 : circleFillColorSelected0));
        canvas->drawCircle(pointX * s, pointY * s, pointRadius * 1.5f, paint);
    }

    if (showRing) {
        renderRings();
    }

    if (selectedPointIndices[0] >= 0) {
        drawSelectionArrows();
    } if (!regionsEqual && showSelectedRegionsByColor && isFocusView && !fieldDataArray.empty()) {
        glm::vec2 center(windowWidth / 2.0f * s, windowHeight / 2.0f * s);
        HEBChartFieldData* fieldData = nullptr;
        if (limitedFieldIdx >= 0) {
            for (auto& fieldDataIt : fieldDataArray) {
                if (limitedFieldIdx == fieldDataIt->selectedFieldIdx) {
                    fieldData = fieldDataIt.get();
                    break;
                }
            }
        } else {
            fieldData = fieldDataArray.front().get();
        }
        float rhi = (totalRadius + std::max(totalRadius * 0.015f, 4.0f)) * s;
        SkPath path;
        paint.setStroke(true);
        paint.setStrokeWidth(3.0f * s);
        for (int i = 0; i < 2; i++) {
            float angle0Deg = (i == 0 ? fieldData->a00 : fieldData->a10) / sgl::PI * 180.0f;
            float angle1Deg = (i == 0 ? fieldData->a01 : fieldData->a11) / sgl::PI * 180.0f;
            const sgl::Color& circleFillColorSelected = getColorSelected(i);
            path.reset();
            path.arcTo(
                    SkRect{center.x - rhi, center.y - rhi, center.x + rhi, center.y + rhi},
                    angle0Deg, angle1Deg - angle0Deg, false);
            paint.setColor(toSkColor(circleFillColorSelected));
            canvas->drawPath(path, paint);
        }
    }

    // Render buttons.
    SkPaint defaultPaint;
    static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(&defaultPaint);
    for (auto& button : buttons) {
        button.render(
                vg,
                canvas, &defaultPaint,
#ifdef SUPPORT_VKVG
                context,
#endif
                isDarkMode, s);
    }

    // Draw color legend.
    if (shallDrawColorLegend) {
        drawColorLegends();
    }
}
#endif

#ifdef SUPPORT_VKVG
void HEBChart::renderBaseVkvg() {
    DiagramBase::renderBaseVkvg();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    if (diagramMode == DiagramMode::CHORD) {
        renderChordDiagramVkvg();
    } else {
        renderCorrelationMatrix();
    }
}

void HEBChart::renderChordDiagramVkvg() {
#ifdef __APPLE__
    auto minMaxMi = getMinMaxCorrelationValue();
    float minMi = minMaxMi.first;
    float maxMi = minMaxMi.second;
#else
    auto [minMi, maxMi] = getMinMaxCorrelationValue();
#endif

    // Draw the B-spline curves.
    sgl::Color curveStrokeColor = sgl::Color(100, 255, 100, 255);
    if (!curvePoints.empty()) {
        vkvg_set_line_width(context, curveThickness * s);
        vkvg_set_source_color(context, curveStrokeColor.getColorRGBA());

        if (colorByValue || opacityByValue) {
            for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
                if (lineIdx == selectedLineIdx) {
                    continue;
                }
                if (colorByValue) {
                    float factor = 1.0f;
                    if (correlationValuesArray.size() > 1) {
                        factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi);
                    }
                    HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(lineIdx)).get();
                    auto color = fieldData->evalColorMap(factor);
                    vkvg_set_source_color(context, color.getColorRGB());
                    vkvg_set_opacity(context, color.getFloatA());
                } else if (opacityByValue) {
                    float factor = 1.0f;
                    if (correlationValuesArray.size() > 1) {
                        factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                    }
                    vkvg_set_opacity(context, curveOpacity * factor);
                }

                glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
                pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
                pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
                vkvg_move_to(context, pt0.x * s, pt0.y * s);
                for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                    glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                    pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                    pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                    vkvg_line_to(context, pt.x * s, pt.y * s);
                }

                vkvg_stroke(context);
            }
        } else {
            vkvg_set_opacity(context, curveOpacity);
            for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
                if (lineIdx == selectedLineIdx) {
                    continue;
                }
                glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
                pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
                pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
                vkvg_move_to(context, pt0.x * s, pt0.y * s);
                for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                    glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                    pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                    pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                    vkvg_line_to(context, pt.x * s, pt.y * s);
                }
            }
            vkvg_stroke(context);
        }

        if (selectedLineIdx >= 0) {
            // Background color outline.
            sgl::Color outlineColor = isDarkMode ? backgroundFillColorDark : backgroundFillColorBright;
            vkvg_set_line_width(context, curveThickness * 3.0f * s);
            vkvg_set_source_color(context, outlineColor.getColorRGB());
            vkvg_set_opacity(context, 1.0f);
            glm::vec2 pt0 = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            vkvg_move_to(context, pt0.x * s, pt0.y * s);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                vkvg_line_to(context, pt.x * s, pt.y * s);
            }
            vkvg_stroke(context);

            // Line itself.
            vkvg_set_line_width(context, curveThickness * 2.0f * s);
            if (colorByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(selectedLineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(selectedLineIdx)).get();
                auto color = fieldData->evalColorMap(factor);
                vkvg_set_source_color(context, color.getColorRGB());
            }
            vkvg_move_to(context, pt0.x * s, pt0.y * s);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(selectedLineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                vkvg_line_to(context, pt.x * s, pt.y * s);
            }
            vkvg_stroke(context);
        }
    }
    vkvg_set_opacity(context, 1.0f);

    // Draw the point circles.
    float pointRadius = curveThickness * pointRadiusBase * s;
    /*for (int i = 0; i < int(nodesList.size()); i++) {
        const auto& leaf = nodesList.at(i);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        vkvg_arc(context, pointX * s, pointY * s, pointRadius, 0.0f, sgl::TWO_PI);
    }*/
    for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
        const auto& leaf = nodesList.at(leafIdx);
        int pointIdx = leafIdx - int(leafIdxOffset);
        if (pointIdx == selectedPointIndices[0] || pointIdx == selectedPointIndices[1]) {
            continue;
        }
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        vkvg_arc(context, pointX * s, pointY * s, pointRadius, 0.0f, sgl::TWO_PI);
    }
    vkvg_set_source_color(context, circleFillColor.getColorRGBA());
    vkvg_fill(context);

    int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
    for (int idx = 0; idx < numPointsSelected; idx++) {
        const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        vkvg_ellipse(context, pointRadius * 1.5f, pointRadius * 1.5f, pointX * s, pointY * s, 0.0f);
        vkvg_set_source_color(
                context, (showSelectedRegionsByColor && idx == 1
                          ? circleFillColorSelected1 : circleFillColorSelected0).getColorRGBA());
        vkvg_fill(context);
    }

    if (showRing) {
        renderRings();
    }

    if (selectedPointIndices[0] >= 0) {
        drawSelectionArrows();
    } else if (!regionsEqual && showSelectedRegionsByColor && isFocusView && !fieldDataArray.empty()) {
        glm::vec2 center(windowWidth / 2.0f * s, windowHeight / 2.0f * s);
        HEBChartFieldData* fieldData = nullptr;
        if (limitedFieldIdx >= 0) {
            for (auto& fieldDataIt : fieldDataArray) {
                if (limitedFieldIdx == fieldDataIt->selectedFieldIdx) {
                    fieldData = fieldDataIt.get();
                    break;
                }
            }
        } else {
            fieldData = fieldDataArray.front().get();
        }
        float rhi = (totalRadius + std::max(totalRadius * 0.015f, 4.0f)) * s;
        for (int i = 0; i < 2; i++) {
            float angle0 = i == 0 ? fieldData->a00 : fieldData->a10;
            float angle1 = i == 0 ? fieldData->a01 : fieldData->a11;
            bool isAnglePositive = angle1 - angle0 > 0.0f;
            const sgl::Color& circleFillColorSelected = getColorSelected(i);
            if (isAnglePositive) {
                vkvg_arc(context, center.x, center.y, rhi, angle0, angle1);
            } else {
                vkvg_arc_negative(context, center.x, center.y, rhi, angle0, angle1);
            }
            vkvg_set_line_width(context, 3.0f * s);
            vkvg_set_source_color(context, circleFillColorSelected.getColorRGBA());
            vkvg_stroke(context);
        }
    }

    // Render buttons.
    for (auto& button : buttons) {
        button.render(
                vg,
#ifdef SUPPORT_SKIA
                canvas, nullptr,
#endif
                context,
                isDarkMode, s);
    }

    // Draw color legend.
    if (shallDrawColorLegend) {
        drawColorLegends();
    }
}
#endif

void HEBChart::renderRings() {
#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr, *gradientPaint = nullptr;
    SkPath* path = nullptr;
    if (canvas) {
        paint = new SkPaint;
        gradientPaint = new SkPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(paint);
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(gradientPaint);
        path = new SkPath;
    }
#endif

    glm::vec2 center(windowWidth / 2.0f, windowHeight / 2.0f);
    auto numFields = int(fieldDataArray.size());
    auto numFieldsReal = 0;
    for (int i = 0; i < numFields; i++) {
        numFieldsReal += fieldDataArray.at(i)->useTwoFields ? 2 : 1;
    }
    int limitedFieldDataIdx = -1;
    if (limitedFieldIdx >= 0) {
        for (int i = 0; i < numFields; i++) {
            if (fieldDataArray.at(i)->selectedFieldIdx == limitedFieldIdx) {
                limitedFieldDataIdx = i;
                break;
            }
        }
    }
    auto numFieldsSize =
            limitedFieldIdx >= 0 && limitedFieldDataIdx >= 0
            ? std::min(numFieldsReal, fieldDataArray.at(limitedFieldDataIdx)->useTwoFields ? 2 : 1)
            : numFieldsReal;
    int i = 0;
    bool secondField = false;
    for (int ix = 0; ix < numFieldsReal; ix++) {
        auto* fieldData = fieldDataArray.at(i).get();
        std::vector<float>* leafStdDevArray = nullptr;
        if (!fieldData->useTwoFields || !secondField) {
            leafStdDevArray = &fieldData->leafStdDevArray;
        } else {
            leafStdDevArray = &fieldData->leafStdDevArray2;
        }
        int ip = ix;
        if (limitedFieldIdx >= 0) {
            if (limitedFieldIdx != fieldData->selectedFieldIdx) {
                i++;
                secondField = false;
                continue;
            } else {
                ip = secondField ? 1 : 0;
            }
        }
        std::pair<float, float> stdDevRange;
        if (useGlobalStdDevRange) {
            if (!fieldData->useTwoFields || !secondField) {
                stdDevRange = getGlobalStdDevRange(fieldData->selectedFieldIdx, 0);
            } else {
                stdDevRange = getGlobalStdDevRange(fieldData->selectedFieldIdx, 1);
            }
        } else {
            stdDevRange = std::make_pair(fieldData->minStdDev, fieldData->maxStdDev);
        }
        float pctLower = float(ip) / float(numFieldsSize);
        float pctMiddle = (float(ip) + 0.5f) / float(numFieldsSize);
        float pctUpper = float(ip + 1) / float(numFieldsSize);
        float rlo = std::max(chartRadius + outerRingOffset + pctLower * outerRingWidth, 1e-6f);
        float rmi = std::max(chartRadius + outerRingOffset + pctMiddle * outerRingWidth, 1e-6f);
        float rhi = std::max(chartRadius + outerRingOffset + pctUpper * outerRingWidth, 1e-6f);
        bool isSaturated =
                (separateColorVarianceAndCorrelation && getIsGrayscaleColorMap(colorMapVariance))
                || !separateColorVarianceAndCorrelation || selectedLineIdx < 0
                || lineFieldIndexArray.at(selectedLineIdx) == i;

        for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
            bool isSingleElementRegion = false;
            if (!regionsEqual) {
                if (leafIdx == int(leafIdxOffset1) - 1) {
                    if (leafIdxOffset + 1 == leafIdxOffset1) {
                        isSingleElementRegion = true;
                    } else {
                        continue;
                    }
                }
                if (leafIdx == int(nodesList.size()) - 1) {
                    if (int(leafIdxOffset1) + 1 == int(nodesList.size())) {
                        isSingleElementRegion = true;
                    } else {
                        continue;
                    }
                }
            }
            //if (!regionsEqual && (leafIdx == int(leafIdxOffset1) - 1 || leafIdx == int(nodesList.size()) - 1)) {
            //    continue;
            //}
            int numLeaves = int(nodesList.size()) - int(leafIdxOffset);
            int nextIdx = (leafIdx + 1 - int(leafIdxOffset)) % numLeaves + int(leafIdxOffset);
            if (isSingleElementRegion) {
                nextIdx = leafIdx;
            }
            const auto &leafCurr = nodesList.at(leafIdx);
            const auto &leafNext = nodesList.at(nextIdx);
            bool isAnglePositive = regionsEqual || leafNext.angle - leafCurr.angle > 0.0f;
            float deltaAngleSign = isAnglePositive ? 1.0f : -1.0f;
            float angle0 = leafCurr.angle;
            float angle1 = leafNext.angle + deltaAngleSign * 0.005f;
            if (isSingleElementRegion) {
                const float angleRangeHalf = sgl::PI * 0.92f;
                angle0 = leafCurr.angle - 0.5f * angleRangeHalf;
                angle1 = leafCurr.angle + 0.5f * angleRangeHalf;
            }
            float angleMid0 = angle0;
            float angleMid1 = angle1;
            bool isStartSegment = false;
            bool isEndSegment = false;
            if (!regionsEqual && (leafIdx == int(leafIdxOffset) || leafIdx == int(leafIdxOffset1))) {
                float deltaAngle = angle1 - angle0;
                if (std::abs(deltaAngle) > 0.1f) {
                    deltaAngle = deltaAngleSign * 0.1f;
                }
                angle0 -= deltaAngle * 0.5f;
                if (leafIdx == int(leafIdxOffset)) {
                    fieldData->a00 = angle0;
                } else {
                    fieldData->a10 = angle0;
                }
                isStartSegment = true;
            }
            if (!regionsEqual && (nextIdx == int(leafIdxOffset1) - 1 || nextIdx == int(nodesList.size()) - 1)) {
                float deltaAngle = angle1 - angle0;
                if (std::abs(deltaAngle) > 0.1f) {
                    deltaAngle = deltaAngleSign * 0.1f;
                }
                angle1 += deltaAngle * 0.5f;
                if (nextIdx == int(leafIdxOffset1) - 1) {
                    fieldData->a01 = angle1;
                } else {
                    fieldData->a11 = angle1;
                }
                isEndSegment = true;
            }
            float cos0 = std::cos(angle0), sin0 = std::sin(angle0);
            float cos1 = std::cos(angle1), sin1 = std::sin(angle1);
            float cosMid0 = std::cos(angleMid0), sinMid0 = std::sin(angleMid0);
            float cosMid1 = std::cos(angleMid1), sinMid1 = std::sin(angleMid1);
            //glm::vec2 lo0 = center + rlo * glm::vec2(cos0, sin0);
            glm::vec2 lo1 = center + rlo * glm::vec2(cos1, sin1);
            glm::vec2 hi0 = center + rhi * glm::vec2(cos0, sin0);
            //glm::vec2 hi1 = center + rhi * glm::vec2(cos1, sin1);
            glm::vec2 mi0 = center + rmi * glm::vec2(cosMid0, sinMid0);
            glm::vec2 mi1 = center + rmi * glm::vec2(cosMid1, sinMid1);

            float stdev0 = leafStdDevArray->at(leafIdx - int(leafIdxOffset));
            float t0 = stdDevRange.first == stdDevRange.second ? 0.0f : (stdev0 - stdDevRange.first) / (stdDevRange.second - stdDevRange.first);
            float stdev1 = leafStdDevArray->at(nextIdx - int(leafIdxOffset));
            float t1 = stdDevRange.first == stdDevRange.second ? 0.0f : (stdev1 - stdDevRange.first) / (stdDevRange.second - stdDevRange.first);

            if (vg) {
                glm::vec4 rgbColor0 = fieldData->evalColorMapVec4Variance(t0, isSaturated);
                rgbColor0.w = 1.0f;
                glm::vec4 rgbColor1 = fieldData->evalColorMapVec4Variance(t1, isSaturated);
                rgbColor1.w = 1.0f;
                NVGcolor fillColor0 = nvgRGBAf(rgbColor0.x, rgbColor0.y, rgbColor0.z, rgbColor0.w);
                NVGcolor fillColor1 = nvgRGBAf(rgbColor1.x, rgbColor1.y, rgbColor1.z, rgbColor1.w);

                nvgBeginPath(vg);
                nvgArc(vg, center.x, center.y, rlo, angle1, angle0, isAnglePositive ? NVG_CCW : NVG_CW);
                if (isStartSegment) {
                    float angleMid = angle0 + arrowAngleRad * deltaAngleSign;
                    nvgLineTo(vg, center.x + rmi * std::cos(angleMid), center.y + rmi * std::sin(angleMid));
                }
                nvgLineTo(vg, hi0.x, hi0.y);
                nvgArc(vg, center.x, center.y, rhi, angle0, angle1, isAnglePositive ? NVG_CW : NVG_CCW);
                if (isEndSegment) {
                    float angleMid = angle1 + arrowAngleRad * deltaAngleSign;
                    nvgLineTo(vg, center.x + rmi * std::cos(angleMid), center.y + rmi * std::sin(angleMid));
                }
                nvgLineTo(vg, lo1.x, lo1.y);
                nvgClosePath(vg);

                NVGpaint paint = nvgLinearGradient(vg, mi0.x, mi0.y, mi1.x, mi1.y, fillColor0, fillColor1);
                nvgFillPaint(vg, paint);
                nvgFill(vg);
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                float angle0Deg = angle0 / sgl::PI * 180.0f;
                float angle1Deg = angle1 / sgl::PI * 180.0f + (nextIdx < leafIdx ? 360.0f : 0.0f);
                sgl::Color rgbColor0 = fieldData->evalColorMapVariance(t0, isSaturated);
                rgbColor0.setA(255);
                sgl::Color rgbColor1 = fieldData->evalColorMapVariance(t1, isSaturated);
                rgbColor1.setA(255);

                SkPoint linearPoints[2] = { { mi0.x * s, mi0.y * s }, { mi1.x * s, mi1.y * s } };
                SkColor linearColors[2] = { toSkColor(rgbColor0), toSkColor(rgbColor1) };
                gradientPaint->setShader(SkGradientShader::MakeLinear(
                        linearPoints, linearColors, nullptr, 2, SkTileMode::kClamp));

                path->reset();
                path->addArc(
                        SkRect{(center.x - rlo) * s, (center.y - rlo) * s, (center.x + rlo) * s, (center.y + rlo) * s},
                        angle1Deg, angle0Deg - angle1Deg);
                if (isStartSegment) {
                    float angleMid = angle0 + arrowAngleRad * deltaAngleSign;
                    path->lineTo((center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                path->lineTo(hi0.x * s, hi0.y * s);
                path->arcTo(
                        SkRect{(center.x - rhi) * s, (center.y - rhi) * s, (center.x + rhi) * s, (center.y + rhi) * s},
                        angle0Deg, angle1Deg - angle0Deg, false);
                if (isEndSegment) {
                    float angleMid = angle1 + arrowAngleRad * deltaAngleSign;
                    path->lineTo((center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                //path.lineTo(lo1.x * s, lo1.y * s);
                path->close();

                canvas->drawPath(*path, *gradientPaint);
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                glm::vec4 rgbColor0 = fieldData->evalColorMapVec4Variance(t0, isSaturated);
                glm::vec4 rgbColor1 = fieldData->evalColorMapVec4Variance(t1, isSaturated);

                if (isAnglePositive) {
                    vkvg_arc_negative(context, center.x * s, center.y * s, rlo * s, angle1, angle0);
                } else {
                    vkvg_arc(context, center.x * s, center.y * s, rlo * s, angle1, angle0);
                }
                if (isStartSegment) {
                    float angleMid = angle0 + arrowAngleRad * deltaAngleSign;
                    vkvg_line_to(context, (center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                vkvg_line_to(context, hi0.x * s, hi0.y * s);
                if (isAnglePositive) {
                    vkvg_arc(context, center.x * s, center.y * s, rhi * s, angle0, angle1);
                } else {
                    vkvg_arc_negative(context, center.x * s, center.y * s, rhi * s, angle0, angle1);
                }
                if (isEndSegment) {
                    float angleMid = angle1 + arrowAngleRad * deltaAngleSign;
                    vkvg_line_to(context, (center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                vkvg_line_to(context, lo1.x * s, lo1.y * s);

                auto pattern = vkvg_pattern_create_linear(mi0.x * s, mi0.y * s, mi1.x * s, mi1.y * s);
                vkvg_pattern_add_color_stop(pattern, 0.0f, rgbColor0.x, rgbColor0.y, rgbColor0.z, 1.0f);
                vkvg_pattern_add_color_stop(pattern, 1.0f, rgbColor1.x, rgbColor1.y, rgbColor1.z, 1.0f);
                vkvg_set_source(context, pattern);
                vkvg_pattern_destroy(pattern);
                vkvg_fill(context);
            }
#endif
        }

        if (fieldData->useTwoFields) {
            if (secondField) {
                i++;
                secondField = false;
            } else {
                secondField = true;
            }
        } else {
            i++;
        }
    }

    int selectedLineFieldPos = 0;
    if (selectedLineIdx >= 0) {
        int selectedLineFieldIdx = lineFieldIndexArray.at(selectedLineIdx);
        for (int j = 0; j < numFields; j++) {
            if (j == selectedLineFieldIdx) {
                break;
            }
            selectedLineFieldPos += fieldDataArray.at(selectedLineFieldIdx)->useTwoFields ? 2 : 1;
        }
    }

    sgl::Color circleStrokeColor = isDarkMode ? circleStrokeColorDark : circleStrokeColorBright;
    sgl::Color currentCircleColor;
    i = 0;
    secondField = false;
    for (int ix = 0; ix < numFieldsReal; ix++) {
        currentCircleColor = circleStrokeColor;
        int fieldIdx = i;
        int ip = ix;
        bool isSelectedRing = false;
        if (separateColorVarianceAndCorrelation && getIsGrayscaleColorMap(colorMapVariance) && numFieldsSize > 1
                && selectedLineIdx >= 0 && limitedFieldIdx < 0) {
            int selectedLineFieldIdx = lineFieldIndexArray.at(selectedLineIdx);
            if (i == numFields - 1) {
                fieldIdx = selectedLineFieldIdx;
                ip = selectedLineFieldPos + (secondField ? 1 : 0);
                currentCircleColor = ringStrokeColorSelected;
                isSelectedRing = true;
            } else if (i >= selectedLineFieldIdx) {
                fieldIdx++;
                ip += fieldDataArray.at(selectedLineFieldIdx)->useTwoFields ? 2 : 1;
            }
        }
        auto* fieldData = fieldDataArray.at(fieldIdx).get();
        if (limitedFieldIdx >= 0) {
            if (limitedFieldIdx != fieldData->selectedFieldIdx) {
                i++;
                secondField = false;
                continue;
            } else {
                ip = 0;
            }
        }
        float pctLower = float(ip) / float(numFieldsSize);
        float pctMiddle = (float(ip) + 0.5f) / float(numFieldsSize);
        float pctUpper = float(ip + 1) / float(numFieldsSize);
        float rlo = std::max(chartRadius + outerRingOffset + pctLower * outerRingWidth, 1e-6f);
        float rmi = std::max(chartRadius + outerRingOffset + pctMiddle * outerRingWidth, 1e-6f);
        float rhi = std::max(chartRadius + outerRingOffset + pctUpper * outerRingWidth, 1e-6f);

        if (vg) {
            nvgBeginPath(vg);
            if (regionsEqual) {
                nvgCircle(vg, center.x, center.y, rlo);
                if (ip == numFieldsSize - 1 || isSelectedRing || limitedFieldIdx >= 0) {
                    nvgCircle(vg, center.x, center.y, rhi);
                }
            } else {
                bool isAnglePos0 = fieldData->a01 - fieldData->a00 > 0.0f;
                nvgArc(vg, center.x, center.y, rlo, fieldData->a00, fieldData->a01, isAnglePos0 ? NVG_CW : NVG_CCW);
                if (useRingArrows) {
                    float angleMid = fieldData->a01 + arrowAngleRad * (isAnglePos0 ? 1.0f : -1.0f);
                    nvgLineTo(vg, center.x + rmi * std::cos(angleMid), center.y + rmi * std::sin(angleMid));
                }
                nvgLineTo(vg, center.x + rhi * std::cos(fieldData->a01), center.y + rhi * std::sin(fieldData->a01));
                nvgArc(vg, center.x, center.y, rhi, fieldData->a01, fieldData->a00, isAnglePos0 ? NVG_CCW : NVG_CW);
                if (useRingArrows) {
                    float angleMid = fieldData->a00 + arrowAngleRad * (isAnglePos0 ? 1.0f : -1.0f);
                    nvgLineTo(vg, center.x + rmi * std::cos(angleMid), center.y + rmi * std::sin(angleMid));
                }
                nvgLineTo(vg, center.x + rlo * std::cos(fieldData->a00), center.y + rlo * std::sin(fieldData->a00));

                bool isAnglePos1 = fieldData->a11 - fieldData->a10 > 0.0f;
                nvgMoveTo(vg, center.x + rlo * std::cos(fieldData->a10), center.y + rlo * std::sin(fieldData->a10));
                nvgArc(vg, center.x, center.y, rlo, fieldData->a10, fieldData->a11, isAnglePos1 ? NVG_CW : NVG_CCW);
                if (useRingArrows) {
                    float angleMid = fieldData->a11 + arrowAngleRad * (isAnglePos1 ? 1.0f : -1.0f);
                    nvgLineTo(vg, center.x + rmi * std::cos(angleMid), center.y + rmi * std::sin(angleMid));
                }
                nvgLineTo(vg, center.x + rhi * std::cos(fieldData->a11), center.y + rhi * std::sin(fieldData->a11));
                nvgArc(vg, center.x, center.y, rhi, fieldData->a11, fieldData->a10, isAnglePos1 ? NVG_CCW : NVG_CW);
                if (useRingArrows) {
                    float angleMid = fieldData->a10 + arrowAngleRad * (isAnglePos1 ? 1.0f : -1.0f);
                    nvgLineTo(vg, center.x + rmi * std::cos(angleMid), center.y + rmi * std::sin(angleMid));
                }
                nvgLineTo(vg, center.x + rlo * std::cos(fieldData->a10), center.y + rlo * std::sin(fieldData->a10));
            }
            nvgStrokeWidth(vg, 1.0f);
            NVGcolor currentCircleColorNvg = nvgRGBA(
                    currentCircleColor.getR(), currentCircleColor.getG(),
                    currentCircleColor.getB(), currentCircleColor.getA());
            nvgStrokeColor(vg, currentCircleColorNvg);
            nvgStroke(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            paint->setStroke(true);
            paint->setStrokeWidth(1.0f * s);
            paint->setColor(toSkColor(currentCircleColor));
            if (regionsEqual) {
                canvas->drawCircle(center.x * s, center.y * s, rlo * s, *paint);
                if (ip == numFieldsSize - 1 || isSelectedRing || limitedFieldIdx >= 0) {
                    canvas->drawCircle(center.x * s, center.y * s, rhi * s, *paint);
                }
            } else {
                float a00Deg = fieldData->a00 / sgl::PI * 180.0f;
                float a01Deg = fieldData->a01 / sgl::PI * 180.0f;
                float a10Deg = fieldData->a10 / sgl::PI * 180.0f;
                float a11Deg = fieldData->a11 / sgl::PI * 180.0f;
                bool isAnglePos0 = fieldData->a01 - fieldData->a00 > 0.0f;
                bool isAnglePos1 = fieldData->a11 - fieldData->a10 > 0.0f;
                path->reset();
                path->arcTo(
                        SkRect{(center.x - rlo) * s, (center.y - rlo) * s, (center.x + rlo) * s, (center.y + rlo) * s},
                        a00Deg, a01Deg - a00Deg, false);
                if (useRingArrows) {
                    float angleMid = fieldData->a01 + arrowAngleRad * (isAnglePos0 ? 1.0f : -1.0f);
                    path->lineTo((center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                path->lineTo((center.x + rhi * std::cos(fieldData->a01)) * s, (center.y + rhi * std::sin(fieldData->a01)) * s);
                path->arcTo(
                        SkRect{(center.x - rhi) * s, (center.y - rhi) * s, (center.x + rhi) * s, (center.y + rhi) * s},
                        a01Deg, a00Deg - a01Deg, false);
                if (useRingArrows) {
                    float angleMid = fieldData->a00 + arrowAngleRad * (isAnglePos0 ? 1.0f : -1.0f);
                    path->lineTo((center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                path->close();
                canvas->drawPath(*path, *paint);

                path->reset();
                path->arcTo(
                        SkRect{(center.x - rlo) * s, (center.y - rlo) * s, (center.x + rlo) * s, (center.y + rlo) * s},
                        a10Deg, a11Deg - a10Deg, false);
                if (useRingArrows) {
                    float angleMid = fieldData->a11 + arrowAngleRad * (isAnglePos1 ? 1.0f : -1.0f);
                    path->lineTo((center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                path->lineTo((center.x + rhi * std::cos(fieldData->a11)) * s, (center.y + rhi * std::sin(fieldData->a11)) * s);
                path->arcTo(
                        SkRect{(center.x - rhi) * s, (center.y - rhi) * s, (center.x + rhi) * s, (center.y + rhi) * s},
                        a11Deg, a10Deg - a11Deg, false);
                if (useRingArrows) {
                    float angleMid = fieldData->a10 + arrowAngleRad * (isAnglePos1 ? 1.0f : -1.0f);
                    path->lineTo((center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                path->close();
                canvas->drawPath(*path, *paint);
            }
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            if (regionsEqual) {
                vkvg_arc(context, center.x * s, center.y * s, rlo * s, 0.0f, sgl::TWO_PI);
                if (ip == numFieldsSize - 1 || isSelectedRing || limitedFieldIdx >= 0) {
                    vkvg_arc(context, center.x * s, center.y * s, rhi * s, 0.0f, sgl::TWO_PI);
                }
            } else {
                bool isAnglePos0 = fieldData->a01 - fieldData->a00 > 0.0f;
                if (isAnglePos0) {
                    vkvg_arc(context, center.x * s, center.y * s, rlo * s, fieldData->a00, fieldData->a01);
                } else {
                    vkvg_arc_negative(context, center.x * s, center.y * s, rlo * s, fieldData->a00, fieldData->a01);
                }
                if (useRingArrows) {
                    float angleMid = fieldData->a01 + arrowAngleRad * (isAnglePos0 ? 1.0f : -1.0f);
                    vkvg_line_to(context, (center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                vkvg_line_to(context, (center.x + rhi * std::cos(fieldData->a01)) * s, (center.y + rhi * std::sin(fieldData->a01)) * s);
                if (isAnglePos0) {
                    vkvg_arc_negative(context, center.x * s, center.y * s, rhi * s, fieldData->a01, fieldData->a00);
                } else {
                    vkvg_arc(context, center.x * s, center.y * s, rhi * s, fieldData->a01, fieldData->a00);
                }
                if (useRingArrows) {
                    float angleMid = fieldData->a00 + arrowAngleRad * (isAnglePos0 ? 1.0f : -1.0f);
                    vkvg_line_to(context, (center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                vkvg_line_to(context, (center.x + rlo * std::cos(fieldData->a00)) * s, (center.y + rlo * std::sin(fieldData->a00)) * s);

                bool isAnglePos1 = fieldData->a11 - fieldData->a10 > 0.0f;
                vkvg_move_to(context, (center.x + rlo * std::cos(fieldData->a10)) * s, (center.y + rlo * std::sin(fieldData->a10)) * s);
                if (isAnglePos1) {
                    vkvg_arc(context, center.x * s, center.y * s, rlo * s, fieldData->a10, fieldData->a11);
                } else {
                    vkvg_arc_negative(context, center.x * s, center.y * s, rlo * s, fieldData->a10, fieldData->a11);
                }
                if (useRingArrows) {
                    float angleMid = fieldData->a11 + arrowAngleRad * (isAnglePos1 ? 1.0f : -1.0f);
                    vkvg_line_to(context, (center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                vkvg_line_to(context, (center.x + rhi * std::cos(fieldData->a11)) * s, (center.y + rhi * std::sin(fieldData->a11)) * s);
                if (isAnglePos1) {
                    vkvg_arc_negative(context, center.x * s, center.y * s, rhi * s, fieldData->a11, fieldData->a10);
                } else {
                    vkvg_arc(context, center.x * s, center.y * s, rhi * s, fieldData->a11, fieldData->a10);
                }
                if (useRingArrows) {
                    float angleMid = fieldData->a10 + arrowAngleRad * (isAnglePos1 ? 1.0f : -1.0f);
                    vkvg_line_to(context, (center.x + rmi * std::cos(angleMid)) * s, (center.y + rmi * std::sin(angleMid)) * s);
                }
                vkvg_line_to(context, (center.x + rlo * std::cos(fieldData->a10)) * s, (center.y + rlo * std::sin(fieldData->a10)) * s);
            }
            vkvg_set_line_width(context, 1.0f * s);
            vkvg_set_source_color(context, currentCircleColor.getColorRGBA());
            vkvg_stroke(context);
        }
#endif

        if (fieldData->useTwoFields) {
            if (secondField) {
                i++;
                secondField = false;
            } else {
                secondField = true;
            }
        } else {
            i++;
        }
    }

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
        delete gradientPaint;
        delete path;
    }
#endif
}

void HEBChart::drawSelectionArrows() {
#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr;
    if (canvas) {
        paint = new SkPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(paint);
    }
#endif

    if (diagramMode == DiagramMode::CHORD) {
        // Draw wedges/arrows pointing at the selected points.
        int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
        for (int idx = 0; idx < numPointsSelected; idx++) {
            auto fillColor = showSelectedRegionsByColor && idx == 1 ? circleFillColorSelected1 : circleFillColorSelected0;
            const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);

            glm::vec2 center(windowWidth / 2.0f, windowHeight / 2.0f);
            glm::vec2 dir(leaf.normalizedPosition.x, leaf.normalizedPosition.y);
            glm::vec2 orthoDir(-dir.y, dir.x);
            float pointRadius = curveThickness * pointRadiusBase * borderSizeX / 10.0f;
            float radius0 = totalRadius;
            float radius1 = pointRadius * 4.0f;
            float width = radius1;
            glm::vec2 p0 = center + radius0 * dir;
            glm::vec2 p1 = p0 + radius1 * dir - width * orthoDir;
            glm::vec2 p2 = p0 + radius1 * dir + width * orthoDir;

            if (vg) {
                NVGcolor fillColorNvg = nvgRGBA(fillColor.getR(), fillColor.getG(), fillColor.getB(), 255);
                nvgBeginPath(vg);
                nvgMoveTo(vg, p0.x, p0.y);
                nvgLineTo(vg, p1.x, p1.y);
                nvgLineTo(vg, p2.x, p2.y);
                nvgClosePath(vg);
                nvgFillColor(vg, fillColorNvg);
                nvgFill(vg);
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                SkPath path;
                path.moveTo(p0.x * s, p0.y * s);
                path.lineTo(p1.x * s, p1.y * s);
                path.lineTo(p2.x * s, p2.y * s);
                path.close();
                paint->setColor(toSkColor(fillColor));
                canvas->drawPath(path, *paint);
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                vkvg_move_to(context, p0.x * s, p0.y * s);
                vkvg_line_to(context, p1.x * s, p1.y * s);
                vkvg_line_to(context, p2.x * s, p2.y * s);
                vkvg_close_path(context);
                vkvg_set_source_color(context, fillColor.getColorRGBA());
                vkvg_fill(context);
            }
#endif
        }
    } else {
        // Draw wedges/arrows pointing at the selected points.
        float matrixWidth = 2.0f * chartRadius;
        float matrixHeight = 2.0f * chartRadius;
        int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
        for (int idx = 0; idx < numPointsSelected; idx++) {
            auto fillColor = showSelectedRegionsByColor && idx == 1 ? circleFillColorSelected1 : circleFillColorSelected0;
            //const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);

            auto lower = int(leafIdxOffset);
            auto upper = int(nodesList.size());
            if (!regionsEqual) {
                if (idx == 0) {
                    upper = int(leafIdxOffset1);
                } else {
                    lower = int(leafIdxOffset1);
                }
            }
            glm::vec2 dir, orthoDir;
            if (idx == 0) {
                dir = glm::vec2(0.0f, -1.0f);
                orthoDir = glm::vec2(1.0f, 0.0f);
            } else {
                dir = glm::vec2(-1.0f, 0.0f);
                orthoDir = glm::vec2(0.0f, 1.0f);
            }

            int offset = regionsEqual || idx == 0 ? 0 : int(leafIdxOffset1 - leafIdxOffset);
            float pct = (float(selectedPointIndices[idx] - offset) + 0.5f) / float(upper - lower);
            glm::vec2 upperLeft(
                    windowWidth / 2.0f - (idx == 0 ? chartRadius : totalRadius),
                    windowHeight / 2.0f - (idx == 0 ? totalRadius : chartRadius));

            float pointRadius = curveThickness * pointRadiusBase * borderSizeX / 10.0f;
            float radius1 = pointRadius * 4.0f;
            float width = radius1;
            // + totalRadius * dir for top/bottom
            glm::vec2 p0 = upperLeft + pct * orthoDir * (idx == 0 ? matrixWidth : matrixHeight);
            glm::vec2 p1 = p0 + radius1 * dir - width * orthoDir;
            glm::vec2 p2 = p0 + radius1 * dir + width * orthoDir;

            if (vg) {
                NVGcolor fillColorNvg = nvgRGBA(fillColor.getR(), fillColor.getG(), fillColor.getB(), 255);
                nvgBeginPath(vg);
                nvgMoveTo(vg, p0.x, p0.y);
                nvgLineTo(vg, p1.x, p1.y);
                nvgLineTo(vg, p2.x, p2.y);
                nvgClosePath(vg);
                nvgFillColor(vg, fillColorNvg);
                nvgFill(vg);
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                SkPath path;
                path.moveTo(p0.x * s, p0.y * s);
                path.lineTo(p1.x * s, p1.y * s);
                path.lineTo(p2.x * s, p2.y * s);
                path.close();
                paint->setColor(toSkColor(fillColor));
                canvas->drawPath(path, *paint);
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                vkvg_move_to(context, p0.x * s, p0.y * s);
                vkvg_line_to(context, p1.x * s, p1.y * s);
                vkvg_line_to(context, p2.x * s, p2.y * s);
                vkvg_close_path(context);
                vkvg_set_source_color(context, fillColor.getColorRGBA());
                vkvg_fill(context);
            }
#endif
        }
    }

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
    }
#endif
}

void HEBChart::drawColorLegends() {
    const auto& fieldDataArrayLocal = fieldDataArray;
    auto numFields = int(fieldDataArray.size());
    auto numFieldsSize = limitedFieldIdx >= 0 ? std::min(int(fieldDataArray.size()), 1) : int(fieldDataArray.size());
    if (numFieldsSize == 0) {
        return;
    }
    bool useColorMapVariance = separateColorVarianceAndCorrelation && outerRingSizePct > 0.0f;
    if (useColorMapVariance) {
        numFieldsSize++;
    }

#ifdef __APPLE__
    auto minMaxMi = getMinMaxCorrelationValue();
    float minMi = minMaxMi.first;
    float maxMi = minMaxMi.second;
#else
    auto [minMi, maxMi] = getMinMaxCorrelationValue();
#endif
    for (int i = 0; i < numFieldsSize; i++) {
        int ix = numFieldsSize - i;
        HEBChartFieldData* fieldData = nullptr;
        std::string variableName;
        std::function<glm::vec4(float)> colorMap;
        std::function<std::string(float)> labelMap;
        bool isEnsembleSpread = useColorMapVariance && i == 0;
        if (isEnsembleSpread) {
            variableName = u8"\u03C3"; //< sigma.
            colorMap = [fieldDataArrayLocal](float t) {
                return fieldDataArrayLocal.front()->evalColorMapVec4Variance(t, true);
            };
            labelMap = [](float t) {
                return t < 0.0001f ? "min" : (t > 0.9999f ? "max" : "");
            };
        } else {
            int fieldIdx = 0;
            if (limitedFieldIdx >= 0) {
                for (int j = 0; j < numFields; j++) {
                    if (fieldDataArray.at(j)->selectedFieldIdx == limitedFieldIdx) {
                        fieldIdx = j;
                    }
                }
            } else {
                fieldIdx = useColorMapVariance ? i - 1 : i;
            }
            fieldData = fieldDataArray.at(fieldIdx).get();
            variableName = fieldData->selectedScalarFieldName;
            colorMap = [fieldData](float t) {
                return fieldData->evalColorMapVec4(t);
            };
            labelMap = [minMi, maxMi](float t) {
                return getNiceNumberString((1.0f - t) * minMi + t * maxMi, 2);
            };
        }

        int numLabels = 5;
        int numTicks = 5;
        if (colorLegendHeight < textSizeLegend * 0.625f) {
            numLabels = 0;
            numTicks = 0;
        } else if (colorLegendHeight < textSizeLegend * 2.0f) {
            numLabels = 2;
            numTicks = 2;
        } else if (colorLegendHeight < textSizeLegend * 4.0f) {
            numLabels = 3;
            numTicks = 3;
        }

        float posX;
        if (arrangeColorLegendsBothSides && i < numFieldsSize / 2) {
            posX = borderSizeX + float(i) * (colorLegendWidth + textWidthMax + colorLegendSpacing);
        } else {
            posX =
                    windowWidth - borderSizeX
                    - float(ix) * (colorLegendWidth + textWidthMax)
                    - float(ix - 1) * colorLegendSpacing;
        }

        float posY = windowHeight - borderSizeY - colorLegendHeight;
        if (diagramMode == DiagramMode::MATRIX && outerRingSizePct > 0.0f) {
            posY -= outerRingWidth + 5;
        }

        drawColorLegend(
                posX, posY, colorLegendWidth, colorLegendHeight, numLabels, numTicks, labelMap, colorMap, variableName);
    }
}

void HEBChart::drawColorLegend(
        float x, float y, float w, float h, int numLabels, int numTicks,
        const std::function<std::string(float)>& labelMap, const std::function<glm::vec4(float)>& colorMap,
        const std::string& textTop) {
    sgl::Color textColor = isDarkMode ? textColorDark : textColorBright;
    NVGcolor textColorNvg;
    if (vg) {
        textColorNvg = nvgRGBA(textColor.getR(), textColor.getG(), textColor.getB(), 255);
    }
#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr, *gradientPaint = nullptr;
    SkFont* font = nullptr;
    SkFontMetrics metrics{};
    if (canvas) {
        paint = new SkPaint;
        gradientPaint = new SkPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(paint);
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(gradientPaint);
        font = new SkFont(typeface, textSizeLegend * s);
        font->getMetrics(&metrics);
    }
#endif

    const int numPoints = 17; // 9
    const int numSubdivisions = numPoints - 1;

    // Draw color bar.
    for (int i = 0; i < numSubdivisions; i++) {
        float t0 = 1.0f - float(i) / float(numSubdivisions);
        float t1 = 1.0f - float(i + 1) / float(numSubdivisions);

        glm::vec4 fillColorVec0 = colorMap(t0);
        glm::vec4 fillColorVec1 = colorMap(t1);
        fillColorVec0.w = 1.0f;
        fillColorVec1.w = 1.0f;
        if (vg) {
            auto fillColor0 = nvgRGBAf(fillColorVec0.x, fillColorVec0.y, fillColorVec0.z, 1.0f);
            auto fillColor1 = nvgRGBAf(fillColorVec1.x, fillColorVec1.y, fillColorVec1.z, 1.0f);
            nvgBeginPath(vg);
            nvgRect(vg, x, y + h * float(i) / float(numSubdivisions), w, h / float(numSubdivisions) + 1e-1f);
            NVGpaint paint = nvgLinearGradient(
                    vg, x, y + h * float(i) / float(numSubdivisions),
                    x, y + h * float(i+1) / float(numSubdivisions),
                    fillColor0, fillColor1);
            nvgFillPaint(vg, paint);
            nvgFill(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            auto fillColor0 = toSkColor(sgl::colorFromVec4(fillColorVec0));
            auto fillColor1 = toSkColor(sgl::colorFromVec4(fillColorVec1));
            SkPoint linearPoints[2] = {
                    { x * s, (y + h * float(i) / float(numSubdivisions)) * s },
                    { x * s, (y + h * float(i + 1) / float(numSubdivisions)) * s }
            };
            SkColor linearColors[2] = { fillColor0, fillColor1 };
            gradientPaint->setShader(SkGradientShader::MakeLinear(
                    linearPoints, linearColors, nullptr, 2, SkTileMode::kClamp));
            canvas->drawRect(
                    SkRect{
                        x * s, (y + h * float(i) / float(numSubdivisions)) * s,
                        (x + w) * s, (y + h * float(i + 1) / float(numSubdivisions) + 1e-1f) * s}, *gradientPaint);
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            auto pattern = vkvg_pattern_create_linear(
                    x * s, (y + h * float(i) / float(numSubdivisions)) * s,
                    x * s, (y + h * float(i + 1) / float(numSubdivisions)) * s);
            vkvg_pattern_add_color_stop(pattern, 0.0f, fillColorVec0.x, fillColorVec0.y, fillColorVec0.z, 1.0f);
            vkvg_pattern_add_color_stop(pattern, 1.0f, fillColorVec1.x, fillColorVec1.y, fillColorVec1.z, 1.0f);
            vkvg_set_source(context, pattern);
            vkvg_pattern_destroy(pattern);
            vkvg_rectangle(
                    context, x * s, (y + h * float(i) / float(numSubdivisions)) * s,
                    w * s, (h / float(numSubdivisions) + 1e-1f) * s);
            vkvg_fill(context);
        }
#endif
    }

    // Draw ticks.
    const float tickWidth = 4.0f;
    const float tickHeight = 1.0f;
    if (vg) {
        nvgBeginPath(vg);
        for (int tickIdx = 0; tickIdx < numTicks; tickIdx++) {
            float centerY = y + float(tickIdx) / float(numTicks - 1) * h;
            nvgRect(vg, x + w, centerY - tickHeight / 2.0f, tickWidth, tickHeight);
        }
        nvgFillColor(vg, textColorNvg);
        nvgFill(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setColor(toSkColor(textColor));
        paint->setStroke(false);
        for (int tickIdx = 0; tickIdx < numTicks; tickIdx++) {
            float centerY = y + float(tickIdx) / float(numTicks - 1) * h;
            canvas->drawRect(
                    SkRect{
                        (x + w) * s, (centerY - tickHeight / 2.0f) * s,
                        (x + w + tickWidth) * s, (centerY + tickHeight / 2.0f) * s}, *paint);
        }
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_source_color(context, textColor.getColorRGBA());
        for (int tickIdx = 0; tickIdx < numTicks; tickIdx++) {
            float centerY = y + float(tickIdx) / float(numTicks - 1) * h;
            vkvg_rectangle(
                    context, (x + w) * s, (centerY - tickHeight / 2.0f) * s, tickWidth * s, tickHeight * s);
        }
        vkvg_fill(context);
    }
#endif

    // Draw on the right.
    if (vg) {
        nvgFontSize(vg, textSizeLegend);
        nvgFontFace(vg, "sans");
        for (int tickIdx = 0; tickIdx < numLabels; tickIdx++) {
            float t = 1.0f - float(tickIdx) / float(numLabels - 1);
            float centerY = y + float(tickIdx) / float(numLabels - 1) * h;
            std::string labelText = labelMap(t);
            nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
            nvgFillColor(vg, textColorNvg);
            nvgText(vg, x + w + 2.0f * tickWidth, centerY, labelText.c_str(), nullptr);
        }
        nvgFillColor(vg, textColorNvg);
        nvgFill(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setColor(toSkColor(textColor));
        for (int tickIdx = 0; tickIdx < numLabels; tickIdx++) {
            float t = 1.0f - float(tickIdx) / float(numLabels - 1);
            float centerY = y + float(tickIdx) / float(numLabels - 1) * h;
            std::string labelText = labelMap(t);
            SkRect bounds{};
            font->measureText(labelText.c_str(), labelText.size(), SkTextEncoding::kUTF8, &bounds);
            canvas->drawString(
                    labelText.c_str(),
                    (x + w + 2.0f * tickWidth) * s, centerY * s + 0.5f * (bounds.height() - metrics.fDescent),
                    *font, *paint);
        }
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_font_size(context, uint32_t(std::round(textSizeLegend * s * 0.75f)));
        //vkvg_select_font_face(context, "sans");
        vkvg_set_source_color(context, textColor.getColorRGBA());
        for (int tickIdx = 0; tickIdx < numLabels; tickIdx++) {
            float t = 1.0f - float(tickIdx) / float(numLabels - 1);
            float centerY = y + float(tickIdx) / float(numLabels - 1) * h;
            std::string labelText = labelMap(t);
            vkvg_text_extents_t te{};
            vkvg_text_extents(context, labelText.c_str(), &te);
            vkvg_font_extents_t fe{};
            vkvg_font_extents(context, &fe);
            vkvg_move_to(context, (x + w + 2.0f * tickWidth) * s, centerY * s + 0.5f * te.height - fe.descent);
            vkvg_show_text(context, labelText.c_str());
        }
    }
#endif

    // Draw text on the top.
    if (vg) {
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_BOTTOM);
        nvgFillColor(vg, textColorNvg);
        nvgText(vg, x + w * 0.5f, y - 4, textTop.c_str(), nullptr);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        SkRect bounds{};
        font->measureText(textTop.c_str(), textTop.size(), SkTextEncoding::kUTF8, &bounds);
        paint->setColor(toSkColor(textColor));
        canvas->drawString(
                textTop.c_str(), (x + w * 0.5f) * s - 0.5f * bounds.width(), (y - 4) * s - metrics.fDescent,
                *font, *paint);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_text_extents_t te{};
        vkvg_text_extents(context, textTop.c_str(), &te);
        vkvg_font_extents_t fe{};
        vkvg_font_extents(context, &fe);
        vkvg_move_to(context, (x + w * 0.5f) * s - 0.5f * te.width, (y - 4) * s - fe.descent);
        vkvg_show_text(context, textTop.c_str());
    }
#endif

    // Draw box outline.
    if (vg) {
        nvgBeginPath(vg);
        nvgRect(vg, x, y, w, h);
        nvgStrokeWidth(vg, 0.75f);
        nvgStrokeColor(vg, textColorNvg);
        nvgStroke(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setStroke(true);
        paint->setStrokeWidth(0.75f * s);
        paint->setColor(toSkColor(textColor));
        canvas->drawRect(SkRect{x * s, y * s, (x + w) * s, (y + h) * s}, *paint);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_line_width(context, 0.75f * s);
        vkvg_set_source_color(context, textColor.getColorRGBA());
        vkvg_rectangle(context, x * s, y * s, w * s, h * s);
        vkvg_stroke(context);
    }
#endif

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
        delete gradientPaint;
        delete font;
    }
#endif
}

template<typename T> inline T sqr(T val) { return val * val; }

float HEBChart::computeColorLegendHeightForNumFields(int numFields, float maxHeight) {
    const float r = totalRadius;
    const float bx = borderSizeX;
    const float by = borderSizeY;
    const float w = windowWidth;
    const float h = windowHeight;
    const float d = colorLegendCircleDist;
    const float cw = float(numFields) * (colorLegendWidth + textWidthMax) + float(numFields - 1) * colorLegendSpacing;

    const float b = -2.0f * (h - by - h * 0.5f);
    const float c = sqr(h - by - h * 0.5f) - sqr(r + d) + sqr(w - bx - cw - w * 0.5f);
    const float discriminant = b * b - 4.0f * c;

    float height;
    if (discriminant > 0.0f) {
        height = (-b - std::sqrt(b * b - 4.0f * c)) * 0.5f - textSize * 2.0f;
        height = std::clamp(height, 1.0f, maxHeight);
    } else {
        height = maxHeight;
    }
    return height;
}

void HEBChart::computeColorLegendHeight() {
    textWidthMax = textWidthMaxBase * textSize / 8.0f;
    auto numFieldsSize = limitedFieldIdx >= 0 ? std::min(int(fieldDataArray.size()), 1) : int(fieldDataArray.size());

    // Add transfer function for ensemble spread.
    if (separateColorVarianceAndCorrelation) {
        numFieldsSize++;
    }

    const float maxHeight = std::min(maxColorLegendHeight, totalRadius * 0.5f);
    colorLegendHeight = computeColorLegendHeightForNumFields(numFieldsSize, maxHeight);
    if (numFieldsSize > 1 && colorLegendHeight < maxHeight * 0.5f) {
        colorLegendHeight = computeColorLegendHeightForNumFields(sgl::iceil(numFieldsSize, 2), maxHeight);
        arrangeColorLegendsBothSides = true;
    } else {
        arrangeColorLegendsBothSides = false;
    }
}

std::pair<float, float> HEBChart::getLocalStdDevRange(int fieldIdx, int varNum) {
    auto numFields = int(fieldDataArray.size());
    for (int i = 0; i < numFields; i++) {
        auto* fieldData = fieldDataArray.at(i).get();
        if (limitedFieldIdx >= 0 && limitedFieldIdx != fieldData->selectedFieldIdx) {
            continue;
        }
        if (fieldData->selectedFieldIdx == fieldIdx) {
            if (fieldData->useTwoFields && varNum != 0 && varNum != 1) {
                return std::make_pair(
                        std::min(fieldData->minStdDev, fieldData->minStdDev2),
                        std::max(fieldData->maxStdDev, fieldData->maxStdDev2));
            } else if (fieldData->useTwoFields && varNum == 1) {
                return std::make_pair(fieldData->minStdDev2, fieldData->maxStdDev2);
            } else {
                return std::make_pair(fieldData->minStdDev, fieldData->maxStdDev);
            }
        }
    }
    return std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
}

void HEBChart::setGlobalStdDevRangeQueryCallback(
        std::function<std::pair<float, float>(int, int )> callback) {
    globalStdDevRangeQueryCallback = std::move(callback);
}

std::pair<float, float> HEBChart::getGlobalStdDevRange(int fieldIdx, int varNum) {
    return globalStdDevRangeQueryCallback(fieldIdx, varNum);
}
