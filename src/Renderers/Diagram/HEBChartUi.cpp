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
#include "HEBChart.hpp"

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
    setDefaultBackendId(defaultBackendId);
}

void HEBChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    chartRadius = 160;
    if (showRing) {
        borderSizeX += outerRingWidth + outerRingOffset;
        borderSizeY += outerRingWidth + outerRingOffset;
    }

    windowWidth = (chartRadius + borderSizeX) * 2.0f;
    windowHeight = (chartRadius + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onWindowSizeChanged();
}

void HEBChart::onUpdatedWindowSize() {
    float minDim = std::min(windowWidth - 2.0f * borderSizeX, windowHeight - 2.0f * borderSizeY);
    chartRadius = std::round(0.5f * minDim);
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
    chartRadius = float(radius);
    windowWidth = (chartRadius + borderSizeX) * 2.0f;
    windowHeight = (chartRadius + borderSizeY) * 2.0f;
    onWindowSizeChanged();
}

void HEBChart::setAlignWithParentWindow(bool _align) {
    alignWithParentWindow = _align;
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

void HEBChart::setColorMap(int fieldIdx, DiagramColorMap _colorMap) {
    fieldDataArray.at(fieldIdx)->colorMap = _colorMap;
    fieldDataArray.at(fieldIdx)->initializeColorPoints();
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
    windowOffsetX = 0;
    windowOffsetY = 0;
    windowWidth = float(parentWidth);
    windowHeight = float(parentHeight);
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void HEBChart::update(float dt) {
    DiagramBase::update(dt);

    glm::vec2 mousePosition(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePosition -= glm::vec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    //else {
    //    mousePosition.y = float(sgl::AppSettings::get()->getMainWindow()->getHeight()) - mousePosition.y - 1;
    //}
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();
    //mousePosition.y = windowHeight - mousePosition.y;

    //std::cout << "mousePosition: " << mousePosition.x << ", " << mousePosition.y << std::endl;

    sgl::AABB2 windowAabb(
            glm::vec2(borderWidth, borderWidth),
            glm::vec2(windowWidth - 2.0f * borderWidth, windowHeight - 2.0f * borderWidth));
    bool isMouseInWindow = windowAabb.contains(mousePosition) && !isMouseGrabbedByParent;
    if (!isMouseInWindow) {
        if (hoveredPointIdx != -1) {
            hoveredPointIdx = -1;
            needsReRender = true;
        }
        if (hoveredLineIdx != -1) {
            hoveredLineIdx = -1;
            needsReRender = true;
        }
    } else {
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
        if (!curvePoints.empty() && hoveredPointIdx < 0) {
            // TODO: Test if point lies in convex hull of control points first (using @see isInsidePolygon).
            int closestLineIdx = -1;
            float closestLineDist = std::numeric_limits<float>::max();
            for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
                for (int ptIdx = 0; ptIdx < NUM_SUBDIVISIONS - 1; ptIdx++) {
                    glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                    pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
                    pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
                    glm::vec2 pt1 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx + 1);
                    pt1.x = windowWidth / 2.0f + pt1.x * chartRadius;
                    pt1.y = windowHeight / 2.0f + pt1.y * chartRadius;
                    float dist = getDistanceToLineSegment(mousePosition, pt0, pt1);
                    if (dist < closestLineDist && dist <= minDist) {
                        closestLineIdx = lineIdx;
                        closestLineDist = dist;
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
    }

    if (isMouseInWindow && sgl::Mouse->buttonReleased(1) && !windowMoveOrResizeJustFinished) {
        clickedLineIdx = -1;
        clickedPointIdx = -1;
        if (hoveredLineIdx >= 0) {
            clickedLineIdx = hoveredLineIdx;
        } else if (hoveredPointIdx >= 0) {
            clickedPointIdx = hoveredPointIdx;
        }
    }

    int newSelectedLineIdx = -1;
    int newSelectedPointIndices[2] = { -1, -1 };
    if (hoveredLineIdx >= 0) {
        newSelectedLineIdx = hoveredLineIdx;
    } else if (clickedLineIdx >= 0 && hoveredPointIdx < 0) {
        newSelectedLineIdx = clickedLineIdx;
    }

    if (newSelectedLineIdx >= 0) {
        const auto& points = connectedPointsArray.at(newSelectedLineIdx);
        newSelectedPointIndices[0] = points.first;
        newSelectedPointIndices[1] = points.second;
    } else if (hoveredPointIdx >= 0) {
        newSelectedPointIndices[0] = hoveredPointIdx;
    } else if (clickedPointIdx >= 0) {
        newSelectedPointIndices[0] = clickedPointIdx;
    }

    if (selectedPointIndices[0] != newSelectedPointIndices[0] || selectedPointIndices[1] != newSelectedPointIndices[1]
            || selectedLineIdx != newSelectedLineIdx) {
        needsReRender = true;
    }
    selectedLineIdx = newSelectedLineIdx;
    selectedPointIndices[0] = newSelectedPointIndices[0];
    selectedPointIndices[1] = newSelectedPointIndices[1];
}

void HEBChartFieldData::initializeColorPoints() {
    colorPoints = getColorPoints(colorMap);
}

glm::vec4 HEBChartFieldData::evalColorMapVec4(float t) {
    t = glm::clamp(t, 0.0f, 1.0f);
    //glm::vec3 c0(208.0f/255.0f, 231.0f/255.0f, 208.0f/255.0f);
    //glm::vec3 c1(100.0f/255.0f, 1.0f, 100.0f/255.0f);
    float opacity = parent->curveOpacity;
    if (parent->opacityByValue) {
        opacity *= t * 0.75f + 0.25f;
    }
    //return glm::vec4(glm::mix(c0, c1, t), opacity);
    auto N = int(colorPoints.size());
    float arrayPosFlt = t * float(N - 1);
    int lastIdx = std::min(int(arrayPosFlt), N - 1);
    int nextIdx = std::min(lastIdx + 1, N - 1);
    float f1 = arrayPosFlt - float(lastIdx);
    const glm::vec3& c0 = colorPoints.at(lastIdx);
    const glm::vec3& c1 = colorPoints.at(nextIdx);
    return glm::vec4(glm::mix(c0, c1, f1), opacity);
}

sgl::Color HEBChartFieldData::evalColorMap(float t) {
    return sgl::colorFromVec4(evalColorMapVec4(t));
}

void HEBChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    NVGcolor circleFillColorNvg = nvgRGBA(
            circleFillColor.getR(), circleFillColor.getG(),
            circleFillColor.getB(), circleFillColor.getA());
    NVGcolor circleFillColorSelectedNvg = nvgRGBA(
            circleFillColorSelected.getR(), circleFillColorSelected.getG(),
            circleFillColorSelected.getB(), circleFillColorSelected.getA());

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    // Draw the B-spline curves.
    NVGcolor curveStrokeColor = nvgRGBA(
            100, 255, 100, uint8_t(std::clamp(int(std::ceil(curveOpacity * 255.0f)), 0, 255)));
    if (!curvePoints.empty()) {
        for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
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

            float strokeWidth = lineIdx == selectedLineIdx ? curveThickness * 2.0f : curveThickness;
            nvgStrokeWidth(vg, strokeWidth);

            if (colorByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    float maxMi = correlationValuesArray.back();
                    float minMi = correlationValuesArray.front();
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(lineIdx)).get();
                glm::vec4 color = fieldData->evalColorMapVec4(factor);
                curveStrokeColor = nvgRGBAf(color.x, color.y, color.z, color.w);
            } else if (opacityByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    float maxMi = correlationValuesArray.back();
                    float minMi = correlationValuesArray.front();
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                }
                curveStrokeColor.a = curveOpacity * factor;
            } else {
                curveStrokeColor.a = curveOpacity;
            }

            curveStrokeColor.a = lineIdx == selectedLineIdx ? 1.0f : curveStrokeColor.a;
            nvgStrokeColor(vg, curveStrokeColor);

            nvgStroke(vg);
        }
    }

    // Draw the point circles.
    float pointRadius = curveThickness * pointRadiusBase;
    nvgBeginPath(vg);
    /*for (int i = 0; i < int(nodesList.size()); i++) {
        const auto& leaf = nodesList.at(i);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgCircle(vg, pointX, pointY, pointRadius);
    }*/
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
    nvgFillColor(vg, circleFillColorNvg);
    nvgFill(vg);

    int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
    for (int idx = 0; idx < numPointsSelected; idx++) {
        const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgBeginPath(vg);
        nvgCircle(vg, pointX, pointY, pointRadius * 1.5f);
        nvgFillColor(vg, circleFillColorSelectedNvg);
        nvgFill(vg);
    }

    if (showRing) {
        glm::vec2 center(windowWidth / 2.0f, windowHeight / 2.0f);

        auto numFields = int(fieldDataArray.size());
        for (int i = 0; i < numFields; i++) {
            auto* fieldData = fieldDataArray.at(i).get();
            float pctLower = float(i) / float(numFields);
            float pctMiddle = (float(i) + 0.5f) / float(numFields);
            float pctUpper = float(i + 1) / float(numFields);
            float rlo = chartRadius + outerRingOffset + pctLower * outerRingWidth;
            float rmi = chartRadius + outerRingOffset + pctMiddle * outerRingWidth;
            float rhi = chartRadius + outerRingOffset + pctUpper * outerRingWidth;

            for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
                int numLeaves = int(nodesList.size()) - int(leafIdxOffset);
                int nextIdx = (leafIdx + 1 - int(leafIdxOffset)) % numLeaves + int(leafIdxOffset);
                const auto &leafCurr = nodesList.at(leafIdx);
                const auto &leafNext = nodesList.at(nextIdx);
                float angle0 = leafCurr.angle;
                float angle1 = leafNext.angle + 0.01f;
                float cos0 = std::cos(angle0), sin0 = std::sin(angle0), cos1 = std::cos(angle1), sin1 = std::sin(
                        angle1);
                //glm::vec2 lo0 = center + rlo * glm::vec2(cos0, sin0);
                glm::vec2 lo1 = center + rlo * glm::vec2(cos1, sin1);
                glm::vec2 hi0 = center + rhi * glm::vec2(cos0, sin0);
                //glm::vec2 hi1 = center + rhi * glm::vec2(cos1, sin1);
                glm::vec2 mi0 = center + rmi * glm::vec2(cos0, sin0);
                glm::vec2 mi1 = center + rmi * glm::vec2(cos1, sin1);
                nvgBeginPath(vg);
                nvgArc(vg, center.x, center.y, rlo, angle1, angle0, NVG_CCW);
                nvgLineTo(vg, hi0.x, hi0.y);
                nvgArc(vg, center.x, center.y, rhi, angle0, angle1, NVG_CW);
                nvgLineTo(vg, lo1.x, lo1.y);
                nvgClosePath(vg);

                float stdev0 = fieldData->leafStdDevArray.at(leafIdx - int(leafIdxOffset));
                float t0 = fieldData->minStdDev == fieldData->maxStdDev ? 0.0f : (stdev0 - fieldData->minStdDev) / (fieldData->maxStdDev - fieldData->minStdDev);
                glm::vec4 rgbColor0 = fieldData->evalColorMapVec4(t0);
                //glm::vec4 rgbColor0 = fieldData->evalColorMapVec4(leafCurr.angle / sgl::TWO_PI);
                rgbColor0.w = 1.0f;
                NVGcolor fillColor0 = nvgRGBAf(rgbColor0.x, rgbColor0.y, rgbColor0.z, rgbColor0.w);
                float stdev1 = fieldData->leafStdDevArray.at(nextIdx - int(leafIdxOffset));
                float t1 = fieldData->minStdDev == fieldData->maxStdDev ? 0.0f : (stdev1 - fieldData->minStdDev) / (fieldData->maxStdDev - fieldData->minStdDev);
                glm::vec4 rgbColor1 = fieldData->evalColorMapVec4(t1);
                //glm::vec4 rgbColor1 = fieldData->evalColorMapVec4(leafNext.angle / sgl::TWO_PI);
                rgbColor1.w = 1.0f;
                NVGcolor fillColor1 = nvgRGBAf(rgbColor1.x, rgbColor1.y, rgbColor1.z, rgbColor1.w);

                NVGpaint paint = nvgLinearGradient(vg, mi0.x, mi0.y, mi1.x, mi1.y, fillColor0, fillColor1);
                nvgFillPaint(vg, paint);
                nvgFill(vg);
            }
        }

        for (int i = 0; i < numFields; i++) {
            float pctLower = float(i) / float(numFields);
            float pctUpper = float(i + 1) / float(numFields);
            float rlo = chartRadius + outerRingOffset + pctLower * outerRingWidth;
            float rhi = chartRadius + outerRingOffset + pctUpper * outerRingWidth;
            NVGcolor circleStrokeColorNvg = nvgRGBA(
                    circleStrokeColor.getR(), circleStrokeColor.getG(),
                    circleStrokeColor.getB(), circleStrokeColor.getA());
            nvgBeginPath(vg);
            nvgCircle(vg, center.x, center.y, rlo);
            if (i == numFields - 1) {
                nvgCircle(vg, center.x, center.y, rhi);
            }
            nvgStrokeWidth(vg, 1.0f);
            nvgStrokeColor(vg, circleStrokeColorNvg);
            nvgStroke(vg);
        }
    }
}

#ifdef SUPPORT_SKIA
void HEBChart::renderBaseSkia() {
    DiagramBase::renderBaseSkia();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    SkPaint paint;
    static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(&paint);

    // Draw the B-spline curves.
    sgl::Color curveStrokeColor = sgl::Color(
            100, 255, 100, uint8_t(std::clamp(int(std::ceil(curveOpacity * 255.0f)), 0, 255)));
    if (!curvePoints.empty()) {
        paint.setStroke(true);
        paint.setStrokeWidth(curveThickness * s);
        paint.setColor(toSkColor(curveStrokeColor));
        for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
            SkPath path;
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
                    float maxMi = correlationValuesArray.back();
                    float minMi = correlationValuesArray.front();
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(lineIdx)).get();
                curveStrokeColor = fieldData->evalColorMap(factor);
                paint.setColor(toSkColor(curveStrokeColor));
            } else if (opacityByValue) {
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    float maxMi = correlationValuesArray.back();
                    float minMi = correlationValuesArray.front();
                    factor = (correlationValuesArray.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                }
                curveStrokeColor.setFloatA(curveOpacity * factor);
                paint.setColor(toSkColor(curveStrokeColor));
            }

            if (selectedLineIdx == lineIdx) {
                sgl::Color curveStrokeColorSelected = curveStrokeColor;
                curveStrokeColorSelected.setA(255);
                paint.setColor(toSkColor(curveStrokeColorSelected));
                paint.setStrokeWidth(curveThickness * 2.0f * s);
            } else {
                if (!colorByValue && !opacityByValue) {
                    paint.setColor(toSkColor(curveStrokeColor));
                }
                paint.setStrokeWidth(curveThickness * s);
            }

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
        paint.setColor(toSkColor(circleFillColorSelected));
        canvas->drawCircle(pointX * s, pointY * s, pointRadius * 1.5f, paint);
    }

    if (showRing) {
        SkPaint gradientPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(&gradientPaint);
        gradientPaint.setStroke(false);
        SkPath path;

        glm::vec2 center(windowWidth / 2.0f * s, windowHeight / 2.0f * s);
        auto numFields = int(fieldDataArray.size());
        for (int i = 0; i < numFields; i++) {
            auto* fieldData = fieldDataArray.at(i).get();
            float pctLower = float(i) / float(numFields);
            float pctMiddle = (float(i) + 0.5f) / float(numFields);
            float pctUpper = float(i + 1) / float(numFields);
            float rlo = (chartRadius + outerRingOffset + pctLower * outerRingWidth) * s;
            float rmi = (chartRadius + outerRingOffset + pctMiddle * outerRingWidth) * s;
            float rhi = (chartRadius + outerRingOffset + pctUpper * outerRingWidth) * s;

            for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
                int numLeaves = int(nodesList.size()) - int(leafIdxOffset);
                int nextIdx = (leafIdx + 1 - int(leafIdxOffset)) % numLeaves + int(leafIdxOffset);
                const auto &leafCurr = nodesList.at(leafIdx);
                const auto &leafNext = nodesList.at(nextIdx);
                float angle0 = leafCurr.angle;
                float angle1 = leafNext.angle + 0.01f;
                float angle0Deg = angle0 / sgl::PI * 180.0f;
                float angle1Deg = angle1 / sgl::PI * 180.0f + (nextIdx < leafIdx ? 360.0f : 0.0f);
                float cos0 = std::cos(angle0), sin0 = std::sin(angle0), cos1 = std::cos(angle1), sin1 = std::sin(
                        angle1);
                //glm::vec2 lo0 = center + rlo * glm::vec2(cos0, sin0);
                //glm::vec2 lo1 = center + rlo * glm::vec2(cos1, sin1);
                glm::vec2 hi0 = center + rhi * glm::vec2(cos0, sin0);
                //glm::vec2 hi1 = center + rhi * glm::vec2(cos1, sin1);
                glm::vec2 mi0 = center + rmi * glm::vec2(cos0, sin0);
                glm::vec2 mi1 = center + rmi * glm::vec2(cos1, sin1);

                float stdev0 = fieldData->leafStdDevArray.at(leafIdx - int(leafIdxOffset));
                float t0 = fieldData->minStdDev == fieldData->maxStdDev ? 0.0f : (stdev0 - fieldData->minStdDev) / (fieldData->maxStdDev - fieldData->minStdDev);
                sgl::Color rgbColor0 = fieldData->evalColorMap(t0);
                rgbColor0.setA(255);
                float stdev1 = fieldData->leafStdDevArray.at(nextIdx - int(leafIdxOffset));
                float t1 = fieldData->minStdDev == fieldData->maxStdDev ? 0.0f : (stdev1 - fieldData->minStdDev) / (fieldData->maxStdDev - fieldData->minStdDev);
                sgl::Color rgbColor1 = fieldData->evalColorMap(t1);
                rgbColor1.setA(255);

                SkPoint linearPoints[2] = {{mi0.x, mi0.y},
                                           {mi1.x, mi1.y}};
                SkColor linearColors[2] = {toSkColor(rgbColor0), toSkColor(rgbColor1)};
                gradientPaint.setShader(SkGradientShader::MakeLinear(
                        linearPoints, linearColors, nullptr, 2, SkTileMode::kClamp));

                path.reset();
                path.addArc(
                        SkRect{center.x - rlo, center.y - rlo, center.x + rlo, center.y + rlo},
                        angle1Deg, angle0Deg - angle1Deg);
                path.lineTo(hi0.x, hi0.y);
                path.arcTo(
                        SkRect{center.x - rhi, center.y - rhi, center.x + rhi, center.y + rhi},
                        angle0Deg, angle1Deg - angle0Deg, false);
                //path.lineTo(lo1.x, lo1.y);
                path.close();

                canvas->drawPath(path, gradientPaint);
            }
        }

        for (int i = 0; i < numFields; i++) {
            float pctLower = float(i) / float(numFields);
            float pctUpper = float(i + 1) / float(numFields);
            float rlo = (chartRadius + outerRingOffset + pctLower * outerRingWidth) * s;
            float rhi = (chartRadius + outerRingOffset + pctUpper * outerRingWidth) * s;
            paint.setStroke(true);
            paint.setStrokeWidth(1.0f * s);
            paint.setColor(toSkColor(circleStrokeColor));
            canvas->drawCircle(center.x, center.y, rlo, paint);
            if (i == numFields - 1) {
                canvas->drawCircle(center.x, center.y, rhi, paint);
            }
        }
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

    // Draw the B-spline curves.
    sgl::Color curveStrokeColor = sgl::Color(100, 255, 100, 255);
    if (!curvePoints.empty()) {
        vkvg_set_line_width(context, curveThickness * s);
        vkvg_set_source_color(context, curveStrokeColor.getColorRGBA());

        if (colorByValue || opacityByValue) {
            float maxMi = correlationValuesArray.empty() ? 0.0f : correlationValuesArray.back();
            float minMi = correlationValuesArray.empty() ? 0.0f : correlationValuesArray.front();
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
            vkvg_set_line_width(context, curveThickness * 2.0f * s);
            if (colorByValue) {
                float maxMi = correlationValuesArray.empty() ? 0.0f : correlationValuesArray.back();
                float minMi = correlationValuesArray.empty() ? 0.0f : correlationValuesArray.front();
                float factor = 1.0f;
                if (correlationValuesArray.size() > 1) {
                    factor = (correlationValuesArray.at(selectedLineIdx) - minMi) / (maxMi - minMi);
                }
                HEBChartFieldData* fieldData = fieldDataArray.at(lineFieldIndexArray.at(selectedLineIdx)).get();
                auto color = fieldData->evalColorMap(factor);
                vkvg_set_source_color(context, color.getColorRGB());
            }
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
        vkvg_set_source_color(context, circleFillColorSelected.getColorRGBA());
        vkvg_fill(context);
    }

    if (showRing) {
        glm::vec2 center(windowWidth / 2.0f * s, windowHeight / 2.0f * s);
        auto numFields = int(fieldDataArray.size());
        for (int i = 0; i < numFields; i++) {
            auto* fieldData = fieldDataArray.at(i).get();
            float pctLower = float(i) / float(numFields);
            float pctMiddle = (float(i) + 0.5f) / float(numFields);
            float pctUpper = float(i + 1) / float(numFields);
            float rlo = (chartRadius + outerRingOffset + pctLower * outerRingWidth) * s;
            float rmi = (chartRadius + outerRingOffset + pctMiddle * outerRingWidth) * s;
            float rhi = (chartRadius + outerRingOffset + pctUpper * outerRingWidth) * s;

            for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
                int numLeaves = int(nodesList.size()) - int(leafIdxOffset);
                int nextIdx = (leafIdx + 1 - int(leafIdxOffset)) % numLeaves + int(leafIdxOffset);
                const auto &leafCurr = nodesList.at(leafIdx);
                const auto &leafNext = nodesList.at(nextIdx);
                float angle0 = leafCurr.angle;
                float angle1 = leafNext.angle + 0.01f;
                float cos0 = std::cos(angle0), sin0 = std::sin(angle0), cos1 = std::cos(angle1), sin1 = std::sin(
                        angle1);
                //glm::vec2 lo0 = center + rlo * glm::vec2(cos0, sin0);
                glm::vec2 lo1 = center + rlo * glm::vec2(cos1, sin1);
                glm::vec2 hi0 = center + rhi * glm::vec2(cos0, sin0);
                //glm::vec2 hi1 = center + rhi * glm::vec2(cos1, sin1);
                glm::vec2 mi0 = center + rmi * glm::vec2(cos0, sin0);
                glm::vec2 mi1 = center + rmi * glm::vec2(cos1, sin1);
                vkvg_arc_negative(context, center.x, center.y, rlo, angle1, angle0);
                vkvg_line_to(context, hi0.x, hi0.y);
                vkvg_arc(context, center.x, center.y, rhi, angle0, angle1);
                vkvg_line_to(context, lo1.x, lo1.y);

                float stdev0 = fieldData->leafStdDevArray.at(leafIdx - int(leafIdxOffset));
                float t0 = fieldData->minStdDev == fieldData->maxStdDev ? 0.0f : (stdev0 - fieldData->minStdDev) / (fieldData->maxStdDev - fieldData->minStdDev);
                glm::vec4 rgbColor0 = fieldData->evalColorMapVec4(t0);
                float stdev1 = fieldData->leafStdDevArray.at(nextIdx - int(leafIdxOffset));
                float t1 = fieldData->minStdDev == fieldData->maxStdDev ? 0.0f : (stdev1 - fieldData->minStdDev) / (fieldData->maxStdDev - fieldData->minStdDev);
                glm::vec4 rgbColor1 = fieldData->evalColorMapVec4(t1);

                auto pattern = vkvg_pattern_create_linear(mi0.x, mi0.y, mi1.x, mi1.y);
                vkvg_pattern_add_color_stop(pattern, 0.0f, rgbColor0.x, rgbColor0.y, rgbColor0.z, 1.0f);
                vkvg_pattern_add_color_stop(pattern, 1.0f, rgbColor1.x, rgbColor1.y, rgbColor1.z, 1.0f);
                vkvg_set_source(context, pattern);
                vkvg_pattern_destroy(pattern);
                vkvg_fill(context);
            }
        }

        for (int i = 0; i < numFields; i++) {
            float pctLower = float(i) / float(numFields);
            float pctUpper = float(i + 1) / float(numFields);
            float rlo = (chartRadius + outerRingOffset + pctLower * outerRingWidth) * s;
            float rhi = (chartRadius + outerRingOffset + pctUpper * outerRingWidth) * s;
            vkvg_arc(context, center.x, center.y, rlo, 0.0f, sgl::TWO_PI);
            if (i == numFields - 1) {
                vkvg_arc(context, center.x, center.y, rhi, 0.0f, sgl::TWO_PI);
            }
            vkvg_set_line_width(context, 1.0f * s);
            vkvg_set_source_color(context, circleStrokeColor.getColorRGBA());
            vkvg_stroke(context);
        }
    }
}
#endif
