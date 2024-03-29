/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#include <Math/Math.hpp>
#include <Input/Mouse.hpp>
#include <Graphics/Vector/nanovg/nanovg.h>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>

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

#ifdef SUPPORT_SKIA
#include "../VectorBackendSkia.hpp"
#endif
#ifdef SUPPORT_VKVG
#include "../VectorBackendVkvg.hpp"
#endif

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "DistributionSimilarityChart.hpp"

DistributionSimilarityChart::DistributionSimilarityChart() {
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

DistributionSimilarityChart::~DistributionSimilarityChart() = default;

void DistributionSimilarityChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    windowWidth = (200 + borderSizeX) * 2.0f;
    windowHeight = (200 + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void DistributionSimilarityChart::onUpdatedWindowSize() {
    borderSizeX = borderSizeY = 10.0f;
    ox = borderSizeX;
    oy = borderSizeY;
    dw = windowWidth - borderSizeX * 2;
    dh = windowHeight - borderSizeY * 2;
    dw = std::max(dw, 1.0f);
    dh = std::max(dh, 1.0f);
}

void DistributionSimilarityChart::updateSizeByParent() {
    auto [parentWidth, parentHeight] = getBlitTargetSize();
    auto ssf = float(blitTargetSupersamplingFactor);
    windowOffsetX = 0;
    windowOffsetY = 0;
    windowWidth = float(parentWidth) / (scaleFactor * float(ssf));
    windowHeight = float(parentHeight) / (scaleFactor * float(ssf));
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void DistributionSimilarityChart::setAlignWithParentWindow(bool _align) {
    alignWithParentWindow = _align;
    renderBackgroundStroke = !_align;
    isWindowFixed = alignWithParentWindow;
    if (alignWithParentWindow) {
        updateSizeByParent();
    }
}

void DistributionSimilarityChart::setPointColor(const sgl::Color& _pointColor) {
    pointColor = _pointColor;
}

void DistributionSimilarityChart::setPointRadius(float _pointRadius) {
    pointRadius = _pointRadius;
}

void DistributionSimilarityChart::setPointData(const std::vector<glm::vec2>& _pointData) {
    pointData = _pointData;
}

void DistributionSimilarityChart::setClusterData(const std::vector<std::vector<size_t>> &_clusterData) {
    clusterData = _clusterData;
    pointToClusterArray.clear();
    pointToClusterArray.resize(pointData.size(), -1);
    for (size_t clusterIdx = 0; clusterIdx < clusterData.size(); clusterIdx++) {
        const std::vector<size_t>& cluster = clusterData.at(clusterIdx);
        for (size_t clusterPointIdx = 0; clusterPointIdx < cluster.size(); clusterPointIdx++) {
            size_t i = cluster.at(clusterPointIdx);
            pointToClusterArray.at(i) = int(clusterIdx);
        }
    }
}

void DistributionSimilarityChart::setBoundingBox(const sgl::AABB2& _bb) {
    bb = _bb;
}

void DistributionSimilarityChart::update(float dt) {
    DiagramBase::update(dt);

    glm::vec2 mousePosition(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePosition -= glm::vec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();

    float pointSizeReal = pointRadius * std::min(windowWidth, windowHeight) / 1024.0f;
    float minX = ox;
    float minY = oy;
    float maxX = ox + dw;
    float maxY = oy + dh;
    float minXPt = minX + pointSizeReal * 1.5f;
    float minYPt = minY + pointSizeReal * 1.5f;
    float maxXPt = maxX - pointSizeReal * 1.5f;
    float maxYPt = maxY - pointSizeReal * 1.5f;
    glm::vec2 mousePosPct = (mousePosition - glm::vec2(minXPt, minYPt)) / glm::vec2(maxXPt - minXPt, maxYPt - minYPt);
    bool isMouseInWindow = mousePosPct.x >= 0.0f && mousePosPct.y >= 0.0f && mousePosPct.x < 1.0f && mousePosPct.y < 1.0f;
    bool isMouseInWindowAll = mousePosition.x >= 0.0f && mousePosition.y >= 0.0f && mousePosition.x < windowWidth && mousePosition.y < windowHeight;

    if (!isMouseInWindow) {
        hoveredPointIdx = -1;
    }
    if (isMouseInWindow && !isMouseGrabbedByParent) {
        int closestPointIdx = -1;
        float closestPointDist = std::numeric_limits<float>::max();
        auto numPoints = int(pointData.size());
        for (int i = 0; i < numPoints; i++) {
            float x = (pointData.at(i).x - bb.min.x) / (bb.max.x - bb.min.x);
            float y = (pointData.at(i).y - bb.min.y) / (bb.max.y - bb.min.y);
            y = 1 - y;
            x = minXPt + (maxXPt - minXPt) * x;
            y = minYPt + (maxYPt - minYPt) * y;
            float distToMouse = glm::distance(glm::vec2(x, y), mousePosition);
            if (distToMouse < closestPointDist) {
                closestPointDist = distToMouse;
                closestPointIdx = i;
            }
        }
        const float minDist = 4.0f;
        if (closestPointDist >= 0 && closestPointDist > minDist) {
            closestPointIdx = -1;
        }
        hoveredPointIdx = closestPointIdx;
    }
    if (isMouseInWindowAll) {
        if ((sgl::Mouse->buttonReleased(1) || sgl::Mouse->buttonReleased(3)) && !windowMoveOrResizeJustFinished) {
            clickedPointIdx = -1;
            if (hoveredPointIdx >= 0) {
                clickedPointIdx = hoveredPointIdx;
            }
        }
    }

    int newSelectedPointIdx = -1;
    if (hoveredPointIdx >= 0) {
        newSelectedPointIdx = hoveredPointIdx;
    } else if (clickedPointIdx >= 0) {
        newSelectedPointIdx = clickedPointIdx;
    }

    if (selectedPointIdx != newSelectedPointIdx) {
        selectedPointIdx = newSelectedPointIdx;
        needsReRender = true;
    }
}

void DistributionSimilarityChart::updateData() {
    ;
}

extern const std::vector<sgl::Color> clusterColorsPredefined;
const std::vector<sgl::Color> clusterColorsPredefined = {
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

void DistributionSimilarityChart::renderScatterPlot() {
    sgl::Color textColor = isDarkMode ? textColorDark : textColorBright;
    NVGcolor textColorNvg;
    if (vg) {
        textColorNvg = nvgRGBA(textColor.getR(), textColor.getG(), textColor.getB(), 255);
    }

#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr;
    if (canvas) {
        paint = new SkPaint;
    }
#endif

    float pointSizeReal = pointRadius * std::min(windowWidth, windowHeight) / 1024.0f;
    float minX = ox;
    float minY = oy;
    float maxX = ox + dw;
    float maxY = oy + dh;
    float minXPt = minX + pointSizeReal * 1.5f;
    float minYPt = minY + pointSizeReal * 1.5f;
    float maxXPt = maxX - pointSizeReal * 1.5f;
    float maxYPt = maxY - pointSizeReal * 1.5f;

    if (vg) {
        nvgBeginPath(vg);
        nvgRect(vg, minX, minY, maxX - minX, maxY - minY);
        nvgStrokeWidth(vg, strokeWidth);
        nvgStrokeColor(vg, textColorNvg);
        nvgStroke(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setColor(toSkColor(textColor));
        paint->setStroke(true);
        paint->setStrokeWidth(strokeWidth * s);
        canvas->drawRect(SkRect{minX * s, minY * s, maxX * s, maxY * s}, *paint);
        paint->setStroke(false);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_source_color(context, textColor.getColorRGBA());
        vkvg_set_opacity(context, 1.0f);
        vkvg_rectangle(context, minX * s, minY * s, (maxX - minX) * s, (maxY - minY) * s);
        vkvg_set_line_width(context, strokeWidth * s);
        vkvg_stroke(context);
    }
#endif

    auto numPoints = int(pointData.size());
    auto pointColorDefault =
            clusterData.empty() ? pointColor : (isDarkMode ? pointColorGreyDark : pointColorGreyBright);
    for (int i = 0; i < numPoints; i++) {
        if (i == hoveredPointIdx || pointToClusterArray.at(i) >= 0) {
            continue;
        }
        float x = (pointData.at(i).x - bb.min.x) / (bb.max.x - bb.min.x);
        float y = (pointData.at(i).y - bb.min.y) / (bb.max.y - bb.min.y);
        y = 1 - y;
        x = minXPt + (maxXPt - minXPt) * x;
        y = minYPt + (maxYPt - minYPt) * y;
        if (vg) {
            nvgBeginPath(vg);
            nvgCircle(vg, x, y, pointSizeReal);
            nvgFillColor(vg, nvgRGB(pointColorDefault.getR(), pointColorDefault.getG(), pointColorDefault.getB()));
            nvgFill(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            paint->setColor(toSkColor(pointColorDefault));
            paint->setStroke(false);
            canvas->drawCircle(x * s, y * s, pointSizeReal * s, *paint);
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            vkvg_set_source_color(context, pointColorDefault.getColorRGBA());
            vkvg_arc(context, x * s, y * s, pointSizeReal * s, 0.0f, sgl::TWO_PI);
            vkvg_fill(context);
        }
#endif
    }

    if (!clusterData.empty()) {
        for (size_t clusterIdx = 0; clusterIdx < clusterData.size(); clusterIdx++) {
            const std::vector<size_t>& cluster = clusterData.at(clusterIdx);
            const sgl::Color& pointColorCluster =
                    clusterColorsPredefined.at(clusterIdx % clusterColorsPredefined.size());
            for (size_t clusterPointIdx = 0; clusterPointIdx < cluster.size(); clusterPointIdx++) {
                auto i = int(cluster.at(clusterPointIdx));
                if (i == hoveredPointIdx) {
                    continue;
                }
                float x = (pointData.at(i).x - bb.min.x) / (bb.max.x - bb.min.x);
                float y = (pointData.at(i).y - bb.min.y) / (bb.max.y - bb.min.y);
                y = 1 - y;
                x = minXPt + (maxXPt - minXPt) * x;
                y = minYPt + (maxYPt - minYPt) * y;
                if (vg) {
                    nvgBeginPath(vg);
                    nvgCircle(vg, x, y, pointSizeReal);
                    nvgFillColor(vg, nvgRGB(pointColorCluster.getR(), pointColorCluster.getG(), pointColorCluster.getB()));
                    nvgFill(vg);
                }
#ifdef SUPPORT_SKIA
                else if (canvas) {
                    paint->setColor(toSkColor(pointColorCluster));
                    paint->setStroke(false);
                    canvas->drawCircle(x * s, y * s, pointSizeReal * s, *paint);
                }
#endif
#ifdef SUPPORT_VKVG
                else if (context) {
                    vkvg_set_source_color(context, pointColorCluster.getColorRGBA());
                    vkvg_arc(context, x * s, y * s, pointSizeReal * s, 0.0f, sgl::TWO_PI);
                    vkvg_fill(context);
                }
#endif
            }
        }
    }

    if (selectedPointIdx >= 0) {
        float x = (pointData.at(selectedPointIdx).x - bb.min.x) / (bb.max.x - bb.min.x);
        float y = (pointData.at(selectedPointIdx).y - bb.min.y) / (bb.max.y - bb.min.y);
        y = 1 - y;
        x = minXPt + (maxXPt - minXPt) * x;
        y = minYPt + (maxYPt - minYPt) * y;
        if (vg) {
            nvgBeginPath(vg);
            nvgCircle(vg, x, y, pointSizeReal * 2.0f);
            nvgFillColor(vg, nvgRGB(hoveredPointColor.getR(), hoveredPointColor.getG(), hoveredPointColor.getB()));
            nvgFill(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            paint->setColor(toSkColor(hoveredPointColor));
            paint->setStroke(false);
            canvas->drawCircle(x * s, y * s, pointSizeReal * 2.0f * s, *paint);
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            vkvg_set_source_color(context, hoveredPointColor.getColorRGBA());
            vkvg_arc(context, x * s, y * s, pointSizeReal * 2.0f * s, 0.0f, sgl::TWO_PI);
            vkvg_fill(context);
        }
#endif
    }

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
    }
#endif
}

void DistributionSimilarityChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}

#ifdef SUPPORT_SKIA
void DistributionSimilarityChart::renderBaseSkia() {
    DiagramBase::renderBaseSkia();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}
#endif

#ifdef SUPPORT_VKVG
void DistributionSimilarityChart::renderBaseVkvg() {
    DiagramBase::renderBaseVkvg();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}
#endif
