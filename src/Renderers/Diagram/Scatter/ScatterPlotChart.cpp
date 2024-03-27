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

#include <Math/Math.hpp>
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
#include "ScatterPlotChart.hpp"

ScatterPlotChart::ScatterPlotChart() {
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

ScatterPlotChart::~ScatterPlotChart() {
}

void ScatterPlotChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    windowWidth = (200 + borderSizeX) * 2.0f;
    windowHeight = (200 + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void ScatterPlotChart::onUpdatedWindowSize() {
    if (windowWidth < 360.0f || windowHeight < 360.0f) {
        borderSizeX = borderSizeY = 10.0f;
    } else {
        borderSizeX = borderSizeY = std::min(windowWidth, windowHeight) / 36.0f;
    }
    float minDim = std::min(windowWidth - 2.0f * borderSizeX, windowHeight - 2.0f * borderSizeY);
    totalRadius = std::round(0.5f * minDim);
}

void ScatterPlotChart::updateSizeByParent() {
    auto [parentWidth, parentHeight] = getBlitTargetSize();
    auto ssf = float(blitTargetSupersamplingFactor);
    windowOffsetX = 0;
    windowOffsetY = 0;
    windowWidth = float(parentWidth) / (scaleFactor * float(ssf));
    windowHeight = float(parentHeight) / (scaleFactor * float(ssf));
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void ScatterPlotChart::setAlignWithParentWindow(bool _align) {
    alignWithParentWindow = _align;
    renderBackgroundStroke = !_align;
    isWindowFixed = alignWithParentWindow;
    if (alignWithParentWindow) {
        updateSizeByParent();
    }
}

void ScatterPlotChart::update(float dt) {
    DiagramBase::update(dt);
}

void ScatterPlotChart::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;
    auto numFieldsBase = int(volumeData->getFieldNamesBase(FieldType::SCALAR).size());
    if (_volumeData->getCurrentTimeStepIdx() != cachedTimeStepIdx
            || _volumeData->getCurrentEnsembleIdx() != cachedEnsembleIdx
            || fieldIdx0 >= numFieldsBase || fieldIdx1 >= numFieldsBase) {
        cachedTimeStepIdx = _volumeData->getCurrentTimeStepIdx();
        cachedEnsembleIdx = _volumeData->getCurrentEnsembleIdx();
        dataDirty = true;
    }
    if (isNewData) {
        isMinMaxDirty = true;
    }
}

void ScatterPlotChart::setIsEnsembleMode(bool _isEnsembleMode) {
    if (isEnsembleMode != _isEnsembleMode) {
        isEnsembleMode = _isEnsembleMode;
        dataDirty = true;
    }
}

void ScatterPlotChart::setField0(int _fieldIdx0, const std::string& _fieldName0) {
    if (fieldIdx0 != _fieldIdx0 || fieldName0 != _fieldName0) {
        fieldIdx0 = _fieldIdx0;
        fieldName0 = _fieldName0;
        dataDirty = true;
        isMinMaxDirty = true;
    }
}

void ScatterPlotChart::setField1(int _fieldIdx1, const std::string& _fieldName1) {
    if (fieldIdx1 != _fieldIdx1 || fieldName1 != _fieldName1) {
        fieldIdx1 = _fieldIdx1;
        fieldName1 = _fieldName1;
        dataDirty = true;
        isMinMaxDirty = true;
    }
}

void ScatterPlotChart::setReferencePoints(const glm::ivec3& pt0, const glm::ivec3& pt1) {
    if (refPoint0 != pt0 || refPoint1 != pt1) {
        refPoint0 = pt0;
        refPoint1 = pt1;
        dataDirty = true;
    }
}

void ScatterPlotChart::setPointColor(const sgl::Color& _pointColor) {
    pointColor = _pointColor;
}

void ScatterPlotChart::setPointRadius(float _pointRadius) {
    pointRadius = _pointRadius;
}

void ScatterPlotChart::setUseGlobalMinMax(bool _useGlobalMinMax) {
    if (useGlobalMinMax != _useGlobalMinMax) {
        useGlobalMinMax = _useGlobalMinMax;
        dataDirty = true;
        isMinMaxDirty = true;
    }
}

int ScatterPlotChart::getCorrelationMemberCount() {
    return isEnsembleMode ? volumeData->getEnsembleMemberCount() : volumeData->getTimeStepCount();
}

VolumeData::HostCacheEntry ScatterPlotChart::getFieldEntryCpu(const std::string& fieldName, int fieldIdx) {
    VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, fieldName, isEnsembleMode ? -1 : fieldIdx, isEnsembleMode ? fieldIdx : -1);
    return ensembleEntryField;
}

std::pair<float, float> ScatterPlotChart::getMinMaxScalarFieldValue(const std::string& fieldName, int fieldIdx) {
    return volumeData->getMinMaxScalarFieldValue(
            fieldName, isEnsembleMode ? -1 : fieldIdx, isEnsembleMode ? fieldIdx : -1);
}

void ScatterPlotChart::updateData() {
    if (!useGlobalMinMax || (useGlobalMinMax && isMinMaxDirty)) {
        minMax0 = std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
        minMax1 = std::make_pair(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
    }

    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int cs = getCorrelationMemberCount();
    values0.resize(cs);
    values1.resize(cs);
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry0 = getFieldEntryCpu(fieldName0, fieldIdx);
        const float* field0 = fieldEntry0->data<float>();
        float val0 = field0[IDXS(refPoint0.x, refPoint0.y, refPoint0.z)];
        values0.at(fieldIdx) = val0;
        VolumeData::HostCacheEntry fieldEntry1 = getFieldEntryCpu(fieldName1, fieldIdx);
        const float* field1 = fieldEntry1->data<float>();
        float val1 = field1[IDXS(refPoint1.x, refPoint1.y, refPoint1.z)];
        values1.at(fieldIdx) = val1;

        if (!useGlobalMinMax) {
            minMax0 = std::make_pair(std::min(minMax0.first, val0), std::max(minMax0.second, val0));
            minMax1 = std::make_pair(std::min(minMax1.first, val1), std::max(minMax1.second, val1));
        }
    }

    if (useGlobalMinMax && isMinMaxDirty) {
        isMinMaxDirty = false;
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            auto minMax = volumeData->getMinMaxScalarFieldValue(fieldName0, fieldIdx);
            minMax0 = std::make_pair(std::min(minMax0.first, minMax.first), std::max(minMax0.second, minMax.second));
        }
        if (fieldIdx0 == fieldIdx1) {
            minMax1 = minMax0;
        } else {
            for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                auto minMax = volumeData->getMinMaxScalarFieldValue(fieldName1, fieldIdx);
                minMax1 = std::make_pair(std::min(minMax1.first, minMax.first), std::max(minMax1.second, minMax.second));
            }
        }
    }

    float diff0 = minMax0.second - minMax0.first;
    float diff1 = minMax1.second - minMax1.first;
    minMaxOff0 = std::make_pair(minMax0.first - offsetPct * diff0, minMax0.second + offsetPct * diff0);
    minMaxOff1 = std::make_pair(minMax1.first - offsetPct * diff1, minMax1.second + offsetPct * diff1);
}

void ScatterPlotChart::renderScatterPlot() {
    sgl::Color textColor = isDarkMode ? textColorDark : textColorBright;
    NVGcolor textColorNvg;
    if (vg) {
        textColorNvg = nvgRGBA(textColor.getR(), textColor.getG(), textColor.getB(), 255);
    }

    float pointSizeReal = pointRadius * std::min(windowWidth, windowHeight) / 1024.0f;
    int cs = getCorrelationMemberCount();
    float minX = windowWidth / 2 - totalRadius;
    float minY = windowHeight / 2 - totalRadius;
    float maxX = windowWidth / 2 + totalRadius;
    float maxY = windowHeight / 2 + totalRadius;

#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr;
    if (canvas) {
        paint = new SkPaint;
    }
#endif

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

    for (int i = 0; i < cs; i++) {
        float x = (values0.at(i) - minMaxOff0.first) / (minMaxOff0.second - minMaxOff0.first);
        float y = (values1.at(i) - minMaxOff1.first) / (minMaxOff1.second - minMaxOff1.first);
        y = 1 - y;
        x = minX + (maxX - minX) * x;
        y = minY + (maxY - minY) * y;
        if (vg) {
            nvgBeginPath(vg);
            nvgCircle(vg, x, y, pointSizeReal);
            nvgFillColor(vg, nvgRGB(pointColor.getR(), pointColor.getG(), pointColor.getB()));
            nvgFill(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            paint->setColor(toSkColor(pointColor));
            paint->setStroke(false);
            canvas->drawCircle(x * s, y * s, pointSizeReal * s, *paint);
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            vkvg_set_source_color(context, pointColor.getColorRGBA());
            vkvg_arc(context, x * s, y * s, pointSizeReal * s, 0.0f, sgl::TWO_PI);
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

void ScatterPlotChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}

#ifdef SUPPORT_SKIA
void ScatterPlotChart::renderBaseSkia() {
    DiagramBase::renderBaseSkia();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}
#endif

#ifdef SUPPORT_VKVG
void ScatterPlotChart::renderBaseVkvg() {
    DiagramBase::renderBaseVkvg();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}
#endif
