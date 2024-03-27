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

#ifndef CORRERENDER_TIMESERIESCORRELATIONCHART_HPP
#define CORRERENDER_TIMESERIESCORRELATIONCHART_HPP

#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/CorrelationMatrix.hpp"
#include "Renderers/Diagram/DiagramColorMap.hpp"
#include "../DiagramBase.hpp"

class HostCacheEntryType;
typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
class VolumeData;
typedef std::shared_ptr<VolumeData> VolumeDataPtr;
class TimeSeriesRasterPass;

class TimeSeriesCorrelationChart : public DiagramBase {
public:
    TimeSeriesCorrelationChart();
    ~TimeSeriesCorrelationChart() override;
    DiagramType getDiagramType() override { return DiagramType::CORRELATION_MATRIX; }
    void initialize() override;
    void onBackendDestroyed() override;
    void update(float dt) override;
    void onWindowSizeChanged() override;
    void updateSizeByParent() override;
    void setAlignWithParentWindow(bool _align);
    void setColorMap(DiagramColorMap _colorMap);
    void setDiagramSelectionCallback(std::function<void(int series, int time)> callback);
    void setCorrelationDataBuffer(
            int _samples, int _numWindows, const sgl::vk::BufferPtr& _correlationDataBuffer);
    void onCorrelationDataRecalculated(
            CorrelationMeasureType _correlationMeasureType,
            const std::pair<float, float>& _minMaxCorrelationValue, bool _isNetworkData);
    void renderPrepare();

protected:
    bool hasData() override {
        return true;
    }
    void renderBaseNanoVG() override;
#ifdef SUPPORT_SKIA
    void renderBaseSkia() override;
#endif
#ifdef SUPPORT_VKVG
    void renderBaseVkvg() override;
#endif

    void renderTimeSeries();
    void onUpdatedWindowSize() override;

private:
    bool dataDirty = true;
    void updateData();

    // GUI data.
    bool alignWithParentWindow = false;
    float ox = 0, oy = 0, dw = 0, dh = 0;

    // Scrolling and zooming.
    void drawScrollBar();
    void recomputeScrollThumbHeight();
    bool useScrollBar = false;
    bool scrollThumbHover = false;
    bool scrollThumbDrag = false;
    float scrollBarWidth = 10.0f;
    float scrollThumbPosition = 0.0f;
    float scrollThumbHeight = 0.0f;
    float scrollTranslationY = 0.0f;
    float thumbDragDelta = 0.0f;
    float zoomFactorVertical = 1.0f;
    float zoomFactor = 1.0f;

    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;
    std::pair<float, float> minMaxCorrelationValue;
    bool isNetworkData = false;

    int samples = 0;
    int numWindows = 0;
    std::function<void(int series, int time)> diagramSelectionCallback;

    sgl::vk::BufferPtr correlationDataBuffer;
#ifdef SUPPORT_OPENGL
    sgl::TexturePtr correlationTextureGl;
#endif
    sgl::vk::TexturePtr correlationTextureVk;
    sgl::vk::ImageViewPtr correlationImageViewVk;
    bool imageHandleDirty = false;
    int imageHandleNvg = -1;
    std::shared_ptr<TimeSeriesRasterPass> timeSeriesRasterPass;

    // Color legend.
    glm::vec4 evalColorMapVec4(float t);
    sgl::Color evalColorMap(float t);
    void drawColorLegends();
    void drawColorLegend(
            float x, float y, float w, float h, int numLabels, int numTicks,
            const std::function<std::string(float)>& labelMap, const std::function<glm::vec4(float)>& colorMap,
            const std::string& textTop);
    void computeColorLegendHeight();
    bool shallDrawColorLegend = true;
    sgl::Color textColorDark = sgl::Color(255, 255, 255, 255);
    sgl::Color textColorBright = sgl::Color(0, 0, 0, 255);
    float colorLegendWidth = 20.0f;
    float colorLegendHeight = 20.0f;
    float maxColorLegendHeight = 200.0f;
    const float textWidthMaxBase = 32;
    float textWidthMax = 32;
    DiagramColorMap colorMap = DiagramColorMap::COOL_TO_WARM;
    std::vector<glm::vec3> colorPoints;
    bool colorMapChanged = true;
};

#endif //CORRERENDER_TIMESERIESCORRELATIONCHART_HPP
