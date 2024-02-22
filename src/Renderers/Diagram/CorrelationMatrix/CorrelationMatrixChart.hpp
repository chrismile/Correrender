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

#ifndef CORRERENDER_CORRELATIONMATRIXCHART_HPP
#define CORRERENDER_CORRELATIONMATRIXCHART_HPP

#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/CorrelationMatrix.hpp"
#include "Renderers/Diagram/DiagramColorMap.hpp"
#include "../DiagramBase.hpp"

class HostCacheEntryType;
typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
class VolumeData;
typedef std::shared_ptr<VolumeData> VolumeDataPtr;

class CorrelationMatrixChart : public DiagramBase {
public:
    CorrelationMatrixChart();
    ~CorrelationMatrixChart() override;
    DiagramType getDiagramType() override { return DiagramType::CORRELATION_MATRIX; }
    void initialize() override;
    void update(float dt) override;
    void updateSizeByParent() override;
    void setAlignWithParentWindow(bool _align);
    void setColorMap(DiagramColorMap _colorMap);
    void setMatrixData(
            CorrelationMeasureType _correlationMeasureType,
            const std::vector<std::string>& _fieldNames,
            const std::shared_ptr<CorrelationMatrix>& _similarityMatrix,
            const std::pair<float, float>& _minMaxCorrelationValue);

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

    void renderScatterPlot();
    void onUpdatedWindowSize() override;

private:
    bool dataDirty = true;
    void updateData();

    // GUI data.
    bool alignWithParentWindow = false;
    float totalRadius = 0.0f;

    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;
    std::vector<std::string> fieldNames;
    std::shared_ptr<CorrelationMatrix> correlationMatrix;
    std::pair<float, float> minMaxCorrelationValue;

    // Color legend.
    glm::vec4 evalColorMapVec4(float t);
    sgl::Color evalColorMap(float t);
    void drawColorLegends();
    void drawColorLegend(
            float x, float y, float w, float h, int numLabels, int numTicks,
            const std::function<std::string(float)>& labelMap, const std::function<glm::vec4(float)>& colorMap,
            const std::string& textTop);
    float computeColorLegendHeightForNumFields(int numFields, float maxHeight);
    void computeColorLegendHeight();
    bool shallDrawColorLegend = true;
    sgl::Color textColorDark = sgl::Color(255, 255, 255, 255);
    sgl::Color textColorBright = sgl::Color(0, 0, 0, 255);
    float colorLegendWidth = 20.0f;
    float colorLegendHeight = 20.0f;
    float maxColorLegendHeight = 200.0f;
    float colorLegendCircleDist = 5.0f;
    const float colorLegendSpacing = 4.0f;
    const float textWidthMaxBase = 32;
    float textWidthMax = 32;
    DiagramColorMap colorMap = DiagramColorMap::VIRIDIS;
    std::vector<glm::vec3> colorPoints;
};

#endif //CORRERENDER_CORRELATIONMATRIXCHART_HPP
