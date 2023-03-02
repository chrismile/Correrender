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

#ifndef CORRERENDER_HEBCHART_HPP
#define CORRERENDER_HEBCHART_HPP

#include <string>
#include <vector>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "DiagramBase.hpp"
#include "../../Calculators/Similarity.hpp"

struct HEBNode {
    HEBNode() {
        parentIdx = std::numeric_limits<uint32_t>::max();
        std::fill_n(childIndices, 8, std::numeric_limits<uint32_t>::max());
    }
    explicit HEBNode(uint32_t parentIdx) : parentIdx(parentIdx) {
        std::fill_n(childIndices, 8, std::numeric_limits<uint32_t>::max());
    }
    glm::vec2 normalizedPosition;
    float angle = 0.0f;
    uint32_t parentIdx;
    uint32_t childIndices[8];
};

/**
 * Hierarchical edge bundling chart. For more details see:
 * - "Hierarchical Edge Bundles: Visualization of Adjacency Relations in Hierarchical Data" (Danny Holten, 2006).
 *   Link: https://ieeexplore.ieee.org/document/4015425
 * - https://r-graph-gallery.com/hierarchical-edge-bundling.html
 */
class HEBChart : public DiagramBase {
public:
    HEBChart();
    DiagramType getDiagramType() override { return DiagramType::HEB_CHART; }
    void initialize() override;
    void update(float dt) override;
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setSelectedScalarField(int selectedFieldIdx, const std::string& _scalarFieldName);
    void setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType);
    void setBeta(float _beta);
    void setDownscalingFactor(int _df);
    void setLineCountFactor(int _factor);
    void setCurveOpacity(float _alpha);
    void setCellDistanceThreshold(int _thresh);
    void setDiagramRadius(int radius);
    void setOpacityByValue(bool _opacityByValue);
    void setColorByValue(bool _colorByValue);
    void setUse2DField(bool _use2dField);

    bool getIsRegionSelected(int idx);
    uint32_t getSelectedPointIndexGrid(int idx);
    sgl::AABB3 getSelectedRegion(int idx);
    std::pair<glm::vec3, glm::vec3> getLinePositions();

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

    void onUpdatedWindowSize() override;

private:
    float chartRadius{};

    // Selected scalar field.
    VolumeDataPtr volumeData;
    int selectedFieldIdx = 0;
    std::string selectedScalarFieldName;
    bool dataDirty = true;

    // Hierarchy data.
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    int df = 32; ///< Downscaling factor.
    int xs = 0, ys = 0, zs = 0;
    int xsd = 0, ysd = 0, zsd = 0;
    bool use2dField = true;
    std::vector<HEBNode> nodesList;
    std::vector<uint32_t> pointToNodeIndexMap;
    uint32_t leafIdxOffset = 0;

    // B-spline data.
    void updateData();
    int NUM_LINES = 0;
    int MAX_NUM_LINES = 100;
    const int NUM_SUBDIVISIONS = 50;
    float beta = 0.75f;
    float curveOpacity = 0.4f;
    int cellDistanceThreshold = 0;
    std::vector<glm::vec2> curvePoints;
    std::vector<float> miValues; ///< per-line values.
    std::vector<std::pair<int, int>> connectedPointsArray; ///< points connected by lines.

    // GUI data.
    float pointRadiusBase = 1.5f;
    int hoveredPointIdx = -1;
    int hoveredLineIdx = -1;
    int clickedPointIdx = -1;
    int clickedLineIdx = -1;
    int selectedPointIndices[2] = { -1, -1 };
    int selectedLineIdx = -1;
    sgl::Color circleFillColor = sgl::Color(180, 180, 180, 255);
    sgl::Color circleFillColorSelected = sgl::Color(180, 80, 80, 255);
    bool opacityByValue = false;
    bool colorByValue = true;
    glm::vec4 evalColorMapVec4(float t);
    sgl::Color evalColorMap(float t);
};

#endif //CORRERENDER_HEBCHART_HPP
