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

#include "DiagramColorMap.hpp"
#include "Region.hpp"
#include "Octree.hpp"
#include "DiagramBase.hpp"
#include "../../Calculators/Similarity.hpp"

struct MIFieldEntry {
    float miValue;
    uint32_t pointIndex0, pointIndex1;

    MIFieldEntry(float miValue, uint32_t pointIndex0, uint32_t pointIndex1)
            : miValue(miValue), pointIndex0(pointIndex0), pointIndex1(pointIndex1) {}
    bool operator<(const MIFieldEntry& rhs) const { return miValue > rhs.miValue; }
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
    void updateSizeByParent();
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setRegions(const std::pair<GridRegion, GridRegion>& _rs);
    void setSelectedScalarField(int selectedFieldIdx, const std::string& _scalarFieldName);
    void setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType);
    void setBeta(float _beta);
    void setDownscalingFactors(int _dfx, int _dfy, int _dfz);
    void setLineCountFactor(int _factor);
    void setCurveOpacity(float _alpha);
    void setDiagramRadius(int radius);
    void setAlignWithParentWindow(bool _align);
    void setOpacityByValue(bool _opacityByValue);
    void setColorByValue(bool _colorByValue);
    void setColorMap(DiagramColorMap _colorMap);
    void setUse2DField(bool _use2dField);

    // Selection query.
    bool getIsRegionSelected(int idx);
    uint32_t getPointIndexGrid(int pointIdx);
    uint32_t getSelectedPointIndexGrid(int idx);
    sgl::AABB3 getSelectedRegion(int idx);
    std::pair<glm::vec3, glm::vec3> getLinePositions();

    // Queries for showing children.
    bool getHasNewFocusSelection(bool& isDeselection);
    std::pair<GridRegion, GridRegion> getFocusSelection();
    GridRegion getGridRegionPointIdx(int idx, uint32_t pointIdx);
    int getLeafIdxGroup(int leafIdx);

    // Range queries.
    glm::vec2 getCorrelationRangeTotal();
    glm::ivec2 getCellDistanceRangeTotal();
    void setCorrelationRange(const glm::vec2& _range);
    void setCellDistanceRange(const glm::ivec2& _range);

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
    int dfx = 32, dfy = 32, dfz = 32; ///< Downscaling factors.
    int xs = 0, ys = 0, zs = 0; //< Grid size.
    int xsd0 = 0, ysd0 = 0, zsd0 = 0; //< Downscaled grid size.
    int xsd1 = 0, ysd1 = 0, zsd1 = 0; //< Downscaled grid size.
    GridRegion r0{}, r1{};
    bool regionsEqual = true;
    bool use2dField = true;
    std::vector<HEBNode> nodesList;
    std::vector<uint32_t> pointToNodeIndexMap0, pointToNodeIndexMap1;
    uint32_t leafIdxOffset = 0, leafIdxOffset1 = 0;
    std::vector<float> leafStdDevArray;

    // B-spline data.
    void updateData();
    void computeDownscaledField(int idx, std::vector<float*>& downscaledEnsembleFields);
    void computeDownscaledFieldVariance(int idx, std::vector<float*>& downscaledEnsembleFields);
    void computeCorrelations(
            std::vector<float*>& downscaledEnsembleFields0,
            std::vector<float*>& downscaledEnsembleFields1,
            std::vector<MIFieldEntry>& miFieldEntries);
    int NUM_LINES = 0;
    int MAX_NUM_LINES = 100;
    const int NUM_SUBDIVISIONS = 50;
    float beta = 0.75f;
    float curveOpacity = 0.4f;
    glm::vec2 correlationRange{}, correlationRangeTotal{};
    glm::ivec2 cellDistanceRange{}, cellDistanceRangeTotal{};
    std::vector<glm::vec2> curvePoints;
    std::vector<float> miValues; ///< per-line values.
    std::vector<std::pair<int, int>> connectedPointsArray; ///< points connected by lines.

    // GUI data.
    void resetSelectedPrimitives();
    int hoveredPointIdx = -1;
    int hoveredLineIdx = -1;
    int clickedPointIdx = -1;
    int clickedLineIdx = -1;
    int selectedPointIndices[2] = { -1, -1 };
    int selectedLineIdx = -1;
    int clickedPointIdxOld = -1, clickedLineIdxOld = -1; //< For getHasNewFocusSelection.
    float pointRadiusBase = 1.5f;
    sgl::Color circleFillColor = sgl::Color(180, 180, 180, 255);
    sgl::Color circleFillColorSelected = sgl::Color(180, 80, 80, 255);
    bool alignWithParentWindow = false;
    bool opacityByValue = false;
    bool colorByValue = true;
    DiagramColorMap colorMap = DiagramColorMap::CIVIDIS;
    std::vector<glm::vec3> colorPoints;
    void initializeColorPoints();
    glm::vec4 evalColorMapVec4(float t);
    sgl::Color evalColorMap(float t);

    // Outer ring.
    bool showRing = true;
    float outerRingOffset = 3.0f;
    const float outerRingWidth = 20.0f;
    float minStdDev = 0.0f, maxStdDev = 0.0f;
};

#endif //CORRERENDER_HEBCHART_HPP
