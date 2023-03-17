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
#include "Sampling.hpp"
#include "DiagramButton.hpp"
#include "DiagramBase.hpp"
#include "../../Calculators/CorrelationDefines.hpp"

typedef std::shared_ptr<float[]> HostCacheEntry;
class CorrelationComputePass;
class HEBChart;

struct MIFieldEntry {
    float correlationValue;
    uint32_t pointIndex0, pointIndex1;

    MIFieldEntry(float correlationValue, uint32_t pointIndex0, uint32_t pointIndex1)
            : correlationValue(correlationValue), pointIndex0(pointIndex0), pointIndex1(pointIndex1) {}
    bool operator<(const MIFieldEntry& rhs) const { return correlationValue > rhs.correlationValue; }
};

struct HEBChartFieldUpdateData {
    std::vector<glm::vec2> curvePoints;
    std::vector<float> correlationValuesArray;
    std::vector<std::pair<int, int>> connectedPointsArray;
};

struct HEBChartFieldData {
    explicit HEBChartFieldData(HEBChart* parent) : parent(parent) {}

    // General data.
    HEBChart* parent = nullptr;
    int selectedFieldIdx = 0;
    std::string selectedScalarFieldName;

    // Line data (lines themselves are stored in diagram class).
    float minCorrelationValue = 0.0f, maxCorrelationValue = 0.0f;

    // Outer ring.
    std::vector<float> leafStdDevArray;
    float minStdDev = 0.0f, maxStdDev = 0.0f;
    // Angle ranges for !regionsEqual.
    float a00 = 0.0f, a01 = 0.0f, a10 = 0.0f, a11 = 0.0f;

    // Transfer function.
    void setColorMap(DiagramColorMap _colorMap); //< Only lines if separateColorVarianceAndCorrelation.
    void setColorMapVariance(DiagramColorMap _colorMap);
    void initializeColorPoints();
    void initializeColorPointsVariance();
    glm::vec4 evalColorMapVec4(float t);
    sgl::Color evalColorMap(float t);
    glm::vec4 evalColorMapVec4Variance(float t, bool saturated);
    sgl::Color evalColorMapVariance(float t, bool saturated);
    DiagramColorMap colorMapLines = DiagramColorMap::CIVIDIS, colorMapVariance = DiagramColorMap::CIVIDIS;
    std::vector<glm::vec3> colorPointsLines, colorPointsVariance;
    bool separateColorVarianceAndCorrelation = true;
};
typedef std::shared_ptr<HEBChartFieldData> HEBChartFieldDataPtr;

/**
 * Hierarchical edge bundling chart. For more details see:
 * - "Hierarchical Edge Bundles: Visualization of Adjacency Relations in Hierarchical Data" (Danny Holten, 2006).
 *   Link: https://ieeexplore.ieee.org/document/4015425
 * - https://r-graph-gallery.com/hierarchical-edge-bundling.html
 */
class HEBChart : public DiagramBase {
    friend struct HEBChartFieldData;
public:
    HEBChart();
    ~HEBChart();
    DiagramType getDiagramType() override { return DiagramType::HEB_CHART; }
    void initialize() override;
    void update(float dt) override;
    void updateSizeByParent() override;
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setIsFocusView(int focusViewLevel);
    void setRegions(const std::pair<GridRegion, GridRegion>& _rs);
    void clearScalarFields();
    void addScalarField(int selectedFieldIdx, const std::string& _scalarFieldName);
    void removeScalarField(int selectedFieldIdx, bool doNotShiftIndicesBack);
    void setColorMap(int fieldIdx, DiagramColorMap _colorMap); //< Only lines if separateColorVarianceAndCorrelation.
    void setColorMapVariance(DiagramColorMap _colorMap);
    void setIsEnsembleMode(bool _isEnsembleMode);
    void setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType);
    void setSamplingMethodType(SamplingMethodType _samplingMethodType);
    void setNumSamples(int _numSamples);
    void setBeta(float _beta);
    void setDownscalingFactors(int _dfx, int _dfy, int _dfz);
    void setLineCountFactor(int _factor);
    void setCurveThickness(float _curveThickness);
    void setCurveOpacity(float _alpha);
    void setDiagramRadius(int radius);
    void setOuterRingSizePercentage(float pct);
    void setAlignWithParentWindow(bool _align);
    void setOpacityByValue(bool _opacityByValue);
    void setColorByValue(bool _colorByValue);
    void setUseSeparateColorVarianceAndCorrelation(bool _separate);
    void setShowSelectedRegionsByColor(bool _show);
    void setUse2DField(bool _use2dField);
    void setUseCorrelationComputationGpu(bool _useGpu);

    // Selection query.
    void resetSelectedPrimitives();
    bool getIsRegionSelected(int idx);
    uint32_t getPointIndexGrid(int pointIdx);
    uint32_t getSelectedPointIndexGrid(int idx);
    sgl::AABB3 getSelectedRegion(int idx);
    std::pair<glm::vec3, glm::vec3> getLinePositions();
    glm::vec3 getLineDirection();
    [[nodiscard]] inline bool getShowSelectedRegionsByColor() const { return showSelectedRegionsByColor; }
    [[nodiscard]] inline const sgl::Color& getColorSelected0() const { return circleFillColorSelected0; }
    [[nodiscard]] inline const sgl::Color& getColorSelected1() const { return circleFillColorSelected1; }
    [[nodiscard]] inline const sgl::Color& getColorSelected(int idx) const {
        return idx == 0 ? circleFillColorSelected0 : circleFillColorSelected1;
    }

    // Queries for showing children.
    void resetFocusSelection();
    bool getHasNewFocusSelection(bool& isDeselection);
    std::pair<GridRegion, GridRegion> getFocusSelection();
    GridRegion getGridRegionPointIdx(int idx, uint32_t pointIdx);
    int getLeafIdxGroup(int leafIdx);

    // Range queries.
    glm::vec2 getCorrelationRangeTotal();
    glm::ivec2 getCellDistanceRangeTotal();
    void setCorrelationRange(const glm::vec2& _range);
    void setCellDistanceRange(const glm::ivec2& _range);

    // Focus+Context state queries.
    [[nodiscard]] inline int getReturnToViewIdx() { int idx = returnToViewIdx; returnToViewIdx = -1; return idx; }

    /// Returns whether ensemble or time correlation mode is used.
    [[nodiscard]] inline bool getIsEnsembleMode() const { return isEnsembleMode; }

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
    VolumeDataPtr volumeData;
    bool isFocusView = false;
    bool dataDirty = true;

    int getCorrelationMemberCount();
    HostCacheEntry getFieldEntryCpu(const std::string& fieldName, int fieldIdx);
    DeviceCacheEntry getFieldEntryDevice(const std::string& fieldName, int fieldIdx);
    std::pair<float, float> getMinMaxScalarFieldValue(const std::string& fieldName, int fieldIdx);
    bool isEnsembleMode = true; //< Ensemble or time mode?

    // Hierarchy data.
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    SamplingMethodType samplingMethodType = SamplingMethodType::MEAN;
    int numSamples = 100;
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

    // B-spline data.
    void updateData();
    void computeDownscaledField(
            HEBChartFieldData* fieldData, int idx, std::vector<float*>& downscaledFields);
    void computeDownscaledFieldVariance(HEBChartFieldData* fieldData, int idx);
    void computeCorrelations(
            HEBChartFieldData* fieldData,
            std::vector<float*>& downscaledFields0, std::vector<float*>& downscaledFields1,
            std::vector<MIFieldEntry>& miFieldEntries);
    void computeCorrelationsMean(
            HEBChartFieldData* fieldData,
            std::vector<float*>& downscaledFields0, std::vector<float*>& downscaledFields1,
            std::vector<MIFieldEntry>& miFieldEntries);
    void computeCorrelationsSamplingCpu(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries);
    void computeCorrelationsSamplingGpu(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries);
    int numLinesTotal = 0;
    int MAX_NUM_LINES = 100;
    const int NUM_SUBDIVISIONS = 50;
    float beta = 0.75f;
    float curveThickness = 1.5f;
    float curveOpacity = 0.4f;
    glm::vec2 correlationRange{}, correlationRangeTotal{};
    glm::ivec2 cellDistanceRange{}, cellDistanceRangeTotal{};

    // GPU computations.
    bool useCorrelationComputationGpu = true;
    bool supportsAsyncCompute = true;
    sgl::vk::Renderer* computeRenderer = nullptr;
    sgl::vk::BufferPtr requestsBuffer{}, requestsStagingBuffer{};
    sgl::vk::BufferPtr correlationOutputBuffer{}, correlationOutputStagingBuffer{};
    sgl::vk::FencePtr fence{};
    VkCommandPool commandPool{};
    VkCommandBuffer commandBuffer{};
    std::shared_ptr<CorrelationComputePass> correlationComputePass;

    // Field data.
    std::vector<HEBChartFieldDataPtr> fieldDataArray;
    // Lines (stored separately from field data, as lines should be rendered from lowest to highest value).
    std::vector<glm::vec2> curvePoints;
    std::vector<float> correlationValuesArray; ///< per-line values.
    std::vector<std::pair<int, int>> connectedPointsArray; ///< points connected by lines.
    std::vector<int> lineFieldIndexArray;
    float minCorrelationValueGlobal = 0.0f, maxCorrelationValueGlobal = 0.0f;
    DiagramColorMap colorMapVariance = DiagramColorMap::VIRIDIS;

    // GUI data.
    float chartRadius{};
    float totalRadius{};
    int hoveredPointIdx = -1;
    int hoveredLineIdx = -1;
    int clickedPointIdx = -1;
    int clickedLineIdx = -1;
    int selectedPointIndices[2] = { -1, -1 };
    int selectedLineIdx = -1;
    int clickedPointIdxOld = -1, clickedLineIdxOld = -1; //< For getHasNewFocusSelection.
    bool isFocusSelectionReset = false;
    float pointRadiusBase = 1.5f;
    sgl::Color circleFillColor = sgl::Color(180, 180, 180, 255);
    sgl::Color circleFillColorSelected0 = sgl::Color(180, 80, 80, 255);
    sgl::Color circleFillColorSelected1 = sgl::Color(50, 100, 180, 255);
    sgl::Color circleStrokeColorDark = sgl::Color(255, 255, 255, 255);
    sgl::Color circleStrokeColorBright = sgl::Color(0, 0, 0, 255);
    bool alignWithParentWindow = false;
    bool opacityByValue = false;
    bool colorByValue = true;
    bool showSelectedRegionsByColor = true;
    bool separateColorVarianceAndCorrelation = true;

    // Outer ring.
    bool showRing = true;
    float outerRingOffset = 3.0f;
    float outerRingWidth = 0.0f; //< Determined automatically.
    float outerRingSizePct = 0.1f;

    // Color legend.
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
    bool arrangeColorLegendsBothSides = false;
    float colorLegendWidth = 20.0f;
    float colorLegendHeight = 20.0f;
    float maxColorLegendHeight = 200.0f;
    float colorLegendCircleDist = 5.0f;
    const float colorLegendSpacing = 4.0f;
    const float textWidthMaxBase = 32;
    float textWidthMax = 32;

    // Buttons for closing the window/going back by one view.
    std::vector<DiagramButton> buttons;
    int returnToViewIdx = -1;
};

#endif //CORRERENDER_HEBCHART_HPP
