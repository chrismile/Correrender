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
#include <optional>
#include <mutex>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <nlopt.hpp>

#include <Math/Geometry/AABB3.hpp>

#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/CorrelationMatrix.hpp"
#include "DiagramColorMap.hpp"
#include "Region.hpp"
#include "Octree.hpp"
#include "Sampling.hpp"
#include "DiagramButton.hpp"
#include "DiagramBase.hpp"

namespace sgl { namespace vk {
class Fence;
typedef std::shared_ptr<Fence> FencePtr;
}}

class HostCacheEntryType;
typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
class DeviceCacheEntryType;
typedef std::shared_ptr<DeviceCacheEntryType> DeviceCacheEntry;
class VolumeData;
typedef std::shared_ptr<VolumeData> VolumeDataPtr;
class CorrelationComputePass;
class HEBChart;
class MultivariateGaussian;

struct MIFieldEntry {
    float correlationValue{};
    uint32_t pointIndex0{}, pointIndex1{};
    bool isSecondField{};

    MIFieldEntry() = default;
    MIFieldEntry(float correlationValue, uint32_t pointIndex0, uint32_t pointIndex1)
            : correlationValue(correlationValue), pointIndex0(pointIndex0), pointIndex1(pointIndex1), isSecondField(false) {}
    MIFieldEntry(float correlationValue, uint32_t pointIndex0, uint32_t pointIndex1, bool isSecondField)
            : correlationValue(correlationValue), pointIndex0(pointIndex0), pointIndex1(pointIndex1), isSecondField(isSecondField) {}
    bool operator<(const MIFieldEntry& rhs) const { return correlationValue > rhs.correlationValue; }
};

struct HEBChartFieldUpdateData {
    std::vector<glm::vec2> curvePoints;
    std::vector<float> correlationValuesArray;
    std::vector<std::pair<int, int>> connectedPointsArray;
};

struct HEBChartFieldData {
    explicit HEBChartFieldData(HEBChart* parent, bool* desaturateUnselectedRing)
            : parent(parent), desaturateUnselectedRing(desaturateUnselectedRing) {}
    ~HEBChartFieldData();

    // General data.
    HEBChart* parent = nullptr;
    int selectedFieldIdx = 0;
    std::string selectedScalarFieldName;

    // Line data (lines themselves are stored in diagram class).
    float minCorrelationValue = 0.0f, maxCorrelationValue = 0.0f;

    // Outer ring.
    std::vector<float> leafStdDevArray;
    float minStdDev = std::numeric_limits<float>::max(), maxStdDev = std::numeric_limits<float>::lowest();
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
    DiagramColorMap colorMapLines = DiagramColorMap::WISTIA, colorMapVariance = DiagramColorMap::GRAY;
    std::vector<glm::vec3> colorPointsLines, colorPointsVariance;
    bool* desaturateUnselectedRing = nullptr;
    bool separateColorVarianceAndCorrelation = true;

    // Mean field cache.
    void createFieldCache(
            VolumeData* _volumeData, bool regionsEqual, GridRegion r0, GridRegion r1, int mdfx, int mdfy, int mdfz,
            bool isEnsembleMode, CorrelationDataMode dataMode, bool useBufferTiling);
    void clearMemoryTokens();
    void computeDownscaledFields(int idx, int varNum, int fieldIdx);
    int getCorrelationMemberCount();
    HostCacheEntry getFieldEntryCpu(const std::string& fieldName, int fieldIdx);
    std::pair<float, float> getMinMaxScalarFieldValue(const std::string& fieldName, int fieldIdx);
    std::mutex volumeDataMutex;
    VolumeData* volumeData = nullptr;
    uint64_t elapsedTimeDownsampling = 0;
    bool cachedRegionsEqual = false;
    GridRegion cachedR0, cachedR1;
    int cachedMdfx = 0, cachedMdfy = 0, cachedMdfz = 0;
    bool cachedIsEnsembleMode = true;
    CorrelationDataMode cachedDataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool cachedUseBufferTiling = true;
    // Data.
    float minFieldVal = std::numeric_limits<float>::max(), maxFieldVal = std::numeric_limits<float>::lowest();
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    std::vector<sgl::vk::BufferPtr> fieldBuffers;
    std::vector<AuxiliaryMemoryToken> deviceMemoryTokens;
    // Optional, if separate regions and downscaled fields are used.
    std::vector<sgl::vk::ImageViewPtr> fieldImageViewsR1;
    std::vector<sgl::vk::BufferPtr> fieldBuffersR1;
    std::vector<AuxiliaryMemoryToken> deviceMemoryTokensR1;

    // For comparing two different fields.
    bool useTwoFields = false;
    bool isSecondFieldMode = false; // Whether to use field 1 -> 2 or 2 -> 1.
    int selectedFieldIdx1 = 0, selectedFieldIdx2 = 0;
    std::string selectedScalarFieldName1, selectedScalarFieldName2;
    std::vector<float> leafStdDevArray2;
    float minStdDev2 = std::numeric_limits<float>::max(), maxStdDev2 = std::numeric_limits<float>::lowest();
    // Data.
    float minFieldVal2 = std::numeric_limits<float>::max(), maxFieldVal2 = std::numeric_limits<float>::lowest();
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews2;
    std::vector<sgl::vk::BufferPtr> fieldBuffers2;
    std::vector<AuxiliaryMemoryToken> deviceMemoryTokens2;
    // Optional, if separate regions and downscaled fields are used.
    std::vector<sgl::vk::ImageViewPtr> fieldImageViewsR12;
    std::vector<sgl::vk::BufferPtr> fieldBuffersR12;
    std::vector<AuxiliaryMemoryToken> deviceMemoryTokensR12;
};
typedef std::shared_ptr<HEBChartFieldData> HEBChartFieldDataPtr;

struct CorrelationRequestData {
    uint32_t xi, yi, zi, i, xj, yj, zj, j;
};

struct HEBChartFieldCache {
    float minFieldVal = std::numeric_limits<float>::max();
    float maxFieldVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::DeviceCacheEntry> fieldEntries;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    std::vector<sgl::vk::BufferPtr> fieldBuffers;
    // Optional, if separate regions and downscaled fields are used.
    std::vector<sgl::vk::ImageViewPtr> fieldImageViewsR1;
    std::vector<sgl::vk::BufferPtr> fieldBuffersR1;

    // For comparing two different fields.
    bool useTwoFields = false;
    bool isSecondFieldMode = false; // Whether to use field 1 -> 2 or 2 -> 1.
    // Data.
    float minFieldVal2 = std::numeric_limits<float>::max(), maxFieldVal2 = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::DeviceCacheEntry> fieldEntries2;
};

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
    ~HEBChart() override;
    DiagramType getDiagramType() override { return DiagramType::HEB_CHART; }
    void initialize() override;
    void update(float dt) override;
    void updateSizeByParent() override;
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setDiagramMode(DiagramMode _diagramMode);
    void setIsFocusView(int focusViewLevel);
    void setRegions(const std::pair<GridRegion, GridRegion>& _rs);
    void clearScalarFields();
    void addScalarField(int selectedFieldIdx, const std::string& _scalarFieldName);
    void removeScalarField(int selectedFieldIdx, bool doNotShiftIndicesBack);
    void setColorMap(int fieldIdx, DiagramColorMap _colorMap); //< Only lines if separateColorVarianceAndCorrelation.
    void setColorMapVariance(DiagramColorMap _colorMap);
    void setIsEnsembleMode(bool _isEnsembleMode);
    void setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType);
    void setUseAbsoluteCorrelationMeasure(bool _useAbsoluteCorrelationMeasure);
    void setNumBins(int _numBins);
    void setKraskovNumNeighbors(int _k);
    void setSamplingMethodType(SamplingMethodType _samplingMethodType);
    void setNumSamples(int _numSamples);
    void setNumInitSamples(int _numInitSamples);
    void setNumBOIterations(int _numBOIterations);
    void setNloptAlgorithm(nlopt::algorithm _algorithm);
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
    void setDesaturateUnselectedRing(bool _desaturate);
    void setUseNeonSelectionColors(bool _useNeonSelectionColors);
    void setUseGlobalStdDevRange(bool _useGlobalStdDevRange);
    void setShowSelectedRegionsByColor(bool _show);
    void setUseCorrelationComputationGpu(bool _useGpu);
    void setDataMode(CorrelationDataMode _dataMode);
    void setUseBufferTiling(bool _useBufferTiling);
    void setShowVariablesForFieldIdxOnly(int _limitedFieldIdx);
    void setOctreeMethod(OctreeMethod _octreeMethod);
    void setRegionWinding(RegionWinding _regionWinding);

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
    [[nodiscard]] bool getHasNewFocusSelection(bool& isDeselection);
    [[nodiscard]] std::pair<GridRegion, GridRegion> getFocusSelection();
    [[nodiscard]] GridRegion getGridRegionPointIdx(int idx, uint32_t pointIdx);
    [[nodiscard]] int getLeafIdxGroup(int leafIdx) const;
    [[nodiscard]] bool getHasFocusSelectionField();
    [[nodiscard]] int getFocusSelectionFieldIndex();

    // Range queries.
    glm::vec2 getCorrelationRangeTotal();
    glm::ivec2 getCellDistanceRangeTotal();
    void setCorrelationRange(const glm::vec2& _range);
    void setCellDistanceRange(const glm::ivec2& _range);

    // Focus+Context state queries.
    [[nodiscard]] inline int getReturnToViewIdx() { int idx = returnToViewIdx; returnToViewIdx = -1; return idx; }

    /// Returns whether ensemble or time correlation mode is used.
    [[nodiscard]] inline bool getIsEnsembleMode() const { return isEnsembleMode; }

    // For computing a global min/max over all diagrams.
    std::pair<float, float> getLocalStdDevRange(int fieldIdx, int varNum);
    void setGlobalStdDevRangeQueryCallback(std::function<std::pair<float, float>(int, int)> callback);
    std::pair<float, float> getGlobalStdDevRange(int fieldIdx, int varNum);

    // For performance tests.
    inline void setIsHeadlessMode(bool _isHeadlessMode)  { isHeadlessMode = _isHeadlessMode; }
    void setSyntheticTestCase(const std::vector<std::pair<uint32_t, uint32_t>>& blockPairs);
    [[nodiscard]] inline bool getForcedUseMeanFields() const { return isUseMeanFieldsForced; }
    void setForcedUseMeanFields(int fx, int fy, int fz);
    void disableForcedUseMeanFields();

    struct PerfStatistics {
        double elapsedTimeMicroseconds{};
        std::vector<float> maximumValues{};
    };
    void createFieldCacheForTests();
    PerfStatistics computeCorrelationsBlockPairs(
            const std::vector<std::pair<uint32_t, uint32_t>>& blockPairs,
            const std::vector<float*>& downscaledFields0, const std::vector<float*>& downscaledFields1);
    void computeAllCorrelationsBlockPair(uint32_t i, uint32_t j, std::vector<float>& allValues);
    void computeDownscaledFieldPerfTest(std::vector<float*>& downscaledFields);

protected:
    bool hasData() override {
        return true;
    }
    void renderCorrelationMatrix();
    void renderBaseNanoVG() override;
    void renderChordDiagramNanoVG();
#ifdef SUPPORT_SKIA
    void renderBaseSkia() override;
    void renderChordDiagramSkia();
#endif
#ifdef SUPPORT_VKVG
    void renderBaseVkvg() override;
    void renderChordDiagramVkvg();
#endif

    void onUpdatedWindowSize() override;

private:
    VolumeDataPtr volumeData;
    bool isFocusView = false;
    bool dataDirty = true;
    bool isHeadlessMode = false;
    DiagramMode diagramMode = DiagramMode::CHORD;

    int getCorrelationMemberCount();
    HostCacheEntry getFieldEntryCpu(const std::string& fieldName, int fieldIdx);
    DeviceCacheEntry getFieldEntryDevice(const std::string& fieldName, int fieldIdx, bool wantsImageData = true);
    std::pair<float, float> getMinMaxScalarFieldValue(const std::string& fieldName, int fieldIdx);
    bool isEnsembleMode = true; //< Ensemble or time mode?

    // Hierarchy data.
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    bool useAbsoluteCorrelationMeasure = true; ///< For non-MI measures.
    int numBins = 80; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_BINNED.
    int k = 3; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    CorrelationDataMode dataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool useBufferTiling = true;
    int dfx = 32, dfy = 32, dfz = 32; ///< Downscaling factors.
    int xs = 0, ys = 0, zs = 0; //< Grid size.
    int xsd0 = 0, ysd0 = 0, zsd0 = 0; //< Downscaled grid size.
    int xsd1 = 0, ysd1 = 0, zsd1 = 0; //< Downscaled grid size.
    GridRegion r0{}, r1{};
    bool regionsEqual = true;
    std::vector<HEBNode> nodesList;
    std::vector<uint32_t> pointToNodeIndexMap0, pointToNodeIndexMap1;
    uint32_t leafIdxOffset = 0, leafIdxOffset1 = 0;

    // Mean field data.
    bool isUseMeanFieldsForced = false;
    bool useMeanFields = false;
    int mdfx = 1, mdfy = 1, mdfz = 1;

    // B-spline data.
    void updateData();
    void updateRegion();
    void computeDownscaledField(
            HEBChartFieldData* fieldData, int idx, std::vector<float*>& downscaledFields);
    void computeDownscaledFieldVariance(HEBChartFieldData* fieldData, int idx, int varNum);
    glm::vec2 correlationRange{}, correlationRangeTotal{};
    glm::ivec2 cellDistanceRange{}, cellDistanceRangeTotal{};

    // Chord diagram data.
    void updateDataVarChord(
            HEBChartFieldData* fieldData, const std::vector<MIFieldEntry>& miFieldEntries,
            HEBChartFieldUpdateData& updateData);
    void updateDataChord(std::vector<HEBChartFieldUpdateData>& updateDataArray);
    OctreeMethod octreeMethod = OctreeMethod::TOP_DOWN_POT;
    RegionWinding regionWinding = RegionWinding::WINDING_AXIS_SYMMETRIC;
    int numLinesTotal = 0;
    int MAX_NUM_LINES = 100;
    const int NUM_SUBDIVISIONS = 50;
    float beta = 0.75f;
    float curveThickness = 1.5f;
    float curveOpacity = 0.4f;
    // Lines (stored separately from field data, as lines should be rendered from lowest to highest value).
    std::vector<glm::vec2> curvePoints;
    std::vector<float> correlationValuesArray; ///< per-line values.
    std::vector<std::pair<int, int>> connectedPointsArray; ///< points connected by lines.
    std::vector<int> lineFieldIndexArray;

    // Correlation matrix data.
    void updateDataVarMatrix(
            HEBChartFieldData* fieldData, const std::vector<MIFieldEntry>& miFieldEntries);
    std::shared_ptr<CorrelationMatrix> correlationMatrix;

    // Correlation sampling.
    void computeCorrelations(
            HEBChartFieldData* fieldData,
            const std::vector<float*>& downscaledFields0, const std::vector<float*>& downscaledFields1,
            std::vector<MIFieldEntry>& miFieldEntries);
    void computeCorrelationsMean(
            HEBChartFieldData* fieldData,
            const std::vector<float*>& downscaledFields0, const std::vector<float*>& downscaledFields1,
            std::vector<MIFieldEntry>& miFieldEntries);
    void computeCorrelationsSamplingCpu(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries);
    void correlationSamplingExecuteCpuDefault(
            HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries,
            const std::vector<const float*>& fields, float minFieldVal, float maxFieldVal,
            const std::vector<const float*>& fields2, float minFieldVal2, float maxFieldVal2);
    void correlationSamplingExecuteCpuBayesian(
            HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries,
            const std::vector<const float*>& fields, float minFieldVal, float maxFieldVal,
            const std::vector<const float*>& fields2, float minFieldVal2, float maxFieldVal2);
    // GPU code.
    void computeCorrelationsSamplingGpu(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries);
    void correlationSamplingExecuteGpuDefault(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries);
    void correlationSamplingExecuteGpuBayesian(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries);
    void clearFieldDeviceData();
    std::shared_ptr<HEBChartFieldCache> getFieldCache(HEBChartFieldData* fieldData);
    sgl::vk::BufferPtr computeCorrelationsForRequests(
            std::vector<CorrelationRequestData>& requests,
            std::shared_ptr<HEBChartFieldCache>& fieldCache, bool isFirstBatch);

    void createBatchCacheData(uint32_t& batchSizeSamplesMax);
    SamplingMethodType samplingMethodType = SamplingMethodType::QUASIRANDOM_PLASTIC;
    int numSamples = 100;
    int numInitSamples = 20;
    int numBOIterations = 100;
    nlopt::algorithm algorithm = nlopt::GN_DIRECT_L_RAND;
    // Performance/quality measurement code.
    bool isSubselection = false;
    std::vector<std::pair<uint32_t, uint32_t>> subselectionBlockPairs;
    bool isSyntheticTestCase = false;
    std::map<std::pair<int, int>, std::shared_ptr<MultivariateGaussian>> syntheticFieldsMap;

    // GPU computations.
    bool useCorrelationComputationGpu = true;
    bool supportsAsyncCompute = true;
    sgl::vk::Renderer* computeRenderer = nullptr;
    uint32_t cachedBatchSizeSamplesMax = 0;
    sgl::vk::BufferPtr requestsBuffer{}, requestsStagingBuffer{};
    sgl::vk::BufferPtr correlationOutputBuffer{}, correlationOutputStagingBuffer{};
    sgl::vk::FencePtr fence{};
    VkCommandPool commandPool{};
    VkCommandBuffer commandBuffer{};
    std::shared_ptr<CorrelationComputePass> correlationComputePass;

    // Field data.
    std::pair<float, float> getMinMaxCorrelationValue();
    std::vector<HEBChartFieldDataPtr> fieldDataArray;
    float minCorrelationValueGlobal = 0.0f, maxCorrelationValueGlobal = 0.0f;
    DiagramColorMap colorMapVariance = DiagramColorMap::VIRIDIS;
    std::function<std::pair<float, float>(int, int)> globalStdDevRangeQueryCallback;
    bool useGlobalStdDevRange = true;
    int limitedFieldIdx = -1;

    // GUI data.
    float chartRadius{};
    float totalRadius{};
    int hoveredPointIdx = -1;
    int hoveredLineIdx = -1; // Chord mode
    std::optional<glm::ivec2> hoveredGridIdx{}; // Matrix mode
    int clickedPointIdx = -1;
    int clickedLineIdx = -1; // Chord mode
    std::optional<glm::ivec2> clickedGridIdx{}; // Matrix mode
    int selectedPointIndices[2] = { -1, -1 };
    int selectedLineIdx = -1; // Chord mode
    std::optional<glm::ivec2> selectedGridIdx{}; // Matrix mode
    int clickedPointIdxOld = -1, clickedLineIdxOld = -1; //< For getHasNewFocusSelection.
    std::optional<glm::ivec2> clickedGridIdxOld{}; // Matrix mode, for getHasNewFocusSelection.
    bool showCorrelationForClickedPoint = false;
    uint32_t clickedPointGridIdx = 0;
    bool isFocusSelectionReset = false;
    float pointRadiusBase = 1.5f;
    sgl::Color circleFillColor = sgl::Color(180, 180, 180, 255);
    sgl::Color circleFillColorSelected0 = sgl::Color(180, 80, 80, 255); //< Overwritten in setUseNeonSelectionColors.
    sgl::Color circleFillColorSelected1 = sgl::Color(50, 100, 180, 255); //< Overwritten in setUseNeonSelectionColors.
    sgl::Color circleStrokeColorDark = sgl::Color(255, 255, 255, 255);
    sgl::Color circleStrokeColorBright = sgl::Color(0, 0, 0, 255);
    bool alignWithParentWindow = false;
    bool opacityByValue = false;
    bool colorByValue = true;
    bool showSelectedRegionsByColor = true;
    bool separateColorVarianceAndCorrelation = true;
    bool desaturateUnselectedRing = true;
    bool useNeonSelectionColors = true;
    bool useRingArrows = true;
    const float arrowAngleRad = 0.01f;

    // Outer ring.
    void renderRings();
    bool showRing = true;
    float outerRingOffset = 3.0f;
    float outerRingWidth = 0.0f; //< Determined automatically.
    float outerRingSizePct = 0.1f;
    sgl::Color ringStrokeColorSelected = sgl::Color(255, 255, 130);

    // Arrow(s) pointing at selected point(s).
    void drawSelectionArrows();

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
    void updateButtonsLayout();
    std::vector<DiagramButton> buttons;
    int returnToViewIdx = -1;
};

#endif //CORRERENDER_HEBCHART_HPP
