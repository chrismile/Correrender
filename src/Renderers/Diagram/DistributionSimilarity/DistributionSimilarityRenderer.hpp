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

#ifndef CORRERENDER_DISTRIBUTIONSIMILARITYRENDERER_HPP
#define CORRERENDER_DISTRIBUTIONSIMILARITYRENDERER_HPP

#include <Graphics/Color.hpp>

#include "../../Renderer.hpp"
#include "Volume/Cache/HostCacheEntry.hpp"
#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/CorrelationMatrix.hpp"
#include "Renderers/IsoSurfaces.hpp"

class HostCacheEntryType;
typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
class DeviceCacheEntryType;
typedef std::shared_ptr<DeviceCacheEntryType> DeviceCacheEntry;

class DistributionSimilarityChart;
class IsoSurfaceRasterPass;

enum class SamplingPattern {
    ALL, QUASIRANDOM_PLASTIC
};
const char* const SAMPLING_PATTERN_NAMES[] = {
        "All", "Quasirandom Plastic"
};

enum class DistributionAnalysisMode {
    GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR,
    GRID_CELL_MEMBER_VALUE_VECTOR,
    MEMBER_GRID_CELL_VALUE_VECTOR
};
const char* const DISTRIBUTION_ANALYSIS_MODE_NAMES[] = {
        "Grid Cell Neighborhood Correlation Vector",
        "Grid Cell Member Value Vector",
        "Member Grid Cell Value Vector"
};

struct TSNESettings {
    float perplexity = 30.0f;
    float theta = 0.5f;
    int randomSeed = 17; // -1 for pseudo-random
    int maxIter = 500;
    int stopLyingIter = 0;
    int momSwitchIter = 700;
};

struct DBSCANSettings {
    float epsilon = 0.05f;
    int minPts = 8;
};

class DistributionSimilarityRenderer : public Renderer {
public:
    explicit DistributionSimilarityRenderer(ViewManager* viewManager);
    ~DistributionSimilarityRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_DISTRIBUTION_SIMILARITY; }
    [[nodiscard]] bool getIsOpaqueRenderer() const override { return false; }
    [[nodiscard]] bool getIsOverlayRenderer() const override { return true; }
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) override;
    void update(float dt, bool isMouseGrabbed) override;
    void onHasMoved(uint32_t viewIdx) override;
    void setClearColor(const sgl::Color& clearColor) override;
    [[nodiscard]] bool getHasGrabbedMouse() const override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void renderViewImpl(uint32_t viewIdx) override;
    void renderViewPreImpl(uint32_t viewIdx) override;
    void renderViewPostOpaqueImpl(uint32_t viewIdx) override;
    void addViewImpl(uint32_t viewIdx) override;
    void removeViewImpl(uint32_t viewIdx) override;
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    void recreateDiagramSwapchain(int diagramIdx = -1);
    void renderDiagramViewSelectionGui(
            sgl::PropertyEditor& propertyEditor, const std::string& name, uint32_t& diagramViewIdx);
    bool adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx);

    void samplePointPositions();
    void computeFeatureVectorsCorrelation(int& numVectors, int& numFeatures, double*& featureVectorArray);
    void computeFeatureVectorsGridCellsEnsembleValues(int& numVectors, int& numFeatures, double*& featureVectorArray);
    void computeFeatureVectorsEnsembleMembersGridCellValues(
            int& numVectors, int& numFeatures, double*& featureVectorArray);
    void recomputeCorrelationMatrix();
    VolumeDataPtr volumeData;
    uint32_t diagramViewIdx = 0;
    bool reRenderTriggeredByDiagram = false;
    std::shared_ptr<DistributionSimilarityChart> parentDiagram; //< Parent diagram.
    bool alignWithParentWindow = false;
    sgl::Color pointColor = sgl::Color(31, 119, 180);
    float pointSize = 5.0f;

    // Internal point data.
    int selectedPointIdx = -1;
    std::vector<glm::ivec3> pointGridPositions;
    int neighborhoodRadius = 3;

    // Correlation computation data.
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;
    bool calculateAbsoluteValue = false; ///< Whether to use absolute value for non-MI correlations.
    int numBins = 80; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_BINNED.
    int k = 3; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    int kMax = 20; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    int fieldIndex = 0, fieldIndexGui = 0;
    int fieldIndex2 = 0, fieldIndex2Gui = 0;
    CorrelationDataMode dataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool useBufferTiling = true;
    bool useSeparateFields = false;
    int cachedTimeStepIdx = -1, cachedEnsembleIdx = -1;

    void setRecomputeFlag();
    bool dataDirty = true;

    bool getSupportsBufferMode();
    int getCorrelationMemberCount();
    HostCacheEntry getFieldEntryCpu(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx);
    DeviceCacheEntry getFieldEntryDevice(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx, bool wantsImageData = true);
    std::pair<float, float> getMinMaxScalarFieldValue(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx);

    void clearFieldDeviceData();
    void onCorrelationMemberCountChanged();
    bool isEnsembleMode = true; //< Ensemble or time mode?
    bool useTimeLagCorrelations = false;
    int timeLagTimeStepIdx = 0;
    //bool useCorrelationMode = true;
    DistributionAnalysisMode distributionAnalysisMode =
            DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR;

    // Grid point sampling settings.
    SamplingPattern samplingPattern = SamplingPattern::QUASIRANDOM_PLASTIC;
    int numRandomPoints = 4000;
    bool usePredicateField = false;
    int predicateFieldIdx = 0, predicateFieldIdxGui = 0;
    bool keepDistanceToBorder = false; //< for DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR

    // t-SNE settings.
    TSNESettings tsneSettings{};

    // DBSCAN settings.
    bool useDbscanClustering = true;
    DBSCANSettings dbscanSettings{};
    void computeClusteringSurfacesByMetaballs(
            const std::vector<glm::vec2>& points, const std::vector<std::vector<size_t>>& clusters);
    bool showClustering3d = true;
    std::vector<std::shared_ptr<IsoSurfaceRasterPass>> isoSurfaceRasterPasses;
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;
    sgl::vk::BufferPtr vertexColorBuffer;
};

#endif //CORRERENDER_DISTRIBUTIONSIMILARITYRENDERER_HPP
