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

#ifndef CORRERENDER_DIAGRAMRENDERER_HPP
#define CORRERENDER_DIAGRAMRENDERER_HPP

#include <utility>

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "../../Calculators/CorrelationDefines.hpp"
#include "../Renderer.hpp"
#include "Octree.hpp"
#include "Sampling.hpp"
#include "DiagramColorMap.hpp"
#include "Region.hpp"

class HEBChart;
class DomainOutlineRasterPass;
class DomainOutlineComputePass;
class ConnectingLineRasterPass;
class SelectionBoxRasterPass;
class ShadowRectRasterPass;

struct OutlineRenderData {
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
};

struct DiagramSelectedFieldData {
    DiagramSelectedFieldData(int first, std::string second, DiagramColorMap third)
            : first(first), second(std::move(second)), third(third) {}
    int first;
    std::string second;
    DiagramColorMap third;
};

class DiagramRenderer : public Renderer {
public:
    explicit DiagramRenderer(ViewManager* viewManager);
    ~DiagramRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_DIAGRAM_RENDERER; }
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
    bool getSupportsBufferMode();
    void updateScalarFieldComboValue();
    void recreateDiagramSwapchain(int diagramIdx = -1);
    void resetSelections(int idx = 0);
    std::pair<float, float> computeGlobalStdDevRange(int fieldIdx, int varNum);
    VolumeDataPtr volumeData;
    void onCorrelationMemberCountChanged();
    int cachedMemberCount = 0;
    uint32_t contextDiagramViewIdx = 0, focusDiagramViewIdx = 0;
    bool reRenderTriggeredByDiagram = false;
    std::shared_ptr<HEBChart> parentDiagram; //< Parent diagram.
    std::vector<std::shared_ptr<HEBChart>> diagrams; //< Diagram stack.
    std::vector<std::pair<GridRegion, GridRegion>> selectedRegionStack; //< Selected regions stack.
    void renderDiagramViewSelectionGui(
            sgl::PropertyEditor& propertyEditor, const std::string& name, uint32_t& diagramViewIdx);
    bool adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx);
    bool renderOnlyLastFocusDiagram = true;

    // UI renderer settings.
    int getCorrelationMemberCount();
    std::vector<std::string> availableFieldNames;
    int numFieldsBase = 0;

    DiagramMode diagramMode = DiagramMode::CHORD;
    std::vector<bool> scalarFieldSelectionArray;
    std::string scalarFieldComboValue;
    std::vector<DiagramSelectedFieldData> selectedScalarFields;
    DiagramColorMap colorMapVariance = DiagramColorMap::GRAY;
    bool separateColorVarianceAndCorrelation = true;
    bool desaturateUnselectedRing = true;
    bool isEnsembleMode = true; //< Ensemble or time mode?
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    bool useAbsoluteCorrelationMeasure = true;
    int numBins = 80; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_BINNED.
    int k = 3; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    int kMax = 20; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    SamplingMethodType samplingMethodTypeContext = SamplingMethodType::QUASIRANDOM_PLASTIC;
    SamplingMethodType samplingMethodTypeFocus = SamplingMethodType::QUASIRANDOM_PLASTIC;
    bool useSeparateSamplingMethodFocus = true;
    int numSamplesContext = 100;
    int numSamplesFocus = 100;
    int numInitSamples = 20;
    int numBOIterations = 100;
    float beta = 0.75f;
    int minDownscalingFactor = 16, maxDownscalingFactor = 64;
    int minDownscalingFactorFocus = 1, maxDownscalingFactorFocus = 16;
    int downscalingFactorX = 32, downscalingFactorY = 32, downscalingFactorZ = 32;
    int downscalingFactorFocusX = 4, downscalingFactorFocusY = 4, downscalingFactorFocusZ = 4;
    bool downscalingPowerOfTwo = true;
    bool downscalingFactorUniform = true;
    int lineCountFactorContext = 100;
    int lineCountFactorFocus = 100;
    float curveThickness = 1.5f;
    float curveOpacityContext = 0.4f;
    float curveOpacityFocus = 0.4f;
    glm::vec2 correlationRange{}, correlationRangeTotal{};
    glm::ivec2 cellDistanceRange{}, cellDistanceRangeTotal{};
    int diagramRadius = 160;
    float outerRingSizePct = 0.1f;
    bool alignWithParentWindow = false;
    bool opacityByValue = false;
    bool colorByValue = true;
    bool showSelectedRegionsByColor = true;
    bool useNeonSelectionColors = true;
    bool useGlobalStdDevRange = true;
    bool useCorrelationComputationGpuContext = true;
    bool useCorrelationComputationGpuFocus = true;
    CorrelationDataMode dataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool useBufferTiling = true;
    bool showOnlySelectedVariableInFocusDiagrams = true;
    OctreeMethod octreeMethod = OctreeMethod::TOP_DOWN_POT;
    RegionWinding regionWinding = RegionWinding::WINDING_AXIS_SYMMETRIC;

    // Selected region.
    float lineWidth = 0.001f;
    std::vector<OutlineRenderData> outlineRenderDataList[2];
    std::vector<std::shared_ptr<DomainOutlineRasterPass>> domainOutlineRasterPasses[2];
    std::vector<std::shared_ptr<DomainOutlineComputePass>> domainOutlineComputePasses[2];
    std::vector<std::shared_ptr<ConnectingLineRasterPass>> connectingLineRasterPass;
    // Opaque selection boxes as an alternative to the outline raster passes.
    std::vector<std::shared_ptr<SelectionBoxRasterPass>> selectionBoxRasterPasses[2];
    std::vector<std::shared_ptr<ShadowRectRasterPass>> shadowRectRasterPasses[2];
    bool useOpaqueSelectionBoxes = true;

    // Camera alignment rotation.
    void updateAlignmentRotation(float dt, HEBChart* diagram);
    bool useAlignmentRotation = false;
    HEBChart* cachedAlignmentRotationDiagram = nullptr;
    uint32_t cachedPointIdx0 = 0, cachedPointIdx1 = 0;
    const float alignmentRotationTotalTimeMax = 4.0f;
    float alignmentRotationTotalTime = 0.0f;
    float alignmentRotationTime = 0.0f;
    float rotationAngleTotal = 0.0f;
    glm::vec3 cameraUpStart{};
    glm::vec3 cameraLookAtStart{};
    glm::vec3 cameraPositionStart{};
};

#endif //CORRERENDER_DIAGRAMRENDERER_HPP
