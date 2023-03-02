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

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "../../Calculators/Similarity.hpp"
#include "../Renderer.hpp"
#include "DiagramColorMap.hpp"

class HEBChart;
class DomainOutlineRasterPass;
class DomainOutlineComputePass;
class ConnectingLineRasterPass;

struct OutlineRenderData {
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
};

class DiagramRenderer : public Renderer {
public:
    explicit DiagramRenderer(ViewManager* viewManager);
    ~DiagramRenderer() override;
    void initialize() override;
    [[nodiscard]] bool getIsOpaqueRenderer() const override { return false; }
    [[nodiscard]] bool getIsOverlayRenderer() const override { return true; }
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) override;
    void update(float dt, bool isMouseGrabbed) override;
    bool getHasGrabbedMouse() const override;

protected:
    void renderViewImpl(uint32_t viewIdx) override;
    void renderViewPreImpl(uint32_t viewIdx) override;
    void addViewImpl(uint32_t viewIdx) override;
    void removeViewImpl(uint32_t viewIdx) override;
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    VolumeDataPtr volumeData;
    std::vector<std::shared_ptr<HEBChart>> diagrams;

    // UI renderer settings.
    int selectedFieldIdx = 0, oldSelectedFieldIdx = 0;
    std::string selectedScalarFieldName;
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    float beta = 0.75f;
    int downscalingFactor = 32;
    int lineCountFactor = 100;
    float curveOpacity = 0.4f;
    int cellDistanceThreshold = 0;
    int diagramRadius = 160;
    bool opacityByValue = false;
    bool colorByValue = true;
    DiagramColorMap colorMap = DiagramColorMap::CIVIDIS;
    bool use2dField = false;

    // Test data.
    std::vector<std::string> variableNames;
    std::vector<std::vector<float>> variableValuesTimeDependent;

    // Selected region.
    float lineWidth = 0.001f;
    std::vector<OutlineRenderData> outlineRenderDataList[2];
    std::vector<std::shared_ptr<DomainOutlineRasterPass>> domainOutlineRasterPasses[2];
    std::vector<std::shared_ptr<DomainOutlineComputePass>> domainOutlineComputePasses[2];
    std::vector<std::shared_ptr<ConnectingLineRasterPass>> connectingLineRasterPass;
};

#endif //CORRERENDER_DIAGRAMRENDERER_HPP
