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

#ifndef CORRERENDER_CORRELATIONMATRIXRENDERER_HPP
#define CORRERENDER_CORRELATIONMATRIXRENDERER_HPP

#include <Graphics/Color.hpp>

#include "../../Renderer.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/CorrelationMatrix.hpp"

class CorrelationMatrixChart;

class CorrelationMatrixRenderer : public Renderer {
public:
    explicit CorrelationMatrixRenderer(ViewManager* viewManager);
    ~CorrelationMatrixRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_CORRELATION_MATRIX; }
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

    void recomputeCorrelationMatrix();
    VolumeDataPtr volumeData;
    int cachedTimeStepIdx = -1, cachedEnsembleIdx = -1;
    uint32_t diagramViewIdx = 0;
    bool reRenderTriggeredByDiagram = false;
    std::shared_ptr<CorrelationMatrixChart> parentDiagram; //< Parent diagram.
    bool alignWithParentWindow = false;
    bool useAllTimeSteps = false;
    bool useAllEnsembleMembers = false;
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;
    int useFieldAccuracyDouble = 1;
    DiagramColorMap colorMap = DiagramColorMap::VIRIDIS;
};

#endif //CORRERENDER_CORRELATIONMATRIXRENDERER_HPP
