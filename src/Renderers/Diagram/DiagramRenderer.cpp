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

#include <random>

#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/imgui_custom.h>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Renderers/DomainOutlineRenderer.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "../RenderingModes.hpp"

#include "ConnectingLineRasterPass.hpp"
#include "RadarBarChart.hpp"
#include "HEBChart.hpp"

#include "DiagramRenderer.hpp"

DiagramRenderer::DiagramRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_DIAGRAM_RENDERER)], viewManager) {
}

DiagramRenderer::~DiagramRenderer() {
    if (!selectedScalarFieldName.empty()) {
        volumeData->releaseTf(this, selectedFieldIdx);
        volumeData->releaseScalarField(this, selectedFieldIdx);
    }
}

void DiagramRenderer::initialize() {
    Renderer::initialize();

    variableNames = {
            "Pressure", "NI_OUT", "NR_OUT", "NS_OUT", "QC", "QG", "QG_OUT", "QH", "QH_OUT", "QI", "QI_OUT",
            "NCCLOUD", "QR", "QR_OUT", "QS", "QS_OUT", "QV", "S", "T", "artificial", "artificial (threshold)",
            "conv_400", "NCGRAUPEL", "conv_600", "dD_rainfrz_gh", "dD_rainfrz_ig", "dT_mult_max", "dT_mult_min",
            "da_HET", "da_ccn_1", "da_ccn_4", "db_HET", "db_ccn_1", "NCHAIL", "db_ccn_3", "db_ccn_4", "dc_ccn_1",
            "dc_ccn_4", "dcloud_c_z", "dd_ccn_1", "dd_ccn_2", "dd_ccn_3", "dd_ccn_4", "dgraupel_a_vel", "NCICE",
            "dgraupel_b_geo", "dgraupel_b_vel", "dgraupel_vsedi_max", "dhail_vsedi_max", "dice_a_f", "dice_a_geo",
            "dice_b_geo", "dice_b_vel", "dice_c_s", "dice_vsedi_max", "NCRAIN", "dinv_z", "dk_r",
            "dp_sat_ice_const_b", "dp_sat_melt", "drain_a_geo", "drain_a_vel", "drain_alpha", "drain_b_geo",
            "drain_b_vel", "drain_beta", "NCSNOW", "drain_c_z", "drain_g1", "drain_g2", "drain_gamma",
            "drain_min_x", "drain_min_x_freezing", "drain_mu", "drain_nu", "drho_vel", "dsnow_a_geo", "NG_OUT",
            "dsnow_b_geo", "dsnow_b_vel", "dsnow_vsedi_max", "mean of artificial", "mean of artificial (threshold)",
            "mean of physical", "mean of physical (high variability)", "physical", "physical (high variability)",
            "time_after_ascent", "NH_OUT", "w", "z"
    };

    {
        std::mt19937 generator(2);
        std::uniform_real_distribution<> dis(0, 1);

        int numVariables = int(variableNames.size());
        int numTimesteps = 10;
        variableValuesTimeDependent.resize(numTimesteps);
        for (int timeStepIdx = 0; timeStepIdx < numTimesteps; timeStepIdx++) {
            std::vector<float>& variableValuesAtTime = variableValuesTimeDependent.at(timeStepIdx);
            variableValuesAtTime.reserve(numVariables);
            for (int varIdx = 1; varIdx <= numVariables; varIdx++) {
                variableValuesAtTime.push_back(float(dis(generator)));
            }
        }
    }

    diagram = std::make_shared<HEBChart>();
    //auto diagram = std::make_shared<RadarBarChart>(true);
    diagram->setRendererVk(renderer);
    diagram->initialize();
    //diagram->setDataTimeDependent(variableNames, variableValuesTimeDependent);
    //diagram->setUseEqualArea(true);
    if (volumeData) {
        diagram->setVolumeData(volumeData, true);
        diagram->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        diagram->setCorrelationMeasureType(correlationMeasureType);
        diagram->setBeta(beta);
        diagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
        diagram->setLineCountFactor(lineCountFactor);
        diagram->setCurveOpacity(curveOpacity);
        diagram->setDiagramRadius(diagramRadius);
        diagram->setAlignWithParentWindow(alignWithParentWindow);
        diagram->setOpacityByValue(opacityByValue);
        diagram->setColorByValue(colorByValue);
        diagram->setColorMap(colorMap);
        diagram->setUse2DField(use2dField);
    }
}

void DiagramRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;
    if (!selectedScalarFieldName.empty()) {
        volumeData->releaseTf(this, oldSelectedFieldIdx);
        volumeData->releaseScalarField(this, oldSelectedFieldIdx);
    }
    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    if (isNewData) {
        selectedFieldIdx = volumeData->getStandardScalarFieldIdx();

        int xs = volumeData->getGridSizeX();
        int ys = volumeData->getGridSizeY();
        int zs = volumeData->getGridSizeZ();

        float minDelta = std::min(volumeData->getDx(), std::min(volumeData->getDy(), volumeData->getDz()));
        int fx = int(std::round(volumeData->getDx() / minDelta));
        int fy = int(std::round(volumeData->getDy() / minDelta));
        int fz = int(std::round(volumeData->getDz() / minDelta));

        downscalingFactorUniform = fx == fy && fy == fz;

        int dsw = int(std::ceil(std::cbrt(float(xs * ys * zs) / 100.0f)));
        dsw = std::max(dsw, 1);
        if (!sgl::isPowerOfTwo(dsw)) {
            dsw = sgl::nextPowerOfTwo(dsw);
        }
        minDownscalingFactor = std::max(dsw / 2, 1);
        maxDownscalingFactor = dsw * 2;
        if (downscalingFactorUniform) {
            downscalingFactorX = downscalingFactorY = downscalingFactorZ = dsw;
        } else {
            downscalingFactorX = downscalingFactorY = downscalingFactorZ = dsw;
        }
    }
    selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
    volumeData->acquireTf(this, selectedFieldIdx);
    volumeData->acquireScalarField(this, selectedFieldIdx);
    oldSelectedFieldIdx = selectedFieldIdx;

    diagram->setVolumeData(volumeData, isNewData);
    if (isNewData) {
        diagram->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        diagram->setCorrelationMeasureType(correlationMeasureType);
        diagram->setBeta(beta);
        diagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
        diagram->setLineCountFactor(lineCountFactor);
        diagram->setCurveOpacity(curveOpacity);
        diagram->setDiagramRadius(diagramRadius);
        diagram->setAlignWithParentWindow(alignWithParentWindow);
        diagram->setOpacityByValue(opacityByValue);
        diagram->setColorByValue(colorByValue);
        diagram->setColorMap(colorMap);
        diagram->setUse2DField(use2dField);

        correlationRangeTotal = correlationRange = diagram->getCorrelationRangeTotal();
        cellDistanceRange = cellDistanceRangeTotal = diagram->getCellDistanceRangeTotal();
        diagram->setCorrelationRange(correlationRangeTotal);
        diagram->setCellDistanceRange(cellDistanceRange);
    }
}

void DiagramRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (selectedFieldIdx == fieldIdx) {
            selectedFieldIdx = 0;
        } else if (selectedFieldIdx > fieldIdx) {
            selectedFieldIdx--;
        }
        if (oldSelectedFieldIdx == fieldIdx) {
            oldSelectedFieldIdx = -1;
            selectedScalarFieldName.clear();
        } else if (oldSelectedFieldIdx > fieldIdx) {
            oldSelectedFieldIdx--;
        }
    }
}

void DiagramRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    if (viewIdx == diagramViewIdx) {
        recreateDiagramSwapchain();
    }
    for (int idx = 0; idx < 2; idx++) {
        domainOutlineRasterPasses[idx].at(viewIdx)->recreateSwapchain(width, height);
        domainOutlineComputePasses[idx].at(viewIdx)->recreateSwapchain(width, height);
    }
    connectingLineRasterPass.at(viewIdx)->recreateSwapchain(width, height);
}

void DiagramRenderer::recreateDiagramSwapchain() {
    SceneData* sceneData = viewManager->getViewSceneData(diagramViewIdx);
    diagram->setBlitTargetVk(
            (*sceneData->sceneTexture)->getImageView(),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (alignWithParentWindow) {
        diagram->updateSizeByParent();
    }
}

void DiagramRenderer::update(float dt, bool isMouseGrabbed) {
    uint32_t viewIdx = 0;
    if (viewIdx == diagramViewIdx) {
        if (isVisibleInView(viewIdx)) {
            diagram->update(dt);
            if (diagram->getNeedsReRender()) {
                reRenderViewArray.at(viewIdx) = true;
            }
        }
        viewIdx++;
    }
}

bool DiagramRenderer::getHasGrabbedMouse() const {
    if (diagram->getIsMouseGrabbed() || diagram->getIsMouseOverDiagramImGui()) {
        return true;
    }
    return false;
}

void DiagramRenderer::renderViewImpl(uint32_t viewIdx) {
    if (viewIdx != diagramViewIdx) {
        return;
    }

    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        diagram->setImGuiWindowOffset(int(pos.x), int(pos.y));
    } else {
        diagram->setImGuiWindowOffset(0, 0);
    }
    diagram->render();

    SceneData* sceneData = viewManager->getViewSceneData(viewIdx);
    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);
    diagram->blitToTargetVk();
}

void DiagramRenderer::renderViewPreImpl(uint32_t viewIdx) {
    for (int idx = 0; idx < 2; idx++) {
        if (diagram->getIsRegionSelected(idx)) {
            domainOutlineComputePasses[idx].at(viewIdx)->setOutlineSettings(diagram->getSelectedRegion(idx), lineWidth);
            domainOutlineComputePasses[idx].at(viewIdx)->render();
            domainOutlineRasterPasses[idx].at(viewIdx)->render();
        }
    }
    if (diagram->getIsRegionSelected(0) && diagram->getIsRegionSelected(1)) {
        connectingLineRasterPass.at(viewIdx)->setLineSettings(diagram->getLinePositions(), lineWidth);
        connectingLineRasterPass.at(viewIdx)->render();
    }
}

void DiagramRenderer::addViewImpl(uint32_t viewIdx) {
    const size_t numEdges = 12;
    for (int idx = 0; idx < 2; idx++) {
        OutlineRenderData outlineRenderData{};
        sgl::vk::Device* device = renderer->getDevice();
        outlineRenderData.vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec3) * numEdges * 8,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        outlineRenderData.indexBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(uint32_t) * numEdges * 6 * 6,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        outlineRenderDataList[idx].push_back(outlineRenderData);

        auto domainOutlineRasterPass = std::make_shared<DomainOutlineRasterPass>(
                renderer, viewManager->getViewSceneData(viewIdx));
        domainOutlineRasterPass->setRenderData(outlineRenderData.indexBuffer, outlineRenderData.vertexPositionBuffer);
        domainOutlineRasterPasses[idx].push_back(domainOutlineRasterPass);

        auto domainOutlineComputePass = std::make_shared<DomainOutlineComputePass>(renderer);
        domainOutlineComputePass->setRenderData(outlineRenderData.indexBuffer, outlineRenderData.vertexPositionBuffer);
        domainOutlineComputePasses[idx].push_back(domainOutlineComputePass);
    }

    connectingLineRasterPass.push_back(std::make_shared<ConnectingLineRasterPass>(
            renderer, viewManager->getViewSceneData(viewIdx)));
}

void DiagramRenderer::removeViewImpl(uint32_t viewIdx) {
    if (diagramViewIdx >= viewIdx && diagramViewIdx != 0) {
        diagramViewIdx--;
    } else if (viewManager->getNumViews() > 0) {
        diagramViewIdx++;
    }

    connectingLineRasterPass.erase(connectingLineRasterPass.begin() + viewIdx);

    for (int idx = 0; idx < 2; idx++) {
        outlineRenderDataList[idx].erase(outlineRenderDataList[idx].begin() + viewIdx);
        domainOutlineRasterPasses[idx].erase(domainOutlineRasterPasses[idx].begin() + viewIdx);
        domainOutlineComputePasses[idx].erase(domainOutlineComputePasses[idx].begin() + viewIdx);
    }
}

void DiagramRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    std::string textDefault = "View " + std::to_string(diagramViewIdx + 1);
    if (propertyEditor.addBeginCombo(
            "Diagram View", textDefault, ImGuiComboFlags_NoArrowButton)) {
        for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
            std::string text = "View " + std::to_string(viewIdx + 1);
            bool showInView = diagramViewIdx == uint32_t(viewIdx);
            if (ImGui::Selectable(
                    text.c_str(), &showInView, ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups)) {
                diagramViewIdx = uint32_t(viewIdx);
                reRender = true;
                recreateDiagramSwapchain();
            }
            if (showInView) {
                ImGui::SetItemDefaultFocus();
            }
        }
        propertyEditor.addEndCombo();
    }

    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        int selectedFieldIdxGui = selectedFieldIdx;
        if (propertyEditor.addCombo("Scalar Field", &selectedFieldIdxGui, fieldNames.data(), int(fieldNames.size()))) {
            selectedFieldIdx = selectedFieldIdxGui;
            selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
            diagram->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
            dirty = true;
            reRender = true;
        }
    }

    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        diagram->setCorrelationMeasureType(correlationMeasureType);
        correlationRangeTotal = correlationRange = diagram->getCorrelationRangeTotal();
        diagram->setCorrelationRange(correlationRangeTotal);
        dirty = true;
        reRender = true;
    }

    if (propertyEditor.addSliderFloatEdit("beta", &beta, 0.0f, 1.0f) == ImGui::EditMode::INPUT_FINISHED) {
        diagram->setBeta(beta);
        reRender = true;
    }

    if (downscalingFactorUniform) {
        if (propertyEditor.addSliderIntPowerOfTwoEdit(
                "Downscaling", &downscalingFactorX, minDownscalingFactor, maxDownscalingFactor)
                    == ImGui::EditMode::INPUT_FINISHED) {
            downscalingFactorY = downscalingFactorX;
            downscalingFactorZ = downscalingFactorX;
            diagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
            reRender = true;
        }
    } else {
        int dfs[3] = { downscalingFactorX, downscalingFactorY, downscalingFactorZ };
        auto editMode = propertyEditor.addSliderInt3PowerOfTwoEdit(
                "Downscaling", dfs, minDownscalingFactor, maxDownscalingFactor);
        downscalingFactorX = dfs[0];
        downscalingFactorY = dfs[1];
        downscalingFactorZ = dfs[2];
        if (editMode == ImGui::EditMode::INPUT_FINISHED) {
            diagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
            reRender = true;
        }
    }

    if (propertyEditor.addSliderIntEdit(
            "#Line Factor", &lineCountFactor, 10, 1000) == ImGui::EditMode::INPUT_FINISHED) {
        diagram->setLineCountFactor(lineCountFactor);
        reRender = true;
    }

    if (propertyEditor.addSliderFloat("Opacity", &curveOpacity, 0.0f, 1.0f)) {
        diagram->setCurveOpacity(curveOpacity);
        reRender = true;
    }

    if (propertyEditor.addSliderFloat2Edit(
            "Correlation Range", &correlationRange.x,
            correlationRangeTotal.x, correlationRangeTotal.y) == ImGui::EditMode::INPUT_FINISHED) {
        diagram->setCorrelationRange(correlationRange);
        reRender = true;
    }

    if (propertyEditor.addSliderInt2Edit(
            "Cell Dist. Range", &cellDistanceRange.x,
            cellDistanceRangeTotal.x, cellDistanceRangeTotal.y) == ImGui::EditMode::INPUT_FINISHED) {
        diagram->setCellDistanceRange(cellDistanceRange);
        reRender = true;
    }

    if (propertyEditor.addSliderInt("Diagram Radius", &diagramRadius, 100, 400)) {
        diagram->setDiagramRadius(diagramRadius);
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Align with Window", &alignWithParentWindow)) {
        diagram->setAlignWithParentWindow(alignWithParentWindow);
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Value -> Opacity", &opacityByValue)) {
        diagram->setOpacityByValue(opacityByValue);
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Value -> Color", &colorByValue)) {
        diagram->setColorByValue(colorByValue);
        reRender = true;
    }

    if (colorByValue && propertyEditor.addCombo(
            "Color Map", (int*)&colorMap, DIAGRAM_COLOR_MAP_NAMES,
            IM_ARRAYSIZE(DIAGRAM_COLOR_MAP_NAMES))) {
        diagram->setColorMap(colorMap);
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Use 2D Field", &use2dField)) {
        diagram->setUse2DField(use2dField);
        reRender = true;
    }

    if (diagram->renderGuiPropertyEditor(propertyEditor)) {
        reRender = true;
    }
}
