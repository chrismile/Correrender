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

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/imgui_custom.h>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Renderers/DomainOutlineRenderer.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "../RenderingModes.hpp"

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
    }
    selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
    volumeData->acquireTf(this, selectedFieldIdx);
    volumeData->acquireScalarField(this, selectedFieldIdx);
    oldSelectedFieldIdx = selectedFieldIdx;

    for (auto& diagram : diagrams) {
        diagram->setVolumeData(volumeData, isNewData);
        diagram->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        diagram->setBeta(beta);
        diagram->setDownscalingFactor(downscalingFactor);
        diagram->setLineCountFactor(lineCountFactor);
        diagram->setCurveOpacity(curveOpacity);
        diagram->setCellDistanceThreshold(cellDistanceThreshold);
        diagram->setUse2DField(use2dField);
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
    //dvrPasses.at(viewIdx)->recreateSwapchain(width, height);
    SceneData* sceneData = viewManager->getViewSceneData(viewIdx);
    diagrams.at(viewIdx)->setBlitTargetVk(
            (*sceneData->sceneTexture)->getImageView(),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    domainOutlineRasterPasses.at(viewIdx)->recreateSwapchain(width, height);
    domainOutlineComputePasses.at(viewIdx)->recreateSwapchain(width, height);
}

void DiagramRenderer::update(float dt) {
    uint32_t viewIdx = 0;
    for (auto& diagram : diagrams) {
        diagram->update(dt);
        if (diagram->getNeedsReRender()) {
            reRenderViewArray.at(viewIdx) = true;
        }
        viewIdx++;
    }
}

void DiagramRenderer::renderViewImpl(uint32_t viewIdx) {
    auto& diagram = diagrams.at(viewIdx);
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
    auto& diagram = diagrams.at(viewIdx);
    if (diagram->getIsRegionSelected()) {
        domainOutlineComputePasses.at(viewIdx)->setOutlineSettings(diagram->getSelectedRegion(), lineWidth);
        domainOutlineComputePasses.at(viewIdx)->render();
        domainOutlineRasterPasses.at(viewIdx)->render();
    }
}

void DiagramRenderer::addViewImpl(uint32_t viewIdx) {
    auto diagram = std::make_shared<HEBChart>();
    //auto diagram = std::make_shared<RadarBarChart>(true);
    diagram->setRendererVk(renderer);
    diagram->initialize();
    //diagram->setDataTimeDependent(variableNames, variableValuesTimeDependent);
    //diagram->setUseEqualArea(true);
    if (volumeData) {
        diagram->setVolumeData(volumeData, true);
        diagram->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        diagram->setBeta(beta);
        diagram->setDownscalingFactor(downscalingFactor);
        diagram->setLineCountFactor(lineCountFactor);
        diagram->setCurveOpacity(curveOpacity);
        diagram->setCellDistanceThreshold(cellDistanceThreshold);
        diagram->setUse2DField(use2dField);
    }
    diagrams.push_back(diagram);

    const size_t numEdges = 12;
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

    outlineRenderDataList.push_back(outlineRenderData);

    auto domainOutlineRasterPass = std::make_shared<DomainOutlineRasterPass>(
            renderer, viewManager->getViewSceneData(viewIdx));
    domainOutlineRasterPass->setRenderData(outlineRenderData.indexBuffer, outlineRenderData.vertexPositionBuffer);
    domainOutlineRasterPasses.push_back(domainOutlineRasterPass);

    auto domainOutlineComputePass = std::make_shared<DomainOutlineComputePass>(renderer);
    domainOutlineComputePass->setRenderData(outlineRenderData.indexBuffer, outlineRenderData.vertexPositionBuffer);
    domainOutlineComputePasses.push_back(domainOutlineComputePass);
}

void DiagramRenderer::removeViewImpl(uint32_t viewIdx) {
    diagrams.erase(diagrams.begin() + viewIdx);

    outlineRenderDataList.erase(outlineRenderDataList.begin() + viewIdx);
    domainOutlineRasterPasses.erase(domainOutlineRasterPasses.begin() + viewIdx);
    domainOutlineComputePasses.erase(domainOutlineComputePasses.begin() + viewIdx);
}

void DiagramRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        int selectedFieldIdxGui = selectedFieldIdx;
        if (propertyEditor.addCombo("Scalar Field", &selectedFieldIdxGui, fieldNames.data(), int(fieldNames.size()))) {
            selectedFieldIdx = selectedFieldIdxGui;
            selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
            for (auto& diagram : diagrams) {
                diagram->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
            }
            dirty = true;
            reRender = true;
        }
    }

    if (propertyEditor.addSliderFloatEdit("beta", &beta, 0.0f, 1.0f) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setBeta(beta);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderIntPowerOfTwoEdit(
            "Downscaling", &downscalingFactor, 16, 128) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setDownscalingFactor(downscalingFactor);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderIntEdit(
            "#Line Factor", &lineCountFactor, 10, 1000) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setLineCountFactor(lineCountFactor);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderFloat("Opacity", &curveOpacity, 0.0f, 1.0f)) {
        for (auto& diagram : diagrams) {
            diagram->setCurveOpacity(curveOpacity);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderInt("Dist. Thresh.", &cellDistanceThreshold, 0, 100)) {
        for (auto& diagram : diagrams) {
            diagram->setCellDistanceThreshold(cellDistanceThreshold);
        }
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Use 2D Field", &use2dField)) {
        for (auto& diagram : diagrams) {
            diagram->setUse2DField(use2dField);
        }
        reRender = true;
    }

    if (diagrams.size() == 1) {
        if (diagrams.front()->renderGuiPropertyEditor(propertyEditor)) {
            reRender = true;
        }
    } else {
        int i = 0;
        for (auto& diagram : diagrams) {
            if (propertyEditor.beginNode("Diagram " + std::to_string(i + 1))) {
                if (diagram->renderGuiPropertyEditor(propertyEditor)) {
                    reRender = true;
                }
                propertyEditor.endNode();
            }
            i++;
        }
    }
}
