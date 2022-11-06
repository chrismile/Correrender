/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "RenderingModes.hpp"
#include "DvrRenderer.hpp"

DvrRenderer::DvrRenderer(ViewManager* viewManager, sgl::TransferFunctionWindow& transferFunctionWindow)
        : Renderer(
                RENDERING_MODE_NAMES[int(RENDERING_MODE_DIRECT_VOLUME_RENDERING)],
                viewManager, transferFunctionWindow) {
    ;
}

void DvrRenderer::initialize() {
    Renderer::initialize();
}

void DvrRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    volumeData = _volumeData;
    if (isNewData) {
        selectedFieldIdx = 0;
    }
    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    selectedScalarFieldName = fieldNames.at(selectedFieldIdx);

    for (auto& dvrPass : dvrPasses) {
        dvrPass->setVolumeData(volumeData, isNewData);
        dvrPass->setSelectedScalarFieldName(selectedScalarFieldName);
        dvrPass->setStepSize(stepSize);
    }
}

void DvrRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    dvrPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void DvrRenderer::renderViewImpl(uint32_t viewIdx) {
    dvrPasses.at(viewIdx)->render();
}

void DvrRenderer::addViewImpl(uint32_t viewIdx) {
    auto dvrPass = std::make_shared<DvrPass>(renderer, viewManager->getViewSceneData(viewIdx));
    if (volumeData) {
        dvrPass->setVolumeData(volumeData, true);
        dvrPass->setSelectedScalarFieldName(selectedScalarFieldName);
    }
    dvrPass->setStepSize(stepSize);
    dvrPass->setAttenuationCoefficient(attenuationCoefficient);
    dvrPasses.push_back(dvrPass);
}

void DvrRenderer::removeViewImpl(uint32_t viewIdx) {
    dvrPasses.erase(dvrPasses.begin() + viewIdx);
}

void DvrRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        if (propertyEditor.addCombo("Scalar Field", &selectedFieldIdx, fieldNames.data(), int(fieldNames.size()))) {
            selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
            dirty = true;
            reRender = true;
        }
    }
    if (propertyEditor.addSliderFloat("Step Size (Voxel)", &stepSize, 0.01f, 1.0f)) {
        stepSize = std::clamp(stepSize, 0.01f, 1.0f);
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setStepSize(stepSize);
        }
        reRender = true;
    }
    if (propertyEditor.addSliderFloat("Attenuation Coefficient", &attenuationCoefficient, 0, 500)) {
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setAttenuationCoefficient(attenuationCoefficient);
        }
        reRender = true;
    }
}



DvrPass::DvrPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : ComputePass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    rendererUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(RenderSettingsData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void DvrPass::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;

    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    float voxelSizeX = aabb.getDimensions().x / float(volumeData->getGridSizeX());
    float voxelSizeY = aabb.getDimensions().y / float(volumeData->getGridSizeY());
    float voxelSizeZ = aabb.getDimensions().z / float(volumeData->getGridSizeZ());
    voxelSize = std::min(voxelSizeX, std::min(voxelSizeY, voxelSizeZ));
    renderSettingsData.minBoundingBox = aabb.min;
    renderSettingsData.maxBoundingBox = aabb.max;
    renderSettingsData.stepSize = voxelSize * stepSize;

    dataDirty = true;
}

void DvrPass::setSelectedScalarFieldName(const std::string& _field) {
    selectedScalarFieldName = _field;
    if (volumeData && computeData) {
        auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
        computeData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    }
}

void DvrPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    if (sceneData->useDepthBuffer) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_DEPTH_BUFFER", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"DvrShader.Compute"}, preprocessorDefines);
}

void DvrPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    volumeData->setRenderDataBindings(computeData);
    computeData->setStaticBuffer(rendererUniformDataBuffer, "RendererUniformDataBuffer");
    computeData->setStaticImageView(sceneImageView, "outputImage");
    computeData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    if (sceneData->useDepthBuffer) {
        computeData->setStaticImageView(sceneData->sceneDepthColorImage, "depthBuffer");
    }
}

void DvrPass::recreateSwapchain(uint32_t width, uint32_t height) {
    sceneImageView = (*sceneData->sceneTexture)->getImageView();

    if (computeData) {
        computeData->setStaticImageView(sceneImageView, "outputImage");
    }
}

void DvrPass::_render() {
    renderSettingsData.stepSize = voxelSize * stepSize;
    renderSettingsData.inverseViewMatrix = glm::inverse((*camera)->getViewMatrix());
    renderSettingsData.inverseProjectionMatrix = glm::inverse((*camera)->getProjectionMatrix());
    renderSettingsData.zNear = (*camera)->getNearClipDistance();
    renderSettingsData.zFar = (*camera)->getFarClipDistance();
    rendererUniformDataBuffer->updateData(
            sizeof(RenderSettingsData), &renderSettingsData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    //renderer->insertImageMemoryBarrier(
    //       sceneImageView->getImage(),
    //       sceneImageView->getImage()->getVkImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
    //       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    //       VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
    sceneData->switchColorState(RenderTargetAccess::COMPUTE);
    if (sceneData->useDepthBuffer) {
        sceneData->switchDepthState(RenderTargetAccess::COMPUTE);
    }

    int width = int(sceneImageView->getImage()->getImageSettings().width);
    int height = int(sceneImageView->getImage()->getImageSettings().height);
    int groupCountX = sgl::iceil(width, 16);
    int groupCountY = sgl::iceil(height, 16);
    renderer->dispatch(computeData, groupCountX, groupCountY, 1);

    //renderer->insertImageMemoryBarrier(
    //        sceneImageView->getImage(),
    //        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    //        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    //        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}
