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
#include "IsoSurfaceRayCastingRenderer.hpp"

IsoSurfaceRayCastingRenderer::IsoSurfaceRayCastingRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_ISOSURFACE_RAYCASTER)], viewManager) {
}

IsoSurfaceRayCastingRenderer::~IsoSurfaceRayCastingRenderer() {
    if (!selectedScalarFieldName.empty()) {
        volumeData->releaseScalarField(this, selectedFieldIdx);
    }
}

void IsoSurfaceRayCastingRenderer::initialize() {
    Renderer::initialize();
}

void IsoSurfaceRayCastingRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    volumeData = _volumeData;
    if (!selectedScalarFieldName.empty()) {
        volumeData->releaseScalarField(this, oldSelectedFieldIdx);
    }
    if (isNewData) {
        selectedFieldIdx = 0;
    }
    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    std::string oldSelectedScalarFieldName = selectedScalarFieldName;
    selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
    minMaxScalarFieldValue = volumeData->getMinMaxScalarFieldValue(selectedScalarFieldName);
    if (isNewData || oldSelectedScalarFieldName != selectedScalarFieldName) {
        isoValue = (minMaxScalarFieldValue.first + minMaxScalarFieldValue.second) / 2.0f;
        for (auto& isoSurfaceRayCastingPass : isoSurfaceRayCastingPasses) {
            isoSurfaceRayCastingPass->setIsoValue(isoValue);
        }
    }
    volumeData->acquireScalarField(this, selectedFieldIdx);
    oldSelectedFieldIdx = selectedFieldIdx;

    for (auto& isoSurfaceRayCastingPass : isoSurfaceRayCastingPasses) {
        isoSurfaceRayCastingPass->setVolumeData(volumeData, isNewData);
        isoSurfaceRayCastingPass->setSelectedScalarFieldName(selectedScalarFieldName);
    }
}

void IsoSurfaceRayCastingRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    isoSurfaceRayCastingPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void IsoSurfaceRayCastingRenderer::renderViewImpl(uint32_t viewIdx) {
    isoSurfaceRayCastingPasses.at(viewIdx)->render();
}

void IsoSurfaceRayCastingRenderer::addViewImpl(uint32_t viewIdx) {
    auto isoSurfaceRayCastingPass = std::make_shared<IsoSurfaceRayCastingPass>(renderer, viewManager->getViewSceneData(viewIdx));
    if (volumeData) {
        isoSurfaceRayCastingPass->setVolumeData(volumeData, true);
        isoSurfaceRayCastingPass->setSelectedScalarFieldName(selectedScalarFieldName);
    }
    isoSurfaceRayCastingPass->setIsoValue(isoValue);
    isoSurfaceRayCastingPass->setAnalyticIntersections(analyticIntersections);
    isoSurfaceRayCastingPass->setStepSize(stepSize);
    isoSurfaceRayCastingPass->setIsoSurfaceColor(isoSurfaceColor);
    isoSurfaceRayCastingPass->setIntersectionSolver(intersectionSolver);
    isoSurfaceRayCastingPasses.push_back(isoSurfaceRayCastingPass);
}

void IsoSurfaceRayCastingRenderer::removeViewImpl(uint32_t viewIdx) {
    isoSurfaceRayCastingPasses.erase(isoSurfaceRayCastingPasses.begin() + viewIdx);
}

void IsoSurfaceRayCastingRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        if (propertyEditor.addCombo("Scalar Field", &selectedFieldIdx, fieldNames.data(), int(fieldNames.size()))) {
            selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
            minMaxScalarFieldValue = volumeData->getMinMaxScalarFieldValue(selectedScalarFieldName);
            isoValue = (minMaxScalarFieldValue.first + minMaxScalarFieldValue.second) / 2.0f;
            dirty = true;
            reRender = true;
        }
    }
    if (propertyEditor.addSliderFloat(
            "Iso Value", &isoValue, minMaxScalarFieldValue.first, minMaxScalarFieldValue.second)) {
        for (auto& isoSurfaceRayCastingPass : isoSurfaceRayCastingPasses) {
            isoSurfaceRayCastingPass->setIsoValue(isoValue);
        }
        reRender = true;
    }
    if (propertyEditor.addCheckbox("Analytic Intersections", &analyticIntersections)) {
        for (auto& isoSurfaceRayCastingPass : isoSurfaceRayCastingPasses) {
            isoSurfaceRayCastingPass->setAnalyticIntersections(analyticIntersections);
        }
        reRender = true;
    }
    if (!analyticIntersections && propertyEditor.addSliderFloat("Step Size (Voxel)", &stepSize, 0.01f, 1.0f)) {
        stepSize = std::clamp(stepSize, 0.01f, 1.0f);
        for (auto& isoSurfaceRayCastingPass : isoSurfaceRayCastingPasses) {
            isoSurfaceRayCastingPass->setStepSize(stepSize);
        }
        reRender = true;
    }
    if (analyticIntersections && propertyEditor.addCombo(
            "Solver", (int*)&intersectionSolver, INTERSECTION_SOLVER_NAMES, IM_ARRAYSIZE(INTERSECTION_SOLVER_NAMES))) {
        for (auto& isoSurfaceRayCastingPass : isoSurfaceRayCastingPasses) {
            isoSurfaceRayCastingPass->setIntersectionSolver(intersectionSolver);
        }
        reRender = true;
    }
    if (propertyEditor.addColorEdit4("Iso Surface Color", &isoSurfaceColor.x)) {
        for (auto& isoSurfaceRayCastingPass : isoSurfaceRayCastingPasses) {
            isoSurfaceRayCastingPass->setIsoSurfaceColor(isoSurfaceColor);
        }
        reRender = true;
    }
}



IsoSurfaceRayCastingPass::IsoSurfaceRayCastingPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : ComputePass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    rendererUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(RenderSettingsData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void IsoSurfaceRayCastingPass::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;

    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    float voxelSizeX = aabb.getDimensions().x / float(volumeData->getGridSizeX());
    float voxelSizeY = aabb.getDimensions().y / float(volumeData->getGridSizeY());
    float voxelSizeZ = aabb.getDimensions().z / float(volumeData->getGridSizeZ());
    voxelSize = std::min(voxelSizeX, std::min(voxelSizeY, voxelSizeZ));
    renderSettingsData.minBoundingBox = aabb.min;
    renderSettingsData.maxBoundingBox = aabb.max;
    renderSettingsData.stepSize = voxelSize * stepSize;
    renderSettingsData.dx = volumeData->getDx();
    renderSettingsData.dy = volumeData->getDy();
    renderSettingsData.dz = volumeData->getDz();

    dataDirty = true;
}

void IsoSurfaceRayCastingPass::setSelectedScalarFieldName(const std::string& _field) {
    selectedScalarFieldName = _field;
    if (volumeData && computeData) {
        auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
        computeData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    }
}

void IsoSurfaceRayCastingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    if (analyticIntersections) {
        preprocessorDefines.insert(std::make_pair("ANALYTIC_INTERSECTIONS", ""));
    }
    if (intersectionSolver == IntersectionSolver::LINEAR_INTERPOLATION) {
        preprocessorDefines.insert(std::make_pair("SOLVER_LINEAR_INTERPOLATION", ""));
    } else if (intersectionSolver == IntersectionSolver::NEUBAUER) {
        preprocessorDefines.insert(std::make_pair("SOLVER_NEUBAUER", ""));
    } else if (intersectionSolver == IntersectionSolver::MARMITT) {
        preprocessorDefines.insert(std::make_pair("SOLVER_MARMITT", ""));
    } else if (intersectionSolver == IntersectionSolver::SCHWARZE) {
        preprocessorDefines.insert(std::make_pair("SOLVER_SCHWARZE", ""));
    }
    if (sceneData->useDepthBuffer) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_DEPTH_BUFFER", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"IsoSurfaceRayCasting.Compute"}, preprocessorDefines);
}

void IsoSurfaceRayCastingPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    volumeData->setRenderDataBindings(computeData);
    computeData->setStaticBuffer(rendererUniformDataBuffer, "RendererUniformDataBuffer");
    computeData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    computeData->setStaticImageView(sceneImageView, "outputImage");
    if (sceneData->useDepthBuffer) {
        computeData->setStaticImageView(sceneData->sceneDepthColorImage, "depthBuffer");
    }
}

void IsoSurfaceRayCastingPass::recreateSwapchain(uint32_t width, uint32_t height) {
    sceneImageView = (*sceneData->sceneTexture)->getImageView();

    if (computeData) {
        computeData->setStaticImageView(sceneImageView, "outputImage");
        if (sceneData->useDepthBuffer) {
            computeData->setStaticImageView(sceneData->sceneDepthColorImage, "depthBuffer");
        }
    }
}

void IsoSurfaceRayCastingPass::_render() {
    renderSettingsData.stepSize = voxelSize * stepSize;
    renderSettingsData.inverseViewMatrix = glm::inverse((*camera)->getViewMatrix());
    renderSettingsData.inverseProjectionMatrix = glm::inverse((*camera)->getProjectionMatrix());
    renderSettingsData.cameraPosition = (*camera)->getPosition();
    renderSettingsData.zNear = (*camera)->getNearClipDistance();
    renderSettingsData.zFar = (*camera)->getFarClipDistance();
    rendererUniformDataBuffer->updateData(
            sizeof(RenderSettingsData), &renderSettingsData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    //renderer->insertImageMemoryBarrier(
    //        sceneImageView->getImage(),
    //        sceneImageView->getImage()->getVkImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
    //        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    //        VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
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
