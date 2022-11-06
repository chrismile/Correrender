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

#include "IsoSurfaceRasterizer.hpp"

#include <Utils/Mesh/IndexMesh.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include <IsosurfaceCpp/src/MarchingCubes.hpp>
#include <IsosurfaceCpp/src/SnapMC.hpp>

#include "Utils/Normalization.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "RenderingModes.hpp"
#include "IsoSurfaceRasterizer.hpp"

IsoSurfaceRasterizer::IsoSurfaceRasterizer(ViewManager* viewManager, sgl::TransferFunctionWindow& transferFunctionWindow)
        : Renderer(
                RENDERING_MODE_NAMES[int(RENDERING_MODE_ISOSURFACE_RASTERIZER)],
                viewManager, transferFunctionWindow) {
    ;
}

void IsoSurfaceRasterizer::initialize() {
    Renderer::initialize();
}

void IsoSurfaceRasterizer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    volumeData = _volumeData;
    if (isNewData) {
        selectedFieldIdx = 0;
    }
    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    selectedScalarFieldName = fieldNames.at(selectedFieldIdx);

    indexBuffer = {};
    vertexPositionBuffer = {};
    vertexNormalBuffer = {};

    sgl::AABB3 gridAabb;
    //gridAabb.min = glm::vec3(0.0f, 0.0f, 0.0f);
    //gridAabb.max = glm::vec3(volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ());
    gridAabb.min = glm::vec3(-0.5f, -0.5f, -0.5f);
    gridAabb.max = glm::vec3(volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ()) - glm::vec3(0.5f, 0.5f, 0.5f);

    auto scalarFieldData = volumeData->getFieldEntryCpu(FieldType::SCALAR, selectedScalarFieldName);

    std::vector<glm::vec3> isosurfaceVertexPositions;
    std::vector<glm::vec3> isosurfaceVertexNormals;
    if (isoSurfaceExtractionTechnique == IsoSurfaceExtractionTechnique::MARCHING_CUBES) {
        polygonizeMarchingCubes(
                scalarFieldData.get(),
                volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ(),
                isoValue, isosurfaceVertexPositions, isosurfaceVertexNormals);
    } else {
        polygonizeSnapMC(
                scalarFieldData.get(),
                volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ(),
                isoValue, gammaSnapMC, isosurfaceVertexPositions, isosurfaceVertexNormals);
    }

    std::vector<uint32_t> triangleIndices;
    std::vector<glm::vec3> vertexPositions;
    std::vector<glm::vec3> vertexNormals;
    computeSharedIndexRepresentation(
            isosurfaceVertexPositions, isosurfaceVertexNormals,
            triangleIndices, vertexPositions, vertexNormals);
    normalizeVertexPositions(vertexPositions, gridAabb, nullptr);

    sgl::vk::Device* device = renderer->getDevice();
    indexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * triangleIndices.size(), triangleIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexNormalBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexNormals.size(), vertexNormals.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    for (auto& isoSurfaceRasterPass : isoSurfaceRasterPasses) {
        isoSurfaceRasterPass->setVolumeData(volumeData, isNewData);
        isoSurfaceRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer);
        isoSurfaceRasterPass->setSelectedScalarFieldName(selectedScalarFieldName);
    }
}

void IsoSurfaceRasterizer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    isoSurfaceRasterPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void IsoSurfaceRasterizer::renderViewImpl(uint32_t viewIdx) {
    isoSurfaceRasterPasses.at(viewIdx)->render();
}

void IsoSurfaceRasterizer::addViewImpl(uint32_t viewIdx) {
    auto isoSurfaceRasterPass = std::make_shared<IsoSurfaceRasterPass>(renderer, viewManager->getViewSceneData(viewIdx));
    if (volumeData) {
        isoSurfaceRasterPass->setVolumeData(volumeData, true);
        isoSurfaceRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer);
        isoSurfaceRasterPass->setSelectedScalarFieldName(selectedScalarFieldName);
    }
    isoSurfaceRasterPass->setIsoValue(isoValue);
    isoSurfaceRasterPass->setIsoSurfaceColor(isoSurfaceColor);
    isoSurfaceRasterPasses.push_back(isoSurfaceRasterPass);
}

void IsoSurfaceRasterizer::removeViewImpl(uint32_t viewIdx) {
    isoSurfaceRasterPasses.erase(isoSurfaceRasterPasses.begin() + viewIdx);
}

void IsoSurfaceRasterizer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        if (propertyEditor.addCombo("Scalar Field", &selectedFieldIdx, fieldNames.data(), int(fieldNames.size()))) {
            selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
            dirty = true;
            reRender = true;
        }
    }
    if (propertyEditor.addSliderFloatEdit("Iso Value", &isoValue, 0.0f, 1.0f) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& isoSurfaceRasterPass : isoSurfaceRasterPasses) {
            isoSurfaceRasterPass->setIsoValue(isoValue);
        }
        dirty = true;
        reRender = true;
    }
    if (propertyEditor.addCombo(
            "Extraction Technique", (int*)&isoSurfaceExtractionTechnique,
            ISO_SURFACE_EXTRACTION_TECHNIQUE_NAMES, IM_ARRAYSIZE(ISO_SURFACE_EXTRACTION_TECHNIQUE_NAMES))) {
        dirty = true;
        reRender = true;
    }
    if (propertyEditor.addColorEdit4("Iso Surface Color", &isoSurfaceColor.x)) {
        for (auto& isoSurfaceRasterPass : isoSurfaceRasterPasses) {
            isoSurfaceRasterPass->setIsoSurfaceColor(isoSurfaceColor);
        }
        reRender = true;
    }
    if (isoSurfaceExtractionTechnique == IsoSurfaceExtractionTechnique::SNAP_MC && propertyEditor.addSliderFloatEdit(
            "Gamma (SnapMC)", &gammaSnapMC, 0.0f, 1.0f) == ImGui::EditMode::INPUT_FINISHED) {
        dirty = true;
        reRender = true;
    }
}



IsoSurfaceRasterPass::IsoSurfaceRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : RasterPass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    rendererUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(RenderSettingsData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void IsoSurfaceRasterPass::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;
    dataDirty = true;
}

void IsoSurfaceRasterPass::setSelectedScalarFieldName(const std::string& _field) {
    selectedScalarFieldName = _field;
    /*if (volumeData && rasterData) {
        auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
        rasterData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    }*/
}

void IsoSurfaceRasterPass::setRenderData(
        const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
        const sgl::vk::BufferPtr& _vertexNormalBuffer) {
    indexBuffer = _indexBuffer;
    vertexPositionBuffer = _vertexPositionBuffer;
    vertexNormalBuffer = _vertexNormalBuffer;

    if (rasterData) {
        rasterData->setIndexBuffer(indexBuffer);
        rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
        rasterData->setVertexBuffer(vertexNormalBuffer, "vertexNormal");
    }
}

void IsoSurfaceRasterPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            {"IsoSurfaceRasterization.Vertex", "IsoSurfaceRasterization.Fragment"}, preprocessorDefines);
}

void IsoSurfaceRasterPass::setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
    pipelineInfo.setInputAssemblyTopology(sgl::vk::PrimitiveTopology::TRIANGLE_LIST);
    pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexNormal", sizeof(glm::vec3));
}

void IsoSurfaceRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    volumeData->setRenderDataBindings(rasterData);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
    rasterData->setVertexBuffer(vertexNormalBuffer, "vertexNormal");
    rasterData->setStaticBuffer(rendererUniformDataBuffer, "RendererUniformDataBuffer");
}

void IsoSurfaceRasterPass::recreateSwapchain(uint32_t width, uint32_t height) {
    framebuffer = std::make_shared<sgl::vk::Framebuffer>(device, width, height);

    sgl::vk::AttachmentState attachmentState;
    attachmentState.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachmentState.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentState.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    framebuffer->setColorAttachment(
            (*sceneData->sceneTexture)->getImageView(), 0, attachmentState,
            sceneData->clearColor->getFloatColorRGBA());

    sgl::vk::AttachmentState depthAttachmentState;
    depthAttachmentState.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachmentState.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachmentState.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    framebuffer->setDepthStencilAttachment(
            (*sceneData->sceneDepthTexture)->getImageView(), depthAttachmentState, 1.0f);

    framebufferDirty = true;
    dataDirty = true;
}

void IsoSurfaceRasterPass::_render() {
    renderSettingsData.cameraPosition = (*camera)->getPosition();
    rendererUniformDataBuffer->updateData(
            sizeof(RenderSettingsData), &renderSettingsData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);
    if (sceneData->useDepthBuffer) {
        sceneData->switchDepthState(RenderTargetAccess::RASTERIZER);
    }

    RasterPass::_render();
}
