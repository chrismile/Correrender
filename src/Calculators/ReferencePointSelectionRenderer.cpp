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

#include <Math/Math.hpp>
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
#include "ReferencePointSelectionRenderer.hpp"

void getSphereSurfaceRenderData(
        const glm::vec3& center, float radius, int sectorCount, int stackCount,
        std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec3>& vertexNormals,
        std::vector<uint32_t>& triangleIndices) {
    float phi, theta, sinPhi, cosPhi;
    float sectorStep = sgl::TWO_PI / float(sectorCount);
    float stackStep = sgl::PI / float(stackCount);

    // 1. Build the vertex buffers.
    for (int stackIdx = 0; stackIdx <= stackCount; ++stackIdx) {
        phi = sgl::HALF_PI - float(stackIdx) * stackStep;
        cosPhi = std::cos(phi);
        sinPhi = std::sin(phi);
        for (int sectorIdx = 0; sectorIdx <= sectorCount; ++sectorIdx) {
            theta = float(sectorIdx) * sectorStep;
            glm::vec3 normal(cosPhi * std::cos(theta), cosPhi * std::sin(theta), sinPhi);
            glm::vec3 position = center + radius * normal;
            vertexPositions.push_back(position);
            vertexNormals.push_back(normal);
        }
    }

    // 2. Build the index buffer.
    uint32_t k1, k2;
    for (int stackIdx = 0; stackIdx <= stackCount; ++stackIdx) {
        k1 = stackIdx * (sectorCount + 1);
        k2 = k1 + sectorCount + 1;
        for (int sectorIdx = 0; sectorIdx <= sectorCount; ++sectorIdx) {
            if(stackIdx != 0) {
                triangleIndices.push_back(k1);
                triangleIndices.push_back(k2);
                triangleIndices.push_back(k1 + 1);
            }
            if(stackIdx != (stackCount-1)) {
                triangleIndices.push_back(k1 + 1);
                triangleIndices.push_back(k2);
                triangleIndices.push_back(k2 + 1);
            }
            k1++;
            k2++;
        }
    }
}

ReferencePointSelectionRenderer::ReferencePointSelectionRenderer(ViewManager* viewManager)
        : Renderer("Reference Point Selection Renderer", viewManager) {
}

void ReferencePointSelectionRenderer::initialize() {
    Renderer::initialize();

    std::vector<glm::vec3> sphereVertexPositions;
    std::vector<glm::vec3> sphereVertexNormals;
    std::vector<uint32_t> sphereIndices;
    getSphereSurfaceRenderData(
            glm::vec3(0,0,0), 1.0f, 32, 32,
            sphereVertexPositions, sphereVertexNormals, sphereIndices);

    sgl::vk::Device* device = renderer->getDevice();
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * sphereVertexPositions.size(), sphereVertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexNormalBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * sphereVertexNormals.size(), sphereVertexNormals.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    indexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * sphereIndices.size(), sphereIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void ReferencePointSelectionRenderer::setVolumeDataPtr(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
    for (auto& referencePointSelectionRasterPass : referencePointSelectionRasterPasses) {
        referencePointSelectionRasterPass->setVolumeData(volumeData, isNewData);
    }
}

void ReferencePointSelectionRenderer::setReferencePosition(const glm::ivec3& _referencePosition) {
    referencePosition = _referencePosition;
    for (auto& referencePointSelectionRasterPass : referencePointSelectionRasterPasses) {
        referencePointSelectionRasterPass->setReferencePosition(referencePosition);
    }
}

void ReferencePointSelectionRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    referencePointSelectionRasterPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void ReferencePointSelectionRenderer::renderViewImpl(uint32_t viewIdx) {
    referencePointSelectionRasterPasses.at(viewIdx)->render();
}

void ReferencePointSelectionRenderer::addViewImpl(uint32_t viewIdx) {
    auto referencePointSelectionRasterPass = std::make_shared<ReferencePointSelectionRasterPass>(
            renderer, viewManager->getViewSceneData(viewIdx));
    if (volumeData) {
        referencePointSelectionRasterPass->setVolumeData(volumeData, true);
    }
    referencePointSelectionRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer);
    referencePointSelectionRasterPass->setReferencePosition(referencePosition);
    referencePointSelectionRasterPass->setSphereRadius(0.006f);
    referencePointSelectionRasterPass->setSphereColor(sgl::Color(255, 40, 0).getFloatColorRGBA());
    referencePointSelectionRasterPasses.push_back(referencePointSelectionRasterPass);
}

void ReferencePointSelectionRenderer::removeViewImpl(uint32_t viewIdx) {
    referencePointSelectionRasterPasses.erase(referencePointSelectionRasterPasses.begin() + viewIdx);
}

void ReferencePointSelectionRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
}



ReferencePointSelectionRasterPass::ReferencePointSelectionRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : RasterPass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void ReferencePointSelectionRasterPass::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    this->volumeData = _volumeData;
    dataDirty = true;
    setReferencePosition(referencePosition);
}

void ReferencePointSelectionRasterPass::setReferencePosition(const glm::ivec3& _referencePosition) {
    referencePosition = _referencePosition;
    if (volumeData) {
        sgl::AABB3 gridAabb = volumeData->getBoundingBoxRendering();
        /*gridAabb.min = glm::vec3(-0.5f, -0.5f, -0.5f);
        gridAabb.max = glm::vec3(volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ()) - glm::vec3(0.5f, 0.5f, 0.5f);
        gridAabb.min *= glm::vec3(volumeData->getDx(), volumeData->getDy(), volumeData->getDz());
        gridAabb.max *= glm::vec3(volumeData->getDx(), volumeData->getDy(), volumeData->getDz());*/
        uniformData.spherePosition = glm::vec3(referencePosition);
        uniformData.spherePosition /= glm::vec3(
                volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
        uniformData.spherePosition = uniformData.spherePosition * (gridAabb.max - gridAabb.min) + gridAabb.min;
        //normalizeVertexPosition(uniformData.spherePosition, gridAabb, nullptr);
    }
}

void ReferencePointSelectionRasterPass::setRenderData(
        const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
        const sgl::vk::BufferPtr& _vertexNormalBuffer) {
    indexBuffer = _indexBuffer;
    vertexPositionBuffer = _vertexPositionBuffer;
    vertexNormalBuffer = _vertexNormalBuffer;

    setDataDirty();
}

void ReferencePointSelectionRasterPass::loadShader() {
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Sphere.Vertex", "Sphere.Fragment"});
}

void ReferencePointSelectionRasterPass::setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
    pipelineInfo.setInputAssemblyTopology(sgl::vk::PrimitiveTopology::TRIANGLE_LIST);
    pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_BACK);
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexNormal", sizeof(glm::vec3));
}

void ReferencePointSelectionRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
    rasterData->setVertexBuffer(vertexNormalBuffer, "vertexNormal");
    rasterData->setStaticBuffer(uniformDataBuffer, "UniformDataBuffer");
}

void ReferencePointSelectionRasterPass::recreateSwapchain(uint32_t width, uint32_t height) {
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

void ReferencePointSelectionRasterPass::_render() {
    glm::vec3 backgroundColor = sceneData->clearColor->getFloatColorRGB();
    glm::vec3 foregroundColor = glm::vec3(1.0f) - backgroundColor;

    uniformData.cameraPosition = (*camera)->getPosition();
    uniformData.foregroundColor = foregroundColor;
    uniformDataBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);
    sceneData->switchDepthState(RenderTargetAccess::RASTERIZER);

    RasterPass::_render();
}
