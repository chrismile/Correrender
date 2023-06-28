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

#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "RenderingModes.hpp"
#include "DomainOutlineRenderer.hpp"

DomainOutlineRenderer::DomainOutlineRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_DOMAIN_OUTLINE_RENDERER)], viewManager) {
}

DomainOutlineRenderer::~DomainOutlineRenderer() {
}

void DomainOutlineRenderer::initialize() {
    Renderer::initialize();

    const size_t numEdges = 12;

    sgl::vk::Device* device = renderer->getDevice();
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * numEdges * 8,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    indexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * numEdges * 6 * 6,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    triangleIndices.reserve(indexBuffer->getSizeInBytes() / sizeof(uint32_t));
    vertexPositions.reserve(vertexPositionBuffer->getSizeInBytes() / sizeof(glm::vec3));
}

void DomainOutlineRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    volumeData = _volumeData;
    recreateBuffers();
}

void DomainOutlineRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    domainOutlineRasterPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void DomainOutlineRenderer::renderViewImpl(uint32_t viewIdx) {
    if (useDepthCues) {
        domainOutlineRasterPasses.at(viewIdx)->setAabb(volumeData->getBoundingBoxRendering());
    }
    domainOutlineRasterPasses.at(viewIdx)->render();
}

void DomainOutlineRenderer::addViewImpl(uint32_t viewIdx) {
    auto domainOutlineRasterPass = std::make_shared<DomainOutlineRasterPass>(
            renderer, viewManager->getViewSceneData(viewIdx));
    domainOutlineRasterPass->setRenderData(indexBuffer, vertexPositionBuffer);
    domainOutlineRasterPass->setUseDepthCues(useDepthCues);
    domainOutlineRasterPasses.push_back(domainOutlineRasterPass);
}

void DomainOutlineRenderer::removeViewImpl(uint32_t viewIdx) {
    domainOutlineRasterPasses.erase(domainOutlineRasterPasses.begin() + viewIdx);
}

void DomainOutlineRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addSliderFloat("Line Width", &lineWidth, 0.001f, 0.020f, "%.4f")) {
        recreateBuffers();
        reRender = true;
    }
    if (propertyEditor.addCheckbox("Depth Cues", &useDepthCues)) {
        for (auto& domainOutlineRasterPass : domainOutlineRasterPasses) {
            domainOutlineRasterPass->setUseDepthCues(useDepthCues);
        }
        reRender = true;
    }
}

void DomainOutlineRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);
    if (settings.getValueOpt("line_width", lineWidth)) {
        recreateBuffers();
        reRender = true;
    }
    if (settings.getValueOpt("use_depth_cues", useDepthCues)) {
        for (auto& domainOutlineRasterPass : domainOutlineRasterPasses) {
            domainOutlineRasterPass->setUseDepthCues(useDepthCues);
        }
        reRender = true;
    }
}

void DomainOutlineRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);
    settings.addKeyValue("line_width", lineWidth);
    settings.addKeyValue("use_depth_cues", useDepthCues);
}

void addEdge(
        const glm::vec3& lower, const glm::vec3& upper,
        std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions) {
    uint32_t indexData[] = {
            0, 1, 2, 1, 3, 2, // front
            4, 6, 5, 5, 6, 7, // back
            0, 2, 4, 4, 2, 6, // left
            1, 5, 3, 5, 7, 3, // right
            0, 4, 1, 1, 4, 5, // bottom
            2, 3, 6, 3, 7, 6, // top
    };
    glm::vec3 vertexData[] = {
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(1.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(1.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
    };

    auto offset = uint32_t(vertexPositions.size());
    for (int i = 0; i < IM_ARRAYSIZE(indexData); i++) {
        triangleIndices.push_back(indexData[i] + offset);
    }
    for (auto pos : vertexData) {
        pos = pos * (upper - lower) + lower;
        vertexPositions.push_back(pos);
    }
}

void DomainOutlineRenderer::recreateBuffers() {
    triangleIndices.clear();
    vertexPositions.clear();

    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    glm::vec3 min0 = aabb.min - glm::vec3(lineWidth / 2.0f);
    glm::vec3 min1 = aabb.min + glm::vec3(lineWidth / 2.0f);
    glm::vec3 max0 = aabb.max - glm::vec3(lineWidth / 2.0f);
    glm::vec3 max1 = aabb.max + glm::vec3(lineWidth / 2.0f);
    addEdge(glm::vec3(min0.x, min0.y, min0.z), glm::vec3(max1.x, min1.y, min1.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(min0.x, min0.y, max0.z), glm::vec3(max1.x, min1.y, max1.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(min0.x, min0.y, min1.z), glm::vec3(min1.x, min1.y, max0.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(max0.x, min0.y, min1.z), glm::vec3(max1.x, min1.y, max0.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(min0.x, max0.y, min0.z), glm::vec3(max1.x, max1.y, min1.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(min0.x, max0.y, max0.z), glm::vec3(max1.x, max1.y, max1.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(min0.x, max0.y, min1.z), glm::vec3(min1.x, max1.y, max0.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(max0.x, max0.y, min1.z), glm::vec3(max1.x, max1.y, max0.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(min0.x, min1.y, min0.z), glm::vec3(min1.x, max0.y, min1.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(max0.x, min1.y, min0.z), glm::vec3(max1.x, max0.y, min1.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(min0.x, min1.y, max0.z), glm::vec3(min1.x, max0.y, max1.z), triangleIndices, vertexPositions);
    addEdge(glm::vec3(max0.x, min1.y, max0.z), glm::vec3(max1.x, max0.y, max1.z), triangleIndices, vertexPositions);

    indexBuffer->uploadData(indexBuffer->getSizeInBytes(), triangleIndices.data());
    vertexPositionBuffer->uploadData(vertexPositionBuffer->getSizeInBytes(), vertexPositions.data());
}



DomainOutlineRasterPass::DomainOutlineRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : RasterPass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void DomainOutlineRasterPass::setRenderData(
        const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer) {
    indexBuffer = _indexBuffer;
    vertexPositionBuffer = _vertexPositionBuffer;

    setDataDirty();
}

void DomainOutlineRasterPass::setCustomColor(const glm::vec4& color) {
    uniformData.objectColor = color;
    useCustomColor = true;
}

void DomainOutlineRasterPass::resetCustomColor() {
    useCustomColor = false;
}

void DomainOutlineRasterPass::setUseDepthCues(bool _useDepthCues) {
    if (useDepthCues != _useDepthCues) {
        useDepthCues = _useDepthCues;
        setShaderDirty();
    }
}

void DomainOutlineRasterPass::setAabb(const sgl::AABB3& aabb) {
    aabbMin = aabb.min;
    aabbMax = aabb.max;
}

void DomainOutlineRasterPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    if (useDepthCues) {
        preprocessorDefines.insert(std::make_pair("USE_DEPTH_CUES", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "DomainOutline.Vertex", "DomainOutline.Fragment" }, preprocessorDefines);
}

void DomainOutlineRasterPass::setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
    pipelineInfo.setInputAssemblyTopology(sgl::vk::PrimitiveTopology::TRIANGLE_LIST);
    pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
}

void DomainOutlineRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
    rasterData->setStaticBuffer(uniformDataBuffer, "UniformDataBuffer");
}

void DomainOutlineRasterPass::recreateSwapchain(uint32_t width, uint32_t height) {
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

void DomainOutlineRasterPass::_render() {
    glm::vec3 backgroundColor = sceneData->clearColor->getFloatColorRGB();
    glm::vec3 foregroundColor = glm::vec3(1.0f) - backgroundColor;

    if (useDepthCues) {
        uniformData.minDepth = std::numeric_limits<float>::max();
        uniformData.maxDepth = std::numeric_limits<float>::lowest();
        glm::vec4 corners[8] = {
                glm::vec4(aabbMin.x, aabbMin.y, aabbMin.z, 1.0f),
                glm::vec4(aabbMax.x, aabbMin.y, aabbMin.z, 1.0f),
                glm::vec4(aabbMin.x, aabbMax.y, aabbMin.z, 1.0f),
                glm::vec4(aabbMax.x, aabbMax.y, aabbMin.z, 1.0f),
                glm::vec4(aabbMin.x, aabbMin.y, aabbMax.z, 1.0f),
                glm::vec4(aabbMax.x, aabbMin.y, aabbMax.z, 1.0f),
                glm::vec4(aabbMin.x, aabbMax.y, aabbMax.z, 1.0f),
                glm::vec4(aabbMax.x, aabbMax.y, aabbMax.z, 1.0f),
        };
        for (int i = 0; i < 8; i++) {
            glm::vec4 screenSpacePosition = sceneData->camera->getViewMatrix() * glm::vec4(corners[i]);
            float depth = -screenSpacePosition.z;
            uniformData.minDepth = std::min(uniformData.minDepth, depth);
            uniformData.maxDepth = std::max(uniformData.maxDepth, depth);
        }
    }

    if (!useCustomColor) {
        uniformData.objectColor = glm::vec4(foregroundColor.x, foregroundColor.y, foregroundColor.z, 1.0f);
    }
    uniformDataBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);
    sceneData->switchDepthState(RenderTargetAccess::RASTERIZER);

    RasterPass::_render();
}



DomainOutlineComputePass::DomainOutlineComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
}

void DomainOutlineComputePass::setRenderData(
        const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer) {
    indexBuffer = _indexBuffer;
    vertexPositionBuffer = _vertexPositionBuffer;
    setDataDirty();
}

void DomainOutlineComputePass::setOutlineSettings(const sgl::AABB3& aabb, float lineWidth, float offset) {
    pushConstants.aabbMin = aabb.min;
    pushConstants.aabbMax = aabb.max;
    pushConstants.lineWidth = lineWidth;
    pushConstants.offset = offset;
}

void DomainOutlineComputePass::loadShader() {
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "DomainOutline.Compute" });
}

void DomainOutlineComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(indexBuffer, "IndexBuffer");
    computeData->setStaticBuffer(vertexPositionBuffer, "VertexBuffer");
}

void DomainOutlineComputePass::_render() {
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstants);
    renderer->dispatch(computeData, 1, 1, 1);
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
}
