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

#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Renderers/SceneData.hpp"
#include "ShadowCircleRasterPass.hpp"

ShadowCircleRasterPass::ShadowCircleRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : RasterPass(renderer), sceneData(sceneData) {
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    const int NUM_SUBDIV = 40;
    std::vector<glm::vec3> vertexPositions;
    vertexPositions.reserve(NUM_SUBDIV + 1);
    for (int i = 0; i < NUM_SUBDIV; i++) {
        float phi = float(i) / float(NUM_SUBDIV) * sgl::TWO_PI;
        vertexPositions.emplace_back(std::cos(phi), std::sin(phi), 0.0f);
    }
    vertexPositions.emplace_back(0.0f, 0.0f, 0.0f);
    vertexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    std::vector<uint32_t> indexData;
    indexData.reserve(NUM_SUBDIV * 3);
    for (int i = 0; i < NUM_SUBDIV; i++) {
        indexData.push_back(i);
        indexData.push_back((i + 1) % NUM_SUBDIV);
        indexData.push_back(NUM_SUBDIV);
    }
    indexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * indexData.size(), indexData.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void ShadowCircleRasterPass::setCenter(const glm::vec3& center) {
    uniformData.center = center;
}

void ShadowCircleRasterPass::setRadius(float radius) {
    uniformData.radius = radius;
}

void ShadowCircleRasterPass::loadShader() {
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"ShadowCircle.Vertex", "ShadowCircle.Fragment"});
}

void ShadowCircleRasterPass::setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
    pipelineInfo.setInputAssemblyTopology(sgl::vk::PrimitiveTopology::TRIANGLE_LIST);
    pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
    pipelineInfo.setDepthWriteEnabled(false);
    pipelineInfo.setBlendMode(sgl::vk::BlendMode::BACK_TO_FRONT_STRAIGHT_ALPHA);
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
}

void ShadowCircleRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexBuffer, "vertexPosition");
    rasterData->setStaticBuffer(uniformDataBuffer, "UniformDataBuffer");
}

void ShadowCircleRasterPass::recreateSwapchain(uint32_t width, uint32_t height) {
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

void ShadowCircleRasterPass::_render() {
    uniformDataBuffer->updateData(sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);
    sceneData->switchDepthState(RenderTargetAccess::RASTERIZER);

    RasterPass::_render();
}
