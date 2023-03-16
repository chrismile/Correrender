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

#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "ConnectingLineRasterPass.hpp"

ConnectingLineRasterPass::ConnectingLineRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : RasterPass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    uniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    std::vector<uint32_t> triangleIndices;
    triangleIndices.reserve(tubeNumSubdivisions * 6);
    const uint32_t indexOffsetCurrent = 0;
    const uint32_t indexOffsetNext = tubeNumSubdivisions;
    for (int k = 0; k < tubeNumSubdivisions; k++) {
        int kNext = (k + 1) % tubeNumSubdivisions;

        triangleIndices.push_back(indexOffsetCurrent + k);
        triangleIndices.push_back(indexOffsetCurrent + kNext);
        triangleIndices.push_back(indexOffsetNext + k);

        triangleIndices.push_back(indexOffsetNext + k);
        triangleIndices.push_back(indexOffsetCurrent + kNext);
        triangleIndices.push_back(indexOffsetNext + kNext);
    }
    indexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * triangleIndices.size(), triangleIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void ConnectingLineRasterPass::setLineSettings(const std::pair<glm::vec3, glm::vec3>& points, float lineWidth) {
    uniformData.p0 = points.first;
    uniformData.p1 = points.second;
    uniformData.lineWidth = lineWidth;
}

void ConnectingLineRasterPass::setCustomColors(const glm::vec4& c0, const glm::vec4& c1) {
    uniformData.c0 = c0;
    uniformData.c1 = c1;
    useCustomColors = true;
}

void ConnectingLineRasterPass::resetCustomColors() {
    useCustomColors = false;
}

void ConnectingLineRasterPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    if (sceneData->useDepthBuffer) {
        preprocessorDefines.insert(std::make_pair("NUM_TUBE_SUBDIVISIONS", std::to_string(tubeNumSubdivisions)));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "ConnectingLine.Vertex", "ConnectingLine.Fragment" }, preprocessorDefines);
}

void ConnectingLineRasterPass::setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
    pipelineInfo.setInputAssemblyTopology(sgl::vk::PrimitiveTopology::TRIANGLE_LIST);
    pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
}

void ConnectingLineRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setStaticBuffer(uniformDataBuffer, "UniformDataBuffer");
}

void ConnectingLineRasterPass::recreateSwapchain(uint32_t width, uint32_t height) {
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

void ConnectingLineRasterPass::_render() {
    glm::vec3 backgroundColor = sceneData->clearColor->getFloatColorRGB();
    glm::vec3 foregroundColor = glm::vec3(1.0f) - backgroundColor;

    if (!useCustomColors) {
        uniformData.c0 = glm::vec4(foregroundColor.x, foregroundColor.y, foregroundColor.z, 1.0f);
        uniformData.c1 = glm::vec4(foregroundColor.x, foregroundColor.y, foregroundColor.z, 1.0f);
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
