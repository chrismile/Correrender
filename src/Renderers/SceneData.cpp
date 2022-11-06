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

#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include "SceneData.hpp"

void SceneData::initDepthColor() {
    sgl::vk::ImageSettings depthColorImageSettings = (*sceneDepthTexture)->getImage()->getImageSettings();
    depthColorImageSettings.format = VK_FORMAT_R32_SFLOAT;
    depthColorImageSettings.usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    auto image = std::make_shared<sgl::vk::Image>((*renderer)->getDevice(), depthColorImageSettings);
    sceneDepthColorImage = std::make_shared<sgl::vk::ImageView>(image, VK_IMAGE_ASPECT_COLOR_BIT);

    sceneDepthColorBuffer = std::make_shared<sgl::vk::Buffer>(
            (*renderer)->getDevice(), depthColorImageSettings.width * depthColorImageSettings.height * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void SceneData::clearRenderTargetState() {
    colorState = RenderTargetAccess::CLEAR;
    depthState = RenderTargetAccess::CLEAR;
}

void SceneData::switchColorState(RenderTargetAccess access) {
    VkImageLayout oldLayout;
    VkImageLayout newLayout;
    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;

    if (colorState == RenderTargetAccess::COMPUTE) {
        oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else if (colorState == RenderTargetAccess::CLEAR) {
        oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (colorState == RenderTargetAccess::RASTERIZER) {
        oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (colorState == RenderTargetAccess::SAMPLED_FRAGMENT_SHADER) {
        oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else {
        sgl::Logfile::get()->throwError("Error in SceneData::switchDepthState: Unimplemented.");
        return;
    }

    if (access == RenderTargetAccess::COMPUTE) {
        newLayout = VK_IMAGE_LAYOUT_GENERAL;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else if (access == RenderTargetAccess::RASTERIZER) {
        newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (access == RenderTargetAccess::SAMPLED_FRAGMENT_SHADER) {
        newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else {
        sgl::Logfile::get()->throwError("Error in SceneData::switchDepthState: Unimplemented.");
        return;
    }

    (*renderer)->insertImageMemoryBarrier(
            (*sceneTexture)->getImage(), oldLayout, newLayout, srcStage, dstStage, srcAccessMask, dstAccessMask);

    colorState = access;
}

void SceneData::switchDepthState(RenderTargetAccess access) {
    VkImageLayout oldLayout;
    VkImageLayout newLayout;
    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;

    if (depthState == RenderTargetAccess::COMPUTE) {
        oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else if (depthState == RenderTargetAccess::CLEAR) {
        oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (depthState == RenderTargetAccess::RASTERIZER) {
        oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        srcStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else {
        sgl::Logfile::get()->throwError("Error in SceneData::switchDepthState: Unimplemented.");
        return;
    }

    if (access == RenderTargetAccess::COMPUTE) {
        newLayout = VK_IMAGE_LAYOUT_GENERAL;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else if (access == RenderTargetAccess::RASTERIZER) {
        newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    } else {
        sgl::Logfile::get()->throwError("Error in SceneData::switchDepthState: Unimplemented.");
        return;
    }

    if (depthState == RenderTargetAccess::CLEAR && access == RenderTargetAccess::COMPUTE) {
        // Avoid copy and clear instead.
        (*renderer)->insertImageMemoryBarrier(
                sceneDepthColorImage->getImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
        sceneDepthColorImage->getImage()->clearColor(
                glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), (*renderer)->getVkCommandBuffer());
    } else if (depthState != RenderTargetAccess::COMPUTE && access == RenderTargetAccess::COMPUTE) {
        (*renderer)->insertImageMemoryBarrier(
                (*sceneDepthTexture)->getImage(),
                oldLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                srcStage, VK_PIPELINE_STAGE_TRANSFER_BIT,
                srcAccessMask, VK_ACCESS_TRANSFER_READ_BIT);
        (*renderer)->insertImageMemoryBarrier(
                sceneDepthColorImage->getImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
        //(*sceneDepthTexture)->getImage()->copyToImage(
        //        sceneDepthColorImage->getImage(), VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        //        (*renderer)->getVkCommandBuffer());
        (*sceneDepthTexture)->getImage()->copyToBuffer(sceneDepthColorBuffer, (*renderer)->getVkCommandBuffer());
        (*renderer)->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                sceneDepthColorBuffer);
        sceneDepthColorImage->getImage()->copyFromBuffer(sceneDepthColorBuffer, (*renderer)->getVkCommandBuffer());
        oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (depthState == RenderTargetAccess::COMPUTE && access != RenderTargetAccess::COMPUTE) {
        (*renderer)->insertImageMemoryBarrier(
                sceneDepthColorImage->getImage(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        (*renderer)->insertImageMemoryBarrier(
                (*sceneDepthTexture)->getImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
        //sceneDepthColorImage->getImage()->copyToImage(
        //        (*sceneDepthTexture)->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
        //        (*renderer)->getVkCommandBuffer());
        sceneDepthColorImage->getImage()->copyToBuffer(sceneDepthColorBuffer, (*renderer)->getVkCommandBuffer());
        (*renderer)->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                sceneDepthColorBuffer);
        (*sceneDepthTexture)->getImage()->copyFromBuffer(sceneDepthColorBuffer, (*renderer)->getVkCommandBuffer());
        oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    }

    sgl::vk::ImagePtr image;
    if (access == RenderTargetAccess::COMPUTE) {
        image = sceneDepthColorImage->getImage();
    } else {
        image = (*sceneDepthTexture)->getImage();
    }
    (*renderer)->insertImageMemoryBarrier(
            image, oldLayout, newLayout, srcStage, dstStage, srcAccessMask, dstAccessMask);

    depthState = access;
}
