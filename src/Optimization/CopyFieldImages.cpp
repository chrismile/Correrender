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

#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#ifdef CUDA_ENABLED
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

#include "CopyFieldImages.hpp"

void copyFieldImages(
        sgl::vk::Device* device, uint32_t xs, uint32_t ys, uint32_t zs,
        const sgl::vk::ImagePtr& fieldImageGT, const sgl::vk::ImagePtr& fieldImageOpt,
        CopyFieldImageDestinationData& copyFieldImageDestinationData,
        uint32_t fieldIdxGT, uint32_t fieldIdxOpt, bool isSampled, bool exportMemory) {
    sgl::vk::ImageViewPtr& inputImageGT = *copyFieldImageDestinationData.inputImageGT;
    sgl::vk::ImageViewPtr& inputImageOpt = *copyFieldImageDestinationData.inputImageOpt;
    auto formatGT = fieldImageGT->getImageSettings().format;
    auto formatOpt = fieldImageOpt->getImageSettings().format;
    if (!inputImageGT || !inputImageGT
            || inputImageGT->getImage()->getImageSettings().format != formatGT
            || inputImageOpt->getImage()->getImageSettings().format != formatOpt
            || inputImageGT->getImage()->getImageSettings().exportMemory != exportMemory
            || inputImageOpt->getImage()->getImageSettings().exportMemory != exportMemory
            || ((inputImageGT->getImage()->getImageSettings().usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) != isSampled
            || ((inputImageOpt->getImage()->getImageSettings().usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0) != isSampled
            || inputImageGT->getImage()->getImageSettings().width != xs
            || inputImageGT->getImage()->getImageSettings().height != ys
            || inputImageGT->getImage()->getImageSettings().depth != zs) {
        sgl::vk::ImageSettings imageSettings;
        imageSettings.width = xs;
        imageSettings.height = ys;
        imageSettings.depth = zs;
        imageSettings.imageType = VK_IMAGE_TYPE_3D;
        if (isSampled) {
            imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        } else {
            imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        }
        imageSettings.exportMemory = exportMemory;
        if (exportMemory) {
            imageSettings.useDedicatedAllocationForExportedMemory = false;
        }
        imageSettings.format = formatGT;
        inputImageGT = std::make_shared<sgl::vk::ImageView>(
                std::make_shared<sgl::vk::Image>(device, imageSettings), VK_IMAGE_VIEW_TYPE_3D);
        imageSettings.format = formatOpt;
        inputImageOpt = std::make_shared<sgl::vk::ImageView>(
                std::make_shared<sgl::vk::Image>(device, imageSettings), VK_IMAGE_VIEW_TYPE_3D);
#ifdef CUDA_ENABLED
        if (exportMemory && copyFieldImageDestinationData.cudaInputImageGT) {
            sgl::vk::TextureCudaExternalMemorySettings texCudaSettings{};
            texCudaSettings.useNormalizedCoordinates = false;
            sgl::vk::ImageSamplerSettings imageSamplerSettings{};
            *copyFieldImageDestinationData.cudaInputImageGT = std::make_shared<sgl::vk::TextureCudaExternalMemoryVk>(
                    inputImageGT->getImage(), imageSamplerSettings, VK_IMAGE_VIEW_TYPE_3D, texCudaSettings);
            *copyFieldImageDestinationData.cudaInputImageOpt = std::make_shared<sgl::vk::TextureCudaExternalMemoryVk>(
                    inputImageOpt->getImage(), imageSamplerSettings, VK_IMAGE_VIEW_TYPE_3D, texCudaSettings);
        }
#endif
    }

    auto layoutOldGT = fieldImageGT->getVkImageLayout();
    auto layoutOldOpt = fieldImageOpt->getVkImageLayout();
    device->waitIdle();

    auto commandBufferGraphics = device->beginSingleTimeCommands(device->getGraphicsQueueIndex());
    fieldImageGT->insertMemoryBarrier(
            commandBufferGraphics,
            layoutOldGT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
            device->getGraphicsQueueIndex(),
            device->getComputeQueueIndex());
    if (fieldIdxGT != fieldIdxOpt) {
        fieldImageOpt->insertMemoryBarrier(
                commandBufferGraphics,
                layoutOldOpt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
                device->getGraphicsQueueIndex(),
                device->getComputeQueueIndex());
    }
    device->endSingleTimeCommands(commandBufferGraphics, device->getGraphicsQueueIndex());

    auto commandBufferCompute = device->beginSingleTimeCommands(device->getComputeQueueIndex());
    fieldImageGT->insertMemoryBarrier(
            commandBufferCompute,
            layoutOldGT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
            device->getGraphicsQueueIndex(),
            device->getComputeQueueIndex());
    if (fieldIdxGT != fieldIdxOpt) {
        fieldImageOpt->insertMemoryBarrier(
                commandBufferCompute,
                layoutOldOpt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
                device->getGraphicsQueueIndex(),
                device->getComputeQueueIndex());
    }
    inputImageGT->getImage()->insertMemoryBarrier(
            commandBufferCompute,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
    inputImageOpt->getImage()->insertMemoryBarrier(
            commandBufferCompute,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
    fieldImageGT->copyToImage(
            inputImageGT->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, commandBufferCompute);
    fieldImageOpt->copyToImage(
            inputImageOpt->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, commandBufferCompute);
    fieldImageGT->insertMemoryBarrier(
            commandBufferCompute,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldGT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
            device->getComputeQueueIndex(),
            device->getGraphicsQueueIndex());
    if (fieldIdxGT != fieldIdxOpt) {
        fieldImageOpt->insertMemoryBarrier(
                commandBufferCompute,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldOpt,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
                device->getComputeQueueIndex(),
                device->getGraphicsQueueIndex());
    }
    VkImageLayout finalLayout = isSampled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    inputImageGT->getImage()->insertMemoryBarrier(
            commandBufferCompute,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, finalLayout,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    inputImageOpt->getImage()->insertMemoryBarrier(
            commandBufferCompute,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, finalLayout,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    device->endSingleTimeCommands(commandBufferCompute, device->getComputeQueueIndex());

    commandBufferGraphics = device->beginSingleTimeCommands(device->getGraphicsQueueIndex());
    fieldImageGT->insertMemoryBarrier(
            commandBufferGraphics,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldGT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
            device->getComputeQueueIndex(),
            device->getGraphicsQueueIndex());
    if (fieldIdxGT != fieldIdxOpt) {
        fieldImageOpt->insertMemoryBarrier(
                commandBufferGraphics,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldOpt,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
                device->getComputeQueueIndex(),
                device->getGraphicsQueueIndex());
    }
    device->endSingleTimeCommands(commandBufferGraphics, device->getGraphicsQueueIndex());
}
