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

#include <utility>

#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>

#include "DeviceCacheEntry.hpp"

DeviceCacheEntryType::DeviceCacheEntryType(sgl::vk::ImagePtr vulkanImage, sgl::vk::ImageSamplerPtr vulkanSampler)
        : vulkanImage(std::move(vulkanImage)), vulkanSampler(std::move(vulkanSampler)) {
}

std::string getImageFormatGlslString(const sgl::vk::ImagePtr& image) {
    auto format = image->getImageSettings().format;
    if (format == VK_FORMAT_R32_SFLOAT) {
        return "r32f";
    } else if (format == VK_FORMAT_R32G32B32A32_SFLOAT) {
        return "rgba32f";
    } else if (format == VK_FORMAT_R8_UNORM) {
        return "r8";
    } else if (format == VK_FORMAT_R8G8B8A8_UNORM) {
        return "rgba8";
    } else if (format == VK_FORMAT_R16_UNORM) {
        return "r16";
    } else if (format == VK_FORMAT_R16G16B16A16_UNORM) {
        return "rgba16";
    } else {
        sgl::Logfile::get()->throwError("Error in getImageFormatGlslString: Invalid format.");
        return "r32f";
    }
}

ScalarDataFormat DeviceCacheEntryType::getScalarDataFormat() const {
    auto format = vulkanImage->getImageSettings().format;
    if (format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R32G32B32A32_SFLOAT) {
        return ScalarDataFormat::FLOAT;
    } else if (format == VK_FORMAT_R8_UNORM || format == VK_FORMAT_R8G8B8A8_UNORM) {
        return ScalarDataFormat::BYTE;
    } else if (format == VK_FORMAT_R16_UNORM || format == VK_FORMAT_R16G16B16A16_UNORM) {
        return ScalarDataFormat::SHORT;
    } else if (format == VK_FORMAT_R16_SFLOAT || format == VK_FORMAT_R16G16B16A16_SFLOAT) {
        return ScalarDataFormat::FLOAT16;
    } else {
        sgl::Logfile::get()->throwError("Error in DeviceCacheEntryType::getScalarDataFormat: Invalid format.");
        return ScalarDataFormat::FLOAT;
    }
}

std::string DeviceCacheEntryType::getImageFormatGlslString() const {
    return ::getImageFormatGlslString(vulkanImage);
}

const sgl::vk::ImageViewPtr& DeviceCacheEntryType::getVulkanImageView() {
    if (!vulkanImageView) {
        vulkanImageView = std::make_shared<sgl::vk::ImageView>(vulkanImage, VK_IMAGE_VIEW_TYPE_3D);
    }
    return vulkanImageView;
}


const sgl::vk::TexturePtr& DeviceCacheEntryType::getVulkanTexture() {
    if (!vulkanTexture) {
        vulkanTexture = std::make_shared<sgl::vk::Texture>(getVulkanImageView(), vulkanSampler);
    }
    return vulkanTexture;
}

#ifdef SUPPORT_CUDA_INTEROP
CUtexObject DeviceCacheEntryType::getCudaTexture() {
    if (!cudaTexture) {
        sgl::vk::TextureCudaExternalMemorySettings texCudaSettings{};
        //texCudaSettings.useNormalizedCoordinates = true;
        texCudaSettings.useNormalizedCoordinates = false;
        cudaTexture = std::make_shared<sgl::vk::TextureCudaExternalMemoryVk>(
                vulkanImage, vulkanSampler->getImageSamplerSettings(), VK_IMAGE_VIEW_TYPE_3D, texCudaSettings);
    }
    return cudaTexture->getCudaTextureObject();
}

sgl::vk::TextureCudaExternalMemoryVkPtr DeviceCacheEntryType::getTextureCudaExternalMemory() {
    if (!cudaTexture) {
        sgl::vk::TextureCudaExternalMemorySettings texCudaSettings{};
        //texCudaSettings.useNormalizedCoordinates = true;
        texCudaSettings.useNormalizedCoordinates = false;
        cudaTexture = std::make_shared<sgl::vk::TextureCudaExternalMemoryVk>(
                vulkanImage, vulkanSampler->getImageSamplerSettings(), VK_IMAGE_VIEW_TYPE_3D, texCudaSettings);
    }
    return cudaTexture;
}

const sgl::vk::ImageCudaExternalMemoryVkPtr& DeviceCacheEntryType::getImageCudaExternalMemory() {
    if (!cudaTexture) {
        sgl::vk::TextureCudaExternalMemorySettings texCudaSettings{};
        //texCudaSettings.useNormalizedCoordinates = true;
        texCudaSettings.useNormalizedCoordinates = false;
        cudaTexture = std::make_shared<sgl::vk::TextureCudaExternalMemoryVk>(
                vulkanImage, vulkanSampler->getImageSamplerSettings(), VK_IMAGE_VIEW_TYPE_3D, texCudaSettings);
    }
    return cudaTexture->getImageCudaExternalMemory();
}
#endif


DeviceCacheEntryType::DeviceCacheEntryType(sgl::vk::BufferPtr vulkanBuffer) : vulkanBuffer(std::move(vulkanBuffer)) {
}

DeviceCacheEntryType::DeviceCacheEntryType(
        sgl::vk::BufferPtr vulkanBuffer, uint32_t tileSizeX, uint32_t tileSizeY, uint32_t tileSizeZ)
        : vulkanBuffer(std::move(vulkanBuffer)), tileSizeX(tileSizeX), tileSizeY(tileSizeY), tileSizeZ(tileSizeZ) {
}

#ifdef SUPPORT_CUDA_INTEROP
CUdeviceptr DeviceCacheEntryType::getCudaBuffer() {
    if (!cudaBuffer) {
        cudaBuffer = std::make_shared<sgl::vk::BufferCudaExternalMemoryVk>(vulkanBuffer);
    }
    return cudaBuffer->getCudaDevicePtr();
}

const sgl::vk::BufferCudaExternalMemoryVkPtr& DeviceCacheEntryType::getBufferCudaExternalMemory() {
    if (!cudaBuffer) {
        cudaBuffer = std::make_shared<sgl::vk::BufferCudaExternalMemoryVk>(vulkanBuffer);
    }
    return cudaBuffer;
}
#endif
