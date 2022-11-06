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

#include "DeviceCacheEntry.hpp"

#include <utility>

DeviceCacheEntryType::DeviceCacheEntryType(sgl::vk::ImagePtr vulkanImage, sgl::vk::ImageSamplerPtr vulkanSampler)
        : vulkanImage(std::move(vulkanImage)), vulkanSampler(std::move(vulkanSampler)) {
}

const sgl::vk::TexturePtr& DeviceCacheEntryType::getVulkanTexture() {
    if (!vulkanTexture) {
        sgl::vk::ImageViewPtr imageView = std::make_shared<sgl::vk::ImageView>(vulkanImage, VK_IMAGE_VIEW_TYPE_3D);
        vulkanTexture = std::make_shared<sgl::vk::Texture>(imageView, vulkanSampler);
    }

    return vulkanTexture;
}

#ifdef SUPPORT_CUDA_INTEROP
CUtexObject DeviceCacheEntryType::getCudaTexture() {
    if (!cudaTexture) {
        cudaTexture = std::make_shared<sgl::vk::TextureCudaExternalMemoryVk>(
                vulkanImage, vulkanSampler->getImageSamplerSettings(), VK_IMAGE_VIEW_TYPE_3D);
    }

    return cudaTexture->getCudaTextureObject();
}

const sgl::vk::ImageCudaExternalMemoryVkPtr& DeviceCacheEntryType::getImageCudaExternalMemory() {
    if (!cudaTexture) {
        cudaTexture = std::make_shared<sgl::vk::TextureCudaExternalMemoryVk>(
                vulkanImage, vulkanSampler->getImageSamplerSettings(), VK_IMAGE_VIEW_TYPE_3D);
    }

    return cudaTexture->getImageCudaExternalMemory();
}
#endif
