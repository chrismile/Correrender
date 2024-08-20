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

#ifndef CORRERENDER_DEVICECACHEENTRY_HPP
#define CORRERENDER_DEVICECACHEENTRY_HPP

#include <string>
#include <memory>
#include <glm/vec3.hpp>

#include <Utils/SciVis/ScalarDataFormat.hpp>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

namespace sgl { namespace vk {
class Image;
typedef std::shared_ptr<Image> ImagePtr;
class ImageView;
typedef std::shared_ptr<ImageView> ImageViewPtr;
class ImageSampler;
typedef std::shared_ptr<ImageSampler> ImageSamplerPtr;
class Texture;
typedef std::shared_ptr<Texture> TexturePtr;
class ImageCudaExternalMemoryVk;
typedef std::shared_ptr<ImageCudaExternalMemoryVk> ImageCudaExternalMemoryVkPtr;
class TextureCudaExternalMemoryVk;
typedef std::shared_ptr<TextureCudaExternalMemoryVk> TextureCudaExternalMemoryVkPtr;
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
}}

std::string getImageFormatGlslString(const sgl::vk::ImagePtr& image);

class DeviceCacheEntryType {
public:
    // Image mode.
    DeviceCacheEntryType(sgl::vk::ImagePtr vulkanImage, sgl::vk::ImageSamplerPtr vulkanSampler);
    inline const sgl::vk::ImagePtr& getVulkanImage() { return vulkanImage; }
    [[nodiscard]] ScalarDataFormat getScalarDataFormat() const;
    [[nodiscard]] bool getIsRGBA() const;
    [[nodiscard]] std::string getImageFormatGlslString() const;
    const sgl::vk::ImageViewPtr& getVulkanImageView();
    const sgl::vk::TexturePtr& getVulkanTexture();
    void setImageSampler(const sgl::vk::ImageSamplerPtr& _vulkanSampler);
#ifdef SUPPORT_CUDA_INTEROP
    CUtexObject getCudaTexture();
    sgl::vk::TextureCudaExternalMemoryVkPtr getTextureCudaExternalMemory();
    const sgl::vk::ImageCudaExternalMemoryVkPtr& getImageCudaExternalMemory();
#endif

    // Buffer mode.
    explicit DeviceCacheEntryType(sgl::vk::BufferPtr vulkanBuffer);
    DeviceCacheEntryType(sgl::vk::BufferPtr vulkanBuffer, uint32_t tileSizeX, uint32_t tileSizeY, uint32_t tileSizeZ);
    inline const sgl::vk::BufferPtr& getVulkanBuffer() { return vulkanBuffer; }
    [[nodiscard]] inline uint32_t getBufferTileSizeX() const { return tileSizeX; }
    [[nodiscard]] inline uint32_t getBufferTileSizeY() const { return tileSizeY; }
    [[nodiscard]] inline uint32_t getBufferTileSizeZ() const { return tileSizeZ; }
    [[nodiscard]] inline glm::uvec3 getBufferTileSize() const { return { tileSizeX, tileSizeY, tileSizeZ }; }
#ifdef SUPPORT_CUDA_INTEROP
    CUdeviceptr getCudaBuffer();
    const sgl::vk::BufferCudaExternalMemoryVkPtr& getBufferCudaExternalMemory();
#endif

private:
    sgl::vk::ImagePtr vulkanImage;
    sgl::vk::ImageSamplerPtr vulkanSampler;
    sgl::vk::BufferPtr vulkanBuffer;
    uint32_t tileSizeX = 1, tileSizeY = 1, tileSizeZ = 1;

    /// Optional, created when @see getVulkanImageView or @see getVulkanTexture are called.
    sgl::vk::ImageViewPtr vulkanImageView;

    /// Optional, created when @see getVulkanTexture is called.
    sgl::vk::TexturePtr vulkanTexture;
#ifdef SUPPORT_CUDA_INTEROP
    /// Optional, created when @see getCudaTexture is called.
    sgl::vk::TextureCudaExternalMemoryVkPtr cudaTexture;
#endif

#ifdef SUPPORT_CUDA_INTEROP
    /// Optional, created when @see getCudaBuffer is called.
    sgl::vk::BufferCudaExternalMemoryVkPtr cudaBuffer;
#endif
};

typedef std::shared_ptr<DeviceCacheEntryType> DeviceCacheEntry;

#endif //CORRERENDER_DEVICECACHEENTRY_HPP
