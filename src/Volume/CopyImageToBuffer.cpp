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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "CopyImageToBuffer.hpp"

ImageToBufferCopyPass::ImageToBufferCopyPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void ImageToBufferCopyPass::setData(
        uint32_t xs, uint32_t ys, uint32_t zs,
        const sgl::vk::ImageViewPtr& _inputImage, const sgl::vk::BufferPtr& _outputBuffer,
        const glm::uvec3& _tileSize) {
    if (xs != uniformData.xs || xs != uniformData.ys || xs != uniformData.zs) {
        uniformData.xs = xs;
        uniformData.ys = ys;
        uniformData.zs = zs;
        uniformBuffer->updateData(sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    inputImage = _inputImage;
    outputBuffer = _outputBuffer;
    if (computeData) {
        bool formatMatches = imageFormatString == getImageFormatGlslString(_inputImage->getImage());
        if (formatMatches && computeData) {
            computeData->setStaticImageView(inputImage, "inputImage");
        }
        if (!formatMatches) {
            setShaderDirty();
        }
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
    imageFormatString = getImageFormatGlslString(inputImage->getImage());

    if (tileSize != _tileSize) {
        tileSize = _tileSize;
        setShaderDirty();
    }
}

void ImageToBufferCopyPass::resetData() {
    inputImage = {};
    outputBuffer = {};
    if (computeData) {
        computeData->setStaticImageView(inputImage, "inputImage");
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void ImageToBufferCopyPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("TILE_SIZE_X", std::to_string(tileSize.x)));
    preprocessorDefines.insert(std::make_pair("TILE_SIZE_Y", std::to_string(tileSize.y)));
    preprocessorDefines.insert(std::make_pair("TILE_SIZE_Z", std::to_string(tileSize.z)));
    preprocessorDefines.insert(std::make_pair("INPUT_IMAGE_FORMAT", imageFormatString));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"CopyImageToBuffer.Compute"}, preprocessorDefines);
}

void ImageToBufferCopyPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticImageView(inputImage, "inputImage");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}

void ImageToBufferCopyPass::_render() {
    renderer->dispatch(
            computeData,
            sgl::uiceil(uniformData.xs, tileSize.x),
            sgl::uiceil(uniformData.ys, tileSize.y),
            sgl::uiceil(uniformData.zs, tileSize.z));
}
