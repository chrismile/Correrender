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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "GradientPass.hpp"

GradientPass::GradientPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
}

void GradientPass::setInputImages(
        const sgl::vk::ImageViewPtr& _inputImageGT,
        const sgl::vk::ImageViewPtr& _inputImageOpt) {
    if (inputImageGT != _inputImageGT) {
        bool formatMatches = true;
        if (inputImageGT) {
            formatMatches =
                    getImageFormatGlslString(inputImageGT->getImage())
                    == getImageFormatGlslString(_inputImageGT->getImage());
        }
        inputImageGT = _inputImageGT;
        if (formatMatches && computeData) {
            computeData->setStaticImageView(inputImageGT, "inputImageGT");
        }
        if (!formatMatches) {
            setShaderDirty();
        }
    }
    if (inputImageOpt != _inputImageOpt) {
        bool formatMatches = true;
        if (inputImageOpt) {
            formatMatches =
                    getImageFormatGlslString(inputImageOpt->getImage())
                    == getImageFormatGlslString(_inputImageOpt->getImage());
        }
        inputImageOpt = _inputImageOpt;
        if (formatMatches && computeData) {
            computeData->setStaticImageView(inputImageOpt, "inputImageOpt");
        }
        if (!formatMatches) {
            setShaderDirty();
        }
    }
}

void GradientPass::setBuffers(
        uint32_t _tfSize,
        const sgl::vk::BufferPtr& _settingsBuffer,
        const sgl::vk::BufferPtr& _tfGTBuffer,
        const sgl::vk::BufferPtr& _tfOptBuffer,
        const sgl::vk::BufferPtr& _tfOptGradientBuffer) {
    tfSize = _tfSize;
    if (uniformBuffer != _settingsBuffer) {
        uniformBuffer = _settingsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
        }
    }
    if (tfGTBuffer != _tfGTBuffer) {
        tfGTBuffer = _tfGTBuffer;
        if (computeData) {
            computeData->setStaticBuffer(tfGTBuffer, "TfGTBuffer");
        }
    }
    if (tfOptBuffer != _tfOptBuffer) {
        tfOptBuffer = _tfOptBuffer;
        if (computeData) {
            computeData->setStaticBuffer(tfOptBuffer, "TfOptBuffer");
        }
    }
    if (tfOptGradientBuffer != _tfOptGradientBuffer) {
        tfOptGradientBuffer = _tfOptGradientBuffer;
        if (computeData) {
            computeData->setStaticBuffer(tfOptGradientBuffer, "TfOptGradientBuffer");
        }
    }
}

void GradientPass::setSettings(LossType _lossType) {
    if (lossType != _lossType) {
        lossType = _lossType;
        setShaderDirty();
    }
}

void GradientPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair(
            "INPUT_IMAGE_0_FORMAT", getImageFormatGlslString(inputImageGT->getImage())));
    preprocessorDefines.insert(std::make_pair(
            "INPUT_IMAGE_1_FORMAT", getImageFormatGlslString(inputImageOpt->getImage())));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    if (lossType == LossType::L1) {
        preprocessorDefines.insert(std::make_pair("L1_LOSS", ""));
    } else if (lossType == LossType::L2) {
        preprocessorDefines.insert(std::make_pair("L2_LOSS", ""));
    }
    if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_BUFFER_FLOAT_ATOMIC_ADD", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "Loss.Compute.Voxels" }, preprocessorDefines);
}

void GradientPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticImageView(inputImageGT, "inputImageGT");
    computeData->setStaticImageView(inputImageOpt, "inputImageOpt");
    computeData->setStaticBuffer(tfGTBuffer, "TfGTBuffer");
    computeData->setStaticBuffer(tfOptBuffer, "TfOptBuffer");
    computeData->setStaticBuffer(tfOptGradientBuffer, "TfOptGradientBuffer");
}

void GradientPass::_render() {
    tfOptGradientBuffer->fill(0, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    auto Nj = float(tfSize - 1u);
    renderer->pushConstants(computeData->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, Nj);
    const auto& imageSettings = inputImageGT->getImage()->getImageSettings();
    renderer->dispatch(
            computeData,
            sgl::uiceil(imageSettings.width, computeBlockSizeX),
            sgl::uiceil(imageSettings.height, computeBlockSizeY),
            sgl::uiceil(imageSettings.depth, computeBlockSizeZ));
}
