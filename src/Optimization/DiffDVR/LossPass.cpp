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
#include "LossPass.hpp"

LossPass::LossPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void LossPass::setBuffers(
        const sgl::vk::BufferPtr& _finalColorsGTBuffer,
        const sgl::vk::BufferPtr& _finalColorsOptBuffer,
        const sgl::vk::BufferPtr& _adjointColorsBuffer) {
    if (finalColorsGTBuffer != _finalColorsGTBuffer) {
        finalColorsGTBuffer = _finalColorsGTBuffer;
        if (computeData) {
            computeData->setStaticBuffer(finalColorsGTBuffer, "FinalColorsGTBuffer");
        }
    }
    if (finalColorsOptBuffer != _finalColorsOptBuffer) {
        finalColorsOptBuffer = _finalColorsOptBuffer;
        if (computeData) {
            computeData->setStaticBuffer(finalColorsOptBuffer, "FinalColorsOptBuffer");
        }
    }
    if (adjointColorsBuffer != _adjointColorsBuffer) {
        adjointColorsBuffer = _adjointColorsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(adjointColorsBuffer, "AdjointColorsBuffer");
        }
    }
}

void LossPass::setSettings(LossType _lossType, uint32_t imageWidth, uint32_t imageHeight, uint32_t batchSize) {
    if (lossType != _lossType) {
        lossType = _lossType;
        setShaderDirty();
    }
    if (imageWidth != uniformData.imageWidth || imageHeight != uniformData.imageHeight
            || batchSize != uniformData.batchSize) {
        uniformData.imageWidth = imageWidth;
        uniformData.imageHeight = imageHeight;
        uniformData.batchSize = batchSize;
        isUniformBufferDirty = true;
    }
}

void LossPass::updateUniformBuffer() {
    if (isUniformBufferDirty) {
        isUniformBufferDirty = false;
        uniformBuffer->updateData(
                sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void LossPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    if (lossType == LossType::L1) {
        preprocessorDefines.insert(std::make_pair("L1_LOSS", ""));
    } else if (lossType == LossType::L2) {
        preprocessorDefines.insert(std::make_pair("L2_LOSS", ""));
    }
    if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_BUFFER_FLOAT_ATOMIC_ADD", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "Loss.Compute.Image" }, preprocessorDefines);
}

void LossPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticBuffer(finalColorsGTBuffer, "FinalColorsGTBuffer");
    computeData->setStaticBuffer(finalColorsOptBuffer, "FinalColorsOptBuffer");
    computeData->setStaticBuffer(adjointColorsBuffer, "AdjointColorsBuffer");
}

void LossPass::_render() {
    uint32_t workSize = uniformData.imageWidth * uniformData.imageHeight * uniformData.batchSize;
    renderer->dispatch(computeData, sgl::uiceil(workSize, computeBlockSize), 1, 1);
}
