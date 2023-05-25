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
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "OptimizerPass.hpp"

OptimizerPass::OptimizerPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void OptimizerPass::setBuffers(
        const sgl::vk::BufferPtr& _tfOptBuffer,
        const sgl::vk::BufferPtr& _tfOptGradientBuffer) {
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
    if (!firstMomentEstimateBuffer
            || firstMomentEstimateBuffer->getSizeInBytes() != tfOptGradientBuffer->getSizeInBytes()) {
        firstMomentEstimateBuffer = std::make_shared<sgl::vk::Buffer>(
                device, tfOptGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        secondMomentEstimateBuffer = std::make_shared<sgl::vk::Buffer>(
                device, tfOptGradientBuffer->getSizeInBytes(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        setDataDirty();
    }
}

void OptimizerPass::setOptimizerType(OptimizerType _optimizerType) {
    if (optimizerType != _optimizerType) {
        optimizerType = _optimizerType;
        setShaderDirty();
    }
}

void OptimizerPass::setSettings(
        LossType _lossType, uint32_t _numTfEntries, float alpha, float beta1, float beta2, float epsilon) {
    if (lossType != _lossType) {
        lossType = _lossType;
        setShaderDirty();
    }
    if (numTfEntries != _numTfEntries) {
        numTfEntries = _numTfEntries;
        setShaderDirty();
    }
    if (uniformData.alpha != alpha || uniformData.beta1 != beta1 || uniformData.beta2 != beta2
            || uniformData.epsilon != epsilon) {
        uniformData.alpha = alpha;
        uniformData.beta1 = beta1;
        uniformData.beta2 = beta2;
        uniformData.epsilon = epsilon;
    }
}

void OptimizerPass::setEpochIndex(int epochIdx) {
    if (epochIdx == 0) {
        if (optimizerType == OptimizerType::ADAM) {
            firstMomentEstimateBuffer->fill(0, renderer->getVkCommandBuffer());
            secondMomentEstimateBuffer->fill(0, renderer->getVkCommandBuffer());
        }
        uniformBuffer->updateData(
                sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    t = float(epochIdx + 1);
}

void OptimizerPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    preprocessorDefines.insert(std::make_pair("NUM_TF_ENTRIES", std::to_string(numTfEntries)));
    if (lossType == LossType::L1) {
        preprocessorDefines.insert(std::make_pair("L1_LOSS", ""));
    } else if (lossType == LossType::L2) {
        preprocessorDefines.insert(std::make_pair("L2_LOSS", ""));
    }
    std::string shaderName;
    if (optimizerType == OptimizerType::SGD) {
        shaderName = "Optimizer.Compute.SGD";
    } else if (optimizerType == OptimizerType::ADAM) {
        shaderName = "Optimizer.Compute.Adam";
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void OptimizerPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "OptimizerSettingsBuffer");
    computeData->setStaticBuffer(tfOptBuffer, "TfOptBuffer");
    computeData->setStaticBuffer(tfOptGradientBuffer, "TfOptGradientBuffer");
    if (optimizerType == OptimizerType::ADAM) {
        computeData->setStaticBuffer(firstMomentEstimateBuffer, "FirstMomentEstimateBuffer");
        computeData->setStaticBuffer(secondMomentEstimateBuffer, "SecondMomentEstimateBuffer");
    }
}

void OptimizerPass::_render() {
    if (optimizerType == OptimizerType::ADAM) {
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, t);
    }
    auto tfSize = uint32_t(tfOptBuffer->getSizeInBytes() / 4);
    renderer->dispatch(computeData, sgl::uiceil(tfSize, computeBlockSize), 1, 1);
}
