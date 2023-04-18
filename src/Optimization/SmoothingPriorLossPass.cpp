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

#include "SmoothingPriorLossPass.hpp"

SmoothingPriorLossPass::SmoothingPriorLossPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void SmoothingPriorLossPass::setBuffers(
        const sgl::vk::BufferPtr& _transferFunctionBuffer,
        const sgl::vk::BufferPtr& _transferFunctionGradientBuffer) {
    if (transferFunctionBuffer != _transferFunctionBuffer) {
        transferFunctionBuffer = _transferFunctionBuffer;
        if (computeData) {
            computeData->setStaticBuffer(transferFunctionBuffer, "TransferFunctionBuffer");
        }
    }
    if (transferFunctionGradientBuffer != _transferFunctionGradientBuffer) {
        transferFunctionGradientBuffer = _transferFunctionGradientBuffer;
        if (computeData) {
            computeData->setStaticBuffer(transferFunctionGradientBuffer, "TransferFunctionGradientBuffer");
        }
    }
}

void SmoothingPriorLossPass::setSettings(float lambda, uint32_t R) {
    if (uniformData.lambda != lambda || uniformData.R != R) {
        uniformData.lambda = lambda;
        uniformData.R = R;
        uniformBuffer->updateData(
                sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
}

void SmoothingPriorLossPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "SmoothingPrior.Compute" }, preprocessorDefines);
}

void SmoothingPriorLossPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "SmoothingPriorSettingsBuffer");
    computeData->setStaticBuffer(transferFunctionBuffer, "TransferFunctionBuffer");
    computeData->setStaticBuffer(transferFunctionGradientBuffer, "TransferFunctionGradientBuffer");
}

void SmoothingPriorLossPass::_render() {
    auto tfSize = uint32_t(transferFunctionBuffer->getSizeInBytes() / 4);
    renderer->dispatch(computeData, sgl::uiceil(tfSize, computeBlockSize), 1, 1);
}
