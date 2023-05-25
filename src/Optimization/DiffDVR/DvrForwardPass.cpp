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

#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "DvrForwardPass.hpp"

DvrForwardPass::DvrForwardPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
}

void DvrForwardPass::setScalarFieldTexture(const sgl::vk::TexturePtr& _scalarFieldTexture) {
    if (scalarFieldTexture != _scalarFieldTexture) {
        scalarFieldTexture = _scalarFieldTexture;
        if (computeData) {
            computeData->setStaticTexture(scalarFieldTexture, "scalarField");
        }
    }
}

void DvrForwardPass::setBuffers(
        const sgl::vk::BufferPtr& _dvrSettingsBuffer,
        const sgl::vk::BufferPtr& _batchSettingsBuffer,
        const sgl::vk::BufferPtr& _tfBuffer,
        const sgl::vk::BufferPtr& _finalColorsBuffer,
        const sgl::vk::BufferPtr& _terminationIndexBuffer) {
    if (dvrSettingsBuffer != _dvrSettingsBuffer) {
        dvrSettingsBuffer = _dvrSettingsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(dvrSettingsBuffer, "DvrSettingsBuffer");
        }
    }
    if (batchSettingsBuffer != _batchSettingsBuffer) {
        batchSettingsBuffer = _batchSettingsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(batchSettingsBuffer, "BatchSettingsBuffer");
        }
    }
    if (tfBuffer != _tfBuffer) {
        tfBuffer = _tfBuffer;
        if (computeData) {
            computeData->setStaticBuffer(tfBuffer, "TfBuffer");
        }
    }
    if (finalColorsBuffer != _finalColorsBuffer) {
        finalColorsBuffer = _finalColorsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(finalColorsBuffer, "FinalColorsBuffer");
        }
    }
    if (terminationIndexBuffer != _terminationIndexBuffer) {
        terminationIndexBuffer = _terminationIndexBuffer;
        if (computeData) {
            computeData->setStaticBuffer(terminationIndexBuffer, "TerminationIndexBuffer");
        }
    }
}

void DvrForwardPass::setSettings(
        bool _isForwardPassOpt, float _minFieldValue, float _maxFieldValue, uint32_t _tfSize,
        uint32_t _imageWidth, uint32_t _imageHeight, uint32_t _batchSize) {
    if (isForwardPassOpt != _isForwardPassOpt) {
        isForwardPassOpt = _isForwardPassOpt;
        setShaderDirty();
    }
    if (minFieldValue != _minFieldValue || maxFieldValue != _maxFieldValue) {
        minFieldValue = _minFieldValue;
        maxFieldValue = _maxFieldValue;
    }
    if (tfSize != _tfSize) {
        tfSize = _tfSize;
    }
    if (imageWidth != _imageWidth || imageHeight != _imageHeight || batchSize != _batchSize) {
        imageWidth = _imageWidth;
        imageHeight = _imageHeight;
        batchSize = _batchSize;
    }
}

void DvrForwardPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    if (isForwardPassOpt) {
        preprocessorDefines.insert(std::make_pair("FORWARD_PASS_OPT", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "DvrForward.Compute" }, preprocessorDefines);
}

void DvrForwardPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(dvrSettingsBuffer, "DvrSettingsBuffer");
    computeData->setStaticBuffer(batchSettingsBuffer, "BatchSettingsBuffer");
    computeData->setStaticTexture(scalarFieldTexture, "scalarField");
    computeData->setStaticBuffer(tfBuffer, "TfBuffer");
    computeData->setStaticBuffer(finalColorsBuffer, "FinalColorsBuffer");
    if (isForwardPassOpt) {
        computeData->setStaticBuffer(terminationIndexBuffer, "TerminationIndexBuffer");
    }
}

void DvrForwardPass::_render() {
    glm::vec3 pushConstants;
    pushConstants[0] = minFieldValue;
    pushConstants[1] = maxFieldValue;
    pushConstants[2] = float(tfSize - 1u);
    renderer->pushConstants(computeData->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstants);
    uint32_t workSize = imageWidth * imageHeight * batchSize;
    renderer->dispatch(computeData, sgl::uiceil(workSize, computeBlockSize), 1, 1);
}
