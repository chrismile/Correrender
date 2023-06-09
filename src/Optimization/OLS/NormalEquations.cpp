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
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "NormalEquations.hpp"

NormalEquationsComputePass::NormalEquationsComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    if (!renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
        sgl::Logfile::get()->writeWarning(
                "Warning in NormalEquationsComputePass::NormalEquationsComputePass: "
                "32-bit float atomics are not supported. Falling back to spinning updates.");
    }
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void NormalEquationsComputePass::setUseDoublePrecision(bool _useDoublePrecision) {
    if (useDoublePrecision != _useDoublePrecision) {
        useDoublePrecision = _useDoublePrecision;
        setShaderDirty();
    }
}

void NormalEquationsComputePass::setInputImages(
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

void NormalEquationsComputePass::setBuffers(
        uint32_t tfNumEntries, float minGT, float maxGT, float minOpt, float maxOpt,
        const sgl::vk::BufferPtr& _lhsBuffer,
        const sgl::vk::BufferPtr& _rhsBuffer,
        const sgl::vk::BufferPtr& _tfGTBuffer) {
    uniformData.tfNumEntries = tfNumEntries;
    uniformData.minGT = minGT;
    uniformData.maxGT = maxGT;
    uniformData.minOpt = minOpt;
    uniformData.maxOpt = maxOpt;
    if (lhsBuffer != _lhsBuffer) {
        lhsBuffer = _lhsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(lhsBuffer, "LhsBuffer");
        }
    }
    if (rhsBuffer != _rhsBuffer) {
        rhsBuffer = _rhsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(rhsBuffer, "RhsBuffer");
        }
    }
    if (transferFunctionGTBuffer != _tfGTBuffer) {
        transferFunctionGTBuffer = _tfGTBuffer;
        if (computeData) {
            computeData->setStaticBuffer(transferFunctionGTBuffer, "TransferFunctionGTBuffer");
        }
    }
}

void NormalEquationsComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair(
            "INPUT_IMAGE_0_FORMAT", getImageFormatGlslString(inputImageGT->getImage())));
    preprocessorDefines.insert(std::make_pair(
            "INPUT_IMAGE_1_FORMAT", getImageFormatGlslString(inputImageOpt->getImage())));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    std::string glslExtensionsString;
    if (useDoublePrecision) {
        preprocessorDefines.insert(std::make_pair("USE_DOUBLE_PRECISION", ""));
        if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat64AtomicAdd) {
            glslExtensionsString = "GL_EXT_shader_explicit_arithmetic_types_float64";
        } else {
            glslExtensionsString = "GL_EXT_shader_explicit_arithmetic_types_int64";
        }
    }
    if ((useDoublePrecision && renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat64AtomicAdd)
            || (!useDoublePrecision && renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd)) {
        if (!glslExtensionsString.empty()) {
            glslExtensionsString += ",";
        }
        glslExtensionsString += "GL_EXT_shader_atomic_float";
        preprocessorDefines.insert(std::make_pair("SUPPORT_BUFFER_FLOAT_ATOMIC_ADD", ""));
    }
    if (!glslExtensionsString.empty()) {
        preprocessorDefines.insert(std::make_pair("__extensions", glslExtensionsString));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "NormalEquations.Compute" }, preprocessorDefines);
}

void NormalEquationsComputePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticImageView(inputImageGT, "inputImageGT");
    computeData->setStaticImageView(inputImageOpt, "inputImageOpt");
    computeData->setStaticBuffer(lhsBuffer, "LhsBuffer");
    computeData->setStaticBuffer(rhsBuffer, "RhsBuffer");
    computeData->setStaticBuffer(transferFunctionGTBuffer, "TransferFunctionGTBuffer");
}

void NormalEquationsComputePass::_render() {
    uniformData.xs = inputImageGT->getImage()->getImageSettings().width;
    uniformData.ys = inputImageGT->getImage()->getImageSettings().height;
    uniformData.zs = inputImageGT->getImage()->getImageSettings().depth;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    auto Nj = float(uniformData.tfNumEntries / 4u - 1u);
    renderer->pushConstants(computeData->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, Nj);

    renderer->dispatch(
            computeData,
            sgl::uiceil(uniformData.xs, computeBlockSizeX),
            sgl::uiceil(uniformData.ys, computeBlockSizeY),
            sgl::uiceil(uniformData.zs, computeBlockSizeZ));
}



NormalEquationsCopySymmetricPass::NormalEquationsCopySymmetricPass(sgl::vk::Renderer* renderer)
        : ComputePass(renderer) {
}

void NormalEquationsCopySymmetricPass::setUseDoublePrecision(bool _useDoublePrecision) {
    if (useDoublePrecision != _useDoublePrecision) {
        useDoublePrecision = _useDoublePrecision;
        setShaderDirty();
    }
}

void NormalEquationsCopySymmetricPass::setBuffers(uint32_t _tfNumEntries, const sgl::vk::BufferPtr& _lhsBuffer) {
    tfNumEntries = _tfNumEntries;
    if (lhsBuffer != _lhsBuffer) {
        lhsBuffer = _lhsBuffer;
        if (computeData) {
            computeData->setStaticBuffer(lhsBuffer, "LhsBuffer");
        }
    }
}

void NormalEquationsCopySymmetricPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    if (useDoublePrecision) {
        preprocessorDefines.insert(std::make_pair("USE_DOUBLE_PRECISION", ""));
        if (renderer->getDevice()->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat64AtomicAdd) {
            preprocessorDefines.insert(std::make_pair("__extensions", "GL_EXT_shader_explicit_arithmetic_types_float64"));
        } else {
            preprocessorDefines.insert(std::make_pair("__extensions", "GL_EXT_shader_explicit_arithmetic_types_int64"));
        }
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "NormalEquations.ComputeSymmetrization" }, preprocessorDefines);
}

void NormalEquationsCopySymmetricPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(lhsBuffer, "LhsBuffer");
}

void NormalEquationsCopySymmetricPass::_render() {
    uint32_t workSize = (tfNumEntries * tfNumEntries - tfNumEntries) / 2;
    renderer->pushConstants(
            computeData->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, tfNumEntries);
    renderer->pushConstants(
            computeData->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(uint32_t), workSize);
    renderer->dispatch(computeData, sgl::uiceil(workSize, computeBlockSize), 1, 1);
}
