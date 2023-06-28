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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "ResidualColorCalculator.hpp"

ResidualColorCalculator::ResidualColorCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
    residualColorComputePass = std::make_shared<ResidualColorComputePass>(renderer);
}

std::string ResidualColorCalculator::getOutputFieldName() {
    std::string outputFieldName = "Residual Color";
    if (calculatorConstructorUseCount > 1) {
        outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
    }
    return outputFieldName;
}

void ResidualColorCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::RESIDUAL_COLOR);
    }
    residualColorComputePass->setVolumeData(volumeData);

    scalarFieldNames = {};
    scalarFieldIndexArray = {};

    std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
    for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
        if (scalarFieldNamesNew.at(i) != getOutputFieldName()) {
            scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
            scalarFieldIndexArray.push_back(i);
        }
    }

    if (isNewData) {
        for (int i = 0; i < 2; i++) {
            scalarFieldIndices[i] = volumeData->getStandardScalarFieldIdx();
            scalarFieldIndicesGui[i] = volumeData->getStandardScalarFieldIdx();
            volumeData->acquireScalarField(this, scalarFieldIndices[i]);
            volumeData->acquireTf(this, scalarFieldIndices[i]);
        }
    }
}

void ResidualColorCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        for (int i = 0; i < 2; i++) {
            if (scalarFieldIndices[i] == fieldIdx) {
                scalarFieldIndices[i] = 0;
                scalarFieldIndicesGui[i] = 0;
                volumeData->acquireScalarField(this, scalarFieldIndices[i]);
                volumeData->acquireTf(this, scalarFieldIndices[i]);
                dirty = true;
            } else if (scalarFieldIndices[i] > fieldIdx) {
                scalarFieldIndices[i]--;
            }
            scalarFieldIndicesGui[i] = scalarFieldIndices[i];
        }
    }
}

void ResidualColorCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    VolumeData::DeviceCacheEntry entryScalarField0 = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndices[0]), timeStepIdx, ensembleIdx);
    VolumeData::DeviceCacheEntry entryScalarField1 = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndices[1]), timeStepIdx, ensembleIdx);
    residualColorComputePass->setInputOutputImages(
            scalarFieldIndices[0], scalarFieldIndices[1],
            entryScalarField0->getVulkanImageView(),
            entryScalarField1->getVulkanImageView(),
            deviceCacheEntry->getVulkanImageView());

    entryScalarField0->getVulkanImage()->transitionImageLayout(VK_IMAGE_LAYOUT_GENERAL, renderer->getVkCommandBuffer());
    entryScalarField1->getVulkanImage()->transitionImageLayout(VK_IMAGE_LAYOUT_GENERAL, renderer->getVkCommandBuffer());
    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
    residualColorComputePass->render();
}

void ResidualColorCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    for (int i = 0; i < 2; i++) {
        std::string comboName = "Scalar Field #" + std::to_string(i);
        if (propertyEditor.addCombo(
                comboName, &scalarFieldIndicesGui[i], scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            volumeData->releaseScalarField(this, scalarFieldIndices[i]);
            volumeData->releaseTf(this, scalarFieldIndices[i]);
            scalarFieldIndices[i] = int(scalarFieldIndexArray.at(scalarFieldIndicesGui[i]));
            volumeData->acquireScalarField(this, scalarFieldIndices[i]);
            volumeData->acquireTf(this, scalarFieldIndices[i]);
            dirty = true;
        }
    }
}

void ResidualColorCalculator::setSettings(const SettingsMap& settings) {
    Calculator::setSettings(settings);
    for (int i = 0; i < 2; i++) {
        std::string keyName = "scalar_field_idx_" + std::to_string(i);
        if (settings.getValueOpt(keyName.c_str(), scalarFieldIndicesGui[i])) {
            volumeData->releaseScalarField(this, scalarFieldIndices[i]);
            volumeData->releaseTf(this, scalarFieldIndices[i]);
            scalarFieldIndices[i] = int(scalarFieldIndexArray.at(scalarFieldIndicesGui[i]));
            volumeData->acquireScalarField(this, scalarFieldIndices[i]);
            volumeData->acquireTf(this, scalarFieldIndices[i]);
            dirty = true;
        }
    }
}

void ResidualColorCalculator::getSettings(SettingsMap& settings) {
    Calculator::getSettings(settings);
    for (int i = 0; i < 2; i++) {
        settings.addKeyValue("scalar_field_idx_" + std::to_string(i), scalarFieldIndicesGui[i]);
    }
}



ResidualColorComputePass::ResidualColorComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void ResidualColorComputePass::setVolumeData(VolumeData* _volumeData) {
    volumeData = _volumeData;
}

void ResidualColorComputePass::setInputOutputImages(
        int fieldIndex0, int fieldIndex1,
        const sgl::vk::ImageViewPtr& _inputImage0,
        const sgl::vk::ImageViewPtr& _inputImage1,
        const sgl::vk::ImageViewPtr& _outputImage) {
    uniformData.fieldIndex0 = uint32_t(fieldIndex0);
    uniformData.fieldIndex1 = uint32_t(fieldIndex1);
    if (inputImage0 != _inputImage0) {
        bool formatMatches = true;
        if (inputImage0) {
            formatMatches =
                    getImageFormatGlslString(inputImage0->getImage())
                    == getImageFormatGlslString(_inputImage0->getImage());
        }
        inputImage0 = _inputImage0;
        if (formatMatches && computeData) {
            computeData->setStaticImageView(inputImage0, "inputImage0");
        }
        if (!formatMatches) {
            setShaderDirty();
        }
    }
    if (inputImage1 != _inputImage1) {
        bool formatMatches = true;
        if (inputImage1) {
            formatMatches =
                    getImageFormatGlslString(inputImage1->getImage())
                    == getImageFormatGlslString(_inputImage1->getImage());
        }
        inputImage1 = _inputImage1;
        if (formatMatches && computeData) {
            computeData->setStaticImageView(inputImage1, "inputImage1");
        }
        if (!formatMatches) {
            setShaderDirty();
        }
    }
    if (outputImage != _outputImage) {
        outputImage = _outputImage;
        if (computeData) {
            computeData->setStaticImageView(outputImage, "outputImage");
        }
    }
}

void ResidualColorComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    volumeData->getPreprocessorDefines(preprocessorDefines);
    preprocessorDefines.insert(std::make_pair(
            "INPUT_IMAGE_0_FORMAT", getImageFormatGlslString(inputImage0->getImage())));
    preprocessorDefines.insert(std::make_pair(
            "INPUT_IMAGE_1_FORMAT", getImageFormatGlslString(inputImage1->getImage())));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    std::string shaderName = "ResidualColorCalculator.Compute";
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void ResidualColorComputePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    volumeData->setRenderDataBindings(computeData);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticImageView(inputImage0, "inputImage0");
    computeData->setStaticImageView(inputImage1, "inputImage1");
    computeData->setStaticImageView(outputImage, "outputImage");
}

void ResidualColorComputePass::_render() {
    uniformData.xs = outputImage->getImage()->getImageSettings().width;
    uniformData.ys = outputImage->getImage()->getImageSettings().height;
    uniformData.zs = outputImage->getImage()->getImageSettings().depth;
    uniformData.es = 0;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    renderer->dispatch(
            computeData,
            sgl::uiceil(uniformData.xs, computeBlockSizeX),
            sgl::uiceil(uniformData.ys, computeBlockSizeY),
            sgl::uiceil(uniformData.zs, computeBlockSizeZ));
}
