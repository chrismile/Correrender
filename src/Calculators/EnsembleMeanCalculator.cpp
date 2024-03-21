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
#include "EnsembleMeanCalculator.hpp"

EnsembleMeanCalculator::EnsembleMeanCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
    ensembleVarianceComputePass = std::make_shared<EnsembleMeanComputePass>(renderer);
}

std::string EnsembleMeanCalculator::getOutputFieldName() {
    std::string outputFieldName = "Ensemble Mean";
    if (calculatorConstructorUseCount > 1) {
        outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
    }
    return outputFieldName;
}

void EnsembleMeanCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::ENSEMBLE_MEAN);
    }
    ensembleVarianceComputePass->setVolumeData(volumeData, isNewData);

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
        scalarFieldIndex = volumeData->getStandardScalarFieldIdx();
        scalarFieldIndexGui = volumeData->getStandardScalarFieldIdx();
        volumeData->acquireScalarField(this, scalarFieldIndex);
    }
}

void EnsembleMeanCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (scalarFieldIndex == fieldIdx) {
            scalarFieldIndex = 0;
            scalarFieldIndexGui = 0;
            volumeData->acquireScalarField(this, scalarFieldIndex);
            dirty = true;
        } else if (scalarFieldIndex > fieldIdx) {
            scalarFieldIndex--;
        }
        scalarFieldIndexGui = scalarFieldIndex;
    }
}

void EnsembleMeanCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();

    std::vector<VolumeData::HostCacheEntry> scalarFieldEntries;
    std::vector<const float*> scalarFields;
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry entryScalarField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndex), timeStepIdx, ensembleIdx);
        const float* scalarField = entryScalarField->data<float>();
        scalarFieldEntries.push_back(entryScalarField);
        scalarFields.push_back(scalarField);
    }

    int numPoints = xs * ys * zs;
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numPoints), [&](auto const& r) {
        for (auto pointIdx = r.begin(); pointIdx != r.end(); pointIdx++) {
#else
    #pragma omp parallel for shared(numPoints, es, scalarFields, buffer) default(none)
    for (int pointIdx = 0; pointIdx < numPoints; pointIdx++) {
#endif
        int numValid = 0;
        float ensembleMean = 0.0f;
        for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
            const float* field = scalarFields.at(ensembleIdx);
            float val = field[pointIdx];
            if (!std::isnan(val)) {
                ensembleMean += val;
                numValid++;
            }
        }
        if (numValid >= 1) {
            ensembleMean = ensembleMean / float(numValid);
        } else {
            ensembleMean = std::numeric_limits<float>::quiet_NaN();
        }
        buffer[pointIdx] = ensembleMean;
    }
#ifdef USE_TBB
    });
#endif
}

void EnsembleMeanCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    int es = volumeData->getEnsembleMemberCount();

    std::vector<VolumeData::DeviceCacheEntry> ensembleEntryFields;
    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    ensembleEntryFields.reserve(es);
    ensembleImageViews.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
                FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleImageViews.push_back(ensembleEntryField->getVulkanImageView());
        if (ensembleEntryField->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            ensembleEntryField->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        }
    }

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);

    ensembleVarianceComputePass->setOutputImage(deviceCacheEntry->getVulkanImageView());
    ensembleVarianceComputePass->setEnsembleImageViews(ensembleImageViews);

    ensembleVarianceComputePass->render();
}

void EnsembleMeanCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    std::string comboName = "Scalar Field";
    if (propertyEditor.addCombo(
            comboName, &scalarFieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }
}

void EnsembleMeanCalculator::setSettings(const SettingsMap& settings) {
    Calculator::setSettings(settings);
    if (settings.getValueOpt("scalar_field_idx", scalarFieldIndexGui)) {
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }
}

void EnsembleMeanCalculator::getSettings(SettingsMap& settings) {
    Calculator::getSettings(settings);
    settings.addKeyValue("scalar_field_idx", scalarFieldIndexGui);
}



EnsembleMeanComputePass::EnsembleMeanComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void EnsembleMeanComputePass::setVolumeData(VolumeData *_volumeData, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.es = uint32_t(volumeData->getEnsembleMemberCount());
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedEnsembleMemberCount != volumeData->getEnsembleMemberCount()) {
        cachedEnsembleMemberCount = volumeData->getEnsembleMemberCount();
        setShaderDirty();
    }
}


void EnsembleMeanComputePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews != _ensembleImageViews) {
        ensembleImageViews = _ensembleImageViews;
        if (computeData) {
            computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
        }
    }
}

void EnsembleMeanComputePass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    if (outputImage != _outputImage) {
        outputImage = _outputImage;
        if (computeData) {
            computeData->setStaticImageView(outputImage, "outputImage");
        }
    }
}

void EnsembleMeanComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(volumeData->getEnsembleMemberCount())));
    std::string shaderName = "EnsembleMeanCalculator.Compute";
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void EnsembleMeanComputePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticImageView(outputImage, "outputImage");
}

void EnsembleMeanComputePass::_render() {
    renderer->dispatch(
            computeData,
            sgl::uiceil(uniformData.xs, computeBlockSizeX),
            sgl::uiceil(uniformData.ys, computeBlockSizeY),
            sgl::uiceil(uniformData.zs, computeBlockSizeZ));
}
