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

#include <chrono>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/imgui_custom.h>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "DKL.hpp"
#include "DKLCalculator.hpp"

DKLCalculator::DKLCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
    dklComputePass = std::make_shared<DKLComputePass>(renderer);
}

std::string DKLCalculator::getOutputFieldName() {
    std::string outputFieldName = "KL-Divergence";
    if (calculatorConstructorUseCount > 1) {
        outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
    }
    return outputFieldName;
}

void DKLCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::DKL_CALCULATOR);
    }
    dklComputePass->setVolumeData(volumeData, isNewData);
    if (isNewData || cachedMemberCount != volumeData->getEnsembleMemberCount()) {
        onMemberCountChanged();
    }

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

        if (!getSupportsBufferMode() || volumeData->getGridSizeZ() < 4) {
            dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        } else {
            dataMode = CorrelationDataMode::BUFFER_ARRAY;
        }
    }
}

void DKLCalculator::onMemberCountChanged() {
    int cs = volumeData->getEnsembleMemberCount();
    k = std::max(sgl::iceil(3 * cs, 100), 1);
    kMax = std::max(sgl::iceil(7 * cs, 100), 20);
    //dklComputePass->setMemberCount(cs); //< Cannot currently happen.
    dklComputePass->setNumNeighbors(k);
    cachedMemberCount = cs;
}

void DKLCalculator::clearFieldDeviceData() {
    dklComputePass->setFieldImageViews({});
    dklComputePass->setFieldBuffers({});
}

bool DKLCalculator::getSupportsBufferMode() {
    bool supportsBufferMode = true;
    if (!volumeData->getScalarFieldSupportsBufferMode(scalarFieldIndex)) {
        supportsBufferMode = false;
    }
    if (!supportsBufferMode && dataMode == CorrelationDataMode::BUFFER_ARRAY) {
        dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        dirty = true;
    }
    return supportsBufferMode;
}

void DKLCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
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

void DKLCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int cs = volumeData->getEnsembleMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    fieldEntries.reserve(cs);
    fields.reserve(cs);
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndex), timeStepIdx, fieldIdx);
        fieldEntries.push_back(fieldEntry);
        fields.push_back(fieldEntry->data<float>());
    }

#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    size_t numGridPoints = size_t(xs) * size_t(ys) * size_t(zs);
    if (estimatorType == DKLEstimatorType::BINNED) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[cs];
            auto* histogram = new double[numBins];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, fields, buffer) default(none)
        {
#endif
            auto* gridPointValues = new float[cs];
            auto* histogram = new double[numBins];
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    gridPointValues[c] = fields.at(c)[gridPointIdx];
                    if (std::isnan(gridPointValues[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                buffer[gridPointIdx] = computeDKLBinned<double>(gridPointValues, numBins, cs, histogram);
            }
            delete[] gridPointValues;
            delete[] histogram;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (estimatorType == DKLEstimatorType::ENTROPY_KNN) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[cs];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, k, fields, buffer) default(none)
        {
#endif
            auto* gridPointValues = new float[cs];
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    gridPointValues[c] = fields.at(c)[gridPointIdx];
                    if (std::isnan(gridPointValues[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                buffer[gridPointIdx] = computeDKLKNNEstimate<double>(gridPointValues, k, cs);
            }
            delete[] gridPointValues;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    }

#ifdef TEST_INFERENCE_SPEED
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}

void DKLCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    // We write to the descriptor set, so wait until the device is idle.
    renderer->getDevice()->waitIdle();

    int es = volumeData->getEnsembleMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

    std::vector<VolumeData::DeviceCacheEntry> fieldEntries;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    std::vector<sgl::vk::BufferPtr> fieldBuffers;
    dklComputePass->setDataMode(dataMode);
    dklComputePass->setUseBufferTiling(useBufferTiling);
    bool useImageArray = dataMode == CorrelationDataMode::IMAGE_3D_ARRAY;
    fieldEntries.reserve(es);
    if (useImageArray) {
        fieldBuffers.reserve(es);
    } else {
        fieldImageViews.reserve(es);
    }
    for (int fieldIdx = 0; fieldIdx < es; fieldIdx++) {
        VolumeData::DeviceCacheEntry fieldEntry = volumeData->getFieldEntryDevice(
                FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndexGui), timeStepIdx, fieldIdx,
                useImageArray, (!useImageArray && useBufferTiling) ? glm::uvec3(8, 8, 4) : glm::uvec3(1, 1, 1));
        fieldEntries.push_back(fieldEntry);
        if (useImageArray) {
            fieldImageViews.push_back(fieldEntry->getVulkanImageView());
            if (fieldEntry->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                fieldEntry->getVulkanImage()->transitionImageLayout(
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
            }
        } else {
            fieldBuffers.push_back(fieldEntry->getVulkanBuffer());
        }
    }
    if (useImageArray) {
        dklComputePass->setFieldImageViews(fieldImageViews);
    } else {
        dklComputePass->setFieldBuffers(fieldBuffers);
    }

#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);

    dklComputePass->setOutputImage(deviceCacheEntry->getVulkanImageView());

    dklComputePass->render();

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}

void DKLCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    std::string comboName = "Scalar Field";
    if (propertyEditor.addCombo(
            comboName, &scalarFieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }

    if (propertyEditor.addCombo(
            "Estimator Type", (int*)&estimatorType,
            DKL_ESTIMATOR_TYPE_NAMES, IM_ARRAYSIZE(DKL_ESTIMATOR_TYPE_NAMES))) {
        hasNameChanged = true;
        dirty = true;
        dklComputePass->setEstimatorType(estimatorType);
    }

    if (propertyEditor.addCheckbox("Use GPU", &useGpu)) {
        hasFilterDeviceChanged = true;
        dirty = true;
    }

    if (estimatorType == DKLEstimatorType::BINNED && propertyEditor.addSliderIntEdit(
            "#Bins", &numBins, 10, 100) == ImGui::EditMode::INPUT_FINISHED) {
        dklComputePass->setNumBins(numBins);
        dirty = true;
    }
    if (estimatorType == DKLEstimatorType::ENTROPY_KNN && propertyEditor.addSliderIntEdit(
            "#Neighbors", &k, 1, kMax) == ImGui::EditMode::INPUT_FINISHED) {
        dklComputePass->setNumNeighbors(k);
        dirty = true;
    }

    if (propertyEditor.beginNode("Advanced Settings")) {
        if (!scalarFieldNames.empty() && getSupportsBufferMode() && propertyEditor.addCombo(
                "Data Mode", (int*)&dataMode, DATA_MODE_NAMES, IM_ARRAYSIZE(DATA_MODE_NAMES))) {
            clearFieldDeviceData();
            dirty = true;
        }
        if (dataMode != CorrelationDataMode::IMAGE_3D_ARRAY && propertyEditor.addCheckbox(
                "Use Buffer Tiling", &useBufferTiling)) {
            clearFieldDeviceData();
            dirty = true;
        }
        propertyEditor.endNode();
    }
}

void DKLCalculator::setSettings(const SettingsMap& settings) {
    Calculator::setSettings(settings);
    if (settings.getValueOpt("scalar_field_idx", scalarFieldIndexGui)) {
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }

    std::string estimatorTypeName;
    if (settings.getValueOpt("estimator_type", estimatorTypeName)) {
        for (int i = 0; i < IM_ARRAYSIZE(DKL_ESTIMATOR_TYPE_NAMES); i++) {
            if (estimatorTypeName == DKL_ESTIMATOR_TYPE_NAMES[i]) {
                estimatorType = DKLEstimatorType(i);
                break;
            }
        }
        dklComputePass->setEstimatorType(estimatorType);
        dirty = true;
    }

    bool useGpuOld = useGpu;
    if (settings.getValueOpt("use_gpu", useGpu)) {
        if (useGpuOld != useGpu) {
            hasFilterDeviceChanged = true;
            dirty = true;
        }
    }

    if (settings.getValueOpt("mi_bins", numBins)) {
        dklComputePass->setNumBins(numBins);
        dirty = true;
    }
    if (settings.getValueOpt("knn_neighbors", k)) {
        dklComputePass->setNumNeighbors(k);
        dirty = true;
    }

    // Advanced settings.
    std::string dataModeString;
    if (settings.getValueOpt("data_mode", dataModeString)) {
        for (int i = 0; i < IM_ARRAYSIZE(DATA_MODE_NAMES); i++) {
            if (dataModeString == DATA_MODE_NAMES[i]) {
                dataMode = CorrelationDataMode(i);
                break;
            }
        }
        clearFieldDeviceData();
        dirty = true;
    }
    if (settings.getValueOpt("use_buffer_tiling", useBufferTiling)) {
        clearFieldDeviceData();
        dirty = true;
    }
}

void DKLCalculator::getSettings(SettingsMap& settings) {
    Calculator::getSettings(settings);
    settings.addKeyValue("scalar_field_idx", scalarFieldIndexGui);
    settings.addKeyValue("estimator_type", DKL_ESTIMATOR_TYPE_NAMES[int(estimatorType)]);
    settings.addKeyValue("use_gpu", useGpu);
    settings.addKeyValue("mi_bins", numBins);
    settings.addKeyValue("knn_neighbors", k);

    // Advanced settings.
    settings.addKeyValue("data_mode", DATA_MODE_NAMES[int(dataMode)]);
    settings.addKeyValue("use_buffer_tiling", useBufferTiling);
}



DKLComputePass::DKLComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void DKLComputePass::setVolumeData(VolumeData *_volumeData, bool isNewData) {
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

void DKLComputePass::setDataMode(CorrelationDataMode _dataMode) {
    if (dataMode != _dataMode) {
        dataMode = _dataMode;
        setShaderDirty();
    }
}

void DKLComputePass::setUseBufferTiling(bool _useBufferTiling) {
    if (useBufferTiling != _useBufferTiling) {
        useBufferTiling = _useBufferTiling;
        setShaderDirty();
    }
}

void DKLComputePass::setFieldBuffers(const std::vector<sgl::vk::BufferPtr>& _fieldBuffers) {
    if (fieldBuffers == _fieldBuffers) {
        return;
    }
    fieldBuffers = _fieldBuffers;
    if (computeData) {
        computeData->setStaticBufferArrayOptional(fieldBuffers, "ScalarFieldBuffers");
    }
}

void DKLComputePass::setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews) {
    if (fieldImageViews == _fieldImageViews) {
        return;
    }
    fieldImageViews = _fieldImageViews;
    if (computeData) {
        computeData->setStaticImageViewArrayOptional(fieldImageViews, "scalarFields");
    }
}

void DKLComputePass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    if (outputImage != _outputImage) {
        outputImage = _outputImage;
        if (computeData) {
            computeData->setStaticImageView(outputImage, "outputImage");
        }
    }
}

void DKLComputePass::setEstimatorType(DKLEstimatorType _estimatorType) {
    if (estimatorType != _estimatorType) {
        estimatorType = _estimatorType;
        setShaderDirty();
    }
}

void DKLComputePass::setNumBins(int _numBins) {
    if (estimatorType == DKLEstimatorType::BINNED && numBins != _numBins) {
        setShaderDirty();
    }
    numBins = _numBins;
}

void DKLComputePass::setNumNeighbors(int _k) {
    if (estimatorType == DKLEstimatorType::ENTROPY_KNN && k != _k) {
        setShaderDirty();
    }
    k = _k;
}

void DKLComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    if (uniformData.zs < 4) {
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSize2dX)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSize2dY)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", "1"));
    } else {
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    }
    preprocessorDefines.insert(std::make_pair(
            "MEMBER_COUNT", std::to_string(volumeData->getEnsembleMemberCount())));
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        preprocessorDefines.insert(std::make_pair("USE_SCALAR_FIELD_IMAGES", ""));
    } else if (useBufferTiling) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_TILING", ""));
    }
    if (estimatorType == DKLEstimatorType::BINNED) {
        preprocessorDefines.insert(std::make_pair("BINNED_ESTIMATOR", ""));
    } else if (estimatorType == DKLEstimatorType::ENTROPY_KNN) {
        preprocessorDefines.insert(std::make_pair("ENTROPY_KNN_ESTIMATOR", ""));
    }
    if (estimatorType == DKLEstimatorType::BINNED) {
        preprocessorDefines.insert(std::make_pair("numBins", std::to_string(numBins)));
    } else if (estimatorType == DKLEstimatorType::ENTROPY_KNN) {
        preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "DKL.Compute" }, preprocessorDefines);
}

void DKLComputePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
        computeData->setStaticImageViewArray(fieldImageViews, "scalarFields");
    } else {
        computeData->setStaticBufferArray(fieldBuffers, "ScalarFieldBuffers");
    }
    computeData->setStaticImageView(outputImage, "outputImage");
}

void DKLComputePass::_render() {
    // TODO: Evaluate if batched rendering is necessary.
    //uint32_t batchCount = 1;
    //bool needsBatchedRendering = false;

    int blockSizeX = uniformData.zs < 4 ? computeBlockSize2dX : computeBlockSizeX;
    int blockSizeY = uniformData.zs < 4 ? computeBlockSize2dY : computeBlockSizeY;
    int blockSizeZ = uniformData.zs < 4 ? 1 : computeBlockSizeZ;

    auto quietNan = std::numeric_limits<float>::quiet_NaN();
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(float), quietNan);
    renderer->dispatch(
            computeData,
            sgl::uiceil(int(uniformData.xs), blockSizeX),
            sgl::uiceil(int(uniformData.ys), blockSizeY),
            sgl::uiceil(int(uniformData.zs), blockSizeZ));
}
