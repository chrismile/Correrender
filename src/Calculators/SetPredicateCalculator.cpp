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
#include "SetPredicateCalculator.hpp"

SetPredicateCalculator::SetPredicateCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
    setPredicateComputePass = std::make_shared<SetPredicateComputePass>(renderer);
    setPredicateComputePass->setComparisonOperatorType(comparisonOperatorType);
}

std::string SetPredicateCalculator::getOutputFieldName() {
    std::string outputFieldName = "Set Predicate";
    if (calculatorConstructorUseCount > 1) {
        outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
    }
    return outputFieldName;
}

void SetPredicateCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::SET_PREDICATE);
    }

    int es = _volumeData->getEnsembleMemberCount();
    int ts = _volumeData->getTimeStepCount();
    if (isEnsembleMode && es <= 1 && ts > 1) {
        isEnsembleMode = false;
    } else if (!isEnsembleMode && ts <= 1 && es > 1) {
        isEnsembleMode = true;
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
    }

    setPredicateComputePass->setVolumeData(volumeData, getCorrelationMemberCount(), isNewData);
    if (isNewData || cachedMemberCount != getCorrelationMemberCount()) {
        onCorrelationMemberCountChanged();
    }
}

void SetPredicateCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
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

int SetPredicateCalculator::getCorrelationMemberCount() const {
    return isEnsembleMode ? volumeData->getEnsembleMemberCount() : volumeData->getTimeStepCount();
}

VolumeData::HostCacheEntry SetPredicateCalculator::getFieldEntryCpu(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx) {
    VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx);
    return ensembleEntryField;
}

VolumeData::DeviceCacheEntry SetPredicateCalculator::getFieldEntryDevice(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx, bool wantsImageData) {
    VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx,
            wantsImageData, (!wantsImageData && useBufferTiling) ? glm::uvec3(8, 8, 4) : glm::uvec3(1, 1, 1));
    return ensembleEntryField;
}

std::pair<float, float> SetPredicateCalculator::getMinMaxScalarFieldValue(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx) {
    return volumeData->getMinMaxScalarFieldValue(
            fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx);
}

void SetPredicateCalculator::onCorrelationMemberCountChanged() {
    int cs = getCorrelationMemberCount();
    setPredicateComputePass->setCorrelationMemberCount(cs);
    cachedMemberCount = cs;

    countLower = getCorrelationMemberCount() / 2;
    countUpper = countLower;
    setPredicateComputePass->setCountLower(countLower);
    setPredicateComputePass->setCountUpper(countUpper);
}

void SetPredicateCalculator::clearFieldDeviceData() {
    setPredicateComputePass->setFieldImageViews({});
    setPredicateComputePass->setFieldBuffers({});
}

void SetPredicateCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int cs = getCorrelationMemberCount();

    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    fieldEntries.reserve(cs);
    fields.reserve(cs);
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(
                scalarFieldNames.at(scalarFieldIndex), fieldIdx, timeStepIdx, ensembleIdx);
        fieldEntries.push_back(fieldEntry);
        fields.push_back(fieldEntry->data<float>());
    }

    std::function<float(float val0, float val1)> comparator;
    if (comparisonOperatorType == ComparisonOperatorType::GREATER) {
        comparator = [](float val0, float val1) { return val0 > val1; };
    } else if (comparisonOperatorType == ComparisonOperatorType::GREATER_EQUAL) {
        comparator = [](float val0, float val1) { return val0 >= val1; };
    } else if (comparisonOperatorType == ComparisonOperatorType::LESS) {
        comparator = [](float val0, float val1) { return val0 < val1; };
    } else if (comparisonOperatorType == ComparisonOperatorType::LESS_EQUAL) {
        comparator = [](float val0, float val1) { return val0 <= val1; };
    } else if (comparisonOperatorType == ComparisonOperatorType::EQUAL) {
        comparator = [](float val0, float val1) { return val0 == val1; };
    } else if (comparisonOperatorType == ComparisonOperatorType::NOT_EQUAL) {
        comparator = [](float val0, float val1) { return val0 != val1; };
    }

    int numPoints = xs * ys * zs;
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numPoints), [&](auto const& r) {
        for (auto pointIdx = r.begin(); pointIdx != r.end(); pointIdx++) {
#else
    #pragma omp parallel for shared(numPoints, cs, fields, comparator, buffer) default(none)
    for (int pointIdx = 0; pointIdx < numPoints; pointIdx++) {
#endif
        int count = 0;
        for (int c = 0; c < cs; c++) {
            if (comparator(fields.at(c)[pointIdx], comparisonValue)) {
                count++;
            }
        }
        if (countLower == countUpper) {
            buffer[pointIdx] = std::clamp(float(count) - float(countLower), 0.0f, 1.0f);
        } else {
            buffer[pointIdx] = std::clamp(
                    (float(count) - float(countLower)) / (float(countUpper) - float(countLower)), 0.0f, 1.0f);
        }
    }
#ifdef USE_TBB
    });
#endif
}

void SetPredicateCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    // We write to the descriptor set, so wait until the device is idle.
    renderer->getDevice()->waitIdle();

    int cs = getCorrelationMemberCount();

    std::vector<VolumeData::DeviceCacheEntry> fieldEntries;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    std::vector<sgl::vk::BufferPtr> fieldBuffers;
    setPredicateComputePass->setDataMode(dataMode);
    setPredicateComputePass->setUseBufferTiling(useBufferTiling);
    bool useImageArray = dataMode == CorrelationDataMode::IMAGE_3D_ARRAY;
    fieldEntries.reserve(cs);
    if (useImageArray) {
        fieldBuffers.reserve(cs);
    } else {
        fieldImageViews.reserve(cs);
    }
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::DeviceCacheEntry fieldEntry = getFieldEntryDevice(
                scalarFieldNames.at(scalarFieldIndex), fieldIdx, timeStepIdx, ensembleIdx, useImageArray);
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
        setPredicateComputePass->setFieldImageViews(fieldImageViews);
    } else {
        setPredicateComputePass->setFieldBuffers(fieldBuffers);
    }

    setPredicateComputePass->setOutputImage(deviceCacheEntry->getVulkanImageView());

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);

    setPredicateComputePass->buildIfNecessary();
    setPredicateComputePass->render();
}

void SetPredicateCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (volumeData->getEnsembleMemberCount() > 1 && volumeData->getTimeStepCount() > 1) {
        int modeIdx = isEnsembleMode ? 0 : 1;
        if (propertyEditor.addCombo("Correlation Mode", &modeIdx, CORRELATION_MODE_NAMES, 2)) {
            isEnsembleMode = modeIdx == 0;
            onCorrelationMemberCountChanged();
            clearFieldDeviceData();
            dirty = true;
        }
    }

    int cs = getCorrelationMemberCount();
    if (useFuzzyLogic) {
        if (propertyEditor.addSliderInt("Count Lower", &countLower, 0, cs)) {
            setPredicateComputePass->setCountLower(countLower);
            dirty = true;
        }
        if (propertyEditor.addSliderInt("Count Upper", &countUpper, 0, cs)) {
            setPredicateComputePass->setCountUpper(countUpper);
            dirty = true;
        }
    } else {
        if (propertyEditor.addSliderInt("Count", &countUpper, 0, cs)) {
            countLower = countUpper;
            setPredicateComputePass->setCountLower(countLower);
            setPredicateComputePass->setCountUpper(countUpper);
            dirty = true;
        }
    }

    if (propertyEditor.addCombo(
            "Scalar Field", &scalarFieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        clearFieldDeviceData();
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }

    if (propertyEditor.addCombo(
            "Operator", (int*)&comparisonOperatorType, COMPARISON_OPERATOR_NAMES, IM_ARRAYSIZE(COMPARISON_OPERATOR_NAMES))) {
        setPredicateComputePass->setComparisonOperatorType(comparisonOperatorType);
        dirty = true;
    }

    if (propertyEditor.addDragFloat("Comparison Value", &comparisonValue)) {
        setPredicateComputePass->setComparisonValue(comparisonValue);
        dirty = true;
    }

    if (propertyEditor.beginNode("Advanced Settings")) {
        renderGuiImplAdvanced(propertyEditor);
        propertyEditor.endNode();
    }
}

void SetPredicateCalculator::renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addCheckbox("Use Fuzzy Logic", &useFuzzyLogic)) {
        if (!useFuzzyLogic) {
            countUpper = countLower;
        }
    }

    if (propertyEditor.addCheckbox("Use GPU", &useGpu)) {
        hasFilterDeviceChanged = true;
        dirty = true;
    }

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
}

bool SetPredicateCalculator::getSupportsBufferMode() {
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

void SetPredicateCalculator::setSettings(const SettingsMap& settings) {
    std::string ensembleModeName;
    if (settings.getValueOpt("correlation_mode", ensembleModeName)) {
        if (ensembleModeName == CORRELATION_MODE_NAMES[0]) {
            isEnsembleMode = true;
        } else {
            isEnsembleMode = false;
        }
        onCorrelationMemberCountChanged();
        clearFieldDeviceData();
        dirty = true;
    }

    if (settings.getValueOpt("count_lower", countLower)) {
        setPredicateComputePass->setCountLower(countLower);
        dirty = true;
    }
    if (settings.getValueOpt("count_upper", countUpper)) {
        setPredicateComputePass->setCountUpper(countUpper);
        dirty = true;
    }

    Calculator::setSettings(settings);
    std::string keyName = "scalar_field_idx";
    if (settings.getValueOpt(keyName.c_str(), scalarFieldIndexGui)) {
        clearFieldDeviceData();
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }

    std::string comparisonOperatorTypeString;
    if (settings.getValueOpt("comparison_operator_type", comparisonOperatorTypeString)) {
        for (int i = 0; i < IM_ARRAYSIZE(COMPARISON_OPERATOR_NAMES); i++) {
            if (comparisonOperatorTypeString == COMPARISON_OPERATOR_NAMES[i]) {
                comparisonOperatorType = ComparisonOperatorType(i);
                break;
            }
        }
        setPredicateComputePass->setComparisonOperatorType(comparisonOperatorType);
        dirty = true;
    }

    if (settings.getValueOpt("comparison_value", comparisonValue)) {
        setPredicateComputePass->setComparisonValue(comparisonValue);
        dirty = true;
    }

    // Advanced settings.
    settings.getValueOpt("use_fuzzy_logic", useFuzzyLogic);
    if (settings.getValueOpt("use_gpu", useGpu)) {
        hasFilterDeviceChanged = true;
        dirty = true;
    }

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

void SetPredicateCalculator::getSettings(SettingsMap& settings) {
    Calculator::getSettings(settings);
    settings.addKeyValue("correlation_mode", CORRELATION_MODE_NAMES[isEnsembleMode ? 0 : 1]);
    settings.addKeyValue("count_lower", countLower);
    settings.addKeyValue("count_upper", countUpper);
    settings.addKeyValue("scalar_field_idx", scalarFieldIndexGui);
    settings.addKeyValue("comparison_operator_type", COMPARISON_OPERATOR_NAMES[int(comparisonOperatorType)]);
    settings.addKeyValue("comparison_value", comparisonValue);

    // Advanced settings.
    settings.addKeyValue("use_fuzzy_logic", useFuzzyLogic);
    settings.addKeyValue("use_gpu", useGpu);
    settings.addKeyValue("data_mode", DATA_MODE_NAMES[int(dataMode)]);
    settings.addKeyValue("use_buffer_tiling", useBufferTiling);
}



SetPredicateComputePass::SetPredicateComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void SetPredicateComputePass::setVolumeData(VolumeData *_volumeData, int correlationMemberCount, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.cs = uint32_t(correlationMemberCount);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
    }
}

void SetPredicateComputePass::setCorrelationMemberCount(int correlationMemberCount) {
    uniformData.cs = uint32_t(correlationMemberCount);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
    }
}

void SetPredicateComputePass::setDataMode(CorrelationDataMode _dataMode) {
    if (dataMode != _dataMode) {
        dataMode = _dataMode;
        setShaderDirty();
    }
}

void SetPredicateComputePass::setUseBufferTiling(bool _useBufferTiling) {
    if (useBufferTiling != _useBufferTiling) {
        useBufferTiling = _useBufferTiling;
        setShaderDirty();
    }
}

void SetPredicateComputePass::setFieldBuffers(const std::vector<sgl::vk::BufferPtr>& _fieldBuffers) {
    if (fieldBuffers == _fieldBuffers) {
        return;
    }
    fieldBuffers = _fieldBuffers;
    if (computeData) {
        computeData->setStaticBufferArrayOptional(fieldBuffers, "ScalarFieldBuffers");
    }
}

void SetPredicateComputePass::setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews) {
    if (fieldImageViews == _fieldImageViews) {
        return;
    }
    fieldImageViews = _fieldImageViews;
    if (computeData) {
        computeData->setStaticImageViewArrayOptional(fieldImageViews, "scalarFields");
    }
}

void SetPredicateComputePass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    if (outputImage != _outputImage) {
        outputImage = _outputImage;
        if (computeData) {
            computeData->setStaticImageView(outputImage, "outputImage");
        }
    }
}

void SetPredicateComputePass::setComparisonOperatorType(ComparisonOperatorType _comparisonOperatorType) {
    if (comparisonOperatorType != _comparisonOperatorType) {
        comparisonOperatorType = _comparisonOperatorType;
        setShaderDirty();
    }
}

void SetPredicateComputePass::setCountLower(int _countLower) {
    uniformData.countLower = _countLower;
}

void SetPredicateComputePass::setCountUpper(int _countUpper) {
    uniformData.countUpper = _countUpper;
}

void SetPredicateComputePass::setComparisonValue(float _comparisonValue) {
    uniformData.comparisonValue = _comparisonValue;
}

void SetPredicateComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair(
            "MEMBER_COUNT", std::to_string(cachedCorrelationMemberCount)));
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        preprocessorDefines.insert(std::make_pair("USE_SCALAR_FIELD_IMAGES", ""));
    } else if (useBufferTiling) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_TILING", ""));
    }
    if (comparisonOperatorType == ComparisonOperatorType::GREATER) {
        preprocessorDefines.insert(std::make_pair("comparator(x, y)", "((x) > (y))"));
    } else if (comparisonOperatorType == ComparisonOperatorType::GREATER_EQUAL) {
        preprocessorDefines.insert(std::make_pair("comparator(x, y)", "((x) >= (y))"));
    } else if (comparisonOperatorType == ComparisonOperatorType::LESS) {
        preprocessorDefines.insert(std::make_pair("comparator(x, y)", "((x) < (y))"));
    } else if (comparisonOperatorType == ComparisonOperatorType::LESS_EQUAL) {
        preprocessorDefines.insert(std::make_pair("comparator(x, y)", "((x) <= (y))"));
    } else if (comparisonOperatorType == ComparisonOperatorType::EQUAL) {
        preprocessorDefines.insert(std::make_pair("comparator(x, y)", "((x) == (y))"));
    } else if (comparisonOperatorType == ComparisonOperatorType::NOT_EQUAL) {
        preprocessorDefines.insert(std::make_pair("comparator(x, y)", "((x) != (y))"));
    }
    std::string shaderName = "SetPredicateCalculator.Compute";
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void SetPredicateComputePass::createComputeData(
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

void SetPredicateComputePass::_render() {
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
