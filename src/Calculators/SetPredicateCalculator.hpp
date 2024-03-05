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

#ifndef CORRERENDER_SETPREDICATECALCULATOR_HPP
#define CORRERENDER_SETPREDICATECALCULATOR_HPP

#include <vector>

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Volume/Cache/HostCacheEntry.hpp"
#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "CorrelationDefines.hpp"
#include "Calculator.hpp"

enum class ComparisonOperatorType {
    GREATER, GREATER_EQUAL, LESS, LESS_EQUAL, EQUAL, NOT_EQUAL
};
const char* const COMPARISON_OPERATOR_NAMES[] = {
        ">", ">=", "<", "<=", "==", "!="
};

typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
typedef std::shared_ptr<DeviceCacheEntryType> DeviceCacheEntry;

class SetPredicateComputePass;

/**
 * Computes how many variables fulfill a predicate on a variable across all ensemble members or time steps.
 * Where [   ] [        ] [   ] [  ] [   ]
 * Where [NUM] [ENS/TIME] [VAR] [OP] [VAL]
 * Where [900] [ens. mem] [REF] [>=] [ 10]
 *
 * Mode: [ENS/TIME]
 * Number: [NUM]
 * Variable: [VAR]
 * Operator: [>=]
 * Value: [VAL]
 */
class SetPredicateCalculator : public Calculator {
public:
    explicit SetPredicateCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::SET_PREDICATE; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    [[nodiscard]] bool getShouldRenderGui() const override { return true; }
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    std::string getOutputFieldName() override;
    FilterDevice getFilterDevice() override { if (useGpu) return FilterDevice::VULKAN; else return FilterDevice::CPU; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;
    void renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor);

private:
    int getCorrelationMemberCount() const;
    HostCacheEntry getFieldEntryCpu(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx);
    DeviceCacheEntry getFieldEntryDevice(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx, bool wantsImageData);
    std::pair<float, float> getMinMaxScalarFieldValue(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx);
    void onCorrelationMemberCountChanged();
    void clearFieldDeviceData();
    bool getSupportsBufferMode();

    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    std::shared_ptr<SetPredicateComputePass> setPredicateComputePass;
    int cachedMemberCount = 0;
    CorrelationDataMode dataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool useGpu = true;
    bool useBufferTiling = true;

    // Settings.
    bool isEnsembleMode = true; //< Ensemble or time mode?
    ComparisonOperatorType comparisonOperatorType = ComparisonOperatorType::GREATER_EQUAL;
    int countLower = 1, countUpper = 1;
    float comparisonValue = 0.0f;
    bool useFuzzyLogic = false;
    int scalarFieldIndex = 0;
    int scalarFieldIndexGui = 0;
};

class SetPredicateComputePass : public sgl::vk::ComputePass {
public:
    explicit SetPredicateComputePass(sgl::vk::Renderer* renderer);
    void setVolumeData(VolumeData* _volumeData, int correlationMemberCount, bool isNewData);
    void setCorrelationMemberCount(int correlationMemberCount);
    void setDataMode(CorrelationDataMode _dataMode);
    void setUseBufferTiling(bool _useBufferTiling);
    void setFieldBuffers(const std::vector<sgl::vk::BufferPtr>& _fieldBuffers);
    void setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews);
    void setOutputImage(const sgl::vk::ImageViewPtr& _outputImage);
    void setComparisonOperatorType(ComparisonOperatorType _comparisonOperatorType);
    void setCountLower(int _countLower);
    void setCountUpper(int _countUpper);
    void setComparisonValue(float _comparisonValue);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    ComparisonOperatorType comparisonOperatorType = ComparisonOperatorType::GREATER_EQUAL;

    VolumeData* volumeData = nullptr;
    int cachedCorrelationMemberCount = 0;

    CorrelationDataMode dataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool useBufferTiling = true;
    std::vector<sgl::vk::BufferPtr> fieldBuffers;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;

    const uint32_t computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    struct UniformData {
        uint32_t xs, ys, zs, cs;
        int countLower, countUpper;
        float comparisonValue;
        float paddingUniform;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    sgl::vk::ImageViewPtr outputImage;
};

#endif //CORRERENDER_SETPREDICATECALCULATOR_HPP
