/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#ifndef CORRERENDER_CORRELATIONCALCULATOR_HPP
#define CORRERENDER_CORRELATIONCALCULATOR_HPP

#include <vector>
#include <glm/vec3.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Volume/Cache/HostCacheEntry.hpp"
#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "Calculator.hpp"
#include "CorrelationDefines.hpp"

#ifdef SUPPORT_CUDA_INTEROP
#include <cuda.h>

namespace sgl { namespace vk {
class SemaphoreVkCudaDriverApiInterop;
typedef std::shared_ptr<SemaphoreVkCudaDriverApiInterop> SemaphoreVkCudaDriverApiInteropPtr;
}}
#endif

class ReferencePointSelectionRenderer;

typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
typedef std::shared_ptr<DeviceCacheEntryType> DeviceCacheEntry;

class ICorrelationCalculator : public Calculator {
public:
    explicit ICorrelationCalculator(sgl::vk::Renderer* renderer);
    void setViewManager(ViewManager* _viewManager) override;
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    [[nodiscard]] inline int getInputFieldIndex() const { return fieldIndex; }
    [[nodiscard]] bool getComputesCorrelation() const override { return true; }
    [[nodiscard]] virtual bool getIsRealtime() const { return false; }
    [[nodiscard]] bool getShouldRenderGui() const override { return true; }
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    [[nodiscard]] bool getHasFixedRange() const override { return false; }
    RendererPtr getCalculatorRenderer() override { return calculatorRenderer; }
    void update(float dt) override;
    void setReferencePoint(const glm::ivec3& referencePoint);
    void setReferencePointFromWorld(const glm::vec3& worldPosition);

    /// Returns whether ensemble or time correlation mode is used.
    [[nodiscard]] inline bool getIsEnsembleMode() const { return isEnsembleMode; }
    int getCorrelationMemberCount();
    HostCacheEntry getFieldEntryCpu(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx);
    DeviceCacheEntry getFieldEntryDevice(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx);
    std::pair<float, float> getMinMaxScalarFieldValue(
            const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx);

protected:
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

    ViewManager* viewManager = nullptr;
    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    int fieldIndex = 0, fieldIndexGui = 0;
    glm::ivec3 referencePointIndex{};
    RendererPtr calculatorRenderer;
    ReferencePointSelectionRenderer* referencePointSelectionRenderer = nullptr;
    bool continuousRecompute = false; ///< Debug option.

    virtual void onCorrelationMemberCountChanged() {}
    bool isEnsembleMode = true; //< Ensemble or time mode?

    // Focus point picking/moving information.
    void setReferencePointFromFocusPoint();
    bool fixPickingZPlane = true;
    bool hasHitInformation = false;
    glm::vec3 focusPoint{};
    glm::vec3 firstHit{}, lastHit{};
    glm::vec3 hitLookingDirection{};
};


class CorrelationComputePass;

/**
 * Correlation calculator with support for computing:
 * - Pearson correlation coefficient.
 * - Spearman rank correlation coefficient.
 * - Kendall rank correlation coefficient (aka. Kendall's tau).
 * - Binned mutual information estimator.
 * - Kraskov mutual information estimator.
 */
class CorrelationCalculator : public ICorrelationCalculator {
public:
    explicit CorrelationCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::CORRELATION; }
    std::string getOutputFieldName() override {
        std::string outputFieldName = CORRELATION_MEASURE_TYPE_NAMES[int(correlationMeasureType)];
        if (int(correlationMeasureType) <= int(CorrelationMeasureType::KENDALL)) {
            outputFieldName += " Correlation";
        }
        if (calculatorConstructorUseCount > 1) {
            outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
        }
        return outputFieldName;
    }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    [[nodiscard]] bool getIsRealtime() const override { return useGpu; }
    FilterDevice getFilterDevice() override;
    [[nodiscard]] bool getHasFixedRange() const override {
        return correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
                && correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    }
    [[nodiscard]] std::pair<float, float> getFixedRange() const override {
        if (correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
                && correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
            if (calculateAbsoluteValue) {
                return std::make_pair(0.0f, 1.0f);
            } else {
                return std::make_pair(-1.0f, 1.0f);
            }
        } else {
            return std::make_pair(0.0f, 1.0f);
        }
    }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

protected:
    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;
    void onCorrelationMemberCountChanged() override;

private:
    std::shared_ptr<CorrelationComputePass> correlationComputePass;
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    bool useGpu = true;
    bool useCuda = false; ///< Currently only for CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    bool calculateAbsoluteValue = false; ///< Whether to use absolute value for non-MI correlations.
    int numBins = 80; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_BINNED.
    int k = 3; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    int kMax = 20; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
    int kraskovEstimatorIndex = 1; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV.
};

class SpearmanReferenceRankComputePass;
struct CorrelationCalculatorKernelCache;

class CorrelationComputePass : public sgl::vk::ComputePass {
public:
    explicit CorrelationComputePass(sgl::vk::Renderer* renderer);
    ~CorrelationComputePass() override;
    void setVolumeData(VolumeData* _volumeData, int correlationMemberCount, bool isNewData);
    void setCorrelationMemberCount(int correlationMemberCount);
    void setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews);
    void setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType);
    void setCalculateAbsoluteValue(bool _calculateAbsoluteValue);
    void setNumBins(int _numBins);
    void setKraskovNumNeighbors(int _k);
    void setKraskovEstimatorIndex(int _kraskovEstimatorIndex);
    void computeCuda(
            CorrelationCalculator* correlationCalculator,
            const std::string& fieldName, int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry,
            glm::ivec3& referencePointIndex);

    // 3D field evaluation mode (one fixed reference point, output image) - for CorrelationCalculator.
    void setReferencePoint(const glm::ivec3& referencePointIndex);
    void setOutputImage(const sgl::vk::ImageViewPtr& _outputImage);

    // Request evaluation mode (one request buffer with reference and query positions, output buffer) - for HEBChart.
    void setUseRequestEvaluationMode(bool _useRequestEvaluationMode);
    void setNumRequests(uint32_t _numRequests);
    void setRequestsBuffer(const sgl::vk::BufferPtr& _requestsBuffer);
    void setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    VolumeData* volumeData = nullptr;
    int cachedCorrelationMemberCount = 0;
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;

    const uint32_t batchCorrelationMemberCountThresholdPearson = 1024;
    const uint32_t batchCorrelationMemberCountThresholdSpearman = 10;
    const uint32_t batchCorrelationMemberCountThresholdKendall = 10;
    const uint32_t batchCorrelationMemberCountThresholdMiBinned = 10;
    const uint32_t batchCorrelationMemberCountThresholdKraskov = 10;
    const int computeBlockSize1D = 256;
    const int computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    struct UniformData {
        uint32_t xs, ys, zs, cs;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;

    // 3D field evaluation mode.
    sgl::vk::ImageViewPtr outputImage;

    // Request evaluation mode.
    bool useRequestEvaluationMode = false;
    uint32_t numRequests = 0;
    sgl::vk::BufferPtr requestsBuffer;
    sgl::vk::BufferPtr outputBuffer;

    // For non-MI correlations.
    bool calculateAbsoluteValue = false;

    // For Spearman correlation.
    std::shared_ptr<SpearmanReferenceRankComputePass> spearmanReferenceRankComputePass;
    sgl::vk::BufferPtr referenceRankBuffer;

    ///< For CorrelationMeasureType::MUTUAL_INFORMATION_BINNED.
    int numBins = 80;

    // For Kraskov mutual information (MI) estimator.
    int k = 3;
    int kraskovEstimatorIndex = 1;

#ifdef SUPPORT_CUDA_INTEROP
    // For CUDA implementation of estimators.
    size_t cachedNumSwapchainImages = 0;
    std::vector<sgl::vk::CommandBufferPtr> postRenderCommandBuffers;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr vulkanFinishedSemaphore, cudaFinishedSemaphore;
    uint64_t timelineValue = 0;
    size_t cachedCorrelationMemberCountDevice = std::numeric_limits<size_t>::max();
    size_t cachedVolumeDataSlice3dSize = 0;
    CUdeviceptr outputImageBufferCu{};
    CUdeviceptr fieldTextureArrayCu{};
    std::vector<CUtexObject> cachedFieldTexturesCu;
    CUstream stream{};
    CorrelationCalculatorKernelCache* kernelCache = nullptr;
#endif
};

class SpearmanReferenceRankComputePass : public sgl::vk::ComputePass {
public:
    SpearmanReferenceRankComputePass(sgl::vk::Renderer* renderer, sgl::vk::BufferPtr uniformBuffer);
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void setCorrelationMemberCount(int correlationMemberCount);
    void setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews);
    inline const sgl::vk::BufferPtr& getReferenceRankBuffer() { return referenceRankBuffer; }

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    VolumeData* volumeData = nullptr;
    int cachedCorrelationMemberCount = 0;

    const int computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    sgl::vk::BufferPtr uniformBuffer;

    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    sgl::vk::BufferPtr referenceRankBuffer;
};

#endif //CORRERENDER_CORRELATIONCALCULATOR_HPP
