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

#ifndef CORRERENDER_DEEPLEARNINGCUDACORRELATIONCALCULATOR_HPP
#define CORRERENDER_DEEPLEARNINGCUDACORRELATIONCALCULATOR_HPP

#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include "CorrelationCalculator.hpp"
#include "SymmetrizerType.hpp"

/// Symmetrizer operation used between encoder and decoder (@see SymmetrizerType.hpp).
const char* const SYMMETRIZER_TYPE_SHORT_NAMES[] = {
        "Add", "AddDiff", "Mul"
};

class DeepLearningCudaCorrelationCalculator : public ICorrelationCalculator {
public:
    /**
     * @param implName E.g., "tiny-cuda-nn" or "QuickMLP".
     * @param implNameKey E.g., "tinyCudaNN" or "quickMLP".
     * @param renderer The renderer object.
     */
    explicit DeepLearningCudaCorrelationCalculator(
            const std::string& implName, const std::string& implNameKey, sgl::vk::Renderer* renderer);
    void initialize() override;
    ~DeepLearningCudaCorrelationCalculator() override;
    std::string getOutputFieldName() override {
        std::string outputFieldName = "Correlation " + implName;
        if (calculatorConstructorUseCount > 1) {
            outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
        }
        return outputFieldName;
    }
    [[nodiscard]] bool getHasFixedRange() const override {
        return !isMutualInformationData;
    }
    [[nodiscard]] std::pair<float, float> getFixedRange() const override {
        if (calculateAbsoluteValue) {
            return std::make_pair(0.0f, 1.0f);
        } else {
            return std::make_pair(-1.0f, 1.0f);
        }
    }
    FilterDevice getFilterDevice() override { return FilterDevice::CUDA; }
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

protected:
    virtual void loadModelFromFile(const std::string& modelPath) = 0;
    void clearFieldImageViews() override {}

    // Inference steps to be implemented by subclasses.
    virtual void callbackBeginCompute() {}
    virtual void callbackEndCompute() {}
    virtual bool getIsModuleLoaded() = 0;
    virtual void recreateCache(int batchSize) = 0;
    virtual bool getCacheNeedsRecreate() { return cacheNeedsRecreate; }
    virtual CUdeviceptr getReferenceInputPointer() = 0;
    virtual CUdeviceptr getQueryInputPointer() = 0;
    virtual void runInferenceReference() = 0;
    virtual void runInferenceBatch(uint32_t batchOffset, uint32_t batchSize) = 0;
    virtual uint32_t getInputChannelAlignment() { return 1; }
    virtual uint32_t getSrnStride() { return 3; }

    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

    std::string modelFilePath;
    std::string fileDialogDirectory;
    NetworkType networkType = NetworkType::MINE;
    std::vector<std::string> modelPresets;
    std::vector<std::string> modelPresetFilenames;
    int modelPresetIndex = 0;

    /// For networkType == NetworkType::MINE.
    SymmetrizerType symmetrizerType = SymmetrizerType::Add;
    const int gpuBatchSize1DBase = 16384;

    /// For networkType == NetworkType::SRN_MINE.
    const int srnGpuBatchSize1DBase = 131072;
    size_t cachedVolumeDataSlice3dSize = 0;
    bool isMutualInformationData = true;
    bool calculateAbsoluteValue = false;

    size_t cachedCorrelationMemberCountDevice = std::numeric_limits<size_t>::max();
    bool cacheNeedsRecreate = false;

    size_t cachedNumSwapchainImages = 0;
    std::vector<sgl::vk::CommandBufferPtr> postRenderCommandBuffers;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr vulkanFinishedSemaphore, cudaFinishedSemaphore;
    uint64_t timelineValue = 0;
    CUdeviceptr permutationIndicesBufferCu{};
    CUdeviceptr outputImageBufferCu{};
    CUdeviceptr fieldTextureArrayCu{};
    std::vector<CUtexObject> cachedFieldTexturesCu;
    CUstream stream{};
    CUmodule combineCorrelationMembersModuleCu{};
    CUfunction combineCorrelationMembersFunctionCu{}, combineCorrelationMembersReferenceFunctionCu{};
    CUfunction combineCorrelationMembersAlignedFunctionCu{}, combineCorrelationMembersReferenceAlignedFunctionCu{};
    // For networkType == NetworkType::SRN_MINE.
    CUfunction writeGridPositionsFunctionCu{}, writeGridPositionReferenceFunctionCu{};

private:
    void parseModelPresetsFile(const std::string& filename);
    std::string implName, implNameKey, implNameKeyUpper, fileDialogKey, fileDialogDescription, modelFilePathSettingsKey;
};

#endif //CORRERENDER_DEEPLEARNINGCUDACORRELATIONCALCULATOR_HPP
