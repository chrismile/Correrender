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

#ifndef CORRERENDER_DEEPLEARNINGCUDASIMILARITYCALCULATOR_HPP
#define CORRERENDER_DEEPLEARNINGCUDASIMILARITYCALCULATOR_HPP

#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include "SimilarityCalculator.hpp"

class DeepLearningCudaSimilarityCalculator : public EnsembleSimilarityCalculator {
public:
    /**
     * @param implName E.g., "tiny-cuda-nn" or "QuickMLP".
     * @param implNameKey E.g., "tinyCudaNN" or "quickMLP".
     * @param renderer The renderer object.
     */
    explicit DeepLearningCudaSimilarityCalculator(
            const std::string& implName, const std::string& implNameKey, sgl::vk::Renderer* renderer);
    void initialize() override;
    ~DeepLearningCudaSimilarityCalculator() override;
    std::string getOutputFieldName() override { return "Similarity Metric (" + implName + ")"; }
    FilterDevice getFilterDevice() override { return FilterDevice::CUDA; }
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

protected:
    virtual void loadModelFromFile(const std::string& modelPath) = 0;

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

    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

    std::string modelFilePath;
    std::string fileDialogDirectory;

    const int gpuBatchSize1DBase = 16384;
    size_t cachedEnsembleSizeDevice = 0;
    bool cacheNeedsRecreate = false;

    size_t cachedNumSwapchainImages = 0;
    std::vector<sgl::vk::CommandBufferPtr> postRenderCommandBuffers;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr vulkanFinishedSemaphore, cudaFinishedSemaphore;
    uint64_t timelineValue = 0;
    CUdeviceptr permutationIndicesBufferCu{};
    CUdeviceptr outputImageBufferCu{};
    CUdeviceptr ensembleTextureArrayCu{};
    std::vector<CUtexObject> cachedEnsembleTexturesCu;
    CUmodule combineEnsemblesModuleCu{};
    CUfunction combineEnsemblesFunctionCu{}, combineEnsemblesReferenceFunctionCu{};
    CUfunction combineEnsemblesAlignedFunctionCu{}, combineEnsemblesReferenceAlignedFunctionCu{};
    CUstream stream{};

private:
    std::string implName, implNameKey, implNameKeyUpper, fileDialogKey, fileDialogDescription, modelFilePathSettingsKey;
};

#endif //CORRERENDER_DEEPLEARNINGCUDASIMILARITYCALCULATOR_HPP
