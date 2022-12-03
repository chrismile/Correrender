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

#ifndef CORRERENDER_TINYCUDANNSIMILARITYCALCULATOR_HPP
#define CORRERENDER_TINYCUDANNSIMILARITYCALCULATOR_HPP

#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include "SimilarityCalculator.hpp"

struct TinyCudaNNModuleWrapper;
struct TinyCudaNNCacheWrapper;

class TinyCudaNNSimilarityCalculator : public EnsembleSimilarityCalculator {
public:
    explicit TinyCudaNNSimilarityCalculator(sgl::vk::Renderer* renderer);
    ~TinyCudaNNSimilarityCalculator() override;
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    std::string getOutputFieldName() override { return "Similarity Metric (tiny-cuda-nn)"; }
    FilterDevice getFilterDevice() override { return FilterDevice::CUDA; }
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

protected:
    void loadModelFromFile(const std::string& modelPath);

    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    std::string modelFilePath;
    std::shared_ptr<TinyCudaNNModuleWrapper> moduleWrapper;
    std::shared_ptr<TinyCudaNNCacheWrapper> cacheWrapper;
    std::string fileDialogDirectory;

    const int gpuBatchSize1DBase = 16384;
    size_t cachedEnsembleSizeDevice = 0;

    size_t cachedNumSwapchainImages = 0;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr referenceInputBufferCu, inputBufferCu;
    std::vector<sgl::vk::CommandBufferPtr> postRenderCommandBuffers;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr vulkanFinishedSemaphore, cudaFinishedSemaphore;
    uint64_t timelineValue = 0;
    CUdeviceptr outputImageBufferCu{};
    CUdeviceptr ensembleTextureArrayCu{};
    std::vector<CUtexObject> cachedEnsembleTexturesCu;
    CUmodule combineEnsemblesModuleCu{};
    CUfunction combineEnsemblesFunctionCu{}, combineEnsemblesReferenceFunctionCu{};
    CUstream stream{};
};

#endif //CORRERENDER_TINYCUDANNSIMILARITYCALCULATOR_HPP
