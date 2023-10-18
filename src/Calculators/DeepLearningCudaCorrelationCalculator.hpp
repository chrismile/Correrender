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

#include "Volume/Cache/AuxiliaryMemoryToken.hpp"
#include "VMLP/Format.hpp"
#include "SymmetrizerType.hpp"
#include "DeepLearningCorrelationCalculator.hpp"

class DeepLearningCudaCorrelationCalculator : public DeepLearningCorrelationCalculator {
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
    FilterDevice getFilterDevice() override { return FilterDevice::CUDA; }
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

protected:
    void computeNanStencilBuffer();
    void clearFieldDeviceData() override {}
    [[nodiscard]] bool getNeedsScalarFieldData() const override { return networkType == NetworkType::MINE; }

    // Inference steps to be implemented by subclasses.
    virtual void callbackBeginCompute() {}
    virtual void callbackEndCompute() {}
    virtual void recreateCache(int batchSize) = 0;
    virtual bool getCacheNeedsRecreate() { return cacheNeedsRecreate; }
    virtual CUdeviceptr getReferenceInputPointer() = 0;
    virtual CUdeviceptr getQueryInputPointer() = 0;
    virtual void runInferenceReference() = 0;
    virtual void runInferenceBatch(uint32_t batchOffset, uint32_t batchSize) = 0;
    virtual uint32_t getInputChannelAlignment() { return 1; }
    virtual uint32_t getSrnStride() { return 3; }

    SymmetrizerType symmetrizerType = SymmetrizerType::Add;

    /// For networkType == NetworkType::MINE.
    const int gpuBatchSize1DBase = 16384;

    /// For networkType == NetworkType::SRN_MINE.
    int srnGpuBatchSize1DBase = 131072;
    size_t cachedVolumeDataSlice3dSize = 0;

    /// NaN stencil for networkType == NetworkType::SRN_MINE.
    CUdeviceptr nonNanIndexBufferCu{};
    CUdeviceptr outputImageBufferUnpackedCu{};
    size_t cachedVolumeDataSlice3dSizeUnpacked = 0;

    size_t cachedCorrelationMemberCountDevice = std::numeric_limits<size_t>::max();
    bool cacheNeedsRecreate = false;

    size_t cachedNumSwapchainImages = 0;
    std::vector<sgl::vk::CommandBufferPtr> postRenderCommandBuffers;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr vulkanFinishedSemaphore, cudaFinishedSemaphore;
    uint64_t timelineValue = 0;
    CUdeviceptr permutationIndicesBufferCu{};
    CUdeviceptr outputImageBufferCu{};
    CUdeviceptr fieldTextureArrayCu{}, fieldBufferArrayCu{};
    std::vector<CUtexObject> cachedFieldTexturesCu;
    std::vector<CUdeviceptr> cachedFieldBuffersCu;
    CUstream stream{};
    CUmodule combineCorrelationMembersModuleCu{};
    // Functions that take a 3D image array as an input.
    CUfunction combineCorrelationMembersFunctionCu{}, combineCorrelationMembersReferenceFunctionCu{};
    CUfunction combineCorrelationMembersAlignedFunctionCu{}, combineCorrelationMembersReferenceAlignedFunctionCu{};
    // Functions that take a buffer as an input.
    CUfunction combineCorrelationMembersBufferFunctionCu{}, combineCorrelationMembersReferenceBufferFunctionCu{};
    CUfunction combineCorrelationMembersAlignedBufferFunctionCu{}, combineCorrelationMembersReferenceAlignedBufferFunctionCu{};
    // Functions that take a tiled buffer as an input.
    CUfunction combineCorrelationMembersBufferTiledFunctionCu{}, combineCorrelationMembersReferenceBufferTiledFunctionCu{};
    CUfunction combineCorrelationMembersAlignedBufferTiledFunctionCu{}, combineCorrelationMembersReferenceAlignedBufferTiledFunctionCu{};
    // For networkType == NetworkType::SRN_MINE.
    CUfunction writeGridPositionsFunctionCu{}, writeGridPositionsStencilFunctionCu{};
    CUfunction writeGridPositionReferenceFunctionCu{};
    CUfunction unpackStencilValuesFunctionCu{};
};

#endif //CORRERENDER_DEEPLEARNINGCUDACORRELATIONCALCULATOR_HPP
