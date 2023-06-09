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

#ifndef CORRERENDER_TFOPTIMIZEROLS_HPP
#define CORRERENDER_TFOPTIMIZEROLS_HPP

#include "../TFOptimizer.hpp"

struct TFOptimizerOLSCache;
class NormalEquationsComputePass;
class NormalEquationsCopySymmetricPass;

class TFOptimizerOLS : public TFOptimizer {
public:
    TFOptimizerOLS(sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute);
    ~TFOptimizerOLS() override;
    void onRequestQueued(VolumeData* volumeData) override;
    void runOptimization(bool& shallStop, bool& hasStopped) override;
    float getProgress() override;

private:
    TFOptimizerOLSCache* cache = nullptr;
    void buildSystemDense();
    void buildSystemSparse();

    std::shared_ptr<NormalEquationsComputePass> normalEquationsComputePass;
    std::shared_ptr<NormalEquationsCopySymmetricPass> normalEquationsCopySymmetricPass;

    // Non-templated cache data.
    uint32_t cachedTfSize = 0;
    uint32_t cachedNumVoxels = 0;
    uint32_t cachedXs = 0, cachedYs = 0, cachedZs = 0;
    std::shared_ptr<HostCacheEntryType> fieldEntryGT, fieldEntryOpt;
    std::pair<float, float> minMaxGT, minMaxOpt;
    std::vector<glm::vec4> tfGT;
    // Implicit matrix.
    sgl::vk::ImageViewPtr inputImageGT, inputImageOpt;
    sgl::vk::BufferPtr lhsBuffer, rhsBuffer;
    sgl::vk::BufferPtr lhsStagingBuffer, rhsStagingBuffer;
    sgl::vk::BufferPtr tfGTBuffer;
#ifdef CUDA_ENABLED
    sgl::vk::TextureCudaExternalMemoryVkPtr cudaInputImageGT, cudaInputImageOpt;
#endif
};

#endif //CORRERENDER_TFOPTIMIZEROLS_HPP
