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

class NormalEquationsComputePass;
class NormalEquationsCopySymmetricPass;

template<class Real>
struct TFOptimizerOLSCacheTyped;

template<class Real>
class TFOptimizerOLSTyped {
public:
    TFOptimizerOLSTyped(
            sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute,
            sgl::vk::FencePtr fence, VkCommandPool commandPool, VkCommandBuffer commandBuffer,
            TFOptimizationWorkerSettings& settings, std::vector<glm::vec4>& tfArrayOpt);
    ~TFOptimizerOLSTyped();
    void clearCache();
    void onRequestQueued(VolumeData* volumeData);
    void runOptimization(bool& shallStop, bool& hasStopped);

private:
    TFOptimizerOLSCacheTyped<Real>* cache = nullptr;
    void buildSystemDense();
    void buildSystemSparse();

    sgl::vk::Renderer* renderer;
    sgl::vk::Renderer* parentRenderer;
    bool supportsAsyncCompute = false;

    sgl::vk::FencePtr fence{};
    VkCommandPool commandPool{};
    VkCommandBuffer commandBuffer{};

    TFOptimizationWorkerSettings& settings;
    std::vector<glm::vec4>& tfArrayOpt;

    std::shared_ptr<NormalEquationsComputePass> normalEquationsComputePass;
    std::shared_ptr<NormalEquationsCopySymmetricPass> normalEquationsCopySymmetricPass;
};

class TFOptimizerOLS : public TFOptimizer {
public:
    TFOptimizerOLS(sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute);
    ~TFOptimizerOLS() override;
    void onRequestQueued(VolumeData* volumeData) override;
    void runOptimization(bool& shallStop, bool& hasStopped) override;
    float getProgress() override;

private:
    TFOptimizerOLSTyped<float>* fopt;
    TFOptimizerOLSTyped<double>* dopt;

};

#endif //CORRERENDER_TFOPTIMIZEROLS_HPP
