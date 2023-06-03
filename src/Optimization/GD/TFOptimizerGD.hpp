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

#ifndef CORRERENDER_TFOPTIMIZERGD_HPP
#define CORRERENDER_TFOPTIMIZERGD_HPP

#include "../TFOptimizer.hpp"

class GradientPass;
class OptimizerPass;

class TFOptimizerGD : public TFOptimizer {
public:
    TFOptimizerGD(sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute);
    ~TFOptimizerGD() override;
    void onRequestQueued(VolumeData* volumeData) override;
    void runOptimization(bool& shallStop, bool& hasStopped) override;
    float getProgress() override;

private:
    void runEpoch();

    int currentEpoch = 0;
    int maxNumEpochs = 0;

    struct UniformSettings {
        uint32_t xs, ys, zs, tfNumEntries;
        float minGT, maxGT, minOpt, maxOpt;
    };

    uint32_t cachedTfSize = 0;
    UniformSettings uniformSettings{};
    sgl::vk::BufferPtr settingsBuffer;
    sgl::vk::BufferPtr tfGTBuffer;
    sgl::vk::BufferPtr tfOptBuffer;
    sgl::vk::BufferPtr tfDownloadStagingBuffer;
    sgl::vk::BufferPtr tfOptGradientBuffer;
    sgl::vk::ImageViewPtr imageViewFieldGT, imageViewFieldOpt;

    // For Adam.
    sgl::vk::BufferPtr firstMomentEstimateBuffer;
    sgl::vk::BufferPtr secondMomentEstimateBuffer;

    // Compute passes.
    std::shared_ptr<GradientPass> gradientPass;
    std::shared_ptr<OptimizerPass> optimizerPass;
};

#endif //CORRERENDER_TFOPTIMIZERGD_HPP
