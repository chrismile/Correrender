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

#ifndef CORRERENDER_OPTIMIZERPASS_HPP
#define CORRERENDER_OPTIMIZERPASS_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Optimization/OptDefines.hpp"

class OptimizerPass : public sgl::vk::ComputePass {
public:
    explicit OptimizerPass(sgl::vk::Renderer* renderer);
    void setOptimizerType(OptimizerType _optimizerType);
    void setBuffers(
            const sgl::vk::BufferPtr& _tfOptBuffer,
            const sgl::vk::BufferPtr& _tfOptGradientBuffer);
    void setSettings(LossType _lossType, uint32_t _numTfEntries, float alpha, float beta1, float beta2, float epsilon);
    void setEpochIndex(int epochIdx);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    OptimizerType optimizerType = OptimizerType::ADAM;
    const uint32_t computeBlockSize = 64;
    struct UniformData {
        float alpha; ///< Learning rate.
        float beta1; ///< First moment update rate.
        float beta2; ///< Second moment update rate.
        float epsilon; ///< Small epsilon value used to avoid division by zero.
    };
    LossType lossType = LossType::L2;
    uint32_t numTfEntries = 0;
    float t = 1.0f; ///< Time step.
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    sgl::vk::BufferPtr tfOptBuffer, tfOptGradientBuffer;
    sgl::vk::BufferPtr firstMomentEstimateBuffer, secondMomentEstimateBuffer;
};

#endif //CORRERENDER_OPTIMIZERPASS_HPP
