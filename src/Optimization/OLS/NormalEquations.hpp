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

#ifndef CORRERENDER_NORMALEQUATIONS_HPP
#define CORRERENDER_NORMALEQUATIONS_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

class NormalEquationsComputePass : public sgl::vk::ComputePass {
public:
    explicit NormalEquationsComputePass(sgl::vk::Renderer* renderer);
    void setUseDoublePrecision(bool _useDoublePrecision);
    void setInputImages(
            const sgl::vk::ImageViewPtr& _inputImageGT,
            const sgl::vk::ImageViewPtr& _inputImageOpt);
    void setBuffers(
            uint32_t tfNumEntries, float minGT, float maxGT, float minOpt, float maxOpt,
            const sgl::vk::BufferPtr& _lhsBuffer,
            const sgl::vk::BufferPtr& _rhsBuffer,
            const sgl::vk::BufferPtr& _tfGTBuffer);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    struct UniformData {
        uint32_t xs, ys, zs, tfNumEntries;
        float minGT, maxGT, minOpt, maxOpt;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    bool useDoublePrecision = false;
    sgl::vk::ImageViewPtr inputImageGT, inputImageOpt;
    sgl::vk::BufferPtr lhsBuffer, rhsBuffer;
    sgl::vk::BufferPtr transferFunctionGTBuffer;
};

class NormalEquationsCopySymmetricPass : public sgl::vk::ComputePass {
public:
    explicit NormalEquationsCopySymmetricPass(sgl::vk::Renderer* renderer);
    void setUseDoublePrecision(bool _useDoublePrecision);
    void setBuffers(uint32_t _tfNumEntries, const sgl::vk::BufferPtr& _lhsBuffer);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t computeBlockSize = 64;
    bool useDoublePrecision = false;
    uint32_t tfNumEntries = 0;
    sgl::vk::BufferPtr lhsBuffer;
};

#endif //CORRERENDER_NORMALEQUATIONS_HPP
