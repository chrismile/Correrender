/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#ifndef CORRERENDER_HISTOGRAM_HPP
#define CORRERENDER_HISTOGRAM_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

class MinMaxBufferWritePass : public sgl::vk::ComputePass {
public:
    explicit MinMaxBufferWritePass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setBuffers(const sgl::vk::BufferPtr& _dataBuffer, const sgl::vk::BufferPtr& _minMaxOutBuffer);

private:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    const uint32_t BLOCK_SIZE = 256;
    sgl::vk::BufferPtr dataBuffer;
    sgl::vk::BufferPtr minMaxOutBuffer;
};

class MinMaxDepthReductionPass : public sgl::vk::ComputePass {
public:
    explicit MinMaxDepthReductionPass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setBuffers(const sgl::vk::BufferPtr& bufferIn, const sgl::vk::BufferPtr& bufferOut);
    inline void setInputSize(uint32_t _inputSize) { inputSize = _inputSize; }

private:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    const uint32_t BLOCK_SIZE = 256;
    sgl::vk::BufferPtr minMaxInBuffer;
    sgl::vk::BufferPtr minMaxOutBuffer;
    uint32_t inputSize = 0;
};

class ComputeHistogramPass : public sgl::vk::ComputePass {
public:
    explicit ComputeHistogramPass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setBuffers(
            const sgl::vk::BufferPtr& _minMaxBuffer, const sgl::vk::BufferPtr& _valuesBuffer,
            uint32_t _histogramSize, const sgl::vk::BufferPtr& _histogramBuffer);

private:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    const uint32_t BLOCK_SIZE = 64;
    const uint32_t WORK_PER_THREAD = 8;
    uint32_t histogramSize = 0;
    sgl::vk::BufferPtr minMaxBuffer;
    sgl::vk::BufferPtr valuesBuffer;
    sgl::vk::BufferPtr histogramBuffer;
};

class ComputeHistogramMaxPass : public sgl::vk::ComputePass {
public:
    explicit ComputeHistogramMaxPass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setBuffers(
            const sgl::vk::BufferPtr& _maxValueBuffer,
            uint32_t _histogramSize, const sgl::vk::BufferPtr& _histogramBuffer);

private:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    const uint32_t BLOCK_SIZE = 64;
    uint32_t histogramSize = 0;
    sgl::vk::BufferPtr maxValueBuffer;
    sgl::vk::BufferPtr histogramBuffer;
};

class ComputeHistogramDividePass : public sgl::vk::ComputePass {
public:
    explicit ComputeHistogramDividePass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setBuffers(
            const sgl::vk::BufferPtr& _maxValueBuffer, uint32_t _histogramSize,
            const sgl::vk::BufferPtr& _histogramUintBuffer, const sgl::vk::BufferPtr& _histogramFloatBuffer);

private:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    const uint32_t BLOCK_SIZE = 64;
    uint32_t histogramSize = 0;
    sgl::vk::BufferPtr maxValueBuffer;
    sgl::vk::BufferPtr histogramUintBuffer;
    sgl::vk::BufferPtr histogramFloatBuffer;
};

class DivergentMinMaxPass : public sgl::vk::ComputePass {
public:
    explicit DivergentMinMaxPass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setBuffers(const sgl::vk::BufferPtr& _minMaxBuffer);

private:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    sgl::vk::BufferPtr minMaxBuffer;
};

#endif //CORRERENDER_HISTOGRAM_HPP
