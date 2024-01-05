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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "Histogram.hpp"

MinMaxBufferWritePass::MinMaxBufferWritePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}

void MinMaxBufferWritePass::setBuffers(
        const sgl::vk::BufferPtr& _dataBuffer, const sgl::vk::BufferPtr& _minMaxOutBuffer) {
    dataBuffer = _dataBuffer;
    minMaxOutBuffer = _minMaxOutBuffer;
    if (computeData) {
        computeData->setStaticBuffer(dataBuffer, "DataBuffer");
        computeData->setStaticBuffer(minMaxOutBuffer, "MinMaxOutBuffer");
    }
}

void MinMaxBufferWritePass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("FLT_MAX", std::to_string(std::numeric_limits<float>::max())));
    preprocessorDefines.insert(std::make_pair("FLT_LOWEST", std::to_string(std::numeric_limits<float>::lowest())));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"MinMaxWrite.Compute"}, preprocessorDefines);
}

void MinMaxBufferWritePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(dataBuffer, "DataBuffer");
    computeData->setStaticBuffer(minMaxOutBuffer, "MinMaxOutBuffer");
}

void MinMaxBufferWritePass::_render() {
    auto inputSize = uint32_t(dataBuffer->getSizeInBytes() / sizeof(float));
    renderer->pushConstants(
            std::static_pointer_cast<sgl::vk::Pipeline>(computeData->getComputePipeline()),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, inputSize);
    renderer->dispatch(computeData, sgl::uiceil(inputSize, BLOCK_SIZE), 1, 1);
}


MinMaxDepthReductionPass::MinMaxDepthReductionPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}

void MinMaxDepthReductionPass::setBuffers(
        const sgl::vk::BufferPtr &bufferIn, const sgl::vk::BufferPtr &bufferOut) {
    minMaxInBuffer = bufferIn;
    minMaxOutBuffer = bufferOut;
    if (computeData) {
        computeData->setStaticBuffer(minMaxInBuffer, "MinMaxInBuffer");
        computeData->setStaticBuffer(minMaxOutBuffer, "MinMaxOutBuffer");
    }
}

void MinMaxDepthReductionPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("FLT_MAX", std::to_string(std::numeric_limits<float>::max())));
    preprocessorDefines.insert(std::make_pair("FLT_LOWEST", std::to_string(std::numeric_limits<float>::lowest())));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"MinMaxReduce.Compute"}, preprocessorDefines);
}

void MinMaxDepthReductionPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(minMaxInBuffer, "MinMaxInBuffer");
    computeData->setStaticBuffer(minMaxOutBuffer, "MinMaxOutBuffer");
}

void MinMaxDepthReductionPass::_render() {
    renderer->pushConstants(
            std::static_pointer_cast<sgl::vk::Pipeline>(computeData->getComputePipeline()),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, inputSize);
    renderer->dispatch(computeData, sgl::uiceil(inputSize, BLOCK_SIZE * 2u), 1, 1);
}


ComputeHistogramPass::ComputeHistogramPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}

void ComputeHistogramPass::setBuffers(
        const sgl::vk::BufferPtr& _minMaxBuffer, const sgl::vk::BufferPtr& _valuesBuffer,
        uint32_t _histogramSize, const sgl::vk::BufferPtr& _histogramBuffer) {
    if (histogramSize != _histogramSize) {
        histogramSize = _histogramSize;
        setShaderDirty();
    }
    minMaxBuffer = _minMaxBuffer;
    valuesBuffer = _valuesBuffer;
    histogramBuffer = _histogramBuffer;
    if (computeData) {
        computeData->setStaticBuffer(minMaxBuffer, "MinMaxBuffer");
        computeData->setStaticBuffer(valuesBuffer, "ValuesBuffer");
        computeData->setStaticBuffer(histogramBuffer, "HistogramBuffer");
    }
}

void ComputeHistogramPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("WORK_PER_THREAD", std::to_string(WORK_PER_THREAD)));
    preprocessorDefines.insert(std::make_pair("HISTOGRAM_SIZE", std::to_string(histogramSize)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"ComputeHistogram.Compute"}, preprocessorDefines);
}

void ComputeHistogramPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(minMaxBuffer, "MinMaxBuffer");
    computeData->setStaticBuffer(valuesBuffer, "ValuesBuffer");
    computeData->setStaticBuffer(histogramBuffer, "HistogramBuffer");
}

void ComputeHistogramPass::_render() {
    const auto inputSize = uint32_t(valuesBuffer->getSizeInBytes() / sizeof(float));
    renderer->pushConstants(
            std::static_pointer_cast<sgl::vk::Pipeline>(computeData->getComputePipeline()),
            VK_SHADER_STAGE_COMPUTE_BIT, 0, inputSize);
    renderer->dispatch(computeData, sgl::uiceil(inputSize, BLOCK_SIZE), 1, 1);
}


ComputeHistogramMaxPass::ComputeHistogramMaxPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}

void ComputeHistogramMaxPass::setBuffers(
        const sgl::vk::BufferPtr& _maxValueBuffer,
        uint32_t _histogramSize, const sgl::vk::BufferPtr& _histogramBuffer) {
    if (histogramSize != _histogramSize) {
        histogramSize = _histogramSize;
        setShaderDirty();
    }
    maxValueBuffer = _maxValueBuffer;
    histogramBuffer = _histogramBuffer;
    if (computeData) {
        computeData->setStaticBuffer(maxValueBuffer, "MaxValueBuffer");
        computeData->setStaticBuffer(histogramBuffer, "HistogramBuffer");
    }
}

void ComputeHistogramMaxPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("HISTOGRAM_SIZE", std::to_string(histogramSize)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"ComputeHistogramMax.Compute"}, preprocessorDefines);
}

void ComputeHistogramMaxPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(maxValueBuffer, "MaxValueBuffer");
    computeData->setStaticBuffer(histogramBuffer, "HistogramBuffer");
}

void ComputeHistogramMaxPass::_render() {
    renderer->dispatch(computeData, sgl::uiceil(histogramSize, BLOCK_SIZE), 1, 1);
}


ComputeHistogramDividePass::ComputeHistogramDividePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}

void ComputeHistogramDividePass::setBuffers(
        const sgl::vk::BufferPtr& _maxValueBuffer, uint32_t _histogramSize,
        const sgl::vk::BufferPtr& _histogramUintBuffer, const sgl::vk::BufferPtr& _histogramFloatBuffer) {
    if (histogramSize != _histogramSize) {
        histogramSize = _histogramSize;
        setShaderDirty();
    }
    maxValueBuffer = _maxValueBuffer;
    histogramUintBuffer = _histogramUintBuffer;
    histogramFloatBuffer = _histogramFloatBuffer;
    if (computeData) {
        computeData->setStaticBuffer(maxValueBuffer, "MaxValueBuffer");
        computeData->setStaticBuffer(histogramUintBuffer, "HistogramUintBuffer");
        computeData->setStaticBuffer(histogramFloatBuffer, "HistogramFloatBuffer");
    }
}

void ComputeHistogramDividePass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("HISTOGRAM_SIZE", std::to_string(histogramSize)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"ComputeHistogramDivide.Compute"}, preprocessorDefines);
}

void ComputeHistogramDividePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(maxValueBuffer, "MaxValueBuffer");
    computeData->setStaticBuffer(histogramUintBuffer, "HistogramUintBuffer");
    computeData->setStaticBuffer(histogramFloatBuffer, "HistogramFloatBuffer");
}

void ComputeHistogramDividePass::_render() {
    renderer->dispatch(computeData, sgl::uiceil(histogramSize, BLOCK_SIZE), 1, 1);
}


DivergentMinMaxPass::DivergentMinMaxPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}

void DivergentMinMaxPass::setBuffers(const sgl::vk::BufferPtr& _minMaxBuffer) {
    minMaxBuffer = _minMaxBuffer;
    if (computeData) {
        computeData->setStaticBuffer(minMaxBuffer, "MinMaxBuffer");
    }
}

void DivergentMinMaxPass::loadShader() {
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"DivergentMinMax.Compute"});
}

void DivergentMinMaxPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(minMaxBuffer, "MinMaxBuffer");
}

void DivergentMinMaxPass::_render() {
    renderer->dispatch(computeData, 1, 1, 1);
}
