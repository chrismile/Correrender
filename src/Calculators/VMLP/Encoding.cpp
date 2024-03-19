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

#include <iostream>
#include <numeric>
#include <json/json.h>

#include <Math/Math.hpp>
#include <Math/half/half.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Encoding.hpp"

namespace vmlp {

Encoding::Encoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding) {
    if (settingsEncoding["otype"] != "Composite") {
        channelOffset = settingsEncoding["dims_to_encode_begin"].asUInt();
        numChannelsEncode = settingsEncoding["n_dims_to_encode"].asUInt();
    }
}

class PaddingPass : public sgl::vk::ComputePass {
public:
    explicit PaddingPass(
            sgl::vk::Renderer* renderer, uint32_t channelOffset, uint32_t numChannelsEncode)
    : ComputePass(renderer), channelOffset(channelOffset), numChannelsEncode(numChannelsEncode) {}

    void setBufferOut(const Matrix& _matrixOut);
    void setBatchSize(uint32_t _batchSize);
    void setFloatFormat(FloatFormat _format);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t BLOCK_SIZE = 256;
    FloatFormat format = FloatFormat::FLOAT32;
    uint32_t channelOffset = 0, numChannelsEncode = 0;
    Matrix matrixOut;
    uint32_t batchSize = 0;
};

void PaddingPass::setBufferOut(const Matrix& _matrixOut) {
    matrixOut = _matrixOut;
    dataDirty = true;
}

void PaddingPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void PaddingPass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void PaddingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("OFFSET_OUT", std::to_string(channelOffset)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT", std::to_string(matrixOut.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_TO_ENCODE", std::to_string(numChannelsEncode)));
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Encodings.Padding.Compute"}, preprocessorDefines);
}

void PaddingPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(matrixOut.getBuffer(), "OutputBuffer");
}

void PaddingPass::_render() {
    const uint32_t numThreads = numChannelsEncode * batchSize;
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numThreads);
    renderer->dispatch(computeData, sgl::uiceil(numThreads, BLOCK_SIZE), 1, 1);
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            matrixOut.getBuffer());
}

CompositeEncoding::CompositeEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsCompositeEncoding)
        : Encoding(renderer, settingsCompositeEncoding) {
    std::string reductionTypeString = settingsCompositeEncoding.get("reduction", "Concatenation").asString();
    if (reductionTypeString == "Concatenation") {
        reductionType = ReductionType::CONCATENATION;
    } else if (reductionTypeString == "Sum") {
        reductionType = ReductionType::SUM;
    } else if (reductionTypeString == "Product") {
        reductionType = ReductionType::PRODUCT;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in CompositeEncoding::CompositeEncoding: Unsupported reduction type \""
                + reductionTypeString + "\".");
    }

    const auto& nested = settingsCompositeEncoding["nested"];
    for (const auto& settingsEncoding : nested) {
        auto encoding = createInputEncoding(renderer, settingsEncoding);
        encodings.push_back(encoding);
        numParameters += encoding->getNumParameters();
        uint32_t numChannelsOutEncoding = encoding->getNumChannelsOut();
        if (reductionType == ReductionType::CONCATENATION) {
            uint32_t alignment = encoding->getOutputAlignment();
            uint32_t alignmentOffset = numChannelsOutPadded % alignment;
            if (alignmentOffset != 0) {
                uint32_t numChannelsPadding = alignment - alignmentOffset;
                auto paddingPass = std::make_shared<PaddingPass>(renderer, numChannelsOutPadded, numChannelsPadding);
                paddingPasses.push_back(paddingPass);
                numChannelsOutPadded += numChannelsPadding;
            } else {
                paddingPasses.push_back(nullptr);
            }
            encodingsChannelOffsets.push_back(numChannelsOutPadded);
            numChannelsOutPadded += numChannelsOutEncoding;
            numChannelsOut += numChannelsOutEncoding;
        } else {
            paddingPasses.push_back(nullptr);
            encodingsChannelOffsets.push_back(0u);
            numChannelsOutPadded = std::max(numChannelsOutPadded, numChannelsOutEncoding);
            numChannelsOut = std::max(numChannelsOut, numChannelsOutEncoding);
        }
        numChannelsEncode = std::max(numChannelsEncode, encoding->getNumChannelsIn());
    }

    uint32_t alignmentOffsetOut = numChannelsOutPadded % ALIGNMENT_MLP;
    if (alignmentOffsetOut != 0) {
        numChannelsOutPadded += ALIGNMENT_MLP - alignmentOffsetOut;
    }
}

void CompositeEncoding::setInputOutputMatrices(const Matrix& input, const Matrix& output) {
    for (size_t i = 0; i < encodings.size(); i++) {
        auto& encoding = encodings.at(i);
        auto& paddingPass = paddingPasses.at(i);
        if (paddingPass) {
            paddingPass->setBufferOut(output);
        }
        uint32_t outputChannelOffset = encodingsChannelOffsets.at(i);
        encoding->setInputOutputMatrices(input, output.viewOffset(outputChannelOffset));
    }
    // For debug purposes.
    //this->input = input;
    //this->output = output;
}

void CompositeEncoding::setBatchSize(uint32_t _batchSize) {
    for (size_t i = 0; i < encodings.size(); i++) {
        auto& encoding = encodings.at(i);
        auto& paddingPass = paddingPasses.at(i);
        if (paddingPass) {
            paddingPass->setBatchSize(_batchSize);
        }
        encoding->setBatchSize(_batchSize);
    }
}

void CompositeEncoding::setFloatFormat(FloatFormat _format) {
    format = _format;
    for (size_t i = 0; i < encodings.size(); i++) {
        auto& encoding = encodings.at(i);
        auto& paddingPass = paddingPasses.at(i);
        if (paddingPass) {
            paddingPass->setFloatFormat(_format);
        }
        encoding->setFloatFormat(_format);
    }
}

void CompositeEncoding::setCompositionMode(CompositionMode _compositionMode) {
    if (!encodings.empty()) {
        std::static_pointer_cast<Encoding>(encodings.front())->setCompositionMode(CompositionMode::NONE);
    }
    for (size_t i = 1; i < encodings.size(); i++) {
        auto& encoding = encodings.at(i);
        std::static_pointer_cast<Encoding>(encoding)->setCompositionMode(_compositionMode);
    }
}

void CompositeEncoding::runInference() {
    for (size_t i = 0; i < encodings.size(); i++) {
        auto& encoding = encodings.at(i);
        auto& paddingPass = paddingPasses.at(i);
        if (paddingPass) {
            paddingPass->render();
        }
        encoding->runInference();
    }

    // For debug purposes.
    //debugPrintBuffer(input.getBuffer(), format, 10);
    //debugPrintBuffer(output.getBuffer(), format, 48);
    //std::cout << std::endl;
}


class IdentityEncodingPass : public sgl::vk::ComputePass {
public:
    explicit IdentityEncodingPass(
            sgl::vk::Renderer* renderer, uint32_t channelOffset, uint32_t numChannelsEncode)
            : ComputePass(renderer), channelOffset(channelOffset), numChannelsEncode(numChannelsEncode) {}

    void setBuffersInOut(const Matrix& _matrixIn, const Matrix& _matrixOut);
    void setBatchSize(uint32_t _batchSize);
    void setFloatFormat(FloatFormat _format);
    void setCompositionMode(CompositionMode _compositionMode);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t BLOCK_SIZE = 256;
    FloatFormat format = FloatFormat::FLOAT32;
    CompositionMode compositionMode = CompositionMode::NONE;
    uint32_t channelOffset = 0, numChannelsEncode = 0;
    Matrix matrixIn, matrixOut;
    uint32_t batchSize = 0;
};

void IdentityEncodingPass::setBuffersInOut(const Matrix& _matrixIn, const Matrix& _matrixOut) {
    matrixIn = _matrixIn;
    matrixOut = _matrixOut;
    dataDirty = true;
}

void IdentityEncodingPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void IdentityEncodingPass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void IdentityEncodingPass::setCompositionMode(CompositionMode _compositionMode) {
    compositionMode = _compositionMode;
    shaderDirty = true;
}

void IdentityEncodingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("OFFSET_IN", std::to_string(channelOffset)));
    preprocessorDefines.insert(std::make_pair("OFFSET_OUT", std::to_string(matrixOut.getOffsetChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN", std::to_string(matrixIn.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT", std::to_string(matrixOut.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_TO_ENCODE", std::to_string(numChannelsEncode)));
    if (compositionMode == CompositionMode::SUM) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_SUM", ""));
    } else if (compositionMode == CompositionMode::PRODUCT) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_PRODUCT", ""));
    }
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Encodings.Identity.Compute"}, preprocessorDefines);
}

void IdentityEncodingPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(matrixIn.getBuffer(), "InputBuffer");
    computeData->setStaticBuffer(matrixOut.getBuffer(), "OutputBuffer");
}

void IdentityEncodingPass::_render() {
    const uint32_t numThreads = numChannelsEncode * batchSize;
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numThreads);
    renderer->dispatch(computeData, sgl::uiceil(numThreads, BLOCK_SIZE), 1, 1);
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            matrixOut.getBuffer());
}

IdentityEncoding::IdentityEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding)
        : Encoding(renderer, settingsEncoding) {
    encodingPass = std::make_shared<IdentityEncodingPass>(renderer, channelOffset, numChannelsEncode);
}

void IdentityEncoding::setInputOutputMatrices(const Matrix& input, const Matrix& output) {
    encodingPass->setBuffersInOut(input, output);
}

void IdentityEncoding::setBatchSize(uint32_t _batchSize) {
    encodingPass->setBatchSize(_batchSize);
}

void IdentityEncoding::setFloatFormat(vmlp::FloatFormat _format) {
    format = _format;
    encodingPass->setFloatFormat(_format);
}

void IdentityEncoding::setCompositionMode(CompositionMode _compositionMode) {
    encodingPass->setCompositionMode(_compositionMode);
}

void IdentityEncoding::runInference() {
    encodingPass->render();
}


class FrequencyEncodingPass : public sgl::vk::ComputePass {
public:
    explicit FrequencyEncodingPass(
            sgl::vk::Renderer* renderer, uint32_t channelOffset, uint32_t numChannelsEncode, uint32_t numFrequencies)
            : ComputePass(renderer), channelOffset(channelOffset), numChannelsEncode(numChannelsEncode),
              numFrequencies(numFrequencies) {}

    void setBuffersInOut(const Matrix& _matrixIn, const Matrix& _matrixOut);
    void setBatchSize(uint32_t _batchSize);
    void setFloatFormat(FloatFormat _format);
    void setCompositionMode(CompositionMode _compositionMode);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t BLOCK_SIZE = 256;
    FloatFormat format = FloatFormat::FLOAT32;
    CompositionMode compositionMode = CompositionMode::NONE;
    uint32_t channelOffset = 0, numChannelsEncode = 0, numFrequencies = 0;
    Matrix matrixIn, matrixOut;
    uint32_t batchSize = 0;
};

void FrequencyEncodingPass::setBuffersInOut(const Matrix& _matrixIn, const Matrix& _matrixOut) {
    matrixIn = _matrixIn;
    matrixOut = _matrixOut;
    dataDirty = true;
}

void FrequencyEncodingPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void FrequencyEncodingPass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void FrequencyEncodingPass::setCompositionMode(CompositionMode _compositionMode) {
    compositionMode = _compositionMode;
    shaderDirty = true;
}

void FrequencyEncodingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("OFFSET_IN", std::to_string(channelOffset)));
    preprocessorDefines.insert(std::make_pair("OFFSET_OUT", std::to_string(matrixOut.getOffsetChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN", std::to_string(matrixIn.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT", std::to_string(matrixOut.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_TO_ENCODE", std::to_string(numChannelsEncode)));
    preprocessorDefines.insert(std::make_pair("NUM_FREQUENCIES", std::to_string(numFrequencies)));
    preprocessorDefines.insert(std::make_pair("PI", std::to_string(sgl::PI)));
    preprocessorDefines.insert(std::make_pair("PI_HALF", std::to_string(sgl::HALF_PI)));
    if (compositionMode == CompositionMode::SUM) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_SUM", ""));
    } else if (compositionMode == CompositionMode::PRODUCT) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_PRODUCT", ""));
    }
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Encodings.Frequency.Compute"}, preprocessorDefines);
}

void FrequencyEncodingPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(matrixIn.getBuffer(), "InputBuffer");
    computeData->setStaticBuffer(matrixOut.getBuffer(), "OutputBuffer");
}

void FrequencyEncodingPass::_render() {
    const uint32_t numThreads = numChannelsEncode * numFrequencies * 2 * batchSize;
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numThreads);
    renderer->dispatch(computeData, sgl::uiceil(numThreads, BLOCK_SIZE), 1, 1);
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            matrixOut.getBuffer());
}

FrequencyEncoding::FrequencyEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding)
        : Encoding(renderer, settingsEncoding) {
    numFrequencies = settingsEncoding["n_frequencies"].asUInt();
    encodingPass = std::make_shared<FrequencyEncodingPass>(renderer, channelOffset, numChannelsEncode, numFrequencies);
}

void FrequencyEncoding::setInputOutputMatrices(const Matrix& input, const Matrix& output) {
    encodingPass->setBuffersInOut(input, output);
}

void FrequencyEncoding::setBatchSize(uint32_t _batchSize) {
    encodingPass->setBatchSize(_batchSize);
}

void FrequencyEncoding::setFloatFormat(vmlp::FloatFormat _format) {
    format = _format;
    encodingPass->setFloatFormat(_format);
}

void FrequencyEncoding::setCompositionMode(CompositionMode _compositionMode) {
    encodingPass->setCompositionMode(_compositionMode);
}

void FrequencyEncoding::runInference() {
    encodingPass->render();
}


class GridEncodingPass : public sgl::vk::ComputePass {
public:
    explicit GridEncodingPass(
            sgl::vk::Renderer* renderer, uint32_t channelOffset, uint32_t numChannelsEncode,
            GridType gridType, InterpolationType interpolationType, HashType hashType,
            uint32_t numLevels, uint32_t numFeaturesPerLevel, uint32_t numFeatures,
            uint32_t log2HashMapSize, uint32_t baseResolution, float perLevelScale,
            sgl::vk::BufferPtr& parametersBuffer, sgl::vk::BufferPtr offsetTableBuffer)
            : ComputePass(renderer), channelOffset(channelOffset), numChannelsEncode(numChannelsEncode),
              gridType(gridType), interpolationType(interpolationType), hashType(hashType),
              numLevels(numLevels), numFeaturesPerLevel(numFeaturesPerLevel), numFeatures(numFeatures),
              log2HashMapSize(log2HashMapSize), baseResolution(baseResolution), perLevelScale(perLevelScale),
              parametersBuffer(parametersBuffer), offsetTableBuffer(std::move(offsetTableBuffer)) {
        uniformData.baseResolution = baseResolution;
        uniformData.log2PerLevelScale = std::log2(perLevelScale);
        uniformBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(UniformData), &uniformData,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    }

    void setBuffersInOut(const Matrix& _matrixIn, const sgl::vk::BufferPtr& _bufferOut);
    void setBatchSize(uint32_t _batchSize);
    void setFloatFormat(FloatFormat _format);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t BLOCK_SIZE = 256;
    FloatFormat format = FloatFormat::FLOAT32;
    uint32_t channelOffset = 0, numChannelsEncode = 0;
    GridType gridType;
    InterpolationType interpolationType;
    HashType hashType;
    uint32_t numLevels;
    uint32_t numFeaturesPerLevel;
    uint32_t numFeatures;
    uint32_t log2HashMapSize;
    uint32_t baseResolution;
    float perLevelScale;
    struct UniformData {
        uint32_t baseResolution;
        float log2PerLevelScale;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;
    sgl::vk::BufferPtr& parametersBuffer;
    sgl::vk::BufferPtr offsetTableBuffer;
    Matrix matrixIn;
    sgl::vk::BufferPtr bufferOut;
    uint32_t batchSize = 0;
};

void GridEncodingPass::setBuffersInOut(const Matrix& _matrixIn, const sgl::vk::BufferPtr& _bufferOut) {
    matrixIn = _matrixIn;
    bufferOut = _bufferOut;
    dataDirty = true;
}

void GridEncodingPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void GridEncodingPass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void GridEncodingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("OFFSET_IN", std::to_string(channelOffset)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN", std::to_string(matrixIn.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_TO_ENCODE", std::to_string(numChannelsEncode)));
    preprocessorDefines.insert(std::make_pair("NUM_FEATURES_PER_LEVEL", std::to_string(numFeaturesPerLevel)));
    if (gridType == GridType::DENSE) {
        preprocessorDefines.insert(std::make_pair("GRID_TYPE_DENSE", ""));
    } else if (gridType == GridType::HASH) {
        preprocessorDefines.insert(std::make_pair("GRID_TYPE_HASH", ""));
    }
    if (interpolationType == InterpolationType::NEAREST) {
        preprocessorDefines.insert(std::make_pair("INTERPOLATION_TYPE_NEAREST", ""));
    } else if (interpolationType == InterpolationType::LINEAR) {
        preprocessorDefines.insert(std::make_pair("INTERPOLATION_TYPE_LINEAR", ""));
    }
    if (hashType == HashType::PRIME) {
        preprocessorDefines.insert(std::make_pair("PRIME_HASH", ""));
    } else if (hashType == HashType::COHERENT_PRIME) {
        preprocessorDefines.insert(std::make_pair("COHERENT_PRIME_HASH", ""));
    } else if (hashType == HashType::REVERSED_PRIME) {
        preprocessorDefines.insert(std::make_pair("REVERSED_PRIME_HASH", ""));
    }
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Encodings.Grid.Compute"}, preprocessorDefines);
}

void GridEncodingPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticBuffer(matrixIn.getBuffer(), "InputBuffer");
    computeData->setStaticBuffer(parametersBuffer, "GridBuffer");
    computeData->setStaticBuffer(offsetTableBuffer, "OffsetTableBuffer");
    computeData->setStaticBuffer(bufferOut, "OutputBuffer");
}

void GridEncodingPass::_render() {
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, batchSize);
    renderer->dispatch(computeData, sgl::uiceil(batchSize, BLOCK_SIZE), numLevels, 1);
}


class EncodedPositionsTransposePass : public sgl::vk::ComputePass {
public:
    explicit EncodedPositionsTransposePass(sgl::vk::Renderer* renderer, uint32_t numFeatures)
            : ComputePass(renderer), numFeatures(numFeatures) {
        uint32_t subgroupSize = device->getPhysicalDeviceSubgroupProperties().subgroupSize;
        BLOCK_SIZE = std::lcm(numFeatures, subgroupSize);
        while (BLOCK_SIZE < 128) {
            BLOCK_SIZE *= 2;
        }
    }

    void setBuffersInOut(const sgl::vk::BufferPtr& _bufferIn, const Matrix& _matrixOut);
    void setBatchSize(uint32_t _batchSize);
    void setFloatFormat(FloatFormat _format);
    void setCompositionMode(CompositionMode _compositionMode);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    uint32_t BLOCK_SIZE;
    FloatFormat format = FloatFormat::FLOAT32;
    CompositionMode compositionMode = CompositionMode::NONE;
    uint32_t numFeatures;
    sgl::vk::BufferPtr bufferIn;
    Matrix matrixOut;
    uint32_t batchSize = 0;
};

void EncodedPositionsTransposePass::setBuffersInOut(const sgl::vk::BufferPtr& _bufferIn, const Matrix& _matrixOut) {
    bufferIn = _bufferIn;
    matrixOut = _matrixOut;
    dataDirty = true;
}

void EncodedPositionsTransposePass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void EncodedPositionsTransposePass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void EncodedPositionsTransposePass::setCompositionMode(CompositionMode _compositionMode) {
    compositionMode = _compositionMode;
    shaderDirty = true;
}

void EncodedPositionsTransposePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("OFFSET_OUT", std::to_string(matrixOut.getOffsetChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT", std::to_string(matrixOut.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_FEATURES", std::to_string(numFeatures)));
    if (compositionMode == CompositionMode::SUM) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_SUM", ""));
    } else if (compositionMode == CompositionMode::PRODUCT) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_PRODUCT", ""));
    }
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Encodings.GridTranspose.Compute"}, preprocessorDefines);
}

void EncodedPositionsTransposePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(bufferIn, "InputBuffer");
    computeData->setStaticBuffer(matrixOut.getBuffer(), "OutputBuffer");
}

void EncodedPositionsTransposePass::_render() {
    uint32_t workSize = batchSize * numFeatures;
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, batchSize);
    renderer->dispatch(computeData, sgl::uiceil(workSize, BLOCK_SIZE), 1, 1);
}


inline float gridScale(uint32_t level, float log2PerLevelScale, uint32_t baseResolution) {
    return std::exp2f(float(level) * log2PerLevelScale) * float(baseResolution) - 1.0f;
}

GridEncoding::GridEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding)
        : Encoding(renderer, settingsEncoding), renderer(renderer), device(renderer->getDevice()) {
    numLevels = settingsEncoding.get("n_levels", 16u).asUInt();
    numFeaturesPerLevel = settingsEncoding.get("n_features_per_level", 2u).asUInt();
    numFeatures =  numLevels * numFeaturesPerLevel;
    log2HashMapSize = settingsEncoding.get("log2_hashmap_size", 19u).asUInt();
    baseResolution = settingsEncoding.get("base_resolution", 16u).asUInt();
    perLevelScale = settingsEncoding.get("per_level_scale", 2.0f).asFloat();

    std::string encodingType = settingsEncoding["otype"].asString();
    std::string defaultGridType = encodingType == "DenseGrid" ? "Dense" : "Hash";
    std::string gridTypeString = settingsEncoding.get("type", defaultGridType).asString();
    if (gridTypeString == "Dense") {
        gridType = GridType::DENSE;
    } else if (gridTypeString == "Hash") {
        gridType = GridType::HASH;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in GridEncoding::GridEncoding: Unsupported grid type \"" + gridTypeString + "\".");
    }
    std::string interpolationTypeString = settingsEncoding.get("interpolation", "Linear").asString();
    if (interpolationTypeString == "Nearest") {
        interpolationType = InterpolationType::NEAREST;
    } else if (interpolationTypeString == "Linear") {
        interpolationType = InterpolationType::LINEAR;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in GridEncoding::GridEncoding: Unsupported interpolation type \""
                + interpolationTypeString + "\".");
    }
    std::string hashTypeString = settingsEncoding.get("hash", "CoherentPrime").asString();
    if (hashTypeString == "CoherentPrime") {
        hashType = HashType::PRIME;
    } else if (hashTypeString == "CoherentPrime") {
        hashType = HashType::COHERENT_PRIME;
    } else if (hashTypeString == "ReversedPrime") {
        hashType = HashType::REVERSED_PRIME;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in GridEncoding::GridEncoding: Unsupported hash type \"" + hashTypeString + "\".");
    }

    // 64-bit integers only necessary for random hash (not supported by VMLP so far).
    //if (!device->getPhysicalDeviceFeatures().shaderInt64) {
    //    sgl::Logfile::get()->throwError("Error in GridEncoding::GridEncoding: shaderInt64 has not been enabled.");
    //}

    offsetTable.resize(numLevels + 1);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < numLevels; ++i) {
        const uint32_t resolution = uint32_t(std::ceil(gridScale(i, std::log2(perLevelScale), baseResolution))) + 1u;
        uint32_t maxParams = std::numeric_limits<uint32_t>::max() / 2;
        auto tmp = uint32_t(std::pow(resolution, numChannelsEncode));
        uint32_t paramsInLevel = nextMultiple(std::min(tmp, maxParams), 8u);
        if (gridType == GridType::HASH) {
            paramsInLevel = std::min(paramsInLevel, (1u << log2HashMapSize));
        }
        offsetTable.at(i) = offset;
        offset += paramsInLevel;
    }
    offsetTable.at(numLevels) = offset;
    numParameters = offsetTable.at(numLevels) * numFeaturesPerLevel;
    offsetTableBuffer = std::make_shared<sgl::vk::Buffer>(
            device, (numLevels + 1) * sizeof(uint32_t), offsetTable.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

    encodingPass = std::make_shared<GridEncodingPass>(
            renderer, channelOffset, numChannelsEncode, gridType, interpolationType, hashType,
            numLevels, numFeaturesPerLevel, numFeatures, log2HashMapSize, baseResolution, perLevelScale,
            parametersBuffer, offsetTableBuffer);
    encodedPositionsTransposePass = std::make_shared<EncodedPositionsTransposePass>(renderer, numFeatures);
}

void GridEncoding::setInputOutputMatrices(const Matrix& input, const Matrix& output) {
    auto batchSize = input.getBatchSize();
    if (cachedBatchSize != batchSize || floatFormatChanged) {
        cachedBatchSize = batchSize;
        floatFormatChanged = false;
        encodedPositionsBuffer = std::make_shared<sgl::vk::Buffer>(
                device, numFeatures * batchSize * FLOAT_FORMAT_SIZES_IN_BYTE[int(format)],
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    }

    encodingPass->setBuffersInOut(input, encodedPositionsBuffer);
    encodedPositionsTransposePass->setBuffersInOut(encodedPositionsBuffer, output);
}

void GridEncoding::setBatchSize(uint32_t _batchSize) {
    encodingPass->setBatchSize(_batchSize);
    encodedPositionsTransposePass->setBatchSize(_batchSize);
}

void GridEncoding::setFloatFormat(vmlp::FloatFormat _format) {
    if (!parametersBuffer || format != _format) {
        parametersBuffer = std::make_shared<sgl::vk::Buffer>(
                device, numParameters * FLOAT_FORMAT_SIZES_IN_BYTE[int(_format)],
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadParametersToBuffer(parametersCpu, getNumParameters(), parametersBuffer, _format);
        floatFormatChanged = true;
    }
    format = _format;
    encodingPass->setFloatFormat(_format);
    encodedPositionsTransposePass->setFloatFormat(_format);
}

void GridEncoding::setCompositionMode(CompositionMode _compositionMode) {
    encodedPositionsTransposePass->setCompositionMode(_compositionMode);
}

void GridEncoding::runInference() {
    //debugPrintBuffer(encodedPositionsBuffer, format, 48);
    encodingPass->render();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            encodedPositionsBuffer);
    encodedPositionsTransposePass->render();
    //debugPrintBuffer(encodedPositionsBuffer, format, 48);
}


class DictionaryEncodingPass : public sgl::vk::ComputePass {
public:
    explicit DictionaryEncodingPass(
            sgl::vk::Renderer* renderer, uint32_t channelOffset, uint32_t numChannelsEncode,
            uint32_t numEmbeddings, uint32_t numFeatures, sgl::vk::BufferPtr& parametersBuffer)
            : ComputePass(renderer), channelOffset(channelOffset), numChannelsEncode(numChannelsEncode),
              numEmbeddings(numEmbeddings), numFeatures(numFeatures), parametersBuffer(parametersBuffer) {}

    void setBuffersInOut(const Matrix& _matrixIn, const Matrix& _matrixOut);
    void setBatchSize(uint32_t _batchSize);
    void setFloatFormat(FloatFormat _format);
    void setCompositionMode(CompositionMode _compositionMode);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t BLOCK_SIZE = 256;
    FloatFormat format = FloatFormat::FLOAT32;
    CompositionMode compositionMode = CompositionMode::NONE;
    uint32_t channelOffset = 0, numChannelsEncode = 0, numEmbeddings = 0, numFeatures = 0;
    Matrix matrixIn, matrixOut;
    uint32_t batchSize = 0;
    sgl::vk::BufferPtr& parametersBuffer;
};

void DictionaryEncodingPass::setBuffersInOut(const Matrix& _matrixIn, const Matrix& _matrixOut) {
    matrixIn = _matrixIn;
    matrixOut = _matrixOut;
    dataDirty = true;
}

void DictionaryEncodingPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void DictionaryEncodingPass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void DictionaryEncodingPass::setCompositionMode(CompositionMode _compositionMode) {
    compositionMode = _compositionMode;
    shaderDirty = true;
}

void DictionaryEncodingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    preprocessorDefines.insert(std::make_pair("OFFSET_IN", std::to_string(channelOffset)));
    preprocessorDefines.insert(std::make_pair("OFFSET_OUT", std::to_string(matrixOut.getOffsetChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN", std::to_string(matrixIn.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT", std::to_string(matrixOut.getNumChannels())));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_TO_ENCODE", std::to_string(numChannelsEncode)));
    preprocessorDefines.insert(std::make_pair("NUM_EMBEDDINGS", std::to_string(numEmbeddings)));
    preprocessorDefines.insert(std::make_pair("NUM_FEATURES", std::to_string(numFeatures)));
    if (compositionMode == CompositionMode::SUM) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_SUM", ""));
    } else if (compositionMode == CompositionMode::PRODUCT) {
        preprocessorDefines.insert(std::make_pair("OUTPUT_OP_PRODUCT", ""));
    }
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Encodings.Dictionary.Compute"}, preprocessorDefines);
}

void DictionaryEncodingPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(matrixIn.getBuffer(), "InputBuffer");
    computeData->setStaticBuffer(matrixOut.getBuffer(), "OutputBuffer");
    computeData->setStaticBuffer(parametersBuffer, "ParametersBuffer");
}

void DictionaryEncodingPass::_render() {
    const uint32_t numThreads = numChannelsEncode * numFeatures * batchSize;
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numThreads);
    renderer->dispatch(computeData, sgl::uiceil(numThreads, BLOCK_SIZE), 1, 1);
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            matrixOut.getBuffer());
}

DictionaryEncoding::DictionaryEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding)
        : Encoding(renderer, settingsEncoding), renderer(renderer), device(renderer->getDevice()) {
    numEmbeddings = settingsEncoding.get("num_embeddings", 1u).asUInt();
    numFeatures = settingsEncoding.get("embedding_dim", 1u).asUInt();

    dictionaryEncodingPass = std::make_shared<DictionaryEncodingPass>(
            renderer, channelOffset, numChannelsEncode, numEmbeddings, numFeatures, parametersBuffer);
}

void DictionaryEncoding::setInputOutputMatrices(const Matrix& input, const Matrix& output) {
    dictionaryEncodingPass->setBuffersInOut(input, output);
}

void DictionaryEncoding::setBatchSize(uint32_t _batchSize) {
    dictionaryEncodingPass->setBatchSize(_batchSize);
}

void DictionaryEncoding::setFloatFormat(vmlp::FloatFormat _format) {
    if (!parametersBuffer || format != _format) {
        parametersBuffer = std::make_shared<sgl::vk::Buffer>(
                device, numEmbeddings * numFeatures * FLOAT_FORMAT_SIZES_IN_BYTE[int(_format)],
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadParametersToBuffer(parametersCpu, getNumParameters(), parametersBuffer, _format);
    }
    format = _format;
    dictionaryEncodingPass->setFloatFormat(_format);
}

void DictionaryEncoding::setCompositionMode(CompositionMode _compositionMode) {
    dictionaryEncodingPass->setCompositionMode(_compositionMode);
}

void DictionaryEncoding::runInference() {
    dictionaryEncodingPass->render();
}

}
