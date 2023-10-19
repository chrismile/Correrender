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
#include <chrono>
#include <utility>
#include <json/json.h>

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Loaders/half/half.h"
#include "Encoding.hpp"
#include "Network.hpp"

namespace vmlp {

void uploadParametersToBuffer(
        float* parameters, uint32_t numParameters,
        const sgl::vk::BufferPtr& parametersBuffer, FloatFormat format) {
    void* data = nullptr;
    if (format == FloatFormat::FLOAT32) {
        data = parameters;
    } else if (format == FloatFormat::FLOAT16) {
        auto* dataHalf = new FLOAT16[numParameters];
        for (uint32_t i = 0; i < numParameters; i++) {
            dataHalf[i] = FLOAT16::ToFloat16(parameters[i]);
        }
        data = dataHalf;
    }
    parametersBuffer->uploadData(numParameters * FLOAT_FORMAT_SIZES_IN_BYTE[int(format)], data);
    if (format == FloatFormat::FLOAT16) {
        delete[] reinterpret_cast<FLOAT16*>(data);
    }
}

uint32_t nextMultiple(uint32_t num, uint32_t denom) {
    auto x = num % denom;
    if (x == 0) {
        return num;
    }
    return num + denom - x;
}

void addPreprocessorDefinesFormat(std::map<std::string, std::string>& preprocessorDefines, FloatFormat format) {
    if (format == FloatFormat::FLOAT16) {
        /*
         * https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt
         * https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_16bit_storage.txt
         */
        preprocessorDefines.insert(std::make_pair(
                "__extensions", "GL_EXT_shader_explicit_arithmetic_types_float16;GL_EXT_shader_16bit_storage"));
    }
    preprocessorDefines.insert(std::make_pair("real", FLOAT_FORMAT_GLSL_NAMES[int(format)]));
    preprocessorDefines.insert(std::make_pair("real2", FLOAT2_FORMAT_GLSL_NAMES[int(format)]));
    preprocessorDefines.insert(std::make_pair("real3", FLOAT3_FORMAT_GLSL_NAMES[int(format)]));
    preprocessorDefines.insert(std::make_pair("real4", FLOAT4_FORMAT_GLSL_NAMES[int(format)]));
}

ActivationFunction activationFunctionFromString(const std::string& activationName) {
    ActivationFunction activationFunction = ActivationFunction::NONE;
    const int num = VMLP_ARRAYSIZE(ACTIVATION_FUNCTION_NAMES);
    int i = 0;
    for (; i < num; i++) {
        if (activationName == ACTIVATION_FUNCTION_NAMES[i]) {
            activationFunction = ActivationFunction(i);
            break;
        }
    }
    if (i == num) {
        sgl::Logfile::get()->throwError(
                "Error in activationFunctionFromString: Unknown activation function name \"" + activationName + "\".");
    }
    return activationFunction;
}

// TODO: Remove after finishing debugging.
static sgl::vk::Renderer* networkRenderer = nullptr;

void debugPrintBuffer(const sgl::vk::BufferPtr& deviceBuffer, FloatFormat format, size_t numEntries) {
    networkRenderer->syncWithCpu();
    auto* device = networkRenderer->getDevice();
    auto commandBuffer = device->beginSingleTimeCommands();
    auto stagingBuffer = std::make_shared<sgl::vk::Buffer>(
            device, deviceBuffer->getSizeInBytes(), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    deviceBuffer->copyDataTo(stagingBuffer, commandBuffer);
    device->endSingleTimeCommands(commandBuffer);
    if (format == FloatFormat::FLOAT32) {
        auto* data = reinterpret_cast<float*>(stagingBuffer->mapMemory());
        for (size_t i = 0; i < numEntries; i++) {
            std::cout << data[i] << std::endl;
        }
        std::cout << std::endl;
        stagingBuffer->unmapMemory();
    } else {
        auto* data = reinterpret_cast<FLOAT16*>(stagingBuffer->mapMemory());
        for (size_t i = 0; i < numEntries; i++) {
            std::cout << FLOAT16::ToFloat32(data[i]) << std::endl;
        }
        std::cout << std::endl;
        stagingBuffer->unmapMemory();
    }
}

std::shared_ptr<Module> createNetwork(sgl::vk::Renderer* renderer, const Json::Value& settingsNetwork) {
    networkRenderer = renderer;
    return std::shared_ptr<Module>(new MlpNetwork(renderer, settingsNetwork));
}

std::shared_ptr<Module> createInputEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding) {
    std::string encodingType = settingsEncoding["otype"].asString();
    Module* module = nullptr;
    if (encodingType == "Composite") {
        module = new CompositeEncoding(renderer, settingsEncoding);
    } else if (encodingType == "Identity") {
        module = new IdentityEncoding(renderer, settingsEncoding);
    } else if (encodingType == "Frequency") {
        module = new FrequencyEncoding(renderer, settingsEncoding);
    } else if (encodingType == "Grid" || encodingType == "HashGrid" || encodingType == "DenseGrid") {
        module = new GridEncoding(renderer, settingsEncoding);
    } else {
        sgl::Logfile::get()->throwError(
                "Error in createInputEncoding: Unsupported encoding type \"" + encodingType + "\".");
    }
    return std::shared_ptr<Module>(module);
}

std::shared_ptr<Module> createNetworkWithInputEncoding(
        sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding, const Json::Value& settingsNetwork) {
    return std::shared_ptr<Module>(new NetworkWithInputEncoding(renderer, settingsEncoding, settingsNetwork));
}

void Module::setParametersCpu(float* parameters, uint32_t numParameters) {
    if (numParameters != getNumParameters()) {
        sgl::Logfile::get()->throwError("Error in MlpNetwork::setParametersCpu: Mismatch in number of parameters.");
    }
    if (numParameters > 0) {
        if (!parametersCpu) {
            parametersCpu = new float[numParameters];
        }
        memcpy(parametersCpu, parameters, sizeof(float) * numParameters);
    }
}

NetworkWithInputEncoding::NetworkWithInputEncoding(
        sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding, Json::Value settingsNetwork)
        : device(renderer->getDevice()) {
    encoding = createInputEncoding(renderer, settingsEncoding);
    settingsNetwork["n_input_dims"] = encoding->getNumChannelsOut();
    network = createNetwork(renderer, settingsNetwork);
}

void NetworkWithInputEncoding::runInference() {
    encoding->runInference();
    network->runInference();

    //debugPrintBuffer(intermediateOutput.getBuffer(), format, 48);

    /*networkRenderer->syncWithCpu();
    auto beginEncoding = std::chrono::system_clock::now();
    encoding->runInference();
    networkRenderer->syncWithCpu();
    auto endEncoding = std::chrono::system_clock::now();
    auto elapsedEncoding = std::chrono::duration_cast<std::chrono::microseconds>(endEncoding - beginEncoding);
    std::cout << "Elapsed time encoding: " << (elapsedEncoding.count() * 1e-3) << "ms" << std::endl;

    auto beginNetwork = std::chrono::system_clock::now();
    network->runInference();
    networkRenderer->syncWithCpu();
    auto endNetwork = std::chrono::system_clock::now();
    auto elapsedNetwork = std::chrono::duration_cast<std::chrono::microseconds>(endNetwork - beginNetwork);
    std::cout << "Elapsed time network: " << (elapsedNetwork.count() * 1e-3) << "ms" << std::endl;*/
}

class MlpPass : public sgl::vk::ComputePass {
public:
    MlpPass(
            sgl::vk::Renderer* renderer, uint32_t layerChannelsIn, uint32_t layerChannelsOut,
            ActivationFunction activationFunction,
            uint32_t parametersBufferOffset, sgl::vk::BufferPtr& parametersBuffer)
            : ComputePass(renderer), layerChannelsIn(layerChannelsIn), layerChannelsOut(layerChannelsOut),
              activationFunction(activationFunction),
              parametersBufferOffset(parametersBufferOffset), parametersBuffer(parametersBuffer) {}

    void setBuffersInOut(const sgl::vk::BufferPtr& _bufferIn, const sgl::vk::BufferPtr& _bufferOut);
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
    uint32_t layerChannelsIn, layerChannelsOut;
    ActivationFunction activationFunction;
    uint32_t parametersBufferOffset;
    sgl::vk::BufferPtr& parametersBuffer;
    sgl::vk::BufferPtr bufferIn, bufferOut;
    uint32_t batchSize = 0;
};

void MlpPass::setBuffersInOut(const sgl::vk::BufferPtr& _bufferIn, const sgl::vk::BufferPtr& _bufferOut) {
    bufferIn = _bufferIn;
    bufferOut = _bufferOut;
    dataDirty = true;
}

void MlpPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void MlpPass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void MlpPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    auto numChannelsInPadded = nextMultiple(layerChannelsIn, 16);
    auto numChannelsOutPadded = nextMultiple(layerChannelsOut, 16);
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN", std::to_string(layerChannelsIn)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT", std::to_string(layerChannelsOut)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN_PADDED", std::to_string(numChannelsInPadded)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT_PADDED", std::to_string(numChannelsOutPadded)));
    preprocessorDefines.insert(std::make_pair("WEIGHT_OFFSET", std::to_string(parametersBufferOffset)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    if (activationFunction != ActivationFunction::NONE) {
        preprocessorDefines.insert(std::make_pair(
                "ACTIVATION_FUNCTION", ACTIVATION_FUNCTION_NAMES[int(activationFunction)]));
    }
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Network.GlobalMemory.Compute"}, preprocessorDefines);
}

void MlpPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(parametersBuffer, "ParametersBuffer");
    computeData->setStaticBuffer(bufferIn, "InputBuffer");
    computeData->setStaticBuffer(bufferOut, "OutputBuffer");
}

void MlpPass::_render() {
    const uint32_t numThreads = layerChannelsOut * batchSize;
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numThreads);
    renderer->dispatch(computeData, sgl::uiceil(numThreads, BLOCK_SIZE), 1, 1);
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            bufferOut);
}


MlpNetwork::MlpNetwork(sgl::vk::Renderer* renderer, const Json::Value& settingsNetwork)
        : device(renderer->getDevice()) {
    numLayers = settingsNetwork["n_hidden_layers"].asUInt() + 1;
    numChannelsIn = settingsNetwork["n_input_dims"].asUInt();
    numChannelsHidden = settingsNetwork["n_neurons"].asUInt();
    numChannelsOut = settingsNetwork["n_output_dims"].asUInt();
    activationFunction = activationFunctionFromString(settingsNetwork["activation"].asString());
    outputActivationFunction = activationFunctionFromString(settingsNetwork["output_activation"].asString());

    numChannelsInPadded = nextMultiple(numChannelsIn, 16);
    numChannelsOutPadded = nextMultiple(numChannelsOut, 16);

    if (numLayers == 1) {
        numParameters = numChannelsInPadded * numChannelsOutPadded;
    } else {
        numParameters =
                numChannelsInPadded * numChannelsHidden
                + numChannelsHidden * numChannelsHidden * (numLayers - 2)
                + numChannelsHidden * numChannelsOutPadded;
    }

    uint32_t parametersBufferOffset = 0;
    layerPasses.reserve(numLayers);
    for (size_t i = 0; i < numLayers; i++) {
        uint32_t layerChannelsIn, layerChannelsOut;
        ActivationFunction layerActivationFunction;
        if (i == 0) {
            layerChannelsIn = numChannelsIn;
        } else {
            layerChannelsIn = numChannelsHidden;
        }
        if (i == numLayers - 1) {
            layerChannelsOut = numChannelsOut;
            layerActivationFunction = outputActivationFunction;
        } else {
            layerChannelsOut = numChannelsHidden;
            layerActivationFunction = activationFunction;
        }
        auto layerPass = std::make_shared<MlpPass>(
                renderer, layerChannelsIn, layerChannelsOut, layerActivationFunction,
                parametersBufferOffset, parametersBuffer);
        layerPasses.push_back(layerPass);
        parametersBufferOffset += layerChannelsIn * layerChannelsOut;
    }
}

void MlpNetwork::setInputOutputMatrices(const Matrix& input, const Matrix& output) {
    auto batchSize = input.getBatchSize();
    if (numLayers > 1 && (cachedBatchSize != batchSize || floatFormatChanged)) {
        cachedBatchSize = batchSize;
        floatFormatChanged = false;
        for (int i = 0; i < 2; i++) {
            // Add "VK_BUFFER_USAGE_TRANSFER_SRC_BIT" for debugging purposes.
            intermediateOutputBuffers[i] = std::make_shared<sgl::vk::Buffer>(
                    device, numChannelsHidden * batchSize * FLOAT_FORMAT_SIZES_IN_BYTE[int(format)],
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY); // TODO
        }
    }

    for (uint32_t i = 0; i < numLayers; i++) {
        auto& layerPass = layerPasses.at(i);
        sgl::vk::BufferPtr bufferIn, bufferOut;
        if (i == 0) {
            bufferIn = input.getBuffer();
        } else {
            bufferIn = intermediateOutputBuffers[i % 2];
        }
        if (i == numLayers - 1) {
            bufferOut = output.getBuffer();
        } else {
            bufferOut = intermediateOutputBuffers[(i + 1) % 2];
        }
        layerPass->setBuffersInOut(bufferIn, bufferOut);
    }
}

void MlpNetwork::setBatchSize(uint32_t _batchSize) {
    for (size_t i = 0; i < numLayers; i++) {
        auto& layerPass = layerPasses.at(i);
        layerPass->setBatchSize(_batchSize);
    }
}

void MlpNetwork::setFloatFormat(vmlp::FloatFormat _format) {
    if (!parametersBuffer || format != _format) {
        parametersBuffer = std::make_shared<sgl::vk::Buffer>(
                device, numParameters * FLOAT_FORMAT_SIZES_IN_BYTE[int(_format)],
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        uploadParametersToBuffer(parametersCpu, getNumParameters(), parametersBuffer, _format);
        floatFormatChanged = true;
    }
    format = _format;
    for (size_t i = 0; i < numLayers; i++) {
        auto& layerPass = layerPasses.at(i);
        layerPass->setFloatFormat(_format);
    }
}

void MlpNetwork::runInference() {
    for (size_t i = 0; i < numLayers; i++) {
        auto& layerPass = layerPasses.at(i);
        layerPass->render();
    }
}

}
