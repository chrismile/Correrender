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
#include <Math/half/half.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

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
        auto* dataHalf = new HalfFloat[numParameters];
        for (uint32_t i = 0; i < numParameters; i++) {
            dataHalf[i] = HalfFloat(parameters[i]);
        }
        data = dataHalf;
    }
    parametersBuffer->uploadData(numParameters * FLOAT_FORMAT_SIZES_IN_BYTE[int(format)], data);
    if (format == FloatFormat::FLOAT16) {
        delete[] reinterpret_cast<HalfFloat*>(data);
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
        auto* data = reinterpret_cast<HalfFloat*>(stagingBuffer->mapMemory());
        std::cout << std::endl;
        for (size_t i = 0; i < numEntries; i++) {
            std::cout << float(data[i]) << std::endl;
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
    } else if (encodingType == "Dictionary") {
        module = new DictionaryEncoding(renderer, settingsEncoding);
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

void NetworkWithInputEncoding::setInputOutputMatrices(const Matrix& input, const Matrix& output) {
    if (input.getBatchSize() != output.getBatchSize()) {
        sgl::Logfile::get()->throwError(
                "Error in NetworkWithInputEncoding::setInputOutputMatrices: "
                "Mismatch in input and output batch size.");
    }

    auto batchSize = input.getBatchSize();
    if (batchSize != intermediateOutput.getBatchSize() || floatFormatChanged) {
        floatFormatChanged = false;
        auto channelsEncodingOut = encoding->getNumChannelsOut();
        // TODO: Remove VK_BUFFER_USAGE_TRANSFER_SRC_BIT.
        auto intermediateBuffer = std::make_shared<sgl::vk::Buffer>(
                device, channelsEncodingOut * batchSize * FLOAT_FORMAT_SIZES_IN_BYTE[int(format)],
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        auto commandBuffer = device->beginSingleTimeCommands();
        intermediateBuffer->fill(0, commandBuffer);
        device->endSingleTimeCommands(commandBuffer);
        intermediateOutput = Matrix(
                intermediateBuffer, FLOAT_FORMAT_SIZES_IN_BYTE[int(format)], channelsEncodingOut, batchSize);
    }

    encoding->setInputOutputMatrices(input, intermediateOutput);
    network->setInputOutputMatrices(intermediateOutput, output);
}

void NetworkWithInputEncoding::runInference() {
    encoding->runInference();
    network->runInference();

    //debugPrintBuffer(intermediateOutput.getBuffer(), format, 48);
    //std::cout << std::endl;

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
            sgl::vk::Renderer* renderer, bool mlpUseSharedMemory, uint32_t layerChannelsIn, uint32_t layerChannelsOut,
            ActivationFunction activationFunction,
            uint32_t parametersBufferOffset, sgl::vk::BufferPtr& parametersBuffer)
            : ComputePass(renderer), mlpUseSharedMemory(mlpUseSharedMemory),
              layerChannelsIn(layerChannelsIn), layerChannelsOut(layerChannelsOut),
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
    bool mlpUseSharedMemory = true;
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

    std::string shaderName;
    if (mlpUseSharedMemory) {
        shaderName = "Network.SharedMemory.Compute";
    } else {
        shaderName = "Network.GlobalMemory.Compute";
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void MlpPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(parametersBuffer, "ParametersBuffer");
    computeData->setStaticBuffer(bufferIn, "InputBuffer");
    computeData->setStaticBuffer(bufferOut, "OutputBuffer");
}

void MlpPass::_render() {
    if (mlpUseSharedMemory) {
        auto numChannelsOutPadded = nextMultiple(layerChannelsOut, 16);
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, batchSize);
        renderer->dispatch(computeData, sgl::uiceil(numChannelsOutPadded, 16u), sgl::uiceil(batchSize, 16u), 1);
    } else {
        const uint32_t numThreads = layerChannelsOut * batchSize;
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numThreads);
        renderer->dispatch(computeData, sgl::uiceil(numThreads, BLOCK_SIZE), 1, 1);
    }
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            bufferOut);
}


class MlpFusedPass : public sgl::vk::ComputePass {
public:
    MlpFusedPass(
            sgl::vk::Renderer* renderer, bool useKHR, uint32_t _M, uint32_t _K, uint32_t _N, uint32_t subgroupSize,
            FusedMlpMemoryType memoryType, bool fusedMlpDirectLoad, bool useSharedMemoryBankSkew,
            uint32_t numLayers, uint32_t numChannelsIn, uint32_t numChannelsOut, uint32_t numChannelsHidden,
            ActivationFunction activationFunction, ActivationFunction outputActivationFunction,
            sgl::vk::BufferPtr& parametersBuffer);

    void setBuffersInOut(const sgl::vk::BufferPtr& _bufferIn, const sgl::vk::BufferPtr& _bufferOut);
    void setBatchSize(uint32_t _batchSize);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    bool useKHR; // Whether to use KHR_cooperative_matrix or NV_cooperative_matrix.
    /**
     * Matrix block sizes for operation: R^{M x N} = R^{M x K} * R^{K x N} + R^{M x N}
     */
    uint32_t M = 16, K = 16, N = 16;
    uint32_t subgroupSize = 32;
    FusedMlpMemoryType memoryType;
    bool fusedMlpDirectLoad;
    bool useSharedMemoryBankSkew;
    uint32_t bankSkew;
    uint32_t numBatches = 8; // Number of batch blocks (determined by shared memory size).
    uint32_t numRows;
    uint32_t sharedMemorySize;
    uint32_t numLayers;
    uint32_t numChannelsIn, numChannelsOut, numChannelsHidden;
    ActivationFunction activationFunction, outputActivationFunction;
    sgl::vk::BufferPtr& parametersBuffer;
    sgl::vk::BufferPtr bufferIn, bufferOut;
    uint32_t batchSize = 0;
};

MlpFusedPass::MlpFusedPass(
        sgl::vk::Renderer* renderer, bool useKHR, uint32_t _M, uint32_t _K, uint32_t _N, uint32_t subgroupSize,
        FusedMlpMemoryType memoryType, bool fusedMlpDirectLoad, bool useSharedMemoryBankSkew,
        uint32_t numLayers, uint32_t numChannelsIn, uint32_t numChannelsOut, uint32_t numChannelsHidden,
        ActivationFunction activationFunction, ActivationFunction outputActivationFunction,
        sgl::vk::BufferPtr& parametersBuffer)
        : ComputePass(renderer), useKHR(useKHR), M(_M), K(_K), N(_N), subgroupSize(subgroupSize),
          memoryType(memoryType), fusedMlpDirectLoad(fusedMlpDirectLoad),
          useSharedMemoryBankSkew(useSharedMemoryBankSkew),
          numLayers(numLayers), numChannelsIn(numChannelsIn), numChannelsOut(numChannelsOut),
          numChannelsHidden(numChannelsHidden),
          activationFunction(activationFunction), outputActivationFunction(outputActivationFunction),
          parametersBuffer(parametersBuffer) {
    auto* device = renderer->getDevice();
    if (numChannelsHidden < numChannelsIn || numChannelsHidden < numChannelsOut) {
        sgl::Logfile::get()->throwError(
                "Error in MlpFusedPass::MlpFusedPass: Number of channels in the hidden layers must be greater or equal "
                "to the number of input and output channels.");
    }
    if (activationFunction != outputActivationFunction && outputActivationFunction != ActivationFunction::NONE) {
        sgl::Logfile::get()->throwError(
                "Error in MlpFusedPass::MlpFusedPass: Output activation function may only be 'None' or the regular "
                "activation function used by all other layers.");
    }
    if (numChannelsHidden % M != 0 || numChannelsHidden % K != 0 || numChannelsHidden % N != 0) {
        sgl::Logfile::get()->throwError(
                "Error in MlpFusedPass::MlpFusedPass: Number of hidden layer channels must be divisible by the "
                "matrix block size.");
    }

    bankSkew = useSharedMemoryBankSkew ? 8 : 0;
    uint32_t numChannelsHiddenSkewed = numChannelsHidden + bankSkew;
    numRows = numChannelsHidden / M;
    uint32_t maxSharedMemSize = device->getLimits().maxComputeSharedMemorySize;
    numBatches = maxSharedMemSize / (sizeof(HalfFloat) * numChannelsHiddenSkewed * N);
    // Do we even need to use this? Non-power-of-two should work just as well.
    //if (!sgl::isPowerOfTwo(int(nBatch))) {
    //    nBatch = uint32_t(sgl::lastPowerOfTwo(int(nBatch)));
    //}
    if (numBatches < 1u) {
        sgl::Logfile::get()->throwError(
                "Error in MlpFusedPass::MlpFusedPass: Insufficient amount of shared memory ("
                + std::to_string(maxSharedMemSize) + ").");
    }
    numBatches = std::clamp(numBatches, 1u, 8u);
    sharedMemorySize = numBatches * numChannelsHiddenSkewed * N * sizeof(HalfFloat);
}

void MlpFusedPass::setBuffersInOut(const sgl::vk::BufferPtr& _bufferIn, const sgl::vk::BufferPtr& _bufferOut) {
    bufferIn = _bufferIn;
    bufferOut = _bufferOut;
    dataDirty = true;
}

void MlpFusedPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void MlpFusedPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;

    auto numChannelsInPadded = nextMultiple(numChannelsIn, 16);
    auto numChannelsOutPadded = nextMultiple(numChannelsOut, 16);
    preprocessorDefines.insert(std::make_pair("NUM_LAYERS", std::to_string(numLayers)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN", std::to_string(numChannelsIn)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT", std::to_string(numChannelsOut)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_IN_PADDED", std::to_string(numChannelsInPadded)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_OUT_PADDED", std::to_string(numChannelsOutPadded)));
    preprocessorDefines.insert(std::make_pair("NUM_CHANNELS_HIDDEN", std::to_string(numChannelsHidden)));

    preprocessorDefines.insert(std::make_pair("M", std::to_string(M)));
    preprocessorDefines.insert(std::make_pair("K", std::to_string(K)));
    preprocessorDefines.insert(std::make_pair("N", std::to_string(N)));
    preprocessorDefines.insert(std::make_pair("SUBGROUP_SIZE", std::to_string(subgroupSize)));
    preprocessorDefines.insert(std::make_pair("NUM_ROWS", std::to_string(numRows)));
    preprocessorDefines.insert(std::make_pair("NUM_BATCHES", std::to_string(numBatches)));
    preprocessorDefines.insert(std::make_pair(
            "SHARED_MEMORY_SIZE", std::to_string(sharedMemorySize / sizeof(HalfFloat))));
    if (useSharedMemoryBankSkew) {
        preprocessorDefines.insert(std::make_pair("BANK_SKEW", std::to_string(bankSkew)));
    }

    preprocessorDefines.insert(std::make_pair(
            "ACTIVATION_FUNCTION", ACTIVATION_FUNCTION_NAMES[int(activationFunction)]));
    if (outputActivationFunction == ActivationFunction::NONE) {
        preprocessorDefines.insert(std::make_pair("NO_OUTPUT_ACTIVATION", ""));
    }

    if (memoryType == FusedMlpMemoryType::FLOAT16_NO_PADDING) {
        preprocessorDefines.insert(std::make_pair(
                "FLOAT16_NO_PADDING", std::to_string(sharedMemorySize / sizeof(HalfFloat))));
    } else if (fusedMlpDirectLoad) {
        std::string headerPath = sgl::vk::ShaderManager->getShaderFileName("NetworkFusedLayer.glsl");
        std::string prependContent;
        std::string headerString = sgl::vk::ShaderManager->loadHeaderFileString(headerPath, prependContent);
        std::string layerCode = "uint layerIdx;\n";
        for (uint32_t layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            if (layerIdx != 0) {
                layerCode += "#define LAYER_USE_SHARED_MEMORY_INPUT\n";
            }
            if (layerIdx != numLayers - 1) {
                layerCode += "#define LAYER_USE_SHARED_MEMORY_OUTPUT\n";
            }
            layerCode += "layerIdx = " + std::to_string(layerIdx) + ";\n";
            layerCode += headerString;
            if (layerIdx != 0) {
                layerCode += "#undef LAYER_USE_SHARED_MEMORY_INPUT\n";
            }
            if (layerIdx != numLayers - 1) {
                layerCode += "#undef LAYER_USE_SHARED_MEMORY_OUTPUT\n";
            }
        }
        preprocessorDefines.insert(std::make_pair("USE_CUSTOM_LAYERS_CODE", ""));
        preprocessorDefines.insert(std::make_pair("CUSTOM_LAYERS_CODE", layerCode));
    }
    std::string storageType;
    int sharedMemoryFactor = 0;
    if (memoryType == FusedMlpMemoryType::FLOAT16_NO_PADDING || memoryType == FusedMlpMemoryType::FLOAT16_PADDING) {
        storageType = "float16_t";
        sharedMemoryFactor = 1;
    } else if (memoryType == FusedMlpMemoryType::UINT) {
        storageType = "uint";
        sharedMemoryFactor = 2;
    } else if (memoryType == FusedMlpMemoryType::UVEC2) {
        storageType = "uvec2";
        sharedMemoryFactor = 4;
    } else if (memoryType == FusedMlpMemoryType::UVEC4) {
        storageType = "uvec4";
        sharedMemoryFactor = 8;
    }
    preprocessorDefines.insert(std::make_pair("STORAGE_TYPE", storageType));
    preprocessorDefines.insert(std::make_pair("SMEM_FACTOR", std::to_string(sharedMemoryFactor)));

    if (useKHR) {
        preprocessorDefines.insert(std::make_pair("__extensions", "GL_KHR_cooperative_matrix"));
        preprocessorDefines.insert(std::make_pair("CoopMatA", "coopmat<float16_t, gl_ScopeSubgroup, M, K, gl_MatrixUseA>"));
        preprocessorDefines.insert(std::make_pair("CoopMatB", "coopmat<float16_t, gl_ScopeSubgroup, K, N, gl_MatrixUseB>"));
        preprocessorDefines.insert(std::make_pair("CoopMatAcc", "coopmat<float16_t, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>"));
        preprocessorDefines.insert(std::make_pair("matLoad", "coopMatLoad"));
        preprocessorDefines.insert(std::make_pair("matStore", "coopMatStore"));
        preprocessorDefines.insert(std::make_pair("matMulAdd", "coopMatMulAdd"));
        preprocessorDefines.insert(std::make_pair("ROW_MAJOR", "gl_CooperativeMatrixLayoutRowMajor"));
        preprocessorDefines.insert(std::make_pair("COL_MAJOR", "gl_CooperativeMatrixLayoutColumnMajor"));
    } else {
        preprocessorDefines.insert(std::make_pair("__extensions", "GL_NV_cooperative_matrix"));
        preprocessorDefines.insert(std::make_pair("CoopMatA", "fcoopmatNV<16, gl_ScopeSubgroup, M, K>"));
        preprocessorDefines.insert(std::make_pair("CoopMatB", "fcoopmatNV<16, gl_ScopeSubgroup, N, N>"));
        preprocessorDefines.insert(std::make_pair("CoopMatAcc", "fcoopmatNV<16, gl_ScopeSubgroup, M, N>"));
        preprocessorDefines.insert(std::make_pair("matLoad", "coopMatLoadNV"));
        preprocessorDefines.insert(std::make_pair("matStore", "coopMatStoreNV"));
        preprocessorDefines.insert(std::make_pair("matMulAdd", "coopMatMulAddNV"));
        preprocessorDefines.insert(std::make_pair("ROW_MAJOR", "false"));
        preprocessorDefines.insert(std::make_pair("COL_MAJOR", "true"));
    }

    // Use required subgroup size if the requested size not equal to the default size.
    uint32_t requiredSubgroupSize = 0;
    if (subgroupSize != device->getPhysicalDeviceSubgroupProperties().subgroupSize) {
        requiredSubgroupSize = subgroupSize;
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            {"NetworkFused.Compute"}, preprocessorDefines, requiredSubgroupSize);
}

void MlpFusedPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(parametersBuffer, "ParametersBuffer");
    computeData->setStaticBuffer(bufferIn, "InputBuffer");
    computeData->setStaticBuffer(bufferOut, "OutputBuffer");
}

void MlpFusedPass::_render() {
    uint32_t typeSize = 0;
    if (memoryType == FusedMlpMemoryType::FLOAT16_NO_PADDING || memoryType == FusedMlpMemoryType::FLOAT16_PADDING) {
        typeSize = 2;
    } else if (memoryType == FusedMlpMemoryType::UINT) {
        typeSize = 4;
    } else if (memoryType == FusedMlpMemoryType::UVEC2) {
        typeSize = 8;
    } else if (memoryType == FusedMlpMemoryType::UVEC4) {
        typeSize = 16;
    }
    glm::uvec3 inputOutputBufferSizesTyped;
    inputOutputBufferSizesTyped.x = batchSize;
    inputOutputBufferSizesTyped.y = sgl::uiceil(batchSize * nextMultiple(numChannelsIn, 16), typeSize);
    inputOutputBufferSizesTyped.z = sgl::uiceil(batchSize * nextMultiple(numChannelsOut, 16), typeSize);
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, inputOutputBufferSizesTyped);
    //renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, batchSize);
    renderer->dispatch(computeData, sgl::uiceil(batchSize, N * numBatches), 1, 1);
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            bufferOut);
}


MlpNetwork::MlpNetwork(sgl::vk::Renderer* renderer, const Json::Value& settingsNetwork)
        : renderer(renderer), device(renderer->getDevice()) {
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

    recreateMlpPass();

    subgroupSize = device->getPhysicalDeviceSubgroupProperties().subgroupSize;
}

void MlpNetwork::setUseFusedMlp(bool _useFusedMlp) {
    if (useFusedMlp != _useFusedMlp) {
        useFusedMlp = _useFusedMlp;
        if (useFusedMlp) {
            shallRecreateFusedPass = true;
        } else {
            fusedPass = {};
        }
    }
}


void MlpNetwork::recreateMlpPass() {
    uint32_t parametersBufferOffset = 0;
    layerPasses.clear();
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
                renderer, mlpUseSharedMemory, layerChannelsIn, layerChannelsOut, layerActivationFunction,
                parametersBufferOffset, parametersBuffer);
        layerPass->setFloatFormat(format);
        layerPasses.push_back(layerPass);
        parametersBufferOffset += layerChannelsIn * layerChannelsOut;

        if (inputBuffer && outputBuffer) {
            sgl::vk::BufferPtr bufferIn, bufferOut;
            if (i == 0) {
                bufferIn = inputBuffer;
            } else {
                bufferIn = intermediateOutputBuffers[i % 2];
            }
            if (i == numLayers - 1) {
                bufferOut = outputBuffer;
            } else {
                bufferOut = intermediateOutputBuffers[(i + 1) % 2];
            }
            layerPass->setBuffersInOut(bufferIn, bufferOut);
        }

        if (batchSize != 0) {
            layerPasses.at(0)->setBatchSize(batchSize);
        }
    }
    shallRecreateMlpPass = false;
}

void MlpNetwork::checkRecreateMlpPass() {
    if (!useFusedMlp && shallRecreateMlpPass) {
        shallRecreateMlpPass = false;
        recreateMlpPass();
    }
}

void MlpNetwork::setMlpUseSharedMemory(bool _useSharedMemory) {
    if (mlpUseSharedMemory != _useSharedMemory) {
        mlpUseSharedMemory = _useSharedMemory;
        if (!useFusedMlp) {
            shallRecreateMlpPass = true;
        }
    }
}


void MlpNetwork::recreateFusedPass() {
    fusedPass = std::make_shared<MlpFusedPass>(
            renderer, useKhrExtension, M, K, N, subgroupSize,
            memoryType, fusedMlpDirectLoad, useSharedMemoryBankSkew,
            numLayers, numChannelsIn, numChannelsOut, numChannelsHidden,
            activationFunction, outputActivationFunction,
            parametersBuffer);
    if (inputBuffer && outputBuffer) {
        fusedPass->setBuffersInOut(inputBuffer, outputBuffer);
    }
    if (batchSize != 0) {
        fusedPass->setBatchSize(batchSize);
    }
    shallRecreateFusedPass = false;
}

void MlpNetwork::checkRecreateFusedPass() {
    if (useFusedMlp && shallRecreateFusedPass) {
        shallRecreateFusedPass = false;
        recreateFusedPass();
    }
}

void MlpNetwork::setFusedMlpMatrixBlockSize(uint32_t _M, uint32_t _K, uint32_t _N) {
    if (M != _M || K != _K || N != _N) {
        M = _M;
        K = _K;
        N = _N;
        if (useFusedMlp) {
            shallRecreateFusedPass = true;
        }
    }
}

void MlpNetwork::setFusedMlpExtension(bool _useKhrExtension) {
    if (useKhrExtension != _useKhrExtension) {
        useKhrExtension = _useKhrExtension;
        if (useFusedMlp) {
            shallRecreateFusedPass = true;
        }
    }
}

void MlpNetwork::setFusedMlpSubgroupSize(uint32_t _subgroupSize) {
    if (subgroupSize != _subgroupSize) {
        subgroupSize = _subgroupSize;
        if (useFusedMlp) {
            shallRecreateFusedPass = true;
        }
    }
}

void MlpNetwork::setFusedMlpSharedMemoryType(FusedMlpMemoryType _memoryType) {
    if (memoryType != _memoryType) {
        memoryType = _memoryType;
        if (useFusedMlp) {
            shallRecreateFusedPass = true;
        }
    }
}

void MlpNetwork::setFusedMlpDirectLoad(bool _fusedMlpDirectLoad) {
    if (fusedMlpDirectLoad != _fusedMlpDirectLoad) {
        fusedMlpDirectLoad = _fusedMlpDirectLoad;
        if (useFusedMlp) {
            shallRecreateFusedPass = true;
        }
    }
}

void MlpNetwork::setUseSharedMemoryBankSkew(bool _useSharedMemoryBankSkew) {
    if (useSharedMemoryBankSkew != _useSharedMemoryBankSkew) {
        useSharedMemoryBankSkew = _useSharedMemoryBankSkew;
        if (useFusedMlp) {
            shallRecreateFusedPass = true;
        }
    }
}

uint32_t MlpNetwork::getNumChannelsOutPadded() {
    if (useFusedMlp && memoryType != FusedMlpMemoryType::FLOAT16_NO_PADDING) {
        return nextMultiple(getNumChannelsOut(), 16);
    } else {
        return getNumChannelsOut();
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

    inputBuffer = input.getBuffer();
    outputBuffer = output.getBuffer();
    if (fusedPass) {
        fusedPass->setBuffersInOut(inputBuffer, outputBuffer);
    }
}

void MlpNetwork::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
    for (size_t i = 0; i < numLayers; i++) {
        auto& layerPass = layerPasses.at(i);
        layerPass->setBatchSize(_batchSize);
    }

    if (fusedPass) {
        fusedPass->setBatchSize(_batchSize);
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
    if (useFusedMlp) {
        fusedPass->render();
    } else {
        for (size_t i = 0; i < numLayers; i++) {
            auto& layerPass = layerPasses.at(i);
            layerPass->render();
        }
    }
}

}
