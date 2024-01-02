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

#ifndef CORRERENDER_NETWORK_HPP
#define CORRERENDER_NETWORK_HPP

#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>

#include "Format.hpp"

namespace Json {
class Value;
}
namespace sgl { namespace vk {
class Device;
class Renderer;
}}

namespace vmlp {

#define VMLP_ARRAYSIZE(_ARR) ((int)(sizeof(_ARR) / sizeof(*(_ARR))))

enum class ActivationFunction {
    NONE, RELU, SNAKE, SNAKE_ALT
};
const char* const ACTIVATION_FUNCTION_NAMES[] {
        "None", "ReLU", "Snake", "SnakeAlt"
};

void uploadParametersToBuffer(
        float* parameters, uint32_t numParameters,
        const sgl::vk::BufferPtr& parametersBuffer, FloatFormat format);
uint32_t nextMultiple(uint32_t num, uint32_t denom);
void addPreprocessorDefinesFormat(std::map<std::string, std::string>& preprocessorDefines, FloatFormat format);
ActivationFunction activationFunctionFromString(const std::string& activationName);
void debugPrintBuffer(const sgl::vk::BufferPtr& deviceBuffer, FloatFormat format, size_t numEntries);

class Matrix {
public:
    Matrix() : buffer(nullptr), numBytesPerComponent(0), offsetChannels(0), numChannels(0), batchSize(0) {}
    Matrix(
            sgl::vk::BufferPtr buffer, uint32_t numBytesPerComponent,
            uint32_t offsetChannels, uint32_t numChannels, uint32_t batchSize)
            : buffer(std::move(buffer)), numBytesPerComponent(numBytesPerComponent),
              offsetChannels(offsetChannels), numChannels(numChannels), batchSize(batchSize) {}
    Matrix(
            sgl::vk::BufferPtr buffer, uint32_t numBytesPerComponent,
            uint32_t numChannels, uint32_t batchSize)
            : buffer(std::move(buffer)), numBytesPerComponent(numBytesPerComponent),
              offsetChannels(0), numChannels(numChannels), batchSize(batchSize) {}
    [[nodiscard]] Matrix viewOffset(uint32_t offset) const {
        return { buffer, numBytesPerComponent, offsetChannels + offset, numChannels, batchSize };
    }
    [[nodiscard]] inline sgl::vk::BufferPtr getBuffer() const {
        return buffer;
    }
    [[nodiscard]] inline uint32_t getOffsetChannels() const {
        return offsetChannels;
    }
    [[nodiscard]] inline uint32_t getNumChannels() const {
        return numChannels;
    }
    [[nodiscard]] inline uint32_t getBatchSize() const {
        return batchSize;
    }

private:
    sgl::vk::BufferPtr buffer;
    uint32_t numBytesPerComponent, offsetChannels, numChannels, batchSize;
};

const uint32_t ALIGNMENT_MLP = 16;

class Module {
public:
    virtual ~Module() {
        if (parametersCpu) {
            delete[] parametersCpu;
            parametersCpu = nullptr;
        }
    }

    // Input/output information functionality.
    virtual uint32_t getOutputAlignment() { return 1; }
    virtual uint32_t getNumChannelsIn()=0;
    virtual uint32_t getNumChannelsOut()=0;
    virtual uint32_t getNumChannelsOutPadded() { return getNumChannelsOut(); }
    virtual void setInputOutputMatrices(const Matrix& input, const Matrix& output)=0;
    virtual void setBatchSize(uint32_t _batchSize)=0;
    virtual void setFloatFormat(FloatFormat _format)=0;

    // Parameter functionality.
    virtual uint32_t getNumParameters()=0;
    virtual void setParametersCpu(float* parameters, uint32_t numParameters);

    // Inference functionality.
    virtual void runInference()=0;

protected:
    FloatFormat format = FloatFormat::FLOAT32;
    float* parametersCpu = nullptr;
};

typedef std::shared_ptr<Module> ModulePtr;

ModulePtr createNetwork(sgl::vk::Renderer* renderer, const Json::Value& settingsNetwork);
ModulePtr createInputEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding);
ModulePtr createNetworkWithInputEncoding(
        sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding, const Json::Value& settingsNetwork);

class NetworkWithInputEncoding : public Module {
public:
    NetworkWithInputEncoding(
            sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding, Json::Value settingsNetwork);

    inline const std::shared_ptr<Module>& getNetwork() { return network; }
    inline const std::shared_ptr<Module>& getEncoding() { return encoding; }

    uint32_t getNumChannelsIn() override {
        return encoding->getNumChannelsIn();
    }
    uint32_t getNumChannelsOut() override {
        return network->getNumChannelsOut();
    }
    uint32_t getNumChannelsOutPadded() override {
        return network->getNumChannelsOutPadded();
    }
    void setInputOutputMatrices(const Matrix& input, const Matrix& output) override;
    void setBatchSize(uint32_t _batchSize) override {
        encoding->setBatchSize(_batchSize);
        network->setBatchSize(_batchSize);
    }
    void setFloatFormat(FloatFormat _format) override {
        if (format != _format) {
            floatFormatChanged = true;
            format = _format;
        }
        encoding->setFloatFormat(_format);
        network->setFloatFormat(_format);
    }

    uint32_t getNumParameters() override {
        return network->getNumParameters() + encoding->getNumParameters();
    }
    void setParametersCpu(float* parameters, uint32_t numParameters) override {
        if (numParameters != getNumParameters()) {
            sgl::Logfile::get()->throwError(
                    "Error in NetworkWithInputEncoding::setParametersCpu: Mismatch in number of parameters.");
        }
        network->setParametersCpu(parameters, network->getNumParameters());
        parameters += network->getNumParameters();
        encoding->setParametersCpu(parameters, encoding->getNumParameters());
    }

    void runInference() override;

private:
    sgl::vk::Device* device;
    bool floatFormatChanged = true;
    Matrix intermediateOutput;
    std::shared_ptr<Module> network = nullptr;
    std::shared_ptr<Module> encoding = nullptr;
};

class MlpPass;
class MlpFusedPass;

class MlpNetwork : public Module {
public:
    MlpNetwork(sgl::vk::Renderer* renderer, const Json::Value& settingsNetwork);
    uint32_t getNumChannelsIn() override {
        return numChannelsIn;
    }
    uint32_t getNumChannelsOut() override {
        return numChannelsOut;
    }
    uint32_t getNumChannelsOutPadded() override;
    void setInputOutputMatrices(const Matrix& input, const Matrix& output) override;
    void setBatchSize(uint32_t _batchSize) override;
    void setFloatFormat(FloatFormat _format) override;

    uint32_t getNumParameters() override {
        return numParameters;
    }

    void runInference() override;

    // Fused MLP settings.
    void checkRecreateFusedPass();
    void setUseFusedMlp(bool _useFusedMlp);
    void setFusedMlpMatrixBlockSize(uint32_t _matrixBlockSize);
    void setFusedMlpExtension(bool _useKhrExtension);
    void setFusedMlpSubgroupSize(uint32_t _subgroupSize);
    void setFusedMlpSharedMemoryType(FusedMlpMemoryType _memoryType);
    void setFusedMlpDirectLoad(bool _fusedMlpDirectLoad);
    void setUseSharedMemoryBankSkew(bool _useSharedMemoryBankSkew);

private:
    sgl::vk::Renderer* renderer;
    sgl::vk::Device* device;
    uint32_t numChannelsIn, numChannelsHidden, numChannelsOut;
    uint32_t numChannelsInPadded, numChannelsOutPadded;
    uint32_t numLayers;
    ActivationFunction activationFunction;
    ActivationFunction outputActivationFunction;
    uint32_t numParameters;
    sgl::vk::BufferPtr parametersBuffer;
    sgl::vk::BufferPtr intermediateOutputBuffers[2];
    bool floatFormatChanged = true;
    uint32_t cachedBatchSize = 0;
    std::vector<std::shared_ptr<MlpPass>> layerPasses;

    // Fused MLP.
    void recreateFusedPass();
    std::shared_ptr<MlpFusedPass> fusedPass;
    sgl::vk::BufferPtr inputBuffer, outputBuffer;
    uint32_t batchSize = 0;
    bool shallRecreateFusePass = false;
    bool useFusedMlp = false;
    uint32_t matrixBlockSize = 16;
    bool useKhrExtension = false;
    uint32_t subgroupSize = false;
    FusedMlpMemoryType memoryType = FusedMlpMemoryType::FLOAT16_NO_PADDING;
    bool fusedMlpDirectLoad = true;
    bool useSharedMemoryBankSkew = true;
};

}

#endif //CORRERENDER_NETWORK_HPP
