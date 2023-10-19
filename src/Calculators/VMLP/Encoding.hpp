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

#ifndef CORRERENDER_ENCODING_HPP
#define CORRERENDER_ENCODING_HPP

#include <vector>

#include "Network.hpp"

namespace Json {
class Value;
}

namespace vmlp {

class Encoding : public Module {
public:
    Encoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding);
    uint32_t getNumChannelsIn() override {
        return numChannelsEncode;
    }

protected:
    uint32_t channelOffset = 0, numChannelsEncode = 0;
};

class PaddingPass;

class CompositeEncoding : public Encoding {
public:
    CompositeEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsCompositeEncoding);
    uint32_t getNumChannelsOut() override {
        return numChannelsOutPadded;
    }
    void setInputOutputMatrices(const Matrix& input, const Matrix& output) override;
    void setBatchSize(uint32_t _batchSize) override;
    void setFloatFormat(FloatFormat _format) override;

    uint32_t getNumParameters() override {
        return numParameters;
    }
    void setParametersCpu(float* parameters, uint32_t _numParameters) override {
        for (auto& encoding : encodings) {
            uint32_t numParamsEncoding = encoding->getNumParameters();
            if (numParamsEncoding > _numParameters) {
                sgl::Logfile::get()->throwError(
                        "Error in CompositeEncoding::setParametersCpu: Not enough parameters provided.");
            }
            encoding->setParametersCpu(parameters, numParamsEncoding);
            parameters += numParamsEncoding;
            _numParameters -= numParamsEncoding;
        }
        if (_numParameters != 0) {
            sgl::Logfile::get()->throwError(
                    "Error in CompositeEncoding::setParametersCpu: Too many parameters provided.");
        }
    }

    uint32_t getOutputAlignment() override { return 1; }
    void runInference() override;

private:
    std::vector<std::shared_ptr<Module>> encodings;
    std::vector<uint32_t> encodingsChannelOffsets;
    std::vector<std::shared_ptr<PaddingPass>> paddingPasses;
    uint32_t numParameters = 0;
    uint32_t numChannelsOut = 0, numChannelsOutPadded = 0;
    //Matrix input, output; // For debug purposes.
};

class IdentityEncodingPass;

class IdentityEncoding : public Encoding {
public:
    IdentityEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding);
    uint32_t getNumChannelsOut() override {
        return numChannelsEncode;
    }
    void setInputOutputMatrices(const Matrix& input, const Matrix& output) override;
    void setBatchSize(uint32_t _batchSize) override;
    void setFloatFormat(FloatFormat _format) override;

    uint32_t getNumParameters() override {
        return 0;
    }

    uint32_t getOutputAlignment() override { return 1; }
    void runInference() override;

private:
    std::shared_ptr<IdentityEncodingPass> encodingPass;
};

class FrequencyEncodingPass;

class FrequencyEncoding : public Encoding {
public:
    FrequencyEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding);
    uint32_t getNumChannelsOut() override {
        return numFrequencies * numChannelsEncode * 2;
    }
    void setInputOutputMatrices(const Matrix& input, const Matrix& output) override;
    void setBatchSize(uint32_t _batchSize) override;
    void setFloatFormat(FloatFormat _format) override;

    uint32_t getNumParameters() override {
        return 0;
    }

    uint32_t getOutputAlignment() override { return 1; }
    void runInference() override;

private:
    uint32_t numFrequencies;
    std::shared_ptr<FrequencyEncodingPass> encodingPass;
};

enum class GridType {
    HASH, DENSE
};

enum class HashType {
    PRIME, COHERENT_PRIME, REVERSED_PRIME
};

enum class InterpolationType {
    NEAREST, LINEAR
};

class GridEncodingPass;
class EncodedPositionsTransposePass;

class GridEncoding : public Encoding {
public:
    GridEncoding(sgl::vk::Renderer* renderer, const Json::Value& settingsEncoding);
    uint32_t getNumChannelsOut() override {
        return numFeatures;
    }
    void setInputOutputMatrices(const Matrix& input, const Matrix& output) override;
    void setBatchSize(uint32_t _batchSize) override;
    void setFloatFormat(FloatFormat _format) override;

    uint32_t getNumParameters() override {
        return numParameters;
    }

    uint32_t getOutputAlignment() override { return numFeaturesPerLevel; }
    void runInference() override;

private:
    sgl::vk::Renderer* renderer;
    sgl::vk::Device* device;
    GridType gridType;
    InterpolationType interpolationType;
    HashType hashType;
    uint32_t numLevels;
    uint32_t numFeaturesPerLevel;
    uint32_t numFeatures;
    uint32_t log2HashMapSize;
    uint32_t baseResolution;
    float perLevelScale;
    uint32_t numParameters;
    sgl::vk::BufferPtr parametersBuffer;
    sgl::vk::BufferPtr encodedPositionsBuffer;
    bool floatFormatChanged = true;
    uint32_t cachedBatchSize = 0;
    std::vector<uint32_t> offsetTable;
    sgl::vk::BufferPtr offsetTableBuffer;
    std::shared_ptr<GridEncodingPass> encodingPass;
    std::shared_ptr<EncodedPositionsTransposePass> encodedPositionsTransposePass;
};

}

#endif //CORRERENDER_ENCODING_HPP
