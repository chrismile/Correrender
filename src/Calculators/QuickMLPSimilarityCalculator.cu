/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#include <filesystem>

#include <qmlp/fused_network.h>

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Archive.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>

#include "Volume/VolumeData.hpp"
#include "MutualInformation.cuh"
#include "QuickMLPSimilarityCalculator.hpp"

struct QuickMLPModuleWrapper {
    nlohmann::json configEncoder;
    nlohmann::json configDecoder;
    std::shared_ptr<qmlp::FusedNetwork> networkEncoder;
    std::shared_ptr<qmlp::FusedNetwork> networkDecoder;
};

struct QuickMLPCacheWrapper {
    qmlp::Tensor referenceInput;
    qmlp::Tensor referenceEncoded;
    qmlp::Tensor queryInput;
    qmlp::Tensor queryEncoded;
    qmlp::Tensor queryEncodedPermuted;
    qmlp::Tensor symmetrizedReferenceInput;
    qmlp::Tensor symmetrizedQueryInput;
    qmlp::Tensor referenceDecoded;
    qmlp::Tensor queryDecoded;
};

const uint32_t QUICK_MLP_PARAMS_FORMAT_FLOAT = 0;
const uint32_t QUICK_MLP_PARAMS_FORMAT_HALF = 1;
struct QuickMLPDataHeader {
    uint32_t format = 0;
    uint32_t numParams = 0;
};


QuickMLPSimilarityCalculator::QuickMLPSimilarityCalculator(sgl::vk::Renderer* renderer)
        : DeepLearningCudaSimilarityCalculator("QuickMLP", "quickMLP", renderer) {
    cacheWrapper = std::make_shared<QuickMLPCacheWrapper>();
}

QuickMLPSimilarityCalculator::~QuickMLPSimilarityCalculator() = default;

void QuickMLPSimilarityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    DeepLearningCudaSimilarityCalculator::setVolumeData(_volumeData, isNewData);
    calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::QUICK_MLP);
}

void loadNetwork(
        std::shared_ptr<qmlp::FusedNetwork>& network, const std::string& modelPath,
        const nlohmann::json& config, const sgl::ArchiveEntry& entry) {
    std::string quickMlpModelsPath = sgl::AppSettings::get()->getDataDirectory() + "QuickMLP/";
    network = std::make_shared<qmlp::FusedNetwork>(
            config, std::filesystem::path(quickMlpModelsPath));
    qmlp::Tensor::Precision precision = network->networkParameterPrecision(qmlp::Tensor::INFERENCE);
    int numEncodings = network->numEncodings();

    auto numParametersTotal = uint32_t(network->networkParameterCount());
    for (int encodingIdx = 0; encodingIdx < numEncodings; encodingIdx++) {
        auto encoding = network->encoding(encodingIdx);
        if (encoding->hasParameters()) {
            numParametersTotal += encoding->parameterCount();
        }
    }

    auto* header = reinterpret_cast<QuickMLPDataHeader*>(entry.bufferData.get());
    uint8_t* paramsDataHost = entry.bufferData.get() + sizeof(QuickMLPDataHeader);
    if (header->numParams != numParametersTotal) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Mismatching network parameter count (" + std::to_string(header->numParams)
                + " vs. " + std::to_string(numParametersTotal) + ") for \"" + modelPath + "\".");
    }
    if (header->format == QUICK_MLP_PARAMS_FORMAT_FLOAT && precision == qmlp::Tensor::Precision::HALF) {
        float* dataOld = reinterpret_cast<float*>(paramsDataHost);
        half* dataNew = new half[header->numParams];
        for (uint32_t i = 0; i < header->numParams; i++) {
            dataNew[i] = half(dataOld[i]);
        }
        paramsDataHost = reinterpret_cast<uint8_t*>(dataNew);
    } else if (header->format == QUICK_MLP_PARAMS_FORMAT_HALF && precision == qmlp::Tensor::Precision::FLOAT) {
        half* dataOld = reinterpret_cast<half*>(paramsDataHost);
        float* dataNew = new float[header->numParams];
        for (uint32_t i = 0; i < header->numParams; i++) {
            dataNew[i] = float(dataOld[i]);
        }
        paramsDataHost = reinterpret_cast<uint8_t*>(dataNew);
    } else if (header->format == QUICK_MLP_PARAMS_FORMAT_FLOAT && precision != qmlp::Tensor::Precision::FLOAT
            || header->format == QUICK_MLP_PARAMS_FORMAT_HALF && precision != qmlp::Tensor::Precision::HALF) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Precision mismatch between QuickMLP JSON configuration and binary data for \""
                + modelPath + "\".");
    }

    qmlp::Tensor parameters(precision, { network->networkParameterCount() });
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
            reinterpret_cast<CUdeviceptr>(parameters.rawPtr()), paramsDataHost,
            qmlp::Tensor::BytesPerEntry[precision] * network->networkParameterCount()), "Error in cuMemcpyHtoD: ");
    network->setNetworkParameter(parameters, qmlp::Tensor::INFERENCE);

    uint32_t parameterOffset = network->networkParameterCount();
    for (int encodingIdx = 0; encodingIdx < numEncodings; encodingIdx++) {
        auto encoding = network->encoding(encodingIdx);
        if (encoding->hasParameters()) {
            qmlp::Tensor::Precision precisionEncoding = encoding->parameterPrecision(qmlp::Tensor::INFERENCE);
            if (precision != precisionEncoding) {
                sgl::Logfile::get()->throwError(
                        "Error in loadNetwork: Precision mismatch between QuickMLP network and encoding in file \""
                        + modelPath + "\".");
            }
            qmlp::Tensor parametersEncoding(precisionEncoding, { encoding->parameterCount() });
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                    reinterpret_cast<CUdeviceptr>(parametersEncoding.rawPtr()), paramsDataHost + parameterOffset,
                    qmlp::Tensor::BytesPerEntry[precisionEncoding] * encoding->parameterCount()), "Error in cuMemcpyHtoD: ");
            encoding->setParameter(parametersEncoding, qmlp::Tensor::INFERENCE);
            parameterOffset += encoding->parameterCount();
        }
    }

    if (header->format == QUICK_MLP_PARAMS_FORMAT_FLOAT && precision == qmlp::Tensor::Precision::HALF
            || header->format == QUICK_MLP_PARAMS_FORMAT_HALF && precision == qmlp::Tensor::Precision::FLOAT) {
        delete[] paramsDataHost;
    }
}

void QuickMLPSimilarityCalculator::loadModelFromFile(const std::string& modelPath) {
    moduleWrapper = std::make_shared<QuickMLPModuleWrapper>();

    std::unordered_map<std::string, sgl::ArchiveEntry> archiveFiles;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadAllFilesFromArchive(modelPath, archiveFiles, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in QuickMLPSimilarityCalculator::loadModelFromFile: Could not load data from model \""
                + modelPath + "\".");
        return;
    }

    auto itConfig = archiveFiles.find("config.json");
    auto itConfigEncoder = archiveFiles.find("config_encoder.json");
    auto itConfigDecoder = archiveFiles.find("config_decoder.json");
    if (itConfig != archiveFiles.end()) {
        const auto& entry = itConfig->second;
        moduleWrapper->configEncoder = moduleWrapper->configDecoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entry.bufferData.get()), entry.bufferSize));
    } else if (itConfigEncoder != archiveFiles.end() && itConfigDecoder != archiveFiles.end()) {
        const auto& entryEncoder = itConfigEncoder->second;
        const auto& entryDecoder = itConfigDecoder->second;
        moduleWrapper->configEncoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entryEncoder.bufferData.get()), entryEncoder.bufferSize));
        moduleWrapper->configDecoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entryDecoder.bufferData.get()), entryDecoder.bufferSize));
    } else {
        sgl::Logfile::get()->writeError(
                "Error in QuickMLPSimilarityCalculator::loadModelFromFile: Could not load config from model \""
                + modelPath + "\".");
        return;
    }

    auto itNetworkEncoder = archiveFiles.find("network_encoder.bin");
    auto itNetworkDecoder = archiveFiles.find("network_decoder.bin");
    if (itNetworkEncoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in QuickMLPSimilarityCalculator::loadModelFromFile: Missing network_encoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    if (itNetworkDecoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in QuickMLPSimilarityCalculator::loadModelFromFile: Missing network_decoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    moduleWrapper->networkEncoder = {};
    moduleWrapper->networkDecoder = {};
    loadNetwork(moduleWrapper->networkEncoder, modelPath, moduleWrapper->configEncoder, itNetworkEncoder->second);
    loadNetwork(moduleWrapper->networkDecoder, modelPath, moduleWrapper->configDecoder, itNetworkDecoder->second);

    // numLayersInDecoder == numLayersOutEncoder when symmetrizer is sum operation.
    numLayersInEncoder = uint32_t(moduleWrapper->networkEncoder->channelsIn());
    numLayersOutEncoder = uint32_t(moduleWrapper->networkEncoder->channelsOut());
    numLayersInDecoder = uint32_t(moduleWrapper->networkDecoder->channelsIn());
    numLayersOutDecoder = uint32_t(moduleWrapper->networkDecoder->channelsOut());
}

void QuickMLPSimilarityCalculator::recreateCache(int batchSize) {
    int es = volumeData->getEnsembleMemberCount();
    if (moduleWrapper->networkEncoder->precisionIn() != qmlp::Tensor::Precision::FLOAT) {
        sgl::Logfile::get()->throwError(
                "Error in QuickMLPSimilarityCalculator::recreateCache: "
                "Encoder input precision is expected to be float.");
    }

    // Network needs multiple of 16 for number of input layers.
    //int numInputLayers = 16;

    cacheWrapper->referenceInput = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionIn(), { es, int(numLayersInEncoder) });
    cacheWrapper->referenceEncoded = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), { es, int(numLayersOutEncoder) });
    cacheWrapper->queryInput = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionIn(), { es * int(batchSize), int(numLayersInEncoder) });
    cacheWrapper->queryEncoded = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), { es * int(batchSize), int(numLayersOutEncoder) });
    cacheWrapper->queryEncodedPermuted = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), { es * int(batchSize), int(numLayersOutEncoder) });
    cacheWrapper->symmetrizedReferenceInput = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionIn(), { es * int(batchSize), int(numLayersInDecoder) });
    cacheWrapper->symmetrizedQueryInput = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionIn(), { es * int(batchSize), int(numLayersInDecoder) });
    cacheWrapper->referenceDecoded = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionOut(), { es * int(batchSize), int(numLayersOutDecoder) });
    cacheWrapper->queryDecoded = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionOut(), { es * int(batchSize), int(numLayersOutDecoder) });
}

CUdeviceptr QuickMLPSimilarityCalculator::getReferenceInputPointer() {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->referenceInput.rawPtr());
}

CUdeviceptr QuickMLPSimilarityCalculator::getQueryInputPointer() {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->queryInput.rawPtr());
}

void QuickMLPSimilarityCalculator::runInferenceReference() {
    moduleWrapper->networkEncoder->inference(cacheWrapper->referenceInput, cacheWrapper->referenceEncoded, stream);
}

void QuickMLPSimilarityCalculator::runInferenceBatch(uint32_t batchOffset, uint32_t batchSize) {
    int es = volumeData->getEnsembleMemberCount();

    moduleWrapper->networkEncoder->inference(cacheWrapper->queryInput, cacheWrapper->queryEncoded, stream);

    /*sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
            (CUdeviceptr)cacheWrapper->queryEncodedPermuted.rawPtr(),
            (CUdeviceptr)cacheWrapper->queryEncoded.rawPtr(),
            sizeof(float) * batchSize, stream), "Error in cuMemcpyAsync: ");*/

    uint32_t* permutationIndicesBuffer = reinterpret_cast<uint32_t*>(permutationIndicesBufferCu);
    generateRandomPermutations<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
            permutationIndicesBuffer, uint32_t(es), batchOffset);

    if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
        //randomShuffleFisherYates<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
        //        cacheWrapper->queryEncodedPermuted.dataPtr<float>(), cacheWrapper->queryEncoded.dataPtr<float>(),
        //        permutationIndicesBuffer, uint32_t(es), numLayersOutEncoder);
        symmetrizer<<<sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<float>(), cacheWrapper->queryEncoded.dataPtr<float>(),
                cacheWrapper->symmetrizedReferenceInput.dataPtr<float>(), uint32_t(es), numLayersOutEncoder);
        symmetrizerPermuted<<<sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<float>(), cacheWrapper->queryEncoded.dataPtr<float>(),
                cacheWrapper->symmetrizedQueryInput.dataPtr<float>(), permutationIndicesBuffer,
                uint32_t(es), numLayersOutEncoder);
    } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
        //randomShuffleFisherYates<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
        //        cacheWrapper->queryEncodedPermuted.dataPtr<half>(), cacheWrapper->queryEncoded.dataPtr<half>(),
        //        permutationIndicesBuffer, uint32_t(es), numLayersOutEncoder);
        symmetrizer<<<sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<half>(), cacheWrapper->queryEncoded.dataPtr<half>(),
                cacheWrapper->symmetrizedReferenceInput.dataPtr<half>(), uint32_t(es), numLayersOutEncoder);
        symmetrizerPermuted<<<sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<half>(), cacheWrapper->queryEncoded.dataPtr<half>(),
                cacheWrapper->symmetrizedQueryInput.dataPtr<half>(), permutationIndicesBuffer,
                uint32_t(es), numLayersOutEncoder);
    }

    moduleWrapper->networkDecoder->inference(
            cacheWrapper->symmetrizedReferenceInput, cacheWrapper->referenceDecoded, stream);
    moduleWrapper->networkDecoder->inference(
            cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded, stream);

    float* miOutput = reinterpret_cast<float*>(outputImageBufferCu) + batchOffset;
    if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
        combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceDecoded.dataPtr<float>(), cacheWrapper->queryDecoded.dataPtr<float>(),
                miOutput, uint32_t(es), numLayersOutDecoder);
    } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
        combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceDecoded.dataPtr<half>(), cacheWrapper->queryDecoded.dataPtr<half>(),
                miOutput, uint32_t(es), numLayersOutDecoder);
    }
}
