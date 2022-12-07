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

QuickMLPSimilarityCalculator::~QuickMLPSimilarityCalculator() {
}

void loadNetwork(
        std::shared_ptr<qmlp::FusedNetwork>& network, const std::string& modelPath,
        const nlohmann::json& config, const sgl::ArchiveEntry& entry) {
    std::string quickMlpModelsPath = sgl::AppSettings::get()->getDataDirectory() + "QuickMLP/";
    network = std::make_shared<qmlp::FusedNetwork>(
            config, std::filesystem::path(quickMlpModelsPath));
    qmlp::Tensor::Precision precision = network->networkParameterPrecision(qmlp::Tensor::INFERENCE);

    auto* header = reinterpret_cast<QuickMLPDataHeader*>(entry.bufferData.get());
    uint8_t* paramsDataHost = entry.bufferData.get() + sizeof(QuickMLPDataHeader);
    if (header->format == QUICK_MLP_PARAMS_FORMAT_FLOAT && precision != qmlp::Tensor::Precision::FLOAT
            || header->format == QUICK_MLP_PARAMS_FORMAT_HALF && precision != qmlp::Tensor::Precision::HALF) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Precision mismatch between QuickMLP JSON configuration and binary data for \""
                + modelPath + "\".");
    }
    if (header->numParams != uint32_t(network->networkParameterCount())) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Mismatching network parameter count (" + std::to_string(header->numParams)
                + " vs. " + std::to_string(network->networkParameterCount()) + ") for \"" + modelPath + "\".");
    }
    if (entry.bufferSize != uint32_t(qmlp::Tensor::BytesPerEntry[precision] * network->networkParameterCount())) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Mismatching parameter byte size (" + std::to_string(entry.bufferSize)
                + " vs. " + std::to_string(qmlp::Tensor::BytesPerEntry[precision] * network->networkParameterCount())
                + ") for \"" + modelPath + "\".");
    }

    qmlp::Tensor parameters(precision, { network->networkParameterCount() });
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
            reinterpret_cast<CUdeviceptr>(parameters.rawPtr()), paramsDataHost,
            qmlp::Tensor::BytesPerEntry[precision] * network->networkParameterCount()), "Error in cuMemcpyHtoD: ");
    network->setNetworkParameter(parameters, qmlp::Tensor::INFERENCE);
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

    cacheWrapper->referenceInput = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionIn(), { 1, es * 4 });
    cacheWrapper->referenceEncoded = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), { 1, int(numLayersOutEncoder) });
    cacheWrapper->queryInput = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionIn(), { int(batchSize), int(es * 4) });
    cacheWrapper->queryEncoded = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), { int(batchSize), int(numLayersOutEncoder) });
    cacheWrapper->queryEncodedPermuted = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), { int(batchSize), int(numLayersOutEncoder) });
    cacheWrapper->symmetrizedReferenceInput = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionIn(), { int(batchSize), int(numLayersInDecoder) });
    cacheWrapper->symmetrizedQueryInput = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionIn(), { int(batchSize), int(numLayersInDecoder) });
    cacheWrapper->referenceDecoded = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionOut(), { int(batchSize), int(numLayersOutDecoder) });
    cacheWrapper->queryDecoded = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionOut(), { int(batchSize), int(numLayersOutDecoder) });
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

    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
            (CUdeviceptr)cacheWrapper->queryEncodedPermuted.rawPtr(),
            (CUdeviceptr)cacheWrapper->queryEncoded.rawPtr(),
            sizeof(float) * batchSize, stream), "Error in cuMemcpyAsync: ");

    if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
        randomShuffleFisherYatesXorshift<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->queryEncodedPermuted.dataPtr<float>(), numLayersOutEncoder);
        symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<float>(), cacheWrapper->queryEncoded.dataPtr<float>(),
                cacheWrapper->symmetrizedReferenceInput.dataPtr<float>(), numLayersOutEncoder);
        symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<float>(), cacheWrapper->queryEncodedPermuted.dataPtr<float>(),
                cacheWrapper->symmetrizedQueryInput.dataPtr<float>(), numLayersOutEncoder);
    } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
        randomShuffleFisherYatesXorshift<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->queryEncodedPermuted.dataPtr<half>(), numLayersOutEncoder);
        symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<half>(), cacheWrapper->queryEncoded.dataPtr<half>(),
                cacheWrapper->symmetrizedReferenceInput.dataPtr<half>(), numLayersOutEncoder);
        symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceEncoded.dataPtr<half>(), cacheWrapper->queryEncodedPermuted.dataPtr<half>(),
                cacheWrapper->symmetrizedQueryInput.dataPtr<half>(), numLayersOutEncoder);
    }

    moduleWrapper->networkDecoder->inference(
            cacheWrapper->symmetrizedReferenceInput, cacheWrapper->referenceDecoded, stream);
    moduleWrapper->networkDecoder->inference(
            cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded, stream);

    float *miOutput = reinterpret_cast<float*>(outputImageBufferCu) + batchOffset;
    if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
        combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceDecoded.dataPtr<float>(), cacheWrapper->queryDecoded.dataPtr<float>(),
                        miOutput, numLayersOutDecoder, 1);
    } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
        combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceDecoded.dataPtr<half>(), cacheWrapper->queryDecoded.dataPtr<half>(),
                miOutput, numLayersOutDecoder, 1);
    }
}
