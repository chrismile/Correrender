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

#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <Math/Math.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Archive.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>

#include "Volume/VolumeData.hpp"
#include "MutualInformation.cuh"
#include "TinyCudaNNSimilarityCalculator.hpp"

using precision_t = tcnn::network_precision_t;

struct TinyCudaNNModuleWrapper {
    nlohmann::json configEncoder;
    nlohmann::json configDecoder;
    std::shared_ptr<tcnn::Network<float, precision_t>> networkEncoder;
#ifdef TCNN_HALF_PRECISION
    std::shared_ptr<tcnn::Network<precision_t, precision_t>> networkEncoderHalf;
#endif
    std::shared_ptr<tcnn::Network<precision_t, precision_t>> networkDecoder;
};

struct TinyCudaNNCacheWrapper {
    tcnn::GPUMatrix<float> referenceInput;
#ifdef TCNN_HALF_PRECISION
    tcnn::GPUMatrix<precision_t> referenceInputHalf;
#endif
    tcnn::GPUMatrix<precision_t> referenceEncoded;
    tcnn::GPUMatrix<float> queryInput;
#ifdef TCNN_HALF_PRECISION
    tcnn::GPUMatrix<precision_t> queryInputHalf;
#endif
    tcnn::GPUMatrix<precision_t> queryEncoded;
    tcnn::GPUMatrix<precision_t> queryEncodedPermuted;
    tcnn::GPUMatrix<precision_t> symmetrizedReferenceInput;
    tcnn::GPUMatrix<precision_t> symmetrizedQueryInput;
    tcnn::GPUMatrix<precision_t> referenceDecoded;
    tcnn::GPUMatrix<precision_t> queryDecoded;
};

const uint32_t TINY_CUDA_NN_PARAMS_FORMAT_FLOAT = 0;
const uint32_t TINY_CUDA_NN_PARAMS_FORMAT_HALF = 1;
struct TinyCudaNNDataHeader {
    uint32_t format = 0;
    uint32_t numParams = 0;
};

TinyCudaNNSimilarityCalculator::TinyCudaNNSimilarityCalculator(sgl::vk::Renderer* renderer)
        : DeepLearningCudaSimilarityCalculator("tiny-cuda-nn", "tinyCudaNN", renderer) {
    cacheWrapper = std::make_shared<TinyCudaNNCacheWrapper>();
}

TinyCudaNNSimilarityCalculator::~TinyCudaNNSimilarityCalculator() {
}

template<class T, class PARAMS_T> static void loadNetwork(
        std::shared_ptr<tcnn::Network<T, PARAMS_T>>& network, const std::string& modelPath,
        const nlohmann::json& config, const sgl::ArchiveEntry& entry) {
    auto* header = reinterpret_cast<TinyCudaNNDataHeader*>(entry.bufferData.get());
    uint8_t* paramsData = entry.bufferData.get() + sizeof(TinyCudaNNDataHeader);
    uint32_t numParams = header->numParams;

    size_t sizePerEntry = header->format == TINY_CUDA_NN_PARAMS_FORMAT_FLOAT ? 4 : 2;
    if (numParams * sizePerEntry + sizeof(TinyCudaNNDataHeader) != entry.bufferSize) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Invalid number of parameters for file size.");
    }

    bool hasInputEncoding = config.find("encoding") != config.end();
    bool isInputEncodingIdentity = false;
    auto encodingOpts = config.value("encoding", nlohmann::json::object());
    auto lossOpts = config.value("loss", nlohmann::json::object());
    auto optimizerOpts = config.value("optimizer", nlohmann::json::object());
    auto networkOpts = config.value("network", nlohmann::json::object());
    if (hasInputEncoding) {
        if (encodingOpts.value("otype", "Identity") == "Identity"
                && encodingOpts.value("scale", 1.0f) == 1.0f
                && encodingOpts.value("offset", 0.0f) == 0.0f) {
            isInputEncodingIdentity = true;
        }
    }

    uint32_t numInputDims = networkOpts["n_input_dims"];
    uint32_t numOutputDims = networkOpts["n_output_dims"];
    std::shared_ptr<tcnn::Loss<precision_t>> loss{tcnn::create_loss<precision_t>(lossOpts)};
    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{tcnn::create_optimizer<precision_t>(optimizerOpts)};
    if (hasInputEncoding && !isInputEncodingIdentity) {
        std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> networkWithEnc =
                std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(
                        numInputDims, numOutputDims, encodingOpts, networkOpts);
        if constexpr (std::is_same<T, float>::value) {
            network = std::static_pointer_cast<tcnn::Network<float, PARAMS_T>>(networkWithEnc);
        }
    } else {
        if constexpr (std::is_same<T, PARAMS_T>::value) {
            network = std::shared_ptr<tcnn::Network<PARAMS_T, PARAMS_T>>(
                    tcnn::create_network<PARAMS_T>(networkOpts));
        }
    }
    auto trainer = std::make_shared<tcnn::Trainer<T, PARAMS_T, PARAMS_T>>(network, optimizer, loss);

    // Do we need padding because the output width is not a multiple of 16?
    if (network->output_width() != network->padded_output_width()) {
        uint32_t numNeurons = networkOpts["n_neurons"];
        uint32_t paddingSize = numNeurons * (network->padded_output_width() - network->output_width());
        size_t numParamsOld = numParams;
        numParams += paddingSize;
        const uint8_t* paramsDataOld = paramsData;
        paramsData = new uint8_t[numParams * sizePerEntry];
        memcpy(paramsData, paramsDataOld, numParamsOld * sizePerEntry);
        memset(paramsData + numParamsOld * sizePerEntry, 0, paddingSize * sizePerEntry);
    }

    if (network->n_params() != numParams) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Mismatching network parameter count (" + std::to_string(numParams)
                + " vs. " + std::to_string(network->n_params()) + ") for \"" + modelPath + "\".");
    }

#ifdef TCNN_HALF_PRECISION
    if (header->format == TINY_CUDA_NN_PARAMS_FORMAT_FLOAT) {
        trainer->set_params_full_precision(reinterpret_cast<float*>(paramsData), numParams, false);
    } else {
        trainer->set_params(reinterpret_cast<precision_t*>(paramsData), 0, false);
    }
#else
    if (header->format == TINY_CUDA_NN_PARAMS_FORMAT_FLOAT) {
        trainer->set_params(reinterpret_cast<float*>(paramsData), 0, false);
    } else {
        sgl::Logfile::get()->throwError(
                "Error in TinyCudaNNSimilarityCalculator::loadNetwork: Half precision build was disabled.");
    }
#endif

    if (network->output_width() != network->padded_output_width()) {
        delete[] paramsData;
    }

    // TODO: Support trainer->serialize()
}

void TinyCudaNNSimilarityCalculator::loadModelFromFile(const std::string& modelPath) {
    moduleWrapper = std::make_shared<TinyCudaNNModuleWrapper>();
    cacheWrapper = std::make_shared<TinyCudaNNCacheWrapper>();

    std::unordered_map<std::string, sgl::ArchiveEntry> archiveFiles;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadAllFilesFromArchive(modelPath, archiveFiles, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Could not load data from model \""
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
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Could not load config from model \""
                + modelPath + "\".");
        return;
    }

    // Set input/output layer configurations for both networks.
    int es = volumeData->getEnsembleMemberCount();
    auto networkOpts = moduleWrapper->configEncoder.value("network", nlohmann::json::object());
    moduleWrapper->configEncoder["network"]["n_input_dims"] = 4;
    moduleWrapper->configDecoder["network"]["n_output_dims"] = 1;
    if (networkOpts.find("n_output_dims") == networkOpts.end()) {
        moduleWrapper->configEncoder["network"]["n_output_dims"] = moduleWrapper->configEncoder["network"]["n_neurons"];
        moduleWrapper->configDecoder["network"]["n_input_dims"] = moduleWrapper->configEncoder["network"]["n_neurons"];
    }

    bool hasInputEncoding = moduleWrapper->configEncoder.find("encoding") != moduleWrapper->configEncoder.end();
    bool isInputEncodingIdentity = false;
    auto encodingOpts = moduleWrapper->configEncoder.value("encoding", nlohmann::json::object());
    if (hasInputEncoding) {
        if (encodingOpts.value("otype", "Identity") == "Identity"
            && encodingOpts.value("scale", 1.0f) == 1.0f
            && encodingOpts.value("offset", 0.0f) == 0.0f) {
            isInputEncodingIdentity = true;
        }
    }

    auto itNetworkEncoder = archiveFiles.find("network_encoder.bin");
    auto itNetworkDecoder = archiveFiles.find("network_decoder.bin");
    if (itNetworkEncoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Missing network_encoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    if (itNetworkDecoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Missing network_decoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    moduleWrapper->networkEncoder = {};
#ifdef TCNN_HALF_PRECISION
    std::shared_ptr<tcnn::Network<precision_t, precision_t>> networkEncoderHalf;
#endif
    moduleWrapper->networkEncoderHalf = {};
    moduleWrapper->networkDecoder = {};
#ifdef TCNN_HALF_PRECISION
    if (hasInputEncoding && !isInputEncodingIdentity) {
#endif
        loadNetwork(moduleWrapper->networkEncoder, modelPath, moduleWrapper->configEncoder, itNetworkEncoder->second);
#ifdef TCNN_HALF_PRECISION
    } else {
        loadNetwork(moduleWrapper->networkEncoderHalf, modelPath, moduleWrapper->configEncoder, itNetworkEncoder->second);
    }
#endif
    loadNetwork(moduleWrapper->networkDecoder, modelPath, moduleWrapper->configDecoder, itNetworkDecoder->second);

    // numLayersInDecoder == numLayersOutEncoder when symmetrizer is sum operation.
    // TODO
#ifdef TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        numLayersInEncoder = uint32_t(moduleWrapper->networkEncoderHalf->input_width());
        numLayersOutEncoder = uint32_t(moduleWrapper->networkEncoderHalf->padded_output_width());
    } else {
        numLayersInEncoder = uint32_t(moduleWrapper->networkEncoder->input_width());
        numLayersOutEncoder = uint32_t(moduleWrapper->networkEncoder->padded_output_width());
    }
#endif
    numLayersInDecoder = uint32_t(moduleWrapper->networkDecoder->input_width());
    numLayersOutDecoder = uint32_t(moduleWrapper->networkDecoder->padded_output_width());
    //moduleWrapper->networkDecoder->channelsIn():
}

void TinyCudaNNSimilarityCalculator::recreateCache(int batchSize) {
    int es = volumeData->getEnsembleMemberCount();

    //cacheWrapper->referenceInput = tcnn::GPUMatrix<float>(uint32_t(es) * 4, 1);
    //cacheWrapper->referenceEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, 1);
    //cacheWrapper->queryInput = tcnn::GPUMatrix<float>(uint32_t(es) * 4, batchSize);
    //cacheWrapper->queryEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, batchSize);
    //cacheWrapper->queryEncodedPermuted = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, batchSize);
    //cacheWrapper->symmetrizedReferenceInput = tcnn::GPUMatrix<precision_t>(numLayersInDecoder, batchSize);
    //cacheWrapper->symmetrizedQueryInput = tcnn::GPUMatrix<precision_t>(numLayersInDecoder, batchSize);
    //cacheWrapper->referenceDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, batchSize);
    //cacheWrapper->queryDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, batchSize);

    cacheWrapper->referenceInput = tcnn::GPUMatrix<float>();
#ifdef TCNN_HALF_PRECISION
    cacheWrapper->referenceInputHalf = tcnn::GPUMatrix<precision_t>();
#endif
    cacheWrapper->referenceEncoded = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->queryInput = tcnn::GPUMatrix<float>();
#ifdef TCNN_HALF_PRECISION
    cacheWrapper->queryInputHalf = tcnn::GPUMatrix<precision_t>();
#endif
    cacheWrapper->queryEncoded = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->queryEncodedPermuted = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->symmetrizedReferenceInput = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->symmetrizedQueryInput = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->referenceDecoded = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->queryDecoded = tcnn::GPUMatrix<precision_t>();

    // mlp_fused_forward needs multiple of 16 for number of input layers.
    uint32_t numInputLayers = 16;
    uint32_t referenceInputBatchSize =
            sgl::uiceil(uint32_t(es), tcnn::batch_size_granularity) * tcnn::batch_size_granularity;
#ifdef TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        cacheWrapper->referenceInputHalf = tcnn::GPUMatrix<precision_t>(numInputLayers, referenceInputBatchSize);
        cacheWrapper->queryInputHalf = tcnn::GPUMatrix<precision_t>(numInputLayers, uint32_t(es) * batchSize);
    }
#endif
    cacheWrapper->referenceInput = tcnn::GPUMatrix<float>(numInputLayers, referenceInputBatchSize);
    cacheWrapper->queryInput = tcnn::GPUMatrix<float>(numInputLayers, uint32_t(es) * batchSize);
    cacheWrapper->referenceEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, referenceInputBatchSize);
    cacheWrapper->queryEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, uint32_t(es) * batchSize);
    cacheWrapper->queryEncodedPermuted = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, uint32_t(es) * batchSize);
    cacheWrapper->symmetrizedReferenceInput = tcnn::GPUMatrix<precision_t>(numLayersInDecoder, uint32_t(es) * batchSize);
    cacheWrapper->symmetrizedQueryInput = tcnn::GPUMatrix<precision_t>(numLayersInDecoder, uint32_t(es) * batchSize);
    cacheWrapper->referenceDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, uint32_t(es) * batchSize);
    cacheWrapper->queryDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, uint32_t(es) * batchSize);
}

CUdeviceptr TinyCudaNNSimilarityCalculator::getReferenceInputPointer()  {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->referenceInput.data());
}

CUdeviceptr TinyCudaNNSimilarityCalculator::getQueryInputPointer()  {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->queryInput.data());
}

void TinyCudaNNSimilarityCalculator::runInferenceReference() {
#ifdef TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        uint32_t arraySize = cacheWrapper->referenceInputHalf.n() * cacheWrapper->referenceInputHalf.m();
        convertFloatToHalfArray<<<sgl::uiceil(arraySize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceInputHalf.data(), cacheWrapper->referenceInput.data(), arraySize);
        moduleWrapper->networkEncoderHalf->inference_mixed_precision(
                stream, cacheWrapper->referenceInputHalf, cacheWrapper->referenceEncoded);
    } else {
        moduleWrapper->networkEncoder->inference_mixed_precision(
                stream, cacheWrapper->referenceInput, cacheWrapper->referenceEncoded);
    }
#else
    moduleWrapper->networkEncoder->inference(
                stream, cacheWrapper->referenceInput, cacheWrapper->referenceEncoded);
#endif
}

void TinyCudaNNSimilarityCalculator::runInferenceBatch(uint32_t batchOffset, uint32_t batchSize)  {
    int es = volumeData->getEnsembleMemberCount();

#ifdef TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        uint32_t arraySize = cacheWrapper->queryInputHalf.n() * cacheWrapper->queryInputHalf.m();
        convertFloatToHalfArray<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->queryInputHalf.data(), cacheWrapper->queryInput.data(), arraySize);
        moduleWrapper->networkEncoderHalf->inference_mixed_precision(
                stream, cacheWrapper->queryInputHalf, cacheWrapper->queryEncoded);
    } else {
        moduleWrapper->networkEncoder->inference_mixed_precision(
                stream, cacheWrapper->queryInput, cacheWrapper->queryEncoded);
    }
#else
    moduleWrapper->networkEncoder->inference(
                stream, cacheWrapper->queryInput, cacheWrapper->queryEncoded);
#endif

    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
            (CUdeviceptr)cacheWrapper->queryEncodedPermuted.data(),
            (CUdeviceptr)cacheWrapper->queryEncoded.data(),
            sizeof(float) * batchSize, stream), "Error in cuMemcpyAsync: ");
    randomShuffleFisherYatesXorshift<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
            cacheWrapper->queryEncodedPermuted.data(), numLayersOutEncoder);

    symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
            cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncoded.data(),
            cacheWrapper->symmetrizedReferenceInput.data(), numLayersOutEncoder);
    symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
            cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncodedPermuted.data(),
            cacheWrapper->symmetrizedQueryInput.data(), numLayersOutEncoder);

#ifdef TCNN_HALF_PRECISION
    moduleWrapper->networkDecoder->inference_mixed_precision(
            stream, cacheWrapper->symmetrizedReferenceInput, cacheWrapper->referenceDecoded);
    moduleWrapper->networkDecoder->inference_mixed_precision(
            stream, cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded);
#else
    moduleWrapper->networkDecoder->inference(
            stream, cacheWrapper->symmetrizedReferenceInput, cacheWrapper->referenceDecoded);
    moduleWrapper->networkDecoder->inference(
            stream, cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded);
#endif

    float *miOutput = reinterpret_cast<float*>(outputImageBufferCu) + batchOffset;
    combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
            cacheWrapper->referenceDecoded.data(), cacheWrapper->queryDecoded.data(), miOutput,
            uint32_t(es), numLayersOutDecoder);
}
