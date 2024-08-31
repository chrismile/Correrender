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

#include <tiny-cuda-nn/evaluator.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <Utils/AppSettings.hpp>
#include <Utils/File/Archive.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Utils/DeviceThreadInfo.hpp>

#include "Volume/VolumeData.hpp"
#include "Volume/Cache/AuxiliaryMemoryToken.hpp"
#include "Calculators/MutualInformation.cuh"
#include "Calculators/VMLP/Format.hpp"
#include "TimeSeriesCorrelationRenderer.hpp"

using precision_t = tcnn::network_precision_t;

struct TinyCudaNNTimeSeriesModuleWrapper {
    nlohmann::json configGeneral;
    nlohmann::json configEncoder;
    nlohmann::json configDecoder;
    std::shared_ptr<tcnn::Network<float, precision_t>> networkEncoder;
    std::shared_ptr<tcnn::Evaluator<float, precision_t, precision_t>> evaluatorEncoder;
#if TCNN_HALF_PRECISION
    std::shared_ptr<tcnn::Network<precision_t, precision_t>> networkEncoderHalf;
    std::shared_ptr<tcnn::Evaluator<precision_t, precision_t, precision_t>> evaluatorEncoderHalf;
#endif
    std::shared_ptr<tcnn::Network<precision_t, precision_t>> networkDecoder;
    std::shared_ptr<tcnn::Evaluator<precision_t, precision_t, precision_t>> evaluatorDecoder;
};

struct TinyCudaNNTimeSeriesCacheWrapper {
    tcnn::GPUMatrix<float> referenceInput;
#if TCNN_HALF_PRECISION
    tcnn::GPUMatrix<precision_t> referenceInputHalf;
#endif
    tcnn::GPUMatrix<precision_t> referenceEncoded;
    tcnn::GPUMatrix<float> queryInput;
#if TCNN_HALF_PRECISION
    tcnn::GPUMatrix<precision_t> queryInputHalf;
#endif
    tcnn::GPUMatrix<precision_t> queryEncoded;
    tcnn::GPUMatrix<precision_t> symmetrizedQueryInput;
    tcnn::GPUMatrix<precision_t> referenceDecoded;
    tcnn::GPUMatrix<precision_t> queryDecoded;
    //AuxiliaryMemoryToken auxMemoryToken{};
};

void TimeSeriesCorrelationRenderer::initializeCuda() {
    cacheWrapper = std::make_shared<TinyCudaNNTimeSeriesCacheWrapper>();

    sgl::vk::Device* device = renderer->getDevice();
    if (device->getDeviceDriverId() != VK_DRIVER_ID_NVIDIA_PROPRIETARY
        || !sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::Logfile::get()->throwError(
                "Error in DeepLearningCudaCorrelationCalculator::DeepLearningCudaCorrelationCalculator: "
                "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
    }

    // e.g., 131072 for RTX 3090 (rounded up from 83968).
    auto deviceThreadInfo = sgl::getDeviceThreadInfo(renderer->getDevice());
    srnGpuBatchSize1DBase = int(deviceThreadInfo.numCoresTotal) * 8;
    if (!sgl::isPowerOfTwo(srnGpuBatchSize1DBase)) {
        srnGpuBatchSize1DBase = sgl::nextPowerOfTwo(srnGpuBatchSize1DBase);
    }
    srnGpuBatchSize1DBase = std::clamp(srnGpuBatchSize1DBase, 256, 131072);

    CUdevice cuDevice = 0;
    bool foundDevice = sgl::vk::getMatchingCudaDevice(renderer->getDevice(), &cuDevice);
    if (foundDevice) {
        CUresult cuResult;
        int computeCapabilityMajor = 7;
        cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
                &computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
        sgl::vk::checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");
        deviceSupporsFullyFusedMlp = computeCapabilityMajor >= 7;
    }

    if (!deviceSupporsFullyFusedMlp) {
        networkImplementation = TinyCudaNNNetworkImplementation::CUTLASS_MLP;
    }

    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamCreate(
            &stream, 0), "Error in cuStreamCreate: ");
}

void TimeSeriesCorrelationRenderer::cleanupCuda() {
    //if (cacheWrapper->auxMemoryToken) {
    //    volumeData->popAuxiliaryMemoryDevice(cacheWrapper->auxMemoryToken);
    //}

    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamDestroy(
            stream), "Error in cuStreamDestroy: ");
}

template<class T, class PARAMS_T> static void loadNetworkTimeSeries(
        std::shared_ptr<tcnn::Network<T, PARAMS_T>>& network,
        std::shared_ptr<tcnn::Evaluator<T, PARAMS_T, PARAMS_T>>& evaluator,
        const std::string& modelPath, const nlohmann::json& config, const sgl::ArchiveEntry& entry) {
    auto* header = reinterpret_cast<NetworkParametersHeader*>(entry.bufferData.get());
    uint8_t* paramsData = entry.bufferData.get() + sizeof(NetworkParametersHeader);
    uint32_t numParams = header->numParams;

    size_t sizePerEntry = header->format == NETWORK_PARAMS_FORMAT_FLOAT ? 4 : 2;
    if (numParams * sizePerEntry + sizeof(NetworkParametersHeader) != entry.bufferSize) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetworkTimeSeries: Invalid number of parameters for file size.");
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
    //std::shared_ptr<tcnn::Loss<precision_t>> loss{tcnn::create_loss<precision_t>(lossOpts)};
    //std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{tcnn::create_optimizer<precision_t>(optimizerOpts)};
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
    //network->set_params();
    evaluator = std::make_shared<tcnn::Evaluator<T, PARAMS_T, PARAMS_T>>(network);

    // Do we need padding because the output width is not a multiple of 16?
    if (network->output_width() != network->padded_output_width() && network->n_params() != numParams) {
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
                "Error in loadNetworkTimeSeries: Mismatching network parameter count (" + std::to_string(numParams)
                + " vs. " + std::to_string(network->n_params()) + ") for \"" + modelPath + "\".");
    }

#if TCNN_HALF_PRECISION
    if (header->format == NETWORK_PARAMS_FORMAT_FLOAT) {
        evaluator->set_params_full_precision(reinterpret_cast<float*>(paramsData), numParams, false);
    } else {
        evaluator->set_params(reinterpret_cast<precision_t*>(paramsData), numParams, false);
    }
#else
    if (header->format == NETWORK_PARAMS_FORMAT_FLOAT) {
        evaluator->set_params(reinterpret_cast<float*>(paramsData), numParams, false);
    } else {
        sgl::Logfile::get()->throwError(
                "Error in TinyCudaNNCorrelationCalculator::loadNetworkTimeSeries: Half precision build was disabled.");
    }
#endif

    if (network->output_width() != network->padded_output_width() && network->n_params() != numParams) {
        delete[] paramsData;
    }
}

void TimeSeriesCorrelationRenderer::unloadModel() {
    moduleWrapper = {};
    cacheWrapper = {};
}

void TimeSeriesCorrelationRenderer::loadModelFromFile(const std::string& modelPath) {
    unloadModel();
    if (modelPath.empty()) {
        return;
    }
    moduleWrapper = std::make_shared<TinyCudaNNTimeSeriesModuleWrapper>();
    cacheWrapper = std::make_shared<TinyCudaNNTimeSeriesCacheWrapper>();

    std::unordered_map<std::string, sgl::ArchiveEntry> archiveFiles;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadAllFilesFromArchive(modelPath, archiveFiles, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNCorrelationCalculator::loadModelFromFile: Could not load data from model \""
                + modelPath + "\".");
        return;
    }

    // A global configuration file is optional.
    auto itConfig = archiveFiles.find("config.json");
    if (itConfig != archiveFiles.end()) {
        const auto& entry = itConfig->second;
        moduleWrapper->configGeneral = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entry.bufferData.get()), entry.bufferSize));

        auto symmetrizerTypeName = moduleWrapper->configGeneral.value(
                "symmetrizer_type", SYMMETRIZER_TYPE_SHORT_NAMES[0]);
        bool foundSymmetrizerType = false;
        for (int i = 0; i < IM_ARRAYSIZE(SYMMETRIZER_TYPE_SHORT_NAMES); i++) {
            if (SYMMETRIZER_TYPE_SHORT_NAMES[i] == symmetrizerTypeName) {
                symmetrizerType = SymmetrizerType(i);
                foundSymmetrizerType = true;
                break;
            }
        }
        if (!foundSymmetrizerType) {
            sgl::Logfile::get()->writeError(
                    "Error in TinyCudaNNCorrelationCalculator::loadModelFromFile: Invalid symmetrizer type \""
                    + symmetrizerTypeName + "\".");
            return;
        }

        isMutualInformationData = moduleWrapper->configGeneral.value("is_mutual_information", true);
    }

    // Encoder and decoder configuration files are mandatory.
    auto itConfigEncoder = archiveFiles.find("config_encoder.json");
    auto itConfigDecoder = archiveFiles.find("config_decoder.json");
    if (itConfigEncoder != archiveFiles.end() && itConfigDecoder != archiveFiles.end()) {
        const auto& entryEncoder = itConfigEncoder->second;
        const auto& entryDecoder = itConfigDecoder->second;
        moduleWrapper->configEncoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entryEncoder.bufferData.get()), entryEncoder.bufferSize));
        moduleWrapper->configDecoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entryDecoder.bufferData.get()), entryDecoder.bufferSize));
    } else {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNCorrelationCalculator::loadModelFromFile: Could not load encoder or decoder "
                "configuration from model \"" + modelPath + "\".");
        return;
    }

    isInputEncodingIdentity = false;
    bool hasInputEncoding = moduleWrapper->configEncoder.find("encoding") != moduleWrapper->configEncoder.end();
    auto encodingOpts = moduleWrapper->configEncoder.value("encoding", nlohmann::json::object());
    if (hasInputEncoding) {
        if (encodingOpts.value("otype", "Identity") == "Identity"
            && encodingOpts.value("scale", 1.0f) == 1.0f
            && encodingOpts.value("offset", 0.0f) == 0.0f) {
            isInputEncodingIdentity = true;
        }
    }

    // Set input/output layer configurations for both networks.
    auto encoderNetworkOpts = moduleWrapper->configEncoder.value("network", nlohmann::json::object());
    auto decoderNetworkOpts = moduleWrapper->configDecoder.value("network", nlohmann::json::object());
    // mlp_fused_forward needs multiple of 16 for number of input layers.
    int numInputLayers = 16;
    if (!isInputEncodingIdentity) {
        numInputLayers = 3;
    }
    moduleWrapper->configEncoder["network"]["n_input_dims"] = numInputLayers;
    moduleWrapper->configDecoder["network"]["n_output_dims"] = 1;
    if (encoderNetworkOpts.find("n_output_dims") == encoderNetworkOpts.end()) {
        moduleWrapper->configEncoder["network"]["n_output_dims"] = moduleWrapper->configEncoder["network"]["n_neurons"];
    }
    uint32_t symmetrizerFactor = symmetrizerType == SymmetrizerType::AddDiff ? 2 : 1;
    if (decoderNetworkOpts.find("n_input_dims") == decoderNetworkOpts.end()) {
        uint32_t encoderOutputDims = moduleWrapper->configEncoder["network"].value("n_output_dims", 0);
        moduleWrapper->configDecoder["network"]["n_input_dims"] = encoderOutputDims * symmetrizerFactor;
    }

    const char* networkTypeName = TINY_CUDA_NN_NETWORK_IMPLEMENTATION_NAMES[int(networkImplementation)];
    moduleWrapper->configEncoder["network"]["otype"] = networkTypeName;
    moduleWrapper->configDecoder["network"]["otype"] = networkTypeName;

    auto itNetworkEncoder = archiveFiles.find("network_encoder.bin");
    auto itNetworkDecoder = archiveFiles.find("network_decoder.bin");
    if (itNetworkEncoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNCorrelationCalculator::loadModelFromFile: Missing network_encoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    if (itNetworkDecoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNCorrelationCalculator::loadModelFromFile: Missing network_decoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    moduleWrapper->networkEncoder = {};
    moduleWrapper->evaluatorEncoder = {};
#if TCNN_HALF_PRECISION
    moduleWrapper->networkEncoderHalf = {};
    moduleWrapper->evaluatorEncoderHalf = {};
#endif
    moduleWrapper->networkDecoder = {};
    moduleWrapper->evaluatorDecoder = {};
#if TCNN_HALF_PRECISION
    if (hasInputEncoding && !isInputEncodingIdentity) {
#endif
        loadNetworkTimeSeries(
                moduleWrapper->networkEncoder, moduleWrapper->evaluatorEncoder, modelPath,
                moduleWrapper->configEncoder, itNetworkEncoder->second);
#if TCNN_HALF_PRECISION
    } else {
        loadNetworkTimeSeries(
                moduleWrapper->networkEncoderHalf, moduleWrapper->evaluatorEncoderHalf, modelPath,
                moduleWrapper->configEncoder, itNetworkEncoder->second);
    }
#endif
    loadNetworkTimeSeries(
            moduleWrapper->networkDecoder, moduleWrapper->evaluatorDecoder, modelPath,
            moduleWrapper->configDecoder, itNetworkDecoder->second);

    // numLayersOutEncoder == numLayersInDecoder when symmetrizer is sum operation.
#if TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        numLayersInEncoder = uint32_t(moduleWrapper->networkEncoderHalf->input_width());
        numLayersOutEncoder = uint32_t(moduleWrapper->networkEncoderHalf->padded_output_width());
    } else {
        numLayersInEncoder = uint32_t(moduleWrapper->networkEncoder->input_width());
        numLayersOutEncoder = uint32_t(moduleWrapper->networkEncoder->padded_output_width());
    }
#else
    numLayersInEncoder = uint32_t(moduleWrapper->networkEncoder->input_width());
    numLayersOutEncoder = uint32_t(moduleWrapper->networkEncoder->padded_output_width());
#endif

    numLayersInDecoder = uint32_t(moduleWrapper->networkDecoder->input_width());
#if TCNN_HALF_PRECISION
    numLayersOutDecoder = uint32_t(moduleWrapper->networkDecoder->padded_output_width());
#else
    // tcnn::DifferentiableObject<T,PARAMS_T,COMPUTE_T>::inference checks output.m() == output_width().
    // For some reason, there is an incompatibility for the CutlassMLP class.
    numLayersOutDecoder = uint32_t(moduleWrapper->networkDecoder->output_width());
#endif

    if (numLayersOutEncoder * symmetrizerFactor != numLayersInDecoder) {
        sgl::Logfile::get()->throwError(
                "Error in TinyCudaNNCorrelationCalculator::loadModelFromFile: Mismatch between encoder output and "
                "decoder input dimensions.");
    }

    if (timeSeriesMetadata.window > 0) {
        numWindows = timeSeriesMetadata.time;
    }
    recreateCache(srnGpuBatchSize1DBase);
    recomputeCorrelationMatrix();
}

void TimeSeriesCorrelationRenderer::recreateCache(int batchSize) {
    cacheWrapper->referenceInput = tcnn::GPUMatrix<float>();
#if TCNN_HALF_PRECISION
    cacheWrapper->referenceInputHalf = tcnn::GPUMatrix<precision_t>();
#endif
    cacheWrapper->referenceEncoded = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->queryInput = tcnn::GPUMatrix<float>();
#if TCNN_HALF_PRECISION
    cacheWrapper->queryInputHalf = tcnn::GPUMatrix<precision_t>();
#endif
    cacheWrapper->queryEncoded = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->symmetrizedQueryInput = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->referenceDecoded = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->queryDecoded = tcnn::GPUMatrix<precision_t>();
    //if (cacheWrapper->auxMemoryToken) {
    //    volumeData->popAuxiliaryMemoryDevice(cacheWrapper->auxMemoryToken);
    //}

    // mlp_fused_forward needs multiple of 16 for number of input layers.
    uint32_t numInputLayers = 16;
    if (!isInputEncodingIdentity) {
        numInputLayers = 3;
    }
    uint32_t referenceInputBatchSize =
            sgl::uiceil(uint32_t(numWindows), tcnn::batch_size_granularity) * tcnn::batch_size_granularity;
#if TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        cacheWrapper->referenceInputHalf = tcnn::GPUMatrix<precision_t>(numInputLayers, referenceInputBatchSize);
        cacheWrapper->queryInputHalf = tcnn::GPUMatrix<precision_t>(numInputLayers, batchSize);
    }
#endif
    cacheWrapper->referenceInput = tcnn::GPUMatrix<float>(numInputLayers, referenceInputBatchSize);
    cacheWrapper->queryInput = tcnn::GPUMatrix<float>(numInputLayers, batchSize);
    cacheWrapper->referenceEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, referenceInputBatchSize);
    cacheWrapper->queryEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, batchSize);
    cacheWrapper->symmetrizedQueryInput = tcnn::GPUMatrix<precision_t>(numLayersInDecoder, batchSize);
    cacheWrapper->referenceDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, batchSize);
    cacheWrapper->queryDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, batchSize);

    size_t auxBuffersSizeInBytes = 0;
#if TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        auxBuffersSizeInBytes += size_t(cacheWrapper->referenceInputHalf.n_bytes());
        auxBuffersSizeInBytes += size_t(cacheWrapper->queryInputHalf.n_bytes());
    }
#endif
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceInput.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryInput.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceEncoded.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryEncoded.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->symmetrizedQueryInput.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceDecoded.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryDecoded.n_bytes());
    //if (volumeData) {
    //    cacheWrapper->auxMemoryToken = volumeData->pushAuxiliaryMemoryDevice(auxBuffersSizeInBytes);
    //}
}

void TimeSeriesCorrelationRenderer::runInferenceReference() {
#if TCNN_HALF_PRECISION
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

template<class T> __global__ void symmetrizerTimeSeriesAdd(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t batchOffset, uint32_t numWindows, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef =
            ((batchOffset + globalThreadIdx / numChannels) % numWindows) * numChannels // Reference offset
            + (globalThreadIdx % numChannels); // Channel index
    outputValues[readOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerTimeSeriesAddDiff_Add(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t batchOffset, uint32_t numWindows, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t readOffsetRef =
            ((batchOffset + globalThreadIdx / numChannels) % numWindows) * numChannels // Reference offset
            + channelIdx; // Channel index
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + channelIdx;
    outputValues[writeOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerTimeSeriesAddDiff_Diff(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t batchOffset, uint32_t numWindows, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t readOffsetRef =
            ((batchOffset + globalThreadIdx / numChannels) % numWindows) * numChannels // Reference offset
            + channelIdx; // Channel index
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + numChannels + channelIdx;
    outputValues[writeOffset] = absT(referenceValues[readOffsetRef] - queryValues[readOffset]);
}

template<class T> __global__ void symmetrizerTimeSeriesMul(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t batchOffset, uint32_t numWindows, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef =
            ((batchOffset + globalThreadIdx / numChannels) % numWindows) * numChannels // Reference offset
            + (globalThreadIdx % numChannels); // Channel index
    outputValues[readOffset] = referenceValues[readOffsetRef] * queryValues[readOffset];
}

template<class T> void symmetrizerTimeSeries(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ symmetrizedValues,
        uint32_t batchOffset, uint32_t batchSize, uint32_t numWindows, uint32_t numLayersOutEncoder,
        SymmetrizerType symmetrizerType, CUstream stream) {
    constexpr uint32_t blockSize = 256;
    const uint32_t numBlocks = sgl::uiceil(batchSize * numLayersOutEncoder, blockSize);
    if (symmetrizerType == SymmetrizerType::Add) {
        symmetrizerTimeSeriesAdd<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, batchOffset, numWindows, numLayersOutEncoder);
    } else if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerTimeSeriesAddDiff_Add<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, batchOffset, numWindows, numLayersOutEncoder);
        symmetrizerTimeSeriesAddDiff_Diff<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, batchOffset, numWindows, numLayersOutEncoder);
    } else if (symmetrizerType == SymmetrizerType::Mul) {
        symmetrizerTimeSeriesMul<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, batchOffset, numWindows, numLayersOutEncoder);
    }
}

void TimeSeriesCorrelationRenderer::runInferenceBatch(uint32_t batchOffset, uint32_t batchSize)  {
#if TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        uint32_t arraySize = cacheWrapper->queryInputHalf.n() * cacheWrapper->queryInputHalf.m();
        convertFloatToHalfArray<<<sgl::uiceil(arraySize, 256), 256, 0, stream>>>(
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

    symmetrizerTimeSeries(
            cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncoded.data(),
            cacheWrapper->symmetrizedQueryInput.data(),
            batchOffset, batchSize, numWindows, numLayersOutEncoder, symmetrizerType, stream);

#if TCNN_HALF_PRECISION
    moduleWrapper->networkDecoder->inference_mixed_precision(
            stream, cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded);
#else
    moduleWrapper->networkDecoder->inference(
            stream, cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded);
#endif

    float* miOutput = reinterpret_cast<float*>(outputImageBufferCu) + batchOffset;
    if (isMutualInformationData) {
        copyDecoderOutputSrnMutualInformation<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->queryDecoded.data(), miOutput, numLayersOutDecoder);
    } else if (calculateAbsoluteValue) {
        copyDecoderOutputSrnCorrelationCoefficientAbs<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->queryDecoded.data(), miOutput, numLayersOutDecoder);
    } else {
        copyDecoderOutputSrnCorrelationCoefficient<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->queryDecoded.data(), miOutput, numLayersOutDecoder);
    }
}

__global__ void writeNetworkInputsReference(
        uint32_t numWindows, uint32_t seriesIdx, uint32_t stride, float* __restrict__ outputBuffer) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < numWindows) {
        uint32_t writeOffset = globalThreadIdx * stride;
        outputBuffer[writeOffset] = float(seriesIdx);
        outputBuffer[writeOffset + 1] = float(globalThreadIdx) / float(numWindows - 1) * 2.0f - 1.0f;
        outputBuffer[writeOffset + 2] = 0;
    }
}

__global__ void writeNetworkInputs(
        uint32_t numSeries, uint32_t numWindows, uint32_t batchOffset, uint32_t batchSize,
        float* __restrict__ outputBuffer, uint32_t stride) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < batchSize) {
        uint32_t writeOffset = globalThreadIdx * stride;
        uint32_t readOffset = globalThreadIdx + batchOffset;
        uint32_t seriesIdx = readOffset / numWindows;
        uint32_t timeIdx = readOffset % numWindows;
        outputBuffer[writeOffset] = float(seriesIdx);
        outputBuffer[writeOffset + 1] = float(timeIdx) / float(numWindows - 1) * 2.0f - 1.0f;
        outputBuffer[writeOffset + 2] = 0;
    }
}

void TimeSeriesCorrelationRenderer::recomputeCorrelationMatrixTcnn() {
    if (!getIsModuleLoaded()) {
        return;
    }

    int gpuBatchSize1D = srnGpuBatchSize1DBase;

    sgl::vk::Swapchain* swapchain = sgl::AppSettings::get()->getSwapchain();
    uint32_t frameIndex = swapchain ? swapchain->getImageIndex() : 0;
    size_t numSwapchainImages = swapchain ? swapchain->getNumImages() : 1;
    if (numSwapchainImages != cachedNumSwapchainImages) {
        cachedNumSwapchainImages = numSwapchainImages;
        sgl::vk::Device* device = renderer->getDevice();
        timelineValue = 0;
        postRenderCommandBuffers.clear();
        sgl::vk::CommandPoolType commandPoolType;
        commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        for (size_t frameIdx = 0; frameIdx < numSwapchainImages; frameIdx++) {
            postRenderCommandBuffers.push_back(std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType));
        }
        vulkanFinishedSemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE, timelineValue);
        cudaFinishedSemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE, timelineValue);
    }
    timelineValue++;

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    sgl::vk::CommandBufferPtr commandBufferRender = renderer->getCommandBuffer();
    vulkanFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
    commandBufferRender->pushSignalSemaphore(vulkanFinishedSemaphore);
    renderer->endCommandBuffer();

    renderer->submitToQueue();

    vulkanFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);

    uint32_t alignmentVec4 = sgl::uiceil(getInputChannelAlignment(), 4);

    uint32_t srnStride = getSrnStride();
    writeNetworkInputsReference<<<sgl::uiceil(numWindows, 256), 256, 0, stream>>>(
            numWindows, sidxRef, srnStride, cacheWrapper->referenceInput.data());
    runInferenceReference();

    uint32_t numSliceEntries = uint32_t(timeSeriesMetadata.samples) * uint32_t(numWindows);
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(gpuBatchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * gpuBatchSize1D;
        uint32_t batchSize = std::min(uint32_t(gpuBatchSize1D), numSliceEntries - batchOffset);
        writeNetworkInputs<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                timeSeriesMetadata.samples, numWindows, batchOffset, batchSize,
                cacheWrapper->queryInput.data(), srnStride);
        runInferenceBatch(batchOffset, batchSize);
    }

    cudaFinishedSemaphore->signalSemaphoreCuda(stream, timelineValue);
    cudaFinishedSemaphore->setWaitSemaphoreValue(timelineValue);
    sgl::vk::CommandBufferPtr postRenderCommandBuffer = postRenderCommandBuffers.at(frameIndex);
    renderer->pushCommandBuffer(postRenderCommandBuffer);
    renderer->beginCommandBuffer();
    postRenderCommandBuffer->pushWaitSemaphore(
            cudaFinishedSemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}
