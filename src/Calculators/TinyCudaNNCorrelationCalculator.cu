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

#ifdef TEST_INFERENCE_SPEED
#include <cuda_profiler_api.h>
#endif

#include <tiny-cuda-nn/evaluator.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/networks/fully_fused_mlp.h>

#include <Math/Math.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Archive.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Volume/VolumeData.hpp"
#include "MutualInformation.cuh"
#include "TinyCudaNNCorrelationCalculator.hpp"

using precision_t = tcnn::network_precision_t;

struct TinyCudaNNModuleWrapper {
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

struct TinyCudaNNCacheWrapper {
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
    tcnn::GPUMatrix<precision_t> symmetrizedReferenceInput;
    tcnn::GPUMatrix<precision_t> symmetrizedQueryInput;
    tcnn::GPUMatrix<precision_t> referenceDecoded;
    tcnn::GPUMatrix<precision_t> queryDecoded;
    AuxiliaryMemoryToken auxMemoryToken{};
};

TinyCudaNNCorrelationCalculator::TinyCudaNNCorrelationCalculator(sgl::vk::Renderer* renderer)
        : DeepLearningCudaCorrelationCalculator("tiny-cuda-nn", "tinyCudaNN", renderer) {
    cacheWrapper = std::make_shared<TinyCudaNNCacheWrapper>();

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
}

TinyCudaNNCorrelationCalculator::~TinyCudaNNCorrelationCalculator() {
    if (cacheWrapper->auxMemoryToken) {
        volumeData->popAuxiliaryMemoryDevice(cacheWrapper->auxMemoryToken);
    }
}

void TinyCudaNNCorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    DeepLearningCudaCorrelationCalculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::TINY_CUDA_NN);
    }
}

void TinyCudaNNCorrelationCalculator::renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) {
    DeepLearningCudaCorrelationCalculator::renderGuiImplAdvanced(propertyEditor);
    if (deviceSupporsFullyFusedMlp && propertyEditor.addCombo(
            "Network", (int*)&networkImplementation,
            TINY_CUDA_NN_NETWORK_IMPLEMENTATION_UI_NAMES, IM_ARRAYSIZE(TINY_CUDA_NN_NETWORK_IMPLEMENTATION_UI_NAMES))) {
        if (sgl::FileUtils::get()->exists(modelFilePath) && !sgl::FileUtils::get()->isDirectory(modelFilePath)) {
            loadModelFromFile(modelFilePath);
        }
        dirty = true;
    }
}

void TinyCudaNNCorrelationCalculator::setSettings(const SettingsMap& settings) {
    std::string networkImplementationString;
    if (settings.getValueOpt("network_implementation", networkImplementationString)) {
        for (int i = 0; i < IM_ARRAYSIZE(TINY_CUDA_NN_NETWORK_IMPLEMENTATION_NAMES); i++) {
            if (networkImplementationString == TINY_CUDA_NN_NETWORK_IMPLEMENTATION_NAMES[i]) {
                networkImplementation = TinyCudaNNNetworkImplementation(i);
                break;
            }
        }
        if (!deviceSupporsFullyFusedMlp) {
            networkImplementation = TinyCudaNNNetworkImplementation::CUTLASS_MLP;
        }
        dirty = true;
    }
    DeepLearningCudaCorrelationCalculator::setSettings(settings);
}

void TinyCudaNNCorrelationCalculator::getSettings(SettingsMap& settings) {
    DeepLearningCudaCorrelationCalculator::getSettings(settings);
    settings.addKeyValue(
            "network_implementation", TINY_CUDA_NN_NETWORK_IMPLEMENTATION_NAMES[int(networkImplementation)]);
}

template<class T, class PARAMS_T> static void loadNetwork(
        std::shared_ptr<tcnn::Network<T, PARAMS_T>>& network,
        std::shared_ptr<tcnn::Evaluator<T, PARAMS_T, PARAMS_T>>& evaluator,
        const std::string& modelPath, const nlohmann::json& config, const sgl::ArchiveEntry& entry) {
    auto* header = reinterpret_cast<NetworkParametersHeader*>(entry.bufferData.get());
    uint8_t* paramsData = entry.bufferData.get() + sizeof(NetworkParametersHeader);
    uint32_t numParams = header->numParams;

    size_t sizePerEntry = header->format == NETWORK_PARAMS_FORMAT_FLOAT ? 4 : 2;
    if (numParams * sizePerEntry + sizeof(NetworkParametersHeader) != entry.bufferSize) {
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
                "Error in loadNetwork: Mismatching network parameter count (" + std::to_string(numParams)
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
                "Error in TinyCudaNNCorrelationCalculator::loadNetwork: Half precision build was disabled.");
    }
#endif

    if (network->output_width() != network->padded_output_width() && network->n_params() != numParams) {
        delete[] paramsData;
    }
}

void TinyCudaNNCorrelationCalculator::loadModelFromFile(const std::string& modelPath) {
    moduleWrapper = std::make_shared<TinyCudaNNModuleWrapper>();
    cacheWrapper = std::make_shared<TinyCudaNNCacheWrapper>();

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

        auto networkTypeName = moduleWrapper->configGeneral.value(
                "network_type", NETWORK_TYPE_SHORT_NAMES[0]);
        bool foundNetworkType = false;
        for (int i = 0; i < IM_ARRAYSIZE(NETWORK_TYPE_SHORT_NAMES); i++) {
            if (NETWORK_TYPE_SHORT_NAMES[i] == networkTypeName) {
                networkType = NetworkType(i);
                foundNetworkType = true;
                break;
            }
        }
        if (!foundNetworkType && networkTypeName == "MINE_SRN") {
            networkType = NetworkType::SRN_MINE;
            foundNetworkType = true;
        }
        if (!foundNetworkType) {
            sgl::Logfile::get()->writeError(
                    "Error in TinyCudaNNCorrelationCalculator::loadModelFromFile: Invalid network type \""
                    + networkTypeName + "\".");
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
        if (networkType == NetworkType::MINE) {
            numInputLayers = 4;
        } else {
            numInputLayers = 3;
        }
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
        loadNetwork(
                moduleWrapper->networkEncoder, moduleWrapper->evaluatorEncoder, modelPath,
                moduleWrapper->configEncoder, itNetworkEncoder->second);
#if TCNN_HALF_PRECISION
    } else {
        loadNetwork(
                moduleWrapper->networkEncoderHalf, moduleWrapper->evaluatorEncoderHalf, modelPath,
                moduleWrapper->configEncoder, itNetworkEncoder->second);
    }
#endif
    loadNetwork(
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

    cacheNeedsRecreate = true;
}

void TinyCudaNNCorrelationCalculator::recreateCache(int batchSize) {
    int cs = networkType == NetworkType::MINE ? getCorrelationMemberCount() : 1;

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
    cacheWrapper->symmetrizedReferenceInput = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->symmetrizedQueryInput = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->referenceDecoded = tcnn::GPUMatrix<precision_t>();
    cacheWrapper->queryDecoded = tcnn::GPUMatrix<precision_t>();
    if (cacheWrapper->auxMemoryToken) {
        volumeData->popAuxiliaryMemoryDevice(cacheWrapper->auxMemoryToken);
    }

    // mlp_fused_forward needs multiple of 16 for number of input layers.
    uint32_t numInputLayers = 16;
    if (!isInputEncodingIdentity) {
        if (networkType == NetworkType::MINE) {
            numInputLayers = 4;
        } else {
            numInputLayers = 3;
        }
    }
    uint32_t referenceInputBatchSize =
            sgl::uiceil(uint32_t(cs), tcnn::batch_size_granularity) * tcnn::batch_size_granularity;
#if TCNN_HALF_PRECISION
    if (moduleWrapper->networkEncoderHalf) {
        cacheWrapper->referenceInputHalf = tcnn::GPUMatrix<precision_t>(numInputLayers, referenceInputBatchSize);
        cacheWrapper->queryInputHalf = tcnn::GPUMatrix<precision_t>(numInputLayers, uint32_t(cs) * batchSize);
    }
#endif
    cacheWrapper->referenceInput = tcnn::GPUMatrix<float>(numInputLayers, referenceInputBatchSize);
    cacheWrapper->queryInput = tcnn::GPUMatrix<float>(numInputLayers, uint32_t(cs) * batchSize);
    cacheWrapper->referenceEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, referenceInputBatchSize);
    cacheWrapper->queryEncoded = tcnn::GPUMatrix<precision_t>(numLayersOutEncoder, uint32_t(cs) * batchSize);
    cacheWrapper->symmetrizedReferenceInput = tcnn::GPUMatrix<precision_t>(numLayersInDecoder, uint32_t(cs) * batchSize);
    cacheWrapper->symmetrizedQueryInput = tcnn::GPUMatrix<precision_t>(numLayersInDecoder, uint32_t(cs) * batchSize);
    cacheWrapper->referenceDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, uint32_t(cs) * batchSize);
    cacheWrapper->queryDecoded = tcnn::GPUMatrix<precision_t>(numLayersOutDecoder, uint32_t(cs) * batchSize);

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
    auxBuffersSizeInBytes += size_t(cacheWrapper->symmetrizedReferenceInput.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->symmetrizedQueryInput.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceDecoded.n_bytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryDecoded.n_bytes());
    cacheWrapper->auxMemoryToken = volumeData->pushAuxiliaryMemoryDevice(auxBuffersSizeInBytes);
}

CUdeviceptr TinyCudaNNCorrelationCalculator::getReferenceInputPointer()  {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->referenceInput.data());
}

CUdeviceptr TinyCudaNNCorrelationCalculator::getQueryInputPointer()  {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->queryInput.data());
}

void TinyCudaNNCorrelationCalculator::runInferenceReference() {
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

    /*float* refInput = new float[3];
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            refInput, (CUdeviceptr)cacheWrapper->referenceInput.data(),
            sizeof(float) * 3, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "refInput:" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << float(refInput[i]);
        if (i != 2) {
            std::cout << ", ";
        } else {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] refInput;*/

    /*int copySize = batchSize * 40;
    int testSize = 160;
    __half* dataHalf = new __half[copySize];
    dataHalf[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataHalf, (CUdeviceptr)cacheWrapper->queryInputHalf.data(),
            sizeof(__half) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "queryInputHalf:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << float(dataHalf[i * numLayersInEncoder * uint32_t(es)]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataHalf;*/

    /*int sizeDataEnc = 4 * 64;
    auto* dataEnc = new __half[sizeDataEnc];
    cudaMemcpy(dataEnc, cacheWrapper->referenceEncoded.data(), sizeDataEnc * sizeof(__half), cudaMemcpyDeviceToHost);
    std::cout << "dataEnc:" << std::endl;
    for (int i = 0; i < sizeDataEnc; i++) {
        std::cout << float(dataEnc[i]);
        if (i != sizeDataEnc - 1) {
            std::cout << ", ";
        }
        if (i % 16 == 15 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataEnc;*/
}

void TinyCudaNNCorrelationCalculator::runInferenceBatch(uint32_t batchOffset, uint32_t batchSize)  {
    int cs = getCorrelationMemberCount();

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

    /*int copySize = 3 * 800;
    int testSize = 160;
    float* dataFloat = new float[copySize];
    dataFloat[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataFloat, (CUdeviceptr)cacheWrapper->queryInput.data(),
            sizeof(float) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "queryInput:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << dataFloat[i];
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataFloat;*/

    /*int copySize = batchSize * 40;
    int testSize = 160;
    float* dataFloat = new float[copySize];
    dataFloat[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataFloat, (CUdeviceptr)cacheWrapper->queryInput.data(),
            sizeof(float) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "queryInput:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        //std::cout << dataFloat[i * numLayersInEncoder * uint32_t(es)];
        std::cout << dataFloat[i];
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataFloat;
    __half* dataHalf = new __half[copySize];
    dataHalf = new __half[copySize];
    dataHalf[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataHalf, (CUdeviceptr)cacheWrapper->queryEncoded.data(),
            sizeof(__half) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "queryEncoded:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        //std::cout << float(dataHalf[i * numLayersOutEncoder * uint32_t(es)]);
        std::cout << float(dataHalf[i]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    std::cout << "queryEncoded(2):" << std::endl;
    for (int i = 0; i < testSize; i++) {
        //std::cout << float(dataHalf[i * numLayersOutEncoder * uint32_t(es) + numLayersOutEncoder]);
        std::cout << float(dataHalf[i]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataHalf;*/

    /*tcnn::FullyFusedMLP<precision_t, 128>* fmlp = (tcnn::FullyFusedMLP<precision_t, 128>*)moduleWrapper->networkEncoderHalf.get();
    const auto& tensor = fmlp->weight_matrix_at(tcnn::WeightUsage::Inference, 1);

    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemsetD16Async(
            (CUdeviceptr)tensor.data(), 0, 1, stream), "Error in cuMemsetD16Async: ");*/

    /*sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
            (CUdeviceptr)cacheWrapper->queryEncodedPermuted.data(),
            (CUdeviceptr)cacheWrapper->queryEncoded.data(),
            cacheWrapper->queryEncoded.n() * cacheWrapper->queryEncoded.m(),
            stream), "Error in cuMemcpyAsync: ");*/

    if (networkType == NetworkType::MINE) {
        uint32_t* permutationIndicesBuffer = reinterpret_cast<uint32_t*>(permutationIndicesBufferCu);
        generateRandomPermutations<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                permutationIndicesBuffer, uint32_t(cs), batchOffset);
        //randomShuffleFisherYates<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
        //        cacheWrapper->queryEncodedPermuted.data(), cacheWrapper->queryEncoded.data(),
        //        permutationIndicesBuffer, uint32_t(es), numLayersOutEncoder);
        symmetrizer(
                cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncoded.data(),
                cacheWrapper->symmetrizedReferenceInput.data(), cacheWrapper->symmetrizedQueryInput.data(),
                permutationIndicesBuffer, batchSize, uint32_t(cs), numLayersOutEncoder, symmetrizerType, stream);
    } else {
        symmetrizerSrn(
                cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncoded.data(),
                cacheWrapper->symmetrizedQueryInput.data(),
                batchSize, numLayersOutEncoder, symmetrizerType, stream);
    }

    /*int testSize = 10 * es;
    auto* dataUint32 = new uint32_t[batchSize * es];
    dataUint32[0] = 1000;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataUint32, (CUdeviceptr)permutationIndicesBuffer,
            sizeof(uint32_t) * batchSize * es, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "permutationIndicesBuffer:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << dataUint32[i];
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % es == (es - 1) && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataUint32;*/

    /*dataHalf = new __half[copySize];
    dataHalf[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataHalf, (CUdeviceptr)cacheWrapper->queryEncodedPermuted.data(),
            sizeof(__half) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "queryEncodedPermuted:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << float(dataHalf[i * numLayersOutEncoder * uint32_t(es)]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataHalf;*/

    /*symmetrizerAdd<<<sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, 256), 256, 0, stream>>>(
            cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncoded.data(),
            cacheWrapper->symmetrizedReferenceInput.data(), uint32_t(es), numLayersOutEncoder);
    //symmetrizerAdd<<<sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, 256), 256, 0, stream>>>(
    //        cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncodedPermuted.data(),
    //        cacheWrapper->symmetrizedQueryInput.data(), uint32_t(es), numLayersOutEncoder);
    symmetrizerAddPermuted<<<sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, 256), 256, 0, stream>>>(
            cacheWrapper->referenceEncoded.data(), cacheWrapper->queryEncoded.data(),
            cacheWrapper->symmetrizedQueryInput.data(), permutationIndicesBuffer,
            uint32_t(es), numLayersOutEncoder);*/

    /*dataHalf = new __half[copySize];
    dataHalf[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataHalf, (CUdeviceptr)cacheWrapper->symmetrizedReferenceInput.data(),
            sizeof(__half) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "symmetrizedReferenceInput:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << float(dataHalf[i]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataHalf, (CUdeviceptr)cacheWrapper->symmetrizedQueryInput.data(),
            sizeof(__half) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "symmetrizedQueryInput:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << float(dataHalf[i]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataHalf;*/

#if TCNN_HALF_PRECISION
    if (networkType == NetworkType::MINE) {
        moduleWrapper->networkDecoder->inference_mixed_precision(
                stream, cacheWrapper->symmetrizedReferenceInput, cacheWrapper->referenceDecoded);
    }
    moduleWrapper->networkDecoder->inference_mixed_precision(
            stream, cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded);
#else
    if (networkType == NetworkType::MINE) {
        moduleWrapper->networkDecoder->inference(
                stream, cacheWrapper->symmetrizedReferenceInput, cacheWrapper->referenceDecoded);
    }
    moduleWrapper->networkDecoder->inference(
            stream, cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded);
#endif

    /*dataHalf = new __half[copySize];
    dataHalf[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataHalf, (CUdeviceptr)cacheWrapper->queryDecoded.data(),
            sizeof(__half) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "queryDecoded:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << float(dataHalf[i * numLayersOutDecoder * uint32_t(es)]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            dataHalf, (CUdeviceptr)cacheWrapper->referenceDecoded.data(),
            sizeof(__half) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "referenceDecoded:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << float(dataHalf[i * numLayersOutDecoder * uint32_t(es)]);
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] dataHalf;*/

    float* miOutput = reinterpret_cast<float*>(outputImageBufferCu) + batchOffset;
    if (networkType == NetworkType::MINE) {
        combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                cacheWrapper->referenceDecoded.data(), cacheWrapper->queryDecoded.data(), miOutput,
                uint32_t(cs), numLayersOutDecoder);
    } else {
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

    /*copySize = batchSize;
    float* data = new float[copySize];
    data[0] = 1000.0f;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            data, (CUdeviceptr)miOutput,
            sizeof(float) * copySize, stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "miOutput:" << std::endl;
    for (int i = 0; i < testSize; i++) {
        std::cout << data[i];
        if (i != testSize - 1) {
            std::cout << ", ";
        }
        if (i % 20 == 19 && i != 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    delete[] data;

    std::cout << std::endl << "END" << std::endl << std::endl;*/
}

void TinyCudaNNCorrelationCalculator::callbackBeginCompute() {
#ifdef TEST_INFERENCE_SPEED
    cudaProfilerStart();
#endif
}

void TinyCudaNNCorrelationCalculator::callbackEndCompute() {
#ifdef TEST_INFERENCE_SPEED
    cudaProfilerStop();
#endif
}
