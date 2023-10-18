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
#include "QuickMLPCorrelationCalculator.hpp"

struct QuickMLPModuleWrapper {
    nlohmann::json configGeneral;
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
    AuxiliaryMemoryToken auxMemoryToken{};
};

QuickMLPCorrelationCalculator::QuickMLPCorrelationCalculator(sgl::vk::Renderer* renderer)
        : DeepLearningCudaCorrelationCalculator("QuickMLP", "quickMLP", renderer) {
    cacheWrapper = std::make_shared<QuickMLPCacheWrapper>();
}

QuickMLPCorrelationCalculator::~QuickMLPCorrelationCalculator() {
    if (cacheWrapper->auxMemoryToken) {
        volumeData->popAuxiliaryMemoryDevice(cacheWrapper->auxMemoryToken);
    }
}

void QuickMLPCorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    DeepLearningCudaCorrelationCalculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::QUICK_MLP);
    }
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

    auto* header = reinterpret_cast<NetworkParametersHeader*>(entry.bufferData.get());
    uint8_t* paramsDataHost = entry.bufferData.get() + sizeof(NetworkParametersHeader);
    if (header->numParams != numParametersTotal) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetwork: Mismatching network parameter count (" + std::to_string(header->numParams)
                + " vs. " + std::to_string(numParametersTotal) + ") for \"" + modelPath + "\".");
    }
    if (header->format == NETWORK_PARAMS_FORMAT_FLOAT && precision == qmlp::Tensor::Precision::HALF) {
        float* dataOld = reinterpret_cast<float*>(paramsDataHost);
        half* dataNew = new half[header->numParams];
        for (uint32_t i = 0; i < header->numParams; i++) {
            dataNew[i] = half(dataOld[i]);
        }
        paramsDataHost = reinterpret_cast<uint8_t*>(dataNew);
    } else if (header->format == NETWORK_PARAMS_FORMAT_HALF && precision == qmlp::Tensor::Precision::FLOAT) {
        half* dataOld = reinterpret_cast<half*>(paramsDataHost);
        float* dataNew = new float[header->numParams];
        for (uint32_t i = 0; i < header->numParams; i++) {
            dataNew[i] = float(dataOld[i]);
        }
        paramsDataHost = reinterpret_cast<uint8_t*>(dataNew);
    } else if (header->format == NETWORK_PARAMS_FORMAT_FLOAT && precision != qmlp::Tensor::Precision::FLOAT
            || header->format == NETWORK_PARAMS_FORMAT_HALF && precision != qmlp::Tensor::Precision::HALF) {
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

    if (header->format == NETWORK_PARAMS_FORMAT_FLOAT && precision == qmlp::Tensor::Precision::HALF
            || header->format == NETWORK_PARAMS_FORMAT_HALF && precision == qmlp::Tensor::Precision::FLOAT) {
        delete[] paramsDataHost;
    }
}

void QuickMLPCorrelationCalculator::loadModelFromFile(const std::string& modelPath) {
    moduleWrapper = std::make_shared<QuickMLPModuleWrapper>();

    std::unordered_map<std::string, sgl::ArchiveEntry> archiveFiles;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadAllFilesFromArchive(modelPath, archiveFiles, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in QuickMLPCorrelationCalculator::loadModelFromFile: Could not load data from model \""
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
                    "Error in QuickMLPCorrelationCalculator::loadModelFromFile: Invalid symmetrizer type \""
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
                "Error in QuickMLPCorrelationCalculator::loadModelFromFile: Could not load encoder or decoder "
                "configuration from model \"" + modelPath + "\".");
        return;
    }

    auto itNetworkEncoder = archiveFiles.find("network_encoder.bin");
    auto itNetworkDecoder = archiveFiles.find("network_decoder.bin");
    if (itNetworkEncoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in QuickMLPCorrelationCalculator::loadModelFromFile: Missing network_encoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    if (itNetworkDecoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in QuickMLPCorrelationCalculator::loadModelFromFile: Missing network_decoder.bin in file \""
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

    uint32_t symmetrizerFactor = symmetrizerType == SymmetrizerType::AddDiff ? 2 : 1;
    if (numLayersOutEncoder * symmetrizerFactor != numLayersInDecoder) {
        sgl::Logfile::get()->throwError(
                "Error in QuickMLPCorrelationCalculator::loadModelFromFile: Mismatch between encoder output and "
                "decoder input dimensions.");
    }

    cacheNeedsRecreate = true;
}

void QuickMLPCorrelationCalculator::recreateCache(int batchSize) {
    int cs = networkType == NetworkType::MINE ? getCorrelationMemberCount() : 1;
    if (moduleWrapper->networkEncoder->precisionIn() != qmlp::Tensor::Precision::FLOAT) {
        sgl::Logfile::get()->throwError(
                "Error in QuickMLPCorrelationCalculator::recreateCache: "
                "Encoder input precision is expected to be float.");
    }

    // Network needs multiple of 16 for number of input layers.
    //int numInputLayers = 16;

    if (cacheWrapper->auxMemoryToken) {
        volumeData->popAuxiliaryMemoryDevice(cacheWrapper->auxMemoryToken);
    }

    cacheWrapper->referenceInput = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionIn(), {cs, int(numLayersInEncoder) });
    cacheWrapper->referenceEncoded = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), {cs, int(numLayersOutEncoder) });
    cacheWrapper->queryInput = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionIn(), {cs * int(batchSize), int(numLayersInEncoder) });
    cacheWrapper->queryEncoded = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), {cs * int(batchSize), int(numLayersOutEncoder) });
    cacheWrapper->queryEncodedPermuted = qmlp::Tensor(
            moduleWrapper->networkEncoder->precisionOut(), {cs * int(batchSize), int(numLayersOutEncoder) });
    cacheWrapper->symmetrizedReferenceInput = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionIn(), {cs * int(batchSize), int(numLayersInDecoder) });
    cacheWrapper->symmetrizedQueryInput = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionIn(), {cs * int(batchSize), int(numLayersInDecoder) });
    cacheWrapper->referenceDecoded = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionOut(), {cs * int(batchSize), int(numLayersOutDecoder) });
    cacheWrapper->queryDecoded = qmlp::Tensor(
            moduleWrapper->networkDecoder->precisionOut(), {cs * int(batchSize), int(numLayersOutDecoder) });

    size_t auxBuffersSizeInBytes = 0;
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceInput.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceEncoded.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryInput.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryEncoded.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryEncodedPermuted.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->symmetrizedReferenceInput.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->symmetrizedQueryInput.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceDecoded.numBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryDecoded.numBytes());
    cacheWrapper->auxMemoryToken = volumeData->pushAuxiliaryMemoryDevice(auxBuffersSizeInBytes);
}

CUdeviceptr QuickMLPCorrelationCalculator::getReferenceInputPointer() {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->referenceInput.rawPtr());
}

CUdeviceptr QuickMLPCorrelationCalculator::getQueryInputPointer() {
    return reinterpret_cast<CUdeviceptr>(cacheWrapper->queryInput.rawPtr());
}

void QuickMLPCorrelationCalculator::runInferenceReference() {
    moduleWrapper->networkEncoder->inference(cacheWrapper->referenceInput, cacheWrapper->referenceEncoded, stream);
}

void QuickMLPCorrelationCalculator::runInferenceBatch(uint32_t batchOffset, uint32_t batchSize) {
    int cs = getCorrelationMemberCount();

    moduleWrapper->networkEncoder->inference(cacheWrapper->queryInput, cacheWrapper->queryEncoded, stream);

    /*sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
            (CUdeviceptr)cacheWrapper->queryEncodedPermuted.rawPtr(),
            (CUdeviceptr)cacheWrapper->queryEncoded.rawPtr(),
            sizeof(float) * batchSize, stream), "Error in cuMemcpyAsync: ");*/

    if (networkType == NetworkType::MINE) {
        uint32_t* permutationIndicesBuffer = reinterpret_cast<uint32_t*>(permutationIndicesBufferCu);
        generateRandomPermutations<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                permutationIndicesBuffer, uint32_t(cs), batchOffset);
        if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
            symmetrizer(
                    cacheWrapper->referenceEncoded.dataPtr<float>(), cacheWrapper->queryEncoded.dataPtr<float>(),
                    cacheWrapper->symmetrizedReferenceInput.dataPtr<float>(),
                    cacheWrapper->symmetrizedQueryInput.dataPtr<float>(),
                    permutationIndicesBuffer, batchSize, uint32_t(cs), numLayersOutEncoder, symmetrizerType, stream);
        } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
            symmetrizer(
                    cacheWrapper->referenceEncoded.dataPtr<half>(), cacheWrapper->queryEncoded.dataPtr<half>(),
                    cacheWrapper->symmetrizedReferenceInput.dataPtr<half>(),
                    cacheWrapper->symmetrizedQueryInput.dataPtr<half>(),
                    permutationIndicesBuffer, batchSize, uint32_t(cs), numLayersOutEncoder, symmetrizerType, stream);
        }
    } else {
        if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
            symmetrizerSrn(
                    cacheWrapper->referenceEncoded.dataPtr<float>(), cacheWrapper->queryEncoded.dataPtr<float>(),
                    cacheWrapper->symmetrizedQueryInput.dataPtr<float>(),
                    batchSize, numLayersOutEncoder, symmetrizerType, stream);
        } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
            symmetrizerSrn(
                    cacheWrapper->referenceEncoded.dataPtr<half>(), cacheWrapper->queryEncoded.dataPtr<half>(),
                    cacheWrapper->symmetrizedQueryInput.dataPtr<half>(),
                    batchSize, numLayersOutEncoder, symmetrizerType, stream);
        }
    }

    if (networkType == NetworkType::MINE) {
        moduleWrapper->networkDecoder->inference(
                cacheWrapper->symmetrizedReferenceInput, cacheWrapper->referenceDecoded, stream);
    }
    moduleWrapper->networkDecoder->inference(
            cacheWrapper->symmetrizedQueryInput, cacheWrapper->queryDecoded, stream);

    float* miOutput = reinterpret_cast<float*>(outputImageBufferCu) + batchOffset;
    if (networkType == NetworkType::MINE) {
        if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
            combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                    cacheWrapper->referenceDecoded.dataPtr<float>(), cacheWrapper->queryDecoded.dataPtr<float>(),
                    miOutput, uint32_t(cs), numLayersOutDecoder);
        } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
            combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                    cacheWrapper->referenceDecoded.dataPtr<half>(), cacheWrapper->queryDecoded.dataPtr<half>(),
                    miOutput, uint32_t(cs), numLayersOutDecoder);
        }
    } else {
        if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::FLOAT) {
            if (isMutualInformationData) {
                copyDecoderOutputSrnMutualInformation<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                        cacheWrapper->queryDecoded.dataPtr<float>(), miOutput, numLayersOutDecoder);
            } else if (calculateAbsoluteValue) {
                copyDecoderOutputSrnCorrelationCoefficientAbs<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                        cacheWrapper->queryDecoded.dataPtr<float>(), miOutput, numLayersOutDecoder);
            } else {
                copyDecoderOutputSrnCorrelationCoefficient<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                        cacheWrapper->queryDecoded.dataPtr<float>(), miOutput, numLayersOutDecoder);
            }
        } else if (moduleWrapper->networkEncoder->precisionOut() == qmlp::Tensor::Precision::HALF) {
            if (isMutualInformationData) {
                copyDecoderOutputSrnMutualInformation<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                        cacheWrapper->queryDecoded.dataPtr<half>(), miOutput, numLayersOutDecoder);
            } else if (calculateAbsoluteValue) {
                copyDecoderOutputSrnCorrelationCoefficientAbs<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                        cacheWrapper->queryDecoded.dataPtr<half>(), miOutput, numLayersOutDecoder);
            } else {
                copyDecoderOutputSrnCorrelationCoefficient<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                        cacheWrapper->queryDecoded.dataPtr<half>(), miOutput, numLayersOutDecoder);
            }
        }
    }
}
