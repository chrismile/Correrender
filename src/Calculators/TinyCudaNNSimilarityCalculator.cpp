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

#include <Utils/AppSettings.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Archive.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "TinyCudaNNSimilarityCalculator.hpp"

using precision_t = tcnn::network_precision_t;

TinyCudaNNSimilarityCalculator::TinyCudaNNSimilarityCalculator(sgl::vk::Renderer* renderer)
        : EnsembleSimilarityCalculator(renderer) {
    sgl::vk::Device* device = renderer->getDevice();
    if (device->getDeviceDriverId() != VK_DRIVER_ID_NVIDIA_PROPRIETARY
            || !sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::Logfile::get()->throwError(
                "Error in TinyCudaNNSimilarityCalculator::TinyCudaNNSimilarityCalculator: "
                "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
    }

    sgl::AppSettings::get()->getSettings().getValueOpt(
            "tinyCudaNNSimilarityCalculatorModelFilePath", modelFilePath);
}

TinyCudaNNSimilarityCalculator::~TinyCudaNNSimilarityCalculator() {
    sgl::AppSettings::get()->getSettings().addKeyValue(
            "tinyCudaNNSimilarityCalculatorModelFilePath", modelFilePath);
}

void TinyCudaNNSimilarityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    ;
}

static nlohmann::json loadJsonConfig(const std::string& configPath) {
    uint8_t* buffer = nullptr;
    size_t bufferSize = 0;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadFileFromArchive(configPath, buffer, bufferSize, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Could not load data from model \""
                + configPath + "\".");
        return false;
    }
    nlohmann::json config = nlohmann::json::parse(std::string(reinterpret_cast<char*>(buffer), bufferSize));
    delete[] buffer;
    return config;
}

const uint32_t TINY_CUDA_NN_PARAMS_FORMAT_FLOAT = 0;
const uint32_t TINY_CUDA_NN_PARAMS_FORMAT_HALF = 1;
struct TinyCudaNNDataHeader {
    uint32_t format = 0;
    uint32_t numParams = 0;
};

static nlohmann::json TinyCudaNNSimilarityCalculator::loadNetwork(const nlohmann::json& config, const std::string& modelPath) {
    uint8_t* buffer = nullptr;
    size_t bufferSize = 0;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadFileFromArchive(modelPath, buffer, bufferSize, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Could not load data from model \""
                + modelPath + "\".");
        return false;
    }

    auto* header = reinterpret_cast<TinyCudaNNDataHeader*>(buffer);
    uint8_t* paramsData = buffer + sizeof(TinyCudaNNDataHeader);

    auto encodingOpts = config.value("encoding", nlohmann::json::object());
    auto lossOpts = config.value("loss", nlohmann::json::object());
    auto optimizerOpts = config.value("optimizer", nlohmann::json::object());
    auto networkOpts = config.value("network", nlohmann::json::object());

    int n_input_dims = 1;
    int n_output_dims = 1;
    std::shared_ptr<tcnn::Loss<precision_t>> loss{tcnn::create_loss<precision_t>(lossOpts)};
    std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{tcnn::create_optimizer<precision_t>(optimizerOpts)};
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(
            n_input_dims, n_output_dims, encodingOpts, networkOpts);
    auto trainer = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network, optimizer, loss);
#ifdef TCNN_HALF_PRECISION
    if (header->format == TINY_CUDA_NN_PARAMS_FORMAT_FLOAT) {
        trainer->set_params_full_precision(reinterpret_cast<float*>(paramsData), header->numParams, false);
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

    delete[] buffer;
}

void TinyCudaNNSimilarityCalculator::loadModelFromFile(const std::string& modelPath) {
    uint8_t* buffer = nullptr;
    size_t bufferSize = 0;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadFileFromArchive(modelPath, buffer, bufferSize, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Could not load data from model \""
                + modelPath + "\".");
        return;
    }
}

void TinyCudaNNSimilarityCalculator::calculateDevice(
        int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    int xs = 1, ys = 1, zs = 1, es = 1;

    CUstream stream{};
    sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamCreate(&stream, 0);

	tcnn::GPUMemory<float> result(xs * ys * zs);
    result.data();
    sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(result.data(), inputPtr, sizeof(float) * xs * ys * zs, stream);

    auto encoding_opts = config.value("encoding", nlohmann::json::object());
    auto network_opts = config.value("network", nlohmann::json::object());


    uint32_t numLayersInEncoder = batchSize;
    uint32_t numLayersOutEncoder;
    // numLayersInDecoder == numLayersOutEncoder when symmetrizer is sum operation.
    uint32_t numLayersInDecoder;
    auto networkOpts = config.value("network", nlohmann::json::object());

    tcnn::GPUMatrix<float> prediction(batchSize, n_coords_padded);
    tcnn::GPUMatrix<float> inference_batch(xs_and_ys.data(), batchSize, n_coords_padded);
    std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(
            n_input_dims, n_output_dims, encoding_opts, network_opts);
    network->inference(stream, inference_batch, prediction);

    sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(stream);
    sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamDestroy(stream);
}
