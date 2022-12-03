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

#include <Utils/AppSettings.hpp>
#include <Utils/File/Archive.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>

#include "MutualInformation.cuh"
#include "QuickMLPSimilarityCalculator.hpp"

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

void test() {
    std::string quickMlpModelsPath = sgl::AppSettings::get()->getDataDirectory() + "QuickMLP/";
    nlohmann::json cfg;
    auto network = std::make_shared<qmlp::FusedNetwork>(
            cfg, std::filesystem::path(quickMlpModelsPath));

    qmlp::Tensor::Precision precision = network->networkParameterPrecision(qmlp::Tensor::INFERENCE);
    qmlp::Tensor parameters(precision, { network->networkParameterCount() });
    void* parametersDevice = parameters.rawPtr();
    /*sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
            reinterpret_cast<CUdeviceptr>(parametersDevice), dataHost,
            qmlp::Tensor::BytesPerEntry[precision] * network->networkParameterCount()), "Error in cuMemcpyHtoD: ");
    network->setNetworkParameter(parameters, qmlp::Tensor::INFERENCE);

    qmlp::Tensor input(network->precisionIn(), { N, network->channelsIn() });
    qmlp::Tensor output(network->precisionOut(), { N, network->channelsOut() });

    CUstream stream = nullptr;
    network->inference(input, output, stream);*/

    __half* testPtr0 = nullptr;
    __half* testPtr1 = nullptr;
    __half* testPtr2 = nullptr;

    randomShuffleFisherYatesXorshift<<<1,1,1>>>(testPtr0, 1);

    symmetrizer<<<1,1,1>>>(testPtr0, testPtr1, testPtr2, 1);

    float* miOutput = nullptr;
    combineDecoderOutput<<<1,1,1>>>(testPtr0, testPtr1, miOutput, 1);
}
