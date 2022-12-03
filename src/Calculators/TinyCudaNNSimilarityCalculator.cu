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
#include <Utils/AppSettings.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Archive.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Volume/VolumeData.hpp"
#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "MutualInformation.cuh"
#include "TinyCudaNNSimilarityCalculator.hpp"

#if CUDA_VERSION < 11020
#error CUDA >= 11.2 is required for timeline semaphore support.
#endif

using precision_t = tcnn::network_precision_t;

struct TinyCudaNNModuleWrapper {
    nlohmann::json config;
    //torch::jit::Module module;
    //torch::jit::Module frozenModule;
};

struct TinyCudaNNCacheWrapper {
    tcnn::GPUMemory<float> referenceInput;
    tcnn::GPUMemory<float> referenceEncoded;
    tcnn::GPUMemory<float> queryInput;
    tcnn::GPUMemory<float> queryEncoded;
    tcnn::GPUMemory<float> queryEncodedPermuted;
    tcnn::GPUMemory<float> symmetrizedReferenceInput;
    tcnn::GPUMemory<float> symmetrizedQueryInput;
    tcnn::GPUMemory<float> referenceDecoded;
    tcnn::GPUMemory<float> queryDecoded;
};

TinyCudaNNSimilarityCalculator::TinyCudaNNSimilarityCalculator(sgl::vk::Renderer* renderer)
        : EnsembleSimilarityCalculator(renderer) {
    sgl::vk::Device* device = renderer->getDevice();
    if (device->getDeviceDriverId() != VK_DRIVER_ID_NVIDIA_PROPRIETARY
            || !sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::Logfile::get()->throwError(
                "Error in TinyCudaNNSimilarityCalculator::TinyCudaNNSimilarityCalculator: "
                "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
    }

    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamCreate(
            &stream, 0), "Error in cuStreamCreate: ");

    sgl::AppSettings::get()->getSettings().getValueOpt(
            "tinyCudaNNSimilarityCalculatorModelFilePath", modelFilePath);
    if (sgl::FileUtils::get()->exists(modelFilePath) && !sgl::FileUtils::get()->isDirectory(modelFilePath)) {
        loadModelFromFile(modelFilePath);
    }
}

TinyCudaNNSimilarityCalculator::~TinyCudaNNSimilarityCalculator() {
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamDestroy(
            stream), "Error in cuStreamDestroy: ");

    sgl::AppSettings::get()->getSettings().addKeyValue(
            "tinyCudaNNSimilarityCalculatorModelFilePath", modelFilePath);
}

void TinyCudaNNSimilarityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    EnsembleSimilarityCalculator::setVolumeData(_volumeData, isNewData);
}

const uint32_t TINY_CUDA_NN_PARAMS_FORMAT_FLOAT = 0;
const uint32_t TINY_CUDA_NN_PARAMS_FORMAT_HALF = 1;
struct TinyCudaNNDataHeader {
    uint32_t format = 0;
    uint32_t numParams = 0;
};

static nlohmann::json loadNetwork(const nlohmann::json& config, const sgl::ArchiveEntry& archiveEntry) {
    /*auto* header = reinterpret_cast<TinyCudaNNDataHeader*>(buffer);
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

    delete[] buffer;*/
}

void TinyCudaNNSimilarityCalculator::loadModelFromFile(const std::string& modelPath) {
    std::unordered_map<std::string, sgl::ArchiveEntry> archiveFiles;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadAllFilesFromArchive(modelPath, archiveFiles, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Could not load data from model \""
                + modelPath + "\".");
        return;
    }

    nlohmann::json configEncoder, configDecoder;
    auto itConfig = archiveFiles.find("config.json");
    auto itConfigEncoder = archiveFiles.find("config_encoder.json");
    auto itConfigDecoder = archiveFiles.find("config_decoder.json");
    if (itConfig != archiveFiles.end()) {
        const auto& entry = itConfig->second;
        configEncoder = configDecoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entry.bufferData.get()), entry.bufferSize));
    } else if (itConfigEncoder != archiveFiles.end() && itConfigDecoder != archiveFiles.end()) {
        const auto& entryEncoder = itConfigEncoder->second;
        const auto& entryDecoder = itConfigDecoder->second;
        configEncoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entryEncoder.bufferData.get()), entryEncoder.bufferSize));
        configDecoder = nlohmann::json::parse(std::string(
                reinterpret_cast<char*>(entryDecoder.bufferData.get()), entryDecoder.bufferSize));
    } else {
        sgl::Logfile::get()->writeError(
                "Error in TinyCudaNNSimilarityCalculator::loadModelFromFile: Could not load config from model \""
                + modelPath + "\".");
        return;
    }

    //loadNetwork(const nlohmann::json& config, const sgl::ArchiveEntry& archiveEntry);
}

void TinyCudaNNSimilarityCalculator::calculateDevice(
        int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);

    if (!moduleWrapper) {
        deviceCacheEntry->getVulkanImage()->clearColor(glm::vec4(0.0f), renderer->getVkCommandBuffer());
        sgl::Logfile::get()->writeWarning(
                "Warning in TinyCudaNNSimilarityCalculator::calculateDevice: Network modules are not loaded.",
                true);
        return;
    }

    int gpuBatchSize1D = gpuBatchSize1DBase;

    auto& referenceInput = cacheWrapper->referenceInput;
    auto& referenceEncoded = cacheWrapper->referenceInput;
    auto& queryInput = cacheWrapper->referenceInput;
    auto& queryEncoded = cacheWrapper->referenceInput;
    auto& queryEncodedPermuted = cacheWrapper->referenceInput;
    auto& symmetrizedReferenceInput = cacheWrapper->referenceInput;
    auto& symmetrizedQueryInput = cacheWrapper->referenceInput;
    auto& referenceDecoded = cacheWrapper->referenceInput;
    auto& queryDecoded = cacheWrapper->referenceInput;

    if (cachedEnsembleSizeDevice != size_t(es)) {
        if (outputImageBufferCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    outputImageBufferCu, stream), "Error in cuMemFreeAsync: ");
        }
        if (ensembleTextureArrayCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    ensembleTextureArrayCu, stream), "Error in cuMemFreeAsync: ");
        }
        cachedEnsembleSizeDevice = size_t(es);
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &outputImageBufferCu, volumeData->getSlice3dSizeInBytes(FieldType::SCALAR), stream), "Error in cuMemAllocAsync: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &ensembleTextureArrayCu, es * sizeof(CUtexObject), stream), "Error in cuMemAllocAsync: ");

        // TODO
        size_t numLayersOutEncoder, numLayersInDecoder, numLayersOutDecoder;

        referenceInput = tcnn::GPUMemory<float>(1, es * 4);
        referenceEncoded = tcnn::GPUMemory<float>(1, numLayersOutEncoder);
        queryInput = tcnn::GPUMemory<float>(gpuBatchSize1D, es * 4);
        queryEncoded = tcnn::GPUMemory<float>(gpuBatchSize1D, numLayersOutEncoder);
        queryEncodedPermuted = tcnn::GPUMemory<float>(gpuBatchSize1D, numLayersOutEncoder);
        symmetrizedReferenceInput = tcnn::GPUMemory<float>(gpuBatchSize1D, numLayersInDecoder);
        symmetrizedQueryInput = tcnn::GPUMemory<float>(gpuBatchSize1D, numLayersInDecoder);
        referenceDecoded = tcnn::GPUMemory<float>(gpuBatchSize1D, numLayersOutDecoder);
        queryDecoded = tcnn::GPUMemory<float>(gpuBatchSize1D, numLayersOutDecoder);
    }

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
    auto startLoad = std::chrono::system_clock::now();
#endif

    float minEnsembleVal = std::numeric_limits<float>::max();
    float maxEnsembleVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::DeviceCacheEntry> ensembleEntryFields;
    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    std::vector<CUtexObject> ensembleTexturesCu;
    ensembleEntryFields.reserve(es);
    ensembleImageViews.reserve(es);
    ensembleTexturesCu.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleImageViews.push_back(ensembleEntryField->getVulkanImageView());
        ensembleTexturesCu.push_back(ensembleEntryField->getCudaTexture());
        if (ensembleEntryField->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            deviceCacheEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        }
        auto [minVal, maxVal] = volumeData->getMinMaxScalarFieldValue(
                scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        minEnsembleVal = std::min(minEnsembleVal, minVal);
        maxEnsembleVal = std::max(maxEnsembleVal, maxVal);
    }

    if (cachedEnsembleTexturesCu != ensembleTexturesCu) {
        cachedEnsembleTexturesCu = ensembleTexturesCu;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                stream), "Error in cuStreamSynchronize: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                ensembleTextureArrayCu, ensembleTexturesCu.data(), sizeof(CUtexObject) * es), "Error in cuMemcpyHtoD: ");
    }

#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    std::vector<int64_t> referenceInputSizes = { 1, es, 4 };

    sgl::vk::CommandBufferPtr commandBufferRender = renderer->getCommandBuffer();
    vulkanFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
    commandBufferRender->pushSignalSemaphore(vulkanFinishedSemaphore);
    renderer->endCommandBuffer();

    renderer->submitToQueue();

    vulkanFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);

    //auto encoding_opts = config.value("encoding", nlohmann::json::object());
    //auto network_opts = config.value("network", nlohmann::json::object());

    uint32_t numLayersInEncoder;
    uint32_t numLayersOutEncoder;
    // numLayersInDecoder == numLayersOutEncoder when symmetrizer is sum operation.
    uint32_t numLayersInDecoder, numLayersOutDecoder;
    //auto networkOpts = config.value("network", nlohmann::json::object());

    uint32_t numSliceEntries = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(gpuBatchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * gpuBatchSize1D;
        uint32_t batchSize = std::min(uint32_t(gpuBatchSize1D), numSliceEntries - batchOffset);


        CUdeviceptr outputBuffer = reinterpret_cast<CUdeviceptr>(queryInput.data());
        CUdeviceptr scalarFieldEnsembles = ensembleTextureArrayCu;
        void* kernelParameters[] = {
                &xs, &ys, &zs, &es, &batchOffset, &batchSize, &minEnsembleVal, &maxEnsembleVal,
                &outputBuffer, &scalarFieldEnsembles
        };
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                combineEnsemblesFunctionCu,
                sgl::uiceil(batchSize, 256), 1, 1, //< Grid size.
                256, 1, 1, //< Block size.
                0, //< Dynamic shared memory size.
                stream,
                kernelParameters, //< Kernel parameters.
                nullptr //< Extra (empty).
        ), "Error in cuLaunchKernel: ");

        // TODO: Support for inference_mixed_precision?
        //network->inference(stream, queryInput, queryEncoded);

        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
                (CUdeviceptr)queryEncodedPermuted.data(),
                (CUdeviceptr)queryEncoded.data(),
                sizeof(float) * batchSize, stream), "Error in cuMemcpyAsync: ");
        randomShuffleFisherYatesXorshift<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                queryEncodedPermuted.data(), numLayersOutEncoder);

        symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                referenceEncoded.data(), queryEncoded.data(), symmetrizedReferenceInput.data(), numLayersOutEncoder);
        symmetrizer<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                referenceEncoded.data(), queryEncodedPermuted.data(), symmetrizedQueryInput.data(), numLayersOutEncoder);

        //network->inference(stream, symmetrizedReferenceInput, referenceDecoded);
        //network->inference(stream, symmetrizedQueryInput, queryDecoded);

        float *miOutput = reinterpret_cast<float*>(outputImageBufferCu) + batchOffset;
        combineDecoderOutput<<<sgl::uiceil(batchSize, 256), 256, 0, stream>>>(
                referenceDecoded.data(), queryDecoded.data(), miOutput, numLayersOutDecoder);
    }

    deviceCacheEntry->getImageCudaExternalMemory()->memcpyCudaDtoA3DAsync(outputImageBufferCu, stream);

    cudaFinishedSemaphore->signalSemaphoreCuda(stream, timelineValue);
    cudaFinishedSemaphore->setWaitSemaphoreValue(timelineValue);
    sgl::vk::CommandBufferPtr postRenderCommandBuffer = postRenderCommandBuffers.at(frameIndex);
    renderer->pushCommandBuffer(postRenderCommandBuffer);
    renderer->beginCommandBuffer();
    postRenderCommandBuffer->pushWaitSemaphore(
            cudaFinishedSemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

#ifdef TEST_INFERENCE_SPEED
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}
