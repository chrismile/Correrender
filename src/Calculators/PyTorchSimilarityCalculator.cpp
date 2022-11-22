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

#include <json/json.h>
#include <torch/script.h>
#include <torch/cuda.h>
#include <c10/core/MemoryFormat.h>
#ifdef SUPPORT_CUDA_INTEROP
#include <c10/cuda/CUDAStream.h>
#endif

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "PyTorchSimilarityCalculator.hpp"

#if CUDA_VERSION < 11020
#error CUDA >= 11.2 is required for timeline semaphore support.
#endif

struct ModuleWrapper {
    torch::jit::Module module;
};

static torch::DeviceType getTorchDeviceType(PyTorchDevice pyTorchDevice) {
    if (pyTorchDevice == PyTorchDevice::CPU) {
        return torch::kCPU;
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (pyTorchDevice == PyTorchDevice::CUDA) {
        return torch::kCUDA;
    }
#endif
    else {
        sgl::Logfile::get()->writeError("Error in getTorchDeviceType: Unsupported device type.");
        return torch::kCPU;
    }
}

PyTorchSimilarityCalculator::PyTorchSimilarityCalculator(sgl::vk::Renderer* renderer)
        : EnsembleSimilarityCalculator(renderer) {
    sgl::vk::Device* device = renderer->getDevice();

#ifdef SUPPORT_CUDA_INTEROP
    // Support CUDA on NVIDIA GPUs using the proprietary driver.
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY && torch::cuda::is_available()) {
        pyTorchDevice = PyTorchDevice::CUDA;
        if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumetricPathTracingModuleRenderer::renderFrameCuda: "
                    "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
        }
        uint8_t* moduleBuffer = nullptr;
        size_t bufferSize = 0;
        sgl::loadFileFromSource(
                sgl::AppSettings::get()->getDataDirectory() + "/__cudacache__/CombineEnsembles.fatbin",
                moduleBuffer, bufferSize, true);
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleLoadFatBinary(
                &combineEnsemblesModuleCu, moduleBuffer), "Error in cuModuleLoadFatBinary: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &combineEnsemblesFunctionCu, combineEnsemblesModuleCu, "combineEnsembles"), "Error in cuModuleGetFunction: ");
    }
#endif

    // TODO: Test
    //pyTorchDevice = PyTorchDevice::CPU;

    sgl::AppSettings::get()->getSettings().getValueOpt(
            "pyTorchSimilarityCalculatorModelFilePathEncoder", modelFilePathEncoder);
    sgl::AppSettings::get()->getSettings().getValueOpt(
            "pyTorchSimilarityCalculatorModelFilePathDecoder", modelFilePathDecoder);
    if (sgl::FileUtils::get()->exists(modelFilePathEncoder)
            && !sgl::FileUtils::get()->isDirectory(modelFilePathEncoder)) {
        loadModelFromFile(0, modelFilePathEncoder);
    }
    if (sgl::FileUtils::get()->exists(modelFilePathDecoder)
            && !sgl::FileUtils::get()->isDirectory(modelFilePathDecoder)) {
        loadModelFromFile(1, modelFilePathDecoder);
    }

    referenceEnsembleCombinePass = std::make_shared<ReferenceEnsembleCombinePass>(renderer);
    ensembleCombinePass = std::make_shared<EnsembleCombinePass>(renderer);
    ensembleCombinePass->setBatchSize(gpuBatchSize1D);
}

PyTorchSimilarityCalculator::~PyTorchSimilarityCalculator() {
    sgl::AppSettings::get()->getSettings().addKeyValue(
            "pyTorchSimilarityCalculatorModelFilePathEncoder", modelFilePathEncoder);
    sgl::AppSettings::get()->getSettings().addKeyValue(
            "pyTorchSimilarityCalculatorModelFilePathDecoder", modelFilePathDecoder);

    if (referenceInputValues) {
        delete[] referenceInputValues;
        referenceInputValues = nullptr;
    }
    if (batchInputValues) {
        delete[] batchInputValues;
        batchInputValues = nullptr;
    }
    if (outputImageBufferCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                outputImageBufferCu), "Error in cuMemFree: ");
    }
    if (ensembleTextureArrayCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                ensembleTextureArrayCu), "Error in cuMemFree: ");
    }
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleUnload(
                combineEnsemblesModuleCu), "Error in cuModuleUnload: ");
    }
}

void PyTorchSimilarityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    EnsembleSimilarityCalculator::setVolumeData(_volumeData, isNewData);

    referenceEnsembleCombinePass->setVolumeData(volumeData, isNewData);
    ensembleCombinePass->setVolumeData(volumeData, isNewData);

    /*isgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    f (referenceInputValues) {
        delete[] referenceInputValues;
        referenceInputValues = nullptr;
    }
    if (batchInputValues) {
        delete[] batchInputValues;
        batchInputValues = nullptr;
    }
    renderImageStagingBuffers = {};
    denoisedImageStagingBuffer = {};
#ifdef SUPPORT_CUDA_INTEROP
    outputImageBufferCu = {};
    //outputImageBufferVk = {};
    postRenderCommandBuffers = {};
    renderFinishedSemaphores = {};
    denoiseFinishedSemaphores = {};
    timelineValue = 0;
#endif*/
}

bool PyTorchSimilarityCalculator::loadModelFromFile(int idx, const std::string& modelPath) {
    torch::DeviceType deviceType = getTorchDeviceType(pyTorchDevice);
    torch::jit::ExtraFilesMap extraFilesMap;
    extraFilesMap["model_info.json"] = "";
    auto& wrapper = idx == 0 ? encoderWrapper : decoderWrapper;
    wrapper = std::make_shared<ModuleWrapper>();
    try {
        // std::shared_ptr<torch::jit::script::Module>
        wrapper->module = torch::jit::load(modelPath, deviceType, extraFilesMap);
        wrapper->module.to(deviceType);
    } catch (const c10::Error& e) {
        sgl::Logfile::get()->writeError("Error: Couldn't load the PyTorch module from \"" + modelPath + "\"!");
        sgl::Logfile::get()->writeError(std::string() + "What: " + e.what());
        wrapper = {};
        return false;
    } catch (const torch::jit::ErrorReport& e) {
        sgl::Logfile::get()->writeError("Error: Couldn't load the PyTorch module from \"" + modelPath + "\"!");
        sgl::Logfile::get()->writeError(std::string() + "What: " + e.what());
        sgl::Logfile::get()->writeError("Call stack: " + e.current_call_stack());
        wrapper = {};
        return false;
    }

    dirty = true;
    return true;
}

void PyTorchSimilarityCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();
    auto ues = uint32_t(es);

    if (!encoderWrapper || !decoderWrapper) {
        memset(buffer, 0, volumeData->getSlice3dSizeInBytes(FieldType::SCALAR));
        sgl::Logfile::get()->writeWarning(
                "Warning in PyTorchSimilarityCalculator::calculateCpu: Encoder or decoder module is not loaded.",
                true);
        return;
    }

    if (cachedEnsembleSizeHost != size_t(es)) {
        if (cachedEnsembleSizeHost != 0) {
            delete[] referenceInputValues;
            delete[] batchInputValues;
        }
        cachedEnsembleSizeHost = size_t(es);
        referenceInputValues = new float[es * 4];
        batchInputValues = new float[es * 4 * batchSize1D];
    }

    std::vector<VolumeData::HostCacheEntry> ensembleEntryFields;
    std::vector<float*> ensembleFields;
    ensembleEntryFields.reserve(es);
    ensembleFields.reserve(es);
    float minEnsembleVal = std::numeric_limits<float>::max();
    float maxEnsembleVal = std::numeric_limits<float>::lowest();
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        auto [minVal, maxVal] = volumeData->getMinMaxScalarFieldValue(
                scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        minEnsembleVal = std::min(minEnsembleVal, minVal);
        maxEnsembleVal = std::max(maxEnsembleVal, maxVal);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleFields.push_back(ensembleEntryField.get());
    }

    std::vector<int64_t> referenceInputSizes = { 1, es, 4 };

    size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
    glm::vec3 referencePointNorm =
            glm::vec3(referencePointIndex) / glm::vec3(xs - 1, ys - 1, zs - 1) * 2.0f - glm::vec3(1.0f);
    for (int e = 0; e < es; e++) {
        referenceInputValues[e * 4] =
                (ensembleFields.at(e)[referencePointIdx] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
        referenceInputValues[e * 4 + 1] = referencePointNorm.x;
        referenceInputValues[e * 4 + 2] = referencePointNorm.y;
        referenceInputValues[e * 4 + 3] = referencePointNorm.z;
    }

    torch::Tensor referenceInputTensor = torch::from_blob(
            referenceInputValues, referenceInputSizes,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    std::vector<torch::jit::IValue> referenceInputs;
    referenceInputs.emplace_back(referenceInputTensor);
    at::Tensor referenceEncodedTensor = encoderWrapper->module.forward(referenceInputs).toTensor();

    std::vector<torch::jit::IValue> encoderInputs;
    encoderInputs.resize(1);
    std::vector<torch::jit::IValue> decoderInputs;
    decoderInputs.resize(2);
    decoderInputs.at(0) = referenceEncodedTensor;

    std::vector<int64_t> inputSizes = { batchSize1D, es, 4 };

    uint32_t numSliceEntries = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(batchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * batchSize1D;
        uint32_t batchSize = std::min(uint32_t(batchSize1D), numSliceEntries - batchOffset);

#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, batchSize), [&](auto const& r) {
            for (auto pointIdx = r.begin(); pointIdx != r.end(); pointIdx++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel for default(none) shared(xs, ys, zs, es, ues, batchSize, batchOffset) \
        shared(ensembleFields, minEnsembleVal, maxEnsembleVal)
#endif
        for (uint32_t pointIdx = 0; pointIdx < batchSize; pointIdx++) {
#endif
            uint32_t pointIdxWriteOffset = pointIdx * es * 4;
            uint32_t pointIdxReadOffset = pointIdx + batchOffset;
            uint32_t x = pointIdxReadOffset % uint32_t(xs);
            uint32_t y = (pointIdxReadOffset / uint32_t(xs)) % uint32_t(ys);
            uint32_t z = pointIdxReadOffset / uint32_t(xs * ys);
            glm::vec3 pointNorm = glm::vec3(x, y, z) / glm::vec3(xs - 1, ys - 1, zs - 1) * 2.0f - glm::vec3(1.0f);
            for (uint32_t e = 0; e < ues; e++) {
                batchInputValues[pointIdxWriteOffset + e * 4] =
                        (ensembleFields.at(e)[pointIdxReadOffset] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
                batchInputValues[pointIdxWriteOffset + e * 4 + 1] = pointNorm.x;
                batchInputValues[pointIdxWriteOffset + e * 4 + 2] = pointNorm.y;
                batchInputValues[pointIdxWriteOffset + e * 4 + 3] = pointNorm.z;
            }
        }
#ifdef USE_TBB
        });
#endif

        torch::Tensor inputTensor = torch::from_blob(
                batchInputValues, inputSizes,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        encoderInputs.at(0) = inputTensor;
        at::Tensor encodedTensor = encoderWrapper->module.forward(encoderInputs).toTensor();
        decoderInputs.at(1) = encodedTensor;
        at::Tensor similarityMetricTensor = decoderWrapper->module.forward(decoderInputs).toTensor();
        if (!similarityMetricTensor.is_contiguous()) {
            if (isFirstContiguousWarning) {
                sgl::Logfile::get()->writeWarning("Error in PyTorchDenoiser::denoise: Output tensor is not contiguous.");
                isFirstContiguousWarning = false;
            }
            similarityMetricTensor = similarityMetricTensor.contiguous();
        }
        memcpy(buffer + batchOffset, similarityMetricTensor.data_ptr(), sizeof(float) * batchSize);
    }
}

void PyTorchSimilarityCalculator::calculateDevice(
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

    if (!encoderWrapper || !decoderWrapper) {
        deviceCacheEntry->getVulkanImage()->clearColor(glm::vec4(0.0f), renderer->getVkCommandBuffer());
        sgl::Logfile::get()->writeWarning(
                "Warning in PyTorchSimilarityCalculator::calculateCpu: Encoder or decoder module is not loaded.",
                true);
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (cachedEnsembleSizeDevice != size_t(es)) {
        referenceInputBufferCu = {};
        inputBufferCu = {};
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

        auto referenceInputBufferVk = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), sizeof(float) * es * 4,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                true, true);
        referenceInputBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(referenceInputBufferVk);

        auto inputBufferVk = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), sizeof(float) * gpuBatchSize1D * es * 4,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                true, true);
        inputBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(inputBufferVk);

        ensembleCombinePass->setOutputBuffer(inputBufferCu->getVulkanBuffer());
        referenceEnsembleCombinePass->setOutputBuffer(referenceInputBufferCu->getVulkanBuffer());
    }

    sgl::vk::Swapchain* swapchain = sgl::AppSettings::get()->getSwapchain();
    uint32_t frameIndex = swapchain ? swapchain->getImageIndex() : 0;
    size_t numSwapchainImages = swapchain ? swapchain->getNumImages() : 1;
    if (numSwapchainImages != cachedNumSwapchainImages) {
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
        if (ensembleEntryField->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
            deviceCacheEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        }
        auto [minVal, maxVal] = volumeData->getMinMaxScalarFieldValue(
                scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        minEnsembleVal = std::min(minEnsembleVal, minVal);
        maxEnsembleVal = std::max(maxEnsembleVal, maxVal);
    }

    referenceEnsembleCombinePass->setEnsembleImageViews(ensembleImageViews);
    referenceEnsembleCombinePass->setEnsembleMinMax(minEnsembleVal, maxEnsembleVal);
    if (createBatchesWithVulkan) {
        ensembleCombinePass->setEnsembleImageViews(ensembleImageViews);
        ensembleCombinePass->setEnsembleMinMax(minEnsembleVal, maxEnsembleVal);
    } else {
        if (cachedEnsembleTexturesCu != ensembleTexturesCu) {
            cachedEnsembleTexturesCu = ensembleTexturesCu;
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                    stream), "Error in cuStreamSynchronize: ");
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                    ensembleTextureArrayCu, ensembleTexturesCu.data(), sizeof(CUtexObject) * es), "Error in cuMemcpyHtoD: ");
        }
    }

    std::vector<int64_t> referenceInputSizes = { 1, es, 4 };

    referenceEnsembleCombinePass->buildIfNecessary();
    renderer->pushConstants(
            referenceEnsembleCombinePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, referencePointIndex);
    referenceEnsembleCombinePass->render();

    sgl::vk::CommandBufferPtr commandBufferRender = renderer->getCommandBuffer();
    vulkanFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
    commandBufferRender->pushSignalSemaphore(vulkanFinishedSemaphore);
    renderer->endCommandBuffer();

    /*
     * 'model->forward()' uses device caching allocators, which may call functions like 'cudaStreamIsCapturing',
     * which enforce synchronization of the stream with the host if more memory needs to be allocated. Thus, we must
     * submit the Vulkan command buffers before this function, as otherwise, CUDA will wait forever on the render
     * finished semaphore!
     */
    renderer->submitToQueue();

#ifdef USE_TIMELINE_SEMAPHORES
    vulkanFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);
#else
    vulkanFinishedSemaphore->waitSemaphoreCuda(stream);
#endif

    torch::Tensor referenceInputTensor = torch::from_blob(
            (void*)referenceInputBufferCu->getCudaDevicePtr(), referenceInputSizes,
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    std::vector<torch::jit::IValue> referenceInputs;
    referenceInputs.emplace_back(referenceInputTensor);
    at::Tensor referenceEncodedTensor = encoderWrapper->module.forward(referenceInputs).toTensor();

    std::vector<torch::jit::IValue> encoderInputs;
    encoderInputs.resize(1);
    std::vector<torch::jit::IValue> decoderInputs;
    decoderInputs.resize(2);
    decoderInputs.at(0) = referenceEncodedTensor;

    std::vector<int64_t> inputSizes = { gpuBatchSize1D, es, 4 };

    uint32_t numSliceEntries = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(gpuBatchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * gpuBatchSize1D;
        uint32_t batchSize = std::min(uint32_t(gpuBatchSize1D), numSliceEntries - batchOffset);

        if (createBatchesWithVulkan) {
            ensembleCombinePass->buildIfNecessary();
            renderer->pushConstants(
                    ensembleCombinePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                    0, batchOffset);
            renderer->pushConstants(
                    ensembleCombinePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                    sizeof(uint32_t), batchSize);
            ensembleCombinePass->render();
        } else {
            CUdeviceptr outputBuffer = inputBufferCu->getCudaDevicePtr();
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
        }

        torch::Tensor inputTensor = torch::from_blob(
                (void*)inputBufferCu->getCudaDevicePtr(), inputSizes,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        encoderInputs.at(0) = inputTensor;
        at::Tensor encodedTensor = encoderWrapper->module.forward(encoderInputs).toTensor();
        decoderInputs.at(1) = encodedTensor;
        at::Tensor similarityMetricTensor = decoderWrapper->module.forward(decoderInputs).toTensor();
        if (!similarityMetricTensor.is_contiguous()) {
            if (isFirstContiguousWarning) {
                sgl::Logfile::get()->writeWarning("Error in PyTorchDenoiser::denoise: Output tensor is not contiguous.");
                isFirstContiguousWarning = false;
            }
            similarityMetricTensor = similarityMetricTensor.contiguous();
        }
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
                outputImageBufferCu + batchOffset * sizeof(float),
                (CUdeviceptr)similarityMetricTensor.data_ptr(),
                sizeof(float) * batchSize, stream), "Error in cuMemcpyAsync: ");
    }

    deviceCacheEntry->getImageCudaExternalMemory()->memcpyCudaDtoA3DAsync(outputImageBufferCu, stream);

    cudaFinishedSemaphore->signalSemaphoreCuda(stream, timelineValue);
    cudaFinishedSemaphore->setWaitSemaphoreValue(timelineValue);
    sgl::vk::CommandBufferPtr postRenderCommandBuffer = postRenderCommandBuffers.at(frameIndex);
    renderer->pushCommandBuffer(postRenderCommandBuffer);
    renderer->beginCommandBuffer();
    postRenderCommandBuffer->pushWaitSemaphore(
            cudaFinishedSemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}

FilterDevice PyTorchSimilarityCalculator::PyTorchSimilarityCalculator::getFilterDevice() {
    if (pyTorchDevice == PyTorchDevice::CUDA) {
        return FilterDevice::CUDA;
    }
    return FilterDevice::CPU;
}

void PyTorchSimilarityCalculator::setPyTorchDevice(PyTorchDevice pyTorchDeviceNew) {
    if (pyTorchDeviceNew == pyTorchDevice) {
        return;
    }

    hasFilterDeviceChanged = true;
    dirty = true;
    pyTorchDevice = pyTorchDeviceNew;
    if (encoderWrapper) {
        encoderWrapper->module.to(getTorchDeviceType(pyTorchDevice));
    }
    if (decoderWrapper) {
        decoderWrapper->module.to(getTorchDeviceType(pyTorchDevice));
    }
}

void PyTorchSimilarityCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    EnsembleSimilarityCalculator::renderGuiImpl(propertyEditor);
    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChoosePyTorchModelFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathName(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPath(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilter(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            // Is this line data set or a volume data file for the scattering line tracer?
            const char* currentPath = IGFD_GetCurrentPath(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            filename += selection.table[0].fileName;
            IGFD_Selection_DestroyContent(&selection);
            if (currentPath) {
                free((void*)currentPath);
                currentPath = nullptr;
            }

            fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);

            if (modelSelectionIndex == 0) {
                modelFilePathEncoder = filename;
            } else {
                modelFilePathDecoder = filename;
            }
            loadModelFromFile(modelSelectionIndex, filename);
            dirty = true;
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    for (int i = 0; i < 2; i++) {
        propertyEditor.addInputAction(
                i == 0 ? ("Model Path (Encoder)##" + std::to_string(i)) : ("Model Path (Decoder)##" + std::to_string(i)),
                i == 0 ? &modelFilePathEncoder : &modelFilePathDecoder);
        if (propertyEditor.addButton("", "Load##" + std::to_string(i))) {
            loadModelFromFile(i, i == 0 ? modelFilePathEncoder : modelFilePathDecoder);
            dirty = true;
        }
        ImGui::SameLine();
        std::string buttonText = "Open from Disk...##" + std::to_string(i);
        if (ImGui::Button(buttonText.c_str())) {
            modelSelectionIndex = i;
            openModelSelectionDialog();
        }
    }

    PyTorchDevice pyTorchDeviceNew = pyTorchDevice;
    if (propertyEditor.addCombo(
            "Device", (int*)&pyTorchDeviceNew,
            PYTORCH_DEVICE_NAMES, IM_ARRAYSIZE(PYTORCH_DEVICE_NAMES))) {
        setPyTorchDevice(pyTorchDeviceNew);
        dirty = true;
    }
}

void PyTorchSimilarityCalculator::openModelSelectionDialog() {
    if (fileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(fileDialogDirectory)) {
        fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "PyTorch/";
        if (!sgl::FileUtils::get()->exists(fileDialogDirectory)) {
            fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
        }
    }
    IGFD_OpenModal(
            fileDialogInstance,
            "ChoosePyTorchModelFile", "Choose TorchScript Model File",
            ".*,.pt,.pth",
            fileDialogDirectory.c_str(),
            "", 1, nullptr,
            ImGuiFileDialogFlags_ConfirmOverwrite);
}



EnsembleCombinePass::EnsembleCombinePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void EnsembleCombinePass::setVolumeData(VolumeData *_volumeData, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.es = uint32_t(volumeData->getEnsembleMemberCount());
    uniformData.boundingBoxMin = volumeData->getBoundingBox().min;
    uniformData.boundingBoxMax = volumeData->getBoundingBox().min;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedEnsembleMemberCount != volumeData->getEnsembleMemberCount()) {
        cachedEnsembleMemberCount = volumeData->getEnsembleMemberCount();
        setShaderDirty();
    }
}

void EnsembleCombinePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews == _ensembleImageViews) {
        return;
    }
    ensembleImageViews = _ensembleImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    }
}

void EnsembleCombinePass::setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer) {
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void EnsembleCombinePass::setEnsembleMinMax(float minEnsembleVal, float maxEnsembleVal) {
    uniformData.minEnsembleVal = minEnsembleVal;
    uniformData.maxEnsembleVal = maxEnsembleVal;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void EnsembleCombinePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(volumeData->getEnsembleMemberCount())));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "CombineEnsembles.Compute" }, preprocessorDefines);
}

void EnsembleCombinePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}

void EnsembleCombinePass::_render() {
    renderer->dispatch(computeData, sgl::iceil(batchSize, computeBlockSize), 1, 1);
}



ReferenceEnsembleCombinePass::ReferenceEnsembleCombinePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void ReferenceEnsembleCombinePass::setVolumeData(VolumeData *_volumeData, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.es = uint32_t(volumeData->getEnsembleMemberCount());
    uniformData.boundingBoxMin = volumeData->getBoundingBox().min;
    uniformData.boundingBoxMax = volumeData->getBoundingBox().min;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedEnsembleMemberCount != volumeData->getEnsembleMemberCount()) {
        cachedEnsembleMemberCount = volumeData->getEnsembleMemberCount();
        setShaderDirty();
    }
}

void ReferenceEnsembleCombinePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews == _ensembleImageViews) {
        return;
    }
    ensembleImageViews = _ensembleImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    }
}

void ReferenceEnsembleCombinePass::setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer) {
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void ReferenceEnsembleCombinePass::setEnsembleMinMax(float minEnsembleVal, float maxEnsembleVal) {
    uniformData.minEnsembleVal = minEnsembleVal;
    uniformData.maxEnsembleVal = maxEnsembleVal;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void ReferenceEnsembleCombinePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(volumeData->getEnsembleMemberCount())));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "CombineEnsembles.Compute.Reference" }, preprocessorDefines);
}

void ReferenceEnsembleCombinePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}

void ReferenceEnsembleCombinePass::_render() {
    renderer->dispatch(computeData, sgl::iceil(volumeData->getEnsembleMemberCount(), computeBlockSize), 1, 1);
}
