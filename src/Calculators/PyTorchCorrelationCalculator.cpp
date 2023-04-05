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

#include <chrono>

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
#include "PyTorchCorrelationCalculator.hpp"

#if CUDA_VERSION < 11020
#error CUDA >= 11.2 is required for timeline semaphore support.
#endif

struct ModuleWrapper {
    torch::jit::Module module;
    torch::jit::Module frozenModule;
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

PyTorchCorrelationCalculator::PyTorchCorrelationCalculator(sgl::vk::Renderer* renderer)
        : ICorrelationCalculator(renderer) {
    sgl::vk::Device* device = renderer->getDevice();

#ifdef SUPPORT_CUDA_INTEROP
    // Support CUDA on NVIDIA GPUs using the proprietary driver.
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY && torch::cuda::is_available()) {
        pyTorchDevice = PyTorchDevice::CUDA;
        if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            sgl::Logfile::get()->throwError(
                    "Error in PyTorchCorrelationCalculator::PyTorchCorrelationCalculator: "
                    "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
        }
        uint8_t* moduleBuffer = nullptr;
        size_t bufferSize = 0;
        sgl::loadFileFromSource(
                sgl::AppSettings::get()->getDataDirectory() + "/__cudacache__/CombineEnsembles.fatbin",
                moduleBuffer, bufferSize, true);
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleLoadFatBinary(
                &combineCorrelationMembersModuleCu, moduleBuffer), "Error in cuModuleLoadFatBinary: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &combineCorrelationMembersFunctionCu, combineCorrelationMembersModuleCu, "combineEnsembles"), "Error in cuModuleGetFunction: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &memcpyFloatClampToZeroFunctionCu, combineCorrelationMembersModuleCu, "memcpyFloatClampToZero"), "Error in cuModuleGetFunction: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &writeGridPositionsFunctionCu, combineCorrelationMembersModuleCu, "writeGridPositions"), "Error in cuModuleGetFunction: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &writeGridPositionReferenceFunctionCu, combineCorrelationMembersModuleCu, "writeGridPositionReference"), "Error in cuModuleGetFunction: ");
        delete[] moduleBuffer;
    }
#endif

    sgl::AppSettings::get()->getSettings().getValueOpt(
            "pyTorchCorrelationCalculatorModelFilePathEncoder", modelFilePathEncoder);
    sgl::AppSettings::get()->getSettings().getValueOpt(
            "pyTorchCorrelationCalculatorModelFilePathDecoder", modelFilePathDecoder);
    if (sgl::FileUtils::get()->exists(modelFilePathEncoder)
            && !sgl::FileUtils::get()->isDirectory(modelFilePathEncoder)) {
        loadModelFromFile(0, modelFilePathEncoder);
    }
    if (sgl::FileUtils::get()->exists(modelFilePathDecoder)
            && !sgl::FileUtils::get()->isDirectory(modelFilePathDecoder)) {
        loadModelFromFile(1, modelFilePathDecoder);
    }

    referenceCorrelationMembersCombinePass = std::make_shared<ReferenceCorrelationMembersCombinePass>(renderer);
    correlationMembersCombinePass = std::make_shared<CorrelationMembersCombinePass>(renderer);
}

PyTorchCorrelationCalculator::~PyTorchCorrelationCalculator() {
    sgl::AppSettings::get()->getSettings().addKeyValue(
            "pyTorchCorrelationCalculatorModelFilePathEncoder", modelFilePathEncoder);
    sgl::AppSettings::get()->getSettings().addKeyValue(
            "pyTorchCorrelationCalculatorModelFilePathDecoder", modelFilePathDecoder);

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
    if (fieldTextureArrayCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                fieldTextureArrayCu), "Error in cuMemFree: ");
    }
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleUnload(
                combineCorrelationMembersModuleCu), "Error in cuModuleUnload: ");
    }
}

void PyTorchCorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    ICorrelationCalculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::TORCH);
    }

    referenceCorrelationMembersCombinePass->setVolumeData(volumeData, getCorrelationMemberCount(), isNewData);
    correlationMembersCombinePass->setVolumeData(volumeData, getCorrelationMemberCount(), isNewData);

    /*sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (referenceInputValues) {
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

void PyTorchCorrelationCalculator::onCorrelationMemberCountChanged() {
    int cs = getCorrelationMemberCount();
    referenceCorrelationMembersCombinePass->setCorrelationMemberCount(cs);
    correlationMembersCombinePass->setCorrelationMemberCount(cs);
}

void PyTorchCorrelationCalculator::clearFieldDeviceData() {
    referenceCorrelationMembersCombinePass->setFieldImageViews({});
    correlationMembersCombinePass->setFieldImageViews({});
}

bool PyTorchCorrelationCalculator::getSupportsBufferMode() {
    dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
    return false;
}

bool PyTorchCorrelationCalculator::getSupportsSeparateFields() {
    useSeparateFields = false;
    return false;
}

bool PyTorchCorrelationCalculator::loadModelFromFile(int idx, const std::string& modelPath) {
    torch::DeviceType deviceType = getTorchDeviceType(pyTorchDevice);
    torch::jit::ExtraFilesMap extraFilesMap;
    extraFilesMap["model_info.json"] = "";
    auto& wrapper = idx == 0 ? encoderWrapper : decoderWrapper;
    wrapper = std::make_shared<ModuleWrapper>();
    try {
        // std::shared_ptr<torch::jit::script::Module>
        wrapper->module = torch::jit::load(modelPath, deviceType, extraFilesMap);
        wrapper->module.to(deviceType);
        wrapper->module.eval();
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

    wrapper->frozenModule = optimize_for_inference(wrapper->module);

    auto it = extraFilesMap.find("model_info.json");
    if (it != extraFilesMap.end()) {
        Json::CharReaderBuilder builder;
        std::unique_ptr<Json::CharReader> const charReader(builder.newCharReader());
        JSONCPP_STRING errorString;
        Json::Value root;
        if (!charReader->parse(
                it->second.c_str(), it->second.c_str() + it->second.size(), &root, &errorString)) {
            sgl::Logfile::get()->writeError("Error in PyTorchCorrelationCalculator::loadModelFromFile: " + errorString);
            wrapper = {};
            return false;
        }

        /*
         * Example: { "network_type": "MINE_SRN" }
         */
        if (root.isMember("network_type")) {
            Json::Value& networkTypeNode = root["network_type"];
            if (!networkTypeNode.isString()) {
                sgl::Logfile::get()->writeError(
                        "Error: Array 'network_type' could not be found in the model_info.json file of the PyTorch "
                        "module loaded from \"" + modelPath + "\"!");
                wrapper = {};
                return false;
            }

            auto networkTypeName = networkTypeNode.asString();
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
                        "Error in PyTorchCorrelationCalculator::loadModelFromFile: Invalid network type \""
                        + networkTypeName + "\".");
                wrapper = {};
                return false;
            }
        }
    }

    dirty = true;
    return true;
}

void PyTorchCorrelationCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    torch::NoGradGuard noGradGuard{};

    if (!encoderWrapper || !decoderWrapper) {
        memset(buffer, 0, volumeData->getSlice3dSizeInBytes(FieldType::SCALAR));
        sgl::Logfile::get()->writeWarning(
                "Warning in PyTorchCorrelationCalculator::calculateCpu: Encoder or decoder module is not loaded.", true);
        return;
    }

    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int cs = networkType == NetworkType::MINE ? getCorrelationMemberCount() : 0;
    auto ucs = uint32_t(cs);

    if (cachedCorrelationMemberCountHost != size_t(cs)) {
        if (cachedCorrelationMemberCountHost != std::numeric_limits<size_t>::max()) {
            delete[] referenceInputValues;
            delete[] batchInputValues;
        }
        cachedCorrelationMemberCountHost = size_t(cs);
        if (networkType == NetworkType::MINE) {
            referenceInputValues = new float[cs * 4];
            batchInputValues = new float[cs * 4 * mineBatchSize1D];
        } else {
            referenceInputValues = new float[3];
            batchInputValues = new float[3 * srnBatchSize1D];
        }
    }

    float minEnsembleVal = std::numeric_limits<float>::max();
    float maxEnsembleVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::HostCacheEntry> ensembleEntryFields;
    std::vector<const float*> ensembleFields;
    if (networkType == NetworkType::MINE) {
        ensembleEntryFields.reserve(cs);
        ensembleFields.reserve(cs);
        for (ensembleIdx = 0; ensembleIdx < cs; ensembleIdx++) {
            VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                    FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
            auto [minVal, maxVal] = volumeData->getMinMaxScalarFieldValue(
                    scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
            minEnsembleVal = std::min(minEnsembleVal, minVal);
            maxEnsembleVal = std::max(maxEnsembleVal, maxVal);
            ensembleEntryFields.push_back(ensembleEntryField);
            ensembleFields.push_back(ensembleEntryField->data<float>());
        }
    }

    size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
    glm::vec3 referencePointNorm =
            glm::vec3(referencePointIndex) / glm::vec3(xs - 1, ys - 1, zs - 1) * 2.0f - glm::vec3(1.0f);
    std::vector<int64_t> referenceInputSizes;
    std::vector<int64_t> inputSizes;
    if (networkType == NetworkType::MINE) {
        for (int c = 0; c < cs; c++) {
            referenceInputValues[c * 4] =
                    (ensembleFields.at(c)[referencePointIdx] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
            referenceInputValues[c * 4 + 1] = referencePointNorm.x;
            referenceInputValues[c * 4 + 2] = referencePointNorm.y;
            referenceInputValues[c * 4 + 3] = referencePointNorm.z;
        }
        referenceInputSizes = {1, cs, 4 };
        inputSizes = {mineBatchSize1D, cs, 4 };
    } else {
        referenceInputValues[0] = referencePointNorm.x;
        referenceInputValues[1] = referencePointNorm.y;
        referenceInputValues[2] = referencePointNorm.z;
        referenceInputSizes = { 1, 3 };
        inputSizes = { srnBatchSize1D, 3 };
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

    uint32_t batchSize1D = networkType == NetworkType::MINE ? mineBatchSize1D : srnBatchSize1D;
    uint32_t numSliceEntries = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(batchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * batchSize1D;
        uint32_t batchSize = std::min(uint32_t(batchSize1D), numSliceEntries - batchOffset);

        if (networkType == NetworkType::MINE) {
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<uint32_t>(0, batchSize), [&](auto const& r) {
                for (auto pointIdx = r.begin(); pointIdx != r.end(); pointIdx++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for default(none) shared(xs, ys, zs, cs, ucs, batchSize, batchOffset) \
            shared(ensembleFields, minEnsembleVal, maxEnsembleVal)
#endif
            for (uint32_t pointIdx = 0; pointIdx < batchSize; pointIdx++) {
#endif
                uint32_t pointIdxWriteOffset = pointIdx * cs * 4;
                uint32_t pointIdxReadOffset = pointIdx + batchOffset;
                uint32_t x = pointIdxReadOffset % uint32_t(xs);
                uint32_t y = (pointIdxReadOffset / uint32_t(xs)) % uint32_t(ys);
                uint32_t z = pointIdxReadOffset / uint32_t(xs * ys);
                glm::vec3 pointNorm = glm::vec3(x, y, z) / glm::vec3(xs - 1, ys - 1, zs - 1) * 2.0f - glm::vec3(1.0f);
                for (uint32_t e = 0; e < ucs; e++) {
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
        } else {
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<uint32_t>(0, batchSize), [&](auto const& r) {
                for (auto pointIdx = r.begin(); pointIdx != r.end(); pointIdx++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for default(none) shared(xs, ys, zs, cs, ucs, batchSize, batchOffset) \
            shared(ensembleFields, minEnsembleVal, maxEnsembleVal)
#endif
            for (uint32_t pointIdx = 0; pointIdx < batchSize; pointIdx++) {
#endif
                uint32_t pointIdxWriteOffset = pointIdx * 3;
                uint32_t pointIdxReadOffset = pointIdx + batchOffset;
                uint32_t x = pointIdxReadOffset % uint32_t(xs);
                uint32_t y = (pointIdxReadOffset / uint32_t(xs)) % uint32_t(ys);
                uint32_t z = pointIdxReadOffset / uint32_t(xs * ys);
                glm::vec3 pointNorm = glm::vec3(x, y, z) / glm::vec3(xs - 1, ys - 1, zs - 1) * 2.0f - glm::vec3(1.0f);
                batchInputValues[pointIdxWriteOffset] = pointNorm.x;
                batchInputValues[pointIdxWriteOffset + 1] = pointNorm.y;
                batchInputValues[pointIdxWriteOffset + 2] = pointNorm.z;
            }
#ifdef USE_TBB
            });
#endif
        }

        torch::Tensor inputTensor = torch::from_blob(
                batchInputValues, inputSizes,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        encoderInputs.at(0) = inputTensor;
        at::Tensor encodedTensor = encoderWrapper->module.forward(encoderInputs).toTensor();
        decoderInputs.at(1) = encodedTensor;
        at::Tensor correlationMetricTensor = decoderWrapper->module.forward(decoderInputs).toTensor();
        if (!correlationMetricTensor.is_contiguous()) {
            if (isFirstContiguousWarning) {
                sgl::Logfile::get()->writeWarning("Error in PyTorchDenoiser::denoise: Output tensor is not contiguous.");
                isFirstContiguousWarning = false;
            }
            correlationMetricTensor = correlationMetricTensor.contiguous();
        }
        memcpy(buffer + batchOffset, correlationMetricTensor.data_ptr(), sizeof(float) * batchSize);

        // Clamp values to zero.
        float* bufferFlt = buffer + batchOffset;
        for (uint32_t i = 0; i < batchSize; i++) {
            if (bufferFlt[i] < 0.0f) {
                bufferFlt[i] = 0.0f;
            }
        }
    }
}

#ifdef SUPPORT_CUDA_INTEROP
void PyTorchCorrelationCalculator::calculateDevice(
        int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    torch::NoGradGuard noGradGuard{};

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);

    if (!encoderWrapper || !decoderWrapper) {
        deviceCacheEntry->getVulkanImage()->clearColor(glm::vec4(0.0f), renderer->getVkCommandBuffer());
        sgl::Logfile::get()->writeWarning(
                "Warning in PyTorchCorrelationCalculator::calculateCpu: Encoder or decoder module is not loaded.",
                true);
        return;
    }

    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int cs = networkType == NetworkType::MINE ? getCorrelationMemberCount() : 0;

    int gpuBatchSize1D;
    if (networkType == NetworkType::MINE) {
        gpuBatchSize1D = gpuBatchSize1DBase;
        if (cs >= 200) {
            gpuBatchSize1D /= 2;
        }
        if (cs >= 400) {
            gpuBatchSize1D /= 2;
        }
        if (cs >= 800) {
            gpuBatchSize1D /= 2;
        }
    } else {
        gpuBatchSize1D = srnGpuBatchSize1DBase;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    size_t volumeDataSlice3dSize = volumeData->getSlice3dSizeInBytes(FieldType::SCALAR);
    if (cachedVolumeDataSlice3dSize != volumeDataSlice3dSize) {
        if (outputImageBufferCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    outputImageBufferCu, stream), "Error in cuMemFreeAsync: ");
        }
        cachedVolumeDataSlice3dSize = volumeDataSlice3dSize;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &outputImageBufferCu, volumeDataSlice3dSize, stream), "Error in cuMemAllocAsync: ");
    }

    if (cachedCorrelationMemberCountDevice != size_t(cs)) {
        referenceInputBufferCu = {};
        inputBufferCu = {};
        if (fieldTextureArrayCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    fieldTextureArrayCu, stream), "Error in cuMemFreeAsync: ");
        }
        cachedCorrelationMemberCountDevice = size_t(cs);

        if (networkType == NetworkType::MINE) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                    &fieldTextureArrayCu, cs * sizeof(CUtexObject), stream), "Error in cuMemAllocAsync: ");
            auto referenceInputBufferVk = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), sizeof(float) * cs * 4,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                    true, true);
            referenceInputBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(referenceInputBufferVk);
            auto inputBufferVk = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), sizeof(float) * gpuBatchSize1D * cs * 4,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                    true, true);
            inputBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(inputBufferVk);

            correlationMembersCombinePass->setOutputBuffer(inputBufferCu->getVulkanBuffer());
            referenceCorrelationMembersCombinePass->setOutputBuffer(referenceInputBufferCu->getVulkanBuffer());
        } else {
            auto referenceInputBufferVk = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), sizeof(float) * 3,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                    true, true);
            referenceInputBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(referenceInputBufferVk);
            auto inputBufferVk = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), sizeof(float) * gpuBatchSize1D * 3,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                    true, true);
            inputBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(inputBufferVk);
        }
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

    float minEnsembleVal = std::numeric_limits<float>::max();
    float maxEnsembleVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::DeviceCacheEntry> ensembleEntryFields;
    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    std::vector<CUtexObject> ensembleTexturesCu;
    if (networkType == NetworkType::MINE) {
#ifdef TEST_INFERENCE_SPEED
        auto startLoad = std::chrono::system_clock::now();
#endif

        ensembleEntryFields.reserve(cs);
        ensembleImageViews.reserve(cs);
        ensembleTexturesCu.reserve(cs);
        for (ensembleIdx = 0; ensembleIdx < cs; ensembleIdx++) {
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

        referenceCorrelationMembersCombinePass->setFieldImageViews(ensembleImageViews);
        referenceCorrelationMembersCombinePass->setFieldMinMax(minEnsembleVal, maxEnsembleVal);
        if (createBatchesWithVulkan) {
            correlationMembersCombinePass->setFieldImageViews(ensembleImageViews);
            correlationMembersCombinePass->setFieldMinMax(minEnsembleVal, maxEnsembleVal);
            correlationMembersCombinePass->setBatchSize(gpuBatchSize1D);
        } else {
            if (cachedFieldTexturesCu != ensembleTexturesCu) {
                cachedFieldTexturesCu = ensembleTexturesCu;
                sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                        stream), "Error in cuStreamSynchronize: ");
                sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                        fieldTextureArrayCu, ensembleTexturesCu.data(), sizeof(CUtexObject) * cs),
                                       "Error in cuMemcpyHtoD: ");
            }
        }

#ifdef TEST_INFERENCE_SPEED
        auto endLoad = std::chrono::system_clock::now();
        auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
        std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif
    }

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    if (networkType == NetworkType::MINE) {
        referenceCorrelationMembersCombinePass->buildIfNecessary();
        renderer->pushConstants(
                referenceCorrelationMembersCombinePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                referencePointIndex);
        referenceCorrelationMembersCombinePass->render();
        renderer->insertMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, 0,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    }

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

    vulkanFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);

    if (networkType == NetworkType::SRN_MINE) {
        CUdeviceptr outputBufferRef = referenceInputBufferCu->getCudaDevicePtr();
        void* kernelParameters[] = {
                &xs, &ys, &zs, &referencePointIndex.x, &outputBufferRef
        };
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                writeGridPositionReferenceFunctionCu,
                1, 1, 1, //< Grid size.
                1, 1, 1, //< Block size.
                0, //< Dynamic shared memory size.
                stream,
                kernelParameters, //< Kernel parameters.
                nullptr //< Extra (empty).
        ), "Error in cuLaunchKernel: ");
    }

    std::vector<int64_t> referenceInputSizes;
    std::vector<int64_t> inputSizes;
    if (networkType == NetworkType::MINE) {
        referenceInputSizes = {1, cs, 4 };
        inputSizes = {gpuBatchSize1D, cs, 4 };
    } else {
        referenceInputSizes = { 1, 3 };
        inputSizes = { gpuBatchSize1D, 3 };
    }

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


    uint32_t numSliceEntries = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(gpuBatchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * gpuBatchSize1D;
        uint32_t batchSize = std::min(uint32_t(gpuBatchSize1D), numSliceEntries - batchOffset);

        if (networkType == NetworkType::MINE && createBatchesWithVulkan) {
            correlationMembersCombinePass->buildIfNecessary();
            renderer->pushConstants(
                    correlationMembersCombinePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                    0, batchOffset);
            renderer->pushConstants(
                    correlationMembersCombinePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                    sizeof(uint32_t), batchSize);
            correlationMembersCombinePass->render();
        } else {
            CUdeviceptr outputBuffer = inputBufferCu->getCudaDevicePtr();
            CUdeviceptr scalarFields = fieldTextureArrayCu;
            std::vector<void*> kernelParameters;
            CUfunction queryInputAssemblyFunction;
            uint32_t srnStride = 3;
            if (networkType == NetworkType::MINE) {
                kernelParameters = {
                        &xs, &ys, &zs, &cs, &batchOffset, &batchSize, &minEnsembleVal, &maxEnsembleVal,
                        &outputBuffer, &scalarFields
                };
                queryInputAssemblyFunction = combineCorrelationMembersFunctionCu;
            } else {
                kernelParameters = {
                        &xs, &ys, &zs, &batchOffset, &batchSize, &outputBuffer, &srnStride
                };
                queryInputAssemblyFunction = writeGridPositionsFunctionCu;
            }
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                    queryInputAssemblyFunction,
                    sgl::uiceil(batchSize, 256), 1, 1, //< Grid size.
                    256, 1, 1, //< Block size.
                    0, //< Dynamic shared memory size.
                    stream,
                    kernelParameters.data(), //< Kernel parameters.
                    nullptr //< Extra (empty).
            ), "Error in cuLaunchKernel: ");
        }

        torch::Tensor inputTensor = torch::from_blob(
                (void*)inputBufferCu->getCudaDevicePtr(), inputSizes,
                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        encoderInputs.at(0) = inputTensor;
        at::Tensor encodedTensor = encoderWrapper->module.forward(encoderInputs).toTensor();
        decoderInputs.at(1) = encodedTensor;
        at::Tensor correlationMetricTensor = decoderWrapper->module.forward(decoderInputs).toTensor();
        if (!correlationMetricTensor.is_contiguous()) {
            if (isFirstContiguousWarning) {
                sgl::Logfile::get()->writeWarning("Error in PyTorchDenoiser::denoise: Output tensor is not contiguous.");
                isFirstContiguousWarning = false;
            }
            correlationMetricTensor = correlationMetricTensor.contiguous();
        }
        //sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyAsync(
        //        outputImageBufferCu + batchOffset * sizeof(float),
        //        (CUdeviceptr)correlationMetricTensor.data_ptr(),
        //        sizeof(float) * batchSize, stream), "Error in cuMemcpyAsync: ");

        CUdeviceptr outputBuffer = outputImageBufferCu + batchOffset * sizeof(float);
        auto inputBuffer = (CUdeviceptr)correlationMetricTensor.data_ptr();
        void* kernelParametersCopy[] = { &outputBuffer, &inputBuffer, &batchSize };
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                memcpyFloatClampToZeroFunctionCu,
                sgl::uiceil(batchSize, 256), 1, 1, //< Grid size.
                256, 1, 1, //< Block size.
                0, //< Dynamic shared memory size.
                stream,
                kernelParametersCopy, //< Kernel parameters.
                nullptr //< Extra (empty).
        ), "Error in cuLaunchKernel: ");
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
#endif

FilterDevice PyTorchCorrelationCalculator::PyTorchCorrelationCalculator::getFilterDevice() {
#ifdef SUPPORT_CUDA_INTEROP
    if (pyTorchDevice == PyTorchDevice::CUDA) {
        return FilterDevice::CUDA;
    }
#endif
    return FilterDevice::CPU;
}

void PyTorchCorrelationCalculator::setPyTorchDevice(PyTorchDevice pyTorchDeviceNew) {
    if (pyTorchDeviceNew == pyTorchDevice) {
        return;
    }

    hasFilterDeviceChanged = true;
    dirty = true;
    pyTorchDevice = pyTorchDeviceNew;
    if (encoderWrapper) {
        encoderWrapper->module.to(getTorchDeviceType(pyTorchDevice));
        encoderWrapper->frozenModule = optimize_for_inference(encoderWrapper->module);
    }
    if (decoderWrapper) {
        decoderWrapper->module.to(getTorchDeviceType(pyTorchDevice));
        decoderWrapper->frozenModule = optimize_for_inference(decoderWrapper->module);
   }
}

void PyTorchCorrelationCalculator::renderGuiImplSub(sgl::PropertyEditor& propertyEditor) {
    ICorrelationCalculator::renderGuiImplSub(propertyEditor);
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

void PyTorchCorrelationCalculator::openModelSelectionDialog() {
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



CorrelationMembersCombinePass::CorrelationMembersCombinePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void CorrelationMembersCombinePass::setVolumeData(
        VolumeData *_volumeData, int correlationMemberCount, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.cs = uint32_t(correlationMemberCount);
    uniformData.boundingBoxMin = volumeData->getBoundingBox().min;
    uniformData.boundingBoxMax = volumeData->getBoundingBox().min;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
    }
}

void CorrelationMembersCombinePass::setCorrelationMemberCount(int correlationMemberCount) {
    uniformData.cs = uint32_t(correlationMemberCount);
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
    }
}

void CorrelationMembersCombinePass::setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews) {
    if (fieldImageViews == _fieldImageViews) {
        return;
    }
    fieldImageViews = _fieldImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(fieldImageViews, "scalarFields");
    }
}

void CorrelationMembersCombinePass::setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer) {
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void CorrelationMembersCombinePass::setFieldMinMax(float minFieldVal, float maxFieldVal) {
    uniformData.minFieldVal = minFieldVal;
    uniformData.maxFieldVal = maxFieldVal;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void CorrelationMembersCombinePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    preprocessorDefines.insert(std::make_pair(
            "MEMBER_COUNT", std::to_string(cachedCorrelationMemberCount)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "CombineEnsembles.Compute" }, preprocessorDefines);
}

void CorrelationMembersCombinePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(fieldImageViews, "scalarFields");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}

void CorrelationMembersCombinePass::_render() {
    renderer->dispatch(computeData, sgl::iceil(batchSize, computeBlockSize), 1, 1);
}



ReferenceCorrelationMembersCombinePass::ReferenceCorrelationMembersCombinePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void ReferenceCorrelationMembersCombinePass::setVolumeData(
        VolumeData *_volumeData, int correlationMemberCount, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.cs = uint32_t(correlationMemberCount);
    uniformData.boundingBoxMin = volumeData->getBoundingBox().min;
    uniformData.boundingBoxMax = volumeData->getBoundingBox().min;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
    }
}

void ReferenceCorrelationMembersCombinePass::setCorrelationMemberCount(int correlationMemberCount) {
    uniformData.cs = uint32_t(correlationMemberCount);
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
    }
}

void ReferenceCorrelationMembersCombinePass::setFieldImageViews(
        const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews) {
    if (fieldImageViews == _fieldImageViews) {
        return;
    }
    fieldImageViews = _fieldImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(fieldImageViews, "scalarFields");
    }
}

void ReferenceCorrelationMembersCombinePass::setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer) {
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void ReferenceCorrelationMembersCombinePass::setFieldMinMax(float minFieldVal, float maxFieldVal) {
    uniformData.minFieldVal = minFieldVal;
    uniformData.maxFieldVal = maxFieldVal;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void ReferenceCorrelationMembersCombinePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(computeBlockSize)));
    preprocessorDefines.insert(std::make_pair(
            "MEMBER_COUNT", std::to_string(cachedCorrelationMemberCount)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "CombineEnsembles.Compute.Reference" }, preprocessorDefines);
}

void ReferenceCorrelationMembersCombinePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(fieldImageViews, "scalarFields");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}

void ReferenceCorrelationMembersCombinePass::_render() {
    renderer->dispatch(computeData, sgl::iceil(cachedCorrelationMemberCount, computeBlockSize), 1, 1);
}
