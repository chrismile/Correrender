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

#include <iostream>
#include <boost/algorithm/string/case_conv.hpp>
#include <json/json.h>

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/File/Archive.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Volume/VolumeData.hpp"
#include "Volume/Cache/DeviceCacheEntry.hpp"
#include "DeepLearningCudaCorrelationCalculator.hpp"

#if CUDA_VERSION < 11020
#error CUDA >= 11.2 is required for timeline semaphore support.
#endif

DeepLearningCudaCorrelationCalculator::DeepLearningCudaCorrelationCalculator(
        const std::string& implName, const std::string& implNameKey, sgl::vk::Renderer* renderer)
        : ICorrelationCalculator(renderer), implName(implName), implNameKey(implNameKey) {
    implNameKeyUpper = implNameKey;
    std::string firstCharUpper = boost::to_upper_copy(implNameKeyUpper);
    implNameKeyUpper.at(0) = firstCharUpper.at(0);
    fileDialogKey = "Choose" + implNameKeyUpper + "ModelFile";
    fileDialogDescription = "Choose " + implName + " Model File";
    modelFilePathSettingsKey = implNameKey + "CorrelationCalculatorModelFilePath";

    sgl::vk::Device* device = renderer->getDevice();
    if (device->getDeviceDriverId() != VK_DRIVER_ID_NVIDIA_PROPRIETARY
            || !sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::Logfile::get()->throwError(
                "Error in DeepLearningCudaCorrelationCalculator::DeepLearningCudaCorrelationCalculator: "
                "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
    }

    std::string implNameKeyLower = boost::to_lower_copy(implNameKeyUpper);
    std::string modelPresetsJsonFilename =
            sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/models-" + implNameKeyLower + ".json";
    if (sgl::FileUtils::get()->exists(modelPresetsJsonFilename)) {
        parseModelPresetsFile(modelPresetsJsonFilename);
    }
}

void DeepLearningCudaCorrelationCalculator::parseModelPresetsFile(const std::string& filename) {
    // Parse the passed JSON file.
    std::ifstream jsonFileStream(filename.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return;
    }
    jsonFileStream.close();

    modelPresets.emplace_back("---");
    modelPresetFilenames.emplace_back("---");

    DataSetInformationPtr dataSetInformationRoot(new DataSetInformation);
    Json::Value& modelsNode = root["models"];
    for (Json::Value& model : modelsNode) {
        modelPresets.push_back(model["name"].asString());
        modelPresetFilenames.push_back(model["filename"].asString());
    }

    if (modelPresets.size() == 1) {
        modelPresets.clear();
        modelPresetFilenames.clear();
    }
}

void DeepLearningCudaCorrelationCalculator::initialize() {
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamCreate(
            &stream, 0), "Error in cuStreamCreate: ");

    uint8_t* moduleBuffer = nullptr;
    size_t bufferSize = 0;
    sgl::loadFileFromSource(
            sgl::AppSettings::get()->getDataDirectory() + "/__cudacache__/CombineCorrelationMembers.fatbin",
            moduleBuffer, bufferSize, true);
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleLoadFatBinary(
            &combineCorrelationMembersModuleCu, moduleBuffer), "Error in cuModuleLoadFatBinary: ");

    // Functions that take a 3D image array as an input.
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembers"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersReferenceFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersReference"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersAlignedFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersAligned"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersReferenceAlignedFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersReferenceAligned"), "Error in cuModuleGetFunction: ");

    // Functions that take a buffer as an input.
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersBufferFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersBuffer"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersReferenceBufferFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersReferenceBuffer"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersAlignedBufferFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersAlignedBuffer"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersReferenceAlignedBufferFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersReferenceAlignedBuffer"), "Error in cuModuleGetFunction: ");

    // Functions that take a tiled buffer as an input.
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersBufferTiledFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersBufferTiled"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersReferenceBufferTiledFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersReferenceBufferTiled"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersAlignedBufferTiledFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersAlignedBufferTiled"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &combineCorrelationMembersReferenceAlignedBufferTiledFunctionCu, combineCorrelationMembersModuleCu,
            "combineCorrelationMembersReferenceAlignedBufferTiled"), "Error in cuModuleGetFunction: ");

    // For networkType == NetworkType::SRN_MINE.
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &writeGridPositionsFunctionCu, combineCorrelationMembersModuleCu,
            "writeGridPositions"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &writeGridPositionsStencilFunctionCu, combineCorrelationMembersModuleCu,
            "writeGridPositionsStencil"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &writeGridPositionReferenceFunctionCu, combineCorrelationMembersModuleCu,
            "writeGridPositionReference"), "Error in cuModuleGetFunction: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
            &unpackStencilValuesFunctionCu, combineCorrelationMembersModuleCu,
            "unpackStencilValues"), "Error in cuModuleGetFunction: ");
    delete[] moduleBuffer;

    sgl::AppSettings::get()->getSettings().getValueOpt(modelFilePathSettingsKey.c_str(), modelFilePath);
    if (sgl::FileUtils::get()->exists(modelFilePath) && !sgl::FileUtils::get()->isDirectory(modelFilePath)) {
        loadModelFromFile(modelFilePath);
    }
}

DeepLearningCudaCorrelationCalculator::~DeepLearningCudaCorrelationCalculator() {
    sgl::AppSettings::get()->getSettings().addKeyValue(modelFilePathSettingsKey, modelFilePath);

    if (permutationIndicesBufferCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                permutationIndicesBufferCu), "Error in cuMemFree: ");
    }
    if (outputImageBufferCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                outputImageBufferCu), "Error in cuMemFree: ");
    }
    if (outputImageBufferUnpackedCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                outputImageBufferUnpackedCu), "Error in cuMemFree: ");
    }
    if (fieldTextureArrayCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                fieldTextureArrayCu), "Error in cuMemFree: ");
    }
    if (fieldBufferArrayCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                fieldBufferArrayCu), "Error in cuMemFree: ");
    }
    if (nonNanIndexBufferCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                nonNanIndexBufferCu), "Error in cuMemFree: ");
    }
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleUnload(
            combineCorrelationMembersModuleCu), "Error in cuModuleUnload: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamDestroy(
            stream), "Error in cuStreamDestroy: ");
}

void DeepLearningCudaCorrelationCalculator::renderGuiImplSub(sgl::PropertyEditor& propertyEditor) {
    ICorrelationCalculator::renderGuiImplSub(propertyEditor);
    if (IGFD_DisplayDialog(
            fileDialogInstance,
            fileDialogKey.c_str(), ImGuiWindowFlags_NoCollapse,
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

            modelFilePath = filename;
            loadModelFromFile(filename);
            dirty = true;
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    propertyEditor.addInputAction("Model Path", &modelFilePath);
    if (propertyEditor.addButton("", "Load")) {
        loadModelFromFile(modelFilePath);
        dirty = true;
    }
    ImGui::SameLine();
    std::string buttonText = "Open from Disk...";
    if (ImGui::Button(buttonText.c_str())) {
        if (fileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(fileDialogDirectory)) {
            fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + implName + "/";
            if (!sgl::FileUtils::get()->exists(fileDialogDirectory)) {
                fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
            }
        }
        IGFD_OpenModal(
                fileDialogInstance,
                fileDialogKey.c_str(), fileDialogDescription.c_str(),
                ".*,.zip,.7z,.tar,.tar.zip,.tar.gz,.tar.bz2,.tar.xz,.tar.lzma,.tar.7z",
                fileDialogDirectory.c_str(),
                "", 1, nullptr,
                ImGuiFileDialogFlags_ConfirmOverwrite);
    }

    if (!modelPresets.empty() && propertyEditor.addCombo(
            "Model Presets", &modelPresetIndex, modelPresets.data(), int(modelPresets.size()))) {
        if (modelPresetIndex != 0) {
            modelFilePath = modelPresetFilenames.at(modelPresetIndex);
            loadModelFromFile(modelFilePath);
            dirty = true;
        }
    }

    if (!isMutualInformationData && propertyEditor.addCheckbox("Absolute Value", &calculateAbsoluteValue)) {
        dirty = true;
    }

    // TODO
    //propertyEditor.addText("Data type:", "Float");
}

void DeepLearningCudaCorrelationCalculator::renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) {
    ICorrelationCalculator::renderGuiImplAdvanced(propertyEditor);
    if (networkType != NetworkType::MINE && propertyEditor.addCheckbox(
            "Use Data NaN Stencil", &useDataNanStencil)) {
        clearFieldDeviceData();
        dirty = true;
    }
}

void DeepLearningCudaCorrelationCalculator::setSettings(const SettingsMap& settings) {
    ICorrelationCalculator::setSettings(settings);
    if (settings.getValueOpt("model_file_path", modelFilePath)) {
        dirty = true;
    }
    if (settings.getValueOpt("model_preset_index", modelPresetIndex)) {
        modelPresetIndex = std::clamp(modelPresetIndex, 0, int(modelPresetFilenames.size()) - 1);
        modelFilePath = modelPresetFilenames.at(modelPresetIndex);
        dirty = true;
    }
    if (settings.getValueOpt("calculate_absolute_value", calculateAbsoluteValue)) {
        dirty = true;
    }
    if (settings.getValueOpt("use_data_nan_stencil", useDataNanStencil)) {
        dirty = true;
    }
}

void DeepLearningCudaCorrelationCalculator::getSettings(SettingsMap& settings) {
    ICorrelationCalculator::getSettings(settings);
    settings.addKeyValue("model_file_path", modelFilePath);
    settings.addKeyValue("model_preset_index", modelPresetIndex);
    settings.addKeyValue("calculate_absolute_value", calculateAbsoluteValue);
    settings.addKeyValue("use_data_nan_stencil", useDataNanStencil);
}

bool DeepLearningCudaCorrelationCalculator::getSupportsSeparateFields() {
    useSeparateFields = false;
    return false;
}

void DeepLearningCudaCorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    ICorrelationCalculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        isNanStencilInitialized = false;
    }
}

void DeepLearningCudaCorrelationCalculator::computeNanStencilBuffer() {
    numNonNanValues = 0;
    VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(scalarFieldNames.at(fieldIndexGui), -1, -1, -1);
    const auto* fieldData = fieldEntry->data<float>();
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    std::vector<uint32_t> nonNanIndexBuffer;
    uint32_t numCells = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    for (uint32_t cellIdx = 0; cellIdx < numCells; cellIdx++) {
        if (!std::isnan(fieldData[cellIdx])) {
            nonNanIndexBuffer.push_back(cellIdx);
            numNonNanValues++;
        }
    }
    if (nonNanIndexBufferCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                nonNanIndexBufferCu, stream), "Error in cuMemFreeAsync: ");
        nonNanIndexBufferCu = 0;
    }
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
            &nonNanIndexBufferCu, sizeof(uint32_t) * numNonNanValues, stream), "Error in cuMemAllocAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoDAsync(
            nonNanIndexBufferCu, nonNanIndexBuffer.data(), sizeof(uint32_t) * numNonNanValues,
            stream), "Error in cuMemcpyHtoDAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
}

void DeepLearningCudaCorrelationCalculator::calculateDevice(
        int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);

    if (!getIsModuleLoaded()) {
        deviceCacheEntry->getVulkanImage()->clearColor(glm::vec4(0.0f), renderer->getVkCommandBuffer());
        sgl::Logfile::get()->writeWarning(
                "Warning in DeepLearningCudaCorrelationCalculator::calculateDevice: Network modules are not loaded.",
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

    size_t volumeDataSlice3dSize = volumeData->getSlice3dSizeInBytes(FieldType::SCALAR);
    if (cachedVolumeDataSlice3dSize != volumeDataSlice3dSize) {
        if (outputImageBufferCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    outputImageBufferCu, stream), "Error in cuMemFreeAsync: ");
            outputImageBufferCu = 0;
        }
        cachedVolumeDataSlice3dSize = volumeDataSlice3dSize;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &outputImageBufferCu, volumeDataSlice3dSize, stream), "Error in cuMemAllocAsync: ");
    }
    bool recreatedUnpackBuffer = false;
    if (useDataNanStencil && cachedVolumeDataSlice3dSizeUnpacked != volumeDataSlice3dSize) {
        if (outputImageBufferUnpackedCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    outputImageBufferUnpackedCu, stream), "Error in cuMemFreeAsync: ");
            outputImageBufferUnpackedCu = 0;
        }
        cachedVolumeDataSlice3dSizeUnpacked = volumeDataSlice3dSize;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &outputImageBufferUnpackedCu, volumeDataSlice3dSize, stream), "Error in cuMemAllocAsync: ");
        recreatedUnpackBuffer = true;
    }
    if (!useDataNanStencil && outputImageBufferUnpackedCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                outputImageBufferUnpackedCu, stream), "Error in cuMemFreeAsync: ");
        outputImageBufferUnpackedCu = 0;
        cachedVolumeDataSlice3dSizeUnpacked = 0;
    }
    if (useDataNanStencil && (recreatedUnpackBuffer || !isNanStencilInitialized)) {
        const float fillValue = std::numeric_limits<float>::quiet_NaN();
        const uint32_t fillValueUint = *reinterpret_cast<const uint32_t*>(&fillValue);
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemsetD32Async(
                outputImageBufferUnpackedCu, fillValueUint, cachedVolumeDataSlice3dSizeUnpacked / sizeof(float),
                stream), "Error in cuMemsetD32Async: ");
    }

    if (cachedCorrelationMemberCountDevice != size_t(cs)) {
        if (permutationIndicesBufferCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    permutationIndicesBufferCu, stream), "Error in cuMemFreeAsync: ");
            permutationIndicesBufferCu = 0;
        }
        if (fieldTextureArrayCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    fieldTextureArrayCu, stream), "Error in cuMemFreeAsync: ");
            fieldTextureArrayCu = 0;
        }
        if (fieldBufferArrayCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    fieldBufferArrayCu, stream), "Error in cuMemFreeAsync: ");
            fieldBufferArrayCu = 0;
        }
        cachedCorrelationMemberCountDevice = size_t(cs);

        if (networkType == NetworkType::MINE) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                    &permutationIndicesBufferCu, gpuBatchSize1D * cs * sizeof(uint32_t), stream), "Error in cuMemAllocAsync: ");
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                    &fieldTextureArrayCu, cs * sizeof(CUtexObject), stream), "Error in cuMemAllocAsync: ");
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                    &fieldBufferArrayCu, cs * sizeof(CUdeviceptr), stream), "Error in cuMemAllocAsync: ");
        }
        cacheNeedsRecreate = true;
    }

    if (cacheNeedsRecreate) {
        recreateCache(gpuBatchSize1D);
        cacheNeedsRecreate = false;
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

    float minFieldVal = std::numeric_limits<float>::max();
    float maxFieldVal = std::numeric_limits<float>::lowest();
    bool useImageArray = dataMode == CorrelationDataMode::IMAGE_3D_ARRAY;
    std::vector<VolumeData::DeviceCacheEntry> fieldEntries;
    std::vector<CUtexObject> fieldTexturesCu;
    std::vector<CUtexObject> fieldBuffersCu;
    if (networkType == NetworkType::MINE) {
#ifdef TEST_INFERENCE_SPEED
        auto startLoad = std::chrono::system_clock::now();
#endif

        fieldEntries.reserve(cs);
        if (useImageArray) {
            fieldTexturesCu.reserve(cs);
        } else {
            fieldBuffersCu.reserve(cs);
        }
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            VolumeData::DeviceCacheEntry fieldEntry = getFieldEntryDevice(
                    scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx, useImageArray);
            fieldEntries.push_back(fieldEntry);
            if (useImageArray) {
                fieldTexturesCu.push_back(fieldEntry->getCudaTexture());
                if (fieldEntry->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    deviceCacheEntry->getVulkanImage()->transitionImageLayout(
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
                }
            } else {
                fieldBuffersCu.push_back(fieldEntry->getCudaBuffer());
            }
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(
                    scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
            minFieldVal = std::min(minFieldVal, minVal);
            maxFieldVal = std::max(maxFieldVal, maxVal);
        }

        if (useImageArray && cachedFieldTexturesCu != fieldTexturesCu) {
            cachedFieldTexturesCu = fieldTexturesCu;
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                    stream), "Error in cuStreamSynchronize: ");
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                    fieldTextureArrayCu, fieldTexturesCu.data(), sizeof(CUtexObject) * cs), "Error in cuMemcpyHtoD: ");
        }
        if (!useImageArray && cachedFieldBuffersCu != fieldBuffersCu) {
            cachedFieldBuffersCu = fieldBuffersCu;
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                    stream), "Error in cuStreamSynchronize: ");
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                    fieldBufferArrayCu, fieldBuffersCu.data(), sizeof(CUdeviceptr) * cs), "Error in cuMemcpyHtoD: ");
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

    sgl::vk::CommandBufferPtr commandBufferRender = renderer->getCommandBuffer();
    vulkanFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
    commandBufferRender->pushSignalSemaphore(vulkanFinishedSemaphore);
    renderer->endCommandBuffer();

    renderer->submitToQueue();

    vulkanFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);

    uint32_t alignmentVec4 = sgl::uiceil(getInputChannelAlignment(), 4);

    if (useDataNanStencil && !isNanStencilInitialized) {
        computeNanStencilBuffer();
        isNanStencilInitialized = true;
    }

    callbackBeginCompute();

    std::vector<void*> kernelParametersRef;
    CUfunction referenceInputAssemblyFunction{};
    CUdeviceptr scalarFields{};
    CUdeviceptr outputBufferRef = getReferenceInputPointer();
    uint32_t srnStride = 0;
    if (networkType == NetworkType::MINE) {
        scalarFields = useImageArray ? fieldTextureArrayCu : fieldBufferArrayCu;
        kernelParametersRef = {
                &xs, &ys, &zs, &cs, &referencePointIndex.x, &minFieldVal, &maxFieldVal,
                &outputBufferRef, &scalarFields, &alignmentVec4
        };
        if (useImageArray) {
            if (alignmentVec4 == 1) {
                referenceInputAssemblyFunction = combineCorrelationMembersReferenceFunctionCu;
            } else {
                referenceInputAssemblyFunction = combineCorrelationMembersReferenceAlignedFunctionCu;
            }
        } else if (useBufferTiling) {
            if (alignmentVec4 == 1) {
                referenceInputAssemblyFunction = combineCorrelationMembersReferenceBufferTiledFunctionCu;
            } else {
                referenceInputAssemblyFunction = combineCorrelationMembersReferenceAlignedBufferTiledFunctionCu;
            }
        } else {
            if (alignmentVec4 == 1) {
                referenceInputAssemblyFunction = combineCorrelationMembersReferenceBufferFunctionCu;
            } else {
                referenceInputAssemblyFunction = combineCorrelationMembersReferenceAlignedBufferFunctionCu;
            }
        }
    } else {
        srnStride = getSrnStride();
        kernelParametersRef = {
                &xs, &ys, &zs, &referencePointIndex.x, &outputBufferRef
        };
        referenceInputAssemblyFunction = writeGridPositionReferenceFunctionCu;
    }
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
            referenceInputAssemblyFunction,
            networkType == NetworkType::MINE ? sgl::uiceil(cs, 256) : 1, 1, 1, //< Grid size.
            networkType == NetworkType::MINE ? 256 : 1, 1, 1, //< Block size.
            0, //< Dynamic shared memory size.
            stream,
            kernelParametersRef.data(), //< Kernel parameters.
            nullptr //< Extra (empty).
    ), "Error in cuLaunchKernel: ");
    runInferenceReference();

    uint32_t numSliceEntries = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    if (useDataNanStencil) {
        numSliceEntries = numNonNanValues;
    }
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(gpuBatchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * gpuBatchSize1D;
        uint32_t batchSize = std::min(uint32_t(gpuBatchSize1D), numSliceEntries - batchOffset);

        CUdeviceptr outputBuffer = getQueryInputPointer();
        std::vector<void*> kernelParameters;
        CUfunction queryInputAssemblyFunction{};
        if (networkType == NetworkType::MINE) {
            kernelParameters = {
                    &xs, &ys, &zs, &cs, &batchOffset, &batchSize, &minFieldVal, &maxFieldVal,
                    &outputBuffer, &scalarFields, &alignmentVec4
            };
            if (useImageArray) {
                if (alignmentVec4 == 1) {
                    queryInputAssemblyFunction = combineCorrelationMembersFunctionCu;
                } else {
                    queryInputAssemblyFunction = combineCorrelationMembersAlignedFunctionCu;
                }
            } else if (useBufferTiling) {
                if (alignmentVec4 == 1) {
                    queryInputAssemblyFunction = combineCorrelationMembersBufferTiledFunctionCu;
                } else {
                    queryInputAssemblyFunction = combineCorrelationMembersAlignedBufferTiledFunctionCu;
                }
            } else {
                if (alignmentVec4 == 1) {
                    queryInputAssemblyFunction = combineCorrelationMembersBufferFunctionCu;
                } else {
                    queryInputAssemblyFunction = combineCorrelationMembersAlignedBufferFunctionCu;
                }
            }
        } else {
            if (useDataNanStencil) {
                kernelParameters = {
                        &xs, &ys, &zs, &batchOffset, &batchSize, &outputBuffer, &srnStride,
                        &nonNanIndexBufferCu
                };
                queryInputAssemblyFunction = writeGridPositionsStencilFunctionCu;
            } else {
                kernelParameters = {
                        &xs, &ys, &zs, &batchOffset, &batchSize, &outputBuffer, &srnStride
                };
                queryInputAssemblyFunction = writeGridPositionsFunctionCu;
            }
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

        runInferenceBatch(batchOffset, batchSize);
    }

    // For debugging purposes.
    /*float miCenter = 0.0f;
    CUdeviceptr offsetCenter = (referencePointIndex.x + referencePointIndex.y * xs + referencePointIndex.z * xs * ys) * sizeof(float);
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyDtoHAsync(
            &miCenter, outputImageBufferCu + offsetCenter,
            sizeof(float), stream), "Error in cuMemcpyDtoHAsync: ");
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
            stream), "Error in cuStreamSynchronize: ");
    std::cout << "MI center: " << miCenter << std::endl << std::endl;*/

    if (useDataNanStencil) {
        void* kernelParameters[] = {
                &numNonNanValues, &nonNanIndexBufferCu, &outputImageBufferCu, &outputImageBufferUnpackedCu
        };
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                unpackStencilValuesFunctionCu,
                sgl::uiceil(numNonNanValues, 256), 1, 1, //< Grid size.
                256, 1, 1, //< Block size.
                0, //< Dynamic shared memory size.
                stream,
                kernelParameters, //< Kernel parameters.
                nullptr //< Extra (empty).
        ), "Error in cuLaunchKernel: ");
        deviceCacheEntry->getImageCudaExternalMemory()->memcpyCudaDtoA3DAsync(outputImageBufferUnpackedCu, stream);
    } else {
        deviceCacheEntry->getImageCudaExternalMemory()->memcpyCudaDtoA3DAsync(outputImageBufferCu, stream);
    }

    cudaFinishedSemaphore->signalSemaphoreCuda(stream, timelineValue);
    cudaFinishedSemaphore->setWaitSemaphoreValue(timelineValue);
    sgl::vk::CommandBufferPtr postRenderCommandBuffer = postRenderCommandBuffers.at(frameIndex);
    renderer->pushCommandBuffer(postRenderCommandBuffer);
    renderer->beginCommandBuffer();
    postRenderCommandBuffer->pushWaitSemaphore(
            cudaFinishedSemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

    callbackEndCompute();

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    //std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}

