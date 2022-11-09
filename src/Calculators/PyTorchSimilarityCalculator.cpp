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

#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>

#include "Volume/VolumeData.hpp"
#include "PyTorchSimilarityCalculator.hpp"

#if CUDA_VERSION >= 11020
#define USE_TIMELINE_SEMAPHORES
#elif defined(_WIN32)
#error Binary semaphore sharing is broken on Windows. Please install CUDA >= 11.2 for timeline semaphore support.
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

    sgl::AppSettings::get()->getSettings().getValueOpt("pyTorchDenoiserModelFilePath", modelFilePath);

#ifdef SUPPORT_CUDA_INTEROP
    // Support CUDA on NVIDIA GPUs using the proprietary driver.
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY && torch::cuda::is_available()) {
        pyTorchDevice = PyTorchDevice::CUDA;
        if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumetricPathTracingModuleRenderer::renderFrameCuda: "
                    "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.");
        }
    }
#endif
}

PyTorchSimilarityCalculator::~PyTorchSimilarityCalculator() {
    sgl::AppSettings::get()->getSettings().addKeyValue("pyTorchDenoiserModelFilePath", modelFilePath);

    /*if (renderedImageData) {
        delete[] renderedImageData;
        renderedImageData = nullptr;
    }
    if (denoisedImageData) {
        delete[] denoisedImageData;
        denoisedImageData = nullptr;
    }*/
}

bool PyTorchSimilarityCalculator::loadModelFromFile(const std::string& modelPath) {
    torch::DeviceType deviceType = getTorchDeviceType(pyTorchDevice);
    torch::jit::ExtraFilesMap extraFilesMap;
    extraFilesMap["model_info.json"] = "";
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

    // Read the model JSON metadata next.
    /*auto inputFeatureMapsUsedOld = inputFeatureMapsUsed;
    inputFeatureMapsUsed.clear();
    inputFeatureMapsIndexMap.clear();
    auto it = extraFilesMap.find("model_info.json");
    if (it == extraFilesMap.end()) {
        sgl::Logfile::get()->writeError(
                "Error: Couldn't find model_info.json in the PyTorch module loaded from \"" + modelPath + "\"!");
        wrapper = {};
        return false;
    }

    Json::CharReaderBuilder builder;
    std::unique_ptr<Json::CharReader> const charReader(builder.newCharReader());
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!charReader->parse(
            it->second.c_str(), it->second.c_str() + it->second.size(), &root, &errorString)) {
        sgl::Logfile::get()->writeError("Error in PyTorchSimilarityCalculator::loadModelFromFile: " + errorString);
        wrapper = {};
        return false;
    }*/

    /*
     * Example: { "input_feature_maps": [ "color", "normal" ] }
     */
    /*if (!root.isMember("input_feature_maps")) {
        sgl::Logfile::get()->writeError(
                "Error: Array 'input_feature_maps' could not be found in the model_info.json file of the PyTorch "
                "module loaded from \"" + modelPath + "\"!");
        wrapper = {};
        return false;
    }
    Json::Value& inputFeatureMapsNode = root["input_feature_maps"];
    if (!inputFeatureMapsNode.isArray()) {
        sgl::Logfile::get()->writeError(
                "Error: 'input_feature_maps' is not an array in the model_info.json file of the PyTorch "
                "module loaded from \"" + modelPath + "\"!");
        wrapper = {};
        return false;
    }

    for (Json::Value& inputFeatureMapNode : inputFeatureMapsNode) {
        if (!inputFeatureMapNode.isString()) {
            sgl::Logfile::get()->writeError(
                    "Error: A child of the array 'input_feature_maps' in the model_info.json file of the PyTorch "
                    "module loaded from \"" + modelPath + "\" is not a string!");
            wrapper = {};
            return false;
        }
        std::string inputFeatureMapName = boost::to_lower_copy(inputFeatureMapNode.asString());
        int i;
        for (i = 0; i < IM_ARRAYSIZE(FEATURE_MAP_NAMES); i++) {
            if (boost::to_lower_copy(std::string() + FEATURE_MAP_NAMES[i]) == inputFeatureMapName) {
                inputFeatureMapsUsed.push_back(FeatureMapType(i));
                break;
            }
        }
        if (i == IM_ARRAYSIZE(FEATURE_MAP_NAMES)) {
            std::string errorStringLogfile = "Error: Invalid feature map name '" + inputFeatureMapName;
            errorStringLogfile +=
                    "' found in the model_info.json file of the PyTorch module loaded from \"" + modelPath + "\"!";
            sgl::Logfile::get()->writeError(errorStringLogfile);
            wrapper = {};
            return false;
        }
    }

    for (size_t i = 0; i < inputFeatureMapsUsed.size(); i++) {
        inputFeatureMapsIndexMap.insert(std::make_pair(inputFeatureMapsUsed.at(i), i));
    }
    inputFeatureMaps.resize(inputFeatureMapsUsed.size());
    computeNumChannels();
    if (inputFeatureMapsUsed != inputFeatureMapsUsedOld) {
        renderer->getDevice()->waitIdle();
        recreateSwapchain(
                inputImageVulkan->getImage()->getImageSettings().width,
                inputImageVulkan->getImage()->getImageSettings().height);
    }
    isFirstContiguousWarning = true;

    // Does the model use 3 or 4 input dimensions?
    if (wrapper->module.parameters().size() > 0) {
        auto paramSizes = (*wrapper->module.parameters().begin()).sizes();
        if (paramSizes.size() == 4) {
            useBatchDimension = true;
        }
    }*/

    return true;
}

void PyTorchSimilarityCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
     ;
}

void PyTorchSimilarityCalculator::calculateDevice(
        int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    /*cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sgl::vk::ImageCudaExternalMemoryVkPtr imageCudaExternalMemory = deviceCacheEntry->getImageCudaExternalMemory();
    imageCudaExternalMemory->memcpyCudaDtoA3DAsync(devicePtr, stream);

    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();

    std::vector<VolumeData::HostCacheEntry> ensembleEntryFields;
    std::vector<float*> ensembleFields;
    ensembleEntryFields.reserve(es);
    ensembleFields.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleFields.push_back(ensembleEntryField.get());
    }

    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
    }

    size_t referencePointIdx =
            size_t(referencePointIndex.x) * size_t(referencePointIndex.y) * size_t(referencePointIndex.z);
    auto* referenceValues = new float[es];
    for (int e = 0; e < es; e++) {
        referenceValues[e] = ensembleFields.at(e)[referencePointIdx];
    }

    // See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    size_t numGridPoints = size_t(xs) * size_t(ys) * size_t(zs);
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
        for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel for shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none)
#endif
    for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
        auto n = float(es);
        float sumX = 0.0f;
        float sumY = 0.0f;
        float sumXY = 0.0f;
        float sumXX = 0.0f;
        float sumYY = 0.0f;
        for (int e = 0; e < es; e++) {
            float x = referenceValues[e];
            float y = ensembleFields.at(e)[gridPointIdx];
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumXX += x * x;
            sumYY += y * y;
        }
        float pearsonCorrelation =
                (n * sumXY - sumX * sumY) / std::sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
        buffer[gridPointIdx] = pearsonCorrelation;
    }
#ifdef USE_TBB
    });
#endif

    delete[] referenceValues;*/
}

void PyTorchSimilarityCalculator::setPyTorchDevice(PyTorchDevice pyTorchDeviceNew) {
    if (pyTorchDeviceNew == pyTorchDevice) {
        return;
    }

    pyTorchDevice = pyTorchDeviceNew;
    /*if (inputImageVulkan) {
        renderer->getDevice()->waitIdle();
        recreateSwapchain(
                inputImageVulkan->getImage()->getImageSettings().width,
                inputImageVulkan->getImage()->getImageSettings().height);
    }
    if (wrapper) {
        wrapper->module.to(getTorchDeviceType(pyTorchDevice));
    }*/
}

void PyTorchSimilarityCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    bool reRender = false;

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

            modelFilePath = filename;
            loadModelFromFile(modelFilePath);
            reRender = true;
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    propertyEditor.addInputAction("Model Path", &modelFilePath);
    if (propertyEditor.addButton("", "Load")) {
        loadModelFromFile(modelFilePath);
        reRender = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Open from Disk...")) {
        if (fileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(fileDialogDirectory)) {
            fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "LineDataSets/";
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

    PyTorchDevice pyTorchDeviceNew = pyTorchDevice;
    if (propertyEditor.addCombo(
            "Device", (int*)&pyTorchDeviceNew,
            PYTORCH_DEVICE_NAMES, IM_ARRAYSIZE(PYTORCH_DEVICE_NAMES))) {
        setPyTorchDevice(pyTorchDeviceNew);
        reRender = true;
    }

    (void)reRender;
    // TODO
    //return reRender;
}
