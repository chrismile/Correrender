/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
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

#include <boost/algorithm/string/case_conv.hpp>
#include <json/json.h>

#include <Utils/AppSettings.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Volume/VolumeData.hpp"
#include "Loaders/DataSetList.hpp"
#include "DeepLearningCorrelationCalculator.hpp"

DeepLearningCorrelationCalculator::DeepLearningCorrelationCalculator(
        const std::string& implName, const std::string& implNameKey, sgl::vk::Renderer* renderer)
        : ICorrelationCalculator(renderer), implName(implName), implNameKey(implNameKey) {
    implNameKeyUpper = implNameKey;
    std::string firstCharUpper = boost::to_upper_copy(implNameKeyUpper);
    implNameKeyUpper.at(0) = firstCharUpper.at(0);
    fileDialogKey = "Choose" + implNameKeyUpper + "ModelFile";
    fileDialogDescription = "Choose " + implName + " Model File";
    modelFilePathSettingsKey = implNameKey + "CorrelationCalculatorModelFilePath";

    std::string implNameKeyLower = boost::to_lower_copy(implNameKeyUpper);
    if (implNameKeyLower == "vmlp") {
        implNameKeyLower = "tinycudann";
    }
    std::string modelPresetsJsonFilename =
            sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/models-" + implNameKeyLower + ".json";
    if (sgl::FileUtils::get()->exists(modelPresetsJsonFilename)) {
        parseModelPresetsFile(modelPresetsJsonFilename);
    }
}

void DeepLearningCorrelationCalculator::parseModelPresetsFile(const std::string& filename) {
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

void DeepLearningCorrelationCalculator::initialize() {
    sgl::AppSettings::get()->getSettings().getValueOpt(modelFilePathSettingsKey.c_str(), modelFilePath);
    if (sgl::FileUtils::get()->exists(modelFilePath) && !sgl::FileUtils::get()->isDirectory(modelFilePath)) {
        loadModelFromFile(modelFilePath);
    }
}

DeepLearningCorrelationCalculator::~DeepLearningCorrelationCalculator() {
    sgl::AppSettings::get()->getSettings().addKeyValue(modelFilePathSettingsKey, modelFilePath);
}

void DeepLearningCorrelationCalculator::renderGuiImplSub(sgl::PropertyEditor& propertyEditor) {
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
}

void DeepLearningCorrelationCalculator::renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) {
    ICorrelationCalculator::renderGuiImplAdvanced(propertyEditor);
    if (networkType != NetworkType::MINE && propertyEditor.addCheckbox(
            "Use Data NaN Stencil", &useDataNanStencil)) {
        clearFieldDeviceData();
        dirty = true;
    }
}

void DeepLearningCorrelationCalculator::setSettings(const SettingsMap& settings) {
    ICorrelationCalculator::setSettings(settings);
    if (settings.getValueOpt("model_file_path", modelFilePath)) {
        loadModelFromFile(modelFilePath);
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

void DeepLearningCorrelationCalculator::getSettings(SettingsMap& settings) {
    ICorrelationCalculator::getSettings(settings);
    settings.addKeyValue("model_file_path", modelFilePath);
    settings.addKeyValue("model_preset_index", modelPresetIndex);
    settings.addKeyValue("calculate_absolute_value", calculateAbsoluteValue);
    settings.addKeyValue("use_data_nan_stencil", useDataNanStencil);
}

bool DeepLearningCorrelationCalculator::getSupportsSeparateFields() {
    correlationFieldMode = CorrelationFieldMode::SINGLE;
    return false;
}

void DeepLearningCorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    ICorrelationCalculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        isNanStencilInitialized = false;
    }
}

std::vector<uint32_t> DeepLearningCorrelationCalculator::computeNanStencilBufferHost() {
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
    return nonNanIndexBuffer;
}

void DeepLearningCorrelationCalculator::calculateDevice(
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
}