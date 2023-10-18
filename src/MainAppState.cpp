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

#include <json/json.h>

#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/imgui.h>

#include "Widgets/DataView.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "Calculators/Calculator.hpp"
#include "Renderers/Renderer.hpp"
#include "Replicability/ReplicabilityState.hpp"
#include "MainApp.hpp"

static Json::Value settingsMapToJson(const SettingsMap& settings) {
    Json::Value node;
    for (const auto& keyValuePair : settings.getMap()) {
        node[keyValuePair.first] = keyValuePair.second;
    }
    return node;
}

static SettingsMap jsonToSettingsMap(const Json::Value& node) {
    SettingsMap settings;
    for (const auto& member : node.getMemberNames()) {
        settings.addKeyValue(member, node[member].asString());
    }
    return settings;
}

static void camToJson(const sgl::CameraPtr& camera, Json::Value& cameraNode) {
    cameraNode["fovy"] = camera->getFOVy();
    cameraNode["yaw"] = camera->getYaw();
    cameraNode["pitch"] = camera->getPitch();
    Json::Value positionNode;
    positionNode["x"] = camera->getPosition().x;
    positionNode["y"] = camera->getPosition().y;
    positionNode["z"] = camera->getPosition().z;
    cameraNode["position"] = positionNode;
    Json::Value lookAtNode;
    lookAtNode["x"] = camera->getLookAtLocation().x;
    lookAtNode["y"] = camera->getLookAtLocation().y;
    lookAtNode["z"] = camera->getLookAtLocation().z;
    cameraNode["lookat"] = lookAtNode;
}

static void jsonToCam(const Json::Value& cameraNode, sgl::CameraPtr& camera) {
    const Json::Value& positionNode = cameraNode["position"];
    const Json::Value& lookAtNode = cameraNode["lookat"];
    camera->setPosition(glm::vec3(
            positionNode["x"].asFloat(), positionNode["y"].asFloat(), positionNode["z"].asFloat()));
    camera->setFOVy(cameraNode["fovy"].asFloat());
    camera->setYaw(cameraNode["yaw"].asFloat());
    camera->setPitch(cameraNode["pitch"].asFloat());
    camera->setLookAtLocation(glm::vec3(
            lookAtNode["x"].asFloat(), lookAtNode["y"].asFloat(), lookAtNode["z"].asFloat()));
}

void MainApp::saveStateToFile(const std::string& stateFilePath) {
    Json::Value root;

    Json::Value windowSizeNode;
    auto* window = sgl::AppSettings::get()->getMainWindow();
    windowSizeNode["x"] = window->getPixelWidth();
    windowSizeNode["y"] = window->getPixelHeight();
    root["window_size"] = windowSizeNode;

    Json::Value globalCameraNode;
    camToJson(camera, globalCameraNode);
    root["global_camera"] = globalCameraNode;

    Json::Value viewsNode;
    for (auto& dataView : dataViews) {
        Json::Value viewNode;
        viewNode["name"] = dataView->getViewName();
        viewNode["sync_with_global_camera"] = dataView->syncWithParentCamera;
        if (!dataView->syncWithParentCamera) {
            Json::Value cameraNode;
            camToJson(dataView->camera, cameraNode);
            viewNode["camera"] = cameraNode;
        }
        viewsNode.append(viewNode);
    }
    root["views"] = viewsNode;

    size_t iniSize = 0;
    const char* settingsString = ImGui::SaveIniSettingsToMemory(&iniSize);
    root["dock_data"] = settingsString;

    if (volumeData) {
        const auto& dataSetInformation = volumeData->getDataSetInformation();
        Json::Value volumeDataNode;
        if (!dataSetInformation.name.empty()) {
            volumeDataNode["name"] = dataSetInformation.name;
        } else {
            volumeDataNode["filename"] = dataSetInformation.filenames.front();
        }
        volumeDataNode["current_time_step_idx"] = volumeData->getCurrentTimeStepIdx();
        volumeDataNode["current_ensemble_idx"] = volumeData->getCurrentEnsembleIdx();

        Json::Value calculatorsNode;
        const auto& calculators = volumeData->getCalculators();
        for (const CalculatorPtr& calculator : calculators) {
            auto type = calculator->getCalculatorType();
            if (type == CalculatorType::VELOCITY || type == CalculatorType::VECTOR_MAGNITUDE
                    || type == CalculatorType::VORTICITY || type == CalculatorType::HELICITY) {
                continue;
            }
            Json::Value calculatorNode;
            calculatorNode["type"] = CALCULATOR_TYPE_IDS[int(calculator->getCalculatorType())];
            SettingsMap settings;
            calculator->getSettings(settings);
            calculatorNode["state"] = settingsMapToJson(settings);
            calculatorsNode.append(calculatorNode);
        }
        root["calculators"] = calculatorsNode;

        Json::Value transferFunctionNodes;
        const auto& scalarFieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        auto numTfs = int(scalarFieldNames.size());
        auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
        for (int varIdx = 0; varIdx < numTfs; varIdx++) {
            Json::Value transferFunctionNode;
            transferFunctionNode["data"] = tfWindow.serializeXmlString(varIdx);
            Json::Value rangeNode;
            rangeNode["min"] = tfWindow.getSelectedRangeMin(varIdx);
            rangeNode["max"] = tfWindow.getSelectedRangeMax(varIdx);
            transferFunctionNode["selected_range"] = rangeNode;
            transferFunctionNode["is_selected_range_fixed"] = tfWindow.getIsSelectedRangeFixed(varIdx);
            transferFunctionNodes.append(transferFunctionNode);
        }
        volumeDataNode["transfer_functions"] = transferFunctionNodes;

        root["volume_data"] = volumeDataNode;
    }

    Json::Value renderersNode;
    for (auto& volumeRenderer : volumeRenderers) {
        auto renderingMode = volumeRenderer->getRenderingMode();
        if (renderingMode == RENDERING_MODE_CUSTOM) {
            continue;
        }
        Json::Value rendererNode;
        rendererNode["type"] = RENDERING_MODE_NAMES_ID[int(renderingMode)];
        SettingsMap settings;
        volumeRenderer->getSettings(settings);
        rendererNode["state"] = settingsMapToJson(settings);
        renderersNode.append(rendererNode);
    }
    root["renderers"] = renderersNode;

    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "    ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    std::ofstream jsonFileStream(stateFilePath.c_str());
    writer->write(root, &jsonFileStream);
    jsonFileStream.close();
}

void MainApp::loadStateFromFile(const std::string& stateFilePath) {
    // Parse the passed JSON file.
    std::ifstream jsonFileStream(stateFilePath.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return;
    }
    jsonFileStream.close();
    loadStateFromJsonObject(root);
}

void MainApp::loadStateFromJsonObject(Json::Value root) {
    // Window size.
    if (root.isMember("window_size")) {
        Json::Value& windowSizeNode = root["window_size"];
        int windowWidth = windowSizeNode["x"].asInt();
        int windowHeight = windowSizeNode["y"].asInt();
        auto* window = sgl::AppSettings::get()->getMainWindow();
        if (windowWidth != window->getPixelWidth() || windowHeight != window->getPixelHeight()) {
            rendererVk->syncWithCpu();
            window->setWindowPixelSize(windowWidth, windowHeight);
        }
    }

    const Json::Value& globalCameraNode = root["global_camera"];
    jsonToCam(globalCameraNode, camera);
    fovDegree = camera->getFOVy() / sgl::PI * 180.0f;

    // Views.
    Json::Value& viewsNode = root["views"];
    while (dataViews.size() > viewsNode.size()) {
        auto viewIdx = int(dataViews.size() - 1);
        viewManager->removeView(viewIdx);
        dataViews.erase(dataViews.begin() + viewIdx);
        for (auto& volumeRenderer : volumeRenderers) {
            volumeRenderer->removeView(viewIdx);
        }
        if (volumeData) {
            volumeData->removeView(viewIdx);
        }
    }
    while (dataViews.size() < viewsNode.size()) {
        addNewDataView();
    }
    for (size_t viewIdx = 0; viewIdx < dataViews.size(); viewIdx++) {
        const auto& viewNode = viewsNode[int(viewIdx)];
        auto& dataView = dataViews.at(viewIdx);
        dataView->getViewName() = viewNode["name"].asString();
        dataView->syncWithParentCamera = viewNode["sync_with_global_camera"].asBool();
        if (!dataView->syncWithParentCamera) {
            const Json::Value& cameraNode = viewNode["camera"];
            jsonToCam(cameraNode, dataView->camera);
        }
    }

    if (root.isMember("volume_data")) {
        const auto& volumeDataNode = root["volume_data"];

        if (volumeDataNode.isMember("name")) {
            std::string dataSetName = volumeDataNode["name"].asString();
            selectedDataSetIndex = -1;
            customDataSetFileName = "";
            for (const auto& dataSetInfo : dataSetInformationList) {
                if (dataSetInfo->name == dataSetName) {
                    selectedDataSetIndex = int(dataSetInfo->sequentialIndex);
                    break;
                }
            }
            if (selectedDataSetIndex < 0) {
                sgl::Logfile::get()->writeError(
                        "Error in MainApp::loadStateFromFile: Unknown data set \"" + dataSetName + "\"");
            }
        } else {
            selectedDataSetIndex = 0;
            customDataSetFileName = volumeDataNode["filename"].asString();
            dataSetType = DataSetType::VOLUME;
        }
        if (selectedDataSetIndex >= 0) {
            loadVolumeDataSet(getSelectedDataSetFilenames());
        }

        rendererVk->syncWithCpu();

        volumeData->setCurrentTimeStepIdx(volumeDataNode["current_time_step_idx"].asInt());
        volumeData->setCurrentEnsembleIdx(volumeDataNode["current_ensemble_idx"].asInt());

        const Json::Value& calculatorsNode = root["calculators"];
        const auto& calculators = volumeData->getCalculators();
        int numCalculatorsBase = 0;
        for (const CalculatorPtr& calculator : calculators) {
            auto type = calculator->getCalculatorType();
            if (type == CalculatorType::VELOCITY || type == CalculatorType::VECTOR_MAGNITUDE
                    || type == CalculatorType::VORTICITY || type == CalculatorType::HELICITY) {
                numCalculatorsBase++;
            }
        }
        auto numCalculatorsOld = int(calculators.size());
        auto numCalculatorsNew = numCalculatorsBase + int(calculatorsNode.size());
        while (numCalculatorsOld > numCalculatorsNew) {
            volumeData->removeCalculator(calculators.at(numCalculatorsOld - 1), numCalculatorsOld - 1);
            volumeData->dirty = true;
            numCalculatorsOld--;
        }
        int calculatorListIdxCurr = 0;
        int calculatorJsonIdxCurr = 0;
        while (calculatorJsonIdxCurr < int(calculatorsNode.size())) {
            const Json::Value& calculatorNode = calculatorsNode[calculatorJsonIdxCurr];
            CalculatorType calculatorTypeNew = CalculatorType::INVALID;
            std::string typeString = calculatorNode["type"].asString();
            for (int i = 0; i < IM_ARRAYSIZE(CALCULATOR_TYPE_IDS); i++) {
                if (typeString == CALCULATOR_TYPE_IDS[i]) {
                    calculatorTypeNew = CalculatorType(i);
                    break;
                }
            }
            if (calculatorTypeNew == CalculatorType::INVALID) {
                sgl::Logfile::get()->writeError(
                        "Error in MainApp::loadStateFromFile: Invalid calculator type \"" + typeString + "\".");
                calculatorTypeNew = CalculatorType::BINARY_OPERATOR;
            }
            if (calculatorListIdxCurr == int(calculators.size())) {
                bool foundCalculatorType = false;
                auto& factoriesCalculator = volumeData->factoriesCalculator;
                for (auto& factory : factoriesCalculator) {
                    if (factory.first == CALCULATOR_NAMES[int(calculatorTypeNew)]) {
                        volumeData->addCalculator(CalculatorPtr(factory.second()));
                        volumeData->dirty = true; //< Necessary for new transfer function map.
                        volumeData->reRender = true;
                        foundCalculatorType = true;
                        break;
                    }
                }
                if (!foundCalculatorType) {
                    sgl::Logfile::get()->throwError(
                            "Error in MainApp::loadStateFromFile: Invalid calculator type \"" + typeString + "\".");
                }
            }

            auto calculator = calculators.at(calculatorListIdxCurr);
            CalculatorType calculatorTypePrev = calculator->getCalculatorType();
            if (calculatorTypePrev == calculatorTypeNew) {
                calculator->setSettings(jsonToSettingsMap(calculatorNode["state"]));
            } else {
                // Remove incompatible renderers
                for (size_t i = calculatorListIdxCurr; i < calculators.size(); i++) {
                    volumeData->removeCalculator(calculators.at(numCalculatorsOld - 1), numCalculatorsOld - 1);
                    volumeData->dirty = true;
                }
                continue;
            }
            calculatorListIdxCurr++;
            calculatorJsonIdxCurr++;
        }

        const Json::Value& transferFunctionNodes = volumeDataNode["transfer_functions"];
        auto numTfs = int(transferFunctionNodes.size());
        auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
        for (int varIdx = 0; varIdx < numTfs; varIdx++) {
            const Json::Value& transferFunctionNode = transferFunctionNodes[varIdx];
            tfWindow.deserializeXmlString(varIdx, transferFunctionNode["data"].asString());
            const Json::Value& rangeNode = transferFunctionNode["selected_range"];
            tfWindow.setSelectedRange(varIdx, glm::vec2(rangeNode["min"].asFloat(), rangeNode["max"].asFloat()));
            tfWindow.setIsSelectedRangeFixed(varIdx, transferFunctionNode["is_selected_range_fixed"].asBool());
        }
    }

    // Renderers.
    Json::Value& renderersNode = root["renderers"];
    auto numRenderersNew = int(renderersNode.size());
    int numRenderersPrev = 0;
    for (auto& volumeRenderer : volumeRenderers) {
        auto renderingMode = volumeRenderer->getRenderingMode();
        if (renderingMode == RENDERING_MODE_CUSTOM) {
            continue;
        }
        numRenderersPrev++;
    }
    int rendererIdxCurr = int(volumeRenderers.size()) - 1;
    while (numRenderersPrev > numRenderersNew) {
        auto volumeRenderer = volumeRenderers.at(rendererIdxCurr);
        auto renderingMode = volumeRenderer->getRenderingMode();
        if (renderingMode == RENDERING_MODE_CUSTOM) {
            rendererIdxCurr--;
            continue;
        }
        volumeRenderers.erase(volumeRenderers.begin() + rendererIdxCurr);
        numRenderersPrev--;
        rendererIdxCurr--;
    }
    int rendererListIdxCurr = 0;
    int rendererJsonIdxCurr = 0;
    while (rendererJsonIdxCurr < numRenderersNew) {
        const Json::Value& rendererNode = renderersNode[rendererJsonIdxCurr];
        RenderingMode renderingModeNew = RENDERING_MODE_NONE;
        std::string typeString = rendererNode["type"].asString();
        for (int i = 0; i < NUM_RENDERING_MODES; i++) {
            if (typeString == RENDERING_MODE_NAMES_ID[i]) {
                renderingModeNew = RenderingMode(i);
                break;
            }
        }
        if (renderingModeNew == RENDERING_MODE_NONE) {
            sgl::Logfile::get()->writeError(
                    "Error in MainApp::loadStateFromFile: Invalid rendering mode \"" + typeString +"\".");
            renderingModeNew = RENDERING_MODE_DIRECT_VOLUME_RENDERING;
        }

        if (rendererListIdxCurr == int(volumeRenderers.size())) {
            addNewRenderer(renderingModeNew);
        }

        auto volumeRenderer = volumeRenderers.at(rendererListIdxCurr);
        auto renderingModeOld = volumeRenderer->getRenderingMode();
        if (renderingModeOld == RENDERING_MODE_CUSTOM) {
            rendererListIdxCurr++;
            continue;
        }
        if (renderingModeOld == renderingModeNew) {
            prepareVisualizationPipeline();
            volumeRenderer->setSettings(jsonToSettingsMap(rendererNode["state"]));
        } else {
            // Remove incompatible renderers
            for (size_t i = rendererListIdxCurr; i < volumeRenderers.size(); i++) {
                if (volumeRenderers.at(i)->getRenderingMode() != RENDERING_MODE_CUSTOM) {
                    volumeRenderers.erase(volumeRenderers.begin() + ptrdiff_t(i));
                    i--;
                }
            }
            continue;
        }
        rendererListIdxCurr++;
        rendererJsonIdxCurr++;
    }

    std::string iniSettingsString = root["dock_data"].asString();
    ImGui::LoadIniSettingsFromMemory(iniSettingsString.c_str(), iniSettingsString.size());

    reRender = true;
    hasMoved();
    onCameraReset();
}

void MainApp::setUseReplicabilityStampMode() {
    useReplicabilityStampMode = true;
}

void MainApp::loadReplicabilityStampState() {
    Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    JSONCPP_STRING errorString;
    Json::Value root;

    int desktopWidth = 0;
    int desktopHeight = 0;
    int refreshRate = 60;
    sgl::AppSettings::get()->getDesktopDisplayMode(desktopWidth, desktopHeight, refreshRate);
    const char* replicabilityStateString;
    int stateStringLength;
    if (desktopWidth == 3840 && desktopHeight == 2160) {
        replicabilityStateString = REPLICABILITY_STATE_STRING_4K;
        stateStringLength = IM_ARRAYSIZE(REPLICABILITY_STATE_STRING_4K);
    } else {
        replicabilityStateString = REPLICABILITY_STATE_STRING_MISC;
        stateStringLength = IM_ARRAYSIZE(REPLICABILITY_STATE_STRING_MISC);
    }

    if (!reader->parse(
            replicabilityStateString, replicabilityStateString + stateStringLength, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return;
    }
    loadStateFromJsonObject(root);
}
