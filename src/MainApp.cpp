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

#include <memory>
#include <stack>
#include <algorithm>
#include <csignal>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/color_space.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <boost/algorithm/string.hpp>

#ifdef USE_ZEROMQ
#include <zmq.h>
#endif

#ifdef SUPPORT_QUICK_MLP
#include <ckl/kernel_loader.h>
#include <qmlp/qmlp.h>
#endif

#include <Utils/Timer.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/Dialog.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/Regex/TransformString.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Input/Keyboard.hpp>
#include <Input/Mouse.hpp>
#include <Math/Math.hpp>
#include <Math/Geometry/MatrixUtil.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#ifdef SUPPORT_OPENCL_INTEROP
#include <Graphics/Vulkan/Utils/InteropOpenCL.hpp>
#endif

#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>
#include <ImGui/imgui_internal.h>
#include <ImGui/imgui_custom.h>
#include <ImGui/imgui_stdlib.h>
#include <ImGui/Widgets/ColorLegendWidget.hpp>

#include "Volume/VolumeData.hpp"
#include "Renderers/DvrRenderer.hpp"
#include "Renderers/IsoSurfaceRayCastingRenderer.hpp"
#include "Renderers/IsoSurfaceRasterizer.hpp"
#include "Renderers/DomainOutlineRenderer.hpp"
#include "Renderers/SliceRenderer.hpp"
#include "Renderers/Diagram/DiagramRenderer.hpp"
#include "Utils/AutomaticPerformanceMeasurer.hpp"

#include "Widgets/ViewManager.hpp"
#include "Widgets/DataView.hpp"
#include "MainApp.hpp"

void vulkanErrorCallback() {
    SDL_CaptureMouse(SDL_FALSE);
    std::cerr << "Application callback" << std::endl;
}

#ifdef __linux__
void signalHandler(int signum) {
    SDL_CaptureMouse(SDL_FALSE);
    std::cerr << "Interrupt signal (" << signum << ") received." << std::endl;
    exit(signum);
}
#endif

MainApp::MainApp()
        : sceneData(
                &rendererVk, &sceneTextureVk, &sceneDepthTextureVk,
                &viewportPositionX, &viewportPositionY, &viewportWidth, &viewportHeight, camera,
                &clearColor, &screenshotTransparentBackground,
                &performanceMeasurer, &continuousRendering, &recording,
                &useCameraFlight, &MOVE_SPEED, &MOUSE_ROT_SPEED,
                &nonBlockingMsgBoxHandles),
#ifdef USE_PYTHON
          //replayWidget(&sceneData, checkpointWindow),
#endif
          boundingBox() {
    sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallback);
    clearColor = sgl::Color(0, 0, 0, 255);
    clearColorSelection = ImColor(0, 0, 0, 255);

#ifdef SUPPORT_CUDA_INTEROP
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        cudaInteropInitialized = true;
        if (!sgl::vk::initializeCudaDeviceApiFunctionTable()) {
            cudaInteropInitialized = false;
            sgl::Logfile::get()->writeError(
                    "Error in MainApp::MainApp: sgl::vk::initializeCudaDeviceApiFunctionTable() returned false.",
                    false);
        }

        if (cudaInteropInitialized) {
            CUresult cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuInit(0);
            if (cuResult == CUDA_ERROR_NO_DEVICE) {
                sgl::Logfile::get()->writeInfo("No CUDA-capable device was found. Disabling CUDA interop support.");
                cudaInteropInitialized = false;
            } else {
                sgl::vk::checkCUresult(cuResult, "Error in cuInit: ");
            }
        }

        if (cudaInteropInitialized) {
            CUresult cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxCreate(
                    &cuContext, CU_CTX_SCHED_SPIN, cuDevice);
            sgl::vk::checkCUresult(cuResult, "Error in cuCtxCreate: ");
        }
    }
#endif

#ifdef SUPPORT_OPENCL_INTEROP
    openclInteropInitialized = true;
    if (!sgl::vk::initializeOpenCLFunctionTable()) {
        openclInteropInitialized = false;
    }
#endif

    viewManager = new ViewManager(rendererVk);

#ifdef USE_PYTHON
    sgl::ColorLegendWidget::setFontScaleStandard(1.0f);

    /*replayWidget.setLoadLineDataCallback([this](const std::string& datasetName) {
        int i;
        int oldSelectedDataSetIndex = selectedDataSetIndex;
        for (i = 0; i < int(dataSetNames.size()); i++) {
            if (dataSetNames.at(i) == datasetName) {
                selectedDataSetIndex = i;
                break;
            }
        }
        if (i != int(dataSetNames.size())) {
            if (selectedDataSetIndex >= NUM_MANUAL_LOADERS && oldSelectedDataSetIndex != selectedDataSetIndex) {
                loadVolumeDataSet(getSelectedDataSetFilenames(), true);
            }
        } else {
            sgl::Logfile::get()->writeError(
                    "Replay widget: loadMeshCallback: Invalid data set name \"" + datasetName + "\".");
        }
    });
    replayWidget.setLoadRendererCallback([this](const std::string& rendererName, int viewIdx) {
        SceneData* sceneDataPtr;
        RenderingMode* renderingModeNew;
        RenderingMode* renderingModeOld;
        VolumeRenderer** volumeRendererPtr;
        if (useDockSpaceMode) {
            if (viewIdx >= int(dataViews.size())) {
                addNewDataView();
            }
            sceneDataPtr = &dataViews.at(viewIdx)->sceneData;
            renderingModeNew = &dataViews.at(viewIdx)->renderingMode;
            renderingModeOld = &dataViews.at(viewIdx)->oldRenderingMode;
            volumeRendererPtr = &dataViews.at(viewIdx)->volumeRenderer;
        } else {
            sceneDataPtr = &sceneData;
            renderingModeNew = &renderingMode;
            renderingModeOld = &oldRenderingMode;
            volumeRendererPtr = &volumeRenderer;
        }
        int i;
        for (i = 0; i < IM_ARRAYSIZE(RENDERING_MODE_NAMES); i++) {
            if (RENDERING_MODE_NAMES[i] == rendererName) {
                *renderingModeNew = RenderingMode(i);
                break;
            }
        }
        if (i == IM_ARRAYSIZE(RENDERING_MODE_NAMES)) {
            sgl::Logfile::get()->writeError(
                    std::string() + "Error in replay widget load renderer callback: Unknown renderer name \""
                    + rendererName + "\".");
        }
        if (*renderingModeNew != *renderingModeOld) {
            setRenderer(*sceneDataPtr, *renderingModeOld, *renderingModeNew, *volumeRendererPtr, viewIdx);
        }
        if (useDockSpaceMode) {
            dataViews.at(viewIdx)->updateCameraMode();
        }
    });
    replayWidget.setLoadTransferFunctionCallback([this](const std::string& tfName) {
        if (volumeData) {
            transferFunctionWindow.loadFunctionFromFile(
                    transferFunctionWindow.getSaveDirectory() + tfName);
            volumeData->onTransferFunctionMapRebuilt();
            sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
                    ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }
    });
    replayWidget.setTransferFunctionRangeCallback([this](const glm::vec2& tfRange) {
        if (volumeData) {
            transferFunctionWindow.setSelectedRange(tfRange);
            updateTransferFunctionRange = true;
            transferFunctionRange = tfRange;
            volumeData->onTransferFunctionMapRebuilt();
            sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
                    ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }
    });
    replayWidget.setLoadMultiVarTransferFunctionsCallback([this](
            const std::vector<std::string>& tfNames) {
        if (volumeData) {
            MultiVarTransferFunctionWindow* multiVarTransferFunctionWindow;
            if (volumeData->getType() == DATA_SET_TYPE_FLOW_VOLUMES) {
                VolumeDataFlow* volumeDataFlow = static_cast<VolumeDataFlow*>(volumeData.get());
                multiVarTransferFunctionWindow = &volumeDataFlow->getMultiVarTransferFunctionWindow();
            } else if (volumeData->getType() == DATA_SET_TYPE_STRESS_VOLUMES) {
                VolumeDataStress* volumeDataStress = static_cast<VolumeDataStress*>(volumeData.get());
                multiVarTransferFunctionWindow = &volumeDataStress->getMultiVarTransferFunctionWindow();
            } else if (volumeData->getType() == DATA_SET_TYPE_TRIANGLE_MESH) {
                TriangleMeshData* triangleMeshData = static_cast<TriangleMeshData*>(volumeData.get());
                multiVarTransferFunctionWindow = &triangleMeshData->getMultiVarTransferFunctionWindow();
            } else {
                sgl::Logfile::get()->writeError(
                        "Error in replay widget load multi-var transfer functions callback: Invalid data type.");
                return;
            }
            multiVarTransferFunctionWindow->loadFromTfNameList(tfNames);
            volumeData->onTransferFunctionMapRebuilt();
            sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
                    ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }
    });
    replayWidget.setMultiVarTransferFunctionsRangesCallback([this](
            const std::vector<glm::vec2>& tfRanges) {
        if (volumeData) {
            MultiVarTransferFunctionWindow* multiVarTransferFunctionWindow;
            if (volumeData->getType() == DATA_SET_TYPE_FLOW_VOLUMES) {
                VolumeDataFlow* volumeDataFlow = static_cast<VolumeDataFlow*>(volumeData.get());
                multiVarTransferFunctionWindow = &volumeDataFlow->getMultiVarTransferFunctionWindow();
            } else if (volumeData->getType() == DATA_SET_TYPE_STRESS_VOLUMES) {
                VolumeDataStress* volumeDataStress = static_cast<VolumeDataStress*>(volumeData.get());
                multiVarTransferFunctionWindow = &volumeDataStress->getMultiVarTransferFunctionWindow();
            } else if (volumeData->getType() == DATA_SET_TYPE_TRIANGLE_MESH) {
                TriangleMeshData* triangleMeshData = static_cast<TriangleMeshData*>(volumeData.get());
                multiVarTransferFunctionWindow = &triangleMeshData->getMultiVarTransferFunctionWindow();
            } else {
                sgl::Logfile::get()->writeError(
                        "Error in replay widget multi-var transfer functions ranges callback: Invalid data type.");
                return;
            }

            for (int varIdx = 0; varIdx < int(tfRanges.size()); varIdx++) {
                multiVarTransferFunctionWindow->setSelectedRange(varIdx, tfRanges.at(varIdx));
            }
            volumeData->onTransferFunctionMapRebuilt();
            sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
                    ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }
    });
    replayWidget.setSaveScreenshotCallback([this](const std::string& screenshotName) {
        if (!screenshotName.empty()) {
            saveFilenameScreenshots = screenshotName;
        }
        screenshot = true;
    });*/
#endif

    checkpointWindow.setStandardWindowSize(1254, 390);
    checkpointWindow.setStandardWindowPosition(841, 53);

    propertyEditor.setInitWidthValues(sgl::ImGuiWrapper::get()->getScaleDependentSize(280.0f));

    camera->setNearClipDistance(0.01f);
    camera->setFarClipDistance(100.0f);

    CAMERA_PATH_TIME_PERFORMANCE_MEASUREMENT = TIME_PERFORMANCE_MEASUREMENT;
    usePerformanceMeasurementMode = false;
    if (sgl::FileUtils::get()->get_argc() > 1) {
        if (strcmp(sgl::FileUtils::get()->get_argv()[1], "--perf") == 0) {
            usePerformanceMeasurementMode = true;
        }
    }
    cameraPath.setApplicationCallback([this](
            const std::string& modelFilename, glm::vec3& centerOffset, float& startAngle, float& pulseFactor,
            float& standardZoom) {
    });

    useDockSpaceMode = true;
    sgl::AppSettings::get()->getSettings().getValueOpt("useDockSpaceMode", useDockSpaceMode);
    sgl::AppSettings::get()->getSettings().getValueOpt("useFixedSizeViewport", useFixedSizeViewport);
    sgl::AppSettings::get()->getSettings().getValueOpt("fixedViewportSizeX", fixedViewportSize.x);
    sgl::AppSettings::get()->getSettings().getValueOpt("fixedViewportSizeY", fixedViewportSize.y);
    fixedViewportSizeEdit = fixedViewportSize;
    showPropertyEditor = true;
    sgl::ImGuiWrapper::get()->setUseDockSpaceMode(useDockSpaceMode);
    //useDockSpaceMode = false;

#ifdef NDEBUG
    showFpsOverlay = false;
#else
    showFpsOverlay = true;
#endif
    sgl::AppSettings::get()->getSettings().getValueOpt("showFpsOverlay", showFpsOverlay);
    sgl::AppSettings::get()->getSettings().getValueOpt("showCoordinateAxesOverlay", showCoordinateAxesOverlay);

    useLinearRGB = false;
    coordinateAxesOverlayWidget.setClearColor(clearColor);

    resolutionChanged(sgl::EventPtr());

    if (usePerformanceMeasurementMode) {
        useCameraFlight = true;
    }
    if (useCameraFlight && recording) {
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        window->setWindowSize(recordingResolution.x, recordingResolution.y);
        realTimeCameraFlight = false;
        loadVolumeDataSet({ sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/test.nc" });
    }

    fileDialogInstance = IGFD_Create();
    customDataSetFileName = sgl::FileUtils::get()->getUserDirectory();
    loadAvailableDataSetInformation();

    if (!recording && !usePerformanceMeasurementMode) {
        // Just for convenience...
        int desktopWidth = 0;
        int desktopHeight = 0;
        int refreshRate = 60;
        sgl::AppSettings::get()->getDesktopDisplayMode(desktopWidth, desktopHeight, refreshRate);
        if (desktopWidth == 3840 && desktopHeight == 2160) {
            sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
            window->setWindowSize(2186, 1358);
        }
    }

    //if (!useDockSpaceMode) {
    //    setRenderer(sceneData, oldRenderingMode, renderingMode, volumeRenderer, 0);
    //}

    if (!sgl::AppSettings::get()->getSettings().hasKey("cameraNavigationMode")) {
        cameraNavigationMode = sgl::CameraNavigationMode::TURNTABLE;
        updateCameraNavigationMode();
    }

    addNewDataView();

    recordingTimeStampStart = sgl::Timer->getTicksMicroseconds();
    usesNewState = true;
    if (usePerformanceMeasurementMode) {
        sgl::FileUtils::get()->ensureDirectoryExists("images");
        performanceMeasurer = new AutomaticPerformanceMeasurer(
                rendererVk, getTestModes(),
                "performance.csv", "depth_complexity.csv",
                [this](const InternalState &newState) { this->setNewState(newState); });
    }

#ifdef __linux__
    signal(SIGSEGV, signalHandler);
#endif
}

MainApp::~MainApp() {
    device->waitIdle();

    if (usePerformanceMeasurementMode) {
        performanceMeasurer->cleanup();
        delete performanceMeasurer;
        performanceMeasurer = nullptr;
    }

    volumeRenderers = {};
    volumeData = {};
    dataViews.clear();
    delete viewManager;
    viewManager = nullptr;

    IGFD_Destroy(fileDialogInstance);

#ifdef SUPPORT_QUICK_MLP
    qmlp::QuickMLP::DeleteInstance();
    ckl::KernelLoader::DeleteInstance();
#endif

#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        if (cuContext) {
            CUresult cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxDestroy(cuContext);
            sgl::vk::checkCUresult(cuResult, "Error in cuCtxDestroy: ");
            cuContext = {};
        }
        sgl::vk::freeCudaDeviceApiFunctionTable();
    }
#endif
#ifdef SUPPORT_OPENCL_INTEROP
    if (sgl::vk::getIsOpenCLFunctionTableInitialized()) {
        sgl::vk::freeOpenCLFunctionTable();
    }
#endif

    for (int i = 0; i < int(nonBlockingMsgBoxHandles.size()); i++) {
        auto& handle = nonBlockingMsgBoxHandles.at(i);
        if (handle->ready(0)) {
            nonBlockingMsgBoxHandles.erase(nonBlockingMsgBoxHandles.begin() + i);
            i--;
        } else {
            handle->kill();
        }
    }
    nonBlockingMsgBoxHandles.clear();

    sgl::AppSettings::get()->getSettings().addKeyValue("useDockSpaceMode", useDockSpaceMode);
    if (!usePerformanceMeasurementMode) {
        sgl::AppSettings::get()->getSettings().addKeyValue("useFixedSizeViewport", useFixedSizeViewport);
        sgl::AppSettings::get()->getSettings().addKeyValue("fixedViewportSizeX", fixedViewportSize.x);
        sgl::AppSettings::get()->getSettings().addKeyValue("fixedViewportSizeY", fixedViewportSize.y);
    }
    sgl::AppSettings::get()->getSettings().addKeyValue("showFpsOverlay", showFpsOverlay);
    sgl::AppSettings::get()->getSettings().addKeyValue("showCoordinateAxesOverlay", showCoordinateAxesOverlay);
}

void MainApp::setNewState(const InternalState &newState) {
    rendererVk->getDevice()->waitIdle();

    if (performanceMeasurer) {
        performanceMeasurer->setCurrentAlgorithmBufferSizeBytes(0);
    }

    // 1. Change the window resolution?
    glm::ivec2 newResolution = newState.windowResolution;
    if (useDockSpaceMode) {
        useFixedSizeViewport = true;
        fixedViewportSizeEdit = newResolution;
        fixedViewportSize = newResolution;
    } else {
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        int currentWindowWidth = window->getWidth();
        int currentWindowHeight = window->getHeight();
        if (newResolution.x > 0 && newResolution.y > 0 && currentWindowWidth != newResolution.x
            && currentWindowHeight != newResolution.y) {
            window->setWindowSize(newResolution.x, newResolution.y);
        }
    }

    // 1.1. Handle the new tiling mode for SSBO accesses.
    /*VolumeRenderer::setNewTilingMode(
            newState.tilingWidth, newState.tilingHeight,
            newState.useMortonCodeForTiling);

    // 1.2. Load the new transfer function if necessary.
    if (!newState.transferFunctionName.empty() && newState.transferFunctionName != lastState.transferFunctionName) {
        transferFunctionWindow.loadFunctionFromFile(
                sgl::AppSettings::get()->getDataDirectory()
                + "TransferFunctions/" + newState.transferFunctionName);
    }

    // 2.1. Do we need to load new renderers?
    if (firstState || newState.renderingMode != lastState.renderingMode
            || newState.rendererSettings != lastState.rendererSettings) {
        dataViews.clear();
        if (useDockSpaceMode) {
            if (dataViews.empty()) {
                addNewDataView();
            }
            RenderingMode newRenderingMode = newState.renderingMode;
            setRenderer(
                    dataViews[0]->sceneData, dataViews[0]->oldRenderingMode, newRenderingMode,
                    dataViews[0]->volumeRenderer, 0);
            dataViews[0]->renderingMode = newRenderingMode;
            dataViews[0]->updateCameraMode();
        } else {
            renderingMode = newState.renderingMode;
            setRenderer(sceneData, oldRenderingMode, renderingMode, volumeRenderer, 0);
        }
    }

    // 2.2. Set the new renderer settings.
    bool reloadGatherShader = false;
    std::vector<bool> reloadGatherShaderDataViewList;
    if (useDockSpaceMode) {
        for (DataViewPtr& dataView : dataViews) {
            bool reloadGatherShaderLocal = reloadGatherShader;
            if (dataView->volumeRenderer) {
                dataView->volumeRenderer->setNewState(newState);
                reloadGatherShaderLocal |= dataView->volumeRenderer->setNewSettings(newState.rendererSettings);
                reloadGatherShaderDataViewList.push_back(reloadGatherShaderLocal);
            }
        }
    } else {
        if (volumeRenderer) {
            volumeRenderer->setNewState(newState);
            reloadGatherShader |= volumeRenderer->setNewSettings(newState.rendererSettings);
        }
    }

    // 3. Load the correct data set file.
    if (newState.dataSetDescriptor != lastState.dataSetDescriptor) {
        selectedDataSetIndex = 0;
        std::string nameLower = boost::algorithm::to_lower_copy(newState.dataSetDescriptor.name);
        for (size_t i = 0; i < dataSetInformationList.size(); i++) {
            if (boost::algorithm::to_lower_copy(dataSetInformationList.at(i)->name) == nameLower) {
                selectedDataSetIndex = int(i) + NUM_MANUAL_LOADERS;
                break;
            }
        }
        if (selectedDataSetIndex == 0) {
            if (dataSetInformationList.at(selectedDataSetIndex - NUM_MANUAL_LOADERS)->type
                    == DATA_SET_TYPE_STRESS_VOLUMES && newState.dataSetDescriptor.enabledFileIndices.size() == 3) {
                VolumeDataStress::setUseMajorPS(newState.dataSetDescriptor.enabledFileIndices.at(0));
                VolumeDataStress::setUseMediumPS(newState.dataSetDescriptor.enabledFileIndices.at(1));
                VolumeDataStress::setUseMinorPS(newState.dataSetDescriptor.enabledFileIndices.at(2));
            }
            loadVolumeDataSet(newState.dataSetDescriptor.filenames, true);
        } else {
            if (newState.dataSetDescriptor.type == DATA_SET_TYPE_STRESS_VOLUMES
                && newState.dataSetDescriptor.enabledFileIndices.size() == 3) {
                VolumeDataStress::setUseMajorPS(newState.dataSetDescriptor.enabledFileIndices.at(0));
                VolumeDataStress::setUseMediumPS(newState.dataSetDescriptor.enabledFileIndices.at(1));
                VolumeDataStress::setUseMinorPS(newState.dataSetDescriptor.enabledFileIndices.at(2));
            }
            loadVolumeDataSet(getSelectedDataSetFilenames(), true);
        }
    }

    // 4. Pass state change to filters to handle internally necessary state changes.
    for (VolumeFilter* filter : dataFilters) {
        filter->setNewState(newState);
    }
    for (size_t i = 0; i < newState.filterSettings.size(); i++) {
        dataFilters.at(i)->setNewSettings(newState.filterSettings.at(i));
    }

    // 5. Pass state change to renderers to handle internally necessary state changes.
    if (volumeData) {
        reloadGatherShader |= volumeData->setNewSettings(newState.dataSetSettings);
    }

    // 6. Reload the gather shader if necessary.
    if (useDockSpaceMode) {
        size_t idx = 0;
        for (DataViewPtr& dataView : dataViews) {
            bool reloadGatherShaderLocal = reloadGatherShader || reloadGatherShaderDataViewList.at(idx);
            if (dataView->volumeRenderer && reloadGatherShaderLocal) {
                dataView->volumeRenderer->reloadGatherShaderExternal();
            }
            idx++;
        }
    } else {
        if (volumeRenderer && reloadGatherShader) {
            volumeRenderer->reloadGatherShaderExternal();
        }
    }*/

    recordingTime = 0.0f;
    recordingTimeLast = 0.0f;
    recordingTimeStampStart = sgl::Timer->getTicksMicroseconds();
    lastState = newState;
    firstState = false;
    usesNewState = true;
}

void MainApp::scheduleRecreateSceneFramebuffer() {
    scheduledRecreateSceneFramebuffer = true;
}

void MainApp::addNewRenderer(RenderingMode renderingMode) {
    RendererPtr volumeRenderer;
    setRenderer(renderingMode, volumeRenderer);

    // Opaque surface renderers are always added before transparent renderers.
    bool isOpaqueRenderer =
            renderingMode != RenderingMode::RENDERING_MODE_DIRECT_VOLUME_RENDERING
            && renderingMode != RenderingMode::RENDERING_MODE_DIAGRAM_RENDERER;
    bool isOverlayRenderer = renderingMode == RenderingMode::RENDERING_MODE_DIAGRAM_RENDERER;
    if (isOpaqueRenderer && !isOverlayRenderer) {
        // Push after last opaque renderer (or at the beginning).
        auto it = volumeRenderers.begin();
        while (it != volumeRenderers.end() && it->get()->getIsOpaqueRenderer()) {
            ++it;
        }
        volumeRenderers.insert(it, volumeRenderer);
    } else if (!isOverlayRenderer) {
        // Push before the last overlay renderer.
        auto it = volumeRenderers.begin();
        while (it != volumeRenderers.end() && !it->get()->getIsOverlayRenderer()) {
            ++it;
        }
        volumeRenderers.insert(it, volumeRenderer);
    } else {
        volumeRenderers.push_back(volumeRenderer);
    }
}

void MainApp::setRenderer(RenderingMode newRenderingMode, RendererPtr& newVolumeRenderer) {
    size_t creationId;
    if (newVolumeRenderer) {
        creationId = newVolumeRenderer->getCreationId();
        device->waitIdle();
        newVolumeRenderer = {};
    } else {
        creationId = rendererCreationCounter;
        rendererCreationCounter++;
    }

    if (newRenderingMode == RENDERING_MODE_DIRECT_VOLUME_RENDERING) {
        newVolumeRenderer = std::make_shared<DvrRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_ISOSURFACE_RAYCASTER) {
        newVolumeRenderer = std::make_shared<IsoSurfaceRayCastingRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_ISOSURFACE_RASTERIZER) {
        newVolumeRenderer = std::make_shared<IsoSurfaceRasterizer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_DOMAIN_OUTLINE_RENDERER) {
        newVolumeRenderer = std::make_shared<DomainOutlineRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_SLICE_RENDERER) {
        newVolumeRenderer = std::make_shared<SliceRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_DIAGRAM_RENDERER) {
        newVolumeRenderer = std::make_shared<DiagramRenderer>(viewManager);
    } else {
        int idx = std::clamp(int(newRenderingMode), 0, IM_ARRAYSIZE(RENDERING_MODE_NAMES) - 1);
        std::string warningText =
                std::string() + "The selected renderer \"" + RENDERING_MODE_NAMES[idx] + "\" is not "
                + "supported in this build configuration or incompatible with this system.";
        onUnsupportedRendererSelected(warningText, newVolumeRenderer);
    }

    newVolumeRenderer->setCreationId(creationId);
    newVolumeRenderer->initialize();
    newVolumeRenderer->setUseLinearRGB(useLinearRGB);
    newVolumeRenderer->setFileDialogInstance(fileDialogInstance);

    for (size_t viewIdx = 0; viewIdx < dataViews.size(); viewIdx++) {
        newVolumeRenderer->addView(uint32_t(viewIdx));
        auto& viewSceneData = dataViews.at(viewIdx)->sceneData;
        if (*viewSceneData.sceneTexture) {
            newVolumeRenderer->recreateSwapchainView(
                    uint32_t(viewIdx), *viewSceneData.viewportWidth, *viewSceneData.viewportHeight);
        }
    }
}

void MainApp::onUnsupportedRendererSelected(const std::string& warningText, RendererPtr& newVolumeRenderer) {
    sgl::Logfile::get()->writeWarning(
            "Warning in MainApp::setRenderer: " + warningText, false);
    auto handle = sgl::dialog::openMessageBox(
            "Unsupported Renderer", warningText, sgl::dialog::Icon::WARNING);
    nonBlockingMsgBoxHandles.push_back(handle);
    newVolumeRenderer = std::make_shared<DvrRenderer>(viewManager);
}

void MainApp::resolutionChanged(sgl::EventPtr event) {
    SciVisApp::resolutionChanged(event);
    if (!useDockSpaceMode) {
        auto* window = sgl::AppSettings::get()->getMainWindow();
        viewportWidth = uint32_t(window->getWidth());
        viewportHeight = uint32_t(window->getHeight());
        for (auto& volumeRenderer : volumeRenderers) {
            volumeRenderer->recreateSwapchainView(0, viewportWidth, viewportHeight);
        }
        if (volumeData) {
            volumeData->recreateSwapchainView(0, viewportWidth, viewportHeight);
        }
    }
}

void MainApp::updateColorSpaceMode() {
    SciVisApp::updateColorSpaceMode();
    volumeData->setUseLinearRGB(useLinearRGB);
    if (useDockSpaceMode) {
        for (DataViewPtr& dataView : dataViews) {
            dataView->useLinearRGB = useLinearRGB;
            dataView->viewportWidth = 0;
            dataView->viewportHeight = 0;
        }
    }
    for (auto& volumeRenderer : volumeRenderers) {
        volumeRenderer->setUseLinearRGB(useLinearRGB);
    }
}

void MainApp::render() {
    if (usePerformanceMeasurementMode) {
        performanceMeasurer->beginRenderFunction();
    }

    if (scheduledRecreateSceneFramebuffer) {
        device->waitIdle();
        sgl::vk::Swapchain* swapchain = sgl::AppSettings::get()->getSwapchain();
        createSceneFramebuffer();
        if (swapchain && sgl::AppSettings::get()->getUseGUI()) {
            sgl::ImGuiWrapper::get()->setVkRenderTarget(compositedTextureVk->getImageView());
            sgl::ImGuiWrapper::get()->onResolutionChanged();
        }
        if (videoWriter) {
            videoWriter->onSwapchainRecreated();
        }
        scheduledRecreateSceneFramebuffer = false;
    }

    SciVisApp::preRender();
    if (useDockSpaceMode) {
        for (DataViewPtr& dataView : dataViews) {
            dataView->saveScreenshotDataIfAvailable();
        }
    }

    if (!useDockSpaceMode) {
        prepareVisualizationPipeline();

        componentOtherThanRendererNeedsReRender = reRender;
        if (volumeData != nullptr) {
            bool volumeDataNeedsReRender = volumeData->needsReRender();
            reRender = reRender || volumeDataNeedsReRender;
            componentOtherThanRendererNeedsReRender = componentOtherThanRendererNeedsReRender || volumeDataNeedsReRender;
        }
    }

    if (!useDockSpaceMode) {
        for (auto& volumeRenderer : volumeRenderers) {
            reRender = reRender || volumeRenderer->needsReRender();
            //componentOtherThanRendererNeedsReRender |= volumeRenderer->needsInternalReRender();
        }
        if (componentOtherThanRendererNeedsReRender) {
            // If the re-rendering was triggered from an outside source, frame accumulation cannot be used!
            for (auto& volumeRenderer : volumeRenderers) {
                volumeRenderer->notifyReRenderTriggeredExternally();
            }
        }

        if (reRender || continuousRendering) {
            if (usePerformanceMeasurementMode) {
                performanceMeasurer->startMeasure(recordingTimeLast);
            }

            SciVisApp::prepareReRender();

            // TODO
            rendererVk->insertImageMemoryBarriers(
                    { sceneTextureVk->getImage(), sceneDepthTextureVk->getImage() },
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
            sceneTextureVk->getImageView()->clearColor(
                    clearColor.getFloatColorRGBA(), rendererVk->getVkCommandBuffer());
            sceneDepthTextureVk->getImageView()->clearDepthStencil(
                    1.0f, 0, rendererVk->getVkCommandBuffer());

            if (volumeData) {
                volumeData->renderViewCalculator(0);
            }
            for (auto& volumeRenderer : volumeRenderers) {
                volumeRenderer->renderView(0);
            }

            if (usePerformanceMeasurementMode) {
                performanceMeasurer->endMeasure();
            }

            reRender = false;
        }
    }

    SciVisApp::postRender();

    if (useDockSpaceMode && !dataViews.empty() && !uiOnScreenshot && recording && !isFirstRecordingFrame) {
        auto dataView = dataViews.at(0);
        if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
            rendererVk->transitionImageLayout(
                    dataView->compositedTextureVk->getImage(),
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            videoWriter->pushFramebufferImage(dataView->compositedTextureVk->getImage());
            rendererVk->transitionImageLayout(
                    dataView->compositedTextureVk->getImage(),
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        }
    }
}

void MainApp::renderGui() {
    focusedWindowIndex = -1;
    mouseHoverWindowIndex = -1;

    if (sgl::Keyboard->keyPressed(SDLK_o) && (sgl::Keyboard->getModifier() & (KMOD_LCTRL | KMOD_RCTRL)) != 0) {
        openFileDialog();
    }

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseDataSetFile", ImGuiWindowFlags_NoCollapse,
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

            const char* currentPath = IGFD_GetCurrentPath(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            if (selection.count != 0) {
                filename += selection.table[0].fileName;
            }
            IGFD_Selection_DestroyContent(&selection);
            if (currentPath) {
                free((void*)currentPath);
                currentPath = nullptr;
            }

            fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);

            std::string filenameLower = boost::to_lower_copy(filename);

            if (boost::ends_with(filenameLower, ".vtk")
                    || boost::ends_with(filenameLower, ".vti")
                    || boost::ends_with(filenameLower, ".vts")
                    || boost::ends_with(filenameLower, ".nc")
                    || boost::ends_with(filenameLower, ".zarr")
                    || boost::ends_with(filenameLower, ".am")
                    || boost::ends_with(filenameLower, ".bin")
                    || boost::ends_with(filenameLower, ".field")
                    || boost::ends_with(filenameLower, ".cvol")
#ifdef USE_ECCODES
                    || boost::ends_with(filenameLower, ".grib")
                    || boost::ends_with(filenameLower, ".grb")
#endif
                    || boost::ends_with(filenameLower, ".dat")
                    || boost::ends_with(filenameLower, ".raw")) {
                selectedDataSetIndex = 0;
                customDataSetFileName = filename;
                dataSetType = DataSetType::VOLUME;
                loadVolumeDataSet(getSelectedDataSetFilenames());
            } else {
                sgl::Logfile::get()->writeError(
                        "The selected file name has an unknown extension \""
                        + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
            }
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseExportFieldFile", ImGuiWindowFlags_NoCollapse,
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

            const char* currentPath = IGFD_GetCurrentPath(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            std::string currentFileName;
            if (filter == ".*") {
                currentFileName = IGFD_GetCurrentFileNameRaw(fileDialogInstance);
            } else {
                currentFileName = IGFD_GetCurrentFileName(fileDialogInstance);
            }
            if (selection.count != 0 && selection.table[0].fileName == currentFileName) {
                filename += selection.table[0].fileName;
            } else {
                filename += currentFileName;
            }
            IGFD_Selection_DestroyContent(&selection);
            if (currentPath) {
                free((void*)currentPath);
                currentPath = nullptr;
            }

            exportFieldFileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);

            std::string filenameLower = boost::to_lower_copy(filename);
            if (boost::ends_with(filenameLower, ".nc") || boost::ends_with(filenameLower, ".cvol")) {
                volumeData->saveFieldToFile(filename, FieldType::SCALAR, selectedFieldIndexExport);
            } else {
                sgl::Logfile::get()->writeError(
                        "The selected file name has an unsupported extension \""
                        + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
            }
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (useDockSpaceMode) {
        if (isFirstFrame && dataViews.size() == 1) {
            if (volumeRenderers.empty()) {
                initializeFirstDataView();
            }
            isFirstFrame = false;
        }

        ImGuiID dockSpaceId = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
        ImGuiDockNode* centralNode = ImGui::DockBuilderGetNode(dockSpaceId);
        static bool isProgramStartup = true;
        if (isProgramStartup && centralNode->IsEmpty()) {
            ImGuiID dockLeftId, dockMainId;
            ImGui::DockBuilderSplitNode(
                    dockSpaceId, ImGuiDir_Left, 0.29f, &dockLeftId, &dockMainId);
            ImGui::DockBuilderDockWindow("Opaque Renderer (1)###data_view_0", dockMainId);

            ImGuiID dockLeftUpId, dockLeftDownId;
            ImGui::DockBuilderSplitNode(
                    dockLeftId, ImGuiDir_Up, 0.45f, &dockLeftUpId, &dockLeftDownId);
            ImGui::DockBuilderDockWindow("Property Editor", dockLeftUpId);

            ImGuiID dockLeftDownUpId, dockLeftDownDownId;
            ImGui::DockBuilderSplitNode(
                    dockLeftDownId, ImGuiDir_Up, 0.28f,
                    &dockLeftDownUpId, &dockLeftDownDownId);
            ImGui::DockBuilderDockWindow("Transfer Function", dockLeftDownDownId);
            ImGui::DockBuilderDockWindow("Multi-Var Transfer Function", dockLeftDownDownId);
            ImGui::DockBuilderDockWindow("Camera Checkpoints", dockLeftDownUpId);
            ImGui::DockBuilderDockWindow("Replay Widget", dockLeftDownUpId);

            ImGui::DockBuilderFinish(dockLeftId);
            ImGui::DockBuilderFinish(dockSpaceId);
        }
        isProgramStartup = false;

        renderGuiMenuBar();

        if (showPropertyEditor) {
            renderGuiPropertyEditorWindow();
        }

        prepareVisualizationPipeline();

        componentOtherThanRendererNeedsReRender = reRender;
        if (volumeData != nullptr) {
            bool volumeDataNeedsReRender = volumeData->needsReRender();
            reRender = reRender || volumeDataNeedsReRender;
            componentOtherThanRendererNeedsReRender = componentOtherThanRendererNeedsReRender || volumeDataNeedsReRender;
        }

        bool rendererNeedsReRender = false;
        bool componentOtherThanRendererNeedsReRenderLocal = componentOtherThanRendererNeedsReRender;
        for (auto& volumeRenderer : volumeRenderers) {
            rendererNeedsReRender |= volumeRenderer->needsReRender();
            //componentOtherThanRendererNeedsReRenderLocal |= volumeRenderer->needsInternalReRender();
        }
        if (componentOtherThanRendererNeedsReRenderLocal) {
            // If the re-rendering was triggered from an outside source, frame accumulation cannot be used!
            for (auto& volumeRenderer : volumeRenderers) {
                volumeRenderer->notifyReRenderTriggeredExternally();
            }
        }

        for (int i = 0; i < int(dataViews.size()); i++) {
            auto viewIdx = uint32_t(i);
            DataViewPtr& dataView = dataViews.at(i);
            if (dataView->showWindow) {
                std::string windowName = dataView->getWindowName(i);
                bool isViewOpen = true;
                sgl::ImGuiWrapper::get()->setNextWindowStandardSize(800, 600);
                ImGui::SetNextTabbarMenu([this] {
                    if (ImGui::BeginPopup("#NewTab")) {
                        addNewDataView();
                        ImGui::EndPopup();
                    }

                    return "#NewTab";
                });
                if (ImGui::Begin(windowName.c_str(), &isViewOpen)) {
                    if (ImGui::IsWindowFocused()) {
                        focusedWindowIndex = i;
                    }
                    sgl::ImGuiWrapper::get()->setWindowViewport(i, ImGui::GetWindowViewport());
                    sgl::ImGuiWrapper::get()->setWindowViewport(i, ImGui::GetWindowViewport());
                    sgl::ImGuiWrapper::get()->setWindowPosAndSize(i, ImGui::GetWindowPos(), ImGui::GetWindowSize());

                    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
                    dataView->viewportPositionX = int32_t(cursorPos.x);
                    dataView->viewportPositionY = int32_t(cursorPos.y);
                    ImVec2 sizeContent = ImGui::GetContentRegionAvail();
                    if (useFixedSizeViewport) {
                        sizeContent = ImVec2(float(fixedViewportSize.x), float(fixedViewportSize.y));
                    }
                    if (int(sizeContent.x) != int(dataView->viewportWidth)
                            || int(sizeContent.y) != int(dataView->viewportHeight)) {
                        rendererVk->getDevice()->waitIdle();
                        dataView->resize(int(sizeContent.x), int(sizeContent.y));
                        if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                            for (auto& volumeRenderer : volumeRenderers) {
                                volumeRenderer->recreateSwapchainView(
                                        viewIdx, dataView->viewportWidth, dataView->viewportHeight);
                            }
                            if (volumeData) {
                                volumeData->recreateSwapchainView(
                                        viewIdx, dataView->viewportWidth, dataView->viewportHeight);
                            }
                        }
                        dataView->reRender = true;
                    }

                    bool reRenderLocal = reRender || dataView->reRender || rendererNeedsReRender;
                    dataView->reRender = false;
                    for (auto& volumeRenderer : volumeRenderers) {
                        reRenderLocal |= volumeRenderer->needsReRenderView(viewIdx);
                    }

                    if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0
                            && (reRenderLocal || continuousRendering)) {
                        dataView->beginRender();

                        if (usePerformanceMeasurementMode) {
                            performanceMeasurer->startMeasure(recordingTimeLast);
                        }

                        if (volumeData) {
                            volumeData->renderViewCalculator(viewIdx);
                        }
                        if (volumeData.get() != nullptr) {
                            for (auto& volumeRenderer : volumeRenderers) {
                                volumeRenderer->renderView(viewIdx);
                            }
                        }

                        if (usePerformanceMeasurementMode) {
                            performanceMeasurer->endMeasure();
                        }

                        reRenderLocal = false;

                        dataView->endRender();
                    }

                    if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                        if (!uiOnScreenshot && screenshot) {
                            printNow = true;
                            std::string screenshotFilename =
                                    saveDirectoryScreenshots + saveFilenameScreenshots
                                    + "_" + sgl::toString(screenshotNumber);
                            if (dataViews.size() > 1) {
                                screenshotFilename += "_view" + sgl::toString(i);
                            }
                            screenshotFilename += ".png";

                            dataView->screenshotReadbackHelper->setScreenshotTransparentBackground(
                                    screenshotTransparentBackground);
                            dataView->saveScreenshot(screenshotFilename);
                            screenshot = false;

                            printNow = false;
                            screenshot = true;
                        }

                        if (isViewOpen) {
                            ImTextureID textureId = dataView->getImGuiTextureId();
                            ImGui::Image(
                                    textureId, sizeContent,
                                    ImVec2(0, 0), ImVec2(1, 1));
                            if (ImGui::IsItemHovered()) {
                                mouseHoverWindowIndex = i;
                            }
                        }

                        if (i == 0 && showFpsOverlay) {
                            renderGuiFpsOverlay();
                        }
                        if (i == 0 && showCoordinateAxesOverlay) {
                            renderGuiCoordinateAxesOverlay(dataView->camera);
                        }

                        for (auto& volumeRenderer : volumeRenderers) {
                            volumeRenderer->renderGuiOverlay(viewIdx);
                        }
                        if (volumeData) {
                            volumeData->renderGuiOverlay(viewIdx);
                        }
                    }
                }
                ImGui::End();

                if (!isViewOpen) {
                    dataViews.erase(dataViews.begin() + i);
                    viewManager->removeView(i);
                    for (auto& volumeRenderer : volumeRenderers) {
                        volumeRenderer->removeView(viewIdx);
                    }
                    if (volumeData) {
                        volumeData->removeView(viewIdx);
                    }
                    i--;
                }
            }
        }

        for (auto& volumeRenderer : volumeRenderers) {
            volumeRenderer->renderGuiWindowSecondary();
        }

        if (!uiOnScreenshot && screenshot) {
            screenshot = false;
            screenshotNumber++;
        }
        reRender = false;
    } else {
        if (showPropertyEditor) {
            renderGuiPropertyEditorWindow();
        }

        for (auto& volumeRenderer : volumeRenderers) {
            volumeRenderer->renderGuiOverlay(0);
        }
        if (volumeData) {
            volumeData->renderGuiOverlay(0);
        }
    }

    if (checkpointWindow.renderGui()) {
        fovDegree = camera->getFOVy() / sgl::PI * 180.0f;
        reRender = true;
        hasMoved();
        onCameraReset();
    }

#ifdef USE_PYTHON
    /*ReplayWidget::ReplayWidgetUpdateType replayWidgetUpdateType = replayWidget.renderGui();
    if (replayWidgetUpdateType == ReplayWidget::REPLAY_WIDGET_UPDATE_LOAD) {
        recordingTime = 0.0f;
        //realTimeReplayUpdates = true;
        realTimeReplayUpdates = false;
        sgl::ColorLegendWidget::setFontScale(1.0f);
    }
    if (replayWidgetUpdateType == ReplayWidget::REPLAY_WIDGET_UPDATE_START_RECORDING) {
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        if (useRecordingResolution && window->getWindowResolution() != recordingResolution
            && !window->isFullscreen()) {
            window->setWindowSize(recordingResolution.x, recordingResolution.y);
        }

        if (videoWriter) {
            delete videoWriter;
            videoWriter = nullptr;
        }

        recordingTime = 0.0f;
        realTimeReplayUpdates = false;
        recordingTimeStampStart = sgl::Timer->getTicksMicroseconds();

        recording = true;
        isFirstRecordingFrame = true;
        sgl::ColorLegendWidget::setFontScale(1.0f);
        videoWriter = new sgl::VideoWriter(
                saveDirectoryVideos + saveFilenameVideos
                + "_" + sgl::toString(videoNumber++) + ".mp4", FRAME_RATE_VIDEOS);
        videoWriter->setRenderer(rendererVk);
    }
    if (replayWidgetUpdateType == ReplayWidget::REPLAY_WIDGET_UPDATE_STOP_RECORDING) {
        recording = false;
        sgl::ColorLegendWidget::resetStandardSize();
        if (videoWriter) {
            delete videoWriter;
            videoWriter = nullptr;
        }
    }
    if (replayWidget.getUseCameraFlight()
        && replayWidgetUpdateType != ReplayWidget::REPLAY_WIDGET_UPDATE_STOP_RECORDING) {
        useCameraFlight = true;
        startedCameraFlightPerUI = true;
        realTimeCameraFlight = false;
        cameraPath.resetTime();
        sgl::ColorLegendWidget::setFontScale(1.0f);
    }
    if (replayWidget.getUseCameraFlight()
        && replayWidgetUpdateType == ReplayWidget::REPLAY_WIDGET_UPDATE_STOP_RECORDING) {
        useCameraFlight = false;
        cameraPath.resetTime();
        sgl::ColorLegendWidget::resetStandardSize();
    }
    if (replayWidgetUpdateType != ReplayWidget::REPLAY_WIDGET_UPDATE_NONE) {
        reRender = true;
    }*/
#endif

    if (volumeData) {
        volumeData->renderGuiWindowSecondary();
    }
}

void MainApp::loadAvailableDataSetInformation() {
    dataSetNames.clear();
    dataSetNames.emplace_back("Local file...");
    selectedDataSetIndex = 0;

    const std::string volumeDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/";
    if (sgl::FileUtils::get()->exists(volumeDataSetsDirectory + "datasets.json")) {
        dataSetInformationRoot = loadDataSetList(volumeDataSetsDirectory + "datasets.json");

        std::stack<std::pair<DataSetInformationPtr, size_t>> dataSetInformationStack;
        dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, 0));
        while (!dataSetInformationStack.empty()) {
            std::pair<DataSetInformationPtr, size_t> dataSetIdxPair = dataSetInformationStack.top();
            DataSetInformationPtr dataSetInformationParent = dataSetIdxPair.first;
            size_t idx = dataSetIdxPair.second;
            dataSetInformationStack.pop();
            while (idx < dataSetInformationParent->children.size()) {
                DataSetInformationPtr dataSetInformationChild =
                        dataSetInformationParent->children.at(idx);
                idx++;
                if (dataSetInformationChild->type == DataSetType::NODE) {
                    dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, idx));
                    dataSetInformationStack.push(std::make_pair(dataSetInformationChild, 0));
                    break;
                } else {
                    dataSetInformationChild->sequentialIndex = int(dataSetNames.size());
                    dataSetInformationList.push_back(dataSetInformationChild);
                    dataSetNames.push_back(dataSetInformationChild->name);
                }
            }
        }
    }
}

std::vector<std::string> MainApp::getSelectedDataSetFilenames() {
    std::vector<std::string> filenames;
    if (selectedDataSetIndex == 0) {
        filenames.push_back(customDataSetFileName);
    } else {
        dataSetType = dataSetInformationList.at(selectedDataSetIndex - NUM_MANUAL_LOADERS)->type;
        for (const std::string& filename : dataSetInformationList.at(
                selectedDataSetIndex - NUM_MANUAL_LOADERS)->filenames) {
            filenames.push_back(filename);
        }
    }
    return filenames;
}

void MainApp::renderGuiGeneralSettingsPropertyEditor() {
    if (propertyEditor.addColorEdit3("Clear Color", (float*)&clearColorSelection, 0)) {
        clearColor = sgl::colorFromFloat(
                clearColorSelection.x, clearColorSelection.y, clearColorSelection.z, clearColorSelection.w);
        coordinateAxesOverlayWidget.setClearColor(clearColor);
        if (volumeData) {
            volumeData->setClearColor(clearColor);
        }
        for (DataViewPtr& dataView : dataViews) {
            dataView->setClearColor(clearColor);
        }
        reRender = true;
    }

    // TODO: Remove option?
    /*newDockSpaceMode = useDockSpaceMode;
    if (propertyEditor.addCheckbox("Use Docking Mode", &newDockSpaceMode)) {
        scheduledDockSpaceModeChange = true;
    }*/

    if (propertyEditor.addCheckbox("Fixed Size Viewport", &useFixedSizeViewport)) {
        reRender = true;
    }
    if (useFixedSizeViewport) {
        if (propertyEditor.addSliderInt2Edit("Viewport Size", &fixedViewportSizeEdit.x, 1, 8192)
            == ImGui::EditMode::INPUT_FINISHED) {
            fixedViewportSize = fixedViewportSizeEdit;
            reRender = true;
        }
    }
}

void MainApp::addNewDataView() {
    DataViewPtr dataView = std::make_shared<DataView>(&sceneData);
    dataView->useLinearRGB = useLinearRGB;
    dataView->clearColor = clearColor;
    dataViews.push_back(dataView);
    viewManager->addView(&dataView->sceneData);
    auto viewIdx = uint32_t(dataViews.size() - 1);
    for (auto& volumeRenderer : volumeRenderers) {
        volumeRenderer->addView(viewIdx);
    }
    if (volumeData) {
        volumeData->addView(viewIdx);
    }
}

void MainApp::initializeFirstDataView() {
    DataViewPtr dataView = dataViews.back();
    addNewRenderer(RENDERING_MODE_DOMAIN_OUTLINE_RENDERER);
    addNewRenderer(RENDERING_MODE_DIRECT_VOLUME_RENDERING);
    prepareVisualizationPipeline();
}

void MainApp::openFileDialog() {
    selectedDataSetIndex = 0;
    if (fileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(fileDialogDirectory)) {
        fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/";
        if (!sgl::FileUtils::get()->exists(fileDialogDirectory)) {
            fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
        }
    }
    IGFD_OpenModal(
            fileDialogInstance,
            "ChooseDataSetFile", "Choose a File",
            ".*,.vtk,.vti,.vts,.nc,.zarr,.am,.bin,.field,.cvol,.grib,.grb,.dat,.raw",
            fileDialogDirectory.c_str(),
            "", 1, nullptr,
            ImGuiFileDialogFlags_None);
}

void MainApp::openExportFieldFileDialog() {
    if (exportFieldFileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(exportFieldFileDialogDirectory)) {
        exportFieldFileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/";
        if (!sgl::FileUtils::get()->exists(exportFieldFileDialogDirectory)) {
            exportFieldFileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
        }
    }
    IGFD_OpenModal(
            fileDialogInstance,
            "ChooseExportFieldFile", "Choose a File",
            ".*,.nc,.cvol",
            exportFieldFileDialogDirectory.c_str(),
            "", 1, nullptr,
            ImGuiFileDialogFlags_ConfirmOverwrite);
}

void MainApp::renderGuiMenuBar() {
    bool openExportFieldDialog = false;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Dataset...", "CTRL+O")) {
                openFileDialog();
            }

            if (ImGui::BeginMenu("Datasets")) {
                if (dataSetInformationRoot) {
                    std::stack<std::pair<DataSetInformationPtr, size_t>> dataSetInformationStack;
                    dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, 0));
                    while (!dataSetInformationStack.empty()) {
                        std::pair<DataSetInformationPtr, size_t> dataSetIdxPair = dataSetInformationStack.top();
                        DataSetInformationPtr dataSetInformationParent = dataSetIdxPair.first;
                        size_t idx = dataSetIdxPair.second;
                        dataSetInformationStack.pop();
                        while (idx < dataSetInformationParent->children.size()) {
                            DataSetInformationPtr dataSetInformationChild =
                                    dataSetInformationParent->children.at(idx);
                            if (dataSetInformationChild->type == DataSetType::NODE) {
                                if (ImGui::BeginMenu(dataSetInformationChild->name.c_str())) {
                                    dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, idx + 1));
                                    dataSetInformationStack.push(std::make_pair(dataSetInformationChild, 0));
                                    break;
                                }
                            } else {
                                if (ImGui::MenuItem(dataSetInformationChild->name.c_str())) {
                                    selectedDataSetIndex = int(dataSetInformationChild->sequentialIndex);
                                    loadVolumeDataSet(getSelectedDataSetFilenames());
                                }
                            }
                            idx++;
                        }

                        if (idx == dataSetInformationParent->children.size() && !dataSetInformationStack.empty()) {
                            ImGui::EndMenu();
                        }
                    }
                }

                ImGui::EndMenu();
            }

            if (ImGui::MenuItem("Quit", "CTRL+Q")) {
                quit();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::BeginMenu("New Renderer...")) {
                for (int i = 0; i < IM_ARRAYSIZE(RENDERING_MODE_NAMES); i++) {
                    if (ImGui::MenuItem(RENDERING_MODE_NAMES[i])) {
                        addNewRenderer(RenderingMode(i));
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::MenuItem("New View...")) {
                addNewDataView();
                if (dataViews.size() == 1) {
                    initializeFirstDataView();
                }
            }
            if (volumeData && ImGui::BeginMenu("New Calculator...")) {
                volumeData->renderGuiNewCalculators();
                ImGui::EndMenu();
            }
            ImGui::Separator();
            for (int i = 0; i < int(dataViews.size()); i++) {
                DataViewPtr& dataView = dataViews.at(i);
                std::string windowName = dataView->getWindowName(i);
                if (ImGui::MenuItem(windowName.c_str(), nullptr, dataView->showWindow)) {
                    dataView->showWindow = !dataView->showWindow;
                }
            }
            if (ImGui::MenuItem("FPS Overlay", nullptr, showFpsOverlay)) {
                showFpsOverlay = !showFpsOverlay;
            }
            if (ImGui::MenuItem("Coordinate Axes Overlay", nullptr, showCoordinateAxesOverlay)) {
                showCoordinateAxesOverlay = !showCoordinateAxesOverlay;
            }
            if (ImGui::MenuItem("Property Editor", nullptr, showPropertyEditor)) {
                showPropertyEditor = !showPropertyEditor;
            }
            if (ImGui::MenuItem("Checkpoint Window", nullptr, checkpointWindow.getShowWindow())) {
                checkpointWindow.setShowWindow(!checkpointWindow.getShowWindow());
            }
            //if (ImGui::MenuItem("Replay Widget", nullptr, replayWidget.getShowWindow())) {
            //    replayWidget.setShowWindow(!replayWidget.getShowWindow());
            //}
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Tools")) {
            if (volumeData && ImGui::MenuItem("Export Field...")) {
                openExportFieldDialog = true;
            }
            if (ImGui::MenuItem("Print Camera State")) {
                std::cout << "Position: (" << camera->getPosition().x << ", " << camera->getPosition().y
                          << ", " << camera->getPosition().z << ")" << std::endl;
                std::cout << "Look At: (" << camera->getLookAtLocation().x << ", " << camera->getLookAtLocation().y
                          << ", " << camera->getLookAtLocation().z << ")" << std::endl;
                std::cout << "Yaw: " << camera->getYaw() << std::endl;
                std::cout << "Pitch: " << camera->getPitch() << std::endl;
                std::cout << "FoVy: " << (camera->getFOVy() / sgl::PI * 180.0f) << std::endl;
            }
            ImGui::EndMenu();
        }

        /*bool isRendererComputationRunning = false;
        for (DataViewPtr& dataView : dataViews) {
            isRendererComputationRunning =
                    isRendererComputationRunning
                    || (dataView->lineRenderer && dataView->lineRenderer->getIsComputationRunning());
            if (isRendererComputationRunning) {
                break;
            }
        }

        if (lineDataRequester.getIsProcessingRequest() || isRendererComputationRunning) {
            ImGui::SetCursorPosX(ImGui::GetWindowContentRegionWidth() - ImGui::GetTextLineHeight());
            ImGui::ProgressSpinner(
                    "##progress-spinner", -1.0f, -1.0f, 4.0f,
                    ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
        }*/

        ImGui::EndMainMenuBar();
    }

    if (openExportFieldDialog) {
        ImGui::OpenPopup("Export Field");
    }

    if (ImGui::BeginPopupModal("Export Field", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        ImGui::Combo("Field Name", &selectedFieldIndexExport, fieldNames.data(), int(fieldNames.size()));
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            openExportFieldFileDialog();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void MainApp::renderGuiPropertyEditorBegin() {
    if (!useDockSpaceMode) {
        renderGuiFpsCounter();

        if (ImGui::Combo(
                "Data Set", &selectedDataSetIndex, dataSetNames.data(),
                int(dataSetNames.size()))) {
            if (selectedDataSetIndex >= NUM_MANUAL_LOADERS) {
                loadVolumeDataSet(getSelectedDataSetFilenames());
            }
        }

        /*if (lineDataRequester.getIsProcessingRequest() || lineRenderer->getIsComputationRunning()) {
            ImGui::SameLine();
            ImGui::ProgressSpinner(
                    "##progress-spinner", -1.0f, -1.0f, 4.0f,
                    ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
        }*/

        if (selectedDataSetIndex == 0) {
            ImGui::InputText("##datasetfilenamelabel", &customDataSetFileName);
            ImGui::SameLine();
            if (ImGui::Button("Load File")) {
                loadVolumeDataSet(getSelectedDataSetFilenames());
            }
        }

        ImGui::Separator();
    }
}

void MainApp::renderGuiPropertyEditorCustomNodes() {
    if (volumeData) {
        volumeData->renderGui(propertyEditor);
    }

    if (useDockSpaceMode && dataViews.size() > 1) {
        for (int i = 0; i < int(dataViews.size()); i++) {
            DataViewPtr& dataView = dataViews.at(i);
            bool beginNode = propertyEditor.beginNode(dataView->getWindowName(i));
            if (beginNode) {
                if (propertyEditor.addCheckbox("Sync with Global Camera", &dataView->syncWithParentCamera)) {
                    dataView->reRender = true;
                    for (auto& volumeRenderer : volumeRenderers) {
                        volumeRenderer->notifyReRenderTriggeredExternally();
                    }
                }
                propertyEditor.endNode();
            }
        }
    }
    for (int i = 0; i < int(volumeRenderers.size()); i++) {
        auto& volumeRenderer = volumeRenderers.at(i);
        bool removeRenderer = false;
        std::string windowName =
                volumeRenderer->getWindowName() + "###renderer_" + std::to_string(volumeRenderer->getCreationId());
        bool beginNode = propertyEditor.beginNode(windowName);
        ImGui::SameLine();
        float indentWidth = ImGui::GetContentRegionAvail().x;
        ImGui::Indent(indentWidth);
        std::string buttonName = "X###x_renderer" + std::to_string(i);
        if (ImGui::Button(buttonName.c_str())) {
            removeRenderer = true;
        }
        ImGui::Unindent(indentWidth);
        if (beginNode) {
            std::string previewValue = volumeRenderer->getWindowName();
            if (propertyEditor.addBeginCombo("Rendering Mode", previewValue)) {
                for (int j = 0; j < IM_ARRAYSIZE(RENDERING_MODE_NAMES); j++) {
                    if (ImGui::Selectable(
                            RENDERING_MODE_NAMES[int(j)], false,
                            ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups)) {
                        ImGui::CloseCurrentPopup();
                        setRenderer(RenderingMode(j), volumeRenderer);
                        prepareVisualizationPipeline();
                        reRender = true;
                    }
                }
                propertyEditor.addEndCombo();
            }

            volumeRenderer->renderGui(propertyEditor);
            propertyEditor.endNode();
        }
        if (removeRenderer) {
            reRender = true;
            volumeRenderers.erase(volumeRenderers.begin() + i);
            i--;
        }
    }

    if (volumeData) {
        volumeData->renderGuiCalculators(propertyEditor);
    }
}

void MainApp::update(float dt) {
    sgl::SciVisApp::update(dt);

    for (int i = 0; i < int(nonBlockingMsgBoxHandles.size()); i++) {
        auto& handle = nonBlockingMsgBoxHandles.at(i);
        if (handle->ready(0)) {
            nonBlockingMsgBoxHandles.erase(nonBlockingMsgBoxHandles.begin() + i);
            i--;
        }
    }

    if (scheduledDockSpaceModeChange) {
        if (useDockSpaceMode) {
            dataViews.clear();
        } else {
            addNewDataView();
        }

        useDockSpaceMode = newDockSpaceMode;
        scheduledDockSpaceModeChange = false;
    }

    if (usePerformanceMeasurementMode && !performanceMeasurer->update(recordingTime)) {
        // All modes were tested -> quit.
        quit();
    }

    updateCameraFlight(volumeData.get() != nullptr, usesNewState);

#ifdef USE_PYTHON
    /*bool stopRecording = false;
    bool stopCameraFlight = false;
    if (replayWidget.update(recordingTime, stopRecording, stopCameraFlight)) {
        if (!useCameraFlight) {
            camera->overwriteViewMatrix(replayWidget.getViewMatrix());
            if (std::abs(camera->getFOVy() - replayWidget.getCameraFovy()) > 1e-6f) {
                camera->setFOVy(replayWidget.getCameraFovy());
                fovDegree = camera->getFOVy() / sgl::PI * 180.0f;
            }
            if (camera->getLookAtLocation() != replayWidget.getLookAtLocation()) {
                camera->setLookAtLocation(replayWidget.getLookAtLocation());
            }
        }
        SettingsMap currentDatasetSettings = replayWidget.getCurrentDatasetSettings();
        bool reloadGatherShader = false;
        if (lineData) {
            reloadGatherShader |= lineData->setNewSettings(currentDatasetSettings);
        }
        if (useDockSpaceMode) {
            for (DataViewPtr& dataView : dataViews) {
                bool reloadGatherShaderLocal = reloadGatherShader;
                if (dataView->lineRenderer) {
                    SettingsMap currentRendererSettings = replayWidget.getCurrentRendererSettings();
                    reloadGatherShaderLocal |= dataView->lineRenderer->setNewSettings(currentRendererSettings);
                }
                if (dataView->lineRenderer && reloadGatherShaderLocal) {
                    dataView->lineRenderer->reloadGatherShaderExternal();
                }
            }
        } else {
            if (lineRenderer) {
                SettingsMap currentRendererSettings = replayWidget.getCurrentRendererSettings();
                reloadGatherShader |= lineRenderer->setNewSettings(currentRendererSettings);
            }
            if (lineRenderer && reloadGatherShader) {
                lineRenderer->reloadGatherShaderExternal();
            }
        }

        if (updateTransferFunctionRange) {
            transferFunctionWindow.setSelectedRange(transferFunctionRange);
            updateTransferFunctionRange = false;
            lineData->onTransferFunctionMapRebuilt();
            sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
                    ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }

        replayWidgetRunning = true;
        reRender = true;
        hasMoved();

        if (!useCameraFlight) {
            if (realTimeReplayUpdates) {
                uint64_t currentTimeStamp = sgl::Timer->getTicksMicroseconds();
                uint64_t timeElapsedMicroSec = currentTimeStamp - recordingTimeStampStart;
                recordingTime = timeElapsedMicroSec * 1e-6f;
            } else {
                recordingTime += FRAME_TIME_CAMERA_PATH;
            }
        }
    } else {
        if (replayWidgetRunning) {
            sgl::ColorLegendWidget::resetStandardSize();
            replayWidgetRunning = false;
        }
    }
    if (stopRecording) {
        recording = false;
        sgl::ColorLegendWidget::resetStandardSize();
        if (videoWriter) {
            delete videoWriter;
            videoWriter = nullptr;
        }
        if (useCameraFlight) {
            useCameraFlight = false;
            cameraPath.resetTime();
        }
    }
    if (stopCameraFlight) {
        sgl::ColorLegendWidget::resetStandardSize();
        useCameraFlight = false;
        cameraPath.resetTime();
    }*/
#endif

    viewManager->setMouseHoverWindowIndex(mouseHoverWindowIndex);

    if (volumeData) {
        volumeData->update(dt);
    }

    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard || recording || focusedWindowIndex != -1) {
        if (useDockSpaceMode) {
            for (int i = 0; i < int(dataViews.size()); i++) {
                DataViewPtr& dataView = dataViews.at(i);
                if (i != focusedWindowIndex) {
                    continue;
                }

                sgl::CameraPtr parentCamera = this->camera;
                bool reRenderOld = reRender;
                if (!dataView->syncWithParentCamera) {
                    this->camera = dataView->camera;
                    hasMovedIndex = i;
                }
                this->reRender = false;
                moveCameraKeyboard(dt);
                if (this->reRender && dataView->syncWithParentCamera) {
                    for (DataViewPtr& dataViewLocal : dataViews) {
                        if (dataViewLocal->syncWithParentCamera) {
                            dataViewLocal->reRender = dataView->reRender || this->reRender;
                        }
                    }
                }
                if (!dataView->syncWithParentCamera) {
                    dataView->reRender = dataView->reRender || this->reRender;
                    this->camera = parentCamera;
                    hasMovedIndex = -1;
                }
                this->reRender = reRenderOld;
            }
        } else {
            moveCameraKeyboard(dt);
        }
    }

    if (!io.WantCaptureMouse || mouseHoverWindowIndex != -1) {
        if (useDockSpaceMode) {
            for (int i = 0; i < int(dataViews.size()); i++) {
                DataViewPtr& dataView = dataViews.at(i);
                if (i != mouseHoverWindowIndex) {
                    continue;
                }

                sgl::CameraPtr parentCamera = this->camera;
                bool reRenderOld = reRender;
                if (!dataView->syncWithParentCamera) {
                    this->camera = dataView->camera;
                    hasMovedIndex = i;
                }
                this->reRender = false;
                moveCameraMouse(dt);
                if (this->reRender && dataView->syncWithParentCamera) {
                    for (DataViewPtr& dataViewLocal : dataViews) {
                        if (dataViewLocal->syncWithParentCamera) {
                            dataViewLocal->reRender = dataView->reRender || this->reRender;
                        }
                    }
                }
                if (!dataView->syncWithParentCamera) {
                    dataView->reRender = dataView->reRender || this->reRender;
                    this->camera = parentCamera;
                    hasMovedIndex = -1;
                }
                this->reRender = reRenderOld;
            }
        } else {
            moveCameraMouse(dt);
        }

        for (auto& volumeRenderer : volumeRenderers) {
            volumeRenderer->update(dt);
        }
    }
}

void MainApp::hasMoved() {
    if (useDockSpaceMode) {
        if (hasMovedIndex < 0) {
            uint32_t viewIdx = 0;
            for (DataViewPtr& dataView : dataViews) {
                if (dataView->syncWithParentCamera) {
                    dataView->syncCamera();
                }
                for (auto& volumeRenderer : volumeRenderers) {
                    volumeRenderer->onHasMoved(viewIdx);
                }
                viewIdx++;
            }
        } else {
            for (auto& volumeRenderer : volumeRenderers) {
                volumeRenderer->onHasMoved(uint32_t(hasMovedIndex));
            }
        }
    } else {
        for (auto& volumeRenderer : volumeRenderers) {
            volumeRenderer->onHasMoved(0);
        }
    }
}

void MainApp::onCameraReset() {
    if (useDockSpaceMode) {
        for (DataViewPtr& dataView : dataViews) {
            dataView->camera->setNearClipDistance(camera->getNearClipDistance());
            dataView->camera->setFarClipDistance(camera->getFarClipDistance());
            dataView->camera->setYaw(camera->getYaw());
            dataView->camera->setPitch(camera->getPitch());
            dataView->camera->setFOVy(camera->getFOVy());
            dataView->camera->setPosition(camera->getPosition());
            dataView->camera->resetLookAtLocation();
        }
    }
}



// --- Visualization pipeline ---

void MainApp::loadVolumeDataSet(const std::vector<std::string>& fileNames) {
    if (fileNames.empty() || fileNames.front().empty()) {
        volumeData = {};
        return;
    }
    currentlyLoadedDataSetIndex = selectedDataSetIndex;

    DataSetInformation selectedDataSetInformation;
    if (selectedDataSetIndex >= NUM_MANUAL_LOADERS && !dataSetInformationList.empty()) {
        selectedDataSetInformation = *dataSetInformationList.at(selectedDataSetIndex - NUM_MANUAL_LOADERS);
    } else {
        selectedDataSetInformation.type = dataSetType;
        selectedDataSetInformation.filenames = fileNames;
    }

    glm::mat4 transformationMatrix = sgl::matrixIdentity();
    glm::mat4* transformationMatrixPtr = nullptr;
    if (selectedDataSetInformation.hasCustomTransform) {
        transformationMatrix *= selectedDataSetInformation.transformMatrix;
        transformationMatrixPtr = &transformationMatrix;
    }
    if (rotateModelBy90DegreeTurns != 0) {
        transformationMatrix *= glm::rotate(float(rotateModelBy90DegreeTurns) * sgl::HALF_PI, modelRotationAxis);
        transformationMatrixPtr = &transformationMatrix;
    }
    if (selectedDataSetInformation.heightScale != 1.0f) {
        transformationMatrix *= glm::scale(glm::vec3(1.0f, selectedDataSetInformation.heightScale, 1.0f));
        transformationMatrixPtr = &transformationMatrix;
    }

    VolumeDataPtr newVolumeData;
    if (dataSetType == DataSetType::VOLUME) {
        newVolumeData = std::make_shared<VolumeData>(rendererVk);
    } else {
        sgl::Logfile::get()->writeError("Error in MainApp::loadVolumeDataSet: Invalid data set type.");
        return;
    }
    newVolumeData->setFileDialogInstance(fileDialogInstance);
    newVolumeData->setViewManager(viewManager);

    bool dataLoaded = newVolumeData->setInputFiles(fileNames, selectedDataSetInformation, transformationMatrixPtr);
    sgl::ColorLegendWidget::resetStandardSize();

    if (dataLoaded) {
        volumeData = newVolumeData;
        //lineData->onMainThreadDataInit();
        volumeData->recomputeHistogram();
        volumeData->setClearColor(clearColor);
        volumeData->setUseLinearRGB(useLinearRGB);
        for (size_t viewIdx = 0; viewIdx < dataViews.size(); viewIdx++) {
            volumeData->addView(uint32_t(viewIdx));
            auto& viewSceneData = dataViews.at(viewIdx)->sceneData;
            if (*viewSceneData.sceneTexture) {
                volumeData->recreateSwapchainView(
                        uint32_t(viewIdx), *viewSceneData.viewportWidth, *viewSceneData.viewportHeight);
            }
        }
        newDataLoaded = true;
        reRender = true;
        boundingBox = volumeData->getBoundingBoxRendering();
        selectedFieldIndexExport = 0;

        std::string meshDescriptorName = fileNames.front();
        if (fileNames.size() > 1) {
            meshDescriptorName += std::string() + "_" + std::to_string(fileNames.size());
        }
        checkpointWindow.onLoadDataSet(meshDescriptorName);

        if (true) { // useCameraFlight
            std::string cameraPathFilename =
                    saveDirectoryCameraPaths + sgl::FileUtils::get()->getPathAsList(meshDescriptorName).back()
                    + ".binpath";
            if (sgl::FileUtils::get()->exists(cameraPathFilename)) {
                cameraPath.fromBinaryFile(cameraPathFilename);
            } else {
                cameraPath.fromCirclePath(
                        boundingBox, meshDescriptorName,
                        usePerformanceMeasurementMode
                        ? CAMERA_PATH_TIME_PERFORMANCE_MEASUREMENT : CAMERA_PATH_TIME_RECORDING,
                        usePerformanceMeasurementMode);
                //cameraPath.saveToBinaryFile(cameraPathFilename);
            }
        }
    }
}

void MainApp::reloadDataSet() {
    loadVolumeDataSet(getSelectedDataSetFilenames());
}

void MainApp::prepareVisualizationPipeline() {
    if (volumeData && !volumeRenderers.empty()) {
        bool isPreviousNodeDirty = volumeData->isDirty();
        for (auto& volumeRenderer : volumeRenderers) {
            if (volumeRenderer->isDirty() || isPreviousNodeDirty) {
                rendererVk->getDevice()->waitIdle();
                volumeRenderer->setVolumeData(volumeData, newDataLoaded);
                volumeRenderer->resetDirty();
            }
        }
        volumeData->resetDirty();
    }
    newDataLoaded = false;
}
