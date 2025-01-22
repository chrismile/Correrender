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

#ifdef USE_ZEROMQ
#include <zmq.h>
#endif

#ifdef SUPPORT_QUICK_MLP
#include <ckl/kernel_loader.h>
#include <qmlp/qmlp.h>
#endif

#include <Utils/Timer.hpp>
#include <Utils/StringUtils.hpp>
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
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
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
#include "Calculators/Similarity.hpp"
#include "Renderers/DvrRenderer.hpp"
#include "Renderers/IsoSurfaceRayCastingRenderer.hpp"
#include "Renderers/IsoSurfaceRasterizer.hpp"
#include "Renderers/DomainOutlineRenderer.hpp"
#include "Renderers/SliceRenderer.hpp"
#include "Renderers/WorldMapRenderer.hpp"
#include "Renderers/Diagram/DiagramRenderer.hpp"
#include "Renderers/Diagram/Scatter/ScatterPlotRenderer.hpp"
#include "Renderers/Diagram/CorrelationMatrix/CorrelationMatrixRenderer.hpp"
#include "Renderers/Diagram/TimeSeriesCorrelation/TimeSeriesCorrelationRenderer.hpp"
#include "Renderers/Diagram/DistributionSimilarity/DistributionSimilarityRenderer.hpp"
#include "Utils/CurlWrapper.hpp"
#include "Utils/AutomaticPerformanceMeasurer.hpp"
#include "Optimization/TFOptimization.hpp"

#include "Widgets/ViewManager.hpp"
#include "Widgets/DataView.hpp"
#include "MainApp.hpp"

void vulkanErrorCallback() {
#ifdef SGL_INPUT_API_V2
    sgl::AppSettings::get()->captureMouse(false);
#else
    SDL_CaptureMouse(SDL_FALSE);
#endif
    std::cerr << "Application callback" << std::endl;
}

#ifdef __linux__
void signalHandler(int signum) {
#ifdef SGL_INPUT_API_V2
    sgl::AppSettings::get()->captureMouse(false);
#else
    SDL_CaptureMouse(SDL_FALSE);
#endif
    std::cerr << "Interrupt signal (" << signum << ") received." << std::endl;
    exit(signum);
}
#endif

MainApp::MainApp()
        : sceneData(
                &rendererVk, &sceneTextureVk, &sceneDepthTextureVk,
                &viewportPositionX, &viewportPositionY,
                &viewportWidth, &viewportHeight, &viewportWidth, &viewportHeight,
                camera, &clearColor, &screenshotTransparentBackground,
                &performanceMeasurer, &continuousRendering, &recording,
                &useCameraFlight, &MOVE_SPEED, &MOUSE_ROT_SPEED,
                &nonBlockingMsgBoxHandles),
          boundingBox() {
    sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallback);
    clearColor = sgl::Color(0, 0, 0, 255);
    clearColorSelection = ImColor(0, 0, 0, 255);
    std::string clearColorString;
    if (sgl::AppSettings::get()->getSettings().getValueOpt("clearColor", clearColorString)) {
        std::vector<std::string> clearColorStringParts;
        sgl::splitString(clearColorString, ',', clearColorStringParts);
        if (clearColorStringParts.size() == 3 || clearColorStringParts.size() == 4) {
            clearColor.setR(uint8_t(sgl::fromString<int>(clearColorStringParts.at(0))));
            clearColor.setG(uint8_t(sgl::fromString<int>(clearColorStringParts.at(1))));
            clearColor.setB(uint8_t(sgl::fromString<int>(clearColorStringParts.at(2))));
            clearColorSelection = ImColor(clearColor.getR(), clearColor.getG(), clearColor.getB(), 255);
        }
    }

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

    if (cudaInteropInitialized) {
        nvrtcInitialized = true;
        if (!sgl::vk::initializeNvrtcFunctionTable()) {
            nvrtcInitialized = false;
            sgl::Logfile::get()->writeWarning(
                    "Warning in MainApp::MainApp: sgl::vk::initializeNvrtcFunctionTable() returned false.",
                    false);
        }
    }
#endif

#ifdef SUPPORT_OPENCL_INTEROP
    openclInteropInitialized = true;
    if (!sgl::vk::initializeOpenCLFunctionTable()) {
        openclInteropInitialized = false;
    }
#endif

    viewManager = new ViewManager(&clearColor, rendererVk);
    sgl::ColorLegendWidget::setFontScaleStandard(1.0f);

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

    curlInitWrapper();

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
    isProgramStart = false;

    const std::string volumeDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/";
    sgl::FileUtils::get()->ensureDirectoryExists(volumeDataSetsDirectory);
#ifdef __linux__
    datasetsWatch.setPath(volumeDataSetsDirectory + "datasets.json", false);
    datasetsWatch.initialize();
#endif

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

    tfOptimization = new TFOptimization(rendererVk);
#ifdef CUDA_ENABLED
    if (cudaInteropInitialized) {
        tfOptimization->setCudaContext(cuContext);
    }
#endif
    tfOptimization->initialize();

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

    delete tfOptimization;
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
    if (sgl::vk::getIsNvrtcFunctionTableInitialized()) {
        sgl::vk::freeNvrtcFunctionTable();
    }
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

    curlFreeWrapper();

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

    std::string clearColorString =
            std::to_string(int(clearColor.getR())) + ","
            + std::to_string(int(clearColor.getG())) + ","
            + std::to_string(int(clearColor.getB()));
    sgl::AppSettings::get()->getSettings().addKeyValue("clearColor", clearColorString);
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
        std::string nameLower = sgl::toLowerCopy(newState.dataSetDescriptor.name);
        for (size_t i = 0; i < dataSetInformationList.size(); i++) {
            if (sgl::toLowerCopy(dataSetInformationList.at(i)->name) == nameLower) {
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
            && renderingMode != RenderingMode::RENDERING_MODE_DIAGRAM_RENDERER
            && renderingMode != RenderingMode::RENDERING_MODE_SCATTER_PLOT
            && renderingMode != RenderingMode::RENDERING_MODE_CORRELATION_MATRIX
            && renderingMode != RenderingMode::RENDERING_MODE_TIME_SERIES_CORRELATION
            && renderingMode != RenderingMode::RENDERING_MODE_DISTRIBUTION_SIMILARITY;
    bool isOverlayRenderer =
            renderingMode == RenderingMode::RENDERING_MODE_DIAGRAM_RENDERER
            || renderingMode == RenderingMode::RENDERING_MODE_SCATTER_PLOT
            || renderingMode == RenderingMode::RENDERING_MODE_CORRELATION_MATRIX
            || renderingMode == RenderingMode::RENDERING_MODE_TIME_SERIES_CORRELATION
            || renderingMode == RenderingMode::RENDERING_MODE_DISTRIBUTION_SIMILARITY;
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
    } else if (newRenderingMode == RENDERING_MODE_WORLD_MAP_RENDERER) {
        newVolumeRenderer = std::make_shared<WorldMapRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_DIAGRAM_RENDERER) {
        newVolumeRenderer = std::make_shared<DiagramRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_SCATTER_PLOT) {
        newVolumeRenderer = std::make_shared<ScatterPlotRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_CORRELATION_MATRIX) {
        newVolumeRenderer = std::make_shared<CorrelationMatrixRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_TIME_SERIES_CORRELATION) {
        newVolumeRenderer = std::make_shared<TimeSeriesCorrelationRenderer>(viewManager);
    } else if (newRenderingMode == RENDERING_MODE_DISTRIBUTION_SIMILARITY) {
        newVolumeRenderer = std::make_shared<DistributionSimilarityRenderer>(viewManager);
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
                    uint32_t(viewIdx), *viewSceneData.viewportWidthVirtual, *viewSceneData.viewportHeightVirtual);
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

void MainApp::beginFrameMarker() {
#ifdef SUPPORT_RENDERDOC_DEBUGGER
    renderDocDebugger.startFrameCapture();
#endif
}

void MainApp::endFrameMarker() {
#ifdef SUPPORT_RENDERDOC_DEBUGGER
    renderDocDebugger.endFrameCapture();
#endif
}

void MainApp::render() {
    // Debug Code.
    /*static bool isFirstFrame = true;
    if (isFirstFrame) {
        selectedDataSetIndex = 0;
        customDataSetFileName = "/home/christoph/datasets/Toy/chord/linear_4x4.nc";
        dataSetType = DataSetType::VOLUME;
        loadVolumeDataSet({ customDataSetFileName });
        isFirstFrame = false;
    }

    static int frameNum = 0;
    frameNum++;
    if (frameNum == 10) {
        quit();
    }*/

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
            if (screenshotTransparentBackground) {
                clearColor.setA(0);
            }
            rendererVk->insertImageMemoryBarriers(
                    { sceneTextureVk->getImage(), sceneDepthTextureVk->getImage() },
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
            sceneTextureVk->getImageView()->clearColor(
                    clearColor.getFloatColorRGBA(), rendererVk->getVkCommandBuffer());
            sceneDepthTextureVk->getImageView()->clearDepthStencil(
                    1.0f, 0, rendererVk->getVkCommandBuffer());
            if (screenshotTransparentBackground) {
                clearColor.setA(255);
            }

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

    if (useReplicabilityStampMode) {
        const auto NUM_STEPS = int(sgl::AppSettings::get()->getSwapchain()->getNumImages());
        if (replicabilityFrameNumber < NUM_STEPS) {
            replicabilityFrameNumber++;
        } else if (replicabilityFrameNumber == NUM_STEPS) {
            useReplicabilityStampMode = false;
            replicabilityFrameNumber++;
            loadReplicabilityStampState();
        }
    }

#ifdef SGL_INPUT_API_V2
    if (sgl::Keyboard->keyPressed(ImGuiKey_O) && sgl::Keyboard->getModifier(ImGuiKey_ModCtrl)) {
        openFileDialog();
    }
#else
    if (sgl::Keyboard->keyPressed(SDLK_o) && (sgl::Keyboard->getModifier() & (KMOD_LCTRL | KMOD_RCTRL)) != 0) {
        openFileDialog();
    }
#endif

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseDataSetFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            if (selection.count != 0) {
                filename += selection.table[0].fileName;
            }
            IGFD_Selection_DestroyContent(&selection);

            fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);

            std::string filenameLower = sgl::toLowerCopy(filename);
            if (checkHasValidExtension(filenameLower)) {
                selectedDataSetIndex = 0;
                customDataSetFileName = filename;
                customDataSetFileNames.clear();
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
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            std::string currentFileName;
            if (filter == ".*") {
                currentFileName = IGFD_GetCurrentFileNameRawString(fileDialogInstance);
            } else {
                currentFileName = IGFD_GetCurrentFileNameString(fileDialogInstance);
            }
            if (selection.count != 0 && selection.table[0].fileName == currentFileName) {
                filename += selection.table[0].fileName;
            } else {
                filename += currentFileName;
            }
            IGFD_Selection_DestroyContent(&selection);

            exportFieldFileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);

            std::string filenameLower = sgl::toLowerCopy(filename);
            if (sgl::endsWith(filenameLower, ".nc") || sgl::endsWith(filenameLower, ".cvol")) {
                volumeData->saveFieldToFile(filename, FieldType::SCALAR, selectedFieldIndexExport);
            } else {
                sgl::Logfile::get()->writeError(
                        "The selected file name has an unsupported extension \""
                        + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
            }
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseStateFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            std::string currentFileName;
            if (filter == ".*") {
                currentFileName = IGFD_GetCurrentFileNameRawString(fileDialogInstance);
            } else {
                currentFileName = IGFD_GetCurrentFileNameString(fileDialogInstance);
            }
            if (selection.count != 0 && selection.table[0].fileName == currentFileName) {
                filename += selection.table[0].fileName;
            } else {
                filename += currentFileName;
            }
            IGFD_Selection_DestroyContent(&selection);

            stateFileDirectory = sgl::FileUtils::get()->getPathToFile(filename);
            if (stateModeSave) {
                saveStateToFile(filename);
            } else {
                loadStateFromFile(filename);
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

        static bool isProgramStartup = true;
        ImGuiID dockSpaceId = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
        if (isProgramStartup) {
            ImGuiDockNode* centralNode = ImGui::DockBuilderGetNode(dockSpaceId);
            if (centralNode->IsEmpty()) {
                auto* window = sgl::AppSettings::get()->getMainWindow();
                //const ImVec2 dockSpaceSize = ImGui::GetMainViewport()->Size;//ImGui::GetContentRegionAvail();
                const ImVec2 dockSpaceSize(float(window->getWidth()), float(window->getHeight()));
                ImGui::DockBuilderSetNodeSize(dockSpaceId, dockSpaceSize);

                ImGuiID dockLeftId, dockMainId;
                ImGui::DockBuilderSplitNode(
                        dockSpaceId, ImGuiDir_Left, 0.29f, &dockLeftId, &dockMainId);
                ImGui::DockBuilderSetNodeSize(dockLeftId, ImVec2(dockSpaceSize.x * 0.29f, dockSpaceSize.y));
                ImGui::DockBuilderDockWindow("Data View###data_view_0", dockMainId);

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

                ImGui::DockBuilderFinish(dockSpaceId);
            }
            isProgramStartup = false;
        }

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

        int first3dViewIdx = 0;
        for (int i = 0; i < int(dataViews.size()); i++) {
            for (auto& renderer : volumeRenderers) {
                if (renderer->isVisibleInView(i) && !renderer->getIsOverlayRenderer()) {
                    first3dViewIdx = i;
                    break;
                }
            }
        }

        for (int i = 0; i < int(dataViews.size()); i++) {
            auto viewIdx = uint32_t(i);
            DataViewPtr& dataView = dataViews.at(i);
            if (dataView->showWindow) {
                std::string windowName = dataView->getWindowNameImGui(dataViews, i);
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
                                        viewIdx, dataView->viewportWidthVirtual, dataView->viewportHeightVirtual);
                            }
                            if (volumeData) {
                                volumeData->recreateSwapchainView(
                                        viewIdx, dataView->viewportWidthVirtual, dataView->viewportHeightVirtual);
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
                            for (auto& volumeRenderer : volumeRenderers) {
                                volumeRenderer->renderViewPre(viewIdx);
                            }
                            int rendererIdx = 0;
                            for (auto& volumeRenderer : volumeRenderers) {
                                volumeRenderer->renderView(viewIdx);
                                if (rendererIdx != int(volumeRenderers.size() - 1) && volumeRenderer->getIsOpaqueRenderer()
                                        && !volumeRenderers.at(rendererIdx + 1)->getIsOpaqueRenderer()) {
                                    volumeData->renderViewCalculatorPostOpaque(viewIdx);
                                    for (auto& volumeRendererPostOpaque : volumeRenderers) {
                                        volumeRendererPostOpaque->renderViewPostOpaque(viewIdx);
                                    }
                                }
                                rendererIdx++;
                            }
                        } else {
                            for (auto& volumeRenderer : volumeRenderers) {
                                if (volumeRenderer->getShallRenderWithoutData()) {
                                    volumeRenderer->renderView(viewIdx);
                                }
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

                        if (i == first3dViewIdx && showFpsOverlay) {
                            renderGuiFpsOverlay();
                        }
                        if (i == first3dViewIdx && showCoordinateAxesOverlay) {
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
                    viewManager->removeView(i);
                    dataViews.erase(dataViews.begin() + i);
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

    if (volumeData) {
        volumeData->renderGuiWindowSecondary();
    }
}

void MainApp::loadAvailableDataSetInformation() {
    const std::string volumeDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/";
    bool datasetsJsonExists = sgl::FileUtils::get()->exists(volumeDataSetsDirectory + "datasets.json");
    DataSetInformationPtr dataSetInformationRootNew;
    if (datasetsJsonExists) {
        dataSetInformationRootNew = loadDataSetList(volumeDataSetsDirectory + "datasets.json", isFileWatchReload);
        if (!dataSetInformationRootNew && !isProgramStart) {
            return;
        }
    } else if (!isProgramStart) {
        return;
    }

    if (!isProgramStart && currentlyLoadedDataSetIndex >= NUM_MANUAL_LOADERS) {
        customDataSetFileNames = dataSetInformationList.at(currentlyLoadedDataSetIndex - NUM_MANUAL_LOADERS)->filenames;
        selectedDataSetIndex = 0;
        currentlyLoadedDataSetIndex = 0;
    }

    dataSetInformationRoot = {};
    dataSetInformationList.clear();
    dataSetNames.clear();
    dataSetNames.emplace_back("Local file...");
    selectedDataSetIndex = 0;

    dataSetInformationRoot = dataSetInformationRootNew;
    if (!datasetsJsonExists || !dataSetInformationRoot) {
        return;
    }

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

std::vector<std::string> MainApp::getSelectedDataSetFilenames() {
    std::vector<std::string> filenames;
    if (selectedDataSetIndex == 0) {
        if (customDataSetFileNames.empty()) {
            filenames.push_back(customDataSetFileName);
        } else {
            filenames = customDataSetFileNames;
        }
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
        for (auto& volumeRenderer : volumeRenderers) {
            volumeRenderer->setClearColor(clearColor);
        }
        reRender = true;
    }

    // TODO: Remove option?
    /*newDockSpaceMode = useDockSpaceMode;
    if (propertyEditor.addCheckbox("Use Docking Mode", &newDockSpaceMode)) {
        scheduledDockSpaceModeChange = true;
    }*/

    if (useDockSpaceMode && propertyEditor.addSliderInt("Supersampling Factor", &supersamplingFactor, 1, 4)) {
        for (DataViewPtr& dataView : dataViews) {
            dataView->supersamplingFactor = supersamplingFactor;
            dataView->viewportWidth = 0;
            dataView->viewportHeight = 0;
        }
    }

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
    dataView->supersamplingFactor = supersamplingFactor;
    dataView->useLinearRGB = useLinearRGB;
    dataView->clearColor = clearColor;
    dataViews.push_back(dataView);
    viewManager->addView(dataView.get(), &dataView->sceneData);
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
    // Debug Code.
    //addNewRenderer(RENDERING_MODE_DIAGRAM_RENDERER);
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
            ".*,.vtk,.vti,.vts,.vtr,.nc,.hdf5,.zarr,.am,.bin,.field,.cvol,.grib,.grb,.dat,.raw,.mhd,.ctl",
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

void MainApp::openSelectStateDialog() {
    if (stateFileDirectory.empty() || !sgl::FileUtils::get()->directoryExists(exportFieldFileDialogDirectory)) {
        stateFileDirectory = sgl::AppSettings::get()->getDataDirectory() + "States/";
        if (!sgl::FileUtils::get()->exists(stateFileDirectory)) {
            sgl::FileUtils::get()->ensureDirectoryExists(stateFileDirectory);
        }
    }
    IGFD_OpenModal(
            fileDialogInstance, "ChooseStateFile", "Choose a File", ".json", stateFileDirectory.c_str(), "", 1, nullptr,
            stateModeSave ? ImGuiFileDialogFlags_ConfirmOverwrite : ImGuiFileDialogFlags_None);
}

void MainApp::renderGuiMenuBar() {
    bool openExportFieldDialog = false;
    bool openFieldSimilarityDialog = false;
    bool openFieldSimilarityResultDialog = false;
    bool openOptimizeTFDialog = false;
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

            if (ImGui::MenuItem("Reload Datasets")) {
                loadAvailableDataSetInformation();
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
                //if (dataViews.size() == 1) {
                //    initializeFirstDataView();
                //}
            }
            if (volumeData && ImGui::BeginMenu("New Calculator...")) {
                volumeData->renderGuiNewCalculators();
                ImGui::EndMenu();
            }
            ImGui::Separator();
            for (int i = 0; i < int(dataViews.size()); i++) {
                DataViewPtr& dataView = dataViews.at(i);
                std::string windowName = dataView->getWindowNameImGui(dataViews, i);
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
            if (volumeData) {
                auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
                if (ImGui::MenuItem("Transfer Function Window", nullptr, tfWindow.getShowWindow())) {
                    tfWindow.setShowWindow(!tfWindow.getShowWindow());
                }
                if (ImGui::MenuItem(
                        "Color Legend Widget Background", nullptr, sgl::ColorLegendWidget::getShowBackground())) {
                    sgl::ColorLegendWidget::setShowBackground(!sgl::ColorLegendWidget::getShowBackground());
                }
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
            if (volumeData && ImGui::MenuItem("Compute Field Similarity...")) {
                openFieldSimilarityDialog = true;
            }
            if (volumeData && ImGui::MenuItem("Optimize Transfer Function...")) {
                openOptimizeTFDialog = true;
            }

            if (ImGui::MenuItem("Load State...")) {
                stateModeSave = false;
                openSelectStateDialog();
            }
            if (ImGui::MenuItem("Save State...")) {
                stateModeSave = true;
                openSelectStateDialog();
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
        selectedFieldIndexExport = std::min(selectedFieldIndexExport, int(fieldNames.size()) - 1);
        ImGui::Combo(
                "Field Name", &selectedFieldIndexExport,
                fieldNames.data(), int(fieldNames.size()));
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

    if (openFieldSimilarityDialog) {
        ImGui::OpenPopup("Compute Field Similarity");
    }
    if (ImGui::BeginPopupModal("Compute Field Similarity", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        similarityFieldIdx0 = std::min(similarityFieldIdx0, int(fieldNames.size()) - 1);
        similarityFieldIdx1 = std::min(similarityFieldIdx1, int(fieldNames.size()) - 1);
        ImGui::Combo(
                "Field #1", &similarityFieldIdx0,
                fieldNames.data(), int(fieldNames.size()));
        ImGui::Combo(
                "Field #2", &similarityFieldIdx1,
                fieldNames.data(), int(fieldNames.size()));
        ImGui::Combo(
                "Correlation Measure", (int*)&correlationMeasureFieldSimilarity,
                CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES));
        ImGui::Combo("Accuracy", (int*)&useFieldAccuracyDouble, FIELD_ACCURACY_NAMES, 2);
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            openFieldSimilarityResultDialog = true;
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    if (openFieldSimilarityResultDialog) {
        ImGui::OpenPopup("Field Similarity Result");
        if (useFieldAccuracyDouble) {
            similarityMetricNumber = computeFieldSimilarity<double>(
                    volumeData.get(), similarityFieldIdx0, similarityFieldIdx1, correlationMeasureFieldSimilarity,
                    maxCorrelationValue, false, false);
        } else {
            similarityMetricNumber = computeFieldSimilarity<float>(
                    volumeData.get(), similarityFieldIdx0, similarityFieldIdx1, correlationMeasureFieldSimilarity,
                    maxCorrelationValue, false, false);
        }
    }
    if (ImGui::BeginPopupModal("Field Similarity Result", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        std::string resultText0 =
                std::string("Similarity Measure: ")
                + CORRELATION_MEASURE_TYPE_NAMES[int(correlationMeasureFieldSimilarity)];
        std::string resultText1 =
                "Fields: " + sgl::toString(fieldNames.at(similarityFieldIdx0)) + ", "
                + sgl::toString(fieldNames.at(similarityFieldIdx1));
        std::string resultText2 = "Value: " + sgl::toString(similarityMetricNumber);
        if (correlationMeasureFieldSimilarity == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
            resultText2 += " (max: " + sgl::toString(maxCorrelationValue) + ")";
        }
        ImGui::TextUnformatted(resultText0.c_str());
        ImGui::TextUnformatted(resultText1.c_str());
        ImGui::TextUnformatted(resultText2.c_str());
        if (ImGui::Button("Close", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::EndPopup();
    }

    if (openOptimizeTFDialog) {
        tfOptimization->openDialog();
    }
    tfOptimization->renderGuiDialog();
    if (tfOptimization->getNeedsReRender()) {
        reRender = true;
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
            bool beginNode = propertyEditor.beginNode(dataView->getWindowNameImGui(dataViews, i));
            if (beginNode) {
                if (propertyEditor.addInputAction("View Name", &dataView->getViewName())) {
                    dataView->reRender = true;
                    for (auto& volumeRenderer : volumeRenderers) {
                        volumeRenderer->notifyReRenderTriggeredExternally();
                    }
                }
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

#ifdef __linux__
    datasetsWatch.update([this] {
        isFileWatchReload = true;
        this->loadAvailableDataSetInformation();
        isFileWatchReload = false;
    });
#endif

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

#ifdef SUPPORT_RENDERDOC_DEBUGGER
    renderDocDebugger.update();
#endif

    updateCameraFlight(volumeData.get() != nullptr, usesNewState);

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

    // Update in inverse order due to back-to-front composited rendering.
    bool hasGrabbedMouse = io.WantCaptureMouse && mouseHoverWindowIndex < 0;
    for (auto it = volumeRenderers.rbegin(); it != volumeRenderers.rend(); it++) {
        auto& volumeRenderer = *it;
        volumeRenderer->update(dt, hasGrabbedMouse);
        hasGrabbedMouse = hasGrabbedMouse || volumeRenderer->getHasGrabbedMouse();
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
                if (!hasGrabbedMouse) {
                    moveCameraMouse(dt);
                }
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
            if (!hasGrabbedMouse) {
                moveCameraMouse(dt);
            }
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

bool MainApp::checkHasValidExtension(const std::string& filenameLower) {
    if (sgl::endsWith(filenameLower, ".vtk")
            || sgl::endsWith(filenameLower, ".vti")
            || sgl::endsWith(filenameLower, ".vts")
            || sgl::endsWith(filenameLower, ".vtr")
            || sgl::endsWith(filenameLower, ".nc")
#ifdef USE_HDF5
            || sgl::endsWith(filenameLower, ".hdf5")
#endif
            || sgl::endsWith(filenameLower, ".zarr")
            || sgl::endsWith(filenameLower, ".am")
            || sgl::endsWith(filenameLower, ".bin")
            || sgl::endsWith(filenameLower, ".field")
            || sgl::endsWith(filenameLower, ".cvol")
            || sgl::endsWith(filenameLower, ".nii")
#ifdef USE_ECCODES
            || sgl::endsWith(filenameLower, ".grib")
            || sgl::endsWith(filenameLower, ".grb")
#endif
            || sgl::endsWith(filenameLower, ".dat")
            || sgl::endsWith(filenameLower, ".raw")
            || sgl::endsWith(filenameLower, ".mhd")
            || sgl::endsWith(filenameLower, ".ctl")) {
        return true;
    }
    return false;
}

void MainApp::onFileDropped(const std::string& droppedFileName) {
    std::string filenameLower = sgl::toLowerCopy(droppedFileName);
    if (checkHasValidExtension(filenameLower)) {
        device->waitIdle();
        fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(droppedFileName);
        selectedDataSetIndex = 0;
        customDataSetFileName = droppedFileName;
        customDataSetFileNames.clear();
        dataSetType = DataSetType::VOLUME;
        loadVolumeDataSet(getSelectedDataSetFilenames());
    } else {
        sgl::Logfile::get()->writeError(
                "The dropped file name has an unknown extension \""
                + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
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
        volumeData->setPrepareVisualizationPipelineCallback([this]() { this->prepareVisualizationPipeline(); });
        //lineData->onMainThreadDataInit();
        volumeData->recomputeHistogram();
        volumeData->setClearColor(clearColor);
        volumeData->setUseLinearRGB(useLinearRGB);
        for (size_t viewIdx = 0; viewIdx < dataViews.size(); viewIdx++) {
            volumeData->addView(uint32_t(viewIdx));
            auto& viewSceneData = dataViews.at(viewIdx)->sceneData;
            if (*viewSceneData.sceneTexture) {
                volumeData->recreateSwapchainView(
                        uint32_t(viewIdx), *viewSceneData.viewportWidthVirtual, *viewSceneData.viewportHeightVirtual);
            }
        }
        newDataLoaded = true;
        reRender = true;
        boundingBox = volumeData->getBoundingBoxRendering();
        selectedFieldIndexExport = 0;
        similarityFieldIdx0 = 0;
        similarityFieldIdx1 = 0;

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
    if (volumeData && volumeData->isDirty()) {
        tfOptimization->setVolumeData(volumeData.get(), newDataLoaded);
    }
    if (volumeData && !volumeRenderers.empty()) {
        bool isPreviousNodeDirty = volumeData->isDirty();
        volumeData->resetDirty();
        for (auto& volumeRenderer : volumeRenderers) {
            if (volumeRenderer->isDirty() || isPreviousNodeDirty) {
                rendererVk->getDevice()->waitIdle();
                volumeRenderer->setVolumeData(volumeData, newDataLoaded);
                volumeRenderer->resetDirty();
            }
        }

        // For rendering restriction enforced by calculators.
        bool reloadRendererShaders = volumeData->getShallReloadRendererShaders();
        if (reloadRendererShaders) {
            rendererVk->getDevice()->waitIdle();
            reRender = true;
            for (auto& volumeRenderer : volumeRenderers) {
                volumeRenderer->reloadShaders();
            }
        }
    }
    newDataLoaded = false;
}
