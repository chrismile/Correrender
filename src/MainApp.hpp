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

#ifndef CORRERENDER_MAINAPP_HPP
#define CORRERENDER_MAINAPP_HPP

#include <string>
#include <vector>
#include <map>

#include <Utils/SciVis/SciVisApp.hpp>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

#ifdef SUPPORT_RENDERDOC_DEBUGGER
#include "Utils/RenderDocDebugger.hpp"
#endif

#include "Utils/InternalState.hpp"
#include "Loaders/DataSetList.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Renderers/SceneData.hpp"

#ifdef USE_PYTHON
//#include "Widgets/ReplayWidget.hpp"
#endif

namespace sgl { namespace dialog {
class MsgBoxHandle;
typedef std::shared_ptr<MsgBoxHandle> MsgBoxHandlePtr;
}}

namespace IGFD {
class FileDialog;
}
typedef IGFD::FileDialog ImGuiFileDialog;

namespace Json {
class Value;
}

class Renderer;
typedef std::shared_ptr<Renderer> RendererPtr;
class VolumeData;
typedef std::shared_ptr<VolumeData> VolumeDataPtr;
class DataView;
typedef std::shared_ptr<DataView> DataViewPtr;
class ViewManager;
class TFOptimization;

class MainApp : public sgl::SciVisApp {
public:
    MainApp();
    ~MainApp() override;
    void render() override;
    void update(float dt) override;
    void resolutionChanged(sgl::EventPtr event) override;

    /// For changing performance measurement modes.
    void setNewState(const InternalState& newState);

    /// Replicability stamp mode.
    void setUseReplicabilityStampMode();

protected:
    void renderGuiGeneralSettingsPropertyEditor() override;
    void beginFrameMarker() override;
    void endFrameMarker() override;

private:
    /// Renders the GUI of the scene settings and all filters and renderers.
    void renderGui() override;
    /// Update the color space (linear RGB vs. sRGB).
    void updateColorSpaceMode() override;
    /// Called when the camera moved.
    void hasMoved() override;
    /// Callback when the camera was reset.
    void onCameraReset() override;

    void scheduleRecreateSceneFramebuffer();
    bool scheduledRecreateSceneFramebuffer = false;
    bool componentOtherThanRendererNeedsReRender = false;

    // Dock space mode.
    void renderGuiMenuBar();
    void renderGuiPropertyEditorBegin() override;
    void renderGuiPropertyEditorCustomNodes() override;
    void addNewDataView();
    void initializeFirstDataView();
    bool scheduledDockSpaceModeChange = false;
    bool newDockSpaceMode = false;
    int focusedWindowIndex = -1;
    int mouseHoverWindowIndex = -1;
    std::vector<DataViewPtr> dataViews;
    int hasMovedIndex = -1;
    bool isFirstFrame = true;

    /// Scene data (e.g., camera, main framebuffer, ...).
    int32_t viewportPositionX = 0;
    int32_t viewportPositionY = 0;
    uint32_t viewportWidth = 0;
    uint32_t viewportHeight = 0;
    int supersamplingFactor = 1;
    SceneData sceneData;

    // This setting lets all data views use the same viewport resolution.
    bool useFixedSizeViewport = false;
    glm::ivec2 fixedViewportSizeEdit{ 2186, 1358 };
    glm::ivec2 fixedViewportSize{ 2186, 1358 };

    // Data set GUI information.
    void loadAvailableDataSetInformation();
    std::vector<std::string> getSelectedDataSetFilenames();
    void openFileDialog();
    DataSetInformationPtr dataSetInformationRoot;
    std::vector<DataSetInformationPtr> dataSetInformationList; //< List of all leaves.
    std::vector<std::string> dataSetNames; //< Contains "Local file..." at beginning, thus starts actually at 1.
    int selectedDataSetIndex = 0; //< Contains "Local file..." at beginning, thus starts actually at 1.
    int currentlyLoadedDataSetIndex = -1;
    std::string customDataSetFileName;
    ImGuiFileDialog* fileDialogInstance = nullptr;
    std::string fileDialogDirectory;
    std::vector<sgl::dialog::MsgBoxHandlePtr> nonBlockingMsgBoxHandles;
    // For volume export dialog.
    void openExportFieldFileDialog();
    int selectedFieldIndexExport = 0;
    std::string exportFieldFileDialogDirectory;
    // For loading and saving the application state.
    void openSelectStateDialog();
    void saveStateToFile(const std::string& stateFilePath);
    void loadStateFromFile(const std::string& stateFilePath);
    void loadStateFromJsonObject(Json::Value root);
    void loadReplicabilityStampState();
    bool useReplicabilityStampMode = false;
    int replicabilityFrameNumber = 0;
    bool stateModeSave = false;
    std::string stateFileDirectory;
    // For field similarity computation.
    CorrelationMeasureType correlationMeasureFieldSimilarity = CorrelationMeasureType::PEARSON;
    int useFieldAccuracyDouble = 1;
    int similarityFieldIdx0 = 0, similarityFieldIdx1 = 0;
    float similarityMetricNumber = 0.0f;
    float maxCorrelationValue = 0.0f;
    TFOptimization* tfOptimization = nullptr;

    // For making performance measurements.
    AutomaticPerformanceMeasurer* performanceMeasurer = nullptr;
    InternalState lastState;
    bool firstState = true;
    bool usesNewState = true;

#ifdef USE_PYTHON
    /*ReplayWidget replayWidget;
    bool replayWidgetRunning = false;
    bool realTimeReplayUpdates = false;
    bool updateTransferFunctionRange = false;
    glm::vec2 transferFunctionRange{};*/
#endif

#ifdef SUPPORT_CUDA_INTEROP
    CUcontext cuContext = {};
    CUdevice cuDevice = 0;
#endif
    bool cudaInteropInitialized = false;
    bool nvrtcInitialized = false;
    bool openclInteropInitialized = false;

#ifdef SUPPORT_RENDERDOC_DEBUGGER
    RenderDocDebugger renderDocDebugger;
#endif


    /// --- Visualization pipeline ---

    /// Loads volume data from a file.
    void loadVolumeDataSet(const std::vector<std::string>& fileName);
    /// Reload the currently loaded data set.
    void reloadDataSet() override;
    /// Prepares the visualization pipeline for rendering.
    void prepareVisualizationPipeline();
    /// Sets the used renderers.
    void addNewRenderer(RenderingMode renderingMode);
    void setRenderer(RenderingMode newRenderingMode, RendererPtr& newVolumeRenderer);
    void onUnsupportedRendererSelected(const std::string& warningText, RendererPtr& newVolumeRenderer);

    /// A list of filters that are applied sequentially on the data.
    ViewManager* viewManager = nullptr;
    std::vector<RendererPtr> volumeRenderers;
    size_t rendererCreationCounter = 0;
    VolumeDataPtr volumeData;
    DataSetType dataSetType = DataSetType::NONE;
    bool newDataLoaded = true;
    sgl::AABB3 boundingBox;
    const int NUM_MANUAL_LOADERS = 1;
};

#endif //CORRERENDER_MAINAPP_HPP
