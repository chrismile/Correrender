/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <json/json.h>

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include "Widgets/DataView.hpp"
#include "Widgets/ViewManager.hpp"
#include "Utils/InternalState.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/Correlation.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Volume/VolumeData.hpp"
#include "TimeSeriesLoader.hpp"
#include "TimeSeriesCorrelationChart.hpp"
#include "TimeSeriesCorrelationRenderer.hpp"

TimeSeriesCorrelationRenderer::TimeSeriesCorrelationRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_TIME_SERIES_CORRELATION)], viewManager) {
    std::string presetsJsonFilename =
            sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/timeseries.json";
    if (sgl::FileUtils::get()->exists(presetsJsonFilename)) {
        parsePresetsFile(presetsJsonFilename);
    }

#ifdef SUPPORT_TINY_CUDA_NN
    initializeCuda();
#endif
}

void TimeSeriesCorrelationRenderer::parsePresetsFile(const std::string& filename) {
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

    timeSeriesNames.emplace_back("---");
    timeSeriesDataPaths.emplace_back("");
    timeSeriesModelPaths.emplace_back("");

    DataSetInformationPtr dataSetInformationRoot(new DataSetInformation);
    Json::Value& modelsNode = root["models"];
    for (Json::Value& model : modelsNode) {
        timeSeriesNames.push_back(model["name"].asString());
        timeSeriesDataPaths.push_back(model["datapath"].asString());
        timeSeriesModelPaths.push_back(model.isMember("modelpath") ? model["modelpath"].asString() : "");
    }

    if (timeSeriesNames.size() == 1) {
        timeSeriesNames.clear();
        timeSeriesDataPaths.clear();
        timeSeriesModelPaths.clear();
    }
}

TimeSeriesCorrelationRenderer::~TimeSeriesCorrelationRenderer() {
    parentDiagram = {};

#ifdef SUPPORT_TINY_CUDA_NN
    cleanupCuda();
#endif
}

void TimeSeriesCorrelationRenderer::initialize() {
    Renderer::initialize();

    parentDiagram = std::make_shared<TimeSeriesCorrelationChart>();
    parentDiagram->setRendererVk(renderer);
    parentDiagram->initialize();
    parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
    parentDiagram->setClearColor(viewManager->getClearColor());
    parentDiagram->setColorMap(colorMap);
    parentDiagram->setDiagramSelectionCallback([this](int series, int time) {
        this->sidxRef = std::clamp(series, 0, timeSeriesMetadata.samples - 1);
        diagramDataDirty = true;
        reRender = true;
        reRenderTriggeredByDiagram = true;
    });
}

void TimeSeriesCorrelationRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
}

void TimeSeriesCorrelationRenderer::loadTimeSeriesFromFile(const std::string& filePath) {
    auto* loader = new TimeSeriesLoader;
    if (!loader->open(filePath)) {
        delete loader;
        return;
    }
    timeSeriesMetadata = loader->getMetadata();
    if (timeSeriesMetadata.window > 0) {
        windowLength = timeSeriesMetadata.window;
    } else {
        windowLength = 32;
    }
    sidxRef = 0;
    if (modelFilePath.empty()) {
        timeSeriesData = loader->loadData();
        recomputeCorrelationMatrix();
    }
    loader->close();
    delete loader;
}

#ifndef SUPPORT_TINY_CUDA_NN
void TimeSeriesCorrelationRenderer::loadModelFromFile(const std::string& filePath) {
}
void TimeSeriesCorrelationRenderer::unloadModel() {
}
#endif

void TimeSeriesCorrelationRenderer::updateCorrelationRange() {
    if (getIsModuleLoaded()) {
#ifdef SUPPORT_TINY_CUDA_NN
        parentDiagram->onCorrelationDataRecalculated(
                isMutualInformationData ? CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
                                        : CorrelationMeasureType::PEARSON,
                {minCorrelationValue, maxCorrelationValue}, true);
#endif
    } else {
        parentDiagram->onCorrelationDataRecalculated(
                correlationMeasureType, {minCorrelationValue, maxCorrelationValue}, false);
    }
}

void TimeSeriesCorrelationRenderer::recomputeCorrelationMatrix() {
    if (!timeSeriesData && !getIsModuleLoaded()) {
        return;
    }

    if (timeSeriesMetadata.window <= 0) {
        numWindows = timeSeriesMetadata.time - windowLength + 1;
    } else {
        numWindows = timeSeriesMetadata.time;
    }
    auto cs = windowLength;
    auto cmt = correlationMeasureType;

    if (numWindows != cachedNumWindows) {
        k = std::clamp(sgl::iceil(3 * cs, 100), 1, 100);
        kMax = std::max(sgl::iceil(7 * cs, 100), 20);
        numBins = sgl::iceil(cs, 10);
    }

    float minFieldVal = 0.0f, maxFieldVal = 0.0f;
    if (isMeasureBinnedMI(correlationMeasureType)) {
        minFieldVal = timeSeriesData->getMinValue();
        maxFieldVal = timeSeriesData->getMaxValue();
    }

    if (cachedNumSamples != timeSeriesMetadata.samples || cachedNumWindows != numWindows
            || memoryExported != getIsModuleLoaded()) {
        cachedNumSamples = timeSeriesMetadata.samples;
        cachedNumWindows = numWindows;
        memoryExported = getIsModuleLoaded();
        correlationDataBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), sizeof(float) * cachedNumSamples * cachedNumWindows,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY, true, memoryExported, true);
        correlationDataStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), sizeof(float) * cachedNumSamples * cachedNumWindows,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU);
        parentDiagram->setCorrelationDataBuffer(timeSeriesMetadata.samples, numWindows, correlationDataBuffer);
#ifdef SUPPORT_TINY_CUDA_NN
        if (getIsModuleLoaded()) {
            outputBufferInterop = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(correlationDataBuffer);
            outputImageBufferCu = outputBufferInterop->getCudaDevicePtr();
        }
#endif
    }

#ifdef SUPPORT_TINY_CUDA_NN
    if (getIsModuleLoaded()) {
        recomputeCorrelationMatrixTcnn();
        minCorrelationValue = 0.0f;
        maxCorrelationValue = 1.0f;
        if (isMutualInformationData) {
            maxCorrelationValue = computeMaximumMutualInformationKraskov(k, cs);
        } else if (!calculateAbsoluteValue) {
            minCorrelationValue = -maxCorrelationValue;
        }
        minCorrelationValueGlobal = minCorrelationValue;
        maxCorrelationValueGlobal = maxCorrelationValue;
        parentDiagram->onCorrelationDataRecalculated(
                isMutualInformationData ? CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV : CorrelationMeasureType::PEARSON,
                {minCorrelationValue, maxCorrelationValue}, true);
        reRender = true;
        reRenderTriggeredByDiagram = true;
        return;
    }
#endif

    minCorrelationValue = 0.0f;
    maxCorrelationValue = 1.0f;
    if (isMeasureKraskovMI(cmt)) {
        if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
            maxCorrelationValue = computeMaximumMutualInformationKraskov(k, cs);
        }
    }
    if (correlationMeasureType == CorrelationMeasureType::PEARSON
            || correlationMeasureType == CorrelationMeasureType::SPEARMAN
            || correlationMeasureType == CorrelationMeasureType::KENDALL) {
        minCorrelationValue = -maxCorrelationValue;
    }
    minCorrelationValueGlobal = minCorrelationValue;
    maxCorrelationValueGlobal = maxCorrelationValue;

    auto* buffer = reinterpret_cast<float*>(correlationDataStagingBuffer->mapMemory());

#ifdef TEST_INFERENCE_SPEED
    auto startCompute = std::chrono::system_clock::now();
#endif

    int numGridPoints = timeSeriesMetadata.samples * numWindows;
    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel for shared(numGridPoints, cs, sidxRef, numWindows, timeSeriesData, buffer) default(none)
#endif
        for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
            if (cs == 1) {
                buffer[gridPointIdx] = 1.0f;
                continue;
            }
            int sidx = gridPointIdx / numWindows;
            int widx = gridPointIdx % numWindows;
            auto* dataRef = timeSeriesData->getWindowData(sidxRef, widx, windowLength);
            auto* dataQuery = timeSeriesData->getWindowData(sidx, widx, windowLength);
            float pearsonCorrelation = computePearson2<float>(dataRef, dataQuery, cs);
            buffer[gridPointIdx] = pearsonCorrelation;
        }
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            auto* referenceRanks = new float[cs];
            auto* queryRanks = new float[cs];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(cs);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, sidxRef, numWindows, timeSeriesData, buffer) default(none)
        {
#endif
            auto* referenceRanks = new float[cs];
            auto* queryRanks = new float[cs];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(cs);
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                int sidx = gridPointIdx / numWindows;
                int widx = gridPointIdx % numWindows;
                auto* dataRef = timeSeriesData->getWindowData(sidxRef, widx, windowLength);
                auto* dataQuery = timeSeriesData->getWindowData(sidx, widx, windowLength);

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    if (std::isnan(dataRef[c]) || std::isnan(dataQuery[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                computeRanks(dataRef, referenceRanks, ordinalRankArray, cs);
                computeRanks(dataQuery, queryRanks, ordinalRankArray, cs);

                float pearsonCorrelation = computePearson2<float>(referenceRanks, queryRanks, cs);
                buffer[gridPointIdx] = pearsonCorrelation;
            }
            delete[] referenceRanks;
            delete[] queryRanks;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            std::vector<float> sortArray;
            std::vector<std::pair<int, int>> stack;
            jointArray.reserve(cs);
            ordinalRankArray.reserve(cs);
            y.reserve(cs);
            sortArray.reserve(cs);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, sidxRef, numWindows, timeSeriesData, buffer) default(none)
        {
#endif
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            std::vector<float> sortArray;
            std::vector<std::pair<int, int>> stack;
            jointArray.reserve(cs);
            ordinalRankArray.reserve(cs);
            y.reserve(cs);
            sortArray.reserve(cs);
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                int sidx = gridPointIdx / numWindows;
                int widx = gridPointIdx % numWindows;
                auto* dataRef = timeSeriesData->getWindowData(sidxRef, widx, windowLength);
                auto* dataQuery = timeSeriesData->getWindowData(sidx, widx, windowLength);

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    if (std::isnan(dataRef[c]) || std::isnan(dataQuery[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float pearsonCorrelation = computeKendall<int32_t>(
                        dataRef, dataQuery, cs, jointArray, ordinalRankArray, y, sortArray, stack);
                buffer[gridPointIdx] = pearsonCorrelation;
            }
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (isMeasureBinnedMI(correlationMeasureType)) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            auto* refValues = new float[cs];
            auto* queryValues = new float[cs];
            auto* histogram0 = new double[numBins];
            auto* histogram1 = new double[numBins];
            auto* histogram2d = new double[numBins * numBins];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, sidxRef, numWindows, timeSeriesData, buffer) default(none) \
        shared(numBins, minFieldVal, maxFieldVal)
        {
#endif
            auto* refValues = new float[cs];
            auto* queryValues = new float[cs];
            auto* histogram0 = new double[numBins];
            auto* histogram1 = new double[numBins];
            auto* histogram2d = new double[numBins * numBins];
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                int sidx = gridPointIdx / numWindows;
                int widx = gridPointIdx % numWindows;
                auto* dataRef = timeSeriesData->getWindowData(sidxRef, widx, windowLength);
                auto* dataQuery = timeSeriesData->getWindowData(sidx, widx, windowLength);

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    if (std::isnan(dataRef[c]) || std::isnan(dataQuery[c])) {
                        isNan = true;
                        break;
                    }
                    refValues[c] =
                            (dataRef[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                    queryValues[c] =
                            (dataQuery[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation = computeMutualInformationBinned<double>(
                        refValues, queryValues, numBins, cs, histogram0, histogram1, histogram2d);
                if (correlationMeasureType == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT) {
                    mutualInformation = std::sqrt(1.0f - std::exp(-2.0f * mutualInformation));
                }
                buffer[gridPointIdx] = mutualInformation;
            }
            delete[] refValues;
            delete[] queryValues;
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (isMeasureKraskovMI(correlationMeasureType)) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[cs];
            KraskovEstimatorCache<double> kraskovEstimatorCache;
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, k, sidxRef, numWindows, timeSeriesData, buffer) default(none)
        {
#endif
            KraskovEstimatorCache<double> kraskovEstimatorCache;
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                int sidx = gridPointIdx / numWindows;
                int widx = gridPointIdx % numWindows;
                auto* dataRef = timeSeriesData->getWindowData(sidxRef, widx, windowLength);
                auto* dataQuery = timeSeriesData->getWindowData(sidx, widx, windowLength);

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    if (std::isnan(dataRef[c]) || std::isnan(dataQuery[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation = computeMutualInformationKraskov<double>(
                        dataRef, dataQuery, k, cs, kraskovEstimatorCache);
                if (correlationMeasureType == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT) {
                    mutualInformation = std::sqrt(1.0f - std::exp(-2.0f * mutualInformation));
                }
                buffer[gridPointIdx] = mutualInformation;
            }
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    }

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endCompute = std::chrono::system_clock::now();
    auto elapsedCompute = std::chrono::duration_cast<std::chrono::milliseconds>(endCompute - startCompute);
    std::cout << "Elapsed time compute: " << elapsedCompute.count() << "ms" << std::endl;
#endif

    correlationDataStagingBuffer->unmapMemory();
    correlationDataStagingBuffer->copyDataTo(correlationDataBuffer, renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            correlationDataBuffer);

    parentDiagram->onCorrelationDataRecalculated(
            correlationMeasureType, {minCorrelationValue, maxCorrelationValue}, false);
    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void TimeSeriesCorrelationRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    if (viewIdx == diagramViewIdx) {
        recreateDiagramSwapchain();
    }
}

void TimeSeriesCorrelationRenderer::recreateDiagramSwapchain(int diagramIdx) {
    SceneData* sceneData = viewManager->getViewSceneData(diagramViewIdx);
    if (!(*sceneData->sceneTexture)) {
        return;
    }
    parentDiagram->setBlitTargetVk(
            (*sceneData->sceneTexture)->getImageView(),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (alignWithParentWindow) {
        parentDiagram->setBlitTargetSupersamplingFactor(
                viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
        parentDiagram->updateSizeByParent();
    }
    reRenderTriggeredByDiagram = true;
}

void TimeSeriesCorrelationRenderer::update(float dt, bool isMouseGrabbed) {
    reRenderTriggeredByDiagram = false;
    parentDiagram->setIsMouseGrabbedByParent(isMouseGrabbed);
    parentDiagram->update(dt);
    if (parentDiagram->getNeedsReRender() && !reRenderViewArray.empty()) {
        for (int viewIdx = 0; viewIdx < int(viewVisibilityArray.size()); viewIdx++) {
            if (viewVisibilityArray.at(viewIdx)) {
                reRenderViewArray.at(viewIdx) = true;
            }
        }
        reRenderTriggeredByDiagram = true;
    }
    //isMouseGrabbed |= parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui();
}

void TimeSeriesCorrelationRenderer::onHasMoved(uint32_t viewIdx) {
}

bool TimeSeriesCorrelationRenderer::getHasGrabbedMouse() const {
    if (parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui()) {
        return true;
    }
    return false;
}

void TimeSeriesCorrelationRenderer::setClearColor(const sgl::Color& clearColor) {
    parentDiagram->setClearColor(clearColor);
    reRenderTriggeredByDiagram = true;
}

void TimeSeriesCorrelationRenderer::renderViewImpl(uint32_t viewIdx) {
    if (viewIdx != diagramViewIdx) {
        return;
    }

    SceneData* sceneData = viewManager->getViewSceneData(viewIdx);
    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);

    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        parentDiagram->setImGuiWindowOffset(int(pos.x), int(pos.y));
    } else {
        parentDiagram->setImGuiWindowOffset(0, 0);
    }
    if (reRenderTriggeredByDiagram || parentDiagram->getIsFirstRender()) {
        if (diagramDataDirty) {
            recomputeCorrelationMatrix();
            diagramDataDirty = false;
        }
        parentDiagram->renderPrepare();
        parentDiagram->render();
    }
    parentDiagram->setBlitTargetSupersamplingFactor(viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
    parentDiagram->blitToTargetVk();
}

void TimeSeriesCorrelationRenderer::renderViewPreImpl(uint32_t viewIdx) {
    if ((viewIdx == diagramViewIdx) && alignWithParentWindow) {
        return;
    }
}

void TimeSeriesCorrelationRenderer::renderViewPostOpaqueImpl(uint32_t viewIdx) {
    if (viewIdx == diagramViewIdx && alignWithParentWindow) {
        return;
    }
}

void TimeSeriesCorrelationRenderer::addViewImpl(uint32_t viewIdx) {
}

bool TimeSeriesCorrelationRenderer::adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx) {
    if (diagramViewIdx >= viewIdx && diagramViewIdx != 0) {
        if (diagramViewIdx != 0) {
            diagramViewIdx--;
            return true;
        } else if (viewManager->getNumViews() > 0) {
            diagramViewIdx++;
            return true;
        }
    }
    return false;
}

void TimeSeriesCorrelationRenderer::removeViewImpl(uint32_t viewIdx) {
    bool diagramViewIdxChanged = false;
    diagramViewIdxChanged |= adaptIdxOnViewRemove(viewIdx, diagramViewIdx);
    if (diagramViewIdxChanged) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
        recreateDiagramSwapchain();
    }
}

void TimeSeriesCorrelationRenderer::renderDiagramViewSelectionGui(
        sgl::PropertyEditor& propertyEditor, const std::string& name, uint32_t& diagramViewIdx) {
    std::string textDefault = "View " + std::to_string(diagramViewIdx + 1);
    if (propertyEditor.addBeginCombo(name, textDefault, ImGuiComboFlags_NoArrowButton)) {
        for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
            std::string text = "View " + std::to_string(viewIdx + 1);
            bool showInView = diagramViewIdx == uint32_t(viewIdx);
            if (ImGui::Selectable(
                    text.c_str(), &showInView, ImGuiSelectableFlags_::ImGuiSelectableFlags_None)) {
                diagramViewIdx = uint32_t(viewIdx);
                reRender = true;
                reRenderTriggeredByDiagram = true;
                recreateDiagramSwapchain();
            }
            if (showInView) {
                ImGui::SetItemDefaultFocus();
            }
        }
        propertyEditor.addEndCombo();
    }
}

void TimeSeriesCorrelationRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    renderDiagramViewSelectionGui(propertyEditor, "Diagram View", diagramViewIdx);

    if (!timeSeriesNames.empty() && propertyEditor.addCombo(
            "Time Series Data", &presetIndex, timeSeriesNames.data(), int(timeSeriesNames.size()))) {
        if (presetIndex != 0) {
            timeSeriesFilePath = timeSeriesDataPaths.at(presetIndex);
            modelFilePath = timeSeriesModelPaths.at(presetIndex);
            unloadModel();
            loadTimeSeriesFromFile(timeSeriesFilePath);
            loadModelFromFile(modelFilePath);
            dirty = true;
        }
    }

    propertyEditor.addInputAction("Time Series Path", &timeSeriesFilePath);
    propertyEditor.addInputAction("Model Path", &modelFilePath);
    if (propertyEditor.addButton("##load-model-label", "Load")) {
        unloadModel();
        loadTimeSeriesFromFile(timeSeriesFilePath);
        loadModelFromFile(modelFilePath);
        dirty = true;
    }

    if (timeSeriesMetadata.window <=  0) {
        if (propertyEditor.addSliderInt("Sliding Window Length", &windowLength, 2, 1024)) {
            recomputeCorrelationMatrix();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
    }

    if (!getIsModuleLoaded() && propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        recomputeCorrelationMatrix();
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
    if (!getIsModuleLoaded()) {
        if (isMeasureBinnedMI(correlationMeasureType) && propertyEditor.addSliderIntEdit(
                "#Bins", &numBins, 10, 100) == ImGui::EditMode::INPUT_FINISHED) {
            recomputeCorrelationMatrix();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
        if (isMeasureKraskovMI(correlationMeasureType) && propertyEditor.addSliderIntEdit(
                "#Neighbors", &k, 1, kMax) == ImGui::EditMode::INPUT_FINISHED) {
            recomputeCorrelationMatrix();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
    }

    if (propertyEditor.addCombo(
            "Color Map", (int*)&colorMap, DIAGRAM_COLOR_MAP_NAMES,
            IM_ARRAYSIZE(DIAGRAM_COLOR_MAP_NAMES))) {
        parentDiagram->setColorMap(colorMap);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    updateCorrelationRange();
    glm::vec2 correlationMinMax(minCorrelationValue, maxCorrelationValue);
    if (propertyEditor.addSliderFloat2(
            "Value Range", (float*)&correlationMinMax.x, minCorrelationValueGlobal, maxCorrelationValueGlobal)) {
        minCorrelationValue = correlationMinMax.x;
        maxCorrelationValue = correlationMinMax.y;
        updateCorrelationRange();
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
    if (propertyEditor.addButton("##reset-range-label", "Reset Range")) {
        minCorrelationValue = minCorrelationValueGlobal;
        maxCorrelationValue = maxCorrelationValueGlobal;
        updateCorrelationRange();
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCheckbox("Align with Window", &alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

#ifdef SUPPORT_TINY_CUDA_NN
    if (propertyEditor.beginNode("Advanced Settings")) {
        if (deviceSupporsFullyFusedMlp && propertyEditor.addCombo(
                "Network", (int*)&networkImplementation,
                TINY_CUDA_NN_NETWORK_IMPLEMENTATION_UI_NAMES, IM_ARRAYSIZE(TINY_CUDA_NN_NETWORK_IMPLEMENTATION_UI_NAMES))) {
            if (sgl::FileUtils::get()->exists(modelFilePath) && !sgl::FileUtils::get()->isDirectory(modelFilePath)) {
                loadModelFromFile(modelFilePath);
            }
            dirty = true;
        }
        if (!isMutualInformationData && propertyEditor.addCheckbox("Absolute Value", &calculateAbsoluteValue)) {
            dirty = true;
        }
        propertyEditor.endNode();
    }
#endif

    if (parentDiagram->renderGuiPropertyEditor(propertyEditor)) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
}

void TimeSeriesCorrelationRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);

    bool diagramChanged = false;
    diagramChanged |= settings.getValueOpt("diagram_view", diagramViewIdx);
    if (diagramChanged) {
        recreateDiagramSwapchain();
    }
    if (settings.getValueOpt("align_with_parent_window", alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
    }

    if (settings.getValueOpt("time_series_file_path", timeSeriesFilePath)) {
        loadTimeSeriesFromFile(timeSeriesFilePath);
        dirty = true;
    }
    if (settings.getValueOpt("model_file_path", modelFilePath)) {
        loadModelFromFile(modelFilePath);
        dirty = true;
    }
    if (settings.getValueOpt("preset_index", presetIndex)) {
        presetIndex = std::clamp(presetIndex, 0, int(timeSeriesDataPaths.size()) - 1);
        timeSeriesFilePath = timeSeriesDataPaths.at(presetIndex);
        modelFilePath = timeSeriesModelPaths.at(presetIndex);
        dirty = true;
    }

    if (settings.getValueOpt("sliding_window_length", windowLength)) {
        recomputeCorrelationMatrix();
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    std::string correlationMeasureTypeName;
    if (settings.getValueOpt("correlation_measure_type", correlationMeasureTypeName)) {
        for (int i = 0; i < IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_IDS); i++) {
            if (correlationMeasureTypeName == CORRELATION_MEASURE_TYPE_IDS[i]) {
                correlationMeasureType = CorrelationMeasureType(i);
                break;
            }
        }
        recomputeCorrelationMatrix();
    }

    std::string colorMapName;
    std::string colorMapIdx = "color_map";
    if (settings.getValueOpt(colorMapIdx.c_str(), colorMapName)) {
        for (int i = 0; i < IM_ARRAYSIZE(DIAGRAM_COLOR_MAP_NAMES); i++) {
            if (colorMapName == DIAGRAM_COLOR_MAP_NAMES[i]) {
                colorMap = DiagramColorMap(i);
                parentDiagram->setColorMap(colorMap);
                break;
            }
        }
    }

    if (settings.getValueOpt("mi_bins", numBins)) {
        dirty = true;
    }
    if (settings.getValueOpt("kmi_neighbors", k)) {
        dirty = true;
    }

#ifdef SUPPORT_TINY_CUDA_NN
    std::string networkImplementationString;
    if (settings.getValueOpt("network_implementation", networkImplementationString)) {
        for (int i = 0; i < IM_ARRAYSIZE(TINY_CUDA_NN_NETWORK_IMPLEMENTATION_NAMES); i++) {
            if (networkImplementationString == TINY_CUDA_NN_NETWORK_IMPLEMENTATION_NAMES[i]) {
                networkImplementation = TinyCudaNNNetworkImplementation(i);
                break;
            }
        }
        if (!deviceSupporsFullyFusedMlp) {
            networkImplementation = TinyCudaNNNetworkImplementation::CUTLASS_MLP;
        }
        dirty = true;
    }
    if (settings.getValueOpt("calculate_absolute_value", calculateAbsoluteValue)) {
        dirty = true;
    }
#endif

    dirty = true;
    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void TimeSeriesCorrelationRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);

    settings.addKeyValue("diagram_view", diagramViewIdx);
    settings.addKeyValue("align_with_parent_window", alignWithParentWindow);

    settings.addKeyValue("time_series_file_path", timeSeriesFilePath);
    settings.addKeyValue("model_file_path", modelFilePath);
    settings.addKeyValue("preset_index", presetIndex);

    settings.addKeyValue("sliding_window_length", windowLength);

    settings.addKeyValue(
            "correlation_measure_type", CORRELATION_MEASURE_TYPE_IDS[int(correlationMeasureType)]);
    settings.addKeyValue("color_map", DIAGRAM_COLOR_MAP_NAMES[int(colorMap)]);
    settings.addKeyValue("mi_bins", numBins);
    settings.addKeyValue("kmi_neighbors", k);

#ifdef SUPPORT_TINY_CUDA_NN
    settings.addKeyValue("calculate_absolute_value", calculateAbsoluteValue);
    settings.addKeyValue(
            "network_implementation", TINY_CUDA_NN_NETWORK_IMPLEMENTATION_NAMES[int(networkImplementation)]);
#endif

    // No vector widget settings for now.
}
