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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/Parallel/Reduction.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/imgui_custom.h>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "bhtsne/tsne.h"

#include "Widgets/DataView.hpp"
#include "Widgets/ViewManager.hpp"
#include "Utils/InternalState.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/Correlation.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Volume/VolumeData.hpp"
#include "../CorrelationCache.hpp"
#include "DistributionSimilarityChart.hpp"
#include "DistributionSimilarityRenderer.hpp"

DistributionSimilarityRenderer::DistributionSimilarityRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_DISTRIBUTION_SIMILARITY)], viewManager) {
}

DistributionSimilarityRenderer::~DistributionSimilarityRenderer() {
    parentDiagram = {};
}

void DistributionSimilarityRenderer::initialize() {
    Renderer::initialize();

    parentDiagram = std::make_shared<DistributionSimilarityChart>();
    parentDiagram->setRendererVk(renderer);
    parentDiagram->initialize();
    if (volumeData) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
        //parentDiagram->setColorMap(colorMap);
    }
}

void DistributionSimilarityRenderer::setRecomputeFlag() {
    dataDirty = true;
    dirty = true;
    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void DistributionSimilarityRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;

    int es = _volumeData->getEnsembleMemberCount();
    int ts = _volumeData->getTimeStepCount();
    if (isEnsembleMode && es <= 1 && ts > 1) {
        isEnsembleMode = false;
    } else if (!isEnsembleMode && ts <= 1 && es > 1) {
        isEnsembleMode = true;
    }

    scalarFieldNames = {};
    scalarFieldIndexArray = {};
    scalarFieldNames.emplace_back("---");
    scalarFieldIndexArray.emplace_back(0xFFFFFF);

    std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
    for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
        scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
        scalarFieldIndexArray.push_back(i);
    }

    if (isNewData) {
        fieldIndex = fieldIndex2 = 0xFFFFFF;
        fieldIndexGui = fieldIndex2Gui = 0;
        //fieldIndex = fieldIndex2 = volumeData->getStandardScalarFieldIdx();
        //fieldIndexGui = fieldIndex2Gui = volumeData->getStandardScalarFieldIdx();
        //volumeData->acquireScalarField(this, fieldIndex);
        //if (useSeparateFields) {
        //    volumeData->acquireScalarField(this, fieldIndex2);
        //}

        if (!getSupportsBufferMode() || volumeData->getGridSizeZ() < 4) {
            dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        } else {
            dataMode = CorrelationDataMode::BUFFER_ARRAY;
        }
    }

    if (fieldIndexGui + 1 >= int(volumeData->getFieldNamesBase(FieldType::SCALAR).size())) {
        dataDirty = true;
    }
    if (useSeparateFields && fieldIndex2Gui + 1 >= int(volumeData->getFieldNamesBase(FieldType::SCALAR).size())) {
        dataDirty = true;
    }
    if (dataDirty || isNewData) {
        recomputeCorrelationMatrix();
    }

    if (isNewData) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
        //parentDiagram->setColorMap(colorMap);
    }
}

bool DistributionSimilarityRenderer::getSupportsBufferMode() {
    bool supportsBufferMode = true;
    if (!volumeData->getScalarFieldSupportsBufferMode(fieldIndex)) {
        supportsBufferMode = false;
    }
    if (useSeparateFields && !volumeData->getScalarFieldSupportsBufferMode(fieldIndex2)) {
        supportsBufferMode = false;
    }
    if (!supportsBufferMode && dataMode == CorrelationDataMode::BUFFER_ARRAY) {
        dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        setRecomputeFlag();
    }
    return supportsBufferMode;
}

void DistributionSimilarityRenderer::onCorrelationMemberCountChanged() {
    int cs = getCorrelationMemberCount();
    k = std::max(sgl::iceil(3 * cs, 100), 1);
    kMax = std::max(sgl::iceil(7 * cs, 100), 20);
    /*correlationComputePass->setCorrelationMemberCount(cs);
    correlationComputePass->setKraskovNumNeighbors(k);
    cachedMemberCount = cs;

#ifdef SUPPORT_CUDA_INTEROP
    bool canUseCuda =
            sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() && sgl::vk::getIsNvrtcFunctionTableInitialized();
#else
    bool canUseCuda = false;
#endif
    referenceValuesBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), cs * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, true, canUseCuda, false);
#ifdef SUPPORT_CUDA_INTEROP
    if (canUseCuda) {
        referenceValuesCudaBuffer = std::make_shared<sgl::vk::BufferCudaExternalMemoryVk>(referenceValuesBuffer);
    }
#endif
    referenceValuesCpu.resize(cs);*/
}

void DistributionSimilarityRenderer::clearFieldDeviceData() {
//    correlationComputePass->setFieldImageViews({});
//    correlationComputePass->setFieldBuffers({});
//    correlationComputePass->setReferenceValuesBuffer({});
//#ifdef SUPPORT_CUDA_INTEROP
//    correlationComputePass->setReferenceValuesCudaBuffer({});
//#endif
}

void DistributionSimilarityRenderer::recomputeCorrelationMatrix() {
    dataDirty = false;
    if (fieldIndexGui == 0 || (useSeparateFields && fieldIndex2Gui == 0)) {
        parentDiagram->setPointData({});
        return;
    }

    auto xs = volumeData->getGridSizeX();
    auto ys = volumeData->getGridSizeY();
    auto zs = volumeData->getGridSizeZ();
    int cs = getCorrelationMemberCount();

    int timeStepIdxReference = -1;
    int ensembleIdx = -1, timeStepIdx = -1;
    if (useTimeLagCorrelations) {
        timeStepIdxReference = timeLagTimeStepIdx;
    }

    float minFieldValRef = std::numeric_limits<float>::max();
    float maxFieldValRef = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(
                scalarFieldNames.at(useSeparateFields ? fieldIndex2Gui : fieldIndexGui), fieldIdx, timeStepIdxReference, ensembleIdx);
        const float *field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
        if (isMeasureBinnedMI(correlationMeasureType)) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(
                    scalarFieldNames.at(useSeparateFields ? fieldIndex2Gui : fieldIndexGui), fieldIdx, timeStepIdxReference, ensembleIdx);
            minFieldValRef = std::min(minFieldValRef, minVal);
            maxFieldValRef = std::max(maxFieldValRef, maxVal);
        }
    }

    float minFieldValQuery = std::numeric_limits<float>::max();
    float maxFieldValQuery = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::HostCacheEntry> fieldEntries2;
    std::vector<const float*> fields2;
    if (useSeparateFields) {
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            VolumeData::HostCacheEntry fieldEntry2 = getFieldEntryCpu(
                    scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
            const float *field2 = fieldEntry2->data<float>();
            fieldEntries2.push_back(fieldEntry2);
            fields2.push_back(field2);
            if (isMeasureBinnedMI(correlationMeasureType)) {
                auto [minVal2, maxVal2] = getMinMaxScalarFieldValue(
                        scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
                minFieldValQuery = std::min(minFieldValQuery, minVal2);
                maxFieldValQuery = std::max(maxFieldValQuery, maxVal2);
            }
        }
    } else {
        minFieldValQuery = minFieldValRef;
        maxFieldValQuery = maxFieldValRef;
        fieldEntries2 = fieldEntries;
        fields2 = fields;
    }

    int numGridPoints = xs * ys * zs;
    int numVectors = numGridPoints;
    int localRadius = 3;
    int featureDimLen = 2 * localRadius + 1;
    int numFeatures = featureDimLen * featureDimLen * featureDimLen;
    auto* featureVectorArray = new double[numFeatures * numVectors];
    auto* outputPoints = new double[2 * numVectors];
    const CorrelationMeasureType cmt = correlationMeasureType;
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            CORRELATION_CACHE;
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel shared(xs, ys, zs, cs, cmt, numGridPoints, numFeatures, featureDimLen, localRadius) \
    shared(minFieldValRef, maxFieldValRef, minFieldValQuery, maxFieldValQuery, featureVectorArray, fields, fields2) default(none)
#endif
    {
        CORRELATION_CACHE;
        #pragma omp for
        for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
            int xr = gridPointIdx % xs;
            int yr = (gridPointIdx / xs) % ys;
            int zr = gridPointIdx / (xs * ys);
            for (int c = 0; c < cs; c++) {
                X[c] = fields.at(c)[gridPointIdx];
                if (std::isnan(X[c])) {
                    X[c] = 0.0f;
                }
            }
            for (int localIdx = 0; localIdx < numFeatures; localIdx++) {
                int xq = xr + (localIdx % featureDimLen) - localRadius;
                int yq = yr + (localIdx / featureDimLen) % featureDimLen - localRadius;
                int zq = zr + localIdx / (featureDimLen * featureDimLen) - localRadius;
                xq = std::clamp(xq, 0, xs - 1);
                yq = std::clamp(yq, 0, ys - 1);
                zq = std::clamp(zq, 0, zs - 1);
                int idxq = IDXS(xq, yq, zq);
                for (int c = 0; c < cs; c++) {
                    Y[c] = fields2.at(c)[idxq];
                    if (std::isnan(Y[c])) {
                        Y[c] = 0.0f;
                    }
                }

                float correlationValue = 0.0f;
                if (cmt == CorrelationMeasureType::PEARSON) {
                    correlationValue = computePearson2<float>(X.data(), Y.data(), cs);
                } else if (cmt == CorrelationMeasureType::SPEARMAN) {
                    computeRanks(X.data(), referenceRanks.data(), ordinalRankArrayRef, cs);
                    computeRanks(Y.data(), gridPointRanks.data(), ordinalRankArraySpearman, cs);
                    correlationValue = computePearson2<float>(referenceRanks.data(), gridPointRanks.data(), cs);
                } else if (cmt == CorrelationMeasureType::KENDALL) {
                    correlationValue = computeKendall<int32_t>(
                            X.data(), Y.data(), cs, jointArray, ordinalRankArray, y, sortArray, stack);
                } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                    for (int c = 0; c < cs; c++) {
                        X[c] = (X[c] - minFieldValRef) / (maxFieldValRef - minFieldValRef);
                        Y[c] = (Y[c] - minFieldValQuery) / (maxFieldValQuery - minFieldValQuery);
                    }
                    correlationValue = computeMutualInformationBinned<double>(
                            X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                    correlationValue = computeMutualInformationKraskov<double>(
                            X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                } else if (cmt == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT) {
                    for (int c = 0; c < cs; c++) {
                        X[c] = (X[c] - minFieldValRef) / (maxFieldValRef - minFieldValRef);
                        Y[c] = (Y[c] - minFieldValQuery) / (maxFieldValQuery - minFieldValQuery);
                    }
                    correlationValue = computeMutualInformationBinned<double>(
                            X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                    correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
                } else if (cmt == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT) {
                    correlationValue = computeMutualInformationKraskov<double>(
                            X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                    correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
                }
                if (calculateAbsoluteValue) {
                    correlationValue = std::abs(correlationValue);
                }
                if (std::isnan(correlationValue)) {
                    correlationValue = 0.0f;
                }

                size_t offsetWrite = size_t(gridPointIdx) * size_t(numFeatures) + size_t(localIdx);
                featureVectorArray[offsetWrite] = correlationValue;
            }
        }
#ifdef USE_TBB
        });
#else
    }
#endif

    float perplexity = 30.0f;
    float theta = 0.5f;
    int randomSeed = 17; // -1 for pseudo-random
    int maxIter = 500;
    int stopLyingIter = 0;
    int momSwitchIter = 700;
    TSNE::run(
            featureVectorArray, numVectors, numFeatures, outputPoints, 2,
            double(perplexity), double(theta), randomSeed, false, maxIter, stopLyingIter, momSwitchIter);

    std::vector<glm::vec2> points(numVectors);
    for (int ptIdx = 0; ptIdx < numVectors; ptIdx++) {
        points.at(ptIdx) = glm::vec2(float(featureVectorArray[ptIdx * 2]), float(featureVectorArray[ptIdx * 2 + 1]));
    }
    parentDiagram->setPointData(points);
    auto bbData = sgl::reduceVec2ArrayAabb(points);
    parentDiagram->setBoundingBox(bbData);
    delete[] featureVectorArray;
    delete[] outputPoints;

    reRender = true;
    reRenderTriggeredByDiagram = true;
}

int DistributionSimilarityRenderer::getCorrelationMemberCount() {
    return isEnsembleMode ? volumeData->getEnsembleMemberCount() : volumeData->getTimeStepCount();
}

VolumeData::HostCacheEntry DistributionSimilarityRenderer::getFieldEntryCpu(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx) {
    VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx);
    return ensembleEntryField;
}

VolumeData::DeviceCacheEntry DistributionSimilarityRenderer::getFieldEntryDevice(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx, bool wantsImageData) {
    VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx,
            wantsImageData, (!wantsImageData && useBufferTiling) ? glm::uvec3(8, 8, 4) : glm::uvec3(1, 1, 1));
    return ensembleEntryField;
}

std::pair<float, float> DistributionSimilarityRenderer::getMinMaxScalarFieldValue(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx) {
    return volumeData->getMinMaxScalarFieldValue(
            fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx);
}

void DistributionSimilarityRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (fieldIndex == fieldIdx) {
            fieldIndex = 0xFFFFFF;
            //volumeData->acquireScalarField(this, fieldIndex);
            fieldIndexGui = 0;
            setRecomputeFlag();
        } else if (fieldIndex > fieldIdx) {
            fieldIndex--;
            fieldIndexGui = fieldIndex + 1;
        }

        if (fieldIndex2 == fieldIdx) {
            fieldIndex2 = 0xFFFFFF;
            if (useSeparateFields) {
                //volumeData->acquireScalarField(this, fieldIndex2);
                setRecomputeFlag();
            }
            fieldIndex2Gui = 0;
        } else if (fieldIndex2 > fieldIdx) {
            fieldIndex2--;
            fieldIndex2Gui = fieldIndex2 + 1;
        }
    }
}

void DistributionSimilarityRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    if (viewIdx == diagramViewIdx) {
        recreateDiagramSwapchain();
    }
}

void DistributionSimilarityRenderer::recreateDiagramSwapchain(int diagramIdx) {
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

void DistributionSimilarityRenderer::update(float dt, bool isMouseGrabbed) {
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

void DistributionSimilarityRenderer::onHasMoved(uint32_t viewIdx) {
}

bool DistributionSimilarityRenderer::getHasGrabbedMouse() const {
    if (parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui()) {
        return true;
    }
    return false;
}

void DistributionSimilarityRenderer::setClearColor(const sgl::Color& clearColor) {
    parentDiagram->setClearColor(clearColor);
    reRenderTriggeredByDiagram = true;
}

void DistributionSimilarityRenderer::renderViewImpl(uint32_t viewIdx) {
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
        parentDiagram->render();
    }
    parentDiagram->setBlitTargetSupersamplingFactor(viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
    parentDiagram->blitToTargetVk();
}

void DistributionSimilarityRenderer::renderViewPreImpl(uint32_t viewIdx) {
    if ((viewIdx == diagramViewIdx) && alignWithParentWindow) {
        return;
    }
}

void DistributionSimilarityRenderer::renderViewPostOpaqueImpl(uint32_t viewIdx) {
    if (viewIdx == diagramViewIdx && alignWithParentWindow) {
        return;
    }
}

void DistributionSimilarityRenderer::addViewImpl(uint32_t viewIdx) {
}

bool DistributionSimilarityRenderer::adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx) {
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

void DistributionSimilarityRenderer::removeViewImpl(uint32_t viewIdx) {
    bool diagramViewIdxChanged = false;
    diagramViewIdxChanged |= adaptIdxOnViewRemove(viewIdx, diagramViewIdx);
    if (diagramViewIdxChanged) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
        recreateDiagramSwapchain();
    }
}

void DistributionSimilarityRenderer::renderDiagramViewSelectionGui(
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

void DistributionSimilarityRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    renderDiagramViewSelectionGui(propertyEditor, "Diagram View", diagramViewIdx);

    if (volumeData && useSeparateFields) {
        if (propertyEditor.addCombo(
                "Scalar Field Reference", &fieldIndex2Gui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            clearFieldDeviceData();
            if (fieldIndex2 != 0xFFFFFF) {
                volumeData->releaseScalarField(this, fieldIndex2);
            }
            fieldIndex2 = int(scalarFieldIndexArray.at(fieldIndex2Gui));
            if (fieldIndex2 != 0xFFFFFF) {
                volumeData->acquireScalarField(this, fieldIndex2);
            }
            setRecomputeFlag();
        }
        if (propertyEditor.addCombo(
                "Scalar Field Query", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            clearFieldDeviceData();
            if (fieldIndex != 0xFFFFFF) {
                volumeData->releaseScalarField(this, fieldIndex);
            }
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            if (fieldIndex != 0xFFFFFF) {
                volumeData->acquireScalarField(this, fieldIndex);
            }
            setRecomputeFlag();
        }
    } else if (volumeData && !useSeparateFields) {
        if (propertyEditor.addCombo(
                "Scalar Field", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            clearFieldDeviceData();
            if (fieldIndex != 0xFFFFFF) {
                volumeData->releaseScalarField(this, fieldIndex);
            }
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            if (fieldIndex != 0xFFFFFF) {
                volumeData->acquireScalarField(this, fieldIndex);
            }
            setRecomputeFlag();
        }
    }

    if (volumeData && volumeData->getEnsembleMemberCount() > 1 && volumeData->getTimeStepCount() > 1) {
        int modeIdx = isEnsembleMode ? 0 : 1;
        if (propertyEditor.addCombo("Correlation Mode", &modeIdx, CORRELATION_MODE_NAMES, 2)) {
            isEnsembleMode = modeIdx == 0;
            onCorrelationMemberCountChanged();
            clearFieldDeviceData();
            setRecomputeFlag();
        }
    }

    if (volumeData && isEnsembleMode && useTimeLagCorrelations && volumeData->getTimeStepCount() > 1) {
        ImGui::EditMode editMode = propertyEditor.addSliderIntEdit(
                "Reference Time Step", &timeLagTimeStepIdx, 0, volumeData->getTimeStepCount() - 1);
        if (editMode == ImGui::EditMode::INPUT_FINISHED) {
            clearFieldDeviceData();
            setRecomputeFlag();
        }
    }

    if (volumeData && scalarFieldNames.size() > 1 && propertyEditor.addCheckbox(
            "Two Fields Mode", &useSeparateFields)) {
        if (fieldIndex2 != 0xFFFFFF) {
            if (useSeparateFields) {
                volumeData->acquireScalarField(this, fieldIndex2);
            } else {
                volumeData->releaseScalarField(this, fieldIndex2);
            }
        }
        clearFieldDeviceData();
        setRecomputeFlag();
    }

    if (volumeData && useSeparateFields && isEnsembleMode && volumeData->getTimeStepCount() > 1
            && propertyEditor.addCheckbox("Time Lag Correlations", &useTimeLagCorrelations)) {
        useTimeLagCorrelations = volumeData->getCurrentTimeStepIdx();
        if (!useTimeLagCorrelations) {
            clearFieldDeviceData();
            setRecomputeFlag();
        }
    }

    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        setRecomputeFlag();
        //correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
    }

    if (!isMeasureMI(correlationMeasureType) && propertyEditor.addCheckbox("Absolute Value", &calculateAbsoluteValue)) {
        //correlationComputePass->setCalculateAbsoluteValue(calculateAbsoluteValue);
        setRecomputeFlag();
    }

    if (isMeasureBinnedMI(correlationMeasureType) && propertyEditor.addSliderIntEdit(
            "#Bins", &numBins, 10, 100) == ImGui::EditMode::INPUT_FINISHED) {
        //correlationComputePass->setNumBins(numBins);
        setRecomputeFlag();
    }
    if (isMeasureKraskovMI(correlationMeasureType) && propertyEditor.addSliderIntEdit(
            "#Neighbors", &k, 1, kMax) == ImGui::EditMode::INPUT_FINISHED) {
        //correlationComputePass->setKraskovNumNeighbors(k);
        setRecomputeFlag();
    }

    if (propertyEditor.addSliderFloat("Point Size", &pointSize, 1.0f, 10.0f)) {
        parentDiagram->setPointRadius(pointSize);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    glm::vec4 pointColorVec = pointColor.getFloatColorRGBA();
    if (propertyEditor.addColorEdit4("Point Color", &pointColorVec.x)) {
        pointColor = sgl::colorFromVec4(pointColorVec);
        parentDiagram->setPointColor(pointColor);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCheckbox("Align with Window", &alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.beginNode("Advanced Settings")) {
        if (!scalarFieldNames.empty() && getSupportsBufferMode() && propertyEditor.addCombo(
                "Data Mode", (int*)&dataMode, DATA_MODE_NAMES, IM_ARRAYSIZE(DATA_MODE_NAMES))) {
            clearFieldDeviceData();
            setRecomputeFlag();
        }
        if (dataMode != CorrelationDataMode::IMAGE_3D_ARRAY && propertyEditor.addCheckbox(
                "Use Buffer Tiling", &useBufferTiling)) {
            clearFieldDeviceData();
            setRecomputeFlag();
        }
        propertyEditor.endNode();
    }

    if (parentDiagram->renderGuiPropertyEditor(propertyEditor)) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
}

void DistributionSimilarityRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);

    bool diagramChanged = false;
    diagramChanged |= settings.getValueOpt("diagram_view", diagramViewIdx);
    if (diagramChanged) {
        recreateDiagramSwapchain();
    }
    if (settings.getValueOpt("align_with_parent_window", alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
    }

    if (settings.getValueOpt("use_separate_fields", useSeparateFields)) {
        if (fieldIndex2 != 0xFFFFFF) {
            if (useSeparateFields) {
                volumeData->acquireScalarField(this, fieldIndex2);
            } else {
                volumeData->releaseScalarField(this, fieldIndex2);
            }
        }
        clearFieldDeviceData();
        setRecomputeFlag();
    }

    if (useSeparateFields) {
        if (settings.getValueOpt("scalar_field_idx_ref", fieldIndex2Gui)) {
            clearFieldDeviceData();
            if (fieldIndex2 != 0xFFFFFF) {
                volumeData->releaseScalarField(this, fieldIndex2);
            }
            fieldIndex2 = int(scalarFieldIndexArray.at(fieldIndex2Gui));
            if (fieldIndex2 != 0xFFFFFF) {
                volumeData->acquireScalarField(this, fieldIndex2);
            }
            setRecomputeFlag();
        }
        if (settings.getValueOpt("scalar_field_idx_query", fieldIndexGui)) {
            clearFieldDeviceData();
            if (fieldIndex != 0xFFFFFF) {
                volumeData->releaseScalarField(this, fieldIndex);
            }
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            if (fieldIndex != 0xFFFFFF) {
                volumeData->acquireScalarField(this, fieldIndex);
            }
            setRecomputeFlag();
        }
    } else {
        if (settings.getValueOpt("scalar_field_idx", fieldIndexGui)) {
            clearFieldDeviceData();
            if (fieldIndex != 0xFFFFFF) {
                volumeData->releaseScalarField(this, fieldIndex);
            }
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            if (fieldIndex != 0xFFFFFF) {
                volumeData->acquireScalarField(this, fieldIndex);
            }
            setRecomputeFlag();
        }
    }

    std::string ensembleModeName;
    if (settings.getValueOpt("correlation_mode", ensembleModeName)) {
        if (ensembleModeName == CORRELATION_MODE_NAMES[0]) {
            isEnsembleMode = true;
        } else {
            isEnsembleMode = false;
        }
        onCorrelationMemberCountChanged();
        clearFieldDeviceData();
        setRecomputeFlag();
    }

    // Advanced settings.
    std::string dataModeString;
    if (settings.getValueOpt("data_mode", dataModeString)) {
        for (int i = 0; i < IM_ARRAYSIZE(DATA_MODE_NAMES); i++) {
            if (dataModeString == DATA_MODE_NAMES[i]) {
                dataMode = CorrelationDataMode(i);
                break;
            }
        }
        clearFieldDeviceData();
        setRecomputeFlag();
    }
    if (settings.getValueOpt("use_buffer_tiling", useBufferTiling)) {
        clearFieldDeviceData();
        setRecomputeFlag();
    }

    if (settings.getValueOpt("use_time_lag_correlations", useTimeLagCorrelations)) {
        clearFieldDeviceData();
        setRecomputeFlag();
    }
    if (settings.getValueOpt("time_lag_time_step_idx", timeLagTimeStepIdx)) {
        clearFieldDeviceData();
        setRecomputeFlag();
    }

    std::string correlationMeasureTypeName;
    if (settings.getValueOpt("correlation_measure_type", correlationMeasureTypeName)) {
        for (int i = 0; i < IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_IDS); i++) {
            if (correlationMeasureTypeName == CORRELATION_MEASURE_TYPE_IDS[i]) {
                correlationMeasureType = CorrelationMeasureType(i);
                break;
            }
        }
        setRecomputeFlag();
        //correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
    }
    /*std::string deviceName;
    if (settings.getValueOpt("device", deviceName)) {
        const char* const choices[] = {
                "CPU", "Vulkan", "CUDA"
        };
        int choice = 1;
        for (int i = 0; i < 3; i++) {
            if (deviceName == choices[i]) {
                choice = i;
                break;
            }
        }
#ifndef SUPPORT_CUDA_INTEROP
        if (choice == 2) {
            choice = 1;
        }
#else
        if (choice == 2 && !sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            choice = 1;
        }
#endif
        bool useGpuOld = useGpu;
        useGpu = choice != 0;
        useCuda = choice != 1;
        hasFilterDeviceChanged = useGpuOld != useGpu;
        setRecomputeFlag();
    }*/
    if (settings.getValueOpt("calculate_absolute_value", calculateAbsoluteValue)) {
        //correlationComputePass->setCalculateAbsoluteValue(calculateAbsoluteValue);
        setRecomputeFlag();
    }
    if (settings.getValueOpt("mi_bins", numBins)) {
        //correlationComputePass->setNumBins(numBins);
        setRecomputeFlag();
    }
    if (settings.getValueOpt("kmi_neighbors", k)) {
        //correlationComputePass->setKraskovNumNeighbors(k);
        setRecomputeFlag();
    }

    if (settings.getValueOpt("point_size", pointSize)) {
        parentDiagram->setPointRadius(pointSize);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
    glm::vec4 pointColorVec = pointColor.getFloatColorRGBA();
    if (settings.getValueOpt("point_color", pointColorVec)) {
        pointColor = sgl::colorFromVec4(pointColorVec);
        parentDiagram->setPointColor(pointColor);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    setRecomputeFlag();
}

void DistributionSimilarityRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);

    settings.addKeyValue("diagram_view", diagramViewIdx);
    settings.addKeyValue("align_with_parent_window", alignWithParentWindow);

    settings.addKeyValue("use_separate_fields", useSeparateFields);
    if (useSeparateFields) {
        settings.addKeyValue("scalar_field_idx_ref", fieldIndex2Gui);
        settings.addKeyValue("scalar_field_idx_query", fieldIndexGui);
    } else {
        settings.addKeyValue("scalar_field_idx", fieldIndexGui);
    }
    settings.addKeyValue("correlation_mode", CORRELATION_MODE_NAMES[isEnsembleMode ? 0 : 1]);

    // Advanced settings.
    settings.addKeyValue("data_mode", DATA_MODE_NAMES[int(dataMode)]);
    settings.addKeyValue("use_buffer_tiling", useBufferTiling);
    settings.addKeyValue("use_time_lag_correlations", useTimeLagCorrelations);
    settings.addKeyValue("time_lag_time_step_idx", timeLagTimeStepIdx);

    settings.addKeyValue("correlation_measure_type", CORRELATION_MEASURE_TYPE_IDS[int(correlationMeasureType)]);
    //const char* const choices[] = {
    //        "CPU", "Vulkan", "CUDA"
    //};
    //settings.addKeyValue("device", choices[!useGpu ? 0 : (!useCuda ? 1 : 2)]);
    settings.addKeyValue("calculate_absolute_value", calculateAbsoluteValue);
    settings.addKeyValue("mi_bins", numBins);
    settings.addKeyValue("kmi_neighbors", k);

    settings.addKeyValue("point_size", pointSize);
    settings.addKeyValue("point_color", pointColor.getFloatColorRGBA());

    // No vector widget settings for now.
}
