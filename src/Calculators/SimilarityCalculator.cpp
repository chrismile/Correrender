/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022-2023, Christoph Neuhauser
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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/AppSettings.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>
#include <Input/Mouse.hpp>
#include <Input/Keyboard.hpp>

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "Widgets/ViewManager.hpp"
#include "MutualInformation.hpp"
#include "ReferencePointSelectionRenderer.hpp"
#include "SimilarityCalculator.hpp"

EnsembleSimilarityCalculator::EnsembleSimilarityCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
}

void EnsembleSimilarityCalculator::setViewManager(ViewManager* _viewManager) {
    viewManager = _viewManager;
    referencePointSelectionRenderer = new ReferencePointSelectionRenderer(viewManager);
    calculatorRenderer = RendererPtr(referencePointSelectionRenderer);
    referencePointSelectionRenderer->initialize();
}

void EnsembleSimilarityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    referencePointSelectionRenderer->setVolumeDataPtr(volumeData, isNewData);

    scalarFieldNames = {};
    scalarFieldIndexArray = {};

    std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
    for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
        if (scalarFieldNamesNew.at(i) != getOutputFieldName()) {
            scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
            scalarFieldIndexArray.push_back(i);
        }
    }

    if (isNewData) {
        referencePointIndex.x = volumeData->getGridSizeX() / 2;
        referencePointIndex.y = volumeData->getGridSizeY() / 2;
        referencePointIndex.z = volumeData->getGridSizeZ() / 2;
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);

        fieldIndex = volumeData->getStandardScalarFieldIdx();
        fieldIndexGui = volumeData->getStandardScalarFieldIdx();
        volumeData->acquireScalarField(this, fieldIndex);
    }
}

void EnsembleSimilarityCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (fieldIndex == fieldIdx) {
            fieldIndex = 0;
            volumeData->acquireScalarField(this, fieldIndex);
            dirty = true;
        } else if (fieldIndex > fieldIdx) {
            fieldIndex--;
        }
        fieldIndexGui = fieldIndex;
    }
}

void EnsembleSimilarityCalculator::update(float dt) {
    // Use mouse for selection of reference point.
    int mouseHoverWindowIndex = viewManager->getMouseHoverWindowIndex();
    if (mouseHoverWindowIndex >= 0) {
        SceneData* sceneData = viewManager->getViewSceneData(uint32_t(mouseHoverWindowIndex));
        uint32_t varIdx = volumeData->getVarIdxForCalculator(this);
        if (volumeData->getIsScalarFieldUsedInView(uint32_t(mouseHoverWindowIndex), varIdx, this)) {
            if (sgl::Keyboard->getModifier() & KMOD_CTRL) {
                if (sgl::Mouse->buttonPressed(1) || (sgl::Mouse->isButtonDown(1) && sgl::Mouse->mouseMoved())) {
                    ImVec2 mousePosGlobal = ImGui::GetMousePos();
                    //int mouseGlobalX = sgl::Mouse->getX();
                    //int mouseGlobalY = sgl::Mouse->getY();
                    bool rayHasHitMesh;
                    if (fixPickingZPlane) {
                        glm::vec3 centerHit;
                        rayHasHitMesh = volumeData->pickPointScreenAtZ(
                                sceneData, int(mousePosGlobal.x), int(mousePosGlobal.y),
                                volumeData->getGridSizeZ() / 2, centerHit);
                        if (rayHasHitMesh) {
                            auto aabb = volumeData->getBoundingBoxRendering();
                            focusPoint = centerHit;
                            firstHit = glm::vec3(centerHit.x, centerHit.y, aabb.max.z);
                            lastHit = glm::vec3(centerHit.x, centerHit.y, aabb.min.z);
                            hitLookingDirection = glm::vec3(0.0f, 0.0f, -glm::sign(sceneData->camera->getPosition().z));
                            hasHitInformation = true;
                            setReferencePointFromFocusPoint();
                        }
                    } else {
                        rayHasHitMesh = volumeData->pickPointScreen(
                                sceneData, int(mousePosGlobal.x), int(mousePosGlobal.y), firstHit, lastHit);
                        if (rayHasHitMesh) {
                            focusPoint = firstHit;
                            hitLookingDirection = glm::normalize(firstHit - sceneData->camera->getPosition());
                            hasHitInformation = true;
                            setReferencePointFromFocusPoint();
                        }
                    }
                }

                if (sgl::Mouse->getScrollWheel() > 0.1f || sgl::Mouse->getScrollWheel() < -0.1f) {
                    if (!hasHitInformation) {
                        glm::mat4 inverseViewMatrix = glm::inverse(sceneData->camera->getViewMatrix());
                        glm::vec3 lookingDirection = glm::vec3(
                                -inverseViewMatrix[2].x, -inverseViewMatrix[2].y, -inverseViewMatrix[2].z);

                        float moveAmount = sgl::Mouse->getScrollWheel() * dt * 0.5f;
                        glm::vec3 moveDirection = focusPoint - sceneData->camera->getPosition();
                        moveDirection *= float(sgl::sign(glm::dot(lookingDirection, moveDirection)));
                        if (glm::length(moveDirection) < 1e-4) {
                            moveDirection = lookingDirection;
                        }
                        moveDirection = glm::normalize(moveDirection);
                        focusPoint = focusPoint + moveAmount * moveDirection;
                    } else {
                        float moveAmount = sgl::Mouse->getScrollWheel() * dt;
                        glm::vec3 newFocusPoint = focusPoint + moveAmount * hitLookingDirection;
                        float t = glm::dot(newFocusPoint - firstHit, hitLookingDirection);
                        t = glm::clamp(t, 0.0f, glm::length(lastHit - firstHit));
                        focusPoint = firstHit + t * hitLookingDirection;
                    }
                    setReferencePointFromFocusPoint();
                }
            }
        }
    }

    if (continuousRecompute) {
        dirty = true;
    }
}

void EnsembleSimilarityCalculator::setReferencePointFromFocusPoint() {
    glm::ivec3 newReferencePointIndex = glm::ivec3(glm::round(focusPoint));
    if (referencePointIndex != newReferencePointIndex) {
        glm::ivec3 maxCoord(
                volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
        sgl::AABB3 gridAabb = volumeData->getBoundingBoxRendering();
        glm::vec3 position = (focusPoint - gridAabb.min) / (gridAabb.max - gridAabb.min);
        position *= glm::vec3(maxCoord);
        referencePointIndex = glm::ivec3(glm::round(position));
        referencePointIndex = glm::clamp(referencePointIndex, glm::ivec3(0), maxCoord);
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        dirty = true;
    }
}

void EnsembleSimilarityCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addCombo(
            "Scalar Field", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
        dirty = true;
    }

    bool isRealtime = getIsRealtime();
    ImGui::EditMode editModes[3];
    editModes[0] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (X)", &referencePointIndex[0], 0, volumeData->getGridSizeX() - 1);
    editModes[1] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Y)", &referencePointIndex[1], 0, volumeData->getGridSizeY() - 1);
    editModes[2] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Z)", &referencePointIndex[2], 0, volumeData->getGridSizeZ() - 1);
    if ((isRealtime && (editModes[0] != ImGui::EditMode::NO_CHANGE
                || editModes[1] != ImGui::EditMode::NO_CHANGE
                || editModes[2] != ImGui::EditMode::NO_CHANGE))
            || (!isRealtime && (editModes[0] == ImGui::EditMode::INPUT_FINISHED
                || editModes[1] == ImGui::EditMode::INPUT_FINISHED
                || editModes[2] == ImGui::EditMode::INPUT_FINISHED))) {
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        dirty = true;
    }

    propertyEditor.addCheckbox("Fix Picking Z", &fixPickingZPlane);
}



PccCalculator::PccCalculator(sgl::vk::Renderer* renderer) : EnsembleSimilarityCalculator(renderer) {
    useCuda = sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() && sgl::vk::getIsNvrtcFunctionTableInitialized();

    pccComputePass = std::make_shared<PccComputePass>(renderer);
    pccComputePass->setCorrelationMeasureType(correlationMeasureType);
    pccComputePass->setNumBins(numBins);
    pccComputePass->setKraskovNumNeighbors(k);
    pccComputePass->setKraskovEstimatorIndex(kraskovEstimatorIndex);
}

void PccCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    EnsembleSimilarityCalculator::setVolumeData(_volumeData, isNewData);
    calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::CORRELATION);
    pccComputePass->setVolumeData(volumeData, isNewData);

    int es = volumeData->getEnsembleMemberCount();
    k = std::max(sgl::iceil(3 * es, 100), 1);
    kMax = std::max(sgl::iceil(7 * es, 100), 20);
    pccComputePass->setKraskovNumNeighbors(k);
}

FilterDevice PccCalculator::getFilterDevice() {
    if (useGpu) {
        if (useCuda) {
            return FilterDevice::CUDA;
        } else {
            return FilterDevice::VULKAN;
        }
    }
    return FilterDevice::CPU;
}

void PccCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    EnsembleSimilarityCalculator::renderGuiImpl(propertyEditor);
    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        hasNameChanged = true;
        dirty = true;
        pccComputePass->setCorrelationMeasureType(correlationMeasureType);
        if (useGpu && (correlationMeasureType == CorrelationMeasureType::KENDALL)) {
            hasFilterDeviceChanged = true;
            useGpu = false;
        }
    }

#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() && sgl::vk::getIsNvrtcFunctionTableInitialized()
            && correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        const char* const choices[] = {
                "CPU", "Vulkan", "CUDA"
        };
        int choice = !useGpu ? 0 : (!useCuda ? 1 : 2);
        if (propertyEditor.addCombo("Device", &choice, choices, 3)) {
            bool useGpuOld = useGpu;
            useGpu = choice != 0;
            useCuda = choice != 1;
            hasFilterDeviceChanged = useGpuOld != useGpu;
            dirty = true;
        }
    } else
#endif
    if (correlationMeasureType != CorrelationMeasureType::KENDALL && propertyEditor.addCheckbox("Use GPU", &useGpu)) {
        hasFilterDeviceChanged = true;
        dirty = true;
    }

    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED && propertyEditor.addSliderIntEdit(
            "#Bins", &numBins, 10, 100) == ImGui::EditMode::INPUT_FINISHED) {
        pccComputePass->setNumBins(numBins);
        dirty = true;
    }
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV && propertyEditor.addSliderIntEdit(
            "#Neighbors", &k, 1, kMax) == ImGui::EditMode::INPUT_FINISHED) {
        pccComputePass->setKraskovNumNeighbors(k);
        dirty = true;
    }
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV && propertyEditor.addSliderIntEdit(
            "Kraskov Estimator", &kraskovEstimatorIndex, 1, 2) == ImGui::EditMode::INPUT_FINISHED) {
        kraskovEstimatorIndex = std::clamp(kraskovEstimatorIndex, 1, 2);
        pccComputePass->setKraskovEstimatorIndex(kraskovEstimatorIndex);
        dirty = true;
    }

#ifdef SHOW_DEBUG_OPTIONS
    if (propertyEditor.addCheckbox("Continuous Recompute", &continuousRecompute)) {
        dirty = true;
    }
#endif
}

template<class T>
inline float computePearson1(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx) {
    auto n = T(es);
    T sumX = 0;
    T sumY = 0;
    T sumXY = 0;
    T sumXX = 0;
    T sumYY = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
        sumYY += y * y;
    }
    float pearsonCorrelation =
            (n * sumXY - sumX * sumY) / std::sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    return (float)pearsonCorrelation;
}

template<class T>
inline float computePearson1(
        const float* referenceValues, const float* queryValues, int es) {
    auto n = T(es);
    T sumX = 0;
    T sumY = 0;
    T sumXY = 0;
    T sumXX = 0;
    T sumYY = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
        sumYY += y * y;
    }
    float pearsonCorrelation =
            (n * sumXY - sumX * sumY) / std::sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    return (float)pearsonCorrelation;
}

template<class T>
inline float computePearson2(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx) {
    auto n = T(es);
    T meanX = 0;
    T meanY = 0;
    T invN = T(1) / n;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        meanX += invN * x;
        meanY += invN * y;
    }
    T varX = 0;
    T varY = 0;
    T invNm1 = T(1) / (n - T(1));
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        T diffX = x - meanX;
        T diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    T stdDevX = std::sqrt(varX);
    T stdDevY = std::sqrt(varY);
    T pearsonCorrelation = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }
    return (float)pearsonCorrelation;
}

template<class T>
inline float computePearson2(
        const float* referenceValues, const float* queryValues, int es) {
    auto n = T(es);
    T meanX = 0;
    T meanY = 0;
    T invN = T(1) / n;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        meanX += invN * x;
        meanY += invN * y;
    }
    T varX = 0;
    T varY = 0;
    T invNm1 = T(1) / (n - T(1));
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        T diffX = x - meanX;
        T diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    T stdDevX = std::sqrt(varX);
    T stdDevY = std::sqrt(varY);
    T pearsonCorrelation = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }
    return (float)pearsonCorrelation;
}

void computeRanks(const float* values, float* ranks, std::vector<std::pair<float, int>>& ordinalRankArray, int es) {
    ordinalRankArray.clear();
    for (int i = 0; i < es; i++) {
        ordinalRankArray.emplace_back(values[i], i);
    }
    std::sort(ordinalRankArray.begin(), ordinalRankArray.end());

    // Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    float currentRank = 1.0f;
    int idx = 0;
    while (idx < es) {
        float value = ordinalRankArray.at(idx).first;
        int idxEqualEnd = idx + 1;
        while (idxEqualEnd < es && value == ordinalRankArray.at(idxEqualEnd).first) {
            idxEqualEnd++;
        }

        int numEqualValues = idxEqualEnd - idx;
        float meanRank = currentRank + float(numEqualValues - 1) * 0.5f;
        for (int offset = 0; offset < numEqualValues; offset++) {
            ranks[ordinalRankArray.at(idx + offset).second] = meanRank;
        }

        idx += numEqualValues;
        currentRank += float(numEqualValues);
    }
}

int computeTiesB(const float* values, std::vector<float>& ordinalRankArray, int es) {
    ordinalRankArray.clear();
    for (int i = 0; i < es; i++) {
        ordinalRankArray.emplace_back(values[i]);
    }
    std::sort(ordinalRankArray.begin(), ordinalRankArray.end());

    // Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    int tiesB = 0;
    int idx = 0;
    while (idx < es) {
        float value = ordinalRankArray.at(idx);
        int idxEqualEnd = idx + 1;
        while (idxEqualEnd < es && value == ordinalRankArray.at(idxEqualEnd)) {
            idxEqualEnd++;
        }

        int numEqualValues = idxEqualEnd - idx;
        tiesB += numEqualValues * (numEqualValues - 1) / 2;
        idx += numEqualValues;
    }

    return tiesB;
}

// https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
int M(const std::vector<float>& L, const std::vector<float>& R) {
    int n = int(L.size());
    int m = int(R.size());
    int i = 0;
    int j = 0;
    int num_swaps = 0;
    while (i < n && j < m) {
        if (R[j] < L[i]) {
            num_swaps += n - i;
            j += 1;
        } else {
            i += 1;
        }
    }
    return num_swaps;
}

// https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
int S(const std::vector<float>& y) {
    int n = int(y.size());
    if (n <= 1) {
        return 0;
    }
    int s = n / 2;
    std::vector<float> y_l = std::vector<float>(y.begin(), y.begin() + s);
    std::vector<float> y_r = std::vector<float>(y.begin() + s, y.end());
    int S_y_l = S(y_l);
    int S_y_r = S(y_r);
    std::sort(y_l.begin(), y_l.end());
    std::sort(y_r.begin(), y_r.end());
    return S_y_l + S_y_r + M(y_l, y_r);
}

float computeKendall(
        const float* referenceValues, const float* queryValues, int es,
        std::vector<std::pair<float, float>>& jointArray, std::vector<float>& ordinalRankArray,
        std::vector<float>& y) {
    int n = es;
    for (int i = 0; i < es; i++) {
        jointArray.emplace_back(referenceValues[i], queryValues[i]);
    }
    std::sort(jointArray.begin(), jointArray.end());
    for (int i = 0; i < es; i++) {
        y.push_back(jointArray[i].second);
    }
    jointArray.clear();
    int S_y = S(y);
    y.clear();
    int n0 = (n * (n - 1)) / 2;
    int n1 = computeTiesB(referenceValues, ordinalRankArray, es);
    ordinalRankArray.clear();
    int n2 = computeTiesB(queryValues, ordinalRankArray, es);
    ordinalRankArray.clear();
    int n3 = 0;  // Joint ties in ref and query, TODO.
    int numerator = n0 - n1 - n2 + n3 - 2 * S_y;
    //auto denominator = float(n0);  // Tau-a
    float denominator = std::sqrt(float((n0 - n1) * (n0 - n2)));
    return float(numerator) / denominator;
}

void PccCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

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

    //size_t referencePointIdx =
    //        size_t(referencePointIndex.x) * size_t(referencePointIndex.y) * size_t(referencePointIndex.z);
    size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
    auto* referenceValues = new float[es];
    for (int e = 0; e < es; e++) {
        referenceValues[e] = ensembleFields.at(e)[referencePointIdx];
    }

    float minEnsembleVal = std::numeric_limits<float>::max();
    float maxEnsembleVal = std::numeric_limits<float>::lowest();
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
            auto [minVal, maxVal] = volumeData->getMinMaxScalarFieldValue(
                    scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
            minEnsembleVal = std::min(minEnsembleVal, minVal);
            maxEnsembleVal = std::max(maxEnsembleVal, maxVal);
        }
        for (int e = 0; e < es; e++) {
            referenceValues[e] = (referenceValues[e] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
        }
    }


#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    float* referenceRanks = nullptr;
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        referenceRanks = new float[es];
        std::vector<std::pair<float, int>> ordinalRankArray;
        ordinalRankArray.reserve(es);
        computeRanks(referenceValues, referenceRanks, ordinalRankArray, es);
    }

    size_t numGridPoints = size_t(xs) * size_t(ys) * size_t(zs);
    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* referenceValues = new float[es];
            for (int e = 0; e < es; e++) {
                referenceValues[e] = ensembleFields.at(e)[referencePointIdx];
            }
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel for shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none)
#endif
        for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
            if (es == 1) {
                buffer[gridPointIdx] = 1.0f;
                continue;
            }
            // See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
#define FORMULA_2_FLOAT
#ifdef FORMULA_1_FLOAT
            float pearsonCorrelation = computePearson1<float>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_1_DOUBLE)
            float pearsonCorrelation = computePearson1<double>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_2_FLOAT)
            float pearsonCorrelation = computePearson2<float>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_2_DOUBLE)
            float pearsonCorrelation = computePearson2<double>(referenceValues, ensembleFields, es, gridPointIdx);
#endif
            buffer[gridPointIdx] = pearsonCorrelation;
        }
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            auto* gridPointRanks = new float[es];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(es);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, referenceRanks, ensembleFields, buffer) default(none)
        {
            auto* gridPointValues = new float[es];
            auto* gridPointRanks = new float[es];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(es);
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                computeRanks(gridPointValues, gridPointRanks, ordinalRankArray, es);

#define FORMULA_2_FLOAT
#ifdef FORMULA_1_FLOAT
                float pearsonCorrelation = computePearson1<float>(referenceRanks, gridPointRanks, es);
#elif defined(FORMULA_1_DOUBLE)
                float pearsonCorrelation = computePearson1<double>(referenceRanks, gridPointRanks, es);
#elif defined(FORMULA_2_FLOAT)
                float pearsonCorrelation = computePearson2<float>(referenceRanks, gridPointRanks, es);
#elif defined(FORMULA_2_DOUBLE)
                float pearsonCorrelation = computePearson2<double>(referenceRanks, gridPointRanks, es);
#endif
                buffer[gridPointIdx] = pearsonCorrelation;
            }
            delete[] gridPointValues;
            delete[] gridPointRanks;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            jointArray.reserve(es);
            ordinalRankArray.reserve(es);
            y.reserve(es);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none)
        {
            auto* gridPointValues = new float[es];
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            jointArray.reserve(es);
            ordinalRankArray.reserve(es);
            y.reserve(es);
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float pearsonCorrelation = computeKendall(
                        referenceValues, gridPointValues, es, jointArray, ordinalRankArray, y);
                buffer[gridPointIdx] = pearsonCorrelation;
            }
            delete[] gridPointValues;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            auto* histogram0 = new double[numBins];
            auto* histogram1 = new double[numBins];
            auto* histogram2d = new double[numBins * numBins];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none) \
        shared(minEnsembleVal, maxEnsembleVal)
        {
            auto* gridPointValues = new float[es];
            auto* histogram0 = new double[numBins];
            auto* histogram1 = new double[numBins];
            auto* histogram2d = new double[numBins * numBins];
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                    gridPointValues[e] = (gridPointValues[e] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation = computeMutualInformationBinned<double>(
                        referenceValues, gridPointValues, numBins, es, histogram0, histogram1, histogram2d);
                buffer[gridPointIdx] = mutualInformation;
            }
            delete[] gridPointValues;
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, k, referenceValues, ensembleFields, buffer) default(none) \
        shared(minEnsembleVal, maxEnsembleVal)
        {
            auto* gridPointValues = new float[es];
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                    //gridPointValues[e] = (gridPointValues[e] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation;
                if (kraskovEstimatorIndex == 1) {
                    mutualInformation = computeMutualInformationKraskov<double>(
                            referenceValues, gridPointValues, k, es);
                } else {
                    mutualInformation = computeMutualInformationKraskov2<double>(
                            referenceValues, gridPointValues, k, es);
                }
                buffer[gridPointIdx] = mutualInformation;
            }
            delete[] gridPointValues;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    }

    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        delete[] referenceRanks;
    }
    delete[] referenceValues;

#ifdef TEST_INFERENCE_SPEED
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}

void PccCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    // We write to the descriptor set, so wait until the device is idle.
    renderer->getDevice()->waitIdle();

    int es = volumeData->getEnsembleMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

    std::vector<VolumeData::DeviceCacheEntry> ensembleEntryFields;
    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    ensembleEntryFields.reserve(es);
    ensembleImageViews.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleImageViews.push_back(ensembleEntryField->getVulkanImageView());
        if (ensembleEntryField->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            ensembleEntryField->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        }
    }

#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    if (correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV || !useCuda) {
        pccComputePass->setOutputImage(deviceCacheEntry->getVulkanImageView());
        pccComputePass->setEnsembleImageViews(ensembleImageViews);

        renderer->insertImageMemoryBarrier(
                deviceCacheEntry->getVulkanImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);

        pccComputePass->buildIfNecessary();
        pccComputePass->setReferencePoint(referencePointIndex);
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            float minEnsembleVal = std::numeric_limits<float>::max();
            float maxEnsembleVal = std::numeric_limits<float>::lowest();
            if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
                    auto [minVal, maxVal] = volumeData->getMinMaxScalarFieldValue(
                            scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
                    minEnsembleVal = std::min(minEnsembleVal, minVal);
                    maxEnsembleVal = std::max(maxEnsembleVal, maxVal);
                }
            }
            renderer->pushConstants(
                    pccComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::vec4),
                    glm::vec2(minEnsembleVal, maxEnsembleVal));
        }
        pccComputePass->render();
    } else {
        pccComputePass->setEnsembleImageViews(ensembleImageViews);
        pccComputePass->computeCuda(
                scalarFieldNames.at(fieldIndexGui), timeStepIdx, deviceCacheEntry, referencePointIndex);
    }

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}



PccComputePass::PccComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    spearmanReferenceRankComputePass = std::make_shared<SpearmanReferenceRankComputePass>(renderer, uniformBuffer);

#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamCreate(
                &stream, 0), "Error in cuStreamCreate: ");
    }
#endif
}

PccComputePass::~PccComputePass() {
#ifdef SUPPORT_CUDA_INTEROP
    if (outputImageBufferCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                outputImageBufferCu), "Error in cuMemFree: ");
    }
    if (ensembleTextureArrayCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                ensembleTextureArrayCu), "Error in cuMemFree: ");
    }
    if (stream) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamDestroy(
                stream), "Error in cuStreamDestroy: ");
    }
#endif
}

void PccComputePass::setVolumeData(VolumeData *_volumeData, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.es = uint32_t(volumeData->getEnsembleMemberCount());
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    spearmanReferenceRankComputePass->setVolumeData(_volumeData, isNewData);
    if (cachedEnsembleMemberCount != volumeData->getEnsembleMemberCount()) {
        cachedEnsembleMemberCount = volumeData->getEnsembleMemberCount();
        setShaderDirty();
        spearmanReferenceRankComputePass->setEnsembleMemberCount(cachedEnsembleMemberCount);
    }
}

void PccComputePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews == _ensembleImageViews) {
        return;
    }
    ensembleImageViews = _ensembleImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    }
    spearmanReferenceRankComputePass->setEnsembleImageViews(_ensembleImageViews);
}

void PccComputePass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    outputImage = _outputImage;
    if (computeData) {
        computeData->setStaticImageView(outputImage, "outputImage");
    }
}

void PccComputePass::setReferencePoint(const glm::ivec3& referencePointIndex) {
    if (correlationMeasureType == CorrelationMeasureType::PEARSON
            || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
            || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, referencePointIndex);
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        spearmanReferenceRankComputePass->buildIfNecessary();
        renderer->pushConstants(
                spearmanReferenceRankComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                0, referencePointIndex);
    }
}

void PccComputePass::setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType) {
    if (correlationMeasureType != _correlationMeasureType) {
        correlationMeasureType = _correlationMeasureType;
        setShaderDirty();
    }
}

void PccComputePass::setNumBins(int _numBins) {
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED && numBins != _numBins) {
        setShaderDirty();
    }
    numBins = _numBins;
}

void PccComputePass::setKraskovNumNeighbors(int _k) {
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV && k != _k) {
        setShaderDirty();
    }
    k = _k;
}

void PccComputePass::setKraskovEstimatorIndex(int _kraskovEstimatorIndex) {
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
            && kraskovEstimatorIndex != _kraskovEstimatorIndex) {
        setShaderDirty();
    }
    kraskovEstimatorIndex = _kraskovEstimatorIndex;
}

void PccComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(volumeData->getEnsembleMemberCount())));
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        preprocessorDefines.insert(std::make_pair("numBins", std::to_string(numBins)));
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        auto maxBinaryTreeLevels = uint32_t(std::ceil(std::log2(volumeData->getEnsembleMemberCount() + 1)));
        preprocessorDefines.insert(std::make_pair(
                "MAX_STACK_SIZE_BUILD", std::to_string(2 * maxBinaryTreeLevels)));
        preprocessorDefines.insert(std::make_pair(
                "MAX_STACK_SIZE_KN", std::to_string(maxBinaryTreeLevels)));
        preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));
        preprocessorDefines.insert(std::make_pair("KRASKOV_ESTIMATOR_INDEX", std::to_string(kraskovEstimatorIndex)));
    }
    std::string shaderName;
    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
        shaderName = "PearsonCorrelation.Compute";
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        shaderName = "SpearmanRankCorrelation.Compute";
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        shaderName = "MutualInformationBinned.Compute";
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        shaderName = "MutualInformationKraskov.Compute";
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void PccComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticImageView(outputImage, "outputImage");
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        computeData->setStaticBuffer(
                spearmanReferenceRankComputePass->getReferenceRankBuffer(), "ReferenceRankBuffer");
    }
}

void PccComputePass::_render() {
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        spearmanReferenceRankComputePass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                spearmanReferenceRankComputePass->getReferenceRankBuffer());
    }

    uint32_t batchCount = 1;
    bool needsBatchedRendering = false;
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        if (volumeData->getEnsembleMemberCount() > int(batchEnsembleCountThreshold)) {
            needsBatchedRendering = true;
            batchCount = sgl::uiceil(uint32_t(volumeData->getEnsembleMemberCount()), batchEnsembleCountThreshold);
        }
    }

    if (needsBatchedRendering) {
        auto blockSizeX = uint32_t(computeBlockSizeX);
        //auto blockSizeY = uint32_t(computeBlockSizeY);
        //auto blockSizeZ = uint32_t(computeBlockSizeZ);
        /*auto batchSizeX =
                sgl::uiceil(uint32_t(volumeData->getGridSizeX()), batchCount * blockSizeX) * blockSizeX;
        auto batchSizeY = uint32_t(volumeData->getGridSizeY());
        auto batchSizeZ = uint32_t(volumeData->getGridSizeZ());
        batchCount = sgl::uiceil(uint32_t(volumeData->getGridSizeX()), batchSizeX);*/
        auto batchSizeX = 2 * blockSizeX;
        auto batchSizeY = uint32_t(volumeData->getGridSizeY());
        auto batchSizeZ = uint32_t(volumeData->getGridSizeZ());
        batchCount = sgl::uiceil(uint32_t(volumeData->getGridSizeX()), batchSizeX);
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                renderer->pushConstants(
                        getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::uvec4),
                        glm::uvec3(batchSizeX * batchIdx, 0, 0));
            }
            if (batchIdx == batchCount - 1) {
                batchSizeX = uint32_t(volumeData->getGridSizeX()) - batchSizeX * batchIdx;
            }
            renderer->dispatch(
                    computeData,
                    sgl::uiceil(batchSizeX, uint32_t(computeBlockSizeX)),
                    sgl::uiceil(batchSizeY, uint32_t(computeBlockSizeY)),
                    sgl::uiceil(batchSizeZ, uint32_t(computeBlockSizeZ)));
            renderer->syncWithCpu();
        }
    } else {
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
            renderer->pushConstants(
                    getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::uvec4), glm::uvec3(0));
        }
        renderer->dispatch(
                computeData,
                sgl::iceil(volumeData->getGridSizeX(), computeBlockSizeX),
                sgl::iceil(volumeData->getGridSizeY(), computeBlockSizeY),
                sgl::iceil(volumeData->getGridSizeZ(), computeBlockSizeZ));
    }
}

struct SimilarityCalculatorKernelCache {
    ~SimilarityCalculatorKernelCache() {
        if (cumodule) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleUnload(
                    cumodule), "Error in cuModuleUnload: ");
        }
    }
    std::string kernelString;
    std::map<std::string, std::string> preprocessorDefines;
    CUmodule cumodule{};
    CUfunction kernel{};
};

struct CudaDeviceCoresInfo {
    uint32_t numMultiprocessors;
    uint32_t warpSize;
    uint32_t numCoresPerMultiprocessor;
    uint32_t numCudaCoresTotal;
};

CudaDeviceCoresInfo getNumCudaCores(CUdevice cuDevice) {
    CudaDeviceCoresInfo info{};

    /*
     * Only use one thread block per shader multiprocessor (SM) to improve chance of fair scheduling.
     * See, e.g.: https://stackoverflow.com/questions/33150040/doubling-buffering-in-cuda-so-the-cpu-can-operate-on-data-produced-by-a-persiste/33158954#33158954%5B/
     */
    CUresult cuResult;
    int numMultiprocessors = 16;
    cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &numMultiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
    sgl::vk::checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");
    info.numMultiprocessors = uint32_t(numMultiprocessors);

    /*
     * Use more threads than warp size. Factor 4 seems to make sense at least for RTX 3090.
     * For more details see: https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
     * Or: https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h
     * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
     * https://developer.nvidia.com/blog/inside-pascal/
     */
    int warpSize = 32;
    cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, cuDevice);
    sgl::vk::checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");
    info.warpSize = uint32_t(warpSize);

    int major = 0;
    cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice);
    sgl::vk::checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");
    int minor = 0;
    cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice);
    sgl::vk::checkCUresult(cuResult, "Error in cuDeviceGetAttribute: ");

    // Use warp size * 4 as fallback for unknown architectures.
    int numCoresPerMultiprocessor = warpSize * 4;

    if (major == 2) {
        if (minor == 1) {
            numCoresPerMultiprocessor = 48;
        } else {
            numCoresPerMultiprocessor = 32;
        }
    } else if (major == 3) {
        numCoresPerMultiprocessor = 192;
    } else if (major == 5) {
        numCoresPerMultiprocessor = 128;
    } else if (major == 6) {
        if (minor == 0) {
            numCoresPerMultiprocessor = 64;
        } else {
            numCoresPerMultiprocessor = 128;
        }
    } else if (major == 7) {
        numCoresPerMultiprocessor = 64;
    } else if (major == 8) {
        if (minor == 0) {
            numCoresPerMultiprocessor = 64;
        } else {
            numCoresPerMultiprocessor = 128;
        }
    } else if (major == 9) {
        numCoresPerMultiprocessor = 128;
    }
    info.numCoresPerMultiprocessor = uint32_t(numCoresPerMultiprocessor);
    info.numCudaCoresTotal = info.numMultiprocessors * info.numCoresPerMultiprocessor;

    return info;
}

void PccComputePass::computeCuda(
        const std::string& fieldName, int timeStepIdx, const DeviceCacheEntry& deviceCacheEntry,
        glm::ivec3& referencePointIndex) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();
    int N = es;
    int M = volumeData->getGridSizeX() * volumeData->getGridSizeY() * volumeData->getGridSizeZ();

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);

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

    if (cachedEnsembleSizeDevice != size_t(es)) {
        if (ensembleTextureArrayCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    ensembleTextureArrayCu, stream), "Error in cuMemFreeAsync: ");
        }
        cachedEnsembleSizeDevice = size_t(es);
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &ensembleTextureArrayCu, es * sizeof(CUtexObject), stream), "Error in cuMemAllocAsync: ");
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

    std::vector<VolumeData::DeviceCacheEntry> ensembleEntryFields;
    std::vector<CUtexObject> ensembleTexturesCu;
    ensembleEntryFields.reserve(es);
    ensembleTexturesCu.reserve(es);
    for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
                FieldType::SCALAR, fieldName, timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleTexturesCu.push_back(ensembleEntryField->getCudaTexture());
        if (ensembleEntryField->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            deviceCacheEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        }
    }

    if (cachedEnsembleTexturesCu != ensembleTexturesCu) {
        cachedEnsembleTexturesCu = ensembleTexturesCu;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                stream), "Error in cuStreamSynchronize: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                ensembleTextureArrayCu, ensembleTexturesCu.data(), sizeof(CUtexObject) * es), "Error in cuMemcpyHtoD: ");
    }

    sgl::vk::CommandBufferPtr commandBufferRender = renderer->getCommandBuffer();
    vulkanFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
    commandBufferRender->pushSignalSemaphore(vulkanFinishedSemaphore);
    renderer->endCommandBuffer();

    renderer->submitToQueue();

    vulkanFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);

    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(N)));
    auto maxBinaryTreeLevels = uint32_t(std::ceil(std::log2(N + 1)));
    preprocessorDefines.insert(std::make_pair(
            "MAX_STACK_SIZE_BUILD", std::to_string(2 * maxBinaryTreeLevels)));
    preprocessorDefines.insert(std::make_pair(
            "MAX_STACK_SIZE_KN", std::to_string(maxBinaryTreeLevels)));
    preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));

    if (!kernelCache || kernelCache->preprocessorDefines != preprocessorDefines) {
        if (kernelCache) {
            delete[] kernelCache;
        }
        kernelCache = new SimilarityCalculatorKernelCache;

        if (kernelCache->kernelString.empty()) {
            std::ifstream inFile(
                    sgl::AppSettings::get()->getDataDirectory() + "Shaders/Similarity/MutualInformationKraskov.cu",
                    std::ios::binary);
            if (!inFile.is_open()) {
                sgl::Logfile::get()->throwError(
                        "Error in PccComputePass::computeCuda: Could not open MutualInformationKraskov.cu.");
            }
            inFile.seekg(0, std::ios::end);
            auto fileSize = inFile.tellg();
            inFile.seekg(0, std::ios::beg);
            kernelCache->kernelString.resize(fileSize);
            inFile.read(kernelCache->kernelString.data(), fileSize);
            inFile.close();
        }

        std::string code;
        for (const auto& entry : preprocessorDefines) {
            code += std::string("#define ") + entry.first + " " + entry.second + "\n";
        }
        code += "#line 1\n";
        code += kernelCache->kernelString;

        nvrtcProgram prog;
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcCreateProgram(
                &prog, code.c_str(), "MutualInformationKraskov.cu", 0, nullptr, nullptr), "Error in nvrtcCreateProgram: ");
        auto retVal = nvrtcCompileProgram(prog, 0, nullptr);
        if (retVal == NVRTC_ERROR_COMPILATION) {
            size_t logSize = 0;
            sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetProgramLogSize(
                    prog, &logSize), "Error in nvrtcGetProgramLogSize: ");
            char* log = new char[logSize];
            sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetProgramLog(
                    prog, log), "Error in nvrtcGetProgramLog: ");
            std::cerr << "NVRTC log:" << std::endl << log << std::endl;
            delete[] log;
            sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcDestroyProgram(
                    &prog), "Error in nvrtcDestroyProgram: ");
            exit(1);
        }

        size_t ptxSize = 0;
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetPTXSize(
                prog, &ptxSize), "Error in nvrtcGetPTXSize: ");
        char* ptx = new char[ptxSize];
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetPTX(
                prog, ptx), "Error in nvrtcGetPTX: ");
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcDestroyProgram(
                &prog), "Error in nvrtcDestroyProgram: ");

        kernelCache->preprocessorDefines = preprocessorDefines;

        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleLoadDataEx(
                &kernelCache->cumodule, ptx, 0, nullptr, nullptr), "Error in cuModuleLoadDataEx: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &kernelCache->kernel, kernelCache->cumodule, "mutualInformationKraskov"), "Error in cuModuleGetFunction: ");
        delete[] ptx;
    }

    int minGridSize = 0;
    int bestBlockSize = 32;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuOccupancyMaxPotentialBlockSize(
            &minGridSize, &bestBlockSize, kernelCache->kernel, nullptr, 0, 0), "Error in cuOccupancyMaxPotentialBlockSize: ");
    //std::cout << "minGridSize: " << minGridSize << ", bestBlockSize: " << bestBlockSize << std::endl;

    CUdevice cuDevice{};
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuDeviceGet(&cuDevice, 0), "Error in cuDeviceGet: ");
    CudaDeviceCoresInfo deviceCoresInfo = getNumCudaCores(cuDevice);
    //std::cout << "numMultiprocessors: " << deviceCoresInfo.numMultiprocessors << ", bestBlockSize: " << deviceCoresInfo.warpSize << std::endl;
    int BLOCK_SIZE = sgl::lastPowerOfTwo(bestBlockSize);
    while (BLOCK_SIZE > int(deviceCoresInfo.warpSize)
           && sgl::iceil(int(M), BLOCK_SIZE) * 2 < int(deviceCoresInfo.numMultiprocessors)) {
        BLOCK_SIZE /= 2;
    }

    /*
     * Estimated time: M * N * log2(N).
     * On a RTX 3090, M = 1.76 * 10^6 and N = 100 takes approx. 1s.
     */
    const uint32_t numCudaCoresRtx3090 = 10496;
    double factorM = double(M) / (1.76 * 1e6);
    double factorN = double(N) / 100.0 * std::log2(double(N) / 100.0 + 1.0);
    double factorCudaCores = double(numCudaCoresRtx3090) / double(deviceCoresInfo.numCudaCoresTotal);
    auto batchCount = uint32_t(std::ceil(factorM * factorN * factorCudaCores));

    // __global__ void mutualInformationKraskov(
    //        cudaTextureObject_t* scalarFieldEnsembles, float* __restrict__ miArray,
    //        uint32_t xs, uint32_t ys, uint32_t zs, uint3 referencePointIdx,
    //        const uint32_t batchOffset, const uint32_t batchSize) {
    CUdeviceptr scalarFieldEnsembles = ensembleTextureArrayCu;
    CUdeviceptr miArray = outputImageBufferCu;
    auto batchSize = uint32_t(M);
    if (batchCount == 1) {
        auto batchOffset = uint32_t(0);
        void* kernelParameters[] = {
                &scalarFieldEnsembles, &miArray, &xs, &ys, &zs, &referencePointIndex.x,
                &batchOffset, &batchSize
        };
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                kernelCache->kernel, sgl::iceil(int(M), BLOCK_SIZE), 1, 1, //< Grid size.
                BLOCK_SIZE, 1, 1, //< Block size.
                0, //< Dynamic shared memory size.
                stream,
                kernelParameters, //< Kernel parameters.
                nullptr
        ), "Error in cuLaunchKernel: "); //< Extra (empty).
    } else {
        auto batchSizeLocal = sgl::uiceil(uint32_t(M), batchCount);
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            auto batchOffset = batchSizeLocal * batchIdx;
            if (batchOffset + batchSizeLocal > uint32_t(M)) {
                batchSizeLocal = uint32_t(M) - batchSizeLocal;
            }
            void* kernelParameters[] = {
                    &scalarFieldEnsembles, &miArray, &xs, &ys, &zs, &referencePointIndex.x,
                    &batchOffset, &batchSize
            };
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                    kernelCache->kernel, sgl::iceil(int(batchSizeLocal), BLOCK_SIZE), 1, 1, //< Grid size.
                    BLOCK_SIZE, 1, 1, //< Block size.
                    0, //< Dynamic shared memory size.
                    stream,
                    kernelParameters, //< Kernel parameters.
                    nullptr
            ), "Error in cuLaunchKernel: "); //< Extra (empty).
        }
    }

    deviceCacheEntry->getImageCudaExternalMemory()->memcpyCudaDtoA3DAsync(outputImageBufferCu, stream);

    cudaFinishedSemaphore->signalSemaphoreCuda(stream, timelineValue);
    cudaFinishedSemaphore->setWaitSemaphoreValue(timelineValue);
    sgl::vk::CommandBufferPtr postRenderCommandBuffer = postRenderCommandBuffers.at(frameIndex);
    renderer->pushCommandBuffer(postRenderCommandBuffer);
    renderer->beginCommandBuffer();
    postRenderCommandBuffer->pushWaitSemaphore(
            cudaFinishedSemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}



SpearmanReferenceRankComputePass::SpearmanReferenceRankComputePass(
        sgl::vk::Renderer* renderer, sgl::vk::BufferPtr uniformBuffer)
        : ComputePass(renderer), uniformBuffer(std::move(uniformBuffer)) {
}

void SpearmanReferenceRankComputePass::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
}

void SpearmanReferenceRankComputePass::setEnsembleMemberCount(int ensembleMemberCount) {
    cachedEnsembleMemberCount = ensembleMemberCount;
    referenceRankBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(float) * cachedEnsembleMemberCount,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    setShaderDirty();
}

void SpearmanReferenceRankComputePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews == _ensembleImageViews) {
        return;
    }
    ensembleImageViews = _ensembleImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    }
}

void SpearmanReferenceRankComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(cachedEnsembleMemberCount)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "SpearmanRankCorrelation.Reference.Compute" }, preprocessorDefines);
}

void SpearmanReferenceRankComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticBuffer(referenceRankBuffer, "ReferenceRankBuffer");
}

void SpearmanReferenceRankComputePass::_render() {
    renderer->dispatch(
            computeData,
            sgl::iceil(volumeData->getGridSizeX(), computeBlockSizeX),
            sgl::iceil(volumeData->getGridSizeY(), computeBlockSizeY),
            sgl::iceil(volumeData->getGridSizeZ(), computeBlockSizeZ));
}
