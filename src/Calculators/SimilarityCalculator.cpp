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

#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>
#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
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

    if (isNewData) {
        referencePointIndex.x = volumeData->getGridSizeX() / 2;
        referencePointIndex.y = volumeData->getGridSizeY() / 2;
        referencePointIndex.z = volumeData->getGridSizeZ() / 2;
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);

        fieldIndex = 0;
        fieldIndexGui = 0;
        scalarFieldNames = {};
        scalarFieldIndexArray = {};

        std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
        for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
            if (scalarFieldNamesNew.at(i) != getOutputFieldName()) {
                scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
                scalarFieldIndexArray.push_back(i);
            }
        }
    }
}

void EnsembleSimilarityCalculator::update(float dt) {
    // TODO: Use mouse for selection of reference point.
}

void EnsembleSimilarityCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addCombo(
            "Scalar Field", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
        dirty = true;
    }

    // TODO: Replace with referencePointSelectionRenderer.
    bool inputFinished[3];
    inputFinished[0] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (X)", &referencePointIndex[0], 0, volumeData->getGridSizeX())
            == ImGui::EditMode::INPUT_FINISHED;
    inputFinished[1] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Y)", &referencePointIndex[1], 0, volumeData->getGridSizeY())
            == ImGui::EditMode::INPUT_FINISHED;
    inputFinished[2] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Z)", &referencePointIndex[2], 0, volumeData->getGridSizeZ())
            == ImGui::EditMode::INPUT_FINISHED;
    if (inputFinished[0] || inputFinished[1] || inputFinished[2]) {
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        dirty = true;
    }
}

void PccCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
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

    //size_t referencePointIdx =
    //        size_t(referencePointIndex.x) * size_t(referencePointIndex.y) * size_t(referencePointIndex.z);
    size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
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
        if (n == 1) {
            buffer[gridPointIdx] = 1.0f;
            continue;
        }
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

    delete[] referenceValues;
}
