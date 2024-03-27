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
#include "Calculators/CorrelationCalculator.hpp"
#include "Volume/VolumeData.hpp"
#include "../CorrelationCache.hpp"
#include "../Sampling.hpp"
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

    if (_volumeData->getCurrentTimeStepIdx() != cachedTimeStepIdx
            || _volumeData->getCurrentEnsembleIdx() != cachedEnsembleIdx) {
        cachedTimeStepIdx = _volumeData->getCurrentTimeStepIdx();
        cachedEnsembleIdx = _volumeData->getCurrentEnsembleIdx();
        dataDirty = true;
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

void DistributionSimilarityRenderer::samplePointPositions() {
    auto xs = volumeData->getGridSizeX();
    auto ys = volumeData->getGridSizeY();
    auto zs = volumeData->getGridSizeZ();

    if (usePredicateField && predicateFieldIdxGui != 0) {
        int ensembleIdx = -1, timeStepIdx = -1;
        VolumeData::HostCacheEntry fieldEntry = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(predicateFieldIdxGui), timeStepIdx, ensembleIdx);
        const float* field = fieldEntry->data<float>();

        if (samplingPattern == SamplingPattern::ALL) {
            int numGridPoints = xs * ys * zs;
            pointGridPositions.clear();
            pointGridPositions.reserve(numGridPoints);
            for (int ptIdx = 0; ptIdx < numGridPoints; ptIdx++) {
                int xr = ptIdx % xs;
                int yr = (ptIdx / xs) % ys;
                int zr = ptIdx / (xs * ys);
                if (field[ptIdx] > 0.5f) {
                    pointGridPositions.emplace_back(xr, yr, zr);
                }
            }
        } else if (samplingPattern == SamplingPattern::QUASIRANDOM_PLASTIC) {
            SampleGenerator3D* sampleGenerator = createSampleGenerator3D(
                    SamplingMethodType::QUASIRANDOM_PLASTIC, false);
            pointGridPositions.clear();
            pointGridPositions.reserve(numRandomPoints);
            int sampleIdx = 0;
            // Use check "sampleIdx < 100 * numRandomPoints" to make sure rejection sampling terminates at some point.
            while (int(pointGridPositions.size()) < numRandomPoints && sampleIdx < 100 * numRandomPoints) {
                glm::vec3 sample = sampleGenerator->next();
                int xr = std::clamp(int(std::round(sample[0] * float(xs) - 0.5f)), 0, xs - 1);
                int yr = std::clamp(int(std::round(sample[1] * float(ys) - 0.5f)), 0, ys - 1);
                int zr = std::clamp(int(std::round(sample[2] * float(zs) - 0.5f)), 0, zs - 1);
                int gridIdx = IDXS(xr, yr, zr);
                if (field[gridIdx] > 0.5f) {
                    pointGridPositions.emplace_back(xr, yr, zr);
                }
                sampleIdx++;
            }
            delete sampleGenerator;
        }
    } else {
        if (samplingPattern == SamplingPattern::ALL) {
            int numGridPoints = xs * ys * zs;
            pointGridPositions.clear();
            pointGridPositions.resize(numGridPoints);
            for (int ptIdx = 0; ptIdx < numGridPoints; ptIdx++) {
                int xr = ptIdx % xs;
                int yr = (ptIdx / xs) % ys;
                int zr = ptIdx / (xs * ys);
                pointGridPositions.at(ptIdx) = glm::ivec3(xr, yr, zr);
            }
        } else if (samplingPattern == SamplingPattern::QUASIRANDOM_PLASTIC) {
            int numGridPoints = numRandomPoints;
            auto* samples = new float[numGridPoints * 3];
            generateSamples3D(samples, numGridPoints, SamplingMethodType::QUASIRANDOM_PLASTIC, false);
            pointGridPositions.clear();
            pointGridPositions.resize(numGridPoints);
            for (int ptIdx = 0; ptIdx < numGridPoints; ptIdx++) {
                int xr = std::clamp(int(std::round(samples[ptIdx * 3 + 0] * float(xs) - 0.5f)), 0, xs - 1);
                int yr = std::clamp(int(std::round(samples[ptIdx * 3 + 1] * float(ys) - 0.5f)), 0, ys - 1);
                int zr = std::clamp(int(std::round(samples[ptIdx * 3 + 2] * float(zs) - 0.5f)), 0, zs - 1);
                pointGridPositions.at(ptIdx) = glm::ivec3(xr, yr, zr);
            }
            delete[] samples;
        }
    }
}

void DistributionSimilarityRenderer::computeFeatureVectorsCorrelation(
        int& numVectors, int& numFeatures, double*& featureVectorArray) {
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

    samplePointPositions();
    auto numGridPoints = int(pointGridPositions.size());

    int localRadius = std::max(neighborhoodRadius, 1);
    int featureDimLen = 2 * localRadius + 1;
    numFeatures = featureDimLen * featureDimLen * featureDimLen;
    numVectors = numGridPoints;
    featureVectorArray = new double[numFeatures * numVectors];

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
            auto gridPt = pointGridPositions.at(gridPointIdx);
            //int xr = gridPointIdx % xs;
            //int yr = (gridPointIdx / xs) % ys;
            //int zr = gridPointIdx / (xs * ys);
            int xr = gridPt.x;
            int yr = gridPt.y;
            int zr = gridPt.z;
            int gridIdx = IDXS(xr, yr, zr);
            for (int c = 0; c < cs; c++) {
                //X[c] = fields.at(c)[gridPointIdx];
                X[c] = fields.at(c)[gridIdx];
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
}

void DistributionSimilarityRenderer::computeFeatureVectorsGridCellsEnsembleValues(
        int& numVectors, int& numFeatures, double*& featureVectorArray) {
    auto xs = volumeData->getGridSizeX();
    auto ys = volumeData->getGridSizeY();
    //auto zs = volumeData->getGridSizeZ();
    int cs = getCorrelationMemberCount();

    int ensembleIdx = -1, timeStepIdx = -1;
    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(
                scalarFieldNames.at(useSeparateFields ? fieldIndex2Gui : fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
        const float *field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
    }

    samplePointPositions();
    auto numGridPoints = int(pointGridPositions.size());

    numFeatures = cs;
    numVectors = numGridPoints;
    featureVectorArray = new double[numFeatures * numVectors];

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel shared(xs, ys, cs, numGridPoints, numFeatures, featureVectorArray, fields) default(none)
#endif
    {
        #pragma omp for
        for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
            auto gridPt = pointGridPositions.at(gridPointIdx);
            int xr = gridPt.x;
            int yr = gridPt.y;
            int zr = gridPt.z;
            int gridIdx = IDXS(xr, yr, zr);
            float val;
            for (int c = 0; c < cs; c++) {
                val = fields.at(c)[gridIdx];
                if (std::isnan(val)) {
                    val = 0.0f;
                }
                size_t offsetWrite = size_t(gridPointIdx) * size_t(numFeatures) + size_t(c);
                featureVectorArray[offsetWrite] = val;
            }
        }
#ifdef USE_TBB
        });
#else
    }
#endif
}

void DistributionSimilarityRenderer::computeFeatureVectorsEnsembleMembersGridCellValues(
        int& numVectors, int& numFeatures, double*& featureVectorArray) {
    auto xs = volumeData->getGridSizeX();
    auto ys = volumeData->getGridSizeY();
    //auto zs = volumeData->getGridSizeZ();
    int cs = getCorrelationMemberCount();

    int ensembleIdx = -1, timeStepIdx = -1;
    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(
                scalarFieldNames.at(useSeparateFields ? fieldIndex2Gui : fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
        const float *field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
    }

    samplePointPositions();
    auto numGridPoints = int(pointGridPositions.size());

    numFeatures = numGridPoints;
    numVectors = cs;
    featureVectorArray = new double[numFeatures * numVectors];

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numGridPoints), [&](auto const& r) {
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
#pragma omp parallel shared(xs, ys, cs, numGridPoints, numFeatures, featureVectorArray, fields) default(none)
#endif
    {
#pragma omp for
        for (int gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
            auto gridPt = pointGridPositions.at(gridPointIdx);
            int xr = gridPt.x;
            int yr = gridPt.y;
            int zr = gridPt.z;
            int gridIdx = IDXS(xr, yr, zr);
            float val;
            for (int c = 0; c < cs; c++) {
                val = fields.at(c)[gridIdx];
                if (std::isnan(val)) {
                    val = 0.0f;
                }
                size_t offsetWrite = size_t(c) * size_t(numFeatures) + size_t(gridPointIdx);
                featureVectorArray[offsetWrite] = val;
            }
        }
#ifdef USE_TBB
        });
#else
    }
#endif
}

void DistributionSimilarityRenderer::recomputeCorrelationMatrix() {
    dataDirty = false;
    if (fieldIndexGui == 0 || (useSeparateFields && fieldIndex2Gui == 0)) {
        parentDiagram->setPointData({});
        return;
    }

    int numVectors = 0;
    int numFeatures = 0;
    double* featureVectorArray = nullptr;
    if (distributionAnalysisMode == DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR) {
        computeFeatureVectorsCorrelation(numVectors, numFeatures, featureVectorArray);
    } else if (distributionAnalysisMode == DistributionAnalysisMode::GRID_CELL_MEMBER_VALUE_VECTOR) {
        computeFeatureVectorsGridCellsEnsembleValues(numVectors, numFeatures, featureVectorArray);
    } else if (distributionAnalysisMode == DistributionAnalysisMode::MEMBER_GRID_CELL_VALUE_VECTOR) {
        computeFeatureVectorsEnsembleMembersGridCellValues(numVectors, numFeatures, featureVectorArray);
    }
    auto* outputPoints = new double[2 * numVectors];

    TSNE::run(
            featureVectorArray, numVectors, numFeatures, outputPoints, 2,
            double(tsneSettings.perplexity), double(tsneSettings.theta), tsneSettings.randomSeed, false,
            tsneSettings.maxIter, tsneSettings.stopLyingIter, tsneSettings.momSwitchIter);

    std::vector<glm::vec2> points(numVectors);
    for (int ptIdx = 0; ptIdx < numVectors; ptIdx++) {
        points.at(ptIdx) = glm::vec2(float(outputPoints[ptIdx * 2]), float(outputPoints[ptIdx * 2 + 1]));
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
    if (distributionAnalysisMode != DistributionAnalysisMode::MEMBER_GRID_CELL_VALUE_VECTOR) {
        auto selectedPointIdxNew = parentDiagram->getSelectedPointIdx();
        if (selectedPointIdxNew >= 0) {
            selectedPointIdx = selectedPointIdxNew;
            auto correlationCalculators = volumeData->getCorrelationCalculatorsUsed();
            for (auto& calculator : correlationCalculators) {
                if (calculator->getIsEnsembleMode() != isEnsembleMode) {
                    continue;
                }
                calculator->setReferencePoint(pointGridPositions.at(selectedPointIdx));
            }
        }
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

    if (volumeData && distributionAnalysisMode == DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR
            && scalarFieldNames.size() > 1 && propertyEditor.addCheckbox("Two Fields Mode", &useSeparateFields)) {
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

    if (volumeData && distributionAnalysisMode == DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR
            && useSeparateFields && isEnsembleMode && volumeData->getTimeStepCount() > 1
            && propertyEditor.addCheckbox("Time Lag Correlations", &useTimeLagCorrelations)) {
        timeLagTimeStepIdx = volumeData->getCurrentTimeStepIdx();
        if (!useTimeLagCorrelations) {
            clearFieldDeviceData();
            setRecomputeFlag();
        }
    }

    if (propertyEditor.addCombo(
            "Use Correlation Mode", (int*)&distributionAnalysisMode, DISTRIBUTION_ANALYSIS_MODE_NAMES,
            IM_ARRAYSIZE(DISTRIBUTION_ANALYSIS_MODE_NAMES))) {
        if (distributionAnalysisMode != DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR) {
            if (useSeparateFields) {
                if (fieldIndex2 != 0xFFFFFF) {
                    volumeData->releaseScalarField(this, fieldIndex2);
                }
                useSeparateFields = false;
            }
            if (useTimeLagCorrelations) {
                useTimeLagCorrelations = false;
            }
        }
        clearFieldDeviceData();
        setRecomputeFlag();
    }

    if (distributionAnalysisMode == DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR) {
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

        if (propertyEditor.addSliderIntEdit(
                "Neighborhood Radius", &neighborhoodRadius, 1, 9) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
    }

    if (propertyEditor.addCombo(
            "Sampling Pattern", (int*)&samplingPattern, SAMPLING_PATTERN_NAMES, IM_ARRAYSIZE(SAMPLING_PATTERN_NAMES))) {
        setRecomputeFlag();
    }
    if (samplingPattern != SamplingPattern::ALL) {
        if (propertyEditor.addSliderIntEdit(
                "#Sampled Points", &numRandomPoints, 10, 10000) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
    }
    if (propertyEditor.addCheckbox("Use Predicate Field", &usePredicateField)) {
        if (predicateFieldIdx != 0xFFFFFF) {
            if (usePredicateField) {
                volumeData->acquireScalarField(this, predicateFieldIdx);
            } else {
                volumeData->releaseScalarField(this, predicateFieldIdx);
            }
        }
        clearFieldDeviceData();
        setRecomputeFlag();
    }
    if (usePredicateField && propertyEditor.addCombo(
            "Predicate Field", &predicateFieldIdxGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        clearFieldDeviceData();
        if (predicateFieldIdx != 0xFFFFFF) {
            volumeData->releaseScalarField(this, predicateFieldIdx);
        }
        predicateFieldIdx = int(scalarFieldIndexArray.at(predicateFieldIdxGui));
        if (predicateFieldIdx != 0xFFFFFF) {
            volumeData->acquireScalarField(this, predicateFieldIdx);
        }
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

    if (propertyEditor.beginNode("t-SNE settings")) {
        if (propertyEditor.addSliderFloatEdit(
                "Perplexity", &tsneSettings.perplexity, 5.0f, 50.0f) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
        if (propertyEditor.addSliderFloatEdit(
                "Theta", &tsneSettings.theta, 0.1f, 1.0f) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
        if (propertyEditor.addSliderIntEdit(
                "Random Seed", &tsneSettings.randomSeed, -1, 50) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
        if (propertyEditor.addSliderIntEdit(
                "#Iterations (max)", &tsneSettings.maxIter, 1, 1000) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
        if (propertyEditor.addSliderIntEdit(
                "#Stop Lying Iterations", &tsneSettings.stopLyingIter, 0, 1000) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
        if (propertyEditor.addSliderIntEdit(
                "#Moment Switch Iterations", &tsneSettings.momSwitchIter, 0, 1000) == ImGui::EditMode::INPUT_FINISHED) {
            setRecomputeFlag();
        }
        propertyEditor.endNode();
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

    bool dataAnalysisModeChanged = false;
    bool useCorrelationMode = true;
    if (settings.getValueOpt("use_correlation_mode", useCorrelationMode)) {
        distributionAnalysisMode =
                useCorrelationMode ? DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR
                                   : DistributionAnalysisMode::GRID_CELL_MEMBER_VALUE_VECTOR;
        dataAnalysisModeChanged = true;
    }
    std::string distributionAnalysisModeString;
    if (settings.getValueOpt("distribution_analysis_mode", distributionAnalysisModeString)) {
        for (int i = 0; i < IM_ARRAYSIZE(DISTRIBUTION_ANALYSIS_MODE_NAMES); i++) {
            if (distributionAnalysisModeString == DISTRIBUTION_ANALYSIS_MODE_NAMES[i]) {
                distributionAnalysisMode = DistributionAnalysisMode(i);
                break;
            }
        }
        dataAnalysisModeChanged = true;
    }
    if (dataAnalysisModeChanged) {
        if (distributionAnalysisMode != DistributionAnalysisMode::GRID_CELL_NEIGHBORHOOD_CORRELATION_VECTOR) {
            if (useSeparateFields) {
                if (fieldIndex2 != 0xFFFFFF) {
                    volumeData->releaseScalarField(this, fieldIndex2);
                }
                useSeparateFields = false;
            }
            if (useTimeLagCorrelations) {
                useTimeLagCorrelations = false;
            }
        }
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

    if (settings.getValueOpt("neighborhood_radius", neighborhoodRadius)) {
        setRecomputeFlag();
    }

    std::string samplingPatternName;
    if (settings.getValueOpt("sampling_pattern", samplingPatternName)) {
        for (int i = 0; i < IM_ARRAYSIZE(SAMPLING_PATTERN_NAMES); i++) {
            if (samplingPatternName == SAMPLING_PATTERN_NAMES[i]) {
                samplingPattern = SamplingPattern(i);
                break;
            }
        }
        setRecomputeFlag();
    }
    if (settings.getValueOpt("num_sampled_points", numRandomPoints)) {
        setRecomputeFlag();
    }
    if (settings.getValueOpt("use_predicate_field", usePredicateField)) {
        if (predicateFieldIdx != 0xFFFFFF) {
            if (usePredicateField) {
                volumeData->acquireScalarField(this, predicateFieldIdx);
            } else {
                volumeData->releaseScalarField(this, predicateFieldIdx);
            }
        }
        clearFieldDeviceData();
        setRecomputeFlag();
    }
    if (settings.getValueOpt("predicate_field_idx", predicateFieldIdxGui)) {
        clearFieldDeviceData();
        if (predicateFieldIdx != 0xFFFFFF) {
            volumeData->releaseScalarField(this, predicateFieldIdx);
        }
        predicateFieldIdx = int(scalarFieldIndexArray.at(predicateFieldIdxGui));
        if (predicateFieldIdx != 0xFFFFFF) {
            volumeData->acquireScalarField(this, predicateFieldIdx);
        }
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

    if (settings.getValueOpt("tsne_perplexity", tsneSettings.perplexity)) {
        setRecomputeFlag();
    }
    if (settings.getValueOpt("tsne_theta", tsneSettings.theta)) {
        setRecomputeFlag();
    }
    if (settings.getValueOpt("tsne_random_seed", tsneSettings.randomSeed)) {
        setRecomputeFlag();
    }
    if (settings.getValueOpt("tsne_max_iter", tsneSettings.maxIter)) {
        setRecomputeFlag();
    }
    if (settings.getValueOpt("tsne_stop_lying_iter", tsneSettings.stopLyingIter)) {
        setRecomputeFlag();
    }
    if (settings.getValueOpt("tsne_mom_switch_iter", tsneSettings.momSwitchIter)) {
        setRecomputeFlag();
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

    settings.addKeyValue("distribution_analysis_mode", DISTRIBUTION_ANALYSIS_MODE_NAMES[int(distributionAnalysisMode)]);

    settings.addKeyValue("correlation_measure_type", CORRELATION_MEASURE_TYPE_IDS[int(correlationMeasureType)]);
    //const char* const choices[] = {
    //        "CPU", "Vulkan", "CUDA"
    //};
    //settings.addKeyValue("device", choices[!useGpu ? 0 : (!useCuda ? 1 : 2)]);
    settings.addKeyValue("calculate_absolute_value", calculateAbsoluteValue);
    settings.addKeyValue("mi_bins", numBins);
    settings.addKeyValue("kmi_neighbors", k);

    settings.addKeyValue("neighborhood_radius", neighborhoodRadius);

    settings.addKeyValue("sampling_pattern", SAMPLING_PATTERN_NAMES[int(samplingPattern)]);
    settings.addKeyValue("num_sampled_points", numRandomPoints);
    settings.addKeyValue("use_predicate_field", usePredicateField);
    settings.addKeyValue("predicate_field_idx", predicateFieldIdxGui);

    settings.addKeyValue("point_size", pointSize);
    settings.addKeyValue("point_color", pointColor.getFloatColorRGBA());

    // t-SNE settings.
    settings.addKeyValue("tsne_perplexity", tsneSettings.perplexity);
    settings.addKeyValue("tsne_theta", tsneSettings.theta);
    settings.addKeyValue("tsne_random_seed", tsneSettings.randomSeed);
    settings.addKeyValue("tsne_max_iter", tsneSettings.maxIter);
    settings.addKeyValue("tsne_stop_lying_iter", tsneSettings.stopLyingIter);
    settings.addKeyValue("tsne_mom_switch_iter", tsneSettings.momSwitchIter);

    // No vector widget settings for now.
}
