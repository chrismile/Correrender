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

#include <iostream>
#include <queue>
#include <stack>
#include <chrono>

//#define USE_TBB

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#elif defined(_OPENMP)
#include <omp.h>
#endif

//#ifndef USE_TBB
//#include <execution>
//#include <algorithm>
//#endif

#include <Math/Math.hpp>
#include <Utils/Parallel/Reduction.hpp>
#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "Loaders/DataSet.hpp"
#include "Calculators/Correlation.hpp"
#include "Calculators/CorrelationCalculator.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Volume/VolumeData.hpp"
#include "BSpline.hpp"
#include "HEBChart.hpp"

HEBChart::~HEBChart() {
    if (computeRenderer) {
        correlationComputePass = {};
        sgl::vk::Device* device = computeRenderer->getDevice();
        device->freeCommandBuffer(commandPool, commandBuffer);
        delete computeRenderer;
    }
}

void HEBChart::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;
    for (auto& fieldData : fieldDataArray) {
        if (fieldData->selectedFieldIdx >= int(volumeData->getFieldNamesBase(FieldType::SCALAR).size())) {
            dataDirty = true;
        }
    }
    if (isNewData) {
        xs = volumeData->getGridSizeX();
        ys = volumeData->getGridSizeY();
        zs = volumeData->getGridSizeZ();
        r0 = r1 = GridRegion(0, 0, 0, xs, ys, zs);
        regionsEqual = true;
        xsd0 = xsd1 = sgl::iceil(xs, dfx);
        ysd0 = ysd1 = sgl::iceil(ys, dfy);
        zsd0 = zsd1 = sgl::iceil(zs, dfz);
        resetSelectedPrimitives();
    }
}

int HEBChart::getCorrelationMemberCount() {
    return isEnsembleMode ? volumeData->getEnsembleMemberCount() : volumeData->getTimeStepCount();
}

VolumeData::HostCacheEntry HEBChart::getFieldEntryCpu(const std::string& fieldName, int fieldIdx) {
    VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, fieldName, isEnsembleMode ? -1 : fieldIdx, isEnsembleMode ? fieldIdx : -1);
    return ensembleEntryField;
}

DeviceCacheEntry HEBChart::getFieldEntryDevice(const std::string& fieldName, int fieldIdx) {
    VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, fieldName, isEnsembleMode ? -1 : fieldIdx, isEnsembleMode ? fieldIdx : -1);
    return ensembleEntryField;
}

std::pair<float, float> HEBChart::getMinMaxScalarFieldValue(const std::string& fieldName, int fieldIdx) {
    return volumeData->getMinMaxScalarFieldValue(
            fieldName, isEnsembleMode ? -1 : fieldIdx, isEnsembleMode ? fieldIdx : -1);
}

void HEBChart::setRegions(const std::pair<GridRegion, GridRegion>& _rs) {
    r0 = _rs.first;
    r1 = _rs.second;
    regionsEqual = r0 == r1;
    xsd0 = sgl::iceil(r0.xsr, dfx);
    ysd0 = sgl::iceil(r0.ysr, dfy);
    zsd0 = sgl::iceil(r0.zsr, dfz);
    xsd1 = sgl::iceil(r1.xsr, dfx);
    ysd1 = sgl::iceil(r1.ysr, dfy);
    zsd1 = sgl::iceil(r1.zsr, dfz);
}

void HEBChart::resetSelectedPrimitives() {
    hoveredPointIdx = -1;
    clickedPointIdx = -1;
    selectedPointIndices[0] = -1;
    selectedPointIndices[1] = -1;
    hoveredLineIdx = -1;
    clickedLineIdx = -1;
    selectedLineIdx = -1;
    clickedLineIdxOld = -1;
    clickedPointIdxOld = -1;
}

void HEBChart::clearScalarFields() {
    fieldDataArray.clear();
}

void HEBChart::addScalarField(int _selectedFieldIdx, const std::string& _scalarFieldName) {
    auto fieldData = std::make_shared<HEBChartFieldData>(this, &desaturateUnselectedRing);
    fieldData->selectedFieldIdx = _selectedFieldIdx;
    fieldData->selectedScalarFieldName = _scalarFieldName;
    fieldData->separateColorVarianceAndCorrelation = separateColorVarianceAndCorrelation;
    fieldData->setColorMapVariance(colorMapVariance);

    bool foundInsertionPosition = false;
    for (size_t i = 0; i < fieldDataArray.size(); i++) {
        if (fieldDataArray.at(i)->selectedFieldIdx > _selectedFieldIdx) {
            fieldDataArray.insert(fieldDataArray.begin() + ptrdiff_t(i), fieldData);
            foundInsertionPosition = true;
            break;
        }
    }
    if (!foundInsertionPosition) {
        fieldDataArray.push_back(fieldData);
    }
    computeColorLegendHeight();
    dataDirty = true;
}

void HEBChart::removeScalarField(int _selectedFieldIdx, bool shiftIndicesBack) {
    for (size_t i = 0; i < fieldDataArray.size(); ) {
        if (fieldDataArray.at(i)->selectedFieldIdx == _selectedFieldIdx) {
            fieldDataArray.erase(fieldDataArray.begin() + ptrdiff_t(i));
            if (shiftIndicesBack) {
                continue;
            } else {
                break;
            }
        } else if (fieldDataArray.at(i)->selectedFieldIdx > _selectedFieldIdx) {
            fieldDataArray.at(i)->selectedFieldIdx--;
        }
        i++;
    }
    computeColorLegendHeight();
    dataDirty = true;
}

void HEBChart::setIsEnsembleMode(bool _isEnsembleMode) {
    isEnsembleMode = _isEnsembleMode;
    dataDirty = true;
}

void HEBChart::setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType) {
    correlationMeasureType = _correlationMeasureType;
    dataDirty = true;
}

void HEBChart::setNumBins(int _numBins) {
    if (numBins != _numBins) {
        numBins = _numBins;
        dataDirty = true;
    }
}

void HEBChart::setKraskovNumNeighbors(int _k) {
    if (k != _k) {
        k = _k;
        dataDirty = true;
    }
}

void HEBChart::setSamplingMethodType(SamplingMethodType _samplingMethodType) {
    samplingMethodType = _samplingMethodType;
    dataDirty = true;
}

void HEBChart::setNumSamples(int _numSamples) {
    numSamples = _numSamples;
    dataDirty = true;
}

void HEBChart::setDownscalingFactors(int _dfx, int _dfy, int _dfz) {
    dfx = _dfx;
    dfy = _dfy;
    dfz = _dfz;
    dataDirty = true;
}

void HEBChart::setUse2DField(bool _use2dField) {
    use2dField = _use2dField;
    dataDirty = true;
}

void HEBChart::setUseCorrelationComputationGpu(bool _useGpu) {
    useCorrelationComputationGpu = _useGpu;
    dataDirty = true;
}

void HEBChart::setShowVariablesForFieldIdxOnly(int _limitedFieldIdx) {
    limitedFieldIdx = _limitedFieldIdx;
    dataDirty = true;
}

void HEBChart::setOctreeMethod(OctreeMethod _octreeMethod) {
    octreeMethod = _octreeMethod;
    dataDirty = true;
}


glm::vec2 HEBChart::getCorrelationRangeTotal() {
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
            || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        int cs = getCorrelationMemberCount();
        correlationRangeTotal.x = 0.0f;
        correlationRangeTotal.y = computeMaximumMutualInformationKraskov(k, cs);
    } else {
        correlationRangeTotal.x = -1.0f;
        correlationRangeTotal.y = 1.0f;
    }
    return correlationRangeTotal;
}

glm::ivec2 HEBChart::getCellDistanceRangeTotal() {
    cellDistanceRangeTotal.x = 0;
    //cellDistanceRangeTotal.y = xsd + ysd + zsd; //< Manhattan distance.
    glm::vec3 pti(0, 0, 0);
    glm::vec3 ptj0(xsd0 - 1, ysd0 - 1, zsd0 - 1);
    glm::vec3 ptj1(xsd1 - 1, ysd1 - 1, zsd1 - 1);
    cellDistanceRangeTotal.y = int(std::ceil(std::max(glm::length(pti - ptj0), glm::length(pti - ptj1))));
    return cellDistanceRangeTotal;
}

void HEBChart::setCorrelationRange(const glm::vec2& _range) {
    correlationRange = _range;
    dataDirty = true;
}

void HEBChart::setCellDistanceRange(const glm::ivec2& _range) {
    cellDistanceRange = _range;
    dataDirty = true;
}


void HEBChart::computeDownscaledField(
        HEBChartFieldData* fieldData, int idx, std::vector<float*>& downscaledFields) {
    int cs = getCorrelationMemberCount();
    int xsd = idx == 0 ? xsd0 : xsd1;
    int ysd = idx == 0 ? ysd0 : ysd1;
    int zsd = idx == 0 ? zsd0 : zsd1;
    int numPoints = xsd * ysd * zsd;
    GridRegion r = idx == 0 ? r0 : r1;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(fieldData->selectedScalarFieldName, fieldIdx);
        const float* field = fieldEntry->data<float>();
        auto* downscaledField = new float[numPoints];

        if (!use2dField) {
            for (int zd = 0; zd < zsd; zd++) {
                for (int yd = 0; yd < ysd; yd++) {
                    for (int xd = 0; xd < xsd; xd++) {
                        float valueMean = 0.0f;
                        int numValid = 0;
                        for (int zo = 0; zo < dfz; zo++) {
                            for (int yo = 0; yo < dfy; yo++) {
                                for (int xo = 0; xo < dfx; xo++) {
                                    int x = r.xoff + xd * dfx + xo;
                                    int y = r.yoff + yd * dfy + yo;
                                    int z = r.zoff + zd * dfz + zo;
                                    if (x <= r.xmax && y <= r.ymax && z <= r.zmax) {
                                        float val = field[IDXS(x, y, z)];
                                        if (!std::isnan(val)) {
                                            valueMean += val;
                                            numValid++;
                                        }
                                    }
                                }
                            }
                        }
                        if (numValid > 0) {
                            valueMean = valueMean / float(numValid);
                        } else {
                            valueMean = std::numeric_limits<float>::quiet_NaN();
                        }
                        downscaledField[IDXSD(xd, yd, zd)] = valueMean;
                    }
                }
            }
        } else {
            int zCenter = zs / 2;
            for (int yd = 0; yd < ysd; yd++) {
                for (int xd = 0; xd < xsd; xd++) {
                    float valueMean = 0.0f;
                    int numValid = 0;
                    for (int yo = 0; yo < dfy; yo++) {
                        for (int xo = 0; xo < dfx; xo++) {
                            int x = r.xoff + xd * dfx + xo;
                            int y = r.yoff + yd * dfy + yo;
                            if (x <= r.xmax && y <= r.ymax) {
                                float val = field[IDXS(x, y, zCenter)];
                                if (!std::isnan(val)) {
                                    valueMean += val;
                                    numValid++;
                                }
                            }
                        }
                    }
                    if (numValid > 0) {
                        valueMean = valueMean / float(numValid);
                    } else {
                        valueMean = std::numeric_limits<float>::quiet_NaN();
                    }
                    downscaledField[IDXSD(xd, yd, 0)] = valueMean;
                }
            }
        }

        downscaledFields.at(fieldIdx) = downscaledField;
    }
}

void HEBChart::computeDownscaledFieldVariance(HEBChartFieldData* fieldData, int idx) {
    // Compute the standard deviation inside the downscaled grids.
    int cs = getCorrelationMemberCount();
    int xsd = idx == 0 ? xsd0 : xsd1;
    int ysd = idx == 0 ? ysd0 : ysd1;
    int zsd = idx == 0 ? zsd0 : zsd1;
    int numPoints = xsd * ysd * zsd;
    GridRegion r = idx == 0 ? r0 : r1;
    auto& pointToNodeIndexMap = idx == 0 ? pointToNodeIndexMap0 : pointToNodeIndexMap1;
    fieldData->leafStdDevArray.resize(xsd0 * ysd0 * zsd0 + (idx == 0 ? 0 : + xsd1 * ysd1 * zsd1));

    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(fieldData->selectedScalarFieldName, fieldIdx);
        const float* field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
    }

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numPoints), [&](auto const& r) {
            for (auto pointIdx = r.begin(); pointIdx != r.end(); pointIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel for default(none) shared(fields, fieldData, pointToNodeIndexMap, xsd, ysd, zsd, r, numPoints, cs)
#endif
    for (int pointIdx = 0; pointIdx < numPoints; pointIdx++) {
#endif
        int gridNumValid = 0;
        float gridVarianceSum = 0.0f;
        int xd = pointIdx % xsd;
        int yd = (pointIdx / xsd) % ysd;
        int zd = pointIdx / (xsd * ysd);

        if (!use2dField) {
            for (int zo = 0; zo < dfz; zo++) {
                for (int yo = 0; yo < dfy; yo++) {
                    for (int xo = 0; xo < dfx; xo++) {
                        int x = r.xoff + xd * dfx + xo;
                        int y = r.yoff + yd * dfy + yo;
                        int z = r.zoff + zd * dfz + zo;
                        if (x <= r.xmax && y <= r.ymax && z <= r.zmax) {
                            int numValid = 0;
                            float fieldMean = 0.0f;
                            for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                                const float* field = fields.at(fieldIdx);
                                float val = field[IDXS(x, y, z)];
                                if (!std::isnan(val)) {
                                    fieldMean += val;
                                    numValid++;
                                }
                            }
                            if (numValid > 1) {
                                fieldMean = fieldMean / float(numValid);
                                float fieldVarianceSum = 0.0f;
                                for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                                    const float* field = fields.at(fieldIdx);
                                    float val = field[IDXS(x, y, z)];
                                    if (!std::isnan(val)) {
                                        float diff = fieldMean - val;
                                        fieldVarianceSum += diff * diff;
                                    }
                                }
                                gridVarianceSum += fieldVarianceSum / float(numValid - 1);
                                gridNumValid += 1;
                            }
                        }
                    }
                }
            }
        } else {
            int zCenter = zs / 2;
            for (int yo = 0; yo < dfy; yo++) {
                for (int xo = 0; xo < dfx; xo++) {
                    int x = r.xoff + xd * dfx + xo;
                    int y = r.yoff + yd * dfy + yo;
                    if (x <= r.xmax && y <= r.ymax) {
                        int numValid = 0;
                        float fieldMean = 0.0f;
                        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                            const float* field = fields.at(fieldIdx);
                            float val = field[IDXS(x, y, zCenter)];
                            if (!std::isnan(val)) {
                                fieldMean += val;
                                numValid++;
                            }
                        }
                        if (numValid > 1) {
                            fieldMean = fieldMean / float(numValid);
                            float fieldVarianceSum = 0.0f;
                            for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                                const float* field = fields.at(fieldIdx);
                                float val = field[IDXS(x, y, zCenter)];
                                if (!std::isnan(val)) {
                                    float diff = fieldMean - val;
                                    fieldVarianceSum += diff * diff;
                                }
                            }
                            gridVarianceSum += fieldVarianceSum / float(numValid - 1);
                            gridNumValid += 1;
                        }
                    }
                }
            }
        }

        float stdDev = 0.0f;
        if (gridNumValid > 0) {
            gridVarianceSum /= float(gridNumValid);
            stdDev = std::sqrt(gridVarianceSum);
        } else {
            stdDev = std::numeric_limits<float>::quiet_NaN();
        }

        uint32_t leafIdx = pointToNodeIndexMap.at(pointIdx) - leafIdxOffset;
        fieldData->leafStdDevArray.at(leafIdx) = stdDev;
    }
#ifdef USE_TBB
    });
#endif
}

void HEBChart::computeCorrelations(
        HEBChartFieldData* fieldData,
        std::vector<float*>& downscaledFields0, std::vector<float*>& downscaledFields1,
        std::vector<MIFieldEntry>& miFieldEntries) {
    auto startTime = std::chrono::system_clock::now();
    if (samplingMethodType == SamplingMethodType::MEAN) {
        computeCorrelationsMean(fieldData, downscaledFields0, downscaledFields1, miFieldEntries);
    } else {
        if (useCorrelationComputationGpu) {
            computeCorrelationsSamplingGpu(fieldData, miFieldEntries);
        } else {
            computeCorrelationsSamplingCpu(fieldData, miFieldEntries);
        }
    }
    auto endTime = std::chrono::system_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Elapsed time correlations: " << elapsedTime.count() << "ms" << std::endl;

    //auto startTimeSort = std::chrono::system_clock::now();
#ifdef USE_TBB
    tbb::parallel_sort(miFieldEntries.begin(), miFieldEntries.end());
//#elif __cpp_lib_parallel_algorithm >= 201603L
    //std::sort(std::execution::par_unseq, miFieldEntries.begin(), miFieldEntries.end());
#else
    std::sort(miFieldEntries.begin(), miFieldEntries.end());
#endif
    //auto endTimeSort = std::chrono::system_clock::now();
    //auto elapsedTimeSort = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeSort - startTimeSort);
    //std::cout << "Elapsed time sort: " << elapsedTimeSort.count() << "ms" << std::endl;
}


#define CORRELATION_CACHE \
        std::vector<float> X(cs); \
        std::vector<float> Y(cs); \
        \
        std::vector<std::pair<float, int>> ordinalRankArraySpearman; \
        float* referenceRanks = nullptr; \
        float* gridPointRanks = nullptr; \
        if (cmt == CorrelationMeasureType::SPEARMAN) { \
            ordinalRankArraySpearman.reserve(cs); \
            referenceRanks = new float[cs]; \
            gridPointRanks = new float[cs]; \
        } \
        \
        std::vector<std::pair<float, float>> jointArray; \
        std::vector<float> ordinalRankArray; \
        std::vector<std::pair<float, int>> ordinalRankArrayRef; \
        std::vector<float> y; \
        if (cmt == CorrelationMeasureType::KENDALL) { \
            jointArray.reserve(cs); \
            ordinalRankArray.reserve(cs); \
            ordinalRankArrayRef.reserve(cs); \
            y.reserve(cs); \
        } \
        \
        double* histogram0 = nullptr; \
        double* histogram1 = nullptr; \
        double* histogram2d = nullptr; \
        if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) { \
            histogram0 = new double[numBins]; \
            histogram1 = new double[numBins]; \
            histogram2d = new double[numBins * numBins]; \
        } \
        \
        KraskovEstimatorCache<double> kraskovEstimatorCache;

void HEBChart::computeCorrelationsMean(
        HEBChartFieldData* fieldData,
        std::vector<float*>& downscaledFields0, std::vector<float*>& downscaledFields1,
        std::vector<MIFieldEntry>& miFieldEntries) {
    int cs = getCorrelationMemberCount();
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;
#ifdef USE_TBB
    miFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPoints0), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const& r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                const CorrelationMeasureType cmt = correlationMeasureType;
                CORRELATION_CACHE;
                float minFieldValRef, maxFieldValRef, minFieldVal, maxFieldVal;

                for (int i = r.begin(); i != r.end(); i++) {
#else
    miFieldEntries.reserve(regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1);
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries, numPoints0, numPoints1, cs, k, numBins) \
    shared(downscaledFields0, downscaledFields1)
#endif
    {
        const CorrelationMeasureType cmt = correlationMeasureType;
        std::vector<MIFieldEntry> miFieldEntriesThread;
        CORRELATION_CACHE;
        float minFieldValRef, maxFieldValRef, minFieldVal, maxFieldVal;

#if _OPENMP >= 201107
        #pragma omp for schedule(dynamic)
#endif
        for (int i = 0; i < numPoints0; i++) {
#endif
            bool isNan = false;
            for (int c = 0; c < cs; c++) {
                X[c] = downscaledFields0.at(c)[i];
                if (std::isnan(X[c])) {
                    isNan = true;
                    break;
                }
            }
            if (isNan) {
                continue;
            }
            if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
                computeRanks(X.data(), referenceRanks, ordinalRankArrayRef, cs);
            }
            if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                minFieldValRef = std::numeric_limits<float>::max();
                maxFieldValRef = std::numeric_limits<float>::lowest();
                for (int c = 0; c < cs; c++) {
                    minFieldValRef = std::min(minFieldValRef, X[c]);
                    maxFieldValRef = std::max(maxFieldValRef, X[c]);
                }
                for (int c = 0; c < cs; c++) {
                    X[c] = (downscaledFields0.at(c)[i] - minFieldValRef) / (maxFieldValRef - minFieldValRef);
                }
            }

            int upperBounds = regionsEqual ? i : numPoints1;
            for (int j = 0; j < upperBounds; j++) {
                if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                    glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                    glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                    float cellDist = glm::length(pti - ptj);
                    if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                        continue;
                    }
                }
                if (regionsEqual && showCorrelationForClickedPoint
                        && clickedPointGridIdx != uint32_t(i) && clickedPointGridIdx != uint32_t(j)) {
                    continue;
                }
                for (int c = 0; c < cs; c++) {
                    Y[c] = downscaledFields1.at(c)[j];
                    if (std::isnan(Y[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (!isNan) {
                    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                        minFieldVal = minFieldValRef;
                        maxFieldVal = maxFieldValRef;
                        for (int c = 0; c < cs; c++) {
                            minFieldVal = std::min(minFieldVal, Y[c]);
                            maxFieldVal = std::max(maxFieldVal, Y[c]);
                        }
                        for (int c = 0; c < cs; c++) {
                            X[c] = (downscaledFields0.at(c)[i] - minFieldVal) / (maxFieldVal - minFieldVal);
                            Y[c] = (downscaledFields1.at(c)[j] - minFieldVal) / (maxFieldVal - minFieldVal);
                        }
                    }

                    float correlationValue = 0.0f;
                    if (cmt == CorrelationMeasureType::PEARSON) {
                        correlationValue = computePearson2<float>(X.data(), Y.data(), cs);
                    } else if (cmt == CorrelationMeasureType::SPEARMAN) {
                        computeRanks(Y.data(), gridPointRanks, ordinalRankArraySpearman, cs);
                        correlationValue = computePearson2<float>(referenceRanks, gridPointRanks, cs);
                    } else if (cmt == CorrelationMeasureType::KENDALL) {
                        correlationValue = computeKendall(
                                X.data(), Y.data(), cs, jointArray, ordinalRankArray, y);
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                        correlationValue = computeMutualInformationBinned<double>(
                                X.data(), Y.data(), numBins, cs, histogram0, histogram1, histogram2d);
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                        correlationValue = computeMutualInformationKraskov<double>(
                                X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                    }
                    if (correlationValue < correlationRange.x || correlationValue > correlationRange.y) {
                        continue;
                    }
                    miFieldEntriesThread.emplace_back(correlationValue, i, j);
                }
            }
        }

        if (cmt == CorrelationMeasureType::SPEARMAN) {
            delete[] referenceRanks;
            delete[] gridPointRanks;
        }
        if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
        }
#ifdef USE_TBB
                return miFieldEntriesThread;
            },
            [&](std::vector<MIFieldEntry> lhs, std::vector<MIFieldEntry> rhs) -> std::vector<MIFieldEntry> {
                std::vector<MIFieldEntry> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
        #pragma omp for ordered schedule(static, 1)
        for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
            #pragma omp ordered
#else
            for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
            {
                miFieldEntries.insert(miFieldEntries.end(), miFieldEntriesThread.begin(), miFieldEntriesThread.end());
            }
        }
    }
#endif
}

void HEBChart::computeCorrelationsSamplingCpu(
        HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries) {
    int cs = getCorrelationMemberCount();
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;

    float minFieldVal = std::numeric_limits<float>::max();
    float maxFieldVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(fieldData->selectedScalarFieldName, fieldIdx);
        const float* field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(fieldData->selectedScalarFieldName, fieldIdx);
            minFieldVal = std::min(minFieldVal, minVal);
            maxFieldVal = std::max(maxFieldVal, maxVal);
        }
    }

    int numPointsDownsampled = regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1;

    auto* samples = new float[6 * numSamples];
    generateSamples(samples, numSamples, samplingMethodType);

#ifdef USE_TBB
    miFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPointsDownsampled), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const& r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                const CorrelationMeasureType cmt = correlationMeasureType;
                CORRELATION_CACHE;

                for (int m = r.begin(); m != r.end(); m++) {
#else
    miFieldEntries.reserve(numPointsDownsampled);
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries, numPoints0, numPoints1, cs, k, numBins) \
    shared(numPointsDownsampled, minFieldVal, maxFieldVal, fields, numSamples, samples)
#endif
    {
        const CorrelationMeasureType cmt = correlationMeasureType;
        std::vector<MIFieldEntry> miFieldEntriesThread;
        CORRELATION_CACHE;

#if _OPENMP >= 201107
        #pragma omp for schedule(dynamic)
#endif
        for (int m = 0; m < numPointsDownsampled; m++) {
#endif
            uint32_t i, j;
            if (regionsEqual) {
                i = (1 + sgl::uisqrt(1 + 8 * uint32_t(m))) / 2;
                j = uint32_t(m) - i * (i - 1) / 2;
            } else {
                i = uint32_t(m / numPoints1);
                j = uint32_t(m % numPoints1);
            }
            if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                float cellDist = glm::length(pti - ptj);
                if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                    continue;
                }
            }
            if (regionsEqual && showCorrelationForClickedPoint
                    && clickedPointGridIdx != uint32_t(i) && clickedPointGridIdx != uint32_t(j)) {
                continue;
            }

            auto region0 = getGridRegionPointIdx(0, i);
            auto region1 = getGridRegionPointIdx(1, j);

            float correlationValueMax = 0.0f;
            bool isValidValue = false;
            for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
                int xi = std::clamp(int(std::round(samples[sampleIdx * 6 + 0] * float(region0.xsr) - 0.5f)), 0, region0.xsr - 1) + region0.xoff;
                int yi = std::clamp(int(std::round(samples[sampleIdx * 6 + 1] * float(region0.ysr) - 0.5f)), 0, region0.ysr - 1) + region0.yoff;
                int zi = std::clamp(int(std::round(samples[sampleIdx * 6 + 2] * float(region0.zsr) - 0.5f)), 0, region0.zsr - 1) + region0.zoff;
                int xj = std::clamp(int(std::round(samples[sampleIdx * 6 + 3] * float(region1.xsr) - 0.5f)), 0, region1.xsr - 1) + region1.xoff;
                int yj = std::clamp(int(std::round(samples[sampleIdx * 6 + 4] * float(region1.ysr) - 0.5f)), 0, region1.ysr - 1) + region1.yoff;
                int zj = std::clamp(int(std::round(samples[sampleIdx * 6 + 5] * float(region1.zsr) - 0.5f)), 0, region1.zsr - 1) + region1.zoff;
                int idxi = IDXS(xi, yi, zi);
                int idxj = IDXS(xj, yj, zj);
                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    X[c] = fields.at(c)[idxi];
                    Y[c] = fields.at(c)[idxj];
                    if (std::isnan(X[c]) || std::isnan(Y[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    continue;
                }

                float correlationValue = 0.0f;
                if (cmt == CorrelationMeasureType::PEARSON) {
                    correlationValue = computePearson2<float>(X.data(), Y.data(), cs);
                } else if (cmt == CorrelationMeasureType::SPEARMAN) {
                    computeRanks(Y.data(), referenceRanks, ordinalRankArrayRef, cs);
                    computeRanks(Y.data(), gridPointRanks, ordinalRankArraySpearman, cs);
                    correlationValue = computePearson2<float>(referenceRanks, gridPointRanks, cs);
                } else if (cmt == CorrelationMeasureType::KENDALL) {
                    correlationValue = computeKendall(
                            X.data(), Y.data(), cs, jointArray, ordinalRankArray, y);
                } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                    for (int c = 0; c < cs; c++) {
                        X[c] = (X[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                        Y[c] = (Y[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                    }
                    correlationValue = computeMutualInformationBinned<double>(
                            X.data(), Y.data(), numBins, cs, histogram0, histogram1, histogram2d);
                } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                    correlationValue = computeMutualInformationKraskov<double>(
                            X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                }
                if (std::abs(correlationValue) >= std::abs(correlationValueMax)) {
                    correlationValueMax = correlationValue;
                    isValidValue = true;
                }
            }

            if (isValidValue && correlationValueMax >= correlationRange.x && correlationValueMax <= correlationRange.y) {
                miFieldEntriesThread.emplace_back(correlationValueMax, i, j);
            }
        }

        if (cmt == CorrelationMeasureType::SPEARMAN) {
            delete[] referenceRanks;
            delete[] gridPointRanks;
        }
        if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
        }
#ifdef USE_TBB
        return miFieldEntriesThread;
            },
            [&](std::vector<MIFieldEntry> lhs, std::vector<MIFieldEntry> rhs) -> std::vector<MIFieldEntry> {
                std::vector<MIFieldEntry> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
        #pragma omp for ordered schedule(static, 1)
        for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
            #pragma omp ordered
#else
            for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
            {
                miFieldEntries.insert(miFieldEntries.end(), miFieldEntriesThread.begin(), miFieldEntriesThread.end());
            }
        }
    }
#endif

    delete[] samples;
}

struct CorrelationRequestData {
    uint32_t xi, yi, zi, i, xj, yj, zj, j;
};

void HEBChart::computeCorrelationsSamplingGpu(
        HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries) {
    int cs = getCorrelationMemberCount();
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;

    int numPointsDownsampled = regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1;

    auto* samples = new float[6 * numSamples];
    generateSamples(samples, numSamples, samplingMethodType);

    const uint32_t batchSizeSamplesMaxAllCs = 1 << 17; // Up to 131072 samples per batch.
    uint32_t batchSizeSamplesMax = batchSizeSamplesMaxAllCs;
    if (cs > 100) {
        double factorN = double(cs) / 100.0 * std::log2(double(cs) / 100.0 + 1.0);
        batchSizeSamplesMax = std::ceil(double(batchSizeSamplesMax) / factorN);
        batchSizeSamplesMax = uint32_t(sgl::nextPowerOfTwo(int(batchSizeSamplesMax)));
    }
    uint32_t batchSizeCellsMax = batchSizeSamplesMax / numSamples;
    uint32_t numBatches = sgl::uiceil(uint32_t(numPointsDownsampled), batchSizeCellsMax);

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (!requestsBuffer) {
        computeRenderer = new sgl::vk::Renderer(device, 100);
        if (device->getGraphicsQueue() == device->getComputeQueue()) {
            supportsAsyncCompute = false;
        }
        fence = std::make_shared<sgl::vk::Fence>(device);

        sgl::vk::CommandPoolType commandPoolType{};
        commandPoolType.queueFamilyIndex = device->getComputeQueueIndex();
        commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandBuffer = device->allocateCommandBuffer(commandPoolType, &commandPool);
        computeRenderer->setCustomCommandBuffer(commandBuffer, false);

        requestsBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(CorrelationRequestData) * batchSizeSamplesMaxAllCs,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        requestsStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(CorrelationRequestData) * batchSizeSamplesMaxAllCs,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU);
        correlationOutputBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(float) * batchSizeSamplesMaxAllCs,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        correlationOutputStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(float) * batchSizeSamplesMaxAllCs,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_TO_CPU);

        correlationComputePass = std::make_shared<CorrelationComputePass>(computeRenderer);
        correlationComputePass->setUseRequestEvaluationMode(true);
        correlationComputePass->setRequestsBuffer(requestsBuffer);
        correlationComputePass->setOutputBuffer(correlationOutputBuffer);
    }

    float minFieldVal = std::numeric_limits<float>::max();
    float maxFieldVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::DeviceCacheEntry> fieldEntries;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    fieldEntries.reserve(cs);
    fieldImageViews.reserve(cs);
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::DeviceCacheEntry fieldEntry = getFieldEntryDevice(fieldData->selectedScalarFieldName, fieldIdx);
        fieldEntries.push_back(fieldEntry);
        fieldImageViews.push_back(fieldEntry->getVulkanImageView());
        if (fieldEntry->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            fieldEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
        }
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(fieldData->selectedScalarFieldName, fieldIdx);
            minFieldVal = std::min(minFieldVal, minVal);
            maxFieldVal = std::max(maxFieldVal, maxVal);
        }
    }

    uint32_t cellIdxOffset = 0;
    auto numCellsLeft = uint32_t(numPointsDownsampled);
    std::vector<CorrelationRequestData> requests;
    requests.reserve(batchSizeCellsMax);
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchSizeCells;
        if (numCellsLeft > batchSizeCellsMax) {
            batchSizeCells = batchSizeCellsMax;
            numCellsLeft -= batchSizeCellsMax;
        } else if (numCellsLeft > 0) {
            batchSizeCells = numCellsLeft;
            numCellsLeft = 0;
        } else {
            break;
        }


        // Create the batch data to upload to the device.
        uint32_t loopMax = cellIdxOffset + batchSizeCells;
        requests.clear();
#ifdef USE_TBB
        requests = tbb::parallel_reduce(
            tbb::blocked_range<uint32_t>(cellIdxOffset, loopMax), std::vector<CorrelationRequestData>(),
            [&](tbb::blocked_range<int> const& r, std::vector<CorrelationRequestData> requestsThread) -> std::vector<CorrelationRequestData> {
                for (int m = r.begin(); m != r.end(); m++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(requests, numPoints0, numPoints1, cs) \
        shared(cellIdxOffset, loopMax, numSamples, samples)
#endif
        {
            std::vector<CorrelationRequestData> requestsThread;

#if _OPENMP >= 201107
            #pragma omp for schedule(dynamic)
#endif
            for (uint32_t m = cellIdxOffset; m < loopMax; m++) {
#endif
                uint32_t i, j;
                if (regionsEqual) {
                    i = (1 + sgl::uisqrt(1 + 8 * m)) / 2;
                    j = uint32_t(m) - i * (i - 1) / 2;
                } else {
                    i = m / uint32_t(numPoints1);
                    j = m % uint32_t(numPoints1);
                }
                if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                    glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                    glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                    float cellDist = glm::length(pti - ptj);
                    if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                        continue;
                    }
                }
                if (regionsEqual && showCorrelationForClickedPoint
                        && clickedPointGridIdx != i && clickedPointGridIdx != j) {
                    continue;
                }

                auto region0 = getGridRegionPointIdx(0, i);
                auto region1 = getGridRegionPointIdx(1, j);

                CorrelationRequestData request;
                for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
                    int xi = std::clamp(int(std::round(samples[sampleIdx * 6 + 0] * float(region0.xsr) - 0.5f)), 0, region0.xsr - 1) + region0.xoff;
                    int yi = std::clamp(int(std::round(samples[sampleIdx * 6 + 1] * float(region0.ysr) - 0.5f)), 0, region0.ysr - 1) + region0.yoff;
                    int zi = std::clamp(int(std::round(samples[sampleIdx * 6 + 2] * float(region0.zsr) - 0.5f)), 0, region0.zsr - 1) + region0.zoff;
                    int xj = std::clamp(int(std::round(samples[sampleIdx * 6 + 3] * float(region1.xsr) - 0.5f)), 0, region1.xsr - 1) + region1.xoff;
                    int yj = std::clamp(int(std::round(samples[sampleIdx * 6 + 4] * float(region1.ysr) - 0.5f)), 0, region1.ysr - 1) + region1.yoff;
                    int zj = std::clamp(int(std::round(samples[sampleIdx * 6 + 5] * float(region1.zsr) - 0.5f)), 0, region1.zsr - 1) + region1.zoff;
                    request.i = i;
                    request.j = j;
                    request.xi = uint32_t(xi);
                    request.yi = uint32_t(yi);
                    request.zi = uint32_t(zi);
                    request.xj = uint32_t(xj);
                    request.yj = uint32_t(yj);
                    request.zj = uint32_t(zj);
                    requestsThread.emplace_back(request);
                }
            }

#ifdef USE_TBB
                return requestsThread;
            },
            [&](std::vector<CorrelationRequestData> lhs, std::vector<CorrelationRequestData> rhs) -> std::vector<CorrelationRequestData> {
                std::vector<CorrelationRequestData> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
            #pragma omp for ordered schedule(static, 1)
            for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
                #pragma omp ordered
#else
                for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
                {
                    requests.insert(requests.end(), requestsThread.begin(), requestsThread.end());
                }
            }
        }
#endif
        cellIdxOffset += batchSizeCells;


        computeRenderer->beginCommandBuffer();
        if (batchIdx == 0) {
            correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
            correlationComputePass->setVolumeData(volumeData.get(), cs, false);
            correlationComputePass->setCorrelationMemberCount(cs);
            correlationComputePass->setNumBins(numBins);
            correlationComputePass->setKraskovNumNeighbors(k);
            correlationComputePass->setFieldImageViews(fieldImageViews);
            correlationComputePass->buildIfNecessary();
        }

        // Upload the requests to the GPU, compute the correlations, and download the correlations back to the CPU.
        requestsStagingBuffer->uploadData(sizeof(CorrelationRequestData) * requests.size(), requests.data());
        requestsStagingBuffer->copyDataTo(requestsBuffer, computeRenderer->getVkCommandBuffer());
        computeRenderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                requestsStagingBuffer);
        computeRenderer->pushConstants(
                correlationComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                glm::uvec2(batchSizeCells * uint32_t(numSamples), uint32_t(cs)));
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            computeRenderer->pushConstants(
                    correlationComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 2 * sizeof(float),
                    glm::vec2(minFieldVal, maxFieldVal));
        }
        correlationComputePass->setNumRequests(uint32_t(requests.size()));
        correlationComputePass->render();
        computeRenderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                requestsStagingBuffer);
        correlationOutputBuffer->copyDataTo(correlationOutputStagingBuffer, computeRenderer->getVkCommandBuffer());
        computeRenderer->endCommandBuffer();
        computeRenderer->submitToQueue({}, {}, fence, VK_PIPELINE_STAGE_TRANSFER_BIT);
        fence->wait();
        fence->reset();


        // Finally, insert the values of this batch.
        auto* correlationValues = static_cast<float*>(correlationOutputStagingBuffer->mapMemory());
        auto numValidCells = int(requests.size() / size_t(numSamples));
#ifdef USE_TBB
        std::vector<MIFieldEntry> newMiFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPointsDownsampled), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const& r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                const CorrelationMeasureType cmt = correlationMeasureType;
                CORRELATION_CACHE;

                for (int m = r.begin(); m != r.end(); m++) {
#else
        miFieldEntries.reserve(numPointsDownsampled);
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(miFieldEntries, requests, correlationValues, numValidCells)
#endif
        {
            std::vector<MIFieldEntry> miFieldEntriesThread;
#if _OPENMP >= 201107
            #pragma omp for
#endif
            for (int cellIdx = 0; cellIdx < numValidCells; cellIdx++) {
#endif
                int requestIdx = cellIdx * numSamples;
                auto i = int(requests.at(requestIdx).i);
                auto j = int(requests.at(requestIdx).j);
                float correlationValueMax = 0.0f;
                bool isValidValue = false;
                for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
                    float correlationValue = correlationValues[requestIdx];
                    if (std::abs(correlationValue) >= std::abs(correlationValueMax)) {
                        correlationValueMax = correlationValue;
                        isValidValue = true;
                    }
                    requestIdx++;
                }
                if (isValidValue && correlationValueMax >= correlationRange.x && correlationValueMax <= correlationRange.y) {
                    miFieldEntriesThread.emplace_back(correlationValueMax, i, j);
                }
            }
#ifdef USE_TBB
            return miFieldEntriesThread;
            },
            [&](std::vector<MIFieldEntry> lhs, std::vector<MIFieldEntry> rhs) -> std::vector<MIFieldEntry> {
                std::vector<MIFieldEntry> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
            #pragma omp for ordered schedule(static, 1)
            for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
                #pragma omp ordered
#else
                for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
                {
                    miFieldEntries.insert(miFieldEntries.end(), miFieldEntriesThread.begin(), miFieldEntriesThread.end());
                }
            }
        }
#endif
#ifdef USE_TBB
        miFieldEntries.insert(miFieldEntries.end(), newMiFieldEntries.begin(), newMiFieldEntries.end());
#endif
        correlationOutputStagingBuffer->unmapMemory();
    }

    delete[] samples;
}

void getControlPoints(
        const std::vector<HEBNode>& nodesList,
        const std::vector<uint32_t>& pointToNodeIndexMap0, const std::vector<uint32_t>& pointToNodeIndexMap1,
        uint32_t pointIndex0, uint32_t pointIndex1, std::vector<glm::vec2>& controlPoints) {
    // The start nodes are leaves at the same level.
    uint32_t nidx0 = pointToNodeIndexMap0.at(pointIndex0);
    uint32_t nidx1 = pointToNodeIndexMap1.at(pointIndex1);

    // Go until lowest common ancestor (LCA).
    std::vector<uint32_t> ancestors0;
    while(nidx0 != std::numeric_limits<uint32_t>::max()) {
        ancestors0.push_back(nidx0);
        nidx0 = nodesList.at(nidx0).parentIdx;
    }
    std::vector<uint32_t> ancestors1;
    while(nidx1 != std::numeric_limits<uint32_t>::max()) {
        ancestors1.push_back(nidx1);
        nidx1 = nodesList.at(nidx1).parentIdx;
    }

    // Find first different ancestor.
    auto idx0 = int(ancestors0.size() - 1);
    auto idx1 = int(ancestors1.size() - 1);
    while (idx0 > 0 && idx1 > 0) {
        if (ancestors0.at(idx0) != ancestors1.at(idx1)) {
            // Least common ancestor at idx0 + 1 / idx1 + 1.
            break;
        }
        idx0--;
        idx1--;
    }
    for (int i = 0; i <= idx0; i++) {
        controlPoints.push_back(nodesList.at(ancestors0.at(i)).normalizedPosition);
    }
    // Original control polygon has more than 3 control points?
    if (idx0 + idx1 + 2 <= 3) {
        controlPoints.push_back(nodesList.at(ancestors0.at(idx0 + 1)).normalizedPosition);
    }
    for (int i = idx1; i >= 0; i--) {
        controlPoints.push_back(nodesList.at(ancestors1.at(i)).normalizedPosition);
    }
}

void smoothControlPoints(std::vector<glm::vec2>& controlPoints, float beta) {
    glm::vec2 p0 = controlPoints.front();
    glm::vec2 pn = controlPoints.back();
    for (int i = 1; i < int(controlPoints.size()) - 1; i++) {
        glm::vec2& p = controlPoints.at(i);
        p = beta * p + (1.0f - beta) * (p0 + float(i) / float(controlPoints.size() - 1) * (pn - p0));
    }
}

void HEBChart::updateData() {
    // Values downscaled by factor 32.
    int cs = getCorrelationMemberCount();
    xsd0 = sgl::iceil(r0.xsr, dfx);
    ysd0 = sgl::iceil(r0.ysr, dfy);
    zsd0 = sgl::iceil(r0.zsr, dfz);
    xsd1 = sgl::iceil(r1.xsr, dfx);
    ysd1 = sgl::iceil(r1.ysr, dfy);
    zsd1 = sgl::iceil(r1.zsr, dfz);
    int numPoints = xsd0 * ysd0 * zsd0 + (regionsEqual ? 0 : xsd1 * ysd1 * zsd1);

    if (selectedLineIdx >= 0 || selectedPointIndices[0] >= 0 || selectedPointIndices[1] >= 0) {
        needsReRender = true;
    }
    resetSelectedPrimitives();

    if (use2dField) {
        numPoints = xsd0 * ysd0;
        zsd0 = zsd1 = 1;
    }

    std::vector<HEBChartFieldUpdateData> updateDataArray(fieldDataArray.size());
    numLinesTotal = 0;
    auto numFields = int(fieldDataArray.size());
    for (int i = 0; i < numFields; i++) {
        auto* fieldData = fieldDataArray.at(i).get();
        if (limitedFieldIdx >= 0 && limitedFieldIdx != fieldData->selectedFieldIdx) {
            continue;
        }

        // Compute the downscaled field.
        std::vector<float*> downscaledFields0, downscaledFields1;
        if (samplingMethodType == SamplingMethodType::MEAN) {
            downscaledFields0.resize(cs);
            computeDownscaledField(fieldData, 0, downscaledFields0);
            if (!regionsEqual) {
                downscaledFields1.resize(cs);
                computeDownscaledField(fieldData, 1, downscaledFields1);
            }
        }

        // Compute the correlation matrix.
        std::vector<MIFieldEntry> miFieldEntries;
        if (regionsEqual) {
            computeCorrelations(fieldData, downscaledFields0, downscaledFields0, miFieldEntries);
        } else {
            computeCorrelations(fieldData, downscaledFields0, downscaledFields1, miFieldEntries);
        }

        // Delete the downscaled fields, as they are no longer used.
        if (samplingMethodType == SamplingMethodType::MEAN) {
            for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                float *downscaledField = downscaledFields0.at(fieldIdx);
                delete[] downscaledField;
            }
            downscaledFields0.clear();
            if (!regionsEqual) {
                for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                    float *downscaledField = downscaledFields1.at(fieldIdx);
                    delete[] downscaledField;
                }
                downscaledFields1.clear();
            }
        }

        // Build the octree.
        nodesList.clear();
        pointToNodeIndexMap0.clear();
        pointToNodeIndexMap1.clear();
        buildHebTree(
                octreeMethod, nodesList, pointToNodeIndexMap0, pointToNodeIndexMap1, leafIdxOffset, leafIdxOffset1,
                regionsEqual, xsd0, ysd0, zsd0, xsd1, ysd1, zsd1);

        // Compute the standard deviation inside the downscaled grids.
        computeDownscaledFieldVariance(fieldData, 0);
        if (!regionsEqual) {
            computeDownscaledFieldVariance(fieldData, 1);
        }
        // Normalize standard deviations for visualization.
        auto [minVal, maxVal] = sgl::reduceFloatArrayMinMax(fieldData->leafStdDevArray);
        fieldData->minStdDev = minVal;
        fieldData->maxStdDev = maxVal;

        int maxNumLines = numPoints * MAX_NUM_LINES / 100;
        int numLinesLocal = std::min(maxNumLines, int(miFieldEntries.size()));
        numLinesTotal += numLinesLocal;
        std::vector<glm::vec2>& curvePointsLocal = updateDataArray.at(i).curvePoints;
        std::vector<float>& correlationValuesArrayLocal = updateDataArray.at(i).correlationValuesArray;
        std::vector<std::pair<int, int>>& connectedPointsArrayLocal = updateDataArray.at(i).connectedPointsArray;
        curvePointsLocal.resize(numLinesLocal * NUM_SUBDIVISIONS);
        correlationValuesArrayLocal.resize(numLinesLocal);
        connectedPointsArrayLocal.resize(numLinesLocal);

        if (!miFieldEntries.empty() && numLinesLocal > 0) {
            fieldData->minCorrelationValue = miFieldEntries.at(numLinesLocal - 1).correlationValue;
            fieldData->maxCorrelationValue = miFieldEntries.at(0).correlationValue;
        } else {
            fieldData->minCorrelationValue = std::numeric_limits<float>::max();
            fieldData->maxCorrelationValue = std::numeric_limits<float>::lowest();
        }

#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<int>(0, NUM_LINES), [&](auto const& r) {
            std::vector<glm::vec2> controlPoints;
            for (auto lineIdx = r.begin(); lineIdx != r.end(); lineIdx++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(miFieldEntries, numLinesLocal, i) \
        shared(curvePointsLocal, correlationValuesArrayLocal, connectedPointsArrayLocal)
#endif
        {
            std::vector<glm::vec2> controlPoints;
#if _OPENMP >= 201107
            #pragma omp for
#endif
            for (int lineIdx = 0; lineIdx < numLinesLocal; lineIdx++) {
#endif
                // Use reverse order so lines are drawn from least to most important.
                const auto& miEntry = miFieldEntries.at(numLinesLocal - lineIdx - 1);
                correlationValuesArrayLocal.at(lineIdx) = miEntry.correlationValue;

                auto idx0 = int(pointToNodeIndexMap0.at(miEntry.pointIndex0) - leafIdxOffset);
                auto idx1 = int(pointToNodeIndexMap1.at(miEntry.pointIndex1) - leafIdxOffset);
                connectedPointsArrayLocal.at(lineIdx) = std::make_pair(idx0, idx1);

                controlPoints.clear();
                getControlPoints(
                        nodesList, pointToNodeIndexMap0, pointToNodeIndexMap1,
                        miEntry.pointIndex0, miEntry.pointIndex1, controlPoints);
                if (beta < 1.0f) {
                    smoothControlPoints(controlPoints, beta);
                }
                for (int ptIdx = 0; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                    float t = float(ptIdx) / float(NUM_SUBDIVISIONS - 1);
                    int k = 4;
                    if (controlPoints.size() == 3) {
                        k = 3;
                    }
                    curvePointsLocal.at(lineIdx * NUM_SUBDIVISIONS + ptIdx) = evaluateBSpline(t, k, controlPoints);
                }
            }
#ifdef USE_TBB
            });
#else
        }
#endif
    }

    curvePoints.resize(numLinesTotal * NUM_SUBDIVISIONS);
    correlationValuesArray.resize(numLinesTotal);
    connectedPointsArray.resize(numLinesTotal);
    lineFieldIndexArray.resize(numLinesTotal);
    std::vector<std::tuple<float, int, int>> lineSortArray; //< (correlationValue, fieldIdx, lineIdx) tuples.
    for (int i = 0; i < numFields; i++) {
        const std::vector<float>& correlationValuesArrayLocal = updateDataArray.at(i).correlationValuesArray;
        auto numLines = int(correlationValuesArrayLocal.size());
        for (int lineIdx = 0; lineIdx < numLines; lineIdx++) {
            lineSortArray.emplace_back(std::abs(correlationValuesArrayLocal.at(lineIdx)), i, lineIdx);
        }
    }
    std::sort(lineSortArray.begin(), lineSortArray.end());
    minCorrelationValueGlobal = std::numeric_limits<float>::max();
    maxCorrelationValueGlobal = std::numeric_limits<float>::lowest();
    for (int lineIdx = 0; lineIdx < numLinesTotal; lineIdx++) {
        auto [correlationValue, i, localLineIdx] = lineSortArray.at(lineIdx);
        correlationValuesArray.at(lineIdx) = updateDataArray.at(i).correlationValuesArray.at(localLineIdx);
        connectedPointsArray.at(lineIdx) = updateDataArray.at(i).connectedPointsArray.at(localLineIdx);
        lineFieldIndexArray.at(lineIdx) = i;
        minCorrelationValueGlobal = std::min(minCorrelationValueGlobal, fieldDataArray.at(i)->minCorrelationValue);
        maxCorrelationValueGlobal = std::max(maxCorrelationValueGlobal, fieldDataArray.at(i)->maxCorrelationValue);
        const std::vector<glm::vec2>& curvePointsLocal = updateDataArray.at(i).curvePoints;
        for (int ptIdx = 0; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
            curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx) =
                    curvePointsLocal.at(localLineIdx * NUM_SUBDIVISIONS + ptIdx);
        }
    }
}

bool HEBChart::getIsRegionSelected(int idx) {
    return selectedPointIndices[idx] >= 0;
}

uint32_t HEBChart::getPointIndexGrid(int pointIdx) {
    int groupIdx = getLeafIdxGroup(pointIdx);
    auto& pointToNodeIndexMap = groupIdx == 0 ? pointToNodeIndexMap0 : pointToNodeIndexMap1;
    return uint32_t(std::find(
            pointToNodeIndexMap.begin(), pointToNodeIndexMap.end(),
            int(leafIdxOffset) + pointIdx) - pointToNodeIndexMap.begin());
}

uint32_t HEBChart::getSelectedPointIndexGrid(int idx) {
    int groupIdx = getLeafIdxGroup(selectedPointIndices[idx]);
    auto& pointToNodeIndexMap = groupIdx == 0 ? pointToNodeIndexMap0 : pointToNodeIndexMap1;
    return uint32_t(std::find(
            pointToNodeIndexMap.begin(), pointToNodeIndexMap.end(),
            int(leafIdxOffset) + selectedPointIndices[idx]) - pointToNodeIndexMap.begin());
}

sgl::AABB3 HEBChart::getSelectedRegion(int idx) {
    //const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedCircleIdx);
    int groupIdx = getLeafIdxGroup(selectedPointIndices[idx]);
    int xsd = groupIdx == 0 ? xsd0 : xsd1;
    int ysd = groupIdx == 0 ? ysd0 : ysd1;
    //int zsd = groupIdx == 0 ? zsd0 : zsd1;
    GridRegion r = groupIdx == 0 ? r0 : r1;

    auto pointIdx = getSelectedPointIndexGrid(idx);
    uint32_t xd = pointIdx % uint32_t(xsd);
    uint32_t yd = (pointIdx / uint32_t(xsd)) % uint32_t(ysd);
    uint32_t zd = pointIdx / uint32_t(xsd * ysd);
    sgl::AABB3 aabb;
    if (use2dField) {
        int zCenter = zs / 2;
        aabb.min = glm::vec3(r.xoff + xd * dfx, r.yoff + yd * dfy, zCenter);
        aabb.max = glm::vec3(
                float(std::min(r.xoff + int(xd + 1) * dfx, r.xmax + 1)),
                float(std::min(r.yoff + int(yd + 1) * dfy, r.ymax + 1)),
                float(std::min(zCenter + 1, zs)));
    } else {
        aabb.min = glm::vec3(r.xoff + xd * dfx, r.yoff + yd * dfy, r.zoff + zd * dfz);
        aabb.max = glm::vec3(
                float(std::min(r.xoff + int(xd + 1) * dfx, r.xmax + 1)),
                float(std::min(r.yoff + int(yd + 1) * dfy, r.ymax + 1)),
                float(std::min(r.zoff + int(zd + 1) * dfz, r.zmax + 1)));
    }
    aabb.min /= glm::vec3(xs, ys, zs);
    aabb.max /= glm::vec3(xs, ys, zs);
    sgl::AABB3 volumeAABB = volumeData->getBoundingBoxRendering();
    aabb.min = volumeAABB.min + (volumeAABB.max - volumeAABB.min) * aabb.min;
    aabb.max = volumeAABB.min + (volumeAABB.max - volumeAABB.min) * aabb.max;
    return aabb;
}

std::pair<glm::vec3, glm::vec3> HEBChart::getLinePositions() {
    int groupIdx0 = getLeafIdxGroup(selectedPointIndices[0]);
    int xsdg0 = groupIdx0 == 0 ? xsd0 : xsd1;
    int ysdg0 = groupIdx0 == 0 ? ysd0 : ysd1;
    const auto& rg0 =  groupIdx0 == 0 ? r0 : r1;
    //int zsdg0 = groupIdx0 == 0 ? zsd0 : zsd1;
    int groupIdx1 = getLeafIdxGroup(selectedPointIndices[1]);
    int xsdg1 = groupIdx1 == 0 ? xsd0 : xsd1;
    int ysdg1 = groupIdx1 == 0 ? ysd0 : ysd1;
    //int zsdg1 = groupIdx1 == 0 ? zsd0 : zsd1;
    const auto& rg1 =  groupIdx1 == 0 ? r0 : r1;

    auto pointIdx0 = getSelectedPointIndexGrid(0);
    uint32_t xd0 = pointIdx0 % uint32_t(xsdg0);
    uint32_t yd0 = (pointIdx0 / uint32_t(xsdg0)) % uint32_t(ysdg0);
    uint32_t zd0 = pointIdx0 / uint32_t(xsdg0 * ysdg0);
    glm::ivec3 c0(rg0.xoff + (int)xd0 * dfx, rg0.yoff + (int)yd0 * dfy, rg0.zoff + (int)zd0 * dfz);
    auto pointIdx1 = getSelectedPointIndexGrid(1);
    uint32_t xd1 = pointIdx1 % uint32_t(xsdg1);
    uint32_t yd1 = (pointIdx1 / uint32_t(xsdg1)) % uint32_t(ysdg1);
    uint32_t zd1 = pointIdx1 / uint32_t(xsdg1 * ysdg1);
    glm::ivec3 c1(rg1.xoff + int(xd1) * dfx, rg1.yoff + (int)yd1 * dfy, rg1.zoff + (int)zd1 * dfz);

    auto b0 = getSelectedRegion(0);
    auto b1 = getSelectedRegion(1);

    glm::vec3 p0, p1;
    p0.x = c0.x < c1.x ? b0.max.x : (c0.x > c1.x ? b0.min.x : 0.5f * (b0.min.x + b0.max.x));
    p0.y = c0.y < c1.y ? b0.max.y : (c0.y > c1.y ? b0.min.y : 0.5f * (b0.min.y + b0.max.y));
    p0.z = c0.z < c1.z ? b0.max.z : (c0.z > c1.z ? b0.min.z : 0.5f * (b0.min.z + b0.max.z));
    p1.x = c1.x < c0.x ? b1.max.x : (c1.x > c0.x ? b1.min.x : 0.5f * (b1.min.x + b1.max.x));
    p1.y = c1.y < c0.y ? b1.max.y : (c1.y > c0.y ? b1.min.y : 0.5f * (b1.min.y + b1.max.y));
    p1.z = c1.z < c0.z ? b1.max.z : (c1.z > c0.z ? b1.min.z : 0.5f * (b1.min.z + b1.max.z));
    return std::make_pair(p0, p1);
}

glm::vec3 HEBChart::getLineDirection() {
    auto linePoints = getLinePositions();
    glm::vec3 diffVector = linePoints.second - linePoints.first;
    float lineLength = glm::length(diffVector);
    if (lineLength > 1e-5f) {
        return diffVector / lineLength;
    } else {
        auto b0 = getSelectedRegion(0);
        auto b1 = getSelectedRegion(1);
        return glm::normalize(b1.getCenter() - b0.getCenter());
    }
}


void HEBChart::resetFocusSelection() {
    clickedLineIdxOld = -1;
    clickedPointIdxOld = -1;
    isFocusSelectionReset = true;
}

bool HEBChart::getHasNewFocusSelection(bool& isDeselection) {
    if (isFocusSelectionReset) {
        isDeselection = false;
        return false;
    }
    bool hasNewFocusSelection =
            (clickedPointIdx >= 0 && clickedPointIdx != clickedPointIdxOld)
            || (clickedLineIdx >= 0 && clickedLineIdx != clickedLineIdxOld);
    if (hasNewFocusSelection) {
        std::pair<GridRegion, GridRegion> regions = getFocusSelection();
        if (regions.first.getNumCells() <= 1 && regions.second.getNumCells() <= 1) {
            hasNewFocusSelection = false;
        }
    }
    isDeselection =
            !hasNewFocusSelection && (clickedPointIdx != clickedPointIdxOld || clickedLineIdx != clickedLineIdxOld);
    clickedPointIdxOld = clickedPointIdx;
    clickedLineIdxOld = clickedLineIdx;
    return hasNewFocusSelection;
}

std::pair<GridRegion, GridRegion> HEBChart::getFocusSelection() {
    if (clickedPointIdx != -1) {
        uint32_t pointIdx = getPointIndexGrid(clickedPointIdx);
        auto pointRegion = getGridRegionPointIdx(getLeafIdxGroup(clickedPointIdx), pointIdx);
        return std::make_pair(pointRegion, pointRegion);
    } else {
        auto points = connectedPointsArray.at(clickedLineIdx);
        uint32_t pointIdx0 = getPointIndexGrid(points.first);
        uint32_t pointIdx1 = getPointIndexGrid(points.second);
        return std::make_pair(
                getGridRegionPointIdx(getLeafIdxGroup(points.first), pointIdx0),
                getGridRegionPointIdx(getLeafIdxGroup(points.second), pointIdx1));
    }
}

GridRegion HEBChart::getGridRegionPointIdx(int idx, uint32_t pointIdx) {
    int xsd = idx == 0 ? xsd0 : xsd1;
    int ysd = idx == 0 ? ysd0 : ysd1;
    //int zsd = idx == 0 ? zsd0 : zsd1;
    int xd = int(pointIdx % uint32_t(xsd));
    int yd = int((pointIdx / uint32_t(xsd)) % uint32_t(ysd));
    int zd = int(pointIdx / uint32_t(xsd * ysd));
    GridRegion r = idx == 0 ? r0 : r1;

    GridRegion rf;
    if (use2dField) {
        int zCenter = zs / 2;
        rf = GridRegion(
                r.xoff + xd * dfx, r.yoff + yd * dfy, zCenter,
                std::min((xd + 1) * dfx, r.xsr) - xd * dfx,
                std::min((yd + 1) * dfy, r.ysr) - yd * dfy,
                1);
    } else {
        rf = GridRegion(
                r.xoff + xd * dfx, r.yoff + yd * dfy, r.zoff + zd * dfz,
                std::min((xd + 1) * dfx, r.xsr) - xd * dfx,
                std::min((yd + 1) * dfy, r.ysr) - yd * dfy,
                std::min((zd + 1) * dfz, r.zsr) - zd * dfz);
    }
    return rf;
}

int HEBChart::getLeafIdxGroup(int leafIdx) const {
    if (regionsEqual) {
        return 0;
    }
    return leafIdx >= int(leafIdxOffset1 - leafIdxOffset) ? 1 : 0;
}

int HEBChart::getFocusSelectionFieldIndex() {
    return fieldDataArray.at(lineFieldIndexArray.at(clickedLineIdx))->selectedFieldIdx;
}
