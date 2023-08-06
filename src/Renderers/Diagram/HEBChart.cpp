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

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "Calculators/MutualInformation.hpp"
#include "BSpline.hpp"
#include "HEBChart.hpp"

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

DeviceCacheEntry HEBChart::getFieldEntryDevice(const std::string& fieldName, int fieldIdx, bool wantsImageData) {
    VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, fieldName, isEnsembleMode ? -1 : fieldIdx, isEnsembleMode ? fieldIdx : -1,
            wantsImageData, (!wantsImageData && useBufferTiling) ? glm::uvec3(8, 8, 4) : glm::uvec3(1, 1, 1));
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
    updateRegion();
}

void HEBChart::resetSelectedPrimitives() {
    hoveredPointIdx = -1;
    clickedPointIdx = -1;
    clickedPointIdxOld = -1;
    selectedPointIndices[0] = -1;
    selectedPointIndices[1] = -1;

    hoveredLineIdx = -1;
    clickedLineIdx = -1;
    clickedLineIdxOld = -1;
    selectedLineIdx = -1;

    hoveredGridIdx = {};
    clickedGridIdx = {};
    clickedGridIdxOld = {};
    selectedGridIdx = {};
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

    std::shared_ptr<HEBChartFieldData> fieldData2;
    //bool isTwoFieldMode = false;
    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    if (fieldData->selectedFieldIdx >= int(fieldNames.size())) {
        /*isTwoFieldMode = true;
        fieldData2 = std::make_shared<HEBChartFieldData>(this, &desaturateUnselectedRing);
        fieldData2->selectedFieldIdx = _selectedFieldIdx;
        fieldData2->selectedScalarFieldName = _scalarFieldName;
        fieldData2->separateColorVarianceAndCorrelation = separateColorVarianceAndCorrelation;
        fieldData2->setColorMapVariance(colorMapVariance);*/

        uint32_t m = uint32_t(fieldData->selectedFieldIdx) - uint32_t(fieldNames.size());
        uint32_t mp = 0;
        for (size_t i = 0; i < fieldNames.size(); i++) {
            for (size_t j = i + 1; j < fieldNames.size(); j++) {
                if (m == mp) {
                    fieldData->selectedFieldIdx1 = int(i);
                    fieldData->selectedFieldIdx2 = int(j);
                    break;
                }
                mp++;
            }
            if (m == mp) {
                break;
            }
        }
        fieldData->selectedScalarFieldName1 = fieldNames.at(fieldData->selectedFieldIdx1);
        fieldData->selectedScalarFieldName2 = fieldNames.at(fieldData->selectedFieldIdx2);
        fieldData->useTwoFields = true;
    } else {
        fieldData->selectedFieldIdx1 = fieldData->selectedFieldIdx;
        fieldData->selectedScalarFieldName1 = fieldData->selectedScalarFieldName;
    }

    bool foundInsertionPosition = false;
    for (size_t i = 0; i < fieldDataArray.size(); i++) {
        if (fieldDataArray.at(i)->selectedFieldIdx > _selectedFieldIdx) {
            fieldDataArray.insert(fieldDataArray.begin() + ptrdiff_t(i), fieldData);
            //if (isTwoFieldMode) {
            //    fieldDataArray.insert(fieldDataArray.begin() + ptrdiff_t(i + 1), fieldData2);
            //}
            foundInsertionPosition = true;
            break;
        }
    }
    if (!foundInsertionPosition) {
        fieldDataArray.push_back(fieldData);
        //if (isTwoFieldMode) {
        //    fieldDataArray.push_back(fieldData2);
        //}
    }
    computeColorLegendHeight();
    clearFieldDeviceData();
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
    clearFieldDeviceData();
    dataDirty = true;
}

void HEBChart::setIsEnsembleMode(bool _isEnsembleMode) {
    if (isEnsembleMode != _isEnsembleMode) {
        isEnsembleMode = _isEnsembleMode;
        dataDirty = true;
        clearFieldDeviceData();
    }
}

void HEBChart::setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType) {
    correlationMeasureType = _correlationMeasureType;
    dataDirty = true;
}

void HEBChart::setUseAbsoluteCorrelationMeasure(bool _useAbsoluteCorrelationMeasure) {
    useAbsoluteCorrelationMeasure = _useAbsoluteCorrelationMeasure;
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

void HEBChart::setNumInitSamples(int _numInitSamples){
    numInitSamples = _numInitSamples;
    dataDirty = true;
}

void HEBChart::setNumBOIterations(int _numBOIterations){
    numBOIterations = _numBOIterations;
    dataDirty = true;
}

void HEBChart::setNloptAlgorithm(nlopt::algorithm _algorithm){
    algorithm = _algorithm;
    dataDirty = true;
}

void HEBChart::setDownscalingFactors(int _dfx, int _dfy, int _dfz) {
    dfx = _dfx;
    dfy = _dfy;
    dfz = _dfz;
    updateRegion();
    dataDirty = true;
}

void HEBChart::setUseCorrelationComputationGpu(bool _useGpu) {
    if (useCorrelationComputationGpu != _useGpu) {
        useCorrelationComputationGpu = _useGpu;
        dataDirty = true;
    }
}

void HEBChart::setDataMode(CorrelationDataMode _dataMode) {
    if (dataMode != _dataMode) {
        dataMode = _dataMode;
        dataDirty = true;
        clearFieldDeviceData();
    }
}

void HEBChart::setUseBufferTiling(bool _useBufferTiling) {
    if (useBufferTiling != _useBufferTiling) {
        useBufferTiling = _useBufferTiling;
        dataDirty = true;
        clearFieldDeviceData();
    }
}

void HEBChart::setShowVariablesForFieldIdxOnly(int _limitedFieldIdx) {
    limitedFieldIdx = _limitedFieldIdx;
    dataDirty = true;
}

void HEBChart::setOctreeMethod(OctreeMethod _octreeMethod) {
    if (octreeMethod != _octreeMethod) {
        octreeMethod = _octreeMethod;
        dataDirty = true;
    }
}

void HEBChart::setRegionWinding(RegionWinding _regionWinding) {
    if (regionWinding != _regionWinding) {
        regionWinding = _regionWinding;
        dataDirty = true;
    }
}


glm::vec2 HEBChart::getCorrelationRangeTotal() {
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
            || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        int cs = getCorrelationMemberCount();
        correlationRangeTotal.x = 0.0f;
        correlationRangeTotal.y = computeMaximumMutualInformationKraskov(k, cs);
    } else if (useAbsoluteCorrelationMeasure || isMeasureCorrelationCoefficientMI(correlationMeasureType)) {
        correlationRangeTotal.x = 0.0f;
        correlationRangeTotal.y = 1.0f;
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

        downscaledFields.at(fieldIdx) = downscaledField;
    }
}

void HEBChart::computeDownscaledFieldPerfTest(std::vector<float*>& downscaledFields) {
    computeDownscaledField(fieldDataArray.front().get(), 0, downscaledFields);
}

void HEBChart::computeDownscaledFieldVariance(HEBChartFieldData* fieldData, int idx, int varNum) {
    // Compute the standard deviation inside the downscaled grids.
    int cs = getCorrelationMemberCount();
    int xsd = idx == 0 ? xsd0 : xsd1;
    int ysd = idx == 0 ? ysd0 : ysd1;
    int zsd = idx == 0 ? zsd0 : zsd1;
    int numPoints = xsd * ysd * zsd;
    GridRegion r = idx == 0 ? r0 : r1;
    auto& pointToNodeIndexMap = idx == 0 ? pointToNodeIndexMap0 : pointToNodeIndexMap1;
    std::vector<float>& leafStdDevArray = varNum == 0 ? fieldData->leafStdDevArray : fieldData->leafStdDevArray2;
    leafStdDevArray.resize(xsd0 * ysd0 * zsd0 + (idx == 0 ? 0 : + xsd1 * ysd1 * zsd1));

    const std::string& selectedScalarFieldName =
            varNum == 0 ? fieldData->selectedScalarFieldName1 : fieldData->selectedScalarFieldName2;

    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(selectedScalarFieldName, fieldIdx);
        const float* field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
    }

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numPoints), [&](auto const& reg) {
            for (auto pointIdx = reg.begin(); pointIdx != reg.end(); pointIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel for default(none) shared(leafStdDevArray) \
    shared(fields, fieldData, pointToNodeIndexMap, xsd, ysd, zsd, r, numPoints, cs)
#endif
    for (int pointIdx = 0; pointIdx < numPoints; pointIdx++) {
#endif
        int gridNumValid = 0;
        float gridVarianceSum = 0.0f;
        int xd = pointIdx % xsd;
        int yd = (pointIdx / xsd) % ysd;
        int zd = pointIdx / (xsd * ysd);

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

        float stdDev = 0.0f;
        if (gridNumValid > 0) {
            gridVarianceSum /= float(gridNumValid);
            stdDev = std::sqrt(gridVarianceSum);
        } else {
            stdDev = std::numeric_limits<float>::quiet_NaN();
        }
        if (fieldData->useTwoFields) {
            stdDev *= 0.5f;
        }

        uint32_t leafIdx = pointToNodeIndexMap.at(pointIdx) - leafIdxOffset;
        leafStdDevArray.at(leafIdx) = stdDev;
    }
#ifdef USE_TBB
    });
#endif
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

void HEBChart::updateRegion() {
    xsd0 = sgl::iceil(r0.xsr, dfx);
    ysd0 = sgl::iceil(r0.ysr, dfy);
    zsd0 = sgl::iceil(r0.zsr, dfz);
    xsd1 = sgl::iceil(r1.xsr, dfx);
    ysd1 = sgl::iceil(r1.ysr, dfy);
    zsd1 = sgl::iceil(r1.zsr, dfz);
    if (xsd0 == 1 && ysd0 == 1 && zsd0 == 1) {
        dfx = 1;
        xsd0 = r0.xsr;
        dfy = 1;
        ysd0 = r0.ysr;
        dfz = 1;
        zsd0 = r0.zsr;
    }
    if (xsd1 == 1 &&  ysd1 == 1 && zsd1 == 1) {
        dfx = 1;
        xsd1 = r1.xsr;
        dfy = 1;
        ysd1 = r1.ysr;
        dfz = 1;
        zsd1 = r1.zsr;
    }

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    auto memoryHeapIndex = uint32_t(device->findMemoryHeapIndex(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT));
    size_t availableVram = device->getMemoryHeapBudgetVma(memoryHeapIndex);
    int cs = getCorrelationMemberCount();
    size_t sizeFields = size_t(xs) * size_t(ys) * size_t(zs) * size_t(cs) * sizeof(float);
    double budgetVram = double(availableVram) * 0.4;
    //double budgetVram = double(sizeFields) * 0.6;
    if (isUseMeanFieldsForced) {
        useMeanFields = mdfx > 1 || mdfy > 1 || mdfz > 1;
    } else {
        if (double(sizeFields) > budgetVram) {
            useMeanFields = true;
        } else {
            useMeanFields = false;
        }
        if (useMeanFields) {
            int xsr = std::max(r0.xsr, r1.xsr);
            int ysr = std::max(r0.ysr, r1.ysr);
            int zsr = std::max(r0.zsr, r1.zsr);
            size_t sizeFieldsAtLevel = size_t(xsr) * size_t(ysr) * size_t(zsr) * size_t(cs) * sizeof(float);
            auto f = std::max(int(std::ceil(std::cbrt(double(sizeFieldsAtLevel) / budgetVram))), 1);
            mdfx = f;
            mdfy = f;
            mdfz = f;
        } else {
            mdfx = 1;
            mdfy = 1;
            mdfz = 1;
        }
    }
}

void HEBChart::setForcedUseMeanFields(int fx, int fy, int fz) {
    isUseMeanFieldsForced = true;
    mdfx = fx;
    mdfy = fy;
    mdfz = fz;
    updateRegion();
}

void HEBChart::disableForcedUseMeanFields() {
    isUseMeanFieldsForced = false;
    updateRegion();
}

void HEBChart::updateData() {
    int cs = getCorrelationMemberCount();
    updateRegion();

    if (selectedLineIdx >= 0 || selectedPointIndices[0] >= 0 || selectedPointIndices[1] >= 0
            || selectedGridIdx.has_value()) {
        needsReRender = true;
    }
    resetSelectedPrimitives();

    std::vector<HEBChartFieldUpdateData> updateDataArray;
    if (diagramMode == DiagramMode::CHORD) {
        updateDataArray.resize(fieldDataArray.size());
    } else {
        curvePoints = {};
        correlationValuesArray = {};
        connectedPointsArray = {};
        lineFieldIndexArray = {};
    }
    correlationMatrix = {};
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
            if (fieldData->useTwoFields && regionsEqual) {
                std::vector<MIFieldEntry> miFieldEntries2;
                fieldData->isSecondFieldMode = true;
                computeCorrelations(fieldData, downscaledFields0, downscaledFields0, miFieldEntries2);
                fieldData->isSecondFieldMode = false;
                miFieldEntries.reserve(miFieldEntries.size() + miFieldEntries2.size());
                for (const auto& entry : miFieldEntries2) {
                    miFieldEntries.emplace_back(entry.correlationValue, entry.pointIndex1, entry.pointIndex0, true);
                }
            }
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
                octreeMethod, regionWinding,
                nodesList, pointToNodeIndexMap0, pointToNodeIndexMap1, leafIdxOffset, leafIdxOffset1,
                regionsEqual, xsd0, ysd0, zsd0, xsd1, ysd1, zsd1);

        // Compute the standard deviation inside the downscaled grids.
        computeDownscaledFieldVariance(fieldData, 0, 0);
        if (!regionsEqual) {
            computeDownscaledFieldVariance(fieldData, 1, 0);
        }
        if (fieldData->useTwoFields) {
            computeDownscaledFieldVariance(fieldData, 0, 1);
            if (!regionsEqual) {
                computeDownscaledFieldVariance(fieldData, 1, 1);
            }
        }
        // Normalize standard deviations for visualization.
        auto [minVal, maxVal] = sgl::reduceFloatArrayMinMax(fieldData->leafStdDevArray);
        fieldData->minStdDev = minVal;
        fieldData->maxStdDev = maxVal;
        if (fieldData->useTwoFields) {
            auto [minVal2, maxVal2] = sgl::reduceFloatArrayMinMax(fieldData->leafStdDevArray2);
            fieldData->minStdDev2 = minVal2;
            fieldData->maxStdDev2 = maxVal2;
        }

        if (diagramMode == DiagramMode::CHORD) {
            updateDataVarChord(fieldData, miFieldEntries, updateDataArray.at(i));
        } else {
            updateDataVarMatrix(fieldData, miFieldEntries);
            break;
        }
    }

    if (diagramMode == DiagramMode::CHORD) {
        updateDataChord(updateDataArray);
    }
}

void HEBChart::updateDataVarMatrix(
        HEBChartFieldData* fieldData, const std::vector<MIFieldEntry>& miFieldEntries) {
    if (!regionsEqual) {
        correlationMatrix = std::make_shared<FullCorrelationMatrix>(xsd0 * ysd0 * zsd0, xsd1 * ysd1 * zsd1);
    } else if (fieldData->useTwoFields) {
        correlationMatrix = std::make_shared<FullCorrelationMatrix>(xsd0 * ysd0 * zsd0, xsd0 * ysd0 * zsd0);
    } else {
        correlationMatrix = std::make_shared<SymmetricCorrelationMatrix>(xsd0 * ysd0 * zsd0);
    }

    if (!miFieldEntries.empty()) {
        fieldData->minCorrelationValue = miFieldEntries.back().correlationValue;
        fieldData->maxCorrelationValue = miFieldEntries.front().correlationValue;
    } else {
        fieldData->minCorrelationValue = std::numeric_limits<float>::max();
        fieldData->maxCorrelationValue = std::numeric_limits<float>::lowest();
    }
    minCorrelationValueGlobal = fieldData->minCorrelationValue;
    maxCorrelationValueGlobal = fieldData->maxCorrelationValue;

    for (const MIFieldEntry& entry : miFieldEntries) {
        int idx0 = int(pointToNodeIndexMap0.at(entry.pointIndex0) - leafIdxOffset);
        int idx1 = int(pointToNodeIndexMap1.at(entry.pointIndex1) - leafIdxOffset1);
        if (regionsEqual && entry.isSecondField != (idx0 > idx1)) {
            int tmp = idx0;
            idx0 = idx1;
            idx1 = tmp;
        }
        //correlationMatrix->set(int(entry.pointIndex0), int(entry.pointIndex1), entry.correlationValue);
        correlationMatrix->set(idx0, idx1, entry.correlationValue);
    }
}

void HEBChart::updateDataVarChord(
        HEBChartFieldData* fieldData, const std::vector<MIFieldEntry>& miFieldEntries,
        HEBChartFieldUpdateData& updateData) {
    int numPoints = xsd0 * ysd0 * zsd0 + (regionsEqual ? 0 : xsd1 * ysd1 * zsd1);
    int maxNumLines = numPoints * MAX_NUM_LINES / 100;
    int numLinesLocal = std::min(maxNumLines, int(miFieldEntries.size()));
    numLinesTotal += numLinesLocal;
    std::vector<glm::vec2>& curvePointsLocal = updateData.curvePoints;
    std::vector<float>& correlationValuesArrayLocal = updateData.correlationValuesArray;
    std::vector<std::pair<int, int>>& connectedPointsArrayLocal = updateData.connectedPointsArray;
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
    tbb::parallel_for(tbb::blocked_range<int>(0, numLinesLocal), [&](auto const& r) {
            std::vector<glm::vec2> controlPoints;
            for (auto lineIdx = r.begin(); lineIdx != r.end(); lineIdx++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(miFieldEntries, numLinesLocal) \
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

void HEBChart::updateDataChord(std::vector<HEBChartFieldUpdateData>& updateDataArray) {
    auto numFields = int(fieldDataArray.size());
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
    aabb.min = glm::vec3(r.xoff + xd * dfx, r.yoff + yd * dfy, r.zoff + zd * dfz);
    aabb.max = glm::vec3(
            float(std::min(r.xoff + int(xd + 1) * dfx, r.xmax + 1)),
            float(std::min(r.yoff + int(yd + 1) * dfy, r.ymax + 1)),
            float(std::min(r.zoff + int(zd + 1) * dfz, r.zmax + 1)));
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
    clickedGridIdxOld = {};
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
            || (clickedLineIdx >= 0 && clickedLineIdx != clickedLineIdxOld)
            || (clickedGridIdx.has_value() && clickedGridIdx != clickedGridIdxOld);
    if (hasNewFocusSelection) {
        std::pair<GridRegion, GridRegion> regions = getFocusSelection();
        if (regions.first.getNumCells() <= 1 && regions.second.getNumCells() <= 1) {
            hasNewFocusSelection = false;
        }
    }
    isDeselection =
            !hasNewFocusSelection
            && (clickedPointIdx != clickedPointIdxOld || clickedLineIdx != clickedLineIdxOld
                    || clickedGridIdx != clickedGridIdxOld);
    clickedPointIdxOld = clickedPointIdx;
    clickedLineIdxOld = clickedLineIdx;
    clickedGridIdxOld = clickedGridIdx;
    return hasNewFocusSelection;
}

std::pair<GridRegion, GridRegion> HEBChart::getFocusSelection() {
    if (clickedPointIdx != -1) {
        uint32_t pointIdx = getPointIndexGrid(clickedPointIdx);
        auto pointRegion = getGridRegionPointIdx(getLeafIdxGroup(clickedPointIdx), pointIdx);
        return std::make_pair(pointRegion, pointRegion);
    } else if (diagramMode == DiagramMode::CHORD) {
        auto points = connectedPointsArray.at(clickedLineIdx);
        uint32_t pointIdx0 = getPointIndexGrid(points.first);
        uint32_t pointIdx1 = getPointIndexGrid(points.second);
        return std::make_pair(
                getGridRegionPointIdx(getLeafIdxGroup(points.first), pointIdx0),
                getGridRegionPointIdx(getLeafIdxGroup(points.second), pointIdx1));
    } else if (diagramMode == DiagramMode::MATRIX) {
        uint32_t pointIdx0 = getPointIndexGrid(clickedGridIdx->x);
        uint32_t pointIdx1 = getPointIndexGrid(clickedGridIdx->y);
        return std::make_pair(
                getGridRegionPointIdx(getLeafIdxGroup(clickedGridIdx->x), pointIdx0),
                getGridRegionPointIdx(getLeafIdxGroup(clickedGridIdx->y), pointIdx1));
    }
    return {};
}

GridRegion HEBChart::getGridRegionPointIdx(int idx, uint32_t pointIdx) {
    int xsd = idx == 0 ? xsd0 : xsd1;
    int ysd = idx == 0 ? ysd0 : ysd1;
    //int zsd = idx == 0 ? zsd0 : zsd1;
    int xd = int(pointIdx % uint32_t(xsd));
    int yd = int((pointIdx / uint32_t(xsd)) % uint32_t(ysd));
    int zd = int(pointIdx / uint32_t(xsd * ysd));
    GridRegion r = idx == 0 ? r0 : r1;
    GridRegion rf = GridRegion(
            r.xoff + xd * dfx, r.yoff + yd * dfy, r.zoff + zd * dfz,
            std::min((xd + 1) * dfx, r.xsr) - xd * dfx,
            std::min((yd + 1) * dfy, r.ysr) - yd * dfy,
            std::min((zd + 1) * dfz, r.zsr) - zd * dfz);
    return rf;
}

int HEBChart::getLeafIdxGroup(int leafIdx) const {
    if (regionsEqual) {
        return 0;
    }
    return leafIdx >= int(leafIdxOffset1 - leafIdxOffset) ? 1 : 0;
}

bool HEBChart::getHasFocusSelectionField() {
    return clickedLineIdx >= 0 || clickedGridIdx.has_value();
}

int HEBChart::getFocusSelectionFieldIndex() {
    if (diagramMode == DiagramMode::CHORD) {
        return fieldDataArray.at(lineFieldIndexArray.at(clickedLineIdx))->selectedFieldIdx;
    } else {
        return fieldDataArray.front()->selectedFieldIdx;
    }
}
