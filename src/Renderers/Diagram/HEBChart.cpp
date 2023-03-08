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
#include "Calculators/Correlation.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Volume/VolumeData.hpp"
#include "BSpline.hpp"
#include "HEBChart.hpp"

void HEBChart::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;
    if (selectedFieldIdx >= int(volumeData->getFieldNamesBase(FieldType::SCALAR).size())) {
        dataDirty = true;
    }
    if (isNewData) {
        xs = volumeData->getGridSizeX();
        ys = volumeData->getGridSizeY();
        zs = volumeData->getGridSizeZ();
        r = GridRegion(0, 0, 0, xs, ys, zs);
        xsd = sgl::iceil(xs, dfx);
        ysd = sgl::iceil(ys, dfy);
        zsd = sgl::iceil(zs, dfz);
        resetSelectedPrimitives();
    }
}

void HEBChart::setRegions(const std::pair<GridRegion, GridRegion>& _rs) {
    // TODO
    r = _rs.first;
    xsd = sgl::iceil(r.xsr, dfx);
    ysd = sgl::iceil(r.ysr, dfy);
    zsd = sgl::iceil(r.zsr, dfz);
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

void HEBChart::setSelectedScalarField(int _selectedFieldIdx, const std::string& _scalarFieldName) {
    selectedFieldIdx = _selectedFieldIdx;
    selectedScalarFieldName = _scalarFieldName;
    dataDirty = true;
}

void HEBChart::setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType) {
    correlationMeasureType = _correlationMeasureType;
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


glm::vec2 HEBChart::getCorrelationRangeTotal() {
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
            || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        int es = volumeData->getEnsembleMemberCount();
        int k = std::max(sgl::iceil(3 * es, 100), 1);
        correlationRangeTotal.x = 0.0f;
        correlationRangeTotal.y = computeMaximumMutualInformationKraskov(k, es);
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
    glm::vec3 ptj(xsd - 1, ysd - 1, zsd - 1);
    cellDistanceRangeTotal.y = int(std::ceil(glm::length(pti - ptj)));
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


void HEBChart::computeDownscaledField(std::vector<float*>& downscaledEnsembleFields) {
    int es = volumeData->getEnsembleMemberCount();
    int numPoints = xsd * ysd * zsd;
    for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, selectedScalarFieldName, -1, ensembleIdx);
        float* field = ensembleEntryField.get();
        auto* dowsncaledField = new float[numPoints];

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
                        dowsncaledField[IDXSD(xd, yd, zd)] = valueMean;
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
                    dowsncaledField[IDXSD(xd, yd, 0)] = valueMean;
                }
            }
        }

        downscaledEnsembleFields.at(ensembleIdx) = dowsncaledField;
    }
}

void HEBChart::computeDownscaledFieldVariance(std::vector<float*>& downscaledEnsembleFields) {
    // Compute the standard deviation inside the downscaled grids.
    int es = volumeData->getEnsembleMemberCount();
    int numPoints = xsd * ysd * zsd;
    leafStdDevArray.resize(numPoints);

    std::vector<VolumeData::HostCacheEntry> ensembleEntryFields;
    std::vector<float*> fields;
    for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, selectedScalarFieldName, -1, ensembleIdx);
        float* field = ensembleEntryField.get();
        ensembleEntryFields.push_back(ensembleEntryField);
        fields.push_back(field);
    }

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, numPoints), [&](auto const& r) {
            for (auto pointIdx = r.begin(); pointIdx != r.end(); pointIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel for default(none) shared(fields, numPoints, es)
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
                            float ensembleMean = 0.0f;
                            for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
                                float* field = fields.at(ensembleIdx);
                                float val = field[IDXS(x, y, z)];
                                if (!std::isnan(val)) {
                                    ensembleMean += val;
                                    numValid++;
                                }
                            }
                            if (numValid > 1) {
                                ensembleMean = ensembleMean / float(numValid);
                                float ensembleVarianceSum = 0.0f;
                                for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
                                    float* field = fields.at(ensembleIdx);
                                    float val = field[IDXS(x, y, z)];
                                    if (!std::isnan(val)) {
                                        float diff = ensembleMean - val;
                                        ensembleVarianceSum += diff * diff;
                                    }
                                }
                                gridVarianceSum += ensembleVarianceSum / float(numValid - 1);
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
                        float ensembleMean = 0.0f;
                        for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
                            float* field = fields.at(ensembleIdx);
                            float val = field[IDXS(x, y, zCenter)];
                            if (!std::isnan(val)) {
                                ensembleMean += val;
                                numValid++;
                            }
                        }
                        if (numValid > 1) {
                            ensembleMean = ensembleMean / float(numValid);
                            float ensembleVarianceSum = 0.0f;
                            for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
                                float* field = fields.at(ensembleIdx);
                                float val = field[IDXS(x, y, zCenter)];
                                if (!std::isnan(val)) {
                                    float diff = ensembleMean - val;
                                    ensembleVarianceSum += diff * diff;
                                }
                            }
                            gridVarianceSum += ensembleVarianceSum / float(numValid - 1);
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
        leafStdDevArray.at(leafIdx) = stdDev;
    }
#ifdef USE_TBB
    });
#endif

    // Normalize standard deviations for visualization.
    auto [minVal, maxVal] = sgl::reduceFloatArrayMinMax(leafStdDevArray);
    minStdDev = minVal;
    maxStdDev = maxVal;
}

void HEBChart::computeCorrelations(
        std::vector<float*>& downscaledEnsembleFields, std::vector<MIFieldEntry>& miFieldEntries) {
    int es = volumeData->getEnsembleMemberCount();
    int k = std::max(sgl::iceil(3 * es, 100), 1); //< CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
    int numBins = 80; //< CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
    int numPoints = xsd * ysd * zsd;
#ifdef USE_TBB
    miFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPoints), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const& r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                std::vector<float> X(es);
                std::vector<float> Y(es);

                std::vector<std::pair<float, int>> ordinalRankArraySpearman;
                float* referenceRanks = nullptr;
                float* gridPointRanks = nullptr;
                if (cmt == CorrelationMeasureType::SPEARMAN) {
                    ordinalRankArraySpearman.reserve(es);
                    referenceRanks = new float[es];
                    gridPointRanks = new float[es];
                }

                std::vector<std::pair<float, float>> jointArray;
                std::vector<float> ordinalRankArray;
                std::vector<float> y;
                if (cmt == CorrelationMeasureType::KENDALL) {
                    jointArray.reserve(es);
                    ordinalRankArray.reserve(es);
                    y.reserve(es);
                }

                double* histogram0 = nullptr;
                double* histogram1 = nullptr;
                double* histogram2d = nullptr;
                if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                    histogram0 = new double[numBins];
                    histogram1 = new double[numBins];
                    histogram2d = new double[numBins * numBins];
                }

                float minEnsembleValRef, maxEnsembleValRef, minEnsembleVal, maxEnsembleVal;

                for (int i = r.begin(); i != r.end(); i++) {
#else
    miFieldEntries.reserve((numPoints * numPoints + numPoints) / 2);
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries, downscaledEnsembleFields, numPoints, es, k, numBins)
#endif
    {
        const CorrelationMeasureType cmt = correlationMeasureType;
        std::vector<MIFieldEntry> miFieldEntriesThread;
        std::vector<float> X(es);
        std::vector<float> Y(es);

        std::vector<std::pair<float, int>> ordinalRankArraySpearman;
        float* referenceRanks = nullptr;
        float* gridPointRanks = nullptr;
        if (cmt == CorrelationMeasureType::SPEARMAN) {
            ordinalRankArraySpearman.reserve(es);
            referenceRanks = new float[es];
            gridPointRanks = new float[es];
        }

        std::vector<std::pair<float, float>> jointArray;
        std::vector<float> ordinalRankArray;
        std::vector<float> y;
        if (cmt == CorrelationMeasureType::KENDALL) {
            jointArray.reserve(es);
            ordinalRankArray.reserve(es);
            y.reserve(es);
        }

        double* histogram0 = nullptr;
        double* histogram1 = nullptr;
        double* histogram2d = nullptr;
        if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            histogram0 = new double[numBins];
            histogram1 = new double[numBins];
            histogram2d = new double[numBins * numBins];
        }

        float minEnsembleValRef, maxEnsembleValRef, minEnsembleVal, maxEnsembleVal;

#if _OPENMP >= 201107
    #pragma omp for schedule(dynamic)
#endif
        for (int i = 0; i < numPoints; i++) {
#endif
            bool isNan = false;
            for (int e = 0; e < es; e++) {
                X[e] = downscaledEnsembleFields.at(e)[i];
                if (std::isnan(X[e])) {
                    isNan = true;
                    break;
                }
            }
            if (isNan) {
                continue;
            }
            if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
                std::vector<std::pair<float, int>> ordinalRankArrayRef;
                ordinalRankArray.reserve(es);
                computeRanks(X.data(), referenceRanks, ordinalRankArrayRef, es);
            }
            if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                minEnsembleValRef = std::numeric_limits<float>::max();
                maxEnsembleValRef = std::numeric_limits<float>::lowest();
                for (int e = 0; e < es; e++) {
                    minEnsembleValRef = std::min(minEnsembleValRef, X[e]);
                    maxEnsembleValRef = std::max(maxEnsembleValRef, X[e]);
                }
                for (int e = 0; e < es; e++) {
                    X[e] = (downscaledEnsembleFields.at(e)[i] - minEnsembleValRef) / (maxEnsembleValRef - minEnsembleValRef);
                }
            }

            for (int j = 0; j < i; j++) {
                if (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y) {
                    glm::vec3 pti(i % uint32_t(xsd), (i / uint32_t(xsd)) % uint32_t(ysd), i / uint32_t(xsd * ysd));
                    glm::vec3 ptj(j % uint32_t(xsd), (j / uint32_t(xsd)) % uint32_t(ysd), j / uint32_t(xsd * ysd));
                    float cellDist = glm::length(pti - ptj);
                    if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                        continue;
                    }
                }
                for (int e = 0; e < es; e++) {
                    Y[e] = downscaledEnsembleFields.at(e)[j];
                    if (std::isnan(Y[e])) {
                        isNan = true;
                        break;
                    }
                }
                if (!isNan) {
                    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                        minEnsembleVal = minEnsembleValRef;
                        maxEnsembleVal = maxEnsembleValRef;
                        for (int e = 0; e < es; e++) {
                            minEnsembleVal = std::min(minEnsembleVal, Y[e]);
                            maxEnsembleVal = std::max(maxEnsembleVal, Y[e]);
                        }
                        for (int e = 0; e < es; e++) {
                            X[e] = (downscaledEnsembleFields.at(e)[i] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
                            Y[e] = (downscaledEnsembleFields.at(e)[j] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
                        }
                    }

                    float miValue = 0.0f;
                    if (cmt == CorrelationMeasureType::PEARSON) {
                        miValue = computePearson2<float>(X.data(), Y.data(), es);
                    } else if (cmt == CorrelationMeasureType::SPEARMAN) {
                        computeRanks(Y.data(), gridPointRanks, ordinalRankArraySpearman, es);
                        miValue = computePearson2<float>(referenceRanks, gridPointRanks, es);
                    } else if (cmt == CorrelationMeasureType::KENDALL) {
                        miValue = computeKendall(
                                X.data(), Y.data(), es, jointArray, ordinalRankArray, y);
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                        miValue = computeMutualInformationBinned<double>(
                                X.data(), Y.data(), numBins, es, histogram0, histogram1, histogram2d);
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                        miValue = computeMutualInformationKraskov<double>(X.data(), Y.data(), k, es);
                    }
                    if (miValue < correlationRange.x || miValue > correlationRange.y) {
                        continue;
                    }
                    miFieldEntriesThread.emplace_back(miValue, i, j);
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
#else
            for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
            #pragma omp ordered
            {
                miFieldEntries.insert(miFieldEntries.end(), miFieldEntriesThread.begin(), miFieldEntriesThread.end());
            }
        }
    }
#endif

#ifdef USE_TBB
    tbb::parallel_sort(miFieldEntries.begin(), miFieldEntries.end());
//#elif __cpp_lib_parallel_algorithm >= 201603L
    //std::sort(std::execution::par_unseq, miFieldEntries.begin(), miFieldEntries.end());
#else
    std::sort(miFieldEntries.begin(), miFieldEntries.end());
#endif
}

void getControlPoints(
        const std::vector<HEBNode>& nodesList, const std::vector<uint32_t>& pointToNodeIndexMap,
        uint32_t pointIndex0, uint32_t pointIndex1, std::vector<glm::vec2>& controlPoints) {
    // The start nodes are leaves at the same level.
    uint32_t nidx0 = pointToNodeIndexMap.at(pointIndex0);
    uint32_t nidx1 = pointToNodeIndexMap.at(pointIndex1);

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
    int es = volumeData->getEnsembleMemberCount();
    xsd = sgl::iceil(r.xsr, dfx);
    ysd = sgl::iceil(r.ysr, dfy);
    zsd = sgl::iceil(r.zsr, dfz);
    int numPoints = xsd * ysd * zsd;

    if (selectedLineIdx >= 0 || selectedPointIndices[0] >= 0 || selectedPointIndices[1] >= 0) {
        needsReRender = true;
    }
    resetSelectedPrimitives();

    if (use2dField) {
        numPoints = xsd * ysd;
        zsd = 1;
    }

    // Compute the downscaled field.
    std::vector<float*> downscaledEnsembleFields;
    downscaledEnsembleFields.resize(es);
    computeDownscaledField(downscaledEnsembleFields);

    // Compute the correlation matrix.
    std::vector<MIFieldEntry> miFieldEntries;
    computeCorrelations(downscaledEnsembleFields, miFieldEntries);

    // Build the octree.
    nodesList.clear();
    pointToNodeIndexMap.clear();
    buildHebTree(nodesList, pointToNodeIndexMap, leafIdxOffset, xsd, ysd, zsd);

    // Compute the standard deviation inside the downscaled grids.
    computeDownscaledFieldVariance(downscaledEnsembleFields);

    // Delete the downscaled field, as it is no longer used.
    for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        float* dowsncaledField = downscaledEnsembleFields.at(ensembleIdx);
        delete[] dowsncaledField;
    }
    downscaledEnsembleFields.clear();

    int maxNumLines = numPoints * MAX_NUM_LINES / 100;
    NUM_LINES = std::min(maxNumLines, int(miFieldEntries.size()));
    curvePoints.resize(NUM_LINES * NUM_SUBDIVISIONS);
    miValues.resize(NUM_LINES);
    connectedPointsArray.resize(NUM_LINES);
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, NUM_LINES), [&](auto const& r) {
        std::vector<glm::vec2> controlPoints;
        for (auto lineIdx = r.begin(); lineIdx != r.end(); lineIdx++) {
#else
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries)
#endif
    {
        std::vector<glm::vec2> controlPoints;
#if _OPENMP >= 201107
        #pragma omp for
#endif
        for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
#endif
            // Use reverse order so lines are drawn from least to most important.
            const auto& miEntry = miFieldEntries.at(NUM_LINES - lineIdx - 1);
            miValues.at(lineIdx) = miEntry.miValue;

            auto idx0 = int(pointToNodeIndexMap.at(miEntry.pointIndex0) - leafIdxOffset);
            auto idx1 = int(pointToNodeIndexMap.at(miEntry.pointIndex1) - leafIdxOffset);
            connectedPointsArray.at(lineIdx) = std::make_pair(idx0, idx1);

            controlPoints.clear();
            getControlPoints(nodesList, pointToNodeIndexMap, miEntry.pointIndex0, miEntry.pointIndex1, controlPoints);
            if (beta < 1.0f) {
                smoothControlPoints(controlPoints, beta);
            }
            for (int ptIdx = 0; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                float t = float(ptIdx) / float(NUM_SUBDIVISIONS - 1);
                int k = 4;
                if (controlPoints.size() == 3) {
                    k = 3;
                }
                curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx) = evaluateBSpline(t, k, controlPoints);
            }
        }
#ifdef USE_TBB
    });
#else
    }
#endif
}

bool HEBChart::getIsRegionSelected(int idx) {
    return selectedPointIndices[idx] >= 0;
}

uint32_t HEBChart::getPointIndexGrid(int pointIdx) {
    return uint32_t(std::find(
            pointToNodeIndexMap.begin(), pointToNodeIndexMap.end(),
            int(leafIdxOffset) + pointIdx) - pointToNodeIndexMap.begin());
}

uint32_t HEBChart::getSelectedPointIndexGrid(int idx) {
    return uint32_t(std::find(
            pointToNodeIndexMap.begin(), pointToNodeIndexMap.end(),
            int(leafIdxOffset) + selectedPointIndices[idx]) - pointToNodeIndexMap.begin());
}

sgl::AABB3 HEBChart::getSelectedRegion(int idx) {
    //const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedCircleIdx);
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
    auto pointIdx0 = getSelectedPointIndexGrid(0);
    uint32_t xd0 = pointIdx0 % uint32_t(xsd);
    uint32_t yd0 = (pointIdx0 / uint32_t(xsd)) % uint32_t(ysd);
    uint32_t zd0 = pointIdx0 / uint32_t(xsd * ysd);
    glm::ivec3 c0((int)xd0, (int)yd0, (int)zd0);
    auto pointIdx1 = getSelectedPointIndexGrid(1);
    uint32_t xd1 = pointIdx1 % uint32_t(xsd);
    uint32_t yd1 = (pointIdx1 / uint32_t(xsd)) % uint32_t(ysd);
    uint32_t zd1 = pointIdx1 / uint32_t(xsd * ysd);
    glm::ivec3 c1((int)xd1, (int)yd1, (int)zd1);

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

bool HEBChart::getHasNewFocusSelection(bool& isDeselection) {
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
        auto pointRegion = getGridRegionPointIdx(pointIdx);
        return std::make_pair(pointRegion, pointRegion);
    } else {
        const auto& points = connectedPointsArray.at(clickedLineIdx);
        uint32_t pointIdx0 = getPointIndexGrid(points.first);
        uint32_t pointIdx1 = getPointIndexGrid(points.second);
        return std::make_pair(getGridRegionPointIdx(pointIdx0), getGridRegionPointIdx(pointIdx1));
    }
}

GridRegion HEBChart::getGridRegionPointIdx(uint32_t pointIdx) {
    int xd = int(pointIdx % uint32_t(xsd));
    int yd = int((pointIdx / uint32_t(xsd)) % uint32_t(ysd));
    int zd = int(pointIdx / uint32_t(xsd * ysd));
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
