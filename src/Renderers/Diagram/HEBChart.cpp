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

#ifdef SUPPORT_SKIA
#include <core/SkCanvas.h>
#include <core/SkPaint.h>
#include <core/SkPath.h>
#endif
#ifdef SUPPORT_VKVG
#include <vkvg.h>
#endif

#include <Math/Math.hpp>
#include <Math/Geometry/Circle.hpp>
#include <Graphics/Vector/nanovg/nanovg.h>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>
#include <Input/Mouse.hpp>
#include <Input/Keyboard.hpp>

#include "Loaders/DataSet.hpp"
#include "Calculators/Correlation.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Volume/VolumeData.hpp"
#ifdef SUPPORT_SKIA
#include "VectorBackendSkia.hpp"
#endif
#ifdef SUPPORT_VKVG
#include "VectorBackendVkvg.hpp"
#endif
#include "BSpline.hpp"
#include "HEBChart.hpp"

#define IDXSD(x,y,z) ((z)*xsd*ysd + (y)*xsd + (x))

struct MIFieldEntry {
    float miValue;
    uint32_t pointIndex0, pointIndex1;

    MIFieldEntry(float miValue, uint32_t pointIndex0, uint32_t pointIndex1)
            : miValue(miValue), pointIndex0(pointIndex0), pointIndex1(pointIndex1) {}
    bool operator<(const MIFieldEntry& rhs) const { return miValue > rhs.miValue; }
};

struct StackDomain {
    StackDomain() {}
    StackDomain(uint32_t nodeIdx, uint32_t depth, const glm::ivec3& min, const glm::ivec3& max)
            : nodeIdx(nodeIdx), depth(depth), min(min), max(max) {}
    uint32_t nodeIdx;
    uint32_t depth;
    glm::ivec3 min, max;
};

void buildTree(
        std::vector<HEBNode>& nodesList, std::vector<uint32_t>& pointToNodeIndexMap, uint32_t& leafIdxOffset,
        int xsd, int ysd, int zsd) {
    auto treeHeightX = uint32_t(std::ceil(std::log2(xsd)));
    auto treeHeightY = uint32_t(std::ceil(std::log2(ysd)));
    auto treeHeightZ = uint32_t(std::ceil(std::log2(zsd)));
    auto treeHeight = std::max(treeHeightX, std::max(treeHeightY, treeHeightZ));
    nodesList.emplace_back();
    nodesList[0].normalizedPosition = glm::vec3(0.0f);
    pointToNodeIndexMap.resize(xsd * ysd * zsd);

    std::queue<StackDomain> domainStack;
    StackDomain rootDomain;
    rootDomain.nodeIdx = 0;
    rootDomain.depth = 0;
    rootDomain.min = glm::ivec3(0, 0, 0);
    rootDomain.max = glm::ivec3(xsd - 1, ysd - 1, zsd - 1);
    domainStack.push(rootDomain);
    leafIdxOffset = std::numeric_limits<uint32_t>::max();
    while (!domainStack.empty()) {
        auto stackEntry = domainStack.front();
        domainStack.pop();
        auto extent = stackEntry.max - stackEntry.min + glm::ivec3(1);
        // Leaf?
        //if (extent.x == 1 && extent.y == 1 && extent.z == 1) {
        if (stackEntry.depth == treeHeight) {
            pointToNodeIndexMap.at(IDXSD(stackEntry.min.x, stackEntry.min.y, stackEntry.min.z)) = stackEntry.nodeIdx;
            if (leafIdxOffset == std::numeric_limits<uint32_t>::max()) {
                leafIdxOffset = stackEntry.nodeIdx;
            }
            continue;
        }
        glm::ivec3 maxHalf = stackEntry.max, minHalf = stackEntry.min;
        minHalf.x = stackEntry.min.x + sgl::iceil(extent.x, 2);
        minHalf.y = stackEntry.min.y + sgl::iceil(extent.y, 2);
        minHalf.z = stackEntry.min.z + sgl::iceil(extent.z, 2);
        maxHalf.x = minHalf.x - 1;
        maxHalf.y = minHalf.y - 1;
        maxHalf.z = minHalf.z - 1;
        auto childrenOffset = uint32_t(nodesList.size());
        domainStack.emplace(
                uint32_t(nodesList.size()), stackEntry.depth + 1,
                glm::vec3(stackEntry.min.x, stackEntry.min.y, stackEntry.min.z),
                glm::vec3(maxHalf.x, maxHalf.y, maxHalf.z));
        HEBNode child(stackEntry.nodeIdx);
        nodesList.push_back(child);
        if (extent.x > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(minHalf.x, stackEntry.min.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, maxHalf.y, maxHalf.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
        }
        if (extent.y > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(stackEntry.min.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(maxHalf.x, stackEntry.max.y, maxHalf.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
        }
        if (extent.x > 1 && extent.y > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(minHalf.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, stackEntry.max.y, maxHalf.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
        }
        if (extent.z > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(stackEntry.min.x, stackEntry.min.y, minHalf.z),
                    glm::vec3(maxHalf.x, maxHalf.y, stackEntry.max.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
            if (extent.x > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1,
                        glm::vec3(minHalf.x, stackEntry.min.y, minHalf.z),
                        glm::vec3(stackEntry.max.x, maxHalf.y, stackEntry.max.z));
                HEBNode child(stackEntry.nodeIdx);
                nodesList.push_back(child);
            }
            if (extent.y > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1,
                        glm::vec3(stackEntry.min.x, minHalf.y, minHalf.z),
                        glm::vec3(maxHalf.x, stackEntry.max.y, stackEntry.max.z));
                HEBNode child(stackEntry.nodeIdx);
                nodesList.push_back(child);
            }
            if (extent.x > 1 && extent.y > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1,
                        glm::vec3(minHalf.x, minHalf.y, minHalf.z),
                        glm::vec3(stackEntry.max.x, stackEntry.max.y, stackEntry.max.z));
                HEBNode child(stackEntry.nodeIdx);
                nodesList.push_back(child);
            }
        }
        uint32_t numChildren = uint32_t(nodesList.size()) - childrenOffset;
        for (uint32_t i = 0; i < numChildren; i++) {
            nodesList[stackEntry.nodeIdx].childIndices[i] = childrenOffset + i;
        }
    }



    // Set node positions.
    // Start with placing the leaves on a unit circle.
    std::unordered_set<uint32_t> prevParentNodeIndices;
    std::unordered_set<uint32_t> nextParentNodeIndices;
    uint32_t leafCounter = 0;
    for (uint32_t leafIdx = leafIdxOffset; leafIdx < uint32_t(nodesList.size()); leafIdx++) {
        prevParentNodeIndices.insert(nodesList[leafIdx].parentIdx);
        float angle = float(leafCounter) / float(pointToNodeIndexMap.size()) * sgl::TWO_PI;
        nodesList[leafIdx].angle = angle;
        nodesList[leafIdx].normalizedPosition = glm::vec2(std::cos(angle), std::sin(angle));
        prevParentNodeIndices.insert(nodesList[leafIdx].parentIdx);
        leafCounter++;
    }
    /*std::stack<uint32_t> traversalStack;
    while (!traversalStack.empty()) {
        uint32_t nodeIdx = traversalStack.top();
        traversalStack.pop();
    }*/

    int currentDepth = int(treeHeight) - 1;
    while (!prevParentNodeIndices.empty()) {
        float radius = float(currentDepth) / float(treeHeight);
        for (uint32_t nodeIdx : prevParentNodeIndices) {
            auto& node = nodesList[nodeIdx];
            float minChildAngle = std::numeric_limits<float>::max();
            float maxChildAngle = std::numeric_limits<float>::lowest();
            for (int i = 0; i < 8; i++) {
                if (node.childIndices[i] == std::numeric_limits<uint32_t>::max()) {
                    break;
                }
                minChildAngle = std::min(minChildAngle, nodesList[node.childIndices[i]].angle);
                maxChildAngle = std::max(maxChildAngle, nodesList[node.childIndices[i]].angle);
            }
            node.angle = 0.5f * (minChildAngle + maxChildAngle);
            node.normalizedPosition = radius * glm::vec2(std::cos(node.angle), std::sin(node.angle));
            if (node.parentIdx != std::numeric_limits<uint32_t>::max()) {
                nextParentNodeIndices.insert(node.parentIdx);
            }
        }

        prevParentNodeIndices = nextParentNodeIndices;
        nextParentNodeIndices.clear();
        currentDepth--;
    }
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

HEBChart::HEBChart() {
#ifdef SUPPORT_SKIA
    registerRenderBackendIfSupported<VectorBackendSkia>([this]() { this->renderBaseSkia(); });
#endif
#ifdef SUPPORT_VKVG
    registerRenderBackendIfSupported<VectorBackendVkvg>([this]() { this->renderBaseVkvg(); });
#endif

    std::string defaultBackendId = sgl::VectorBackendNanoVG::getClassID();
#if defined(SUPPORT_SKIA) || defined(SUPPORT_VKVG)
    // NanoVG Vulkan port is broken at the moment, so use Skia or VKVG if OpenGL NanoVG cannot be used.
    if (!sgl::AppSettings::get()->getOffscreenContext()) {
#if defined(SUPPORT_SKIA)
        defaultBackendId = VectorBackendSkia::getClassID();
#elif defined(SUPPORT_VKVG)
        defaultBackendId = VectorBackendVkvg::getClassID();
#endif
    }
#endif
    setDefaultBackendId(defaultBackendId);
}

void HEBChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    chartRadius = 160;
    windowWidth = (chartRadius + borderSizeX) * 2.0f;
    windowHeight = (chartRadius + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onWindowSizeChanged();
}

void HEBChart::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;
    if (selectedFieldIdx >= int(volumeData->getFieldNamesBase(FieldType::SCALAR).size())) {
        dataDirty = true;
    }
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

void HEBChart::setBeta(float _beta) {
    beta = _beta;
    dataDirty = true;
}

void HEBChart::setDownscalingFactor(int _df) {
    df = _df;
    dataDirty = true;
}

void HEBChart::setLineCountFactor(int _factor) {
    MAX_NUM_LINES = _factor;
    dataDirty = true;
}

void HEBChart::setCurveOpacity(float _alpha) {
    curveOpacity = _alpha;
}

void HEBChart::setCellDistanceThreshold(int _thresh) {
    cellDistanceThreshold = _thresh;
    dataDirty = true;
}

void HEBChart::setDiagramRadius(int radius) {
    chartRadius = float(radius);
    windowWidth = (chartRadius + borderSizeX) * 2.0f;
    windowHeight = (chartRadius + borderSizeY) * 2.0f;
    onWindowSizeChanged();
}

void HEBChart::setUse2DField(bool _use2dField) {
    use2dField = _use2dField;
    dataDirty = true;
}

void HEBChart::updateData() {
    // Values downscaled by factor 32.
    int es = volumeData->getEnsembleMemberCount();
    xs = volumeData->getGridSizeX();
    ys = volumeData->getGridSizeY();
    zs = volumeData->getGridSizeZ();
    xsd = sgl::iceil(xs, df);
    ysd = sgl::iceil(ys, df);
    zsd = sgl::iceil(zs, df);
    int numPoints = xsd * ysd * zsd;

    if (use2dField) {
        numPoints = xsd * ysd;
        zsd = 1;
    }

    // Compute the downscaled field.
    std::vector<float*> downscaledEnsembleFields;
    downscaledEnsembleFields.resize(es);
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
                        for (int zo = 0; zo < df; zo++) {
                            for (int yo = 0; yo < df; yo++) {
                                for (int xo = 0; xo < df; xo++) {
                                    int x = xd * df + xo;
                                    int y = yd * df + yo;
                                    int z = zd * df + zo;
                                    if (x < xs && y < ys && z < zs) {
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
                    for (int yo = 0; yo < df; yo++) {
                        for (int xo = 0; xo < df; xo++) {
                            int x = xd * df + xo;
                            int y = yd * df + yo;
                            if (x < xs && y < ys) {
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

    // Compute the mutual information matrix.
    int k = std::max(sgl::iceil(3 * es, 100), 1); //< CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
    int numBins = 80; //< CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
#ifdef USE_TBB
    std::vector<MIFieldEntry> miFieldEntries = tbb::parallel_reduce(
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
    std::vector<MIFieldEntry> miFieldEntries;
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
                if (cellDistanceThreshold > 0) {
                    glm::vec3 pti(i % uint32_t(xsd), (i / uint32_t(xsd)) % uint32_t(ys), i / uint32_t(xsd * ysd));
                    glm::vec3 ptj(j % uint32_t(xsd), (j / uint32_t(xsd)) % uint32_t(ys), j / uint32_t(xsd * ysd));
                    if (glm::length(pti - ptj) < float(cellDistanceThreshold)) {
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

    // Delete the downscaled field, as it is no longer used.
    for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        float* dowsncaledField = downscaledEnsembleFields.at(ensembleIdx);
        delete[] dowsncaledField;
    }
    downscaledEnsembleFields.clear();

    // Build the octree.
    nodesList.clear();
    pointToNodeIndexMap.clear();
    buildTree(nodesList, pointToNodeIndexMap, leafIdxOffset, xsd, ysd, zsd);

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
            const auto& miEntry = miFieldEntries.at(lineIdx);
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

/**
 * Computes the distance of a point to a line segment.
 * See: http://geomalgorithms.com/a02-_lines.html
 *
 * @param p The position of the point.
 * @param l0 The first line point.
 * @param l1 The second line point.
 * @return The distance of p to the line segment.
 */
inline float getDistanceToLineSegment(glm::vec2 p, glm::vec2 l0, glm::vec2 l1) {
    glm::vec2 v = l1 - l0;
    glm::vec2 w = p - l0;
    float c1 = glm::dot(v, w);
    if (c1 <= 0.0) {
        return glm::length(p - l0);
    }

    float c2 = glm::dot(v, v);
    if (c2 <= c1) {
        return glm::length(p - l1);
    }

    float b = c1 / c2;
    glm::vec2 pb = l0 + b * v;
    return glm::length(p - pb);
}

inline int sign(float x) { return x > 0.0f ? 1 : (x < 0.0f ? -1 : 0); }

bool isInsidePolygon(const std::vector<glm::vec2>& polygon, const glm::vec2& pt) {
    int firstSide = 0;
    auto n = int(polygon.size());
    for (int i = 0; i < n; i++) {
        const glm::vec2& p0 = polygon.at(i);
        const glm::vec2& p1 = polygon.at((i + 1) % n);
        int side = sign((p1[0] - p0[0]) * (pt[1] - p0[1]) - (p1[1] - p0[1]) * (pt[0] - p0[0]));
        if (i == 0) {
            firstSide = side;
        } else if (firstSide != side) {
            return false;
        }
    }
    return true;
}

void HEBChart::update(float dt) {
    DiagramBase::update(dt);

    glm::vec2 mousePosition(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePosition -= glm::vec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    //else {
    //    mousePosition.y = float(sgl::AppSettings::get()->getMainWindow()->getHeight()) - mousePosition.y - 1;
    //}
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();
    //mousePosition.y = windowHeight - mousePosition.y;

    //std::cout << "mousePosition: " << mousePosition.x << ", " << mousePosition.y << std::endl;

    sgl::AABB2 windowAabb(
            glm::vec2(borderWidth, borderWidth),
            glm::vec2(windowWidth - 2.0f * borderWidth, windowHeight - 2.0f * borderWidth));
    if (!windowAabb.contains(mousePosition)) {
        if (hoveredPointIdx != -1) {
            hoveredPointIdx = -1;
            needsReRender = true;
        }
        if (hoveredLineIdx != -1) {
            hoveredLineIdx = -1;
            needsReRender = true;
        }
    } else {
        glm::vec2 centeredMousePos = mousePosition - glm::vec2(windowWidth / 2.0f, windowHeight / 2.0f);
        float radiusMouse = std::sqrt(centeredMousePos.x * centeredMousePos.x + centeredMousePos.y * centeredMousePos.y);
        float phiMouse = std::fmod(std::atan2(centeredMousePos.y, centeredMousePos.x) + sgl::TWO_PI, sgl::TWO_PI);

        // Factor 4 is used so the user does not need to exactly hit the (potentially very small) points.
        float minRadius = chartRadius - pointRadiusBase * 4.0f;
        float maxRadius = chartRadius + pointRadiusBase * 4.0f;
        auto numLeaves = int(pointToNodeIndexMap.size());
        int sectorIdx = int(std::round(phiMouse / sgl::TWO_PI * float(numLeaves))) % numLeaves;

        //std::cout << "sector idx: " << sectorIdx << std::endl;

        if (radiusMouse >= minRadius && radiusMouse <= maxRadius) {
            float sectorCenterAngle = float(sectorIdx) / float(numLeaves) * sgl::TWO_PI;
            sgl::Circle circle(
                    chartRadius * glm::vec2(std::cos(sectorCenterAngle), std::sin(sectorCenterAngle)),
                    pointRadiusBase * 4.0f);
            if (circle.contains(centeredMousePos)) {
                hoveredPointIdx = sectorIdx;
            } else {
                hoveredPointIdx = -1;
            }
        } else {
            hoveredPointIdx = -1;
        }

        // Select a line.
        const float minDist = 4.0f;
        if (!curvePoints.empty()) {
            // TODO: Test if point lies in convex hull of control points first (using @see isInsidePolygon).
            int closestLineIdx = -1;
            float closestLineDist = std::numeric_limits<float>::max();
            for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
                for (int ptIdx = 0; ptIdx < NUM_SUBDIVISIONS - 1; ptIdx++) {
                    glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                    pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
                    pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
                    glm::vec2 pt1 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx + 1);
                    pt1.x = windowWidth / 2.0f + pt1.x * chartRadius;
                    pt1.y = windowHeight / 2.0f + pt1.y * chartRadius;
                    float dist = getDistanceToLineSegment(mousePosition, pt0, pt1);
                    if (dist < closestLineDist && dist <= minDist) {
                        closestLineIdx = lineIdx;
                        closestLineDist = dist;
                    }
                }
            }
            if (closestLineIdx >= 0 && closestLineDist <= minDist) {
                hoveredLineIdx = closestLineIdx;
            } else {
                hoveredLineIdx = -1;
            }
        } else {
            hoveredLineIdx = -1;
        }
    }

    if (sgl::Mouse->buttonPressed(1)) {
        clickedLineIdx = -1;
        clickedPointIdx = -1;
        if (hoveredLineIdx >= 0) {
            clickedLineIdx = hoveredLineIdx;
        } else if (hoveredPointIdx >= 0) {
            clickedPointIdx = hoveredPointIdx;
        }
    }

    int newSelectedLineIdx = -1;
    int newSelectedPointIndices[2] = { -1, -1 };
    if (hoveredLineIdx >= 0) {
        newSelectedLineIdx = hoveredLineIdx;
    } else if (clickedLineIdx >= 0 && hoveredPointIdx < 0) {
        newSelectedLineIdx = clickedLineIdx;
    }

    if (newSelectedLineIdx >= 0) {
        const auto& points = connectedPointsArray.at(newSelectedLineIdx);
        newSelectedPointIndices[0] = points.first;
        newSelectedPointIndices[1] = points.second;
    } else if (hoveredPointIdx >= 0) {
        newSelectedPointIndices[0] = hoveredPointIdx;
    } else if (clickedPointIdx >= 0) {
        newSelectedPointIndices[0] = clickedPointIdx;
    }

    if (selectedPointIndices[0] != newSelectedPointIndices[0] || selectedPointIndices[1] != newSelectedPointIndices[1]
            || selectedLineIdx != newSelectedLineIdx) {
        needsReRender = true;
    }
    selectedLineIdx = newSelectedLineIdx;
    selectedPointIndices[0] = newSelectedPointIndices[0];
    selectedPointIndices[1] = newSelectedPointIndices[1];
}

bool HEBChart::getIsRegionSelected(int idx) {
    return selectedPointIndices[idx] >= 0;
}

sgl::AABB3 HEBChart::getSelectedRegion(int idx) {
    //const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedCircleIdx);
    auto pointIdx =
            uint32_t(std::find(pointToNodeIndexMap.begin(), pointToNodeIndexMap.end(), int(leafIdxOffset) + selectedPointIndices[idx])
            - pointToNodeIndexMap.begin());
    uint32_t xd = pointIdx % uint32_t(xsd);
    uint32_t yd = (pointIdx / uint32_t(xsd)) % uint32_t(ysd);
    uint32_t zd = pointIdx / uint32_t(xsd * ysd);
    sgl::AABB3 aabb;
    if (use2dField) {
        int zCenter = zs / 2;
        aabb.min = glm::vec3(xd * df, yd * df, zCenter);
        aabb.max = glm::vec3(
                float(std::min(int(xd + 1) * df, xs)),
                float(std::min(int(yd + 1) * df, ys)),
                float(std::min(zCenter + 1, zs)));
    } else {
        aabb.min = glm::vec3(xd * df, yd * df, zd * df);
        aabb.max = glm::vec3(
                float(std::min(int(xd + 1) * df, xs)),
                float(std::min(int(yd + 1) * df, ys)),
                float(std::min(int(zd + 1) * df, zs)));
    }
    aabb.min /= glm::vec3(xs, ys, zs);
    aabb.max /= glm::vec3(xs, ys, zs);
    sgl::AABB3 volumeAABB = volumeData->getBoundingBoxRendering();
    aabb.min = volumeAABB.min + (volumeAABB.max - volumeAABB.min) * aabb.min;
    aabb.max = volumeAABB.min + (volumeAABB.max - volumeAABB.min) * aabb.max;
    return aabb;
}

void HEBChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    NVGcolor circleFillColorNvg = nvgRGBA(
            circleFillColor.getR(), circleFillColor.getG(),
            circleFillColor.getB(), circleFillColor.getA());
    NVGcolor circleFillColorSelectedNvg = nvgRGBA(
            circleFillColorSelected.getR(), circleFillColorSelected.getG(),
            circleFillColorSelected.getB(), circleFillColorSelected.getA());

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    // Draw the B-spline curves. TODO: Port to Vulkan or OpenGL.
    NVGcolor curveStrokeColor = nvgRGBA(
            100, 255, 100, uint8_t(std::clamp(int(std::ceil(curveOpacity * 255.0f)), 0, 255)));
    if (!curvePoints.empty()) {
        for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
            nvgBeginPath(vg);
            glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            nvgMoveTo(vg, pt0.x, pt0.y);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                nvgLineTo(vg, pt.x, pt.y);
            }

            float strokeWidth = lineIdx == hoveredLineIdx ? 1.5f : 1.0f;
            nvgStrokeWidth(vg, strokeWidth);

            bool opacityByValue = true;
            if (opacityByValue) {
                float maxMi = miValues.front();
                float minMi = miValues.back();
                float factor = (miValues.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                curveStrokeColor.a = curveOpacity * factor;
            }
            curveStrokeColor.a = lineIdx == selectedLineIdx ? 1.0f : curveStrokeColor.a;
            nvgStrokeColor(vg, curveStrokeColor);

            nvgStroke(vg);
        }
    }

    // Draw the point circles.
    float pointRadius = pointRadiusBase;
    nvgBeginPath(vg);
    /*for (int i = 0; i < int(nodesList.size()); i++) {
        const auto& leaf = nodesList.at(i);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgCircle(vg, pointX, pointY, pointRadius);
    }*/
    for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
        const auto& leaf = nodesList.at(leafIdx);
        int pointIdx = leafIdx - int(leafIdxOffset);
        if (pointIdx == selectedPointIndices[0] || pointIdx == selectedPointIndices[1]) {
            continue;
        }
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgCircle(vg, pointX, pointY, pointRadius);
    }
    nvgFillColor(vg, circleFillColorNvg);
    nvgFill(vg);

    int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
    for (int idx = 0; idx < numPointsSelected; idx++) {
        const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgBeginPath(vg);
        nvgCircle(vg, pointX, pointY, pointRadius * 1.5f);
        nvgFillColor(vg, circleFillColorSelectedNvg);
        nvgFill(vg);
    }
}

#ifdef SUPPORT_SKIA
void HEBChart::renderBaseSkia() {
    DiagramBase::renderBaseSkia();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    SkPaint paint;
    static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(&paint);

    // Draw the B-spline curves.
    sgl::Color curveStrokeColor = sgl::Color(
            100, 255, 100, uint8_t(std::clamp(int(std::ceil(curveOpacity * 255.0f)), 0, 255)));
    if (!curvePoints.empty()) {
        paint.setStroke(true);
        paint.setStrokeWidth(1.0f * s);
        paint.setColor(toSkColor(curveStrokeColor));
        for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
            SkPath path;
            glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            path.moveTo(pt0.x * s, pt0.y * s);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                path.lineTo(pt.x * s, pt.y * s);
            }

            bool opacityByValue = true;
            if (opacityByValue) {
                float maxMi = miValues.front();
                float minMi = miValues.back();
                float factor = (miValues.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                curveStrokeColor.setFloatA(curveOpacity * factor);
                paint.setColor(toSkColor(curveStrokeColor));
            }

            if (selectedLineIdx == lineIdx) {
                sgl::Color curveStrokeColorSelected = curveStrokeColor;
                curveStrokeColorSelected.setA(255);
                paint.setColor(toSkColor(curveStrokeColorSelected));
                paint.setStrokeWidth(1.5f * s);
            } else {
                if (!opacityByValue) {
                    paint.setColor(toSkColor(curveStrokeColor));
                }
                paint.setStrokeWidth(1.0f * s);
            }

            canvas->drawPath(path, paint);
        }
    }

    // Draw the point circles.
    float pointRadius = pointRadiusBase * s;
    paint.setColor(toSkColor(circleFillColor));
    paint.setStroke(false);
    /*for (int i = 0; i < int(nodesList.size()); i++) {
        const auto& leaf = nodesList.at(i);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        canvas->drawCircle(pointX * s, pointY * s, pointRadius, paint);
    }*/
    for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
        const auto& leaf = nodesList.at(leafIdx);
        int pointIdx = leafIdx - int(leafIdxOffset);
        if (pointIdx == selectedPointIndices[0] || pointIdx == selectedPointIndices[1]) {
            continue;
        }
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        canvas->drawCircle(pointX * s, pointY * s, pointRadius, paint);
    }

    int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
    for (int idx = 0; idx < numPointsSelected; idx++) {
        const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        paint.setColor(toSkColor(circleFillColorSelected));
        canvas->drawCircle(pointX * s, pointY * s, pointRadius * 1.5f, paint);
    }
}
#endif

#ifdef SUPPORT_VKVG
void HEBChart::renderBaseVkvg() {
    DiagramBase::renderBaseVkvg();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    // Draw the B-spline curves.
    sgl::Color curveStrokeColor = sgl::Color(100, 255, 100, 255);
    if (!curvePoints.empty()) {
        vkvg_set_line_width(context, 1.0f * s);
        vkvg_set_source_color(context, curveStrokeColor.getColorRGBA());

        bool opacityByValue = true;
        if (opacityByValue) {
            float maxMi = miValues.front();
            float minMi = miValues.back();
            for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
                if (lineIdx == selectedLineIdx) {
                    continue;
                }
                float factor = (miValues.at(lineIdx) - minMi) / (maxMi - minMi) * 0.75f + 0.25f;
                vkvg_set_opacity(context, curveOpacity * factor);

                glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
                pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
                pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
                vkvg_move_to(context, pt0.x * s, pt0.y * s);
                for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                    glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                    pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                    pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                    vkvg_line_to(context, pt.x * s, pt.y * s);
                }

                vkvg_stroke(context);
            }
        } else {
            vkvg_set_opacity(context, curveOpacity);
            for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
                if (lineIdx == selectedLineIdx) {
                    continue;
                }
                glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
                pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
                pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
                vkvg_move_to(context, pt0.x * s, pt0.y * s);
                for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                    glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                    pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                    pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                    vkvg_line_to(context, pt.x * s, pt.y * s);
                }
            }
            vkvg_stroke(context);
        }

        if (hoveredLineIdx >= 0) {
            vkvg_set_line_width(context, 1.5f * s);
            vkvg_set_opacity(context, 1.0f);

            glm::vec2 pt0 = curvePoints.at(hoveredLineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.y * chartRadius;
            vkvg_move_to(context, pt0.x * s, pt0.y * s);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(hoveredLineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.y * chartRadius;
                vkvg_line_to(context, pt.x * s, pt.y * s);
            }

            vkvg_stroke(context);
        }
    }
    vkvg_set_opacity(context, 1.0f);

    // Draw the point circles.
    float pointRadius = pointRadiusBase * s;
    /*for (int i = 0; i < int(nodesList.size()); i++) {
        const auto& leaf = nodesList.at(i);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        vkvg_ellipse(context, pointRadius, pointRadius, pointX * s, pointY * s, 0.0f);
    }*/
    for (int leafIdx = int(leafIdxOffset); leafIdx < int(nodesList.size()); leafIdx++) {
        const auto& leaf = nodesList.at(leafIdx);
        int pointIdx = leafIdx - int(leafIdxOffset);
        if (pointIdx == selectedPointIndices[0] || pointIdx == selectedPointIndices[1]) {
            continue;
        }
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        vkvg_ellipse(context, pointRadius, pointRadius, pointX * s, pointY * s, 0.0f);
    }
    vkvg_set_source_color(context, circleFillColor.getColorRGBA());
    vkvg_fill(context);

    int numPointsSelected = selectedPointIndices[0] < 0 ? 0 : (selectedPointIndices[1] < 0 ? 1 : 2);
    for (int idx = 0; idx < numPointsSelected; idx++) {
        const auto& leaf = nodesList.at(int(leafIdxOffset) + selectedPointIndices[idx]);
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        vkvg_ellipse(context, pointRadius * 1.5f, pointRadius * 1.5f, pointX * s, pointY * s, 0.0f);
        vkvg_set_source_color(context, circleFillColorSelected.getColorRGBA());
        vkvg_fill(context);
    }
}
#endif
