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

#include <stack>

#include <Math/Math.hpp>
#include <Graphics/Vulkan/libs/nanovg/nanovg.h>

#include "Loaders/DataSet.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Volume/VolumeData.hpp"
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
    StackDomain(uint32_t nodeIdx, const glm::ivec3& min, const glm::ivec3& max)
            : nodeIdx(nodeIdx), min(min), max(max) {}
    uint32_t nodeIdx;
    glm::ivec3 min, max;
};

void buildTree(
        std::vector<HEBNode>& nodesList, std::vector<uint32_t>& pointToNodeIndexMap, int xsd, int ysd, int zsd) {
    int treeHeightX = std::ceil(std::log2(xsd));
    int treeHeightY = std::ceil(std::log2(ysd));
    int treeHeightZ = std::ceil(std::log2(zsd));
    int treeHeight = std::max(treeHeightX, std::max(treeHeightY, treeHeightZ));
    nodesList.emplace_back();
    nodesList[0].normalizedPosition = glm::vec3(0.0f);
    pointToNodeIndexMap.resize(xsd * ysd * zsd);

    std::stack<StackDomain> domainStack;
    StackDomain rootDomain;
    rootDomain.nodeIdx = 0;
    rootDomain.min = glm::ivec3(0, 0, 0);
    rootDomain.max = glm::ivec3(xsd - 1, ysd - 1, zsd - 1);
    domainStack.push(rootDomain);
    while (!domainStack.empty()) {
        auto stackEntry = domainStack.top();
        domainStack.pop();
        auto extent = stackEntry.max - stackEntry.min + glm::ivec3(1);
        // Leaf?
        if (extent.x == 1 && extent.y == 1 && extent.z == 1) {
            pointToNodeIndexMap.at(IDXSD(stackEntry.min.x, stackEntry.min.y, stackEntry.min.z)) = stackEntry.nodeIdx;
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
                uint32_t(nodesList.size()), glm::vec3(stackEntry.min.x, stackEntry.min.y, stackEntry.min.z),
                glm::vec3(maxHalf.x, maxHalf.y, maxHalf.z));
        if (extent.x > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), glm::vec3(minHalf.x, stackEntry.min.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, maxHalf.y, maxHalf.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
        }
        if (extent.y > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), glm::vec3(stackEntry.min.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(maxHalf.x, stackEntry.max.y, maxHalf.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
        }
        if (extent.x > 1 && extent.y > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), glm::vec3(minHalf.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, stackEntry.max.y, maxHalf.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
        }
        if (extent.z > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), glm::vec3(stackEntry.min.x, stackEntry.min.y, minHalf.z),
                    glm::vec3(maxHalf.x, maxHalf.y, stackEntry.max.z));
            HEBNode child(stackEntry.nodeIdx);
            nodesList.push_back(child);
            if (extent.x > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), glm::vec3(minHalf.x, stackEntry.min.y, minHalf.z),
                        glm::vec3(stackEntry.max.x, maxHalf.y, stackEntry.max.z));
                HEBNode child(stackEntry.nodeIdx);
                nodesList.push_back(child);
            }
            if (extent.y > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), glm::vec3(stackEntry.min.x, minHalf.y, minHalf.z),
                        glm::vec3(maxHalf.x, stackEntry.max.y, stackEntry.max.z));
                HEBNode child(stackEntry.nodeIdx);
                nodesList.push_back(child);
            }
            if (extent.x > 1 && extent.y > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), glm::vec3(minHalf.x, minHalf.y, minHalf.z),
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
    for (uint32_t leafIdx : pointToNodeIndexMap) {
        prevParentNodeIndices.insert(nodesList[leafIdx].parentIdx);
        float angle = float(leafCounter) / float(pointToNodeIndexMap.size()) * sgl::TWO_PI;
        nodesList[leafIdx].angle = angle;
        nodesList[leafIdx].normalizedPosition = glm::vec2(std::cos(angle), std::sin(angle));
        leafCounter++;
    }

    int currentDepth = treeHeight - 1;
    while (nextParentNodeIndices.size() > 1) {
        float radius = float(currentDepth) / float(treeHeight);
        for (uint32_t nodeIdx : nextParentNodeIndices) {
            auto& node = nodesList[nodeIdx];
            float minChildAngle = std::numeric_limits<float>::max();
            float maxChildAngle = std::numeric_limits<float>::lowest();
            for (int i = 0; i < 8; i++) {
                if (node.childIndices[i] == std::numeric_limits<uint32_t>::max()) {
                    break;
                }
                minChildAngle = nodesList[node.childIndices[i]].angle;
            }
            node.angle = 0.5f * (minChildAngle + maxChildAngle);
            node.normalizedPosition = radius * glm::vec2(std::cos(node.angle), std::sin(node.angle));
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
    uint32_t nidx1 = pointToNodeIndexMap.at(pointIndex0);

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
        controlPoints.push_back(nodesList.at(ancestors0.at(i)).normalizedPosition);
    }
}

HEBChart::HEBChart() = default;

void HEBChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    chartRadius = 160;
    windowWidth = (chartRadius + borderSizeX) * 2.0f;
    windowHeight = (chartRadius + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onWindowSizeChanged();
}

void HEBChart::update(float dt) {
    DiagramBase::update(dt);
}

void HEBChart::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;
    dataDirty = true;
}

void HEBChart::setSelectedScalarField(int _selectedFieldIdx, const std::string& _scalarFieldName) {
    selectedFieldIdx = _selectedFieldIdx;
    selectedScalarFieldName = _scalarFieldName;
    dataDirty = true;
}

void HEBChart::updateData() {
    // Values downscaled by factor 32.
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();
    int df = 32;
    int xsd = sgl::iceil(xs, df); // 8
    int ysd = sgl::iceil(ys, df); // 11
    int zsd = sgl::iceil(zs, df); // 1
    int numPoints = xsd * ysd * zsd;

    // Compute the downscaled field.
    std::vector<float*> downscaledEnsembleFields;
    downscaledEnsembleFields.resize(es);
    for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, selectedScalarFieldName, -1, ensembleIdx);
        float* field = ensembleEntryField.get();
        auto* dowsncaledField = new float[numPoints];
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
        downscaledEnsembleFields.at(ensembleIdx) = dowsncaledField;
    }

    // Compute the mutual information matrix.
    int k = std::max(sgl::iceil(3 * es, 100), 1);
    std::vector<MIFieldEntry> miFieldEntries;
    miFieldEntries.reserve((numPoints * numPoints + numPoints) / 2);
    std::vector<float> X(es);
    std::vector<float> Y(es);
    for (int i = 0; i < numPoints; i++) {
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
        for (int j = 0; j < i; j++) {
            for (int e = 0; e < es; e++) {
                Y[e] = downscaledEnsembleFields.at(e)[i];
                if (std::isnan(Y[e])) {
                    isNan = true;
                    break;
                }
            }
            if (!isNan) {
                float miValue = computeMutualInformationKraskov<double>(X.data(), Y.data(), k, es);
                miFieldEntries.emplace_back(miValue, i, j);
            }
        }
    }
    std::sort(miFieldEntries.begin(), miFieldEntries.end());

    // Delete the downscaled field, as it is no longer used.
    for (int ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        float* dowsncaledField = downscaledEnsembleFields.at(ensembleIdx);
        delete[] dowsncaledField;
    }
    downscaledEnsembleFields.clear();

    // Build the octree.
    nodesList.clear();
    pointToNodeIndexMap.clear();
    buildTree(nodesList, pointToNodeIndexMap, xsd, ysd, zsd);

    curvePoints.resize(NUM_LINES * NUM_SUBDIVISIONS);
    miValues.resize(NUM_LINES);
    std::vector<glm::vec2> controlPoints;
    for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
        const auto& miEntry = miFieldEntries.at(lineIdx);
        miValues.at(lineIdx) = miEntry.miValue;
        controlPoints.clear();
        getControlPoints(nodesList, pointToNodeIndexMap, miEntry.pointIndex0, miEntry.pointIndex1, controlPoints);
        for (int ptIdx = 0; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
            float t = float(ptIdx) / float(NUM_SUBDIVISIONS - 1);
            curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx) = evaluateBSpline(t, 4, controlPoints);
        }
    }
}

void HEBChart::renderBase() {
    DiagramBase::renderBase();

    NVGcolor circleFillColor = nvgRGBA(180, 180, 180, 255);
    /*NVGcolor textColor = nvgRGBA(0, 0, 0, 255);
     NVGcolor circleFillColor = nvgRGBA(180, 180, 180, 70);
     NVGcolor circleStrokeColor = nvgRGBA(120, 120, 120, 120);
     NVGcolor dashedCircleStrokeColor = nvgRGBA(120, 120, 120, 120);

     // Render the central radial chart area.
     glm::vec2 center(windowWidth / 2.0f, windowHeight / 2.0f);
     nvgBeginPath(vg);
     nvgCircle(vg, center.x, center.y, chartRadius);
     nvgFillColor(vg, circleFillColor);
     nvgFill(vg);
     nvgStrokeColor(vg, circleStrokeColor);
     nvgStroke(vg);*/

    float pointRadius = 1.5f;
    auto numLeaves = int(pointToNodeIndexMap.size());
    nvgBeginPath(vg);
    for (int i = 0; i < numLeaves; i++) {
        const auto& leaf = nodesList.at(pointToNodeIndexMap.at(i));
        float pointX = windowWidth / 2.0f + leaf.normalizedPosition.x * chartRadius;
        float pointY = windowHeight / 2.0f + leaf.normalizedPosition.y * chartRadius;
        nvgCircle(vg, pointX, pointY, pointRadius);
    }
    nvgFillColor(vg, circleFillColor);
    nvgFill(vg);

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    // Draw the B-spline curves. TODO: Port to Vulkan or OpenGL.
    NVGcolor curveStrokeColor = nvgRGBA(100, 255, 100, 100);
    if (!curvePoints.empty()) {
        for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
            nvgBeginPath(vg);
            glm::vec2 pt0 = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + 0);
            pt0.x = windowWidth / 2.0f + pt0.x * chartRadius;
            pt0.y = windowHeight / 2.0f + pt0.x * chartRadius;
            nvgMoveTo(vg, pt0.x, pt0.y);
            for (int ptIdx = 1; ptIdx < NUM_SUBDIVISIONS; ptIdx++) {
                glm::vec2 pt = curvePoints.at(lineIdx * NUM_SUBDIVISIONS + ptIdx);
                pt.x = windowWidth / 2.0f + pt.x * chartRadius;
                pt.y = windowHeight / 2.0f + pt.x * chartRadius;
                nvgLineTo(vg, pt.x, pt.y);
            }
            nvgStrokeWidth(vg, 1.0f);
            nvgStrokeColor(vg, curveStrokeColor);
            nvgStroke(vg);
        }
    }
}
