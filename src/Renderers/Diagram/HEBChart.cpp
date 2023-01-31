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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/libs/nanovg/nanovg.h>

#include "Loaders/DataSet.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Volume/VolumeData.hpp"
#include "BSpline.hpp"
#include "HEBChart.hpp"

struct HEBNode {
    glm::vec2 screenPosition;
    uint32_t parentIdx;
    uint32_t childIndices[8];
};

struct MIFieldEntry {
    float miValue;
    uint32_t pointIndex0, pointIndex1;

    MIFieldEntry(float miValue, uint32_t pointIndex0, uint32_t pointIndex1)
            : miValue(miValue), pointIndex0(pointIndex0), pointIndex1(pointIndex1) {}
    bool operator<(const MIFieldEntry& rhs) const { return miValue > rhs.miValue; }
};

void buildTree(
        std::vector<HEBNode>& nodesList, int xsd, int ysd, int zsd) {
    int treeHeightX = std::ceil(std::log2(xsd));
    int treeHeightY = std::ceil(std::log2(ysd));
    int treeHeightZ = std::ceil(std::log2(zsd));
    int treeHeight = std::max(treeHeightX, std::max(treeHeightY, treeHeightZ));
    nodesList.emplace_back();

    // TODO
    /*HEBNode* rootNode;
    rootNode.parentIdx = std::numeric_limits<uint32_t>::max();
    rootNode.screenPosition = glm::vec2(0.0f);*/
}

void getControlPoints(
        const std::vector<HEBNode>& nodesList, uint32_t pointIndex0, uint32_t pointIndex1,
        std::vector<glm::vec2>& controlPoints) {
    // The start nodes are leaves at the same level.
    const HEBNode* node0 = &nodesList.at(pointIndex0);
    const HEBNode* node1 = &nodesList.at(pointIndex1);

    // Go until lowest common ancestor (LCA).
    std::vector<glm::vec2> controlPointsReverse;
    while(node0 != node1) {
        controlPoints.push_back(node0->screenPosition);
        controlPointsReverse.push_back(node1->screenPosition);
        node0 = &nodesList.at(node0->parentIdx);
        node1 = &nodesList.at(node1->parentIdx);
    }
    controlPoints.push_back(node0->screenPosition);
    for (const glm::vec2& pt : controlPointsReverse) {
        controlPoints.push_back(pt);
    }
}

#define IDXSD(x,y,z) ((z)*xsd*ysd + (y)*xsd + (x))

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
    float esInv = 1.0f / float(es);
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
                    for (int zo = 0; zo < df; zo++) {
                        for (int yo = 0; yo < df; yo++) {
                            for (int xo = 0; xo < df; xo++) {
                                int x = xd * df + xo;
                                int y = yd * df + yo;
                                int z = zd * df + zo;
                                if (x < xs && y < ys && z < zs) {
                                    valueMean += field[IDXS(x, y, z)] * esInv;
                                }
                            }
                        }
                    }
                    field[IDXSD(xd, yd, zd)] = valueMean;
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
    std::vector<HEBNode> nodesList;
    buildTree(nodesList, xsd, ysd, zsd);

    curvePoints.resize(NUM_LINES * NUM_SUBDIVISIONS);
    miValues.resize(NUM_LINES);
    std::vector<glm::vec2> controlPoints;
    for (int lineIdx = 0; lineIdx < NUM_LINES; lineIdx++) {
        const auto& miEntry = miFieldEntries.at(lineIdx);
        miValues.at(lineIdx) = miEntry.miValue;
        controlPoints.clear();
        getControlPoints(nodesList, miEntry.pointIndex0, miEntry.pointIndex1, controlPoints);
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

    float pointRadius = 1.0f;
    int numLeaves = 100;
    nvgBeginPath(vg);
    for (int i = 0; i < numLeaves; i++) {
        float angle = float(i) / float(numLeaves) * sgl::TWO_PI;
        float pointX = windowWidth / 2.0f + std::cos(angle) * chartRadius;
        float pointY = windowHeight / 2.0f + std::sin(angle) * chartRadius;
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
