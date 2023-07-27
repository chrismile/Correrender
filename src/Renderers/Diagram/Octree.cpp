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
#include <unordered_set>
#include <queue>
#include <cmath>
#include <glm/vec3.hpp>

#include <Math/Math.hpp>

#include "Octree.hpp"

struct StackDomain {
    StackDomain() {}
    StackDomain(uint32_t nodeIdx, uint32_t depth, const glm::ivec3& min, const glm::ivec3& max)
            : nodeIdx(nodeIdx), depth(depth), min(min), max(max) {}
    uint32_t nodeIdx;
    uint32_t depth;
    glm::ivec3 min, max;
};

struct StackDomain2 {
    StackDomain2() {}
    StackDomain2(uint32_t nodeIdx, uint32_t depth, int subdivSize, const glm::ivec3& min, const glm::ivec3& max)
            : nodeIdx(nodeIdx), depth(depth), subdivSize(subdivSize), min(min), max(max) {}
    uint32_t nodeIdx;
    uint32_t depth;
    int subdivSize;
    glm::ivec3 min, max;
};

void buildHebTreeIterativeTopDown(
        std::vector<HEBNode>& nodesList, std::vector<uint32_t>& pointToNodeIndexMap, uint32_t& leafIdxOffset,
        bool regionsEqual, int groupIdx, int xsd, int ysd, int zsd, uint32_t treeHeight) {
    std::queue<StackDomain> domainStack;
    StackDomain rootDomain;
    rootDomain.nodeIdx = regionsEqual ? 0 : (groupIdx == 0 ? 1 : 2);
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
        nodesList.emplace_back(stackEntry.nodeIdx);
        if (extent.x > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(minHalf.x, stackEntry.min.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, maxHalf.y, maxHalf.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
        }
        if (extent.y > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(stackEntry.min.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(maxHalf.x, stackEntry.max.y, maxHalf.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
        }
        if (extent.x > 1 && extent.y > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(minHalf.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, stackEntry.max.y, maxHalf.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
        }
        if (extent.z > 1) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1,
                    glm::vec3(stackEntry.min.x, stackEntry.min.y, minHalf.z),
                    glm::vec3(maxHalf.x, maxHalf.y, stackEntry.max.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
            if (extent.x > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1,
                        glm::vec3(minHalf.x, stackEntry.min.y, minHalf.z),
                        glm::vec3(stackEntry.max.x, maxHalf.y, stackEntry.max.z));
                nodesList.emplace_back(stackEntry.nodeIdx);
            }
            if (extent.y > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1,
                        glm::vec3(stackEntry.min.x, minHalf.y, minHalf.z),
                        glm::vec3(maxHalf.x, stackEntry.max.y, stackEntry.max.z));
                nodesList.emplace_back(stackEntry.nodeIdx);
            }
            if (extent.x > 1 && extent.y > 1) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1,
                        glm::vec3(minHalf.x, minHalf.y, minHalf.z),
                        glm::vec3(stackEntry.max.x, stackEntry.max.y, stackEntry.max.z));
                nodesList.emplace_back(stackEntry.nodeIdx);
            }
        }
        uint32_t numChildren = uint32_t(nodesList.size()) - childrenOffset;
        for (uint32_t i = 0; i < numChildren; i++) {
            nodesList[stackEntry.nodeIdx].childIndices[i] = childrenOffset + i;
        }
    }
}

void buildHebTreeIterativeTopDownZOrder(
        std::vector<HEBNode>& nodesList, std::vector<uint32_t>& pointToNodeIndexMap, uint32_t& leafIdxOffset,
        bool regionsEqual, int groupIdx, int xsd, int ysd, int zsd, uint32_t treeHeight) {
    int maxDim = std::max(xsd, std::max(ysd, zsd));
    if (!sgl::isPowerOfTwo(maxDim)) {
        maxDim = sgl::nextPowerOfTwo(maxDim);
    }
    std::queue<StackDomain2> domainStack;
    StackDomain2 rootDomain;
    rootDomain.nodeIdx = regionsEqual ? 0 : (groupIdx == 0 ? 1 : 2);
    rootDomain.depth = 0;
    rootDomain.min = glm::ivec3(0, 0, 0);
    rootDomain.max = glm::ivec3(xsd - 1, ysd - 1, zsd - 1);
    rootDomain.subdivSize = std::max(maxDim / 2, 1);
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
        minHalf.x = stackEntry.min.x + std::min(stackEntry.subdivSize, extent.x);
        minHalf.y = stackEntry.min.y + std::min(stackEntry.subdivSize, extent.y);
        minHalf.z = stackEntry.min.z + std::min(stackEntry.subdivSize, extent.z);
        maxHalf.x = minHalf.x - 1;
        maxHalf.y = minHalf.y - 1;
        maxHalf.z = minHalf.z - 1;
        auto childrenOffset = uint32_t(nodesList.size());
        domainStack.emplace(
                uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                glm::vec3(stackEntry.min.x, stackEntry.min.y, stackEntry.min.z),
                glm::vec3(maxHalf.x, maxHalf.y, maxHalf.z));
        nodesList.emplace_back(stackEntry.nodeIdx);
        bool subdivX = stackEntry.max.x - minHalf.x >= 0;
        bool subdivY = stackEntry.max.y - minHalf.y >= 0;
        bool subdivZ = stackEntry.max.z - minHalf.z >= 0;
        if (subdivX) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                    glm::vec3(minHalf.x, stackEntry.min.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, maxHalf.y, maxHalf.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
        }
        if (subdivY) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                    glm::vec3(stackEntry.min.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(maxHalf.x, stackEntry.max.y, maxHalf.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
        }
        if (subdivX && subdivY) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                    glm::vec3(minHalf.x, minHalf.y, stackEntry.min.z),
                    glm::vec3(stackEntry.max.x, stackEntry.max.y, maxHalf.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
        }
        if (subdivZ) {
            domainStack.emplace(
                    uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                    glm::vec3(stackEntry.min.x, stackEntry.min.y, minHalf.z),
                    glm::vec3(maxHalf.x, maxHalf.y, stackEntry.max.z));
            nodesList.emplace_back(stackEntry.nodeIdx);
            if (subdivX) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                        glm::vec3(minHalf.x, stackEntry.min.y, minHalf.z),
                        glm::vec3(stackEntry.max.x, maxHalf.y, stackEntry.max.z));
                nodesList.emplace_back(stackEntry.nodeIdx);
            }
            if (subdivY) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                        glm::vec3(stackEntry.min.x, minHalf.y, minHalf.z),
                        glm::vec3(maxHalf.x, stackEntry.max.y, stackEntry.max.z));
                nodesList.emplace_back(stackEntry.nodeIdx);
            }
            if (subdivX && subdivY) {
                domainStack.emplace(
                        uint32_t(nodesList.size()), stackEntry.depth + 1, stackEntry.subdivSize / 2,
                        glm::vec3(minHalf.x, minHalf.y, minHalf.z),
                        glm::vec3(stackEntry.max.x, stackEntry.max.y, stackEntry.max.z));
                nodesList.emplace_back(stackEntry.nodeIdx);
            }
        }
        uint32_t numChildren = uint32_t(nodesList.size()) - childrenOffset;
        for (uint32_t i = 0; i < numChildren; i++) {
            nodesList[stackEntry.nodeIdx].childIndices[i] = childrenOffset + i;
        }
    }
}

void buildHebTree(
        OctreeMethod octreeMethod, RegionWinding regionWinding, std::vector<HEBNode>& nodesList,
        std::vector<uint32_t>& pointToNodeIndexMap0, std::vector<uint32_t>& pointToNodeIndexMap1,
        uint32_t& leafIdxOffset0, uint32_t& leafIdxOffset1,
        bool regionsEqual, int xsd0, int ysd0, int zsd0, int xsd1, int ysd1, int zsd1) {
    auto treeHeight0X = uint32_t(std::ceil(std::log2(xsd0)));
    auto treeHeight0Y = uint32_t(std::ceil(std::log2(ysd0)));
    auto treeHeight0Z = uint32_t(std::ceil(std::log2(zsd0)));
    auto treeHeight0 = std::max(treeHeight0X, std::max(treeHeight0Y, treeHeight0Z));
    treeHeight0 = std::max(treeHeight0, uint32_t(1));
    auto treeHeight1X = uint32_t(std::ceil(std::log2(xsd1)));
    auto treeHeight1Y = uint32_t(std::ceil(std::log2(ysd1)));
    auto treeHeight1Z = uint32_t(std::ceil(std::log2(zsd1)));
    auto treeHeight1 = std::max(treeHeight1X, std::max(treeHeight1Y, treeHeight1Z));
    treeHeight1 = std::max(treeHeight1, uint32_t(1));
    auto treeHeight = std::max(treeHeight0, treeHeight1) + (regionsEqual ? 0 : 1);
    nodesList.emplace_back();
    nodesList[0].normalizedPosition = glm::vec3(0.0f);
    if (!regionsEqual) {
        nodesList.emplace_back();
        nodesList.emplace_back();
        nodesList[0].childIndices[0] = 1;
        nodesList[0].childIndices[1] = 2;
        nodesList[1].parentIdx = 0;
        nodesList[2].parentIdx = 0;
    }
    pointToNodeIndexMap0.resize(xsd0 * ysd0 * zsd0);
    pointToNodeIndexMap1.resize(xsd1 * ysd1 * zsd1);
    if (octreeMethod == OctreeMethod::TOP_DOWN_CEIL) {
        buildHebTreeIterativeTopDown(
                nodesList, pointToNodeIndexMap0, leafIdxOffset0,
                regionsEqual, 0, xsd0, ysd0, zsd0, treeHeight0);
    } else {
        buildHebTreeIterativeTopDownZOrder(
                nodesList, pointToNodeIndexMap0, leafIdxOffset0,
                regionsEqual, 0, xsd0, ysd0, zsd0, treeHeight0);
    }
    if (!regionsEqual) {
        std::vector<HEBNode> leaves0 = { nodesList.begin() + leafIdxOffset0, nodesList.end() };
        nodesList.resize(leafIdxOffset0);
        uint32_t leafIdxOffset0Old = leafIdxOffset0;
        if (octreeMethod == OctreeMethod::TOP_DOWN_CEIL) {
            buildHebTreeIterativeTopDown(
                    nodesList, pointToNodeIndexMap1, leafIdxOffset1,
                    regionsEqual, 1, xsd1, ysd1, zsd1, treeHeight1);
        } else {
            buildHebTreeIterativeTopDownZOrder(
                    nodesList, pointToNodeIndexMap1, leafIdxOffset1,
                    regionsEqual, 1, xsd1, ysd1, zsd1, treeHeight1);
        }
        std::vector<HEBNode> leaves1 = { nodesList.begin() + leafIdxOffset1, nodesList.end() };
        nodesList.resize(leafIdxOffset1);
        uint32_t leafIdxOffset1Old = leafIdxOffset1;
        leafIdxOffset0 = uint32_t(nodesList.size());
        nodesList.insert(nodesList.end(), leaves0.begin(), leaves0.end());
        leafIdxOffset1 = uint32_t(nodesList.size());
        nodesList.insert(nodesList.end(), leaves1.begin(), leaves1.end());

        uint32_t shift0 = leafIdxOffset0 - leafIdxOffset0Old;
        std::unordered_set<uint32_t> parents0;
        for (HEBNode& leaf : leaves0) {
            parents0.insert(leaf.parentIdx);
        }
        for (uint32_t parentIdx : parents0) {
            HEBNode& parent = nodesList[parentIdx];
            for (uint32_t& childIdx : parent.childIndices) {
                if (childIdx != std::numeric_limits<uint32_t>::max()) {
                    childIdx += shift0;
                }
            }
        }
        for (uint32_t& nodeIdx : pointToNodeIndexMap0) {
            nodeIdx += shift0;
        }

        uint32_t shift1 = leafIdxOffset1 - leafIdxOffset1Old;
        std::unordered_set<uint32_t> parents1;
        for (HEBNode& leaf : leaves1) {
            parents1.insert(leaf.parentIdx);
        }
        for (uint32_t parentIdx : parents1) {
            HEBNode& parent = nodesList[parentIdx];
            for (uint32_t& childIdx : parent.childIndices) {
                if (childIdx != std::numeric_limits<uint32_t>::max()) {
                    childIdx += shift1;
                }
            }
        }
        for (uint32_t& nodeIdx : pointToNodeIndexMap1) {
            nodeIdx += shift1;
        }
    } else {
        pointToNodeIndexMap1 = pointToNodeIndexMap0;
        leafIdxOffset1 = leafIdxOffset0;
    }


    // Set node positions.
    std::unordered_set<uint32_t> prevParentNodeIndices;
    std::unordered_set<uint32_t> nextParentNodeIndices;
    if (regionsEqual) {
        // Start with placing the leaves on a unit circle.
        auto numLeaves = int(pointToNodeIndexMap0.size() + (regionsEqual ? 0 : pointToNodeIndexMap1.size()));
        uint32_t leafCounter = 0;
        for (uint32_t leafIdx = leafIdxOffset0; leafIdx < uint32_t(nodesList.size()); leafIdx++) {
            float angle;
            if (numLeaves == 1) {
                // Should never happen, unless the data set has size 1x1x1.
                angle = 0.0f;
            } else {
                angle = float(leafCounter) / float(numLeaves) * sgl::TWO_PI;
            }
            nodesList[leafIdx].angle = angle;
            nodesList[leafIdx].normalizedPosition = glm::vec2(std::cos(angle), std::sin(angle));
            if (nodesList[leafIdx].parentIdx != std::numeric_limits<uint32_t>::max()) {
                prevParentNodeIndices.insert(nodesList[leafIdx].parentIdx);
            }
            leafCounter++;
        }
    } else {
        // Start with placing the leaves on a unit circle.
        const float angleRangeHalf = sgl::PI * 0.92f;
        float angleOffset0;
        if (regionWinding == RegionWinding::WINDING_POINT_SYMMETRIC) {
            angleOffset0 = 0.5f * (sgl::PI - angleRangeHalf);
        } else {
            angleOffset0 = sgl::PI + 0.5f * (sgl::PI - angleRangeHalf);
        }
        auto numLeaves0 = int(pointToNodeIndexMap0.size());
        uint32_t leafCounter0 = 0;
        for (uint32_t leafIdx = leafIdxOffset0; leafIdx < leafIdxOffset1; leafIdx++) {
            float angle;
            if (numLeaves0 == 1) {
                angle = sgl::PI * 0.5f;
            } else {
                angle = angleOffset0 + float(leafCounter0) / float(numLeaves0 - 1) * angleRangeHalf;
            }
            nodesList[leafIdx].angle = angle;
            nodesList[leafIdx].normalizedPosition = glm::vec2(std::cos(angle), std::sin(angle));
            if (nodesList[leafIdx].parentIdx != std::numeric_limits<uint32_t>::max()) {
                prevParentNodeIndices.insert(nodesList[leafIdx].parentIdx);
            }
            leafCounter0++;
        }
        float angleOffset1;
        if (regionWinding == RegionWinding::WINDING_POINT_SYMMETRIC) {
            angleOffset1 = sgl::PI + 0.5f * (sgl::PI - angleRangeHalf);
        } else {
            angleOffset1 = 0.5f * (sgl::PI - angleRangeHalf);
        }
        auto numLeaves1 = int(pointToNodeIndexMap1.size());
        uint32_t leafCounter1 = 0;
        for (uint32_t leafIdx = leafIdxOffset1; leafIdx < uint32_t(nodesList.size()); leafIdx++) {
            float angle;
            if (numLeaves1 == 1) {
                angle = sgl::PI * 1.5f;
            } else {
                if (regionWinding == RegionWinding::WINDING_POINT_SYMMETRIC) {
                    angle = angleOffset1 + float(leafCounter1) / float(numLeaves1 - 1) * angleRangeHalf;
                } else {
                    angle = angleOffset1 + float(numLeaves1 - 1 - leafCounter1) / float(numLeaves1 - 1) * angleRangeHalf;
                }
            }
            nodesList[leafIdx].angle = angle;
            nodesList[leafIdx].normalizedPosition = glm::vec2(std::cos(angle), std::sin(angle));
            if (nodesList[leafIdx].parentIdx != std::numeric_limits<uint32_t>::max()) {
                prevParentNodeIndices.insert(nodesList[leafIdx].parentIdx);
            }
            leafCounter1++;
        }
    }

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
