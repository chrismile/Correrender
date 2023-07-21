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

#ifndef CORRERENDER_OCTREE_HPP
#define CORRERENDER_OCTREE_HPP

#include <algorithm>
#include <glm/vec2.hpp>

#define IDXSD(x,y,z) ((z)*xsd*ysd + (y)*xsd + (x))

enum class OctreeMethod {
    TOP_DOWN_CEIL, TOP_DOWN_POT
};
const char* const OCTREE_METHOD_NAMES[] = {
        "Top Down (ceil)", "Top Down (PoT)"
};

enum class RegionWinding {
    WINDING_POINT_SYMMETRIC, WINDING_AXIS_SYMMETRIC
};
const char* const REGION_WINDING_NAMES[] = {
        "Point Symmetric", "Axis Symmetric"
};

struct HEBNode {
    HEBNode() {
        parentIdx = std::numeric_limits<uint32_t>::max();
        std::fill_n(childIndices, 8, std::numeric_limits<uint32_t>::max());
    }
    explicit HEBNode(uint32_t parentIdx) : parentIdx(parentIdx) {
        std::fill_n(childIndices, 8, std::numeric_limits<uint32_t>::max());
    }
    glm::vec2 normalizedPosition;
    float angle = 0.0f;
    uint32_t parentIdx;
    uint32_t childIndices[8];
};

void buildHebTree(
        OctreeMethod octreeMethod, RegionWinding regionWinding, std::vector<HEBNode>& nodesList,
        std::vector<uint32_t>& pointToNodeIndexMap0, std::vector<uint32_t>& pointToNodeIndexMap1,
        uint32_t& leafIdxOffset0, uint32_t& leafIdxOffset1,
        bool regionsEqual, int xsd0, int ysd0, int zsd0, int xsd1, int ysd1, int zsd1);

#endif //CORRERENDER_OCTREE_HPP
