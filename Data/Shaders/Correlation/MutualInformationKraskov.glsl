/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020-2022, Christoph Neuhauser
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

-- Sort2D

void swapElements2D(uint i, uint j) {
    float temp = referenceValues[i];
    referenceValues[i] = referenceValues[j];
    referenceValues[j] = temp;
    temp = queryValues[i];
    queryValues[i] = queryValues[j];
    queryValues[j] = temp;
}

void heapify2D(uint startIdx, uint i, uint numElements, uint axis) {
    uint child;
    float childValue0, childValue1, arrayI;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = axis == 0u ? referenceValues[startIdx + child] : queryValues[startIdx + child];
        childValue1 = axis == 0u ? referenceValues[startIdx + child + 1] : queryValues[startIdx + child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        arrayI = axis == 0u ? referenceValues[startIdx + i] : queryValues[startIdx + i];
        if (arrayI >= childValue0) {
            break;
        }
        swapElements2D(startIdx + i, startIdx + child);
        i = child;
    }
}

void heapSort2D(uint startIdx, uint endIdx, uint axis) {
    uint numElements = endIdx - startIdx;

    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = numElements / 2; i > 0; i--) {
        heapify2D(startIdx, i - 1, numElements, axis);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < numElements; i++) {
        swapElements2D(startIdx, startIdx + numElements - i);
        heapify2D(startIdx, 0, numElements - i, axis);
    }
}

-- Sort1D

void swapElements(uint i, uint j) {
    //float temp = valueArray[i];
    //valueArray[i] = valueArray[j];
    //valueArray[j] = temp;
    float temp = referenceValues[i];
    referenceValues[i] = referenceValues[j];
    referenceValues[j] = temp;
    temp = queryValues[i];
    queryValues[i] = queryValues[j];
    queryValues[j] = temp;
    temp = kthNeighborDistances[i];
    kthNeighborDistances[i] = kthNeighborDistances[j];
    kthNeighborDistances[j] = temp;
}

void heapify(uint i, uint numElements) {
    uint child;
    float childValue0, childValue1;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = valueArray[child];
        childValue1 = valueArray[child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        if (valueArray[i] >= childValue0) {
            break;
        }
        swapElements(i, child);
        i = child;
    }
}

void heapSort() {
    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = MEMBER_COUNT / 2; i > 0; i--) {
        heapify(i - 1, MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < MEMBER_COUNT; i++) {
        swapElements(0, MEMBER_COUNT - i);
        heapify(0, MEMBER_COUNT - i);
    }
}

float averageDigamma() {
    heapSort();

    float factor = 1.0 / float(cs);
    float meanDigammaValue = 0.0;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float kthDist = kthNeighborDistances[c] - EPSILON;
        float currentValue = valueArray[c];
        float searchValueLower = currentValue - kthDist;
        float searchValueUpper = currentValue + kthDist;
        int lower = 0;
        int upper = MEMBER_COUNT;
        int middle = 0;
        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            float middleValue = valueArray[middle];
            if (middleValue < searchValueLower) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }

        int startRange = upper;
        lower = startRange;
        upper = MEMBER_COUNT;

        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            float middleValue = valueArray[middle];
            if (middleValue < searchValueUpper) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        int endRange = upper - 1;

        uint numPoints = max(uint(endRange + 1 - startRange), 1u);
        meanDigammaValue += factor * digamma(numPoints);
    }
    return meanDigammaValue;
}

-- Compute

#version 450 core
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_nonuniform_qualifier : require
//#extension GL_EXT_debug_printf : enable

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

#ifdef USE_REQUESTS_BUFFER
#include "RequestsBuffer.glsl"
#else
layout (binding = 1, r32f) uniform writeonly image3D outputImage;
layout(push_constant) uniform PushConstants {
    ivec3 referencePointIdx;
    int padding0;
    uvec3 batchOffset;
    uint padding1;
};
#endif

#define MUTUAL_INFORMATION_KRASKOV
#include "ScalarFields.glsl"


/*
 * For more details, please refer to:
 * - https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
 * - https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py
 */

/*
 * Global defines:
 * - MEMBER_COUNT
 * - MAX_STACK_SIZE_BUILD: 2 * uint32_t(ceil(log(MEMBER_COUNT + 1))); 2*10 for 1000 ensemble members.
 * - MAX_STACK_SIZE_KN: uint32_t(ceil(log(MEMBER_COUNT + 1))); 10 for 1000 ensemble members.
 */

//const uint k = 3; //< Set via define.
//const uint base = 2;

float referenceValues[MEMBER_COUNT];
float queryValues[MEMBER_COUNT];
float kthNeighborDistances[MEMBER_COUNT];

const uint INVALID_NODE = 0xFFFFu;
#define FLT_MAX 3.402823466e+38

//#define KD_TREE_ITERATIVE_WITH_STACK

struct KdNode {
    vec2 point;
#ifdef KD_TREE_ITERATIVE_WITH_STACK
    uint axis;
    uint leftRightIdx;
#endif
};
KdNode nodes[MEMBER_COUNT];

struct StackEntryBuild {
    uint startIdx;
    uint endIdx;
    uint depth;
};

#import ".Sort2D"

#ifdef KD_TREE_ITERATIVE_WITH_STACK
void buildKdTree() {
    uint nodeCounter = 0;
    StackEntryBuild stack[MAX_STACK_SIZE_BUILD];
    uint stackSize = 1u;
    stack[0] = StackEntryBuild(0u, MEMBER_COUNT, 0u);
    StackEntryBuild stackEntry;
    while (stackSize > 0u) {
        stackSize--;
        stackEntry = stack[stackSize];

        uint axis = stackEntry.depth % 2u;
        heapSort2D(stackEntry.startIdx, stackEntry.endIdx, axis);
        uint medianIndex = stackEntry.startIdx + (stackEntry.endIdx - stackEntry.startIdx) / 2u;

        uint rightIdx;
        if (stackEntry.endIdx - medianIndex - 1 == 0u) {
            rightIdx = INVALID_NODE;
        } else {
            rightIdx = nodeCounter + medianIndex - stackEntry.startIdx + 1;
            stack[stackSize] = StackEntryBuild(medianIndex + 1, stackEntry.endIdx, stackEntry.depth + 1);
            stackSize++;
        }

        uint leftIdx;
        if (medianIndex - stackEntry.startIdx == 0u) {
            leftIdx = INVALID_NODE;
        } else {
            leftIdx = nodeCounter + 1;
            stack[stackSize] = StackEntryBuild(stackEntry.startIdx, medianIndex, stackEntry.depth + 1);
            stackSize++;
        }

        uint leftRightIdx = (leftIdx | rightIdx << 16u);
        nodes[nodeCounter] = KdNode(vec2(referenceValues[medianIndex], queryValues[medianIndex]), axis, leftRightIdx);
        nodeCounter++;
    }
}
#else
void buildKdTree() {
    StackEntryBuild stack[MAX_STACK_SIZE_BUILD];
    uint stackSize = 1u;
    stack[0] = StackEntryBuild(0u, MEMBER_COUNT, 0u);
    StackEntryBuild stackEntry;
    while (stackSize > 0u) {
        stackSize--;
        stackEntry = stack[stackSize];

        uint curr = stackEntry.depth;
        //uint axis = (31 - __clz(curr + 1)) % 2u;
        uint axis = uint(findMSB(curr + 1)) % 2u;
        heapSort2D(stackEntry.startIdx, stackEntry.endIdx, axis);

        int n = int(stackEntry.endIdx - stackEntry.startIdx);
        //int H = getTreeHeight(n);
        //int H = 32 - __clz(n);
        int H = findMSB(n) + 1;
        uint medianIndex = stackEntry.startIdx;
        if (n > 1) {
            medianIndex += uint((1 << (H - 2)) - 1 + min(1 << (H - 2), n - (1 << (H - 1)) + 1));
        }
        //uint medianIndex = stackEntry.startIdx + (stackEntry.endIdx - stackEntry.startIdx) / 2u;

        if (medianIndex - stackEntry.startIdx != 0u) {
            stack[stackSize] = StackEntryBuild(stackEntry.startIdx, medianIndex, curr * 2u + 1u);
            stackSize++;
        }

        if (stackEntry.endIdx - medianIndex - 1 != 0u) {
            stack[stackSize] = StackEntryBuild(medianIndex + 1, stackEntry.endIdx, curr * 2u + 2u);
            stackSize++;
        }

        nodes[curr] = KdNode(vec2(referenceValues[medianIndex], queryValues[medianIndex]));
    }
}
#endif


#ifdef KD_TREE_ITERATIVE_WITH_STACK
float findKNearestNeighbors(vec2 point, uint c) {
    float distances[k + 1];
    [[unroll]] for (int i = 0; i <= k; i++) {
        distances[i] = FLT_MAX;
    }

    uint stack[MAX_STACK_SIZE_KN];
    uint stackSize = 0u;
    uint currNodeIdx = 0u;
    KdNode currNode;
    while (currNodeIdx != INVALID_NODE || stackSize > 0u) {
        while (currNodeIdx != INVALID_NODE) {
            stack[stackSize] = currNodeIdx;
            stackSize++;
            currNode = nodes[currNodeIdx];

            // Descend on side of split planes where the point lies.
            bool isPointOnLeftSide = point[currNode.axis] <= currNode.point[currNode.axis];
            if (isPointOnLeftSide) {
                currNodeIdx = currNode.leftRightIdx & 0x0000FFFFu;
            } else {
                currNodeIdx = (currNode.leftRightIdx & 0xFFFF0000u) >> 16u;
            }
        }

        stackSize--;
        currNodeIdx = stack[stackSize];
        currNode = nodes[currNodeIdx];

        // Compute the distance of this node to the point.
        vec2 diff = abs(point - currNode.point);
        float newDistance = max(diff.x, diff.y);
        if (newDistance < distances[k]) {
            float tempDistance;
            for (int i = 0; i <= k; i++) {
                if (newDistance < distances[i]) {
                    tempDistance = newDistance;
                    newDistance = distances[i];
                    distances[i] = tempDistance;
                }
            }
        }

        // Check whether there could be a closer point on the opposite side.
        bool isPointOnLeftSide = point[currNode.axis] <= currNode.point[currNode.axis];
        if (isPointOnLeftSide && point[currNode.axis] + distances[k] >= currNode.point[currNode.axis]) {
            currNodeIdx = (currNode.leftRightIdx & 0xFFFF0000u) >> 16u;
        } else if (!isPointOnLeftSide && point[currNode.axis] - distances[k] <= currNode.point[currNode.axis]) {
            currNodeIdx = currNode.leftRightIdx & 0x0000FFFFu;
        } else {
            currNodeIdx = INVALID_NODE;
        }
    }

    return distances[k];
}
#else
/*
 * Stack-free kd-tree traversal based on the following paper:
 * "A Stack-Free Traversal Algorithm for Left-Balanced k-d Trees", Ingo Wald (2022).
 * https://arxiv.org/pdf/2210.12859.pdf
 */
float findKNearestNeighbors(vec2 point, uint c) {
    float distances[k + 1];
    [[unroll]] for (int i = 0; i <= k; i++) {
        distances[i] = FLT_MAX;
    }

    int curr = 0;
    int prev = -1;
    KdNode currNode;
    while (true) {
        int parent = (curr + 1) / 2 - 1;
        if (curr >= MEMBER_COUNT) {
            prev = curr;
            curr = parent;
            continue;
        }
        currNode = nodes[curr];

        bool prevIsParent = prev < curr;
        if (prevIsParent) {
            // Compute the distance of this node to the point.
            vec2 diff = abs(point - currNode.point);
            float newDistance = max(diff.x, diff.y);
            if (newDistance < distances[k]) {
                float tempDistance;
                for (int i = 0; i <= k; i++) {
                    if (newDistance < distances[i]) {
                        tempDistance = newDistance;
                        newDistance = distances[i];
                        distances[i] = tempDistance;
                    }
                }
            }
        }

        //int splitDim = (31 - __clz(curr + 1)) % 2;
        int splitDim = findMSB(curr + 1) % 2;
        float splitPos = currNode.point[splitDim];
        float signedDist = point[splitDim] - splitPos;
        int closeSide = int(signedDist > 0.0);
        int closeChild = 2 * curr + 1 + closeSide;
        int farChild = 2 * curr + 2 - closeSide;
        bool farInRange = abs(signedDist) <= distances[k];

        int next;
        if (prevIsParent) {
            next = closeChild;
        } else if (prev == closeChild) {
            next = farInRange ? farChild : parent;
        } else {
            next = parent;
        }
        if (next == -1) {
            break;
        }

        prev = curr;
        curr = next;
    }

    return distances[k];
}
#endif

/**
 * Lanczos approximation of digamma function using weights by Viktor T. Toth.
 * - digamma = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x) (https://en.wikipedia.org/wiki/Digamma_function)
 * - Lanczos approximation: https://www.rskey.org/CMS/index.php/the-library/11
 * - Weights: https://www.rskey.org/CMS/index.php/the-library/11
 *
 * This function could be extended for values < 1 by:
 * - float z = 1 - iz;
 * - return digammaValue - M_PI * cos(M_PI * iz) / sin(M_PI * iz);
 */
#define G (5.15)
#define P0 (2.50662827563479526904)
#define P1 (225.525584619175212544)
#define P2 (-268.295973841304927459)
#define P3 (80.9030806934622512966)
#define P4 (-5.00757863970517583837)
#define P5 (0.0114684895434781459556)
float digamma(uint iz) {
    if (iz == 1u) {
        return -0.57721566490153287;
    }
    float z = float(iz);
    float zh = z - 0.5;
    float z1 = z + 1.0;
    float z2 = z + 2.0;
    float z3 = z + 3.0;
    float z4 = z + 4.0;
    float ZP = P0 + P1 / z + P2 / z1 + P3 / z2 + P4 / z3 + P5 / z4;
    float dZP = P1 / (z * z) + P2 / (z1 * z1) + P3 / (z2 * z2) + P4 / (z3 * z3) + P5 / (z4 * z4);
    float digammaValue = log(zh + G) + zh / (zh + G) - dZP / ZP - 1.0;

    /*
     * Alternative formulation:
     * float zh = z + 0.5f;
     * float z1 = z + 1.0f;
     * float z2 = z + 2.0f;
     * float z3 = z + 3.0f;
     * float z4 = z + 4.0f;
     * float z5 = z + 5.0f;
     * float ZP = P0 + P1 / z1 + P2 / z2 + P3 / z3 + P4 / z4 + P5 / z5;
     * float dZP = P1 / (z1 * z1) + P2 / (z2 * z2) + P3 / (z3 * z3) + P4 / (z4 * z4) + P5 / (z5 * z5);
     * float digammaValue = log(zh + G) - (zh + G + G * z)/(z * (zh + G)) - dZP / ZP;
     */

    return digammaValue;
}


const float EPSILON = 1e-6;
const float EPSILON_NOISE = 1e-5;

#define REF 111
#define averageDigamma averageDigammaReference
#define valueArray referenceValues
#define swapElements swapElementsReference
#define heapify heapifyReference
#define heapSort heapSortReference
#import ".Sort1D"
#undef averageDigamma
#undef valueArray
#undef swapElements
#undef heapify
#undef heapSort

#undef REF
#define REF 222
#define averageDigamma averageDigammaQuery
#define valueArray queryValues
#define swapElements swapElementsQuery
#define heapify heapifyQuery
#define heapSort heapSortQuery
#import ".Sort1D"
#undef averageDigamma
#undef valueArray
#undef swapElements
#undef heapify
#undef heapSort

#define KRASKOV_USE_RANDOM_NOISE

#ifdef KRASKOV_USE_RANDOM_NOISE
float getRandomFloatNorm(inout uvec3 rngState) {
    rngState.x ^= rngState.x << 16;
    rngState.x ^= rngState.x >> 5;
    rngState.x ^= rngState.x << 1;

    uint t = rngState.x;
    rngState.x = rngState.y;
    rngState.y = rngState.z;
    rngState.z = t ^ rngState.x ^ rngState.y;

    return rngState.z / float(4294967295u) * 2.0 - 1.0;
}
#endif

void main() {
#include "CorrelationMain.glsl"

    float nanValue = 0.0;
    float value;

#ifdef KRASKOV_USE_RANDOM_NOISE
#ifdef USE_REQUESTS_BUFFER
    uint globalThreadIdx = gl_GlobalInvocationID.x;
#else
    uint globalThreadIdx = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * xs + gl_GlobalInvocationID.z * xs * ys;
#endif
    uint seed = 17u * globalThreadIdx + 240167u;

    // Use Xorshift random numbers with period 2^96-1.
    uvec3 rngState;
    rngState.x = 123456789u ^ seed;
    rngState.y = 362436069u ^ seed;
    rngState.z = 521288629u ^ seed;

#if !defined(USE_SCALAR_FIELD_IMAGES) && !defined(SEPARATE_REFERENCE_AND_QUERY_FIELDS)
    uint referenceIdx = IDXSR(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z);
#endif
#if !defined(USE_SCALAR_FIELD_IMAGES)
    uint queryIdx = IDXSQ(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

    // Optionally add noise.
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        value =
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
                referenceValuesOrig[c]
#else
#ifdef USE_SCALAR_FIELD_IMAGES
                texelFetch(sampler3D(scalarFieldsRef[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r
#else
                scalarFieldsRef[nonuniformEXT(c)].values[referenceIdx]
#endif
#endif
                + EPSILON_NOISE * getRandomFloatNorm(rngState);
        if (isnan(value)) {
            nanValue = value;
        }
        referenceValues[c] = value;
        value =
#ifdef USE_SCALAR_FIELD_IMAGES
                texelFetch(sampler3D(scalarFieldsQuery[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r
#else
                scalarFieldsQuery[nonuniformEXT(c)].values[queryIdx]
#endif
                + EPSILON_NOISE * getRandomFloatNorm(rngState);
        if (isnan(value)) {
            nanValue = value;
        }
        queryValues[c] = value;
    }
#else
    float nanValue = 0.0;
    float value;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        value = referenceValuesOrig[c];
#else
#ifdef USE_SCALAR_FIELD_IMAGES
        value = texelFetch(sampler3D(scalarFieldsRef[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
#else
        value = scalarFieldsRef[nonuniformEXT(c)].values[referenceIdx];
#endif
#endif
        if (isnan(value)) {
            nanValue = value;
        }
        referenceValues[c] = value;
#ifdef USE_SCALAR_FIELD_IMAGES
        value = texelFetch(sampler3D(scalarFieldsQuery[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
#else
        value = scalarFieldsQuery[nonuniformEXT(c)].values[queryIdx];
#endif
        if (isnan(value)) {
            nanValue = value;
        }
        queryValues[c] = value;
    }
#endif

    buildKdTree();

    for (uint c = 0; c < MEMBER_COUNT; c++) {
        kthNeighborDistances[c] = findKNearestNeighbors(vec2(referenceValues[c], queryValues[c]), c);
    }

    float a = averageDigammaReference();
    float b = averageDigammaQuery();
    float c = digamma(k);
    float d = digamma(cs);
    //float mi = (-a - b + c + d) / log(base);
    float mi = -a - b + c + d;
    mi = max(mi, 0.0);

#ifdef MI_CORRELATION_COEFFICIENT
    mi = sqrt(1.0 - exp(-2.0 * mi));
#endif

#ifdef USE_REQUESTS_BUFFER
    outputBuffer[requestIdx] = isnan(nanValue) ? nanValue : mi;
#else
    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : mi));
#endif
}
