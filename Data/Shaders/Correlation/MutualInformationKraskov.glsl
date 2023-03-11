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
    //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
    //    debugPrintfEXT("refs %f %f %f %f", referenceValues[0], referenceValues[1], referenceValues[2], referenceValues[3]);
    //}
    float factor = 1.0 / float(cs);
    float meanDigammaValue = 0.0;
    for (uint c = 0; c < cs; c++) {
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

        uint numPoints = uint(endRange + 1 - startRange);
        //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("np %u", numPoints);
        //}
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

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, cs;
};
layout (binding = 1, r32f) uniform writeonly image3D outputImage;
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFields[MEMBER_COUNT];

layout(push_constant) uniform PushConstants {
    ivec3 referencePointIdx;
    int padding0;
    uvec3 batchOffset;
    uint padding1;
};


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

struct KdNode {
    vec2 point;
    uint axis;
    uint leftRightIdx;
};
KdNode nodes[MEMBER_COUNT];

struct StackEntryBuild {
    uint startIdx;
    uint endIdx;
    uint depth;
};

#import ".Sort2D"

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

        //if (nodeCounter == 2 && gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("vs %f %f", referenceValues[stackEntry.startIdx], referenceValues[stackEntry.endIdx-1]);
        //}

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
        //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("add %u %u %f %f %u %u", nodeCounter, axis, referenceValues[medianIndex], queryValues[medianIndex], leftIdx, rightIdx);
        //}
        //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("p %f %f", nodes[nodeCounter].point.x, nodes[nodeCounter].point.y);
        //}
        //if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
        //    debugPrintfEXT("bd (%u): %u", nodeCounter, leftRightIdx);
        //}
        //if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
        //    debugPrintfEXT("bd %u / %i", stackSize, MAX_STACK_SIZE_BUILD);
        //}
        nodeCounter++;
    }
}

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

            //if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
            //    debugPrintfEXT("gd (%u): %u", currNodeIdx, currNode.leftRightIdx);
            //}
            //if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
            //    debugPrintfEXT("bd %u / %i", stackSize, MAX_STACK_SIZE_KN);
            //}

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
        //if (e == 4 && gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("n %f %f %u", currNode.point[0], currNode.point[1], currNode.axis);
        //}
        //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("p %f %f, n %f %f", point[0], point[1], currNode.point[0], currNode.point[1]);
        //}
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
        //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("dist %f %f %f %f", distances[0], distances[1], distances[2], distances[3]);
        //}
        //if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
        //    debugPrintfEXT("kn (%u): %u", currNodeIdx, currNode.leftRightIdx);
        //}

        // Check whether there could be a closer point on the opposite side.
        bool isPointOnLeftSide = point[currNode.axis] <= currNode.point[currNode.axis];
        if (isPointOnLeftSide && point[currNode.axis] + distances[k] >= currNode.point[currNode.axis]) {
            currNodeIdx = (currNode.leftRightIdx & 0xFFFF0000u) >> 16u;
        } else if (!isPointOnLeftSide && point[currNode.axis] - distances[k] <= currNode.point[currNode.axis]) {
            currNodeIdx = currNode.leftRightIdx & 0x0000FFFFu;
        } else {
            currNodeIdx = INVALID_NODE;
        }

        //if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
        //    debugPrintfEXT("fn (%u)", currNodeIdx);
        //}
    }
    //if (gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0) {
    //    debugPrintfEXT("fin");
    //}
    return distances[k];
}


/**
 * Lanczos approximation of digamma function using weights by Viktor T. Toth.
 * - digamma = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x) (https://en.wikipedia.org/wiki/Digamma_function)
 * - Lanczos approximation: https://www.rskey.org/CMS/index.php/the-library/11
 * - Weights: https://www.rskey.org/CMS/index.php/the-library/11
 * - GLSL implementation by: https://www.shadertoy.com/view/3lfGD7
 */
#define M_PI 3.14159265358979323846
#define LG 5.65
#define P0 2.50662827563479526904
#define P1 225.525584619175212544
#define P2 (-268.295973841304927459)
#define P3 80.9030806934622512966
#define P4 (-5.00757863970517583837)
#define P5 0.0114684895434781459556
float digamma(uint ix) {
    if (ix == 1u) {
        return -0.57721566490153287;
    }
    float x = float(ix);
    float xx = x > 1.0 ? x : 1.0 - x;
    float sum1 =
            P1 / ((xx + 1.0) * (xx + 1.0)) + P2/((xx + 2.0) * (xx + 2.0)) + P3 / ((xx + 3.0) * (xx + 3.0))
            + P4 / ((xx + 4.0) * (xx + 4.0)) + P5 / ((xx + 5.0) * (xx + 5.0));
    float sum2 = P0 + P1 / (xx + 1.0) + P2 / (xx + 2.0) + P3/(xx + 3.0) + P4 / (xx + 4.0) + P5 / (xx + 5.0);
    float xh = xx + LG;
    float y = log(xh) - (xh + (LG - 0.5) * xx) / (xx * xh) - sum1 / sum2;
    return x > 1.0 ? y : y - M_PI * cos(M_PI * x) / sin(M_PI * x);
}


const float EPSILON = 1e-6;
const float EPSILON_NOISE = 1e-5;

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
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz + batchOffset);
    if (currentPointIdx.x >= xs || currentPointIdx.y >= ys || currentPointIdx.z >= zs) {
        return;
    }

#ifdef KRASKOV_USE_RANDOM_NOISE
    uint globalThreadIdx = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * xs + gl_GlobalInvocationID.z * xs * ys;
    uint seed = 17u * globalThreadIdx + 240167u;

    // Use Xorshift random numbers with period 2^96-1.
    uvec3 rngState;
    rngState.x = 123456789u ^ seed;
    rngState.y = 362436069u ^ seed;
    rngState.z = 521288629u ^ seed;

    // Optionally add noise.
    float nanValue = 0.0;
    float value;
    for (uint c = 0; c < cs; c++) {
        referenceValues[c] =
                texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r
                + EPSILON_NOISE * getRandomFloatNorm(rngState);
        value =
                texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r
                + EPSILON_NOISE * getRandomFloatNorm(rngState);
        if (isnan(value)) {
            nanValue = value;
        }
        queryValues[c] = value;
    }
#else
    float nanValue = 0.0;
    float value;
    for (uint c = 0; c < cs; c++) {
        referenceValues[c] = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
        //queryValues[c] = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
        value = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
        if (isnan(value)) {
            nanValue = value;
        }
        queryValues[c] = value;
    }
#endif

    buildKdTree();

    //rngState.x = 123456789u ^ seed;
    //rngState.y = 362436069u ^ seed;
    //rngState.z = 521288629u ^ seed;
    //for (uint c = 0; c < cs; c++) {
    //    referenceValues[c] =
    //            texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r
    //            + EPSILON_NOISE * getRandomFloatNorm(rngState);
    //    queryValues[c] =
    //            texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r
    //            + EPSILON_NOISE * getRandomFloatNorm(rngState);
    //}

    for (uint c = 0; c < cs; c++) {
        //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("kref %f, kquery %f", referenceValues[e], queryValues[e]);
        //}
        kthNeighborDistances[c] = findKNearestNeighbors(vec2(referenceValues[c], queryValues[c]), c);
        //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
        //    debugPrintfEXT("kn (%u): %f", e, kthNeighborDistances[e]);
        //}
    }

    float a = averageDigammaReference();
    float b = averageDigammaQuery();
    float c = digamma(k);
    float d = digamma(cs);
    //float mi = (-a - b + c + d) / log(base);
    float mi = -a - b + c + d;

    //if (gl_GlobalInvocationID.x == 29 && gl_GlobalInvocationID.y == 176 && gl_GlobalInvocationID.z == 10) {
    //    debugPrintfEXT("abcd %f %f %f %f", a, b, c, d);
    //    debugPrintfEXT("mi %f", mi);
    //}

    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : max(mi, 0.0)));
}
