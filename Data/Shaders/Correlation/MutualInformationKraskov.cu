/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020-2023, Christoph Neuhauser
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

extern "C" {

typedef unsigned uint;
typedef unsigned uint32_t;
typedef uint3 uvec3;
typedef float2 vec2;

/*
 * For more details, please refer to:
 * - https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
 *
 * Global defines:
 * - MEMBER_COUNT: Number of ensemble members.
 * - k: Number of neighbors used in search.
 * - MAX_STACK_SIZE_BUILD: 2 * uint32_t(ceil(log(MEMBER_COUNT + 1))); 2*10 for 1000 ensemble members.
 * - MAX_STACK_SIZE_KN: uint32_t(ceil(log(MEMBER_COUNT + 1))); 10 for 1000 ensemble members.
 */

const float EPSILON = 1e-6;
const float EPSILON_NOISE = 1e-5;

#define KRASKOV_USE_RANDOM_NOISE

#ifdef KRASKOV_USE_RANDOM_NOISE
__device__ float getRandomFloatNorm(uvec3& rngState) {
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


/**
 * Lanczos approximation of digamma function using weights by Viktor T. Toth.
 * - digamma = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x) (https://en.wikipedia.org/wiki/Digamma_function)
 * - Lanczos approximation: https://www.rskey.org/CMS/index.php/the-library/11
 * - Weights: https://www.rskey.org/CMS/index.php/the-library/11
 *
 * This function could be extended for values < 1 by:
 * - float z = 1 - iz;
 * - if (iz < 1) return digammaValue - M_PI * cosf(M_PI * iz) / sinf(M_PI * iz);
 */
#define G (5.15f)
#define P0 (2.50662827563479526904f)
#define P1 (225.525584619175212544f)
#define P2 (-268.295973841304927459f)
#define P3 (80.9030806934622512966f)
#define P4 (-5.00757863970517583837f)
#define P5 (0.0114684895434781459556f)
__device__ float digamma(uint iz) {
    if (iz == 1u) {
        return -0.57721566490153287f;
    }
    float z = float(iz);
    float zh = z - 0.5f;
    float z1 = z + 1.0f;
    float z2 = z + 2.0f;
    float z3 = z + 3.0f;
    float z4 = z + 4.0f;
    float ZP = P0 + P1 / z + P2 / z1 + P3 / z2 + P4 / z3 + P5 / z4;
    float dZP = P1 / (z * z) + P2 / (z1 * z1) + P3 / (z2 * z2) + P4 / (z3 * z3) + P5 / (z4 * z4);
    float digammaValue = logf(zh + G) + zh / (zh + G) - dZP / ZP - 1.0f;
    return digammaValue;
}


// ----------------------------------------------------------------------------------
/*
 * Sort2D.
 */
__device__ void swapElements2D(float* referenceValues, float* queryValues, uint i, uint j) {
    float temp = referenceValues[i];
    referenceValues[i] = referenceValues[j];
    referenceValues[j] = temp;
    temp = queryValues[i];
    queryValues[i] = queryValues[j];
    queryValues[j] = temp;
}

__device__ void heapify2D(
        float* referenceValues, float* queryValues, float* valuesAxis,
        uint startIdx, uint i, uint numElements) {
    uint child;
    float childValue0, childValue1, arrayI;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = valuesAxis[startIdx + child];
        childValue1 = valuesAxis[startIdx + child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        arrayI = valuesAxis[startIdx + i];
        if (arrayI >= childValue0) {
            break;
        }
        swapElements2D(referenceValues, queryValues, startIdx + i, startIdx + child);
        i = child;
    }
}

__device__ void heapSort2D(
        float* referenceValues, float* queryValues, float* valuesAxis,
        uint startIdx, uint endIdx) {
    uint numElements = endIdx - startIdx;

    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = numElements / 2; i > 0; i--) {
        heapify2D(referenceValues, queryValues, valuesAxis, startIdx, i - 1, numElements);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < numElements; i++) {
        swapElements2D(referenceValues, queryValues, startIdx, startIdx + numElements - i);
        heapify2D(referenceValues, queryValues, valuesAxis, startIdx, 0, numElements - i);
    }
}
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
/*
 * Sort1D.
 */
__device__ void swapElements(
        float* kthNeighborDistances, float* valueArray, uint i, uint j) {
    float temp = valueArray[i];
    valueArray[i] = valueArray[j];
    valueArray[j] = temp;
    temp = kthNeighborDistances[i];
    kthNeighborDistances[i] = kthNeighborDistances[j];
    kthNeighborDistances[j] = temp;
}

__device__ void heapify(
        float* kthNeighborDistances, float* valueArray,
        uint i, uint numElements) {
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
        swapElements(kthNeighborDistances, valueArray, i, child);
        i = child;
    }
}

__device__ void heapSort(float* kthNeighborDistances, float* valueArray) {
    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = MEMBER_COUNT / 2; i > 0; i--) {
        heapify(kthNeighborDistances, valueArray, i - 1, MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < MEMBER_COUNT; i++) {
        swapElements(kthNeighborDistances, valueArray, 0, MEMBER_COUNT - i);
        heapify(kthNeighborDistances, valueArray, 0, MEMBER_COUNT - i);
    }
}

__device__ float averageDigamma(float* kthNeighborDistances, float* valueArray) {
    heapSort(kthNeighborDistances, valueArray);
    float factor = 1.0 / float(MEMBER_COUNT);
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
// ----------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------
/*
 * k-d tree.
 */
const uint INVALID_NODE = 0xFFFFu;
#define FLT_MAX 3.402823466e+38

//#define KD_TREE_ITERATIVE_WITH_STACK

struct KdNode {
    __device__ KdNode() {}
#ifdef KD_TREE_ITERATIVE_WITH_STACK
    __device__ KdNode(float x, float y, uint axis, uint leftRightIdx) : axis(axis), leftRightIdx(leftRightIdx) {
        this->point[0] = x;
        this->point[1] = y;
    }
    float point[2];
    uint axis;
    uint leftRightIdx;
#else
    __device__ KdNode(float x, float y) {
        this->point[0] = x;
        this->point[1] = y;
    }
    float point[2];
#endif
};

struct StackEntryBuild {
    __device__ StackEntryBuild() {}
    __device__ StackEntryBuild(uint startIdx, uint endIdx, uint depth) : startIdx(startIdx), endIdx(endIdx), depth(depth) {}
    uint startIdx;
    uint endIdx;
    uint depth;
};

#ifdef KD_TREE_ITERATIVE_WITH_STACK
__device__ void buildKdTree(KdNode* nodes, float* referenceValues, float* queryValues) {
    uint nodeCounter = 0;
    StackEntryBuild stack[MAX_STACK_SIZE_BUILD];
    uint stackSize = 1u;
    stack[0] = StackEntryBuild(0u, MEMBER_COUNT, 0u);
    StackEntryBuild stackEntry;
    while (stackSize > 0u) {
        stackSize--;
        stackEntry = stack[stackSize];

        uint axis = stackEntry.depth % 2u;
        heapSort2D(
                referenceValues, queryValues, axis == 0u ? referenceValues : queryValues,
                stackEntry.startIdx, stackEntry.endIdx);
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
        nodes[nodeCounter] = KdNode(referenceValues[medianIndex], queryValues[medianIndex], axis, leftRightIdx);
        nodeCounter++;
    }
}
#else
__device__ void buildKdTree(KdNode* nodes, float* referenceValues, float* queryValues) {
    StackEntryBuild stack[MAX_STACK_SIZE_BUILD];
    uint stackSize = 1u;
    stack[0] = StackEntryBuild(0u, MEMBER_COUNT, 0u);
    StackEntryBuild stackEntry;
    while (stackSize > 0u) {
        stackSize--;
        stackEntry = stack[stackSize];

        uint curr = stackEntry.depth;
        uint axis = (31 - __clz(curr + 1)) % 2u;
        //uint axis = uint(findMSB(curr + 1)) % 2u;
        heapSort2D(
                referenceValues, queryValues, axis == 0u ? referenceValues : queryValues,
                stackEntry.startIdx, stackEntry.endIdx);

        int n = int(stackEntry.endIdx - stackEntry.startIdx);
        //int H = getTreeHeight(n);
        int H = 32 - __clz(n);
        //int H = findMSB(n) + 1;
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

        nodes[curr] = KdNode(referenceValues[medianIndex], queryValues[medianIndex]);
    }
}
#endif

#ifdef KD_TREE_ITERATIVE_WITH_STACK
__device__ float findKNearestNeighbors(KdNode* nodes, float point[2], uint e) {
    float distances[k + 1];
    #pragma unroll
    for (int i = 0; i <= k; i++) {
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
        vec2 diff = make_float2(fabs(point[0] - currNode.point[0]), fabs(point[1] - currNode.point[1]));
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
__device__ float findKNearestNeighbors(KdNode* nodes, float point[2], uint e) {
    float distances[k + 1];
    #pragma unroll
    for (int i = 0; i <= k; i++) {
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
            vec2 diff = make_float2(fabs(point[0] - currNode.point[0]), fabs(point[1] - currNode.point[1]));
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

        int splitDim = (31 - __clz(curr + 1)) % 2;
        //int splitDim = findMSB(curr + 1) % 2;
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


// ----------------------------------------------------------------------------------

#ifdef SUPPORT_TILING
#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
#define TILE_SIZE_Z 4
__device__ inline uint IDXST(uint x, uint y, uint z, uint xs, uint ys, uint zs) {
    uint xst = (xs - 1) / TILE_SIZE_X + 1;
    uint yst = (ys - 1) / TILE_SIZE_Y + 1;
    //uint zst = (zs - 1) / TILE_SIZE_Z + 1;
    uint xt = x / TILE_SIZE_X;
    uint yt = y / TILE_SIZE_Y;
    uint zt = z / TILE_SIZE_Z;
    uint tileAddressLinear = (xt + yt * xst + zt * xst * yst) * (TILE_SIZE_X * TILE_SIZE_Y * TILE_SIZE_Z);
    uint vx = x & (TILE_SIZE_X - 1u);
    uint vy = y & (TILE_SIZE_Y - 1u);
    uint vz = z & (TILE_SIZE_Z - 1u);
    uint voxelAddressLinear = vx + vy * TILE_SIZE_X + vz * TILE_SIZE_X * TILE_SIZE_Y;
    return tileAddressLinear | voxelAddressLinear;
}
#define IDXS(x,y,z) IDXST(x, y, z, xs, ys, zs)
#else
#define IDXS(x,y,z) ((z)*xs*ys + (y)*xs + (x))
#endif

__global__ void mutualInformationKraskov(
#ifdef USE_SCALAR_FIELD_IMAGES
        cudaTextureObject_t* scalarFields,
#else
        const float** __restrict__ scalarFields,
#endif
        const float* __restrict__ referenceValuesOrig,
        float* __restrict__ miArray,
        uint32_t xs, uint32_t ys, uint32_t zs, uint3 referencePointIdx,
        const uint32_t batchOffset, const uint32_t batchSize) {
    uint globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x + batchOffset;
    if (globalThreadIdx >= batchSize) {
        return;
    }

    uint32_t x = globalThreadIdx % xs;
    uint32_t y = (globalThreadIdx / xs) % ys;
    uint32_t z = globalThreadIdx / (xs * ys);

    float referenceValues[MEMBER_COUNT];
    float queryValues[MEMBER_COUNT];

#ifdef KRASKOV_USE_RANDOM_NOISE
    // Optionally add noise.
    uint seed = 17u * globalThreadIdx + 240167u;

    // Use Xorshift random numbers with period 2^96-1.
    uvec3 rngState;
    rngState.x = 123456789u ^ seed;
    rngState.y = 362436069u ^ seed;
    rngState.z = 521288629u ^ seed;
#endif

#if !defined(USE_SCALAR_FIELD_IMAGES) && !defined(SEPARATE_REFERENCE_AND_QUERY_FIELDS)
    uint referenceIdx = IDXS(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z);
#endif
#if !defined(USE_SCALAR_FIELD_IMAGES)
    uint queryIdx = IDXS(x, y, z);
#endif

    float nanValue = 0.0;
    float referenceValue, queryValue;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        referenceValue = referenceValuesOrig[c];
#else
#ifdef USE_SCALAR_FIELD_IMAGES
#ifdef USE_NORMALIZED_COORDINATES
        referenceValue = tex3D<float>(
                scalarFields[c],
                (float(referencePointIdx.x) + 0.5f) / float(xs),
                (float(referencePointIdx.y) + 0.5f) / float(ys),
                (float(referencePointIdx.z) + 0.5f) / float(zs));
#else
        referenceValue = tex3D<float>(
                scalarFields[c], float(referencePointIdx.x) + 0.5f,
                float(referencePointIdx.y) + 0.5f, float(referencePointIdx.z) + 0.5f);
#endif
#else
        referenceValue = scalarFields[c][referenceIdx];
#endif
#endif

#ifdef USE_SCALAR_FIELD_IMAGES
#ifdef USE_NORMALIZED_COORDINATES
        queryValue = tex3D<float>(
                scalarFields[c],
                (float(x) + 0.5f) / float(xs),
                (float(y) + 0.5f) / float(ys),
                (float(z) + 0.5f) / float(zs));
#else
        queryValue = tex3D<float>(
                scalarFields[c], float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
#endif
#else
        queryValue = scalarFields[c][queryIdx];
#endif

#ifdef KRASKOV_USE_RANDOM_NOISE
        referenceValue += EPSILON_NOISE * getRandomFloatNorm(rngState);
        queryValue += EPSILON_NOISE * getRandomFloatNorm(rngState);
#endif
        if (isnan(queryValue)) {
            nanValue = queryValue;
        }
        referenceValues[c] = referenceValue;
        queryValues[c] = queryValue;
    }

    KdNode nodes[MEMBER_COUNT];
    buildKdTree(nodes, referenceValues, queryValues);

    float point[2];
    float kthNeighborDistances0[MEMBER_COUNT];
    float kthNeighborDistances1[MEMBER_COUNT];
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        point[0] = referenceValues[c];
        point[1] = queryValues[c];
        float val = findKNearestNeighbors(nodes, point, c);
        kthNeighborDistances0[c] = val;
        kthNeighborDistances1[c] = val;
    }

    float a = averageDigamma(kthNeighborDistances0, referenceValues);
    float b = averageDigamma(kthNeighborDistances1, queryValues);
    float c = digamma(k);
    float d = digamma(MEMBER_COUNT);
    //float mi = (-a - b + c + d) / log(base);
    float mi = -a - b + c + d;
    mi = fmaxf(mi, 0.0f);

#ifdef MI_CORRELATION_COEFFICIENT
    // For more information on MICC see "An informational measure of correlation", Linfoot 1957.
    mi = sqrtf(1.0f - expf(-2.0f * mi));
#endif

    miArray[globalThreadIdx] = isnan(nanValue) ? nanValue : mi;
}

}
