-- SortJoint

void swapElementsJoint(uint i, uint j) {
    float temp = referenceValues[i];
    referenceValues[i] = referenceValues[j];
    referenceValues[j] = temp;
    temp = queryValues[i];
    queryValues[i] = queryValues[j];
    queryValues[j] = temp;
}

void heapifyJoint(uint i, uint numElements) {
    uint child;
    float childValue0, childValue1;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = referenceValues[child];
        childValue1 = referenceValues[child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        if (referenceValues[i] >= childValue0) {
            break;
        }
        swapElementsJoint(i, child);
        i = child;
    }
}

void heapSortJoint() {
    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = MEMBER_COUNT / 2; i > 0; i--) {
        heapifyJoint(i - 1, MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < MEMBER_COUNT; i++) {
        swapElementsJoint(0, MEMBER_COUNT - i);
        heapifyJoint(0, MEMBER_COUNT - i);
    }
}


-- SortRange

void swapElementsRange(uint i, uint j) {
    float temp = sortArray[i];
    sortArray[i] = sortArray[j];
    sortArray[j] = temp;
}

void heapifyRange(uint startIdx, uint i, uint numElements) {
    uint child;
    float childValue0, childValue1, arrayI;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = sortArray[startIdx + child];
        childValue1 = sortArray[startIdx + child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        if (sortArray[startIdx + i] >= childValue0) {
            break;
        }
        swapElementsRange(startIdx + i, startIdx + child);
        i = child;
    }
}

void heapSortRange(uvec2 range) {
    uint numElements = range.y - range.x + 1;

    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = numElements / 2; i > 0; i--) {
        heapifyRange(range.x, i - 1, numElements);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < numElements; i++) {
        swapElementsRange(range.x, range.x + numElements - i);
        heapifyRange(range.x, 0, numElements - i);
    }
}


-- Compute

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

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

#define KENDALL_RANK_CORRELATION
#include "ScalarFields.glsl"


/*
 * Global defines:
 * - MEMBER_COUNT: Number of entries to compute the correlation for.
 * - MAX_STACK_SIZE: uint32_t(ceil(log(MEMBER_COUNT))) + 1; 11 for 1000 ensemble members.
 */

float referenceValues[MEMBER_COUNT];
float queryValues[MEMBER_COUNT];
float sortArray[MEMBER_COUNT];

#import ".SortJoint"
#import ".SortRange"

// https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
uint M(uvec2 leftRange, uvec2 rightRange) {
    uint i = leftRange.x;
    uint j = rightRange.x;
    uint numSwaps = 0;
    while (i <= leftRange.y && j <= rightRange.y) {
        if (sortArray[j] < sortArray[i]) {
            numSwaps += leftRange.y + 1 - i;
            j += 1;
        } else {
            i += 1;
        }
    }
    return numSwaps;
}

uint S() {
    uint sum = 0;
    uvec2 stack[MAX_STACK_SIZE];
    stack[0] = ivec2(0, MEMBER_COUNT - 1);
    uint stackSize = 1;
    while (stackSize > 0) {
        uvec2 range = stack[stackSize - 1];
        stackSize--;
        if (range.y - range.x == 0) {
            continue;
        }
        uint s = (range.y - range.x + 1) / 2;
        for (uint i = range.x; i <= range.y; i++) {
            sortArray[i] = queryValues[i];
        }
        uvec2 rangeLeft = uvec2(range.x, range.x + s - 1);
        uvec2 rangeRight = uvec2(range.x + s, range.y);
        heapSortRange(rangeLeft);
        heapSortRange(rangeRight);
        sum += M(rangeLeft, rangeRight);
        stack[stackSize] = rangeLeft;
        stack[stackSize + 1] = rangeRight;
        stackSize += 2;
    }
    return sum;
}

void main() {
#include "CorrelationMain.glsl"

#if !defined(USE_SCALAR_FIELD_IMAGES) && !defined(SEPARATE_REFERENCE_AND_QUERY_FIELDS)
    uint referenceIdx = IDXSR(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z);
#endif
#if !defined(USE_SCALAR_FIELD_IMAGES)
    uint queryIdx = IDXSQ(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

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

    heapSortJoint();
    int S_y = int(S());

    int n0 = (MEMBER_COUNT * (MEMBER_COUNT - 1)) / 2;

    // Use Tau-a statistic without accounting for ties for now.
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;

    int numerator = n0 - n1 - n2 + n3 - 2 * S_y;
    // The square root needs to be taken separately to avoid integer overflow.
    float denominator = sqrt(float(n0 - n1)) * sqrt(float(n0 - n2));
    float correlationValue = float(numerator) / denominator;

#ifdef CALCULATE_ABSOLUTE_VALUE
    correlationValue = abs(correlationValue);
#endif

#ifdef USE_REQUESTS_BUFFER
    outputBuffer[requestIdx] = isnan(nanValue) ? nanValue : correlationValue;
#else
    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : correlationValue));
#endif
}
