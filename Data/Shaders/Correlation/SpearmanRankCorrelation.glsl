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

-- Common

void swapElements(uint i, uint j) {
    float ftemp = valueArray[i];
    valueArray[i] = valueArray[j];
    valueArray[j] = ftemp;
    uint utemp = ordinalRankArray[i];
    ordinalRankArray[i] = ordinalRankArray[j];
    ordinalRankArray[j] = utemp;
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


-- FractionalRanking

void computeFractionalRanking() {
    float currentRank = 1.0f;
    int idx = 0;
    while (idx < MEMBER_COUNT) {
        float value = valueArray[idx];
        int idxEqualEnd = idx + 1;
        while (idxEqualEnd < MEMBER_COUNT && value == valueArray[idxEqualEnd]) {
            idxEqualEnd++;
        }

        int numEqualValues = idxEqualEnd - idx;
        float meanRank = currentRank + float(numEqualValues - 1) * 0.5f;
        for (int offset = 0; offset < numEqualValues; offset++) {
            rankArray[ordinalRankArray[idx + offset]] = meanRank;
        }

        idx += numEqualValues;
        currentRank += float(numEqualValues);
    }
}


-- Compute

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

#ifdef USE_REQUESTS_BUFFER
#include "RequestsBuffer.glsl"
#else
layout(binding = 1, r32f) uniform writeonly image3D outputImage;
layout(push_constant) uniform PushConstants {
    ivec3 referencePointIdx;
    int padding0;
    uvec3 batchOffset;
    uint padding1;
};
layout(binding = 6) readonly buffer ReferenceRankBuffer {
    float referenceRankArray[MEMBER_COUNT];
};
#endif

#include "ScalarFields.glsl"

float valueArray[MEMBER_COUNT];
uint ordinalRankArray[MEMBER_COUNT];
#ifdef USE_REQUESTS_BUFFER
float referenceRankArray[MEMBER_COUNT];
#endif
float queryRankArray[MEMBER_COUNT];

#import ".Common"

#ifdef USE_REQUESTS_BUFFER
#define  computeFractionalRanking computeFractionalRankingReference
#define rankArray referenceRankArray
#import ".FractionalRanking"
#undef computeFractionalRanking
#undef rankArray
#endif

#define  computeFractionalRanking computeFractionalRankingQuery
#define rankArray queryRankArray
#import ".FractionalRanking"
#undef computeFractionalRanking
#undef rankArray

float pearsonCorrelation() {
    float n = float(MEMBER_COUNT);
    float meanX = 0.0;
    float meanY = 0.0;
    float invN = 1.0 / n;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceRankArray[c];
        float y = queryRankArray[c];
        meanX += invN * x;
        meanY += invN * y;
    }
    float varX = 0;
    float varY = 0;
    float invNm1 = 1.0 / (n - 1.0);
    float correlationValue = 0.0;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceRankArray[c];
        float y = queryRankArray[c];
        float diffX = x - meanX;
        float diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
        correlationValue += invNm1 * (x - meanX) * (y - meanY);
    }
    float stdDevX = sqrt(varX);
    float stdDevY = sqrt(varY);
    correlationValue /= stdDevX * stdDevY;
    return correlationValue;
}

void main() {
#include "CorrelationMain.glsl"

    float nanValue = 0.0;
    float value;

#if !defined(USE_SCALAR_FIELD_IMAGES) && !defined(SEPARATE_REFERENCE_AND_QUERY_FIELDS)
    uint referenceIdx = IDXSR(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z);
#endif
#if !defined(USE_SCALAR_FIELD_IMAGES)
    uint queryIdx = IDXSQ(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

#ifdef USE_REQUESTS_BUFFER
    for (uint c = 0; c < MEMBER_COUNT; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
        value = texelFetch(sampler3D(scalarFieldsRef[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
#else
        value = scalarFieldsRef[nonuniformEXT(c)].values[referenceIdx];
#endif
        if (isnan(value)) {
            nanValue = value;
        }
        valueArray[c] = value;
        ordinalRankArray[c] = c;
    }

    heapSort();
    computeFractionalRankingReference();
#endif

    // 1. Fill the value array.
    for (uint c = 0; c < MEMBER_COUNT; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
        value = texelFetch(sampler3D(scalarFieldsQuery[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
#else
        value = scalarFieldsQuery[nonuniformEXT(c)].values[queryIdx];
#endif
        if (isnan(value)) {
            nanValue = value;
        }
        valueArray[c] = value;
        ordinalRankArray[c] = c;
    }

    // 2. Sort both arrays.
    heapSort();

    // 3. Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    computeFractionalRankingQuery();

    // 4. Compute the Pearson correlation of the ranks.
    float correlationValue = pearsonCorrelation();

#ifdef CALCULATE_ABSOLUTE_VALUE
    correlationValue = abs(correlationValue);
#endif

#ifdef USE_REQUESTS_BUFFER
    outputBuffer[requestIdx] = isnan(nanValue) ? nanValue : correlationValue;
#else
    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : correlationValue));
#endif
}


-- Reference.Compute

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "ScalarFields.glsl"

layout(binding = 6, std430) writeonly buffer ReferenceRankBuffer {
    float rankArray[MEMBER_COUNT];
};

layout(push_constant) uniform PushConstants {
    ivec3 referencePointIdx;
};

float valueArray[MEMBER_COUNT];
uint ordinalRankArray[MEMBER_COUNT];

#import ".Common"
#import ".FractionalRanking"

void main() {
#if !defined(USE_SCALAR_FIELD_IMAGES) && !defined(SEPARATE_REFERENCE_AND_QUERY_FIELDS)
    uint referenceIdx = IDXSR(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z);
#endif

    // 1. Fill the value array.
    for (uint c = 0; c < MEMBER_COUNT; c++) {
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        valueArray[c] = referenceValues[c];
#else
#ifdef USE_SCALAR_FIELD_IMAGES
        valueArray[c] = texelFetch(sampler3D(scalarFieldsRef[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
#else
        valueArray[c] = scalarFieldsRef[nonuniformEXT(c)].values[referenceIdx];
#endif
#endif
        ordinalRankArray[c] = c;
    }

    // 2. Sort both arrays.
    heapSort();

    // 3. Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    computeFractionalRanking();
}
