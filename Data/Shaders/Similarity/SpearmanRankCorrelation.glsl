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
    for (i = ENSEMBLE_MEMBER_COUNT / 2; i > 0; i--) {
        heapify(i - 1, ENSEMBLE_MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < ENSEMBLE_MEMBER_COUNT; i++) {
        swapElements(0, ENSEMBLE_MEMBER_COUNT - i);
        heapify(0, ENSEMBLE_MEMBER_COUNT - i);
    }
}

void computeFractionalRanking() {
    float currentRank = 1.0f;
    int idx = 0;
    while (idx < es) {
        float value = valueArray[idx];
        int idxEqualEnd = idx + 1;
        while (idxEqualEnd < es && value == valueArray[idxEqualEnd]) {
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

layout(binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, es;
};
layout(binding = 1, r32f) uniform writeonly image3D outputImage;
layout(binding = 2) uniform sampler scalarFieldSampler;
layout(binding = 3) uniform texture3D scalarFieldEnsembles[ENSEMBLE_MEMBER_COUNT];

layout(binding = 4) readonly buffer ReferenceRankBuffer {
    float referenceRankArray[ENSEMBLE_MEMBER_COUNT];
};

float valueArray[ENSEMBLE_MEMBER_COUNT];
uint ordinalRankArray[ENSEMBLE_MEMBER_COUNT];
float rankArray[ENSEMBLE_MEMBER_COUNT];

#import ".Common"

float pearsonCorrelation() {
    float n = float(es);
    float meanX = 0;
    float meanY = 0;
    float invN = float(1) / n;
    for (uint e = 0; e < es; e++) {
        float x = referenceRankArray[e];
        float y = rankArray[e];
        meanX += invN * x;
        meanY += invN * y;
    }
    float varX = 0;
    float varY = 0;
    float invNm1 = float(1) / (n - float(1));
    for (uint e = 0; e < es; e++) {
        float x = referenceRankArray[e];
        float y = rankArray[e];
        float diffX = x - meanX;
        float diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    float stdDevX = sqrt(varX);
    float stdDevY = sqrt(varY);
    float pearsonCorrelation = 0;
    for (uint e = 0; e < es; e++) {
        float x = referenceRankArray[e];
        float y = rankArray[e];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }
    return pearsonCorrelation;
}

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }

    // 1. Fill the value array.
    float nanValue = 0.0;
    float value;
    for (uint e = 0; e < es; e++) {
        value = texelFetch(sampler3D(scalarFieldEnsembles[nonuniformEXT(e)], scalarFieldSampler), currentPointIdx, 0).r;
        if (isnan(value)) {
            nanValue = value;
        }
        valueArray[e] = value;
        ordinalRankArray[e] = e;
    }

    // 2. Sort both arrays.
    heapSort();

    // 3. Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    computeFractionalRanking();

    // 4. Compute the Pearson correlation of the ranks.
    float correlation = pearsonCorrelation();

    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : correlation));
}


-- Reference.Compute

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout(binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, es;
};
layout(binding = 2) uniform sampler scalarFieldSampler;
layout(binding = 3) uniform texture3D scalarFieldEnsembles[ENSEMBLE_MEMBER_COUNT];

layout(binding = 4) writeonly buffer ReferenceRankBuffer {
    float rankArray[ENSEMBLE_MEMBER_COUNT];
};

layout(push_constant) uniform PushConstants {
    ivec3 referencePointIdx;
};

float valueArray[ENSEMBLE_MEMBER_COUNT];
uint ordinalRankArray[ENSEMBLE_MEMBER_COUNT];

#import ".Common"

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }

    // 1. Fill the value array.
    for (uint e = 0; e < es; e++) {
        valueArray[e] = texelFetch(sampler3D(scalarFieldEnsembles[nonuniformEXT(e)], scalarFieldSampler), referencePointIdx, 0).r;
        ordinalRankArray[e] = e;
    }

    // 2. Sort both arrays.
    heapSort();

    // 3. Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    computeFractionalRanking();
}
