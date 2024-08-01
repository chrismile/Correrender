/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

-- Compute

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

#define NEEDS_MIN_MAX_FIELD_VALUE
#ifdef USE_REQUESTS_BUFFER
#include "RequestsBuffer.glsl"
#else
layout (binding = 1, r32f) uniform writeonly image3D outputImage;
layout(push_constant) uniform PushConstants {
    ivec3 referencePointIdx;
    int padding0;
    uvec3 batchOffset;
    uint padding1;
    float minFieldValRef, maxFieldValRef, minFieldValQuery, maxFieldValQuery;
};
#endif

#include "ScalarFields.glsl"

void main() {
#include "CorrelationMain.glsl"

    float histogram2d[numBins * numBins];
    float histogram0[numBins];
    float histogram1[numBins];

    // Initialize the histograms with zeros.
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        histogram0[binIdx] = 0;
        histogram1[binIdx] = 0;
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] = 0;
        }
    }

#if !defined(USE_SCALAR_FIELD_IMAGES) && !defined(SEPARATE_REFERENCE_AND_QUERY_FIELDS)
    uint referenceIdx = IDXSR(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z);
#endif
#if !defined(USE_SCALAR_FIELD_IMAGES)
    uint queryIdx = IDXSQ(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

    // Compute the 2D joint histogram.
    float nanValue = 0.0;
    for (int c = 0; c < MEMBER_COUNT; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        float val0 = referenceValues[c];
#else
        float val0 = texelFetch(sampler3D(scalarFieldsRef[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
#endif
        float val1 = texelFetch(sampler3D(scalarFieldsQuery[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
#else
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        float val0 = referenceValues[c];
#else
        float val0 = scalarFieldsRef[nonuniformEXT(c)].values[referenceIdx];
#endif
        float val1 = scalarFieldsQuery[nonuniformEXT(c)].values[queryIdx];
#endif
        if (!isnan(val0) && !isnan(val1)) {
            val0 = (val0 - minFieldValRef) / (maxFieldValRef - minFieldValRef);
            val1 = (val1 - minFieldValQuery) / (maxFieldValQuery - minFieldValQuery);
            int binIdx0 = clamp(int(val0 * float(numBins)), 0, numBins - 1);
            int binIdx1 = clamp(int(val1 * float(numBins)), 0, numBins - 1);
            histogram2d[binIdx0 * numBins + binIdx1] += 1;
        } else {
            nanValue = isnan(val0) ? val0 : val1;
        }
    }

    // Normalize the histograms.
    float totalSum2d = 0.0;
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            totalSum2d += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] /= totalSum2d;
        }
    }

    // Marginalization of joint distribution.
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram0[binIdx0] += histogram2d[binIdx0 * numBins + binIdx1];
            histogram1[binIdx1] += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }

    /*
     * Compute the mutual information metric. Two possible ways of calculation:
     * a) $MI = H(x) + H(y) - H(x, y)$
     * with the Shannon entropy $H(x) = -\sum_i p_x(i) \log p_x(i)$
     * and the joint entropy $H(x, y) = -\sum_i \sum_j p_{xy}(i, j) \log p_{xy}(i, j)$
     * b) $MI = \sum_i \sum_j p_{xy}(i, j) \log \frac{p_{xy}(i, j)}{p_x(i) p_y(j)}$
     */
    const float EPSILON_1D = 0.5 / float(cs);
    const float EPSILON_2D = 0.5 / float(cs * cs);
    float mi = 0.0;
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        float p_x = histogram0[binIdx];
        float p_y = histogram1[binIdx];
        if (p_x > EPSILON_1D) {
            mi -= p_x * log(p_x);
        }
        if (p_y > EPSILON_1D) {
            mi -= p_y * log(p_y);
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            float p_xy = histogram2d[binIdx0 * numBins + binIdx1];
            if (p_xy > EPSILON_2D) {
                mi += p_xy * log(p_xy);
            }
        }
    }

#ifdef MI_CORRELATION_COEFFICIENT
    // For more information on MICC see "An informational measure of correlation", Linfoot 1957.
    mi = sqrt(1.0 - exp(-2.0 * mi));
#endif

#ifdef USE_REQUESTS_BUFFER
    outputBuffer[requestIdx] = isnan(nanValue) ? nanValue : mi;
#else
    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : mi));
#endif
}
