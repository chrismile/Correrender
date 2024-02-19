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

#include "ScalarFields.glsl"


void main() {
#include "CorrelationMain.glsl"

#ifndef SEPARATE_REFERENCE_AND_QUERY_FIELDS
    float referenceValues[MEMBER_COUNT];
#endif
    float queryValues[MEMBER_COUNT];

#if !defined(USE_SCALAR_FIELD_IMAGES) && !defined(SEPARATE_REFERENCE_AND_QUERY_FIELDS)
    uint referenceIdx = IDXSR(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z);
#endif
#if !defined(USE_SCALAR_FIELD_IMAGES)
    uint queryIdx = IDXSQ(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

    float n = float(MEMBER_COUNT);
    float meanX = 0;
    float meanY = 0;
    float invN = 1.0 / n;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        float x = referenceValues[c];
#else
        float x = texelFetch(sampler3D(scalarFieldsRef[nonuniformEXT(c)], scalarFieldSampler), referencePointIdx, 0).r;
#endif
        float y = texelFetch(sampler3D(scalarFieldsQuery[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
#else
#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        float x = referenceValues[c];
#else
        float x = scalarFieldsRef[nonuniformEXT(c)].values[referenceIdx];
#endif
        float y = scalarFieldsQuery[nonuniformEXT(c)].values[queryIdx];
#endif
        meanX += invN * x;
        meanY += invN * y;
#ifndef SEPARATE_REFERENCE_AND_QUERY_FIELDS
        referenceValues[c] = x;
#endif
        queryValues[c] = y;
    }
    float varX = 0.0;
    float varY = 0.0;
    float invNm1 = 1.0 / (n - 1.0);
    float correlationValue = 0.0;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float x = referenceValues[c];
        float y = queryValues[c];
        float diffX = x - meanX;
        float diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
        correlationValue += invNm1 * (x - meanX) * (y - meanY);
    }
    float stdDevX = sqrt(varX);
    float stdDevY = sqrt(varY);
    correlationValue /= (stdDevX * stdDevY + 1e-16);

#ifdef CALCULATE_ABSOLUTE_VALUE
    correlationValue = abs(correlationValue);
#endif

#ifdef USE_REQUESTS_BUFFER
    outputBuffer[requestIdx] = correlationValue;
#else
    imageStore(outputImage, currentPointIdx, vec4(correlationValue));
#endif
}
