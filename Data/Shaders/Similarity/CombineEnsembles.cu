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

#include <cuda_runtime.h>
#include <cstdint>

typedef unsigned uint32_t;

//#define USE_NORMALIZED_COORDINATES

extern "C" __global__ void memcpyFloatClampToZero(
        float* outputBuffer, const float* inputBuffer, uint32_t numElements) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < numElements) {
        float value = inputBuffer[globalThreadIdx];
        if (value < 0.0f) {
            value = 0.0f;
        }
        outputBuffer[globalThreadIdx] = value;
    }
}

extern "C" __global__ void combineEnsembles(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t es, uint32_t batchOffset, uint32_t batchSize,
        float minEnsembleVal, float maxEnsembleVal,
        float4* outputBuffer, cudaTextureObject_t* scalarFieldEnsembles) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * es;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t e = 0; e < es; e++) {
        //float ensembleValue = tex3Dfetch(scalarFieldEnsembles[e], make_int4(x, y, z, 0)).x;
#ifdef USE_NORMALIZED_COORDINATES
        float ensembleValue = tex3D<float>(
                scalarFieldEnsembles[e],
                (float(x) + 0.5f) / float(xs),
                (float(y) + 0.5f) / float(ys),
                (float(z) + 0.5f) / float(zs));
#else
        float ensembleValue = tex3D<float>(scalarFieldEnsembles[e], float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
#endif
        ensembleValue = (ensembleValue - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
        outputBuffer[pointIdxWriteOffset + e] = make_float4(ensembleValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}

extern "C" __global__ void combineEnsemblesReference(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t es, uint3 referencePointIdx,
        float minEnsembleVal, float maxEnsembleVal,
        float4* outputBuffer, cudaTextureObject_t* scalarFieldEnsembles) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= es) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
#ifdef USE_NORMALIZED_COORDINATES
    float ensembleValue = tex3D<float>(
                scalarFieldEnsembles[e],
                (float(referencePointIdx.x) + 0.5f) / float(xs),
                (float(referencePointIdx.y) + 0.5f) / float(ys),
                (float(referencePointIdx.z) + 0.5f) / float(zs));
#else
    float ensembleValue = tex3D<float>(
            scalarFieldEnsembles[e],
            float(referencePointIdx.x) + 0.5f,
            float(referencePointIdx.y) + 0.5f,
            float(referencePointIdx.z) + 0.5f);
#endif
    ensembleValue = (ensembleValue - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
    outputBuffer[e] = make_float4(ensembleValue, pointCoords.x, pointCoords.y, pointCoords.z);
}

extern "C" __global__ void combineEnsemblesAligned(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t es, uint32_t batchOffset, uint32_t batchSize,
        float minEnsembleVal, float maxEnsembleVal,
        float4* outputBuffer, cudaTextureObject_t* scalarFieldEnsembles, uint32_t alignment) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * es;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t e = 0; e < es; e++) {
        //float ensembleValue = tex3Dfetch(scalarFieldEnsembles[e], make_int4(x, y, z, 0)).x;
#ifdef USE_NORMALIZED_COORDINATES
        float ensembleValue = tex3D<float>(
                scalarFieldEnsembles[e],
                (float(x) + 0.5f) / float(xs),
                (float(y) + 0.5f) / float(ys),
                (float(z) + 0.5f) / float(zs));
#else
        float ensembleValue = tex3D<float>(scalarFieldEnsembles[e], float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
#endif
        ensembleValue = (ensembleValue - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
        outputBuffer[(pointIdxWriteOffset + e) * alignment] =
                make_float4(ensembleValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}

extern "C" __global__ void combineEnsemblesReferenceAligned(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t es, uint3 referencePointIdx,
        float minEnsembleVal, float maxEnsembleVal,
        float4* outputBuffer, cudaTextureObject_t* scalarFieldEnsembles, uint32_t alignment) {
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= es) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
#ifdef USE_NORMALIZED_COORDINATES
    float ensembleValue = tex3D<float>(
                scalarFieldEnsembles[e],
                (float(referencePointIdx.x) + 0.5f) / float(xs),
                (float(referencePointIdx.y) + 0.5f) / float(ys),
                (float(referencePointIdx.z) + 0.5f) / float(zs));
#else
    float ensembleValue = tex3D<float>(
            scalarFieldEnsembles[e],
            float(referencePointIdx.x) + 0.5f,
            float(referencePointIdx.y) + 0.5f,
            float(referencePointIdx.z) + 0.5f);
#endif
    ensembleValue = (ensembleValue - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
    outputBuffer[e * alignment] = make_float4(ensembleValue, pointCoords.x, pointCoords.y, pointCoords.z);
}

extern "C" __global__ void writeGridPositions(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t batchOffset, uint32_t batchSize, float* outputBuffer,
        uint32_t stride) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * stride;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    outputBuffer[pointIdxWriteOffset] = 2.0f * float(x) / float(xs - 1) - 1.0f;
    outputBuffer[pointIdxWriteOffset + 1] = 2.0f * float(y) / float(ys - 1) - 1.0f;
    outputBuffer[pointIdxWriteOffset + 2] = 2.0f * float(z) / float(zs - 1) - 1.0f;
}

extern "C" __global__ void writeGridPositionReference(
        uint32_t xs, uint32_t ys, uint32_t zs, uint3 referencePointIdx, float* outputBuffer) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= 1) {
        return;
    }
    outputBuffer[0] = 2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f;
    outputBuffer[1] = 2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f;
    outputBuffer[2] = 2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f;
}
