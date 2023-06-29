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

#define IDXS(x,y,z) ((z)*xs*ys + (y)*xs + (x))

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
#define TILE_SIZE_Z 4
__device__ inline uint32_t IDXSTf(uint32_t x, uint32_t y, uint32_t z, uint32_t xs, uint32_t ys, uint32_t zs) {
    uint32_t xst = (xs - 1) / TILE_SIZE_X + 1;
    uint32_t yst = (ys - 1) / TILE_SIZE_Y + 1;
    //uint32_t zst = (zs - 1) / TILE_SIZE_Z + 1;
    uint32_t xt = x / TILE_SIZE_X;
    uint32_t yt = y / TILE_SIZE_Y;
    uint32_t zt = z / TILE_SIZE_Z;
    uint32_t tileAddressLinear = (xt + yt * xst + zt * xst * yst) * (TILE_SIZE_X * TILE_SIZE_Y * TILE_SIZE_Z);
    uint32_t vx = x & (TILE_SIZE_X - 1u);
    uint32_t vy = y & (TILE_SIZE_Y - 1u);
    uint32_t vz = z & (TILE_SIZE_Z - 1u);
    uint32_t voxelAddressLinear = vx + vy * TILE_SIZE_X + vz * TILE_SIZE_X * TILE_SIZE_Y;
    return tileAddressLinear | voxelAddressLinear;
}
#define IDXST(x,y,z) IDXSTf(x, y, z, xs, ys, zs)

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


// Start: ------------- Combine correlation members query -------------
extern "C" __global__ void combineCorrelationMembers(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint32_t batchOffset, uint32_t batchSize,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, cudaTextureObject_t* scalarFields) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * cs;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t c = 0; c < cs; c++) {
        //float fieldValue = tex3Dfetch(scalarFields[c], make_int4(x, y, z, 0)).x;
#ifdef USE_NORMALIZED_COORDINATES
        float fieldValue = tex3D<float>(
                scalarFields[c],
                (float(x) + 0.5f) / float(xs),
                (float(y) + 0.5f) / float(ys),
                (float(z) + 0.5f) / float(zs));
#else
        float fieldValue = tex3D<float>(scalarFields[c], float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
#endif
        fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
        outputBuffer[pointIdxWriteOffset + c] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}

extern "C" __global__ void combineCorrelationMembersBuffer(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint32_t batchOffset, uint32_t batchSize,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * cs;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t c = 0; c < cs; c++) {
        float fieldValue = scalarFields[c][IDXS(x, y, z)];
        fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
        outputBuffer[pointIdxWriteOffset + c] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}

extern "C" __global__ void combineCorrelationMembersBufferTiled(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint32_t batchOffset, uint32_t batchSize,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * cs;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t c = 0; c < cs; c++) {
        float fieldValue = scalarFields[c][IDXST(x, y, z)];
        fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
        outputBuffer[pointIdxWriteOffset + c] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}
// End: ------------- Combine correlation members query -------------


// Start: ------------- Combine correlation members query aligned -------------
extern "C" __global__ void combineCorrelationMembersAligned(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint32_t batchOffset, uint32_t batchSize,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, cudaTextureObject_t* scalarFields, uint32_t alignment) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * cs;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t c = 0; c < cs; c++) {
        //float fieldValue = tex3Dfetch(scalarFields[c], make_int4(x, y, z, 0)).x;
#ifdef USE_NORMALIZED_COORDINATES
        float fieldValue = tex3D<float>(
                scalarFields[c],
                (float(x) + 0.5f) / float(xs),
                (float(y) + 0.5f) / float(ys),
                (float(z) + 0.5f) / float(zs));
#else
        float fieldValue = tex3D<float>(scalarFields[c], float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);
#endif
        fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
        outputBuffer[(pointIdxWriteOffset + c) * alignment] =
                make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}

extern "C" __global__ void combineCorrelationMembersAlignedBuffer(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint32_t batchOffset, uint32_t batchSize,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields, uint32_t alignment) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * cs;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t c = 0; c < cs; c++) {
        float fieldValue = scalarFields[c][IDXS(x, y, z)];
        fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
        outputBuffer[(pointIdxWriteOffset + c) * alignment] =
                make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}

extern "C" __global__ void combineCorrelationMembersAlignedBufferTiled(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint32_t batchOffset, uint32_t batchSize,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields, uint32_t alignment) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * cs;
    uint32_t pointIdxReadOffset = globalThreadIdx + batchOffset;
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    float3 pointCoords = make_float3(
            2.0f * float(x) / float(xs - 1) - 1.0f,
            2.0f * float(y) / float(ys - 1) - 1.0f,
            2.0f * float(z) / float(zs - 1) - 1.0f);
    for (uint32_t c = 0; c < cs; c++) {
        float fieldValue = scalarFields[c][IDXST(x, y, z)];
        fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
        outputBuffer[(pointIdxWriteOffset + c) * alignment] =
                make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
    }
}
// End: ------------- Combine correlation members query aligned -------------


// Start: ------------- Combine correlation members reference -------------
extern "C" __global__ void combineCorrelationMembersReference(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint3 referencePointIdx,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, cudaTextureObject_t* scalarFields) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cs) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
#ifdef USE_NORMALIZED_COORDINATES
    float fieldValue = tex3D<float>(
                scalarFields[c],
                (float(referencePointIdx.x) + 0.5f) / float(xs),
                (float(referencePointIdx.y) + 0.5f) / float(ys),
                (float(referencePointIdx.z) + 0.5f) / float(zs));
#else
    float fieldValue = tex3D<float>(
            scalarFields[c],
            float(referencePointIdx.x) + 0.5f,
            float(referencePointIdx.y) + 0.5f,
            float(referencePointIdx.z) + 0.5f);
#endif
    fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
    outputBuffer[c] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
}

extern "C" __global__ void combineCorrelationMembersReferenceBuffer(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint3 referencePointIdx,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cs) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
    float fieldValue = scalarFields[c][IDXS(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z)];
    fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
    outputBuffer[c] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
}

extern "C" __global__ void combineCorrelationMembersReferenceBufferTiled(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint3 referencePointIdx,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cs) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
    float fieldValue = scalarFields[c][IDXST(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z)];
    fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
    outputBuffer[c] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
}
// End: ------------- Combine correlation members reference -------------


// Start: ------------- Combine correlation members reference aligned -------------
extern "C" __global__ void combineCorrelationMembersReferenceAligned(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint3 referencePointIdx,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, cudaTextureObject_t* scalarFields, uint32_t alignment) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cs) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
#ifdef USE_NORMALIZED_COORDINATES
    float fieldValue = tex3D<float>(
                scalarFields[c],
                (float(referencePointIdx.x) + 0.5f) / float(xs),
                (float(referencePointIdx.y) + 0.5f) / float(ys),
                (float(referencePointIdx.z) + 0.5f) / float(zs));
#else
    float fieldValue = tex3D<float>(
            scalarFields[c],
            float(referencePointIdx.x) + 0.5f,
            float(referencePointIdx.y) + 0.5f,
            float(referencePointIdx.z) + 0.5f);
#endif
    fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
    outputBuffer[c * alignment] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
}

extern "C" __global__ void combineCorrelationMembersReferenceAlignedBuffer(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint3 referencePointIdx,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields, uint32_t alignment) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cs) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
    float fieldValue = scalarFields[c][IDXS(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z)];
    fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
    outputBuffer[c * alignment] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
}

extern "C" __global__ void combineCorrelationMembersReferenceAlignedBufferTiled(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t cs, uint3 referencePointIdx,
        float minFieldVal, float maxFieldVal,
        float4* __restrict__ outputBuffer, const float** __restrict__ scalarFields, uint32_t alignment) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cs) {
        return;
    }

    float3 pointCoords = make_float3(
            2.0f * float(referencePointIdx.x) / float(xs - 1) - 1.0f,
            2.0f * float(referencePointIdx.y) / float(ys - 1) - 1.0f,
            2.0f * float(referencePointIdx.z) / float(zs - 1) - 1.0f);
    float fieldValue = scalarFields[c][IDXST(referencePointIdx.x, referencePointIdx.y, referencePointIdx.z)];
    fieldValue = (fieldValue - minFieldVal) / (maxFieldVal - minFieldVal);
    outputBuffer[c * alignment] = make_float4(fieldValue, pointCoords.x, pointCoords.y, pointCoords.z);
}
// End: ------------- Combine correlation members reference aligned -------------


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

extern "C" __global__ void writeGridPositionsStencil(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t batchOffset, uint32_t batchSize,
        float* __restrict__ outputBuffer, uint32_t stride, const uint32_t* __restrict__ nonNanIndexBuffer) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= batchSize) {
        return;
    }
    uint32_t pointIdxWriteOffset = globalThreadIdx * stride;
    uint32_t pointIdxReadOffset = nonNanIndexBuffer[globalThreadIdx + batchOffset];
    uint32_t x = pointIdxReadOffset % xs;
    uint32_t y = (pointIdxReadOffset / xs) % ys;
    uint32_t z = pointIdxReadOffset / (xs * ys);
    outputBuffer[pointIdxWriteOffset] = 2.0f * float(x) / float(xs - 1) - 1.0f;
    outputBuffer[pointIdxWriteOffset + 1] = 2.0f * float(y) / float(ys - 1) - 1.0f;
    outputBuffer[pointIdxWriteOffset + 2] = 2.0f * float(z) / float(zs - 1) - 1.0f;
}

extern "C" __global__ void unpackStencilValues(
        uint32_t numNonNanValues, const uint32_t* __restrict__ nonNanIndexBuffer,
        const float* __restrict__ outputImageBuffer, float* __restrict__ outputImageBufferUnpacked) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= numNonNanValues) {
        return;
    }
    uint32_t pointIdxReadOffset = globalThreadIdx;
    uint32_t pointIdxWriteOffset = nonNanIndexBuffer[globalThreadIdx];
    outputImageBufferUnpacked[pointIdxWriteOffset] = outputImageBuffer[pointIdxReadOffset];
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
