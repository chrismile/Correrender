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

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cuda_fp16.h>

typedef unsigned uint32_t;

__global__ void convertFloatToHalfArray(
        __half* __restrict__ halfValues, const float* __restrict__ floatValues, uint32_t arraySize) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < arraySize) {
        halfValues[globalThreadIdx] = floatValues[globalThreadIdx];
    }
}

#define USE_XORSHIFT
#define ROUNDS_TEA 16

/**
 * TEA algorithm as described in the following work:
 * "GPU Random Numbers via the Tiny Encryption Algorithm". Fahad Zafar, Marc Olano, Aaron Curtis. 2010. HPG '10.
 * @param v0 The seed value.
 * @param v1 The sequence number.
 * @return
 */
inline __device__ uint32_t encryptTea(uint32_t v0, uint32_t v1) {
    uint32_t sum = 0;
    #pragma unroll
    for (uint32_t n = 0; n < ROUNDS_TEA; n++) {
        sum += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + sum) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + sum) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

__global__ void generateRandomPermutations(uint32_t* permutationIndicesBuffer, uint32_t es) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t* permutationIndices = permutationIndicesBuffer + globalThreadIdx * es;

#ifdef USE_XORSHIFT
    // Use Xorshift random numbers with period 2^96-1.
    uint32_t seed = 17u * globalThreadIdx + 240167u;
    uint3 rngState;
    rngState.x = 123456789ul ^ seed;
    rngState.y = 362436069ul ^ seed;
    rngState.z = 521288629ul ^ seed;
#endif

    for (uint32_t i = 0; i < es; i++) {
        permutationIndices[i] = i;
    }

    uint32_t tmp;
    for (uint32_t i = es - 1; i > 0; i--) {
#ifdef USE_XORSHIFT
        rngState.x ^= rngState.x << 16;
        rngState.x ^= rngState.x >> 5;
        rngState.x ^= rngState.x << 1;

        uint32_t t = rngState.x;
        rngState.x = rngState.y;
        rngState.y = rngState.z;
        rngState.z = t ^ rngState.x ^ rngState.y;

        uint32_t j = rngState.z % (i + 1);
#else
        uint32_t j = encryptTea(globalThreadIdx, i) % (i + 1);
#endif

        tmp = permutationIndices[i];
        permutationIndices[i] = permutationIndices[j];
        permutationIndices[j] = tmp;
    }
}

template<class T> __global__ void randomShuffleFisherYates(
        T* __restrict__ outputArray, const T* __restrict__ inputArray,
        const uint32_t* __restrict__ permutationIndicesBuffer, uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t offset = globalThreadIdx * es * numChannels;
    const uint32_t* permutationIndices = permutationIndicesBuffer + globalThreadIdx * es;

    for (uint32_t memberIdx = 0; memberIdx < es; memberIdx++) {
        uint32_t shuffledIdx = permutationIndices[memberIdx];
        uint32_t offsetMemberRead = offset + shuffledIdx * numChannels;
        uint32_t offsetMemberWrite = offset + memberIdx * numChannels;
        for (uint32_t channelIdx = 0; channelIdx < numChannels; channelIdx++) {
            outputArray[offsetMemberWrite + channelIdx] = inputArray[offsetMemberRead + channelIdx];
        }
    }
}

template<class T> __global__ void symmetrizerBE(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx * numChannels;
    uint32_t readOffsetRef = (globalThreadIdx % es) * numChannels;

    for (uint32_t c = 0; c < numChannels; c++) {
        outputValues[readOffset + c] = referenceValues[readOffsetRef + c] + queryValues[readOffset + c];
    }
}

template<class T> __global__ void symmetrizer(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[readOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerPermuted(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        const uint32_t* __restrict__ permutationIndicesBuffer, uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t ensembleIdx = (globalThreadIdx / numChannels) % es;
    uint32_t batchIdx = globalThreadIdx / (numChannels * es);
    uint32_t ensembleMemberPermuted = permutationIndicesBuffer[ensembleIdx];
    uint32_t writeOffset = globalThreadIdx;
    uint32_t readOffset = channelIdx + (ensembleMemberPermuted + batchIdx * es) * numChannels;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[writeOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

//#define USE_FAST_CUDA_MATH

template<class T> __global__ void combineDecoderOutput(
        const T* __restrict__ referenceDecoded, const T* __restrict__ queryDecoded,
        float* __restrict__ miValues, uint32_t numChannels, uint32_t paddingFactor) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx * numChannels;

    float meanReference = 0.0f;
    for (uint32_t c = 0; c < numChannels; c++) {
        meanReference += float(referenceDecoded[(readOffset + c) * paddingFactor]);
    }
    meanReference /= float(numChannels);

    float queryMax = -FLT_MAX;
    for (uint32_t c = 0; c < numChannels; c++) {
        queryMax = max(queryMax, float(queryDecoded[(readOffset + c) * paddingFactor]));
    }

    float queryExpSum = 0.0f;
    for (uint32_t c = 0; c < numChannels; c++) {
#ifdef USE_FAST_CUDA_MATH
        queryExpSum += __expf(float(queryDecoded[(readOffset + c) * paddingFactor]) - queryMax);
#else
        queryExpSum += expf(float(queryDecoded[(readOffset + c) * paddingFactor]) - queryMax);
#endif
    }
    float meanQuery = -logf(float(numChannels)) + logf(queryExpSum) + queryMax;

    //miValues[globalThreadIdx] = float(referenceDecoded[(readOffset + 0) * paddingFactor]);
    miValues[globalThreadIdx] = fmaxf(meanReference - meanQuery, 0.0f);
}
