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

#ifndef CORRERENDER_MUTUALINFORMATION_CUH
#define CORRERENDER_MUTUALINFORMATION_CUH

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include <cuda_fp16.h>

template<class T> __global__ void randomShuffleFisherYatesXorshift(T* valueArray, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seed = 17u * globalThreadIdx + 240167u;

    // Use Xorshift random numbers with period 2^96-1.
    uint3 rngState;
    rngState.x = 123456789ul ^ seed;
    rngState.y = 362436069ul ^ seed;
    rngState.z = 521288629ul ^ seed;

    T tmp;
    for (uint32_t i = numChannels - 1; i > 0; i--) {
        rngState.x ^= rngState.x << 16;
        rngState.x ^= rngState.x >> 5;
        rngState.x ^= rngState.x << 1;

        uint32_t t = rngState.x;
        rngState.x = rngState.y;
        rngState.y = rngState.z;
        rngState.z = t ^ rngState.x ^ rngState.y;

        uint32_t j = rngState.z % (i + 1);
        tmp = valueArray[i];
        valueArray[i] = valueArray[j];
        valueArray[j] = tmp;
    }
}

template<class T> __global__ void symmetrizer(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues,
        T* __restrict__ outputValues, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx * numChannels;

    for (uint32_t c = 0; c < numChannels; c++) {
        outputValues[readOffset + c] = referenceValues[c] + queryValues[readOffset + c];
    }
}

//#define USE_FAST_CUDA_MATH

template<class T> __global__ void combineDecoderOutput(
        const T* __restrict__ referenceDecoded, const T* __restrict__ queryDecoded,
        float* __restrict__ miValues, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx * numChannels;

    float meanReference = 0.0f;
    for (uint32_t c = 0; c < numChannels; c++) {
        meanReference += float(referenceDecoded[readOffset + c]);
    }
    meanReference /= float(numChannels);

    float queryMax = -FLT_MAX;
    for (uint32_t c = 0; c < numChannels; c++) {
        queryMax = max(queryMax, float(queryDecoded[readOffset + c]));
    }

    float queryExpSum = 0.0f;
    for (uint32_t c = 0; c < numChannels; c++) {
#ifdef USE_FAST_CUDA_MATH
        queryExpSum += __expf(float(queryDecoded[readOffset + c]) - queryExpSum);
#else
        queryExpSum += expf(float(queryDecoded[readOffset + c]) - queryExpSum);
#endif
    }
    float meanQuery = -logf(float(numChannels)) + logf(queryExpSum) + queryMax;

    miValues[globalThreadIdx] = meanReference - meanQuery;
}

#endif //CORRERENDER_MUTUALINFORMATION_CUH
