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

#include <Math/Math.hpp>

#include "SymmetrizerType.hpp"

__global__ void convertFloatToHalfArray(
        __half* __restrict__ halfValues, const float* __restrict__ floatValues, uint32_t arraySize);

__global__ void generateRandomPermutations(uint32_t* permutationIndicesBuffer, uint32_t es, uint32_t batchOffset);

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

template<class T> __global__ void symmetrizerAdd(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[readOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerAddPermuted(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        const uint32_t* __restrict__ permutationIndicesBuffer, uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t ensembleIdx = (globalThreadIdx / numChannels) % es;
    uint32_t batchIdx = globalThreadIdx / (numChannels * es);
    uint32_t ensembleMemberPermuted = permutationIndicesBuffer[ensembleIdx + batchIdx * es];
    uint32_t writeOffset = globalThreadIdx;
    uint32_t readOffset = channelIdx + (ensembleMemberPermuted + batchIdx * es) * numChannels;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[writeOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

__forceinline__ __device__ float absT(float val) { return fabsf(val); }
__forceinline__ __device__ __half absT(__half val) { return __habs(val); }

template<class T> __global__ void symmetrizerAddDiff_Add(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + channelIdx;
    outputValues[writeOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerAddDiff_Diff(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + numChannels + channelIdx;
    outputValues[writeOffset] = absT(referenceValues[readOffsetRef] - queryValues[readOffset]);
}

template<class T> __global__ void symmetrizerAddDiffPermuted_Add(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        const uint32_t* __restrict__ permutationIndicesBuffer, uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t ensembleIdx = (globalThreadIdx / numChannels) % es;
    uint32_t batchIdx = globalThreadIdx / (numChannels * es);
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t ensembleMemberPermuted = permutationIndicesBuffer[ensembleIdx + batchIdx * es];
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + channelIdx;
    uint32_t readOffset = channelIdx + (ensembleMemberPermuted + batchIdx * es) * numChannels;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[writeOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerAddDiffPermuted_Diff(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        const uint32_t* __restrict__ permutationIndicesBuffer, uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t ensembleIdx = (globalThreadIdx / numChannels) % es;
    uint32_t batchIdx = globalThreadIdx / (numChannels * es);
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t ensembleMemberPermuted = permutationIndicesBuffer[ensembleIdx + batchIdx * es];
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + numChannels + channelIdx;
    uint32_t readOffset = channelIdx + (ensembleMemberPermuted + batchIdx * es) * numChannels;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[writeOffset] = absT(referenceValues[readOffsetRef] - queryValues[readOffset]);
}

template<class T> __global__ void symmetrizerMul(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[readOffset] = referenceValues[readOffsetRef] * queryValues[readOffset];
}

template<class T> __global__ void symmetrizerMulPermuted(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        const uint32_t* __restrict__ permutationIndicesBuffer, uint32_t es, uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t ensembleIdx = (globalThreadIdx / numChannels) % es;
    uint32_t batchIdx = globalThreadIdx / (numChannels * es);
    uint32_t ensembleMemberPermuted = permutationIndicesBuffer[ensembleIdx + batchIdx * es];
    uint32_t writeOffset = globalThreadIdx;
    uint32_t readOffset = channelIdx + (ensembleMemberPermuted + batchIdx * es) * numChannels;
    uint32_t readOffsetRef = globalThreadIdx % (es * numChannels);
    outputValues[writeOffset] = referenceValues[readOffsetRef] * queryValues[readOffset];
}

template<class T> void symmetrizer(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues,
        T* __restrict__ symmetrizedReferenceValues, T* __restrict__ symmetrizedQueryValues,
        const uint32_t* __restrict__ permutationIndicesBuffer,
        uint32_t batchSize, uint32_t es, uint32_t numLayersOutEncoder,
        SymmetrizerType symmetrizerType, CUstream stream) {
    constexpr uint32_t blockSize = 256;
    const uint32_t numBlocks = sgl::uiceil(batchSize * uint32_t(es) * numLayersOutEncoder, blockSize);
    if (symmetrizerType == SymmetrizerType::Add) {
        symmetrizerAdd<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedReferenceValues, uint32_t(es), numLayersOutEncoder);
        symmetrizerAddPermuted<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedQueryValues, permutationIndicesBuffer,
                uint32_t(es), numLayersOutEncoder);
    } else if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerAddDiff_Add<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedReferenceValues, uint32_t(es), numLayersOutEncoder);
        symmetrizerAddDiff_Diff<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedReferenceValues, uint32_t(es), numLayersOutEncoder);
        symmetrizerAddDiffPermuted_Add<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedQueryValues, permutationIndicesBuffer,
                uint32_t(es), numLayersOutEncoder);
        symmetrizerAddDiffPermuted_Diff<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedQueryValues, permutationIndicesBuffer,
                uint32_t(es), numLayersOutEncoder);
    } else if (symmetrizerType == SymmetrizerType::Mul) {
        symmetrizerMul<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedReferenceValues, uint32_t(es), numLayersOutEncoder);
        symmetrizerMulPermuted<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedQueryValues, permutationIndicesBuffer,
                uint32_t(es), numLayersOutEncoder);
    }
}


template<class T> __global__ void symmetrizerSrnAdd(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef = globalThreadIdx % numChannels;
    outputValues[readOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerSrnAddDiff_Add(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t readOffsetRef = channelIdx;
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + channelIdx;
    outputValues[writeOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}

template<class T> __global__ void symmetrizerSrnAddDiff_Diff(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t channelIdx = globalThreadIdx % numChannels;
    uint32_t readOffsetRef = channelIdx;
    uint32_t batchEnsembleIdx = globalThreadIdx / numChannels;
    uint32_t writeOffset = batchEnsembleIdx * numChannels * 2 + numChannels + channelIdx;
    outputValues[writeOffset] = absT(referenceValues[readOffsetRef] - queryValues[readOffset]);
}

template<class T> __global__ void symmetrizerSrnMul(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ outputValues,
        uint32_t numChannels) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t readOffset = globalThreadIdx;
    uint32_t readOffsetRef = globalThreadIdx % numChannels;
    outputValues[readOffset] = referenceValues[readOffsetRef] * queryValues[readOffset];
}

template<class T> void symmetrizerSrn(
        const T* __restrict__ referenceValues, const T* __restrict__ queryValues, T* __restrict__ symmetrizedValues,
        uint32_t batchSize, uint32_t numLayersOutEncoder, SymmetrizerType symmetrizerType, CUstream stream) {
    constexpr uint32_t blockSize = 256;
    const uint32_t numBlocks = sgl::uiceil(batchSize * numLayersOutEncoder, blockSize);
    if (symmetrizerType == SymmetrizerType::Add) {
        symmetrizerSrnAdd<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, numLayersOutEncoder);
    } else if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerSrnAddDiff_Add<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, numLayersOutEncoder);
        symmetrizerSrnAddDiff_Diff<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, numLayersOutEncoder);
    } else if (symmetrizerType == SymmetrizerType::Mul) {
        symmetrizerSrnMul<<<numBlocks, blockSize, 0, stream>>>(
                referenceValues, queryValues, symmetrizedValues, numLayersOutEncoder);
    }
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

    miValues[globalThreadIdx] = fmaxf(meanReference - meanQuery, 0.0f);
}

template<class T> __global__ void copyDecoderOutputSrnMutualInformation(
        const T* __restrict__ decodedValues, float* __restrict__ miValues, uint32_t paddingFactor) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    miValues[globalThreadIdx] = fmaxf(decodedValues[globalThreadIdx * paddingFactor], 0.0f);
}

template<class T> __global__ void copyDecoderOutputSrnCorrelationCoefficient(
        const T* __restrict__ decodedValues, float* __restrict__ miValues, uint32_t paddingFactor) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    miValues[globalThreadIdx] = decodedValues[globalThreadIdx * paddingFactor];
}

template<class T> __global__ void copyDecoderOutputSrnCorrelationCoefficientAbs(
        const T* __restrict__ decodedValues, float* __restrict__ miValues, uint32_t paddingFactor) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    miValues[globalThreadIdx] = fabsf(decodedValues[globalThreadIdx * paddingFactor]);
}

#endif //CORRERENDER_MUTUALINFORMATION_CUH
