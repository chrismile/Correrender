/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
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

#ifndef CORRERENDER_PREFIXSUMSCAN_CUH
#define CORRERENDER_PREFIXSUMSCAN_CUH

__global__ void writeFinalElementKernel(int N, const int* __restrict__ dataIn, int* __restrict__ dataOut) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx == 0) {
        dataOut[N] = dataOut[N - 1] + dataIn[N - 1];
    }
}



/**
 * Adds the blockDim.x elements of sumIn to the block elements in dataOut.
 * @param N The number of elements in dataOut.
 * @param dataOut The data array of size N.
 * @param sumIn Array of block increments to be added to the elements in dataOut.
 */
__global__ void prefixSumBlockIncrementKernel(int N, int* __restrict__ dataOut, const int* __restrict__ sumIn) {
    __shared__ int a;
    if (threadIdx.x == 0) {
        a = sumIn[blockIdx.x];
    }
    __syncthreads();

    uint32_t globalOffset = blockDim.x * 2 * blockIdx.x;
    uint32_t idx0 = globalOffset + threadIdx.x * 2;
    uint32_t idx1 = globalOffset + threadIdx.x * 2 + 1;
    if (idx0 < N) {
        dataOut[idx0] += a;
    }
    if (idx1 < N) {
        dataOut[idx1] += a;
    }
}



#define AVOID_MEMORY_BANK_CONFLICTS
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

/**
 * Based on the work-efficient parallel sum scan algorithm [Blelloch 1990] implementation from:
 * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
 * @param N The number of values in the array.
 * @param dataIn An input array with N entries.
 * @param dataOut A (possibly uninitialized, but allocated) output array with N entries.
 * @param sumOut Array of block increments to be added to the elements in dataOut.
 */
#ifdef AVOID_MEMORY_BANK_CONFLICTS
__global__ void prefixSumScanKernel(
        int N, const int* __restrict__ dataIn, int* __restrict__ dataOut, int* __restrict__ sumOut) {
    extern __shared__ int sdata[];

    uint32_t blockDimTimes2 = blockDim.x * 2;
    uint32_t globalOffset = blockDimTimes2 * blockIdx.x;
    uint32_t localIdx = threadIdx.x;
    uint32_t localIdxTimes2 = localIdx * 2;

    // Copy the data to the input array.
    uint32_t idxA = localIdx;
    uint32_t idxB = localIdx + blockDim.x;
    uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(idxA);
    uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(idxB);
    if (globalOffset + idxA < N) {
        sdata[idxA + bankOffsetA] = dataIn[globalOffset + idxA];
    } else {
        sdata[idxA + bankOffsetA] = 0;
    }
    if (globalOffset + idxB < N) {
        sdata[idxB + bankOffsetB] = dataIn[globalOffset + idxB];
    } else {
        sdata[idxB + bankOffsetB] = 0;
    }

    // Up-sweep (reduce) phase.
    uint32_t offset = 1;
    for (uint32_t i = blockDim.x; i > 0; i >>= 1) {
        __syncthreads();

        if (localIdx < i) {
            uint32_t idxA = offset * (localIdxTimes2 + 1) - 1;
            uint32_t idxB = offset * (localIdxTimes2 + 2) - 1;
            idxA += CONFLICT_FREE_OFFSET(idxA);
            idxB += CONFLICT_FREE_OFFSET(idxB);
            sdata[idxB] += sdata[idxA];
        }
        offset <<= 1;
    }

    if (localIdx == 0) {
        uint32_t lastElementIdx = blockDimTimes2 - 1;
        lastElementIdx += CONFLICT_FREE_OFFSET(lastElementIdx);
        sumOut[blockIdx.x] = sdata[lastElementIdx];
        sdata[lastElementIdx] = 0;
    }

    // Down-sweep phase.
    for (uint32_t i = 1; i < blockDimTimes2; i <<= 1) {
        offset >>= 1;
        __syncthreads();

        if (localIdx < i) {
            uint32_t idxA = offset * (localIdxTimes2 + 1) - 1;
            uint32_t idxB = offset * (localIdxTimes2 + 2) - 1;
            idxA += CONFLICT_FREE_OFFSET(idxA);
            idxB += CONFLICT_FREE_OFFSET(idxB);
            uint32_t temp = sdata[idxA];
            sdata[idxA] = sdata[idxB];
            sdata[idxB] += temp;
        }
    }
    __syncthreads();

    // Copy partial sum scan to the output array.
    if (globalOffset + idxA < N) {
        dataOut[globalOffset + idxA] = sdata[idxA + bankOffsetA];
    }
    if (globalOffset + idxB < N) {
        dataOut[globalOffset + idxB] = sdata[idxB + bankOffsetB];
    }
}
#else
// No handling of shared memory bank conflicts:
__global__ void prefixSumScanKernel(
        int N, const int* __restrict__ dataIn, int* __restrict__ dataOut, int* __restrict__ sumOut) {
    extern __shared__ int sdata[];

    uint32_t blockDimTimes2 = blockDim.x * 2;
    uint32_t globalOffset = blockDimTimes2 * blockIdx.x;
    uint32_t localIdx = threadIdx.x;
    uint32_t localIdxTimes2 = localIdx * 2;

    // Copy the data to the input array.
    if (globalOffset + localIdxTimes2 < N) {
        sdata[localIdxTimes2] = dataIn[globalOffset + localIdxTimes2];
    } else {
        sdata[localIdxTimes2] = 0;
    }
    if (globalOffset + localIdxTimes2 + 1 < N) {
        sdata[localIdxTimes2 + 1] = dataIn[globalOffset + localIdxTimes2 + 1];
    } else {
        sdata[localIdxTimes2 + 1] = 0;
    }

    // Up-sweep (reduce) phase.
    uint32_t offset = 1;
    for (uint32_t i = blockDim.x; i > 0; i >>= 1) {
        __syncthreads();

        if (localIdx < i) {
            uint32_t data0 = offset * (localIdxTimes2 + 1) - 1;
            uint32_t data1 = offset * (localIdxTimes2 + 2) - 1;
            sdata[data1] += sdata[data0];
        }
        offset <<= 1;
    }

    if (localIdx == 0) {
        sumOut[blockIdx.x] = sdata[blockDimTimes2 - 1];
        sdata[blockDimTimes2 - 1] = 0;
    }

    // Down-sweep phase.
    for (uint32_t i = 1; i < blockDimTimes2; i <<= 1) {
        offset >>= 1;
        __syncthreads();

        if (localIdx < i) {
            uint32_t ai = offset * (localIdxTimes2 + 1) - 1;
            uint32_t bi = offset * (localIdxTimes2 + 2) - 1;
            uint32_t temp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += temp;
        }
    }
    __syncthreads();

    // Copy partial sum scan to the output array.
    if (globalOffset + localIdxTimes2 < N) {
        dataOut[globalOffset + localIdxTimes2] = sdata[localIdxTimes2];
    }
    if (globalOffset + localIdxTimes2 + 1 < N) {
        dataOut[globalOffset + localIdxTimes2 + 1] = sdata[localIdxTimes2 + 1];
    }
}
#endif



void allocateParallelPrefixSumScanBufferCache(int N, std::vector<int*>& bufferCache) {
    const int BLOCK_SIZE = 256;
    while (N > 0) {
        const int numBlocks = sgl::iceil(int(N), 2 * BLOCK_SIZE);
        int* buffer0 = nullptr, *buffer1 = nullptr;
        cudaErrorCheck(cudaMalloc((void**)&buffer0, sizeof(int) * numBlocks));
        cudaErrorCheck(cudaMalloc((void**)&buffer1, sizeof(int) * numBlocks));
        bufferCache.push_back(buffer0);
        bufferCache.push_back(buffer1);
        N = numBlocks;
        if (numBlocks == 1) {
            break;
        }
    }
}

void freeParallelPrefixSumScanBufferCache(std::vector<int*>& bufferCache) {
    for (int* buffer : bufferCache) {
        cudaErrorCheck(cudaFree(buffer));
    }
    bufferCache.clear();
}

void parallelPrefixSumScanRecursive(
        cudaStream_t stream, int N, const int* bufferIn, int* bufferOut, std::vector<int*>& bufferCache,
        int& bufferCacheIdx) {
    if (N <= 0) {
        return;
    }

    const int BLOCK_SIZE = 256;
    const int numBlocks = sgl::iceil(int(N), 2 * BLOCK_SIZE);

    auto* sumIn = bufferCache.at(bufferCacheIdx);
    auto* sumOut = bufferCache.at(bufferCacheIdx + 1);
    bufferCacheIdx += 2;

    int sharedMemorySize = (2 * BLOCK_SIZE + 2 * BLOCK_SIZE / NUM_BANKS) * sizeof(int);
    prefixSumScanKernel<<<numBlocks, BLOCK_SIZE, sharedMemorySize, stream>>>(N, bufferIn, bufferOut, sumIn);

    // Array of block increments that is added to the elements in the blocks.
    if (numBlocks > 1) {
        parallelPrefixSumScanRecursive(stream, numBlocks, sumIn, sumOut, bufferCache, bufferCacheIdx);
        prefixSumBlockIncrementKernel<<<numBlocks, BLOCK_SIZE, 0, stream>>>(N, bufferOut, sumOut);
    }
}

void parallelPrefixSumScan(
        cudaStream_t stream, int N, const int* bufferIn, int* bufferOut, std::vector<int*>& bufferCache) {
    if (N <= 0) {
        return;
    }

    int bufferCacheIdx = 0;
    parallelPrefixSumScanRecursive(stream, N, bufferIn, bufferOut, bufferCache, bufferCacheIdx);
    writeFinalElementKernel<<<1, 1, 0, stream>>>(N, bufferIn, bufferOut);
}

#endif //CORRERENDER_PREFIXSUMSCAN_CUH
