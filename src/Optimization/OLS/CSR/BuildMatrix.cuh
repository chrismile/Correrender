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

#ifndef CORRERENDER_BUILDMATRIX_CUH
#define CORRERENDER_BUILDMATRIX_CUH

#include "../PrecisionDefines.hpp"
#include "../CudaDefines.cuh"

// https://developer.nvidia.com/blog/lerp-faster-cuda/
__host__ __device__ inline float4 mix(const float4& v0, const float4&  v1, float t) {
    return make_float4(
            fmaf(t, v1.x, fmaf(-t, v0.x, v0.x)),
            fmaf(t, v1.y, fmaf(-t, v0.y, v0.y)),
            fmaf(t, v1.z, fmaf(-t, v0.z, v0.z)),
            fmaf(t, v1.w, fmaf(-t, v0.w, v0.w))
    );
}

__global__ void computeNnzKernel(
        int xs, int ys, int zs, float Nj, float minGT, float maxGT, float minOpt, float maxOpt,
        cudaTextureObject_t scalarFieldGT, cudaTextureObject_t scalarFieldOpt,
        int* __restrict__ rowsHasNonZero, int* __restrict__ rowsNumNonZero) {
    uint32_t globalThreadIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t globalThreadIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t globalThreadIdxZ = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t globalThreadIdx = globalThreadIdxX + (globalThreadIdxY + globalThreadIdxZ * ys) * xs;
    if (globalThreadIdxX >= xs || globalThreadIdxY >= ys || globalThreadIdxZ >= zs) {
        return;
    }
    auto scalarGT = tex3D<float>(
            scalarFieldGT, float(globalThreadIdxX) + 0.5f,
            float(globalThreadIdxY) + 0.5f, float(globalThreadIdxZ) + 0.5f);
    auto scalarOpt = tex3D<float>(
            scalarFieldOpt, float(globalThreadIdxX) + 0.5f,
            float(globalThreadIdxY) + 0.5f, float(globalThreadIdxZ) + 0.5f);
    int hasNonZero = 0, numNonZero = 0;
    if (!isnan(scalarGT) && !isnan(scalarOpt)) {
        float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
        float tOpt0 = clamp(floor(tOpt * Nj), 0.0f, Nj);
        float tOpt1 = clamp(ceil(tOpt * Nj), 0.0f, Nj);
        uint32_t jOpt0 = uint32_t(tOpt0);
        uint32_t jOpt1 = uint32_t(tOpt1);
        hasNonZero = 1;
        if (jOpt0 == jOpt1) {
            numNonZero = 1;
        } else {
            numNonZero = 2;
        }
    }
    rowsHasNonZero[globalThreadIdx] = hasNonZero;
    rowsNumNonZero[globalThreadIdx] = numNonZero;
}

template<class Real>
__global__ void writeCsrKernel(
        int xs, int ys, int zs, float Nj, float minGT, float maxGT, float minOpt, float maxOpt,
        const float4* __restrict__ tfGT, cudaTextureObject_t scalarFieldGT, cudaTextureObject_t scalarFieldOpt,
        const int* __restrict__ rowsHasNonZero, const int* __restrict__ hasNonZeroPrefixSum,
        const int* __restrict__ rowsNumNonZero, const int* __restrict__ numNonZeroPrefixSum,
        int nnz, Real* __restrict__ csrVals, int* __restrict__ csrRowPtr, int* __restrict__ csrColInd,
        typename MakeVec<Real, 4>::type* __restrict__ b) {
    uint32_t globalThreadIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t globalThreadIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t globalThreadIdxZ = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t globalThreadIdx = globalThreadIdxX + (globalThreadIdxY + globalThreadIdxZ * ys) * xs;
    if (globalThreadIdxX >= xs || globalThreadIdxY >= ys || globalThreadIdxZ >= zs) {
        return;
    }
    uint32_t numNonZero = rowsNumNonZero[globalThreadIdx];
    if (numNonZero == 0) {
        return;
    }
    int writePosRow = hasNonZeroPrefixSum[globalThreadIdx];
    int writePosMat = numNonZeroPrefixSum[globalThreadIdx];

    auto scalarGT = tex3D<float>(
            scalarFieldGT, float(globalThreadIdxX) + 0.5f,
            float(globalThreadIdxY) + 0.5f, float(globalThreadIdxZ) + 0.5f);
    auto scalarOpt = tex3D<float>(
            scalarFieldOpt, float(globalThreadIdxX) + 0.5f,
            float(globalThreadIdxY) + 0.5f, float(globalThreadIdxZ) + 0.5f);

    float tGT = (scalarGT - minGT) / (maxGT - minGT);
    float tGT0 = clamp(floor(tGT * Nj), 0.0f, Nj);
    float tGT1 = clamp(ceil(tGT * Nj), 0.0f, Nj);
    float fGT = tGT * Nj - tGT0;
    int jGT0 = int(tGT0);
    int jGT1 = int(tGT1);
    float4 cGT0 = tfGT[jGT0];
    float4 cGT1 = tfGT[jGT1];
    float4 colorGT = mix(cGT0, cGT1, fGT);
    if constexpr (std::is_same<Real, float>()) {
        b[writePosRow] = colorGT;
    } else {
        b[writePosRow] = make_double4(double(colorGT.x), double(colorGT.y), double(colorGT.z), double(colorGT.w));
    }

    float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
    float tOpt0 = clamp(floor(tOpt * Nj), 0.0f, Nj);
    float tOpt1 = clamp(ceil(tOpt * Nj), 0.0f, Nj);
    auto fOpt = Real(tOpt * Nj - tOpt0);
    uint32_t jOpt0 = uint32_t(tOpt0);
    uint32_t jOpt1 = uint32_t(tOpt1);

    auto fOpt0 = Real(1) - Real(fOpt);
    int j0 = int(jOpt0) * 4;
    int j1 = int(jOpt1) * 4;
    int ir = writePosRow * 4;
    int im = writePosMat * 4;
    for (int c = 0; c < 4; c++) {
        csrRowPtr[ir] = im;
        csrColInd[im] = int(j0);
        csrVals[im] = fOpt0;
        im++;
        if (jOpt0 != jOpt1) {
            csrColInd[im] = int(j1);
            csrVals[im] = fOpt;
            im++;
        }
        j0++;
        j1++;
        ir++;
    }
}

__global__ void writeFinalCsrRowPtr(int nnz, int numRows, int* __restrict__ csrRowPtr) {
    uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx == 0) {
        csrRowPtr[numRows] = nnz;
    }
}

#endif //CORRERENDER_BUILDMATRIX_CUH
