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

#ifndef CORRERENDER_CUDASUBROUTINES_CUH
#define CORRERENDER_CUDASUBROUTINES_CUH

#include <cublas_v2.h>
#include <cusparse.h>

#include "CudaHelpers.cuh"

// axpy computes y[i] = alpha * x[i] + y[i].
inline void axpy(
        cublasHandle_t handle, int n, double alpha, const double* x, double* y) {
    cudaErrorCheck(cublasDaxpy(handle, n, &alpha, x, 1, y, 1));
}

inline void axpy(
        cublasHandle_t handle, int n, float alpha, const float* x, float* y) {
    cudaErrorCheck(cublasSaxpy(handle, n, &alpha, x, 1, y, 1));
}

// scal computes x[i] = alpha * x[i].
inline void scal(
        cublasHandle_t handle, int n, double alpha, double* x) {
    cudaErrorCheck(cublasDscal(handle, n, &alpha, x, 1));
}

inline void scal(
        cublasHandle_t handle, int n, float alpha, float* x) {
    cudaErrorCheck(cublasSscal(handle, n, &alpha, x, 1));
}

// copy computes y[i] = x[i].
inline void copy(
        cublasHandle_t handle, int n, const double* x, double* y) {
    cudaErrorCheck(cublasDcopy(handle, n, x, 1, y, 1));
}

inline void copy(
        cublasHandle_t handle, int n, const float* x, float* y) {
    cudaErrorCheck(cublasScopy(handle, n, x, 1, y, 1));
}

// Computes the two-norm of the vector x.
inline void nrm2(cublasHandle_t handle, int n, const double* x, double* result) {
    cudaErrorCheck(cublasDnrm2(handle, n, x, 1, result));
}

inline void nrm2(cublasHandle_t handle, int n, const float* x, float* result) {
    cudaErrorCheck(cublasSnrm2(handle, n, x, 1, result));
}

inline void nrm2(cublasHandle_t handle, int n, const float* x, double* result) {
    float resultFloat;
    cudaErrorCheck(cublasSnrm2(handle, n, x, 1, &resultFloat));
    *result = static_cast<double>(resultFloat);
}

template<typename Real>
inline Real nrm2(cudaStream_t stream, cublasHandle_t handle, int n, const Real* x) {
    Real result;
    nrm2(handle, n, x, &result);
    cudaErrorCheck(cudaStreamSynchronize(stream));
    return result;
}

inline void nrm2sqr(cublasHandle_t handle, int n, const double* x, double* result) {
    cudaErrorCheck(cublasDdot(handle, n, x, 1, x, 1, result));
    *result = std::sqrt(*result);
}

inline void nrm2sqr(cublasHandle_t handle, int n, const float* x, float* result) {
    cudaErrorCheck(cublasSdot(handle, n, x, 1, x, 1, result));
    *result = std::sqrt(*result);
}

inline void nrm2sqr(cublasHandle_t handle, int n, const float* x, double* result) {
    float resultFloat;
    cudaErrorCheck(cublasSdot(handle, n, x, 1, x, 1, &resultFloat));
    *result = std::sqrt(static_cast<double>(resultFloat));
}

template<typename Real>
inline Real nrm2sqr(cudaStream_t stream, cublasHandle_t handle, int n, const Real* x) {
    Real result;
    nrm2sqr(handle, n, x, &result);
    cudaErrorCheck(cudaStreamSynchronize(stream));
    return result;
}

// Dense vector wrapper for cuSPARSE.
class DnVector {
private:
    cusparseDnVecDescr_t vecDescr = {};
    void* devicePointer = nullptr;

public:
    DnVector(void* values, size_t size, cudaDataType valueType) {
        devicePointer = values;
        cudaErrorCheck(cusparseCreateDnVec(&vecDescr, int64_t(size), devicePointer, valueType));
    }
    DnVector(double* values, size_t size) : DnVector((void*)values, size, CUDA_R_64F) {}
    DnVector(float* values, size_t size) : DnVector((void*)values, size, CUDA_R_32F) {}

    ~DnVector() {
        cudaErrorCheck(cusparseDestroyDnVec(vecDescr));
    }

    inline cusparseDnVecDescr_t getCusparseDnVecDescr() { return vecDescr; }
};

// Auxiliary buffer for cuSPARSE routines.
class AuxBuffer {
private:
    size_t sizeInBytes = 0;
    void* dataPtr = nullptr;

public:
    ~AuxBuffer() {
        if (dataPtr) {
            cudaErrorCheck(cudaFree(dataPtr));
        }
    }

    inline void reserve(size_t _sizeInBytes) {
        if (_sizeInBytes > sizeInBytes) {
            if (dataPtr) {
                cudaErrorCheck(cudaFree(dataPtr));
            }
            sizeInBytes = _sizeInBytes;
            cudaErrorCheck(cudaMalloc(&dataPtr, sizeInBytes));
        }
    }
    inline void* getPointer() { return dataPtr; }
};

// Sparse matrix-vector multiplication.
template<typename Real>
class SpMV {
private:
    cudaStream_t stream{};
    cusparseHandle_t handle{};
    cusparseSpMatDescr_t matDescr{};
    int m, n, nnz;
    cusparseSpMVAlg_t spMVAlg = CUSPARSE_SPMV_ALG_DEFAULT;

public:
    SpMV(
            cudaStream_t _stream, cusparseHandle_t _cusparseHandle,
            int _m, int _n, int _nnz, const Real* csrValues, const int* csrRowOffsets, const int* csrColInd)
            : stream(_stream), handle(_cusparseHandle), m(_m), n(_n), nnz(_nnz) {
        cudaDataType dataType;
        if constexpr(std::is_same<Real, double>::value) {
            dataType = CUDA_R_64F;
        } else if constexpr(std::is_same<Real, float>::value) {
            dataType = CUDA_R_32F;
        }
        cudaErrorCheck(cusparseCreateCsr(
                &matDescr, m, n, nnz,
                (void*)csrRowOffsets, (void*)csrColInd, (void*)csrValues,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dataType));
    }

    ~SpMV() {
        cudaErrorCheck(cusparseDestroySpMat(matDescr));
    }

    // Computes y = alpha * op(A) * x + beta * y.
    void operator()(
            cusparseOperation_t op, const Real alpha, DnVector& x, const Real beta, DnVector& y, AuxBuffer& auxBuffer);
    void eval(
            cudaDataType dataType, cusparseOperation_t op,
            const void* alpha, DnVector& x, const void* beta, DnVector& y, AuxBuffer& auxBuffer) {
        size_t auxBufferSize = 0;
        cudaErrorCheck(cusparseSpMV_bufferSize(
                handle, op, alpha, matDescr,
                x.getCusparseDnVecDescr(), beta, y.getCusparseDnVecDescr(),
                dataType, spMVAlg, &auxBufferSize));
        cudaErrorCheck(cudaStreamSynchronize(stream));
        auxBuffer.reserve(auxBufferSize);

        cudaErrorCheck(cusparseSpMV(
                handle, op, alpha, matDescr,
                x.getCusparseDnVecDescr(), beta, y.getCusparseDnVecDescr(),
                dataType, spMVAlg, auxBuffer.getPointer()));
    }
};

template<>
inline void SpMV<double>::operator()(
        cusparseOperation_t op, const double alpha, DnVector& x, const double beta, DnVector& y, AuxBuffer& auxBuffer) {
    eval(CUDA_R_64F, op, &alpha, x, &beta, y, auxBuffer);
}

template<>
inline void SpMV<float>::operator()(
        cusparseOperation_t op, const float alpha, DnVector& x, const float beta, DnVector& y, AuxBuffer& auxBuffer) {
    eval(CUDA_R_32F, op, &alpha, x, &beta, y, auxBuffer);
}

#endif //CORRERENDER_CUDASUBROUTINES_CUH
