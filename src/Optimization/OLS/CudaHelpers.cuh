/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020-2021, Christoph Neuhauser
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

#ifndef SH_AUG_CUDAHELPERS_CUH
#define SH_AUG_CUDAHELPERS_CUH

#include <iostream>
#include <unordered_map>

const std::unordered_map<cublasStatus_t, std::string> cublasStatusNames = {
        {CUBLAS_STATUS_SUCCESS, "CUBLAS_STATUS_SUCCESS"},
        {CUBLAS_STATUS_NOT_INITIALIZED, "CUBLAS_STATUS_NOT_INITIALIZED"},
        {CUBLAS_STATUS_ALLOC_FAILED, "CUBLAS_STATUS_ALLOC_FAILED"},
        {CUBLAS_STATUS_INVALID_VALUE, "CUBLAS_STATUS_INVALID_VALUE"},
        {CUBLAS_STATUS_ARCH_MISMATCH, "CUBLAS_STATUS_ARCH_MISMATCH"},
        {CUBLAS_STATUS_MAPPING_ERROR, "CUBLAS_STATUS_MAPPING_ERROR"},
        {CUBLAS_STATUS_EXECUTION_FAILED, "CUBLAS_STATUS_EXECUTION_FAILED"},
        {CUBLAS_STATUS_INTERNAL_ERROR, "CUBLAS_STATUS_INTERNAL_ERROR"},
        {CUBLAS_STATUS_NOT_SUPPORTED, "CUBLAS_STATUS_NOT_SUPPORTED"},
        {CUBLAS_STATUS_LICENSE_ERROR, "CUBLAS_STATUS_LICENSE_ERROR"},
};

const std::unordered_map<cusparseStatus_t, std::string> cusparseStatusNames = {
        {CUSPARSE_STATUS_SUCCESS, "CUSPARSE_STATUS_SUCCESS"},
        {CUSPARSE_STATUS_NOT_INITIALIZED, "CUSPARSE_STATUS_NOT_INITIALIZED"},
        {CUSPARSE_STATUS_ALLOC_FAILED, "CUSPARSE_STATUS_ALLOC_FAILED"},
        {CUSPARSE_STATUS_INVALID_VALUE, "CUSPARSE_STATUS_INVALID_VALUE"},
        {CUSPARSE_STATUS_ARCH_MISMATCH, "CUSPARSE_STATUS_ARCH_MISMATCH"},
        {CUSPARSE_STATUS_MAPPING_ERROR, "CUSPARSE_STATUS_MAPPING_ERROR"},
        {CUSPARSE_STATUS_EXECUTION_FAILED, "CUSPARSE_STATUS_EXECUTION_FAILED"},
        {CUSPARSE_STATUS_INTERNAL_ERROR, "CUSPARSE_STATUS_INTERNAL_ERROR"},
        {CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"},
        {CUSPARSE_STATUS_ZERO_PIVOT, "CUSPARSE_STATUS_ZERO_PIVOT"},
        {CUSPARSE_STATUS_NOT_SUPPORTED, "CUSPARSE_STATUS_NOT_SUPPORTED"},
        {CUSPARSE_STATUS_INSUFFICIENT_RESOURCES, "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES"},
};

const std::unordered_map<cusolverStatus_t, std::string> cusolverStatusNames = {
        {CUSOLVER_STATUS_SUCCESS, "CUSOLVER_STATUS_SUCCESS"},
        {CUSOLVER_STATUS_NOT_INITIALIZED, "CUSOLVER_STATUS_NOT_INITIALIZED"},
        {CUSOLVER_STATUS_ALLOC_FAILED, "CUSOLVER_STATUS_ALLOC_FAILED"},
        {CUSOLVER_STATUS_INVALID_VALUE, "CUSOLVER_STATUS_INVALID_VALUE"},
        {CUSOLVER_STATUS_ARCH_MISMATCH, "CUSOLVER_STATUS_ARCH_MISMATCH"},
        {CUSOLVER_STATUS_MAPPING_ERROR, "CUSOLVER_STATUS_MAPPING_ERROR"},
        {CUSOLVER_STATUS_EXECUTION_FAILED, "CUSOLVER_STATUS_EXECUTION_FAILED"},
        {CUSOLVER_STATUS_INTERNAL_ERROR, "CUSOLVER_STATUS_INTERNAL_ERROR"},
        {CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED"},
        {CUSOLVER_STATUS_NOT_SUPPORTED, "CUSOLVER_STATUS_NOT_SUPPORTED"},
        {CUSOLVER_STATUS_ZERO_PIVOT, "CUSOLVER_STATUS_ZERO_PIVOT"},
        {CUSOLVER_STATUS_INVALID_LICENSE, "CUSOLVER_STATUS_INVALID_LICENSE"},
        {CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED, "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED"},
        {CUSOLVER_STATUS_IRS_PARAMS_INVALID, "CUSOLVER_STATUS_IRS_PARAMS_INVALID"},
        {CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC, "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC"},
        {CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE, "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE"},
        {CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER, "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER"},
        {CUSOLVER_STATUS_IRS_INTERNAL_ERROR, "CUSOLVER_STATUS_IRS_INTERNAL_ERROR"},
        {CUSOLVER_STATUS_IRS_NOT_SUPPORTED, "CUSOLVER_STATUS_IRS_NOT_SUPPORTED"},
        {CUSOLVER_STATUS_IRS_OUT_OF_RANGE, "CUSOLVER_STATUS_IRS_OUT_OF_RANGE"},
        {CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES, "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES"},
        {CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED, "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED"},
        {CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED, "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED"},
        {CUSOLVER_STATUS_IRS_MATRIX_SINGULAR, "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR"},
        {CUSOLVER_STATUS_INVALID_WORKSPACE, "CUSOLVER_STATUS_INVALID_WORKSPACE"},
};

// Code from: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define cudaErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t errorCode, const char* file, int line, bool abort = true) {
    if (errorCode != cudaSuccess) {
        std::cerr << "cudaAssert: " << cudaGetErrorString(errorCode)
                << " " << file << " " << line << std::endl;
        if (abort) {
            exit(errorCode);
        }
    }
}
inline void cudaAssert(cublasStatus_t errorCode, const char* file, int line, bool abort = true) {
    if (errorCode != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cudaAssert (cuBLAS): " << cublasStatusNames.at(errorCode)
                  << " " << file << " " << line << std::endl;
        if (abort) {
            exit(errorCode);
        }
    }
}
inline void cudaAssert(cusparseStatus_t errorCode, const char* file, int line, bool abort = true) {
    if (errorCode != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cudaAssert (cuSPARSE): " << cusparseStatusNames.at(errorCode)
                  << " " << file << " " << line << std::endl;
        if (abort) {
            exit(errorCode);
        }
    }
}
inline void cudaAssert(cusolverStatus_t errorCode, const char* file, int line, bool abort = true) {
    if (errorCode != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cudaAssert (cuSOLVER): " << cusolverStatusNames.at(errorCode)
                << " " << file << " " << line << std::endl;
        if (abort) {
            exit(errorCode);
        }
    }
}
// To test error in kernel:
//cudaErrorCheck(cudaPeekAtLastError());
//cudaErrorCheck(cudaDeviceSynchronize());


// Integer ceiling operation, i.e., ceil(x/y)
inline int iceil(int x, int y) {
    return 1 + ((x - 1) / y);
}

// Integer binary logarithm.
inline int ilog2(int x) {
    int log2x = 0;
    while ((x >>= 1) != 0) {
        ++log2x;
    }
    return log2x;
}


template<typename T>
__device__ inline T clamp(T x, T a, T b) {
    return x <= a ? a : (x >= b ? b : x);
}

#endif //SH_AUG_CUDAHELPERS_CUH
