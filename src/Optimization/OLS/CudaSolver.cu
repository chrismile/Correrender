/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Christoph Neuhauser
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

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "CudaHelpers.cuh"
#include "CudaSolver.hpp"

static bool useCustomCudaStream = false;
static cudaStream_t stream = nullptr;
static cusolverDnHandle_t cusolverHandle = nullptr;
static cublasHandle_t cublasHandle = nullptr;

void cudaInit(void* cudaStream) {
    // Initialize cuBLAS and cuSOLVER.
    if (cudaStream) {
        stream = cudaStream_t(cudaStream);
        useCustomCudaStream = true;
    } else {
        cudaErrorCheck(cudaStreamCreate(&stream));
    }
    cudaErrorCheck(cusolverDnCreate(&cusolverHandle));
    cudaErrorCheck(cublasCreate(&cublasHandle));
    cudaErrorCheck(cusolverDnSetStream(cusolverHandle, stream));
    cudaErrorCheck(cublasSetStream(cublasHandle, stream));
}

void cudaRelease() {
    // Free cuBLAS and cuSOLVER.
    if (cusolverHandle) {
        cudaErrorCheck(cusolverDnDestroy(cusolverHandle));
        cusolverHandle = nullptr;
    }
    if (cublasHandle) {
        cudaErrorCheck(cublasDestroy(cublasHandle));
        cublasHandle = nullptr;
    }
    if (stream) {
        if (!useCustomCudaStream) {
            cudaErrorCheck(cudaStreamDestroy(stream));
        }
        stream = nullptr;
    }
}

// https://eigen.tuxfamily.org/dox/TopicCUDA.html
// https://docs.nvidia.com/cuda/cusolver/index.html#introduction
void solveSystemOfLinearEquationsCuda(
        CudaSolverType cudaSolverType, bool useRelaxation, const Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    // A \in R^NxL, I \in R^NxD1, l \in R^LxD1, A*l = I.
    const int N = int(A.rows());
    constexpr int L = 9; //A.cols();
    constexpr int D1 = 1; //I.cols();
    assert(A.rows() == b.rows());
    assert(L == A.cols());//assert(L == 9);
    assert(D1 == b.cols());//assert(D1 == 1);
    x = Eigen::Matrix<Real, L, D1>();

    Eigen::MatrixXr M_I = Eigen::Matrix<Real, L, L>::Identity();

    // Allocate memory on the device.
    Real* dA = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&dA, sizeof(Real) * N * L));
    cudaErrorCheck(cudaMemcpy(dA, A.data(), sizeof(Real) * N * L, cudaMemcpyHostToDevice));
    Real* db = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&db, sizeof(Real) * N * D1));
    cudaErrorCheck(cudaMemcpy(db, b.data(), sizeof(Real) * N * D1, cudaMemcpyHostToDevice));
    Real* dx = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&dx, sizeof(Real) * L * D1));

    // lhs = A^T*A + lambda_l*M_I
    Real* dLhs = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&dLhs, sizeof(Real) * L * L));
    cudaMemcpy(dLhs, M_I.data(), sizeof(Real) * L * L, cudaMemcpyHostToDevice);
    // rhs = A^T*b
    Real* dRhs = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&dRhs, sizeof(Real) * L * D1));

    const int lda = N;
    const int ldb = N;
    constexpr int ldLhs = L;
    constexpr int ldRhs = L;

    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
    // cublasSgemm: C = alpha*op(A)op(B) + beta*C
    const Real alpha = Real(1.0);
    // Compute: lhs = A^T*A + lambda_l*M_I.
    const Real beta0 = lambdaL;
    cudaErrorCheck(cublasRgemm(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            L, L, N, &alpha, dA, lda, dA, lda, &beta0, dLhs, ldLhs));
    // Compute: rhs = A^T*b.
    const Real beta1 = Real(0.0);
    cudaErrorCheck(cublasRgemm(
            cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
            L, D1, N, &alpha, dA, lda, db, ldb, &beta1, dRhs, ldRhs));

    int lwork = 0;
    Real* dWork = nullptr;
    int* dInfo = nullptr;
    int hInfo = 0;
    cudaErrorCheck(cudaMalloc ((void**)&dInfo, sizeof(int)));

    // Now, solve lhs*l = rhs.
    switch(cudaSolverType) {
        // See: https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
        case CudaSolverType::LU: case CudaSolverType::LU_PIVOT: {
            bool usePivot = cudaSolverType == CudaSolverType::LU_PIVOT;

            // Query working space required by getrf.
            cudaErrorCheck(cusolverDnRgetrf_bufferSize(
                    cusolverHandle, L, L, dLhs, ldLhs, &lwork));
            cudaErrorCheck(cudaMalloc((void**)&dWork, sizeof(Real)*lwork));

            // Create data for pivot if user requests pivoting.
            int* dbpiv = nullptr;
            if (usePivot) {
                cudaErrorCheck(cudaMalloc((void**)&dbpiv, sizeof(int) * L));
            }

            // LU factorization.
            cudaErrorCheck(cusolverDnRgetrf(
                    cusolverHandle, L, L, dLhs, ldLhs, dWork, dbpiv, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the factorization was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                if (usePivot) {
                    std::cerr << "ERROR: CudaSolverType::LU_PIVOT failed!" << std::endl;
                } else {
                    std::cerr << "ERROR: CudaSolverType::LU failed!" << std::endl;
                }
                exit(1);
            }

            // Solve A*l = LU*l = I.
            cudaErrorCheck(cusolverDnRgetrs(
                    cusolverHandle, CUBLAS_OP_N, L, D1, dLhs, ldLhs, dbpiv, dRhs, ldRhs, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(x.data(), dRhs, sizeof(Real) * L * D1, cudaMemcpyDeviceToHost));

            if (usePivot && dbpiv) {
                cudaErrorCheck(cudaFree(dbpiv));
            }

            break;
        }

        // See: https://docs.nvidia.com/cuda/cusolver/index.html#qr_examples
        case CudaSolverType::QR: {
            const Real one = 1.0f;
            int lwork_geqrf = 0;
            int lwork_ormqr = 0;
            Real* dTau = nullptr;

            cudaErrorCheck(cudaMalloc((void**)&dTau, sizeof(Real) * L));

            // Query working space required by geqrf and ormqr.
            cudaErrorCheck(cusolverDnRgeqrf_bufferSize(
                    cusolverHandle, L, L, dLhs, ldLhs, &lwork_geqrf));
            cudaErrorCheck(cusolverDnRormqr_bufferSize(
                    cusolverHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                    L, D1, L, dLhs, ldLhs, dTau, dRhs, ldRhs, &lwork_ormqr));
            lwork = std::max(lwork_geqrf, lwork_ormqr);
            cudaErrorCheck(cudaMalloc((void**)&dWork, sizeof(Real)*lwork));

            // Compute the QR factorization.
            cudaErrorCheck(cusolverDnRgeqrf(
                    cusolverHandle, L, L, dLhs, ldLhs, dTau, dWork, lwork, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the factorization was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::QR failed!" << std::endl;
                exit(1);
            }

            // Compute Q^T*I.
            cudaErrorCheck(cusolverDnRormqr(
                    cusolverHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                    L, D1, L, dLhs, ldLhs, dTau, dRhs, ldRhs, dWork, lwork, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the factorization was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::QR failed!" << std::endl;
                exit(1);
            }

            // Solve R*l = Q^T*I (i.e., l = R \ Q^T*I).
            cudaErrorCheck(cublasRtrsm(
                    cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    L, D1, &one, dLhs, ldLhs, dRhs, ldRhs));
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(x.data(), dRhs, sizeof(Real) * L * D1, cudaMemcpyDeviceToHost));

            if (dTau) {
                cudaErrorCheck(cudaFree(dTau));
            }

            break;
        }

        // https://docs.nvidia.com/cuda/cusolver/index.html#chol_examples
        case CudaSolverType::CHOL: {
            const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

            // Query working space required by potrf.
            cudaErrorCheck(cusolverDnRpotrf_bufferSize(
                    cusolverHandle, uplo, L, dLhs, ldLhs, &lwork));
            cudaErrorCheck(cudaMalloc((void**)&dWork, sizeof(Real)*lwork));

            // Cholesky factorization.
            cudaErrorCheck(cusolverDnRpotrf(
                    cusolverHandle, uplo, L, dLhs, ldLhs, dWork, lwork, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the factorization was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::CHOL failed!" << std::endl;
                exit(1);
            }

            // Solving step.
            cudaErrorCheck(cusolverDnRpotrs(
                    cusolverHandle, uplo, L, D1, dLhs, ldLhs, dRhs, ldRhs, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the solving step was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::CHOL failed!" << std::endl;
                exit(1);
            }

            cudaErrorCheck(cudaMemcpy(x.data(), dRhs, sizeof(Real) * L * D1, cudaMemcpyDeviceToHost));

            break;
        }
    }

    // Free the allocated memory.
    if (dWork) {
        cudaErrorCheck(cudaFree(dWork));
    }
    if (dInfo) {
        cudaErrorCheck(cudaFree(dInfo));
    }
    if (dLhs) {
        cudaErrorCheck(cudaFree(dLhs));
    }
    if (dRhs) {
        cudaErrorCheck(cudaFree(dRhs));
    }
    if (dA) {
        cudaErrorCheck(cudaFree(dA));
    }
    if (db) {
        cudaErrorCheck(cudaFree(db));
    }
    if (dx) {
        cudaErrorCheck(cudaFree(dx));
    }
}
