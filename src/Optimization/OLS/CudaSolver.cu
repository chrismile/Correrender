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
#include <cusolverSp.h>

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>

#include "CudaHelpers.cuh"
#include "CudaForward.hpp"
#include "CudaSubroutines.cuh"
#include "lsqr.cuh"
#include "cgls.cuh"
#include "EigenSolver.hpp"
#include "CudaSolver.hpp"
#include "CSR/PrefixSumScan.cuh"
#include "CSR/BuildMatrix.cuh"

static bool useCustomCudaStream = false;
static bool useCustomCudaContext = false;
static CUcontext cuContext = nullptr;
static CUstream cuStream = nullptr;
static cudaStream_t stream = nullptr;
static cublasHandle_t cublasHandle = nullptr;
static cusparseHandle_t cusparseHandle = nullptr;
static cusolverDnHandle_t cusolverHandle = nullptr;
static cusolverSpHandle_t cusolverSpHandle = nullptr;

void cudaInit(bool isMainThread, CUcontext cudaContext, void* cudaStream) {
    if (cudaContext) {
        cuContext = cudaContext;
        if (!isMainThread) {
            cuCtxSetCurrent(cudaContext);
        }
        useCustomCudaContext = true;
    } else {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxCreate(
                &cuContext, CU_CTX_SCHED_AUTO, 0), "Error in cuCtxCreate: ");
    }
    if (cudaStream) {
        stream = cudaStream_t(cudaStream);
        useCustomCudaStream = true;
    } else {
        //cudaErrorCheck(cudaStreamCreate(&stream));
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamCreate(
                &cuStream, 0), "Error in cuStreamCreate: ");
        stream = cuStream;
    }

    // Initialize cuBLAS and cuSOLVER.
    cudaErrorCheck(cublasCreate(&cublasHandle));
    cudaErrorCheck(cusparseCreate(&cusparseHandle));
    cudaErrorCheck(cusolverDnCreate(&cusolverHandle));
    cudaErrorCheck(cusolverSpCreate(&cusolverSpHandle));

    cudaErrorCheck(cublasSetStream(cublasHandle, stream));
    cudaErrorCheck(cusparseSetStream(cusparseHandle, stream));
    cudaErrorCheck(cusolverDnSetStream(cusolverHandle, stream));
    cudaErrorCheck(cusolverSpSetStream(cusolverSpHandle, stream));
}

void cudaRelease() {
    // Free cuBLAS and cuSOLVER.
    if (cusolverHandle) {
        cudaErrorCheck(cusolverDnDestroy(cusolverHandle));
        cusolverHandle = nullptr;
    }
    if (cusolverSpHandle) {
        cudaErrorCheck(cusolverSpDestroy(cusolverSpHandle));
        cusolverSpHandle = nullptr;
    }
    if (cublasHandle) {
        cudaErrorCheck(cublasDestroy(cublasHandle));
        cublasHandle = nullptr;
    }
    if (cusparseHandle) {
        cudaErrorCheck(cusparseDestroy(cusparseHandle));
        cusparseHandle = nullptr;
    }
    if (stream) {
        if (!useCustomCudaStream) {
            //cudaErrorCheck(cudaStreamDestroy(stream));
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamDestroy(
                    cuStream), "Error in cuStreamDestroy: ");
        }
        if (!useCustomCudaContext) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxDestroy(
                    cuContext), "Error in cuCtxDestroy: ");
        }
        stream = nullptr;
        cuStream = nullptr;
        cuContext = nullptr;
    }
}

// https://eigen.tuxfamily.org/dox/TopicCUDA.html
// https://docs.nvidia.com/cuda/cusolver/index.html#introduction
template<class Real>
void solveLeastSquaresCudaDense(
        CudaSolverType cudaSolverType, bool useNormalEquations, const Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    if (!useNormalEquations && cudaSolverType != CudaSolverType::QR) {
        sgl::Logfile::get()->writeError(
                "Error in solveLeastSquaresCudaDense: Only QR can solve non-square matrices. Switching to QR.");
        cudaSolverType = CudaSolverType::QR;
    }

    // A \in R^MxN, I \in R^Nx1, l \in R^Lx1, A*l = I.
    const int M = int(A.rows());
    const int N = int(A.cols());
    assert(A.rows() == b.rows());
    assert(1 == b.cols());
    x = Eigen::MatrixXr(N, 1);

    // Allocate memory on the device.
    Real* dA = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&dA, sizeof(Real) * M * N));
    cudaErrorCheck(cudaMemcpy(dA, A.data(), sizeof(Real) * M * N, cudaMemcpyHostToDevice));
    Real* db = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&db, sizeof(Real) * M));
    cudaErrorCheck(cudaMemcpy(db, b.data(), sizeof(Real) * M, cudaMemcpyHostToDevice));
    Real* dx = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&dx, sizeof(Real) * N));

    // lhs = A^T*A + lambda_l*M_I
    Real* dLhs = nullptr;
    // rhs = A^T*b
    Real* dRhs = nullptr;
    if (useNormalEquations) {
        Eigen::MatrixXr M_I = Eigen::MatrixXr::Identity(N, N);
        cudaErrorCheck(cudaMalloc((void**)&dLhs, sizeof(Real) * N * N));
        cudaMemcpy(dLhs, M_I.data(), sizeof(Real) * N * N, cudaMemcpyHostToDevice);
        cudaErrorCheck(cudaMalloc((void**)&dRhs, sizeof(Real) * N));
    } else {
        dLhs = dA;
        dRhs = db;
    }

    const int lda = M;
    const int ldb = M;
    const int lhsM = useNormalEquations ? N : M;
    const int lhsN = N;
    const int ldLhs = useNormalEquations ? N : M;
    const int ldRhs = useNormalEquations ? N : M;

    if (useNormalEquations) {
        // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
        // cublasSgemm: C = alpha*op(A)op(B) + beta*C
        const Real alpha = Real(1.0);
        // Compute: lhs = A^T*A + lambda_l*M_I.
        const Real beta0 = lambdaL;
        cudaErrorCheck(cublasRgemm<Real>(
                cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, M, &alpha, dA, lda, dA, lda, &beta0, dLhs, ldLhs));
        // Compute: rhs = A^T*b.
        const Real beta1 = Real(0.0);
        cudaErrorCheck(cublasRgemm<Real>(
                cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                N, 1, M, &alpha, dA, lda, db, ldb, &beta1, dRhs, ldRhs));
    }

    int lwork = 0;
    Real* dWork = nullptr;
    int* dInfo = nullptr;
    int hInfo = 0;
    cudaErrorCheck(cudaMalloc((void**)&dInfo, sizeof(int)));

    // Now, solve lhs*x = rhs.
    switch(cudaSolverType) {
        // See: https://docs.nvidia.com/cuda/cusolver/index.html#lu_examples
        case CudaSolverType::LU: case CudaSolverType::LU_PIVOT: {
            bool usePivot = cudaSolverType == CudaSolverType::LU_PIVOT;

            // Query working space required by getrf.
            cudaErrorCheck(cusolverDnRgetrf_bufferSize<Real>(
                    cusolverHandle, N, N, dLhs, ldLhs, &lwork));
            cudaErrorCheck(cudaMalloc((void**)&dWork, sizeof(Real)*lwork));

            // Create data for pivot if user requests pivoting.
            int* dbpiv = nullptr;
            if (usePivot) {
                cudaErrorCheck(cudaMalloc((void**)&dbpiv, sizeof(int) * N));
            }

            // LU factorization.
            cudaErrorCheck(cusolverDnRgetrf<Real>(
                    cusolverHandle, N, N, dLhs, ldLhs, dWork, dbpiv, dInfo));
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
            cudaErrorCheck(cusolverDnRgetrs<Real>(
                    cusolverHandle, CUBLAS_OP_N, N, 1, dLhs, ldLhs, dbpiv, dRhs, ldRhs, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(x.data(), dRhs, sizeof(Real) * N, cudaMemcpyDeviceToHost));

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

            cudaErrorCheck(cudaMalloc((void**)&dTau, sizeof(Real) * N));

            // Query working space required by geqrf and ormqr.
            cudaErrorCheck(cusolverDnRgeqrf_bufferSize<Real>(
                    cusolverHandle, lhsM, lhsN, dLhs, ldLhs, &lwork_geqrf));
            cudaErrorCheck(cusolverDnRormqr_bufferSize<Real>(
                    cusolverHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                    lhsM, 1, lhsN, dLhs, ldLhs, dTau, dRhs, ldRhs, &lwork_ormqr));
            lwork = std::max(lwork_geqrf, lwork_ormqr);
            cudaErrorCheck(cudaMalloc((void**)&dWork, sizeof(Real)*lwork));

            // Compute the QR factorization.
            cudaErrorCheck(cusolverDnRgeqrf<Real>(
                    cusolverHandle, lhsM, lhsN, dLhs, ldLhs, dTau, dWork, lwork, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the factorization was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::QR failed!" << std::endl;
                exit(1);
            }

            // Compute Q^T*I.
            cudaErrorCheck(cusolverDnRormqr<Real>(
                    cusolverHandle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                    lhsM, 1, lhsN, dLhs, ldLhs, dTau, dRhs, ldRhs, dWork, lwork, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the factorization was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::QR failed!" << std::endl;
                exit(1);
            }

            // Solve R*l = Q^T*I (i.e., l = R \ Q^T*I).
            cudaErrorCheck(cublasRtrsm<Real>(
                    cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    lhsN, 1, &one, dLhs, ldLhs, dRhs, ldRhs));
            cudaErrorCheck(cudaDeviceSynchronize());
            cudaErrorCheck(cudaMemcpy(x.data(), dRhs, sizeof(Real) * N, cudaMemcpyDeviceToHost));

            if (dTau) {
                cudaErrorCheck(cudaFree(dTau));
            }

            break;
        }

        // https://docs.nvidia.com/cuda/cusolver/index.html#chol_examples
        case CudaSolverType::CHOL: {
            const cublasFillMode_t fillMode = CUBLAS_FILL_MODE_LOWER;

            // Query working space required by potrf.
            cudaErrorCheck(cusolverDnRpotrf_bufferSize<Real>(
                    cusolverHandle, fillMode, N, dLhs, ldLhs, &lwork));
            cudaErrorCheck(cudaMalloc((void**)&dWork, sizeof(Real)*lwork));

            // Cholesky factorization.
            cudaErrorCheck(cusolverDnRpotrf<Real>(
                    cusolverHandle, fillMode, N, dLhs, ldLhs, dWork, lwork, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the factorization was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::CHOL failed!" << std::endl;
                exit(1);
            }

            // Solving step.
            cudaErrorCheck(cusolverDnRpotrs<Real>(
                    cusolverHandle, fillMode, N, 1, dLhs, ldLhs, dRhs, ldRhs, dInfo));
            cudaErrorCheck(cudaDeviceSynchronize());

            // Check whether the solving step was successful.
            cudaErrorCheck(cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (hInfo != 0) {
                std::cerr << "ERROR: CudaSolverType::CHOL failed!" << std::endl;
                exit(1);
            }

            cudaErrorCheck(cudaMemcpy(x.data(), dRhs, sizeof(Real) * N, cudaMemcpyDeviceToHost));

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
    if (useNormalEquations) {
        if (dLhs) {
            cudaErrorCheck(cudaFree(dLhs));
        }
        if (dRhs) {
            cudaErrorCheck(cudaFree(dRhs));
        }
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
template
void solveLeastSquaresCudaDense<float>(
        CudaSolverType cudaSolverType, bool useNormalEquations, const float lambdaL,
        const Eigen::MatrixXf& A, const Eigen::MatrixXf& b, Eigen::MatrixXf& x);
template
void solveLeastSquaresCudaDense<double>(
        CudaSolverType cudaSolverType, bool useNormalEquations, const double lambdaL,
        const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, Eigen::MatrixXd& x);

template<class Real>
void solveLeastSquaresCudaSparse(
        CudaSparseSolverType cudaSparseSolverType, bool isDevicePtr, Real lambdaL,
        int m, int n, int nnz, Real* csrVals, int* csrRowPtr, int* csrColInd,
        Real* b, Eigen::MatrixXr& x) {
    if (cudaSparseSolverType == CudaSparseSolverType::CUSOLVER_QR) {
        auto* p = new int[n];
        x = Eigen::MatrixXr(n, 1);
        int rank = 0;
        auto minNorm = Real(0);
        cusparseMatDescr_t matDesc{};
        cudaErrorCheck(cusparseCreateMatDescr(&matDesc));
        if (isDevicePtr) {
            auto* hcsrVals = new Real[nnz];
            auto* hcsrRowPtr = new int[nnz];
            auto* hcsrColInd = new int[nnz];
            auto* hb = new Real[nnz];
            cudaErrorCheck(cudaMemcpy(hcsrVals, csrVals, sizeof(Real) * nnz, cudaMemcpyDeviceToHost));
            cudaErrorCheck(cudaMemcpy(hcsrRowPtr, csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));
            cudaErrorCheck(cudaMemcpy(hcsrColInd, csrColInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost));
            cudaErrorCheck(cudaMemcpy(hb, b, sizeof(Real) * m, cudaMemcpyDeviceToHost));
            csrVals = hcsrVals;
            csrRowPtr = hcsrRowPtr;
            csrColInd = hcsrColInd;
            b = hb;
        }
        cudaErrorCheck(cusolverSpRcsrlsqvqrHost<Real>(
                cusolverSpHandle, m, n, nnz, matDesc, csrVals, csrRowPtr, csrColInd, b,
                Real(1e-5f), &rank, x.data(), p, &minNorm));
        cudaErrorCheck(cusparseDestroyMatDescr(matDesc));
        if (isDevicePtr) {
            delete[] csrVals;
            delete[] csrRowPtr;
            delete[] csrColInd;
            delete[] b;
        }
        delete[] p;
    } else {
        bool quiet = false;
        float tol = 1e-6f;
        int maxit = 100;
        float s = lambdaL;
        Real* dx = nullptr, *db = nullptr, *dcsrVals = nullptr;
        int* dcsrRowPtr = nullptr, *dcsrColInd = nullptr;
        cudaErrorCheck(cudaMalloc((void**)&dx, sizeof(Real) * n));
        if (isDevicePtr) {
            db = b;
            dcsrVals = csrVals;
            dcsrRowPtr = csrRowPtr;
            dcsrColInd = csrColInd;
        } else {
            cudaErrorCheck(cudaMalloc((void**)&db, sizeof(Real) * m));
            cudaErrorCheck(cudaMalloc((void**)&dcsrVals, sizeof(Real) * nnz));
            cudaErrorCheck(cudaMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
            cudaErrorCheck(cudaMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
            cudaErrorCheck(cudaMemcpy(db, b, sizeof(Real) * m, cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMemcpy(dcsrVals, csrVals, sizeof(Real) * nnz, cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMemcpy(dcsrRowPtr, csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice));
            cudaErrorCheck(cudaMemcpy(dcsrColInd, csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice));
        }
        cudaErrorCheck(cudaMemset(dx, 0, sizeof(Real) * n));
        if (cudaSparseSolverType == CudaSparseSolverType::LSQR) {
            solveLeastSquaresCudaLSQR<Real>(
                    stream, cublasHandle, cusparseHandle, dcsrRowPtr, dcsrColInd, dcsrVals, m, n, nnz, db, dx, maxit);
        } else {
            solveLeastSquaresCudaCGLS<Real>(
                    stream, cublasHandle, cusparseHandle, dcsrVals, dcsrRowPtr, dcsrColInd,
                    m, n, nnz, db, dx, s, tol, maxit, quiet);
        }
        cudaErrorCheck(cudaMemcpy(x.data(), dx, sizeof(Real) * n, cudaMemcpyDeviceToHost));
        cudaErrorCheck(cudaFree(dx));
        if (!isDevicePtr) {
            cudaErrorCheck(cudaFree(db));
            cudaErrorCheck(cudaFree(dcsrVals));
            cudaErrorCheck(cudaFree(dcsrRowPtr));
            cudaErrorCheck(cudaFree(dcsrColInd));
        }
    }
}
template
void solveLeastSquaresCudaSparse<float>(
        CudaSparseSolverType cudaSparseSolverType, bool isDevicePtr, float lambdaL,
        int m, int n, int nnz, float * csrVals, int* csrRowPtr, int* csrColInd,
        float * b, Eigen::MatrixXf& x);
template
void solveLeastSquaresCudaSparse<double>(
        CudaSparseSolverType cudaSparseSolverType, bool isDevicePtr, double lambdaL,
        int m, int n, int nnz, double * csrVals, int* csrRowPtr, int* csrColInd,
        double* b, Eigen::MatrixXd& x);



template<class Real>
void solveLeastSquaresCudaSparseNormalEquations(
        CudaSparseSolverType cudaSparseSolverType, EigenSolverType eigenSolverType, bool isDevicePtr, Real lambdaL,
        int m, int n, int nnz, Real* csrVals, int* csrRowPtr, int* csrColInd,
        Real* b, Eigen::MatrixXr& x) {
    Real* db = nullptr, *dlhs = nullptr, *drhs = nullptr;
    Real* dcsrVals = nullptr, *dcscVals = nullptr, *dcsrValsATA = nullptr;
    int* dcsrRowPtr = nullptr, *dcsrColInd = nullptr, *dcscColPtr = nullptr, *dcscRowInd = nullptr;
    int* dcsrRowPtrATA = nullptr, *dcsrColIndATA = nullptr;
    if (isDevicePtr) {
        db = b;
        dcsrVals = csrVals;
        dcsrRowPtr = csrRowPtr;
        dcsrColInd = csrColInd;
    } else {
        cudaErrorCheck(cudaMalloc((void**)&db, sizeof(Real) * m));
        cudaErrorCheck(cudaMalloc((void**)&dcsrVals, sizeof(Real) * nnz));
        cudaErrorCheck(cudaMalloc((void**)&dcsrRowPtr, sizeof(int) * (m + 1)));
        cudaErrorCheck(cudaMalloc((void**)&dcsrColInd, sizeof(int) * nnz));
        cudaErrorCheck(cudaMemcpy(db, b, sizeof(Real) * m, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(dcsrVals, csrVals, sizeof(Real) * nnz, cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(dcsrRowPtr, csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(dcsrColInd, csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    }
    cudaErrorCheck(cudaMalloc((void**)&dlhs, sizeof(Real) * n * n));
    cudaErrorCheck(cudaMalloc((void**)&drhs, sizeof(Real) * n));
    cudaErrorCheck(cudaMalloc((void**)&dcscVals, sizeof(Real) * nnz));
    cudaErrorCheck(cudaMalloc((void**)&dcscColPtr, sizeof(int) * (n + 1)));
    cudaErrorCheck(cudaMalloc((void**)&dcscRowInd, sizeof(int) * nnz));
    cudaErrorCheck(cudaMalloc((void**)&dcsrRowPtrATA, sizeof(int) * (n + 1)));
    Eigen::MatrixXr lhs = Eigen::MatrixXr(n, n);
    Eigen::MatrixXr rhs = Eigen::VectorXr(n);

    AuxBuffer auxBuffer;

    cudaDataType dataType;
    if constexpr(std::is_same<Real, double>::value) {
        dataType = CUDA_R_64F;
    } else if constexpr(std::is_same<Real, float>::value) {
        dataType = CUDA_R_32F;
    }

    size_t bufferSize = 0;
    cudaErrorCheck(cusparseCsr2cscEx2_bufferSize(
            cusparseHandle, m, n, nnz,
            dcsrVals, dcsrRowPtr, dcsrColInd,
            dcscVals, dcscColPtr, dcscRowInd,
            dataType, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, // CUSPARSE_CSR2CSC_ALG2
            &bufferSize));
    cudaErrorCheck(cudaStreamSynchronize(stream));
    auxBuffer.reserve(bufferSize);
    cudaErrorCheck(cusparseCsr2cscEx2(
            cusparseHandle, m, n, nnz,
            dcsrVals, dcsrRowPtr, dcsrColInd,
            dcscVals, dcscColPtr, dcscRowInd,
            dataType, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, // CUSPARSE_CSR2CSC_ALG2
            auxBuffer.getPointer()));

    cusparseSpMatDescr_t matDescrA{}, matDescrAT{};
    cudaErrorCheck(cusparseCreateCsr(
            &matDescrA, m, n, nnz,
            (void*)dcsrRowPtr, (void*)dcsrColInd, (void*)dcsrVals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dataType));
    cudaErrorCheck(cusparseCreateCsr(
            &matDescrAT, n, m, nnz,
            (void*)dcscColPtr, (void*)dcscRowInd, (void*)dcscVals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dataType));

    // Compute A^T b.
    auto alpha = Real(1);
    auto beta = Real(0);
    DnVector bvec(db, m);
    DnVector rhsvec(drhs, n);
    cudaErrorCheck(cusparseSpMV_bufferSize(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, bvec.getCusparseDnVecDescr(),
            &beta, rhsvec.getCusparseDnVecDescr(), dataType,
            CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    cudaErrorCheck(cudaStreamSynchronize(stream));
    auxBuffer.reserve(bufferSize);
    cudaErrorCheck(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, bvec.getCusparseDnVecDescr(),
            &beta, rhsvec.getCusparseDnVecDescr(), dataType,
            CUSPARSE_SPMV_ALG_DEFAULT, auxBuffer.getPointer()));
    cudaErrorCheck(cudaMemcpy(rhs.data(), drhs, sizeof(Real) * n, cudaMemcpyDeviceToHost));

    // Compute A^T A.
#if CUDART_VERSION >= 12000
    cusparseSpGEMMAlg_t spGemmAlg = CUSPARSE_SPGEMM_ALG3;
#else
    cusparseSpGEMMAlg_t spGemmAlg = CUSPARSE_SPGEMM_DEFAULT;
#endif
    cusparseSpMatDescr_t matDescrATA{};
    cudaErrorCheck(cusparseCreateCsr(
            &matDescrATA, n, n, 0, nullptr, nullptr, nullptr,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dataType));
    cusparseSpGEMMDescr_t spGemmDesc;
    cudaErrorCheck(cusparseSpGEMM_createDescr(&spGemmDesc));
    void* dBuffer1 = nullptr, *dBuffer2 = nullptr, *dBuffer3 = nullptr;
    size_t bufferSize1 = 0, bufferSize2 = 0, bufferSize3 = 0;
    int64_t numProducts, numRowsATA, numColsATA, nnzATA;
    const float chunkFraction = 0.2;

    cudaErrorCheck(cusparseSpGEMM_workEstimation(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, matDescrA, &beta, matDescrATA, dataType, spGemmAlg,
            spGemmDesc, &bufferSize1, nullptr));
    cudaErrorCheck(cudaMalloc((void**) &dBuffer1, bufferSize1));
    cudaErrorCheck(cusparseSpGEMM_workEstimation(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, matDescrA, &beta, matDescrATA, dataType, spGemmAlg,
            spGemmDesc, &bufferSize1, dBuffer1));

#if CUDART_VERSION >= 12000
    cudaErrorCheck(cusparseSpGEMM_getNumProducts(spGemmDesc, &numProducts));
    cudaErrorCheck(cusparseSpGEMM_estimateMemory(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, matDescrA, &beta, matDescrATA, dataType, spGemmAlg,
            spGemmDesc, chunkFraction, &bufferSize3, nullptr, nullptr));
    cudaErrorCheck(cudaMalloc((void**)&dBuffer3, bufferSize3));
    cudaErrorCheck(cusparseSpGEMM_estimateMemory(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, matDescrA, &beta, matDescrATA, dataType, spGemmAlg,
            spGemmDesc, chunkFraction, &bufferSize3, dBuffer3, &bufferSize2));
    cudaErrorCheck(cudaFree(dBuffer3));
#else
    cudaErrorCheck(cusparseSpGEMM_compute(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, matDescrA, &beta, matDescrATA,
            dataType, spGemmAlg, spGemmDesc, &bufferSize2, nullptr));
#endif

    cudaErrorCheck(cudaMalloc((void**)&dBuffer2, bufferSize2));

    cudaErrorCheck(cusparseSpGEMM_compute(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, matDescrA, &beta, matDescrATA,
            dataType, spGemmAlg, spGemmDesc, &bufferSize2, dBuffer2));
    cudaErrorCheck(cusparseSpMatGetSize(matDescrATA, &numRowsATA, &numColsATA, &nnzATA));
    cudaErrorCheck(cudaMalloc((void**)&dcsrColIndATA, nnzATA * sizeof(int)));
    cudaErrorCheck(cudaMalloc((void**)&dcsrValsATA, nnzATA * sizeof(Real)));

    cudaErrorCheck(cusparseCsrSetPointers(matDescrATA, dcsrRowPtrATA, dcsrColIndATA, dcsrValsATA));
    cudaErrorCheck(cusparseSpGEMM_copy(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matDescrAT, matDescrA, &beta, matDescrATA,
            dataType, spGemmAlg, spGemmDesc));
    cudaErrorCheck(cusparseSpGEMM_destroyDescr(spGemmDesc));
    cudaErrorCheck(cudaFree(dBuffer1));
    cudaErrorCheck(cudaFree(dBuffer2));

    // Convert A^T A to a sparse matrix and download it.
    cusparseDnMatDescr_t matDescrATADense{};
    cudaErrorCheck(cusparseCreateDnMat(
            &matDescrATADense, n, n, n, dlhs, dataType, CUSPARSE_ORDER_COL));
    cudaErrorCheck(cusparseSparseToDense_bufferSize(
            cusparseHandle, matDescrATA, matDescrATADense,
            CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize));
    cudaErrorCheck(cudaStreamSynchronize(stream));
    auxBuffer.reserve(bufferSize);
    cudaErrorCheck(cusparseSparseToDense(
            cusparseHandle, matDescrATA, matDescrATADense,
            CUSPARSE_SPARSETODENSE_ALG_DEFAULT, auxBuffer.getPointer()));
    cudaErrorCheck(cudaMemcpy(lhs.data(), dlhs, sizeof(Real) * n * n, cudaMemcpyDeviceToHost));

    cudaErrorCheck(cusparseDestroyDnMat(matDescrATADense));
    cudaErrorCheck(cusparseDestroySpMat(matDescrA));
    cudaErrorCheck(cusparseDestroySpMat(matDescrAT));
    cudaErrorCheck(cusparseDestroySpMat(matDescrATA));
    if (!isDevicePtr) {
        cudaErrorCheck(cudaFree(db));
        cudaErrorCheck(cudaFree(dcsrVals));
        cudaErrorCheck(cudaFree(dcsrRowPtr));
        cudaErrorCheck(cudaFree(dcsrColInd));
    }
    cudaErrorCheck(cudaFree(drhs));
    cudaErrorCheck(cudaFree(dlhs));
    cudaErrorCheck(cudaFree(dcscVals));
    cudaErrorCheck(cudaFree(dcscColPtr));
    cudaErrorCheck(cudaFree(dcscRowInd));
    cudaErrorCheck(cudaFree(dcsrValsATA));
    cudaErrorCheck(cudaFree(dcsrRowPtrATA));
    cudaErrorCheck(cudaFree(dcsrColIndATA));

    solveLinearSystemEigenSymmetric(eigenSolverType, lambdaL, lhs, rhs, x);
}
template
void solveLeastSquaresCudaSparseNormalEquations<float>(
        CudaSparseSolverType cudaSparseSolverType, EigenSolverType eigenSolverType, bool isDevicePtr, float lambdaL,
        int m, int n, int nnz, float* csrVals, int* csrRowPtr, int* csrColInd,
        float* b, Eigen::MatrixXf& x);
template
void solveLeastSquaresCudaSparseNormalEquations<double>(
        CudaSparseSolverType cudaSparseSolverType, EigenSolverType eigenSolverType, bool isDevicePtr, double lambdaL,
        int m, int n, int nnz, double* csrVals, int* csrRowPtr, int* csrColInd,
        double* b, Eigen::MatrixXd& x);



template<class Real>
void createSystemMatrixCudaSparse(
        int xs, int ys, int zs, int tfSize, float minGT, float maxGT, float minOpt, float maxOpt,
        CUtexObject scalarFieldGT, CUtexObject scalarFieldOpt, const float* tfGT,
        int& numRows, int& nnz, Real*& csrVals, int*& csrRowPtr, int*& csrColInd, Real*& b) {
    const int numVoxels = xs * ys * zs;
    //const int m = numVoxels * 4;
    //const int n = tfSize * 4;
    const auto Nj = float(tfSize - 1);
    float* dtfGT = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&dtfGT, sizeof(float4) * tfSize));
    cudaErrorCheck(cudaMemcpy(dtfGT, tfGT, sizeof(float4) * tfSize, cudaMemcpyHostToDevice));

    const dim3 blockSize3D = dim3(8, 8, 4);
    const dim3 gridSize3D = dim3(
            sgl::uiceil(uint32_t(xs), blockSize3D.x),
            sgl::uiceil(uint32_t(ys), blockSize3D.y),
            sgl::uiceil(uint32_t(zs), blockSize3D.z));

    // In the first step, scan which rows contain how many non-zero values.
    int* rowsHasNonZero = nullptr, *rowsNumNonZero = nullptr;
    int* rowsHasNonZeroPrefixSum = nullptr, *rowsNumNonZeroPrefixSum = nullptr;
    cudaErrorCheck(cudaMalloc((void**)&rowsHasNonZero, sizeof(int) * numVoxels));
    cudaErrorCheck(cudaMalloc((void**)&rowsNumNonZero, sizeof(int) * numVoxels));
    cudaErrorCheck(cudaMalloc((void**)&rowsHasNonZeroPrefixSum, sizeof(int) * (numVoxels + 1)));
    cudaErrorCheck(cudaMalloc((void**)&rowsNumNonZeroPrefixSum, sizeof(int) * (numVoxels + 1)));
    computeNnzKernel<<<gridSize3D, blockSize3D, 0, stream>>>(
            xs, ys, zs, Nj, minGT, maxGT, minOpt, maxOpt,
            scalarFieldGT, scalarFieldOpt, rowsHasNonZero, rowsNumNonZero);

    // Next, use a parallel prefix scan to assign for each thread an index where to write to.
    std::vector<int*> bufferCache;
    allocateParallelPrefixSumScanBufferCache(numVoxels, bufferCache);
    parallelPrefixSumScan(stream, numVoxels, rowsHasNonZero, rowsHasNonZeroPrefixSum, bufferCache);
    cudaErrorCheck(cudaMemcpyAsync(
            &numRows, rowsHasNonZeroPrefixSum + numVoxels, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaErrorCheck(cudaStreamSynchronize(stream));
    parallelPrefixSumScan(stream, numVoxels, rowsNumNonZero, rowsNumNonZeroPrefixSum, bufferCache);
    cudaErrorCheck(cudaMemcpyAsync(
            &nnz, rowsNumNonZeroPrefixSum + numVoxels, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaErrorCheck(cudaStreamSynchronize(stream));
    freeParallelPrefixSumScanBufferCache(bufferCache);
    numRows *= 4;
    nnz *= 4;

    // Allocate the memory for the matrix and the vector.
    typedef typename MakeVec<Real, 4>::type real4;
    cudaErrorCheck(cudaMalloc((void**)&b, sizeof(real4) * numVoxels));
    cudaErrorCheck(cudaMalloc((void**)&csrVals, sizeof(Real) * nnz));
    cudaErrorCheck(cudaMalloc((void**)&csrRowPtr, sizeof(int) * (numRows + 1)));
    cudaErrorCheck(cudaMalloc((void**)&csrColInd, sizeof(int) * nnz));

    // Write the non-zero values to the correct position.
    writeCsrKernel<<<gridSize3D, blockSize3D, 0, stream>>>(
            xs, ys, zs, Nj, minGT, maxGT, minOpt, maxOpt,
            reinterpret_cast<const float4*>(dtfGT), scalarFieldGT, scalarFieldOpt,
            rowsHasNonZero, rowsHasNonZeroPrefixSum,
            rowsNumNonZero, rowsNumNonZeroPrefixSum,
            nnz, csrVals, csrRowPtr, csrColInd,
            reinterpret_cast<real4*>(b));
    writeFinalCsrRowPtr<<<1, 1, 0, stream>>>(nnz, numRows, csrRowPtr);

    cudaErrorCheck(cudaFree(rowsHasNonZero));
    cudaErrorCheck(cudaFree(rowsNumNonZero));
    cudaErrorCheck(cudaFree(rowsHasNonZeroPrefixSum));
    cudaErrorCheck(cudaFree(rowsNumNonZeroPrefixSum));
    cudaErrorCheck(cudaFree(dtfGT));
}
template
void createSystemMatrixCudaSparse<float>(
        int xs, int ys, int zs, int tfSize, float minGT, float maxGT, float minOpt, float maxOpt,
        CUtexObject scalarFieldGT, CUtexObject scalarFieldOpt, const float* tfGT,
        int& numRows, int& nnz, float*& csrVals, int*& csrRowPtr, int*& csrColInd, float*& b);
template
void createSystemMatrixCudaSparse<double>(
        int xs, int ys, int zs, int tfSize, float minGT, float maxGT, float minOpt, float maxOpt,
        CUtexObject scalarFieldGT, CUtexObject scalarFieldOpt, const float* tfGT,
        int& numRows, int& nnz, double*& csrVals, int*& csrRowPtr, int*& csrColInd, double*& b);



template<class Real>
void freeSystemMatrixCudaSparse(Real* csrVals, int* csrRowPtr, int* csrColInd, Real* b) {
    cudaErrorCheck(cudaFree(csrVals));
    cudaErrorCheck(cudaFree(csrRowPtr));
    cudaErrorCheck(cudaFree(csrColInd));
    cudaErrorCheck(cudaFree(b));
}
template
void freeSystemMatrixCudaSparse<float>(float* csrVals, int* csrRowPtr, int* csrColInd, float* b);
template
void freeSystemMatrixCudaSparse<double>(double* csrVals, int* csrRowPtr, int* csrColInd, double* b);
