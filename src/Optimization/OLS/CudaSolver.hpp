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

#ifndef SH_AUG_CUDASOLVER_HPP
#define SH_AUG_CUDASOLVER_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Graphics/Vulkan/Utils/InteropCuda.hpp>

#include "PrecisionDefines.hpp"
#include "../OptDefines.hpp"

/**
 * Initializes CUDA, cuBLAS and cuSOLVE for use in @see solveLeastSquaresCudaDense.
 */
void cudaInit(bool isMainThread, CUcontext cudaContext = nullptr, void* cudaStream = nullptr);

/**
 * Releases CUDA, cuBLAS and cuSOLVE.
 * @see cudaInit must be called exactly once before calling cudaRelease.
 * @see solveLeastSquaresCudaDense must no longer be called after a call to cudaRelease.
 */
void cudaRelease();

/**
 * Solves A*x = b for the vector x in the least squares sense.
 * This solver uses the libraries cuBLAS and cuSOLVER.
 * @param cudaSolverType The type of solver to use.
 * @param useNormalEquations Whether to use relaxation/a regularizer.
 * @param lambdaL The relaxation/regularization factor.
 * @param A The system matrix A.
 * @param b The right-hand side vector.
 * @param x The solution of the least squares problem.
 */
template<class Real>
void solveLeastSquaresCudaDense(
        CudaSolverType cudaSolverType, bool useNormalEquations, const Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x);
extern template
void solveLeastSquaresCudaDense<float>(
        CudaSolverType cudaSolverType, bool useNormalEquations, const float lambdaL,
        const Eigen::MatrixXf& A, const Eigen::MatrixXf& b, Eigen::MatrixXf& x);
extern template
void solveLeastSquaresCudaDense<double>(
        CudaSolverType cudaSolverType, bool useNormalEquations, const double lambdaL,
        const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, Eigen::MatrixXd& x);

/**
 * Solves A*x = b for the vector x in the least squares sense.
 * This solver uses the libraries cuSPARSE and cuSOLVER.
 * @param m The number of rows of the matrix.
 * @param n The number of columns of the matrix.
 * @param nnz The number of non-zero values.
 * @param csrVals The list of non-zero matrix values (CSR format).
 * @param csrRowPtr The row pointers of the matrix data (CSR format).
 * @param csrColInd The column indices of the values in csrVals (CSR format).
 * @param b The right-hand side vector.
 * @param x The solution of the least squares problem.
 */
template<class Real>
void solveLeastSquaresCudaSparse(
        CudaSparseSolverType cudaSparseSolverType, bool isDevicePtr, Real lambdaL,
        int m, int n, int nnz, Real* csrVals, int* csrRowPtr, int* csrColInd,
        Real* b, Eigen::MatrixXr& x);
extern template
void solveLeastSquaresCudaSparse<float>(
        CudaSparseSolverType cudaSparseSolverType, bool isDevicePtr, float lambdaL,
        int m, int n, int nnz, float * csrVals, int* csrRowPtr, int* csrColInd,
        float * b, Eigen::MatrixXf& x);
extern template
void solveLeastSquaresCudaSparse<double>(
        CudaSparseSolverType cudaSparseSolverType, bool isDevicePtr, double lambdaL,
        int m, int n, int nnz, double * csrVals, int* csrRowPtr, int* csrColInd,
        double* b, Eigen::MatrixXd& x);

/**
 * Solves A*x = b for the vector x using the normal equations, i.e., A^T A x = A^T b.
 * This solver uses the libraries cuSPARSE and cuSOLVER.
 * @param m The number of rows of the matrix.
 * @param n The number of columns of the matrix.
 * @param nnz The number of non-zero values.
 * @param csrVals The list of non-zero matrix values (CSR format).
 * @param csrRowPtr The row pointers of the matrix data (CSR format).
 * @param csrColInd The column indices of the values in csrVals (CSR format).
 * @param b The right-hand side vector.
 * @param x The solution of the least squares problem.
 */
template<class Real>
void solveLeastSquaresCudaSparseNormalEquations(
        CudaSparseSolverType cudaSparseSolverType, EigenSolverType eigenSolverType, bool isDevicePtr, Real lambdaL,
        int m, int n, int nnz, Real* csrVals, int* csrRowPtr, int* csrColInd,
        Real* b, Eigen::MatrixXr& x);
extern template
void solveLeastSquaresCudaSparseNormalEquations<float>(
        CudaSparseSolverType cudaSparseSolverType, EigenSolverType eigenSolverType, bool isDevicePtr, float lambdaL,
        int m, int n, int nnz, float* csrVals, int* csrRowPtr, int* csrColInd,
        float* b, Eigen::MatrixXf& x);
extern template
void solveLeastSquaresCudaSparseNormalEquations<double>(
        CudaSparseSolverType cudaSparseSolverType, EigenSolverType eigenSolverType, bool isDevicePtr, double lambdaL,
        int m, int n, int nnz, double* csrVals, int* csrRowPtr, int* csrColInd,
        double* b, Eigen::MatrixXd& x);

template<class Real>
void createSystemMatrixCudaSparse(
        int xs, int ys, int zs, int tfSize, float minGT, float maxGT, float minOpt, float maxOpt,
        CUtexObject scalarFieldGT, CUtexObject scalarFieldOpt, const float* tfGT,
        int& numRows, int& nnz, Real*& csrVals, int*& csrRowPtr, int*& csrColInd, Real*& b);
extern template
void createSystemMatrixCudaSparse<float>(
        int xs, int ys, int zs, int tfSize, float minGT, float maxGT, float minOpt, float maxOpt,
        CUtexObject scalarFieldGT, CUtexObject scalarFieldOpt, const float* tfGT,
        int& numRows, int& nnz, float*& csrVals, int*& csrRowPtr, int*& csrColInd, float*& b);
extern template
void createSystemMatrixCudaSparse<double>(
        int xs, int ys, int zs, int tfSize, float minGT, float maxGT, float minOpt, float maxOpt,
        CUtexObject scalarFieldGT, CUtexObject scalarFieldOpt, const float* tfGT,
        int& numRows, int& nnz, double*& csrVals, int*& csrRowPtr, int*& csrColInd, double*& b);

template<class Real>
void freeSystemMatrixCudaSparse(Real* csrVals, int* csrRowPtr, int* csrColInd, Real* b);
extern template
void freeSystemMatrixCudaSparse<float>(float* csrVals, int* csrRowPtr, int* csrColInd, float* b);
extern template
void freeSystemMatrixCudaSparse<double>(double* csrVals, int* csrRowPtr, int* csrColInd, double* b);

#endif //SH_AUG_CUDASOLVER_HPP
