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
#include "PrecisionDefines.hpp"
#include "../OptDefines.hpp"

/**
 * Initializes CUDA, cuBLAS and cuSOLVE for use in @see solveLeastSquaresCudaDense.
 */
void cudaInit(void* cudaStream = nullptr);

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
void solveLeastSquaresCudaDense(
        CudaSolverType cudaSolverType, bool useNormalEquations, const Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x);

/**
 * Solves A*x = b for the vector x in the least squares sense.
 * This solver uses the libraries cuSPARSE and cuSOLVER.
 * NOTE: It seems like the cuSOLVER solver is broken. Do not use this function.
 * @param m The number of rows of the matrix.
 * @param n The number of columns of the matrix.
 * @param nnz The number of non-zero values.
 * @param csrVals The list of non-zero matrix values (CSR format).
 * @param csrRowPtr The row pointers of the matrix data (CSR format).
 * @param csrColInd The column indices of the values in csrVals (CSR format).
 * @param b The right-hand side vector.
 * @param x The solution of the least squares problem.
 */
void solveLeastSquaresCudaSparse(
        int m, int n, int nnz, const Real* csrVals, const int* csrRowPtr, const int* csrColInd,
        const Real* b, Eigen::MatrixXr& x);

#endif //SH_AUG_CUDASOLVER_HPP