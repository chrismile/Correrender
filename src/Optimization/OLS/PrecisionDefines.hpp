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

#ifndef SH_AUG_DEFINES_HPP
#define SH_AUG_DEFINES_HPP

#define SPARSE_ROW_MAJOR
#ifdef SPARSE_ROW_MAJOR
#define SparseMatrixXr SparseMatrixRowXr
#else
#define SparseMatrixXr SparseMatrixColXr
#endif

/**
 * The floating point type used for the simulation can be changed at compile time.
 */
//#define USE_DOUBLE_PRECISION

#ifdef USE_DOUBLE_PRECISION
// --- Real -> double

typedef double Real;
#define stringToReal std::stof
constexpr Real REAL_EPSILON = 1e-8;//std::numeric_limits<Real>::epsilon();

#define cublasRgemm cublasDgemm
#define cublasRtrsm cublasDtrsm
// LU decomposition
#define cusolverDnRgetrf_bufferSize cusolverDnDgetrf_bufferSize
#define cusolverDnRgetrf cusolverDnDgetrf
#define cusolverDnRgetrs cusolverDnDgetrs
// QR decomposition
#define cusolverDnRgeqrf_bufferSize cusolverDnDgeqrf_bufferSize
#define cusolverDnRormqr_bufferSize cusolverDnDormqr_bufferSize
#define cusolverDnRgeqrf cusolverDnDgeqrf
#define cusolverDnRormqr cusolverDnDormqr
// Cholesky decomposition
#define cusolverDnRpotrf_bufferSize cusolverDnDpotrf_bufferSize
#define cusolverDnRpotrf cusolverDnDpotrf
#define cusolverDnRpotrs cusolverDnDpotrs

// Sparse QR decomposition
#define cusolverSpRcsrlsqvqrHost cusolverSpDcsrlsqvqrHost

// Other sparse matrix functions
#define cusparseRcsrgemm2 cusparseDcsrgemm2

#ifdef EIGEN_CORE_H
namespace Eigen {
typedef Vector3d Vector3r;
typedef MatrixXd MatrixXr;
typedef RowVectorXd RowVectorXr;
typedef VectorXd VectorXr;
typedef SparseMatrix<double, Eigen::ColMajor> SparseMatrixColXr;
typedef SparseMatrix<double, Eigen::RowMajor> SparseMatrixRowXr;
}
#endif


#else
// --- Real -> float

typedef float Real;
#define stringToReal std::stod
constexpr Real REAL_EPSILON = 1e-3f;//std::numeric_limits<Real>::epsilon();

// LU decomposition
#define cusolverDnRgetrf_bufferSize cusolverDnSgetrf_bufferSize
#define cusolverDnRgetrf cusolverDnSgetrf
#define cusolverDnRgetrs cusolverDnSgetrs
// QR decomposition
#define cublasRgemm cublasSgemm
#define cublasRtrsm cublasStrsm
#define cusolverDnRgeqrf_bufferSize cusolverDnSgeqrf_bufferSize
#define cusolverDnRormqr_bufferSize cusolverDnSormqr_bufferSize
#define cusolverDnRgeqrf cusolverDnSgeqrf
#define cusolverDnRormqr cusolverDnSormqr
// Cholesky decomposition
#define cusolverDnRpotrf_bufferSize cusolverDnSpotrf_bufferSize
#define cusolverDnRpotrf cusolverDnSpotrf
#define cusolverDnRpotrs cusolverDnSpotrs

// Sparse QR decomposition
#define cusolverSpRcsrlsqvqrHost cusolverSpScsrlsqvqrHost

// Other sparse matrix functions
#define cusparseRcsrgemm2 cusparseScsrgemm2

#ifdef EIGEN_CORE_H
namespace Eigen {
typedef Vector3f Vector3r;
typedef MatrixXf MatrixXr;
typedef RowVectorXf RowVectorXr;
typedef VectorXf VectorXr;
typedef SparseMatrix<float, Eigen::ColMajor> SparseMatrixColXr;
typedef SparseMatrix<float, Eigen::RowMajor> SparseMatrixRowXr;
}
#endif

#endif

#endif //SH_AUG_DEFINES_HPP
