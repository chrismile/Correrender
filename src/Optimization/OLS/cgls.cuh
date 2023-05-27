/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2015-2023, Christopher Fougner, Christoph Neuhauser
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

#ifndef CORRERENDER_CGLS_CUH
#define CORRERENDER_CGLS_CUH

//  CGLS Conjugate Gradient Least Squares
//  Attempts to solve the least squares problem
//
//    min. ||Ax - b||_2^2 + s ||x||_2^2
//
//  using the Conjugate Gradient for Least Squares method. This is more stable
//  than applying CG to the normal equations. Supports both generic operators
//  for computing Ax and A^Tx as well as a sparse matrix version.
//
//  ------------------------------ GENERIC  ------------------------------------
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  F          - Generic GEMV-like functor type with signature
//               int gemv(char op, T alpha, const T *x, T beta, T *y). Upon
//               exit, y should take on the value y := alpha*op(A)x + beta*y.
//               If successful the functor must return 0, otherwise a non-zero
//               value should be returned.
//
//  Function Arguments:
//  A          - Operator that computes Ax and A^Tx.
//
//  (m, n)     - Matrix dimensions of A.
//
//  b          - Pointer to right-hand-side vector.
//
//  x          - Pointer to solution. This vector will also be used as an
//               initial guess, so it must be initialized (eg. to 0).
//
//  shift      - Regularization parameter s. Solves (A'*A + shift*I)*x = A'*b.
//
//  tol        - Specifies tolerance (recommended 1e-6).
//
//  maxit      - Maximum number of iterations (recommended > 100).
//
//  quiet      - Disable printing to console.
//
//  ------------------------------ SPARSE --------------------------------------
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  O          - Sparse ordering (cgls::CSC or cgls::CSR).
//
//  Function Arguments:
//  val        - Array of matrix values. The array should be of length nnz.
//
//  ptr        - Column pointer if (O is CSC) or row pointer if (O is CSR).
//               The array should be of length m+1.
//
//  ind        - Row indices if (O is CSC) or column indices if (O is CSR).
//               The array should be of length nnz.
//
//  (m, n)     - Matrix dimensions of A.
//
//  nnz        - Number of non-zeros in A.
//
//  b          - Pointer to right-hand-side vector.
//
//  x          - Pointer to solution. This vector will also be used as an
//               initial guess, so it must be initialized (eg. to 0).
//
//  shift      - Regularization parameter s. Solves (A'*A + shift*I)*x = A'*b.
//
//  tol        - Specifies tolerance (recommended 1e-6).
//
//  maxit      - Maximum number of iterations (recommended > 100).
//
//  quiet      - Disable printing to console.
//
//  ----------------------------------------------------------------------------
//
//  Returns:
//  0 : CGLS converged to the desired tolerance tol within maxit iterations.
//  1 : The vector b had norm less than eps, solution likely x = 0.
//  2 : CGLS iterated maxit times, but did not converge.
//  3 : Matrix (A'*A + shift*I) seems to be singular or indefinite.
//  4 : Likely instable, (A'*A + shift*I) indefinite and norm(x) decreased.
//
//  Reference:
//  http://web.stanford.edu/group/SOL/software/cgls/
//

#include <algorithm>
#include <cstdio>

#include <cublas_v2.h>
#include <cusparse.h>

#include "cgls_common.hpp"

// Sparse CGLS (Conjugate Gradient Least Squares).
template<typename Real>
RetValCGLS solveLeastSquaresCudaCGLS(
        cudaStream_t stream, cublasHandle_t cublasHandle, cusparseHandle_t cusparseHandle,
        const Real* val, const int* ptr, const int* ind, const int m,
        const int n, const int nnz, const Real* b, Real* x, const double shift,
        const double tol, const int maxit, bool quiet) {
    SpMV<Real> A(stream, cusparseHandle, m, n, nnz, val, ptr, ind);

    Real* p, *q, *r, *s;
    double gamma, normp, normq, norms, norms0, normx, xmax;
    char fmt[] = "%5d %9.2e %12.5g\n";
    int loopIdx = 0;
    RetValCGLS flag = RetValCGLS::CONVERGED;
    bool indefinite = false;

    AuxBuffer auxBuffer;
    cudaMalloc(&p, n * sizeof(Real));
    cudaMalloc(&q, m * sizeof(Real));
    cudaMalloc(&r, m * sizeof(Real));
    cudaMalloc(&s, n * sizeof(Real));
    DnVector pDesc(p, n);
    DnVector qDesc(q, m);
    DnVector rDesc(r, m);
    DnVector sDesc(s, n);
    DnVector xDesc(x, n);

    cudaErrorCheck(cudaMemcpyAsync(r, b, m * sizeof(Real), cudaMemcpyDeviceToDevice, stream));
    cudaErrorCheck(cudaMemcpyAsync(s, x, n * sizeof(Real), cudaMemcpyDeviceToDevice, stream));

    // r = b - A*x.
    nrm2(cublasHandle, n, x, &normx);
    cudaErrorCheck(cudaStreamSynchronize(stream));
    if (normx > 0.0) {
        A(CUSPARSE_OPERATION_NON_TRANSPOSE, Real(-1), xDesc, Real(1), rDesc, auxBuffer);
    }

    // s = A'*r - shift*x.
    A(CUSPARSE_OPERATION_TRANSPOSE, Real(1), rDesc, Real(-shift), sDesc, auxBuffer);

    // Initialize.
    copy(cublasHandle, n, s, p);
    nrm2(cublasHandle, n, s, &norms);
    cudaErrorCheck(cudaStreamSynchronize(stream));
    norms0 = norms;
    gamma = norms0 * norms0;
    nrm2(cublasHandle, n, x, &normx);
    cudaErrorCheck(cudaStreamSynchronize(stream));
    xmax = normx;

    if (norms < Epsilon<Real>()) {
        flag = RetValCGLS::INITIAL_GUESS_IS_SOLUTION;
    }

    if (!quiet) {
        printf("    k     normx        resNE\n");
    }

    for (; loopIdx < maxit && flag == RetValCGLS::CONVERGED; ++loopIdx) {
        A(CUSPARSE_OPERATION_NON_TRANSPOSE, Real(1), pDesc, Real(0), qDesc, auxBuffer); // q = A * p

        // delta = norm(p)^2 + shift*norm(q)^2.
        nrm2(cublasHandle, n, p, &normp);
        nrm2(cublasHandle, m, q, &normq);
        cudaErrorCheck(cudaStreamSynchronize(stream));
        double delta = normq * normq + shift * normp * normp;

        if (delta <= 0.0) {
            indefinite = true;
        }
        if (delta == 0.0) {
            delta = Epsilon<Real>();
        }
        Real alpha = Real(gamma / delta);
        axpy(cublasHandle, n, alpha, p, x); // x = x + alpha * p
        axpy(cublasHandle, m, -alpha, q, r); // r = r - alpha * q

        // s = A^T * r - shift * x.
        copy(cublasHandle, n, x, s);
        A(CUSPARSE_OPERATION_TRANSPOSE, Real(1), rDesc, Real(-shift), sDesc, auxBuffer);

        // Compute beta.
        nrm2(cublasHandle, n, s, &norms);
        cudaErrorCheck(cudaStreamSynchronize(stream));
        double gamma1 = gamma;
        gamma = norms * norms;
        auto beta = Real(gamma / gamma1);

        // p = s + beta*p.
        axpy(cublasHandle, n, beta, p, s); // s = s + beta * p
        copy(cublasHandle, n, s, p); // p = s

        // Convergence check.
        nrm2(cublasHandle, n, x, &normx);
        cudaErrorCheck(cudaStreamSynchronize(stream));
        xmax = std::max(xmax, normx);
        bool converged = (norms <= norms0 * tol) || (normx * tol >= 1.0);
        if (!quiet && (converged || loopIdx % 10 == 0)) {
            printf(fmt, loopIdx, normx, norms / norms0);
        }
        if (converged) {
            break;
        }
    }

    // Determine exit status.
    double shrink = normx / xmax;
    if (loopIdx == maxit) {
        flag = RetValCGLS::REACHED_MAX_IT;
    } else if (indefinite) {
        flag = RetValCGLS::INDEFINITE;
    } else if (shrink * shrink <= tol) {
        flag = RetValCGLS::INSTABLE;
    }

    // Free variables and return.
    cudaErrorCheck(cudaFree(p));
    cudaErrorCheck(cudaFree(q));
    cudaErrorCheck(cudaFree(r));
    cudaErrorCheck(cudaFree(s));
    return RetValCGLS(flag);
}

#endif // CORRERENDER_CGLS_CUH
