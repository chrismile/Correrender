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

#ifndef CORRERENDER_CGLS_HPP
#define CORRERENDER_CGLS_HPP

// Eigen/C++ port of the code in cgls.cuh.

#include <Eigen/SparseCore>
#include "PrecisionDefines.hpp"
#include "cgls_common.hpp"

// Sparse CGLS (Conjugate Gradient Least Squares).
template<class Real>
RetValCGLS solveLeastSquaresEigenCGLS(
        const Eigen::SparseMatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x,
        const double shift, const double tol, const int maxit, bool quiet) {
    auto m = int(A.rows());
    auto n = int(A.cols());
    auto AT = A.transpose();

    double gamma, normp, normq, norms, norms0, normx, xmax;
    char fmt[] = "%5d %9.2e %12.5g\n";
    int loopIdx = 0;
    RetValCGLS flag = RetValCGLS::CONVERGED;
    bool indefinite = false;

    x = Eigen::VectorXr(n);
    x.setZero();
    auto p = Eigen::VectorXr(n);
    auto q = Eigen::VectorXr(m);
    auto r = Eigen::VectorXr(m);
    auto s = Eigen::VectorXr(n);

    memcpy(r.data(), b.data(), m * sizeof(Real));

    // r = b - A*x.
    normx = x.norm();
    if (normx > 0.0) {
        r = b - A * x;
    }

    s = AT * r - shift * x;

    // Initialize.
    p = s;
    norms = s.norm();
    norms0 = norms;
    gamma = norms0 * norms0;
    normx = x.norm();
    xmax = normx;

    if (norms < Epsilon<Real>()) {
        flag = RetValCGLS::INITIAL_GUESS_IS_SOLUTION;
    }

    if (!quiet) {
        printf("    k     normx        resNE\n");
    }

    for (; loopIdx < maxit && flag == RetValCGLS::CONVERGED; ++loopIdx) {
        q = A * p;

        // delta = norm(p)^2 + shift*norm(q)^2.
        normp = p.norm();
        normq = q.norm();
        double delta = normq * normq + shift * normp * normp;

        if (delta <= 0.0) {
            indefinite = true;
        }
        if (delta == 0.0) {
            delta = Epsilon<Real>();
        }
        Real alpha = Real(gamma / delta);
        x = x + alpha * p;
        r = r - alpha * q;

        s = AT * r - shift * x;

        // Compute beta.
        norms = s.norm();
        double gamma1 = gamma;
        gamma = norms * norms;
        auto beta = Real(gamma / gamma1);

        p = s + beta * p;

        // Convergence check.
        normx = x.norm();
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

    return RetValCGLS(flag);
}

#endif //CORRERENDER_CGLS_HPP
