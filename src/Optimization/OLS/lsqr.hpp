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

#ifndef CORRERENDER_LSQR_HPP
#define CORRERENDER_LSQR_HPP

#include <Eigen/SparseCore>
#include "PrecisionDefines.hpp"

/*
 * Algorithm from:
 * LSQR: An Algorithm for Sparse Linear Equations and Sparse Least Squares.
 * Christopher C. Paige, Michael A. Saunders. 1982.
 * ACM Transactions on Mathematical Software, Volume 8, Issue 1.
 */
template<class Real>
void solveLeastSquaresEigenLSQR(
        const Eigen::SparseMatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x, int maxIter) {
    constexpr auto eps = Real(std::is_same<Real, double>::value ? 1e-17 : 1e-7);
    auto atol = Real(1e-6);
    auto btol = Real(1e-6);
    //auto conlim = Real(1.0 / (10.0 * std::sqrt(eps)));
    auto conlim = Real(1.0 / eps);

    auto AT = A.transpose();

    // (1) Initialization.
    Real beta = b.norm();
    Real normb = beta;
    Eigen::VectorXr u = b / beta;
    Eigen::VectorXr v = AT * u;
    Real alpha = v.norm();
    v /= alpha;
    Eigen::VectorXr w = v;
    x = Eigen::VectorXr(int(A.cols()));
    x.setZero();
    Real phiBar = beta;
    Real rhoBar = alpha;

    // (2) Loop.
    auto normBkSum = std::sqrt(alpha * alpha + beta * beta);
    auto normdd = Real(0);
    auto cs2 = Real(-1);
    auto sn2 = Real(0);
    auto normxx = Real(0);
    auto z = Real(0);
    int loopIdx = 0;
    while (loopIdx < maxIter) {
        loopIdx++;

        // (3) Continue the bidiagonalization.
        u = A * v - alpha * u;
        beta = u.norm();
        u /= beta;
        v = AT * u - beta * v;
        alpha = v.norm();
        v /= alpha;

        // (4) Construct and apply the next orthogonal transformation
        Real rho = std::sqrt(rhoBar * rhoBar + beta * beta);
        Real c = rhoBar / rho;
        Real s = beta / rho;
        Real theta = s * alpha;
        rhoBar = -c * alpha;
        Real phi = c * phiBar;
        phiBar = s * phiBar;

        // (5) Update x, w.
        x = x + (phi / rho) * w;
        w = v - (theta / rho) * w;

        // (6) Test for convergence
        Real normrk = phiBar; // Eq. (5.2).
        Real normATrk = phiBar * alpha * std::abs(c); // Eq. (5.4).
        //normATrk = alpha * std::abs(s * phi); // alternative formulation; difference?
        normBkSum = std::sqrt(normBkSum * normBkSum + alpha * alpha + beta * beta); // Sec. 5.3, sec. 3.
        Real normA = normBkSum;
        // Eq. (5.5) - (5.7).
        // Computation of z from: https://github.com/harusametime/LSQRwithEigen/blob/master/LSQR.cpp
        Real gammaBar = -cs2 * rho;
        Real gammaRhs = (phi - sn2 * rho * z);
        Real zBar = gammaRhs / gammaBar;
        Real normxk = std::sqrt(normxx + zBar * zBar);
        Real gamma = std::sqrt(gammaBar * gammaBar + theta * theta);
        cs2 = gammaBar / gamma;
        sn2 = theta / gamma;
        z = gammaRhs / gamma;
        normxx += z * z;

        // S1.
        if (normrk <= btol * normb + atol * normA * normxk) {
            break;
        }

        // S2.
        if (normATrk / (normA * normrk) <= atol) {
            break;
        }

        // S3.
        auto rhow = (Real(1) / rho) * w;
        normdd += rhow.squaredNorm();
        Real condA = normA + std::sqrt(normdd);
        if (condA >= conlim) {
            break;
        }
    }
}

#endif //CORRERENDER_LSQR_HPP
