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

#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SparseQR>
#include <Eigen/IterativeLinearSolvers>

#include <Utils/File/Logfile.hpp>

#ifdef SUPPORT_OSQP
#include "osqp.h"
#endif

#include "lsqr.hpp"
#include "cgls.hpp"
#include "eigen-qp.hpp"
#include "QuadProgpp/QuadProg++.hh"
#include "EigenSolver.hpp"

/**
 * Uses the QuadProg++ library for quadratic programming.
 * Link: https://github.com/jarredbarber/eigen-QP
 * Solves: min_x 1/2 x^T Q x + c^T x, constraintMat x <= constraintVec
 * For the constraint, we want 0 <= x_i <= 1. This can be expressed as x_i <= 1 and -x_i <= 0.
 * @param lhs The matrix Q, Q = A^T A.
 * @param rhs The vector c, c = A^T b.
 * @param x The solution vector.
 */
template<class Real>
void solveQuadprogBoxConstrainedEigenQP(const Eigen::MatrixXr& lhs, const Eigen::MatrixXr& rhs, Eigen::MatrixXr& x) {
    const auto n = int(lhs.cols());
    Eigen::MatrixXr Q = lhs;
    Eigen::VectorXr c = -rhs;
    Eigen::MatrixXr constraintMat(2 * n, n);
    auto I_n = Eigen::MatrixXr::Identity(n, n);
    constraintMat << I_n, -I_n;
    Eigen::VectorXr constraintVec(2 * n);
    constraintVec << Eigen::VectorXr::Ones(n), Eigen::VectorXr::Zero(n);
    Eigen::VectorXr xvec(n);
    EigenQP::quadprog<Real, -1, -1>(Q, c, constraintMat, constraintVec, xvec);
    x = xvec;
}


/**
 * Uses the QuadProg++ library for quadratic programming.
 * Link: https://github.com/liuq/QuadProgpp
 * Solves: min_x 1/2 x^T G x + g0^T x, l x <= u
 * As the constraint, we have 0 <= x_i <= 1.
 * @param lhs The matrix P, P = A^T A.
 * @param rhs The vector q, q = A^T b.
 * @param x The solution vector.
 */
template<class Real>
void solveQuadprogBoxConstrainedQuadProgpp(const Eigen::MatrixXr& lhs, const Eigen::MatrixXr& rhs, Eigen::MatrixXr& x) {
    const auto n = int(lhs.cols());
    x = Eigen::MatrixXr(n, 1);

    quadprogpp::Matrix<double> G, CE, CI;
    quadprogpp::Vector<double> g0, ce0, ci0, xsol;

    G.resize(n, n);
    g0.resize(n);
    xsol.resize(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            G[i][j] = double(lhs(i, j));
        }
    }
    for (int i = 0; i < n; i++) {
        g0[i] = double(-rhs(i));
    }

    CE.resize(n, 0);
    CI.resize(n, 2 * n);
    ce0.resize(0);
    ci0.resize(2 * n);
    /*for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            CE[i][j] = double(0);
        }
    }
    for (int i = 0; i < n; i++) {
        ce0[i] = double(0);
    }*/
    int twoN = 2 * n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < twoN; j++) {
            double val = 0.0;
            if (i == j) {
                val = 1.0;
            } else if (i + n == j) {
                val = -1.0;
            }
            CI[i][j] = val;
        }
    }
    for (int i = 0; i < n; i++) {
        ci0[i] = 0.0;
        ci0[n + i] = 1.0;
    }

    auto f_value = quadprogpp::solve_quadprog(G, g0, CE, ce0, CI, ci0, xsol);
    if (std::isnan(f_value)) {
        sgl::Logfile::get()->writeError("Error in quadprogpp::solve_quadprog: Cholesky decomposition failed!");
    }
    for (int i = 0; i < n; i++) {
        x(i) = Real(xsol[i]);
    }
}

#ifdef SUPPORT_OSQP
/**
 * Uses the OSQP library for quadratic programming.
 * Link: https://github.com/osqp/osqp
 * Solves: min_x 1/2 x^T P x + q^T x, l x <= u
 * As the constraint, we have 0 <= x_i <= 1.
 * @param lhs The matrix P, P = A^T A.
 * @param rhs The vector q, q = A^T b.
 * @param x The solution vector.
 */
template<class Real>
void solveQuadprogBoxConstrainedOSQP(const Eigen::MatrixXr& lhs, const Eigen::MatrixXr& rhs, Eigen::MatrixXr& x) {
    const auto n = int(lhs.cols());
    x = Eigen::MatrixXr(n, 1);

    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.polishing = 1;

    // Build an identity matrix for the constraints.
    auto* constrMatCscVals = new OSQPFloat[n];
    auto* constrMatCscColPtr = new OSQPInt[n + 1];
    auto* constrMatCscRowInd = new OSQPInt[n];
    for (int i = 0; i < n; i++) {
        constrMatCscVals[i] = OSQPFloat(1);
        constrMatCscColPtr[i] = OSQPInt(i);
        constrMatCscRowInd[i] = OSQPInt(i);
    }
    constrMatCscColPtr[n] = OSQPInt(n);
    OSQPCscMatrix constrMat;
    csc_set_data(&constrMat, n, n, n, constrMatCscVals, constrMatCscRowInd, constrMatCscColPtr);

    // Convert the three (two in upper triangular part) non-zero diagonals to the CSC format.
    const int maxNnz = 2 * n - 4; // 3 * n - 8 for complete matrix, not just upper triangular part.
    std::vector<OSQPFloat> pCscVals;
    pCscVals.reserve(maxNnz);
    std::vector<OSQPInt> pCscColPtr;
    pCscColPtr.reserve(n + 1);
    std::vector<OSQPInt> pCscRowInd;
    pCscRowInd.reserve(maxNnz);
    for (int colIdx = 0; colIdx < n; colIdx++) {
        pCscColPtr.push_back(OSQPInt(pCscVals.size()));
        if (colIdx >= 4) {
            pCscVals.push_back(lhs(colIdx - 4, colIdx));
            pCscRowInd.push_back(colIdx - 4);
        }
        pCscVals.push_back(lhs(colIdx, colIdx));
        pCscRowInd.push_back(colIdx);
        // Only upper triangular part.
    }
    auto nnz = OSQPInt(pCscVals.size());
    pCscColPtr.push_back(nnz);
    OSQPCscMatrix P;
    csc_set_data(&P, n, n, nnz, pCscVals.data(), pCscRowInd.data(), pCscColPtr.data());

    // Lower and upper constraints.
    auto* l = new OSQPFloat[n], *u = new OSQPFloat[n];
    for (int i = 0; i < n; i++) {
        l[i] = OSQPFloat(0);
        u[i] = OSQPFloat(1);
    }

    auto* q = new OSQPFloat[n];
    for (int i = 0; i < n; i++) {
        q[i] = OSQPFloat(-rhs(i));
    }

    OSQPSolver* solver = nullptr;
    OSQPInt errorCode = osqp_setup(&solver, &P, q, &constrMat, l, u, n, n, &settings);
    if (errorCode == 0) {
        errorCode = osqp_solve(solver);
        if (errorCode == 0) {
            for (int i = 0; i < n; i++) {
                x(i) = Real(solver->solution->x[i]);
            }
        } else {
            sgl::Logfile::get()->writeError(
                    std::string() + "Error in solver OSQP: osqp_solve failed: " + solver->info->status);
        }
        osqp_cleanup(solver);
    } else {
        sgl::Logfile::get()->writeError("Error in solver OSQP: osqp_setup failed.");
        return;
    }

    delete[] constrMatCscVals;
    delete[] constrMatCscColPtr;
    delete[] constrMatCscRowInd;
    delete[] l;
    delete[] u;
    delete[] q;
}
#endif

#define TEST_CONDITION_NUMBER

#ifdef TEST_CONDITION_NUMBER
template<class Real>
void printConditionNumber(const std::string& outputPrefix, const Eigen::MatrixXr& M) {
    Eigen::JacobiSVD<Eigen::MatrixXr> svd(M);
    const auto& singularValues = svd.singularValues();
    Real cond = singularValues(0) / singularValues(singularValues.size() - 1);
    std::cout << outputPrefix << cond << std::endl;
}
#endif



template<class Real>
void solveLeastSquaresEigenDense(
        EigenSolverType eigenSolverType, bool useRelaxation, Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    if (useRelaxation) {
        const auto c = A.cols();
        Eigen::MatrixXr M_I = Eigen::MatrixXr::Identity(c, c);

        Eigen::MatrixXr A_T = A.transpose();
        Eigen::MatrixXr lhs = A_T * A;
        if (lambdaL > Real(0)) {
            lhs += lambdaL * M_I;
        }
        Eigen::MatrixXr rhs = A_T * b;

#ifdef TEST_CONDITION_NUMBER
        printConditionNumber("Condition number A^T A: ", lhs);
#endif

        switch(eigenSolverType) {
            case EigenSolverType::PartialPivLU:
                x = lhs.partialPivLu().solve(rhs);
                break;
            case EigenSolverType::FullPivLU:
                x = lhs.fullPivLu().solve(rhs);
                break;
            case EigenSolverType::HouseholderQR:
                x = lhs.householderQr().solve(rhs);
                break;
            case EigenSolverType::ColPivHouseholderQR:
                x = lhs.colPivHouseholderQr().solve(rhs);
                break;
            case EigenSolverType::FullPivHouseholderQR:
                x = lhs.fullPivHouseholderQr().solve(rhs);
                break;
            case EigenSolverType::CompleteOrthogonalDecomposition:
                x = lhs.completeOrthogonalDecomposition().solve(rhs);
                break;
            case EigenSolverType::LLT:
                x = lhs.llt().solve(rhs);
                break;
            case EigenSolverType::LDLT:
                x = lhs.ldlt().solve(rhs);
                break;
            case EigenSolverType::BDCSVD:
                x = lhs.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
                break;
            case EigenSolverType::JacobiSVD:
                x = lhs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
                break;
#ifdef SUPPORT_OSQP
            case EigenSolverType::OSQP:
                solveQuadprogBoxConstrainedOSQP(lhs, rhs, x);
                break;
#endif
            case EigenSolverType::QUADPROGPP:
                solveQuadprogBoxConstrainedQuadProgpp(lhs, rhs, x);
                break;
            case EigenSolverType::EIGEN_QP:
                solveQuadprogBoxConstrainedEigenQP(lhs, rhs, x);
                break;
        }
    } else {
#ifdef TEST_CONDITION_NUMBER
        printConditionNumber("Condition number A: ", A);
#endif
        switch(eigenSolverType) {
            case EigenSolverType::PartialPivLU:
                x = A.partialPivLu().solve(b);
                break;
            case EigenSolverType::FullPivLU:
                x = A.fullPivLu().solve(b);
                break;
            case EigenSolverType::HouseholderQR:
                x = A.householderQr().solve(b);
                break;
            case EigenSolverType::ColPivHouseholderQR:
                x = A.colPivHouseholderQr().solve(b);
                break;
            case EigenSolverType::FullPivHouseholderQR:
                x = A.fullPivHouseholderQr().solve(b);
                break;
            case EigenSolverType::CompleteOrthogonalDecomposition:
                x = A.completeOrthogonalDecomposition().solve(b);
                break;
            case EigenSolverType::LLT:
                x = A.llt().solve(b);
                break;
            case EigenSolverType::LDLT:
                x = A.ldlt().solve(b);
                break;
            case EigenSolverType::BDCSVD:
                x = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                break;
            case EigenSolverType::JacobiSVD:
                x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
                break;
#ifdef SUPPORT_OSQP
            case EigenSolverType::OSQP:
#endif
            case EigenSolverType::QUADPROGPP:
            case EigenSolverType::EIGEN_QP:
                Eigen::MatrixXr A_T = A.transpose();
                Eigen::MatrixXr lhs = A_T * A;
                Eigen::MatrixXr rhs = A_T * b;
                if (eigenSolverType == EigenSolverType::EIGEN_QP) {
                    solveQuadprogBoxConstrainedEigenQP(lhs, rhs, x);
                } else if (eigenSolverType == EigenSolverType::QUADPROGPP) {
                    solveQuadprogBoxConstrainedQuadProgpp(lhs, rhs, x);
                }
#ifdef SUPPORT_OSQP
                else if (eigenSolverType == EigenSolverType::OSQP) {
                    solveQuadprogBoxConstrainedOSQP(lhs, rhs, x);
                }
#endif
                break;
        }
    }
}
template
void solveLeastSquaresEigenDense<float>(
        EigenSolverType eigenSolverType, bool useRelaxation, float lambdaL,
        const Eigen::MatrixXf& A, const Eigen::MatrixXf& b, Eigen::MatrixXf& x);
template
void solveLeastSquaresEigenDense<double>(
        EigenSolverType eigenSolverType, bool useRelaxation, double lambdaL,
        const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, Eigen::MatrixXd& x);



template<class Real>
void solveLinearSystemEigenSymmetric(
        EigenSolverType eigenSolverType, Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    const auto c = A.cols();
    Eigen::MatrixXr M_I = Eigen::MatrixXr::Identity(c, c);

    Eigen::MatrixXr A_T = A.transpose();
    Eigen::MatrixXr lhs = A_T * A;
    if (lambdaL > Real(0)) {
        lhs += lambdaL * M_I;
    }
    Eigen::MatrixXr rhs = A_T * b;

#ifdef TEST_CONDITION_NUMBER
    printConditionNumber("Condition number A^T A: ", lhs);
#endif

    switch(eigenSolverType) {
        case EigenSolverType::PartialPivLU:
            x = lhs.partialPivLu().solve(rhs);
            break;
        case EigenSolverType::FullPivLU:
            x = lhs.fullPivLu().solve(rhs);
            break;
        case EigenSolverType::HouseholderQR:
            x = lhs.householderQr().solve(rhs);
            break;
        case EigenSolverType::ColPivHouseholderQR:
            x = lhs.colPivHouseholderQr().solve(rhs);
            break;
        case EigenSolverType::FullPivHouseholderQR:
            x = lhs.fullPivHouseholderQr().solve(rhs);
            break;
        case EigenSolverType::CompleteOrthogonalDecomposition:
            x = lhs.completeOrthogonalDecomposition().solve(rhs);
            break;
        case EigenSolverType::LLT:
            x = lhs.llt().solve(rhs);
            break;
        case EigenSolverType::LDLT:
            x = lhs.ldlt().solve(rhs);
            break;
        case EigenSolverType::BDCSVD:
            x = lhs.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
            break;
        case EigenSolverType::JacobiSVD:
            x = lhs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
            break;
#ifdef SUPPORT_OSQP
        case EigenSolverType::OSQP:
            solveQuadprogBoxConstrainedOSQP(lhs, rhs, x);
            break;
#endif
        case EigenSolverType::QUADPROGPP:
            solveQuadprogBoxConstrainedQuadProgpp(lhs, rhs, x);
            break;
        case EigenSolverType::EIGEN_QP:
            solveQuadprogBoxConstrainedEigenQP(lhs, rhs, x);
            break;
    }
}
template
void solveLinearSystemEigenSymmetric<float>(
        EigenSolverType eigenSolverType, float lambdaL,
        const Eigen::MatrixXf& A, const Eigen::MatrixXf& b, Eigen::MatrixXf& x);
template
void solveLinearSystemEigenSymmetric<double>(
        EigenSolverType eigenSolverType, double lambdaL,
        const Eigen::MatrixXd& A, const Eigen::MatrixXd& b, Eigen::MatrixXd& x);



template<class Real>
void solveLeastSquaresEigenSparse(
        EigenSparseSolverType solverType, Real lambdaL,
        const Eigen::SparseMatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    if (solverType == EigenSparseSolverType::QR) {
        //Eigen::SparseQR<Eigen::SparseMatrixXr, Eigen::COLAMDOrdering<int>> sparseQr;
        //Eigen::SparseQR<Eigen::SparseMatrixXr, Eigen::NaturalOrdering<int>> sparseQr;
        Eigen::SparseQR<Eigen::SparseMatrixXr, Eigen::AMDOrdering<int>> sparseQr;
        sparseQr.compute(A);
        if (sparseQr.info() != Eigen::Success) {
            sgl::Logfile::get()->writeError("Error in solveLeastSquaresEigenSparse: QR decomposition failed!");
            return;
        }
        x = sparseQr.solve(b);
        if (sparseQr.info() != Eigen::Success) {
            sgl::Logfile::get()->writeError("Error in solveLeastSquaresEigenSparse: QR solver failed!");
            return;
        }
    } else if (solverType == EigenSparseSolverType::LEAST_SQUARES_CG) {
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrixXr, Eigen::IdentityPreconditioner> cgSolver;
        cgSolver.compute(A);
        if (cgSolver.info() != Eigen::Success) {
            sgl::Logfile::get()->writeError("Error in solveLeastSquaresEigenSparse: CG decomposition failed!");
            return;
        }
        x = cgSolver.solve(b);
        if (cgSolver.info() != Eigen::Success) {
            sgl::Logfile::get()->writeError("Error in solveLeastSquaresEigenSparse: CG solver failed!");
            return;
        }
        std::cout << "CG solver iterations: " << cgSolver.iterations() << std::endl;
        std::cout << "CG solver error: " << cgSolver.error() << std::endl;
    } else if (solverType == EigenSparseSolverType::LEAST_SQUARES_CG_PRECONDITIONED) {
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrixXr, Eigen::LeastSquareDiagonalPreconditioner<Real>> cgSolver;
        cgSolver.compute(A);
        if (cgSolver.info() != Eigen::Success) {
            sgl::Logfile::get()->writeError("Error in solveLeastSquaresEigenSparse: CG decomposition failed!");
            return;
        }
        x = cgSolver.solve(b);
        if (cgSolver.info() != Eigen::Success) {
            sgl::Logfile::get()->writeError("Error in solveLeastSquaresEigenSparse: CG solver failed!");
            return;
        }
        std::cout << "CG solver iterations: " << cgSolver.iterations() << std::endl;
        std::cout << "CG solver error: " << cgSolver.error() << std::endl;
    } else if (solverType == EigenSparseSolverType::LSQR) {
        solveLeastSquaresEigenLSQR(A, b, x, 100);
    } else if (solverType == EigenSparseSolverType::CGLS) {
        bool quiet = false;
        int maxit = 100;
        double tol = 1e-6;
        auto s = double(lambdaL);
        solveLeastSquaresEigenCGLS(A, b, x, s, tol, maxit, quiet);
    }
}
template
void solveLeastSquaresEigenSparse<float>(
        EigenSparseSolverType solverType, float lambdaL,
        const Eigen::SparseMatrixXf& A, const Eigen::MatrixXf& b, Eigen::MatrixXf& x);
template
void solveLeastSquaresEigenSparse<double>(
        EigenSparseSolverType solverType, double lambdaL,
        const Eigen::SparseMatrixXd& A, const Eigen::MatrixXd& b, Eigen::MatrixXd& x);



template<class Real>
void solveLeastSquaresEigenSparseNormalEquations(
        EigenSolverType eigenSolverType, Real lambdaL,
        const Eigen::SparseMatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    auto AT = A.adjoint();
    auto lhs = Eigen::MatrixXr(AT * A);
    auto rhs = Eigen::MatrixXr(AT * b);
    if (lambdaL > 0.0f) {
        const auto c = A.cols();
        Eigen::MatrixXr M_I = Eigen::MatrixXr::Identity(c, c);
        lhs += lambdaL * M_I;
    }
#ifdef TEST_CONDITION_NUMBER
    printConditionNumber("Condition number A^T A: ", lhs);
#endif
    switch(eigenSolverType) {
        case EigenSolverType::PartialPivLU:
            x = lhs.partialPivLu().solve(rhs);
            break;
        case EigenSolverType::FullPivLU:
            x = lhs.fullPivLu().solve(rhs);
            break;
        case EigenSolverType::HouseholderQR:
            x = lhs.householderQr().solve(rhs);
            break;
        case EigenSolverType::ColPivHouseholderQR:
            x = lhs.colPivHouseholderQr().solve(rhs);
            break;
        case EigenSolverType::FullPivHouseholderQR:
            x = lhs.fullPivHouseholderQr().solve(rhs);
            break;
        case EigenSolverType::CompleteOrthogonalDecomposition:
            x = lhs.completeOrthogonalDecomposition().solve(rhs);
            break;
        case EigenSolverType::LLT:
            x = lhs.llt().solve(rhs);
            break;
        case EigenSolverType::LDLT:
            x = lhs.ldlt().solve(rhs);
            break;
        case EigenSolverType::BDCSVD:
            x = lhs.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
            break;
        case EigenSolverType::JacobiSVD:
            x = lhs.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rhs);
            break;
#ifdef SUPPORT_OSQP
        case EigenSolverType::OSQP:
            solveQuadprogBoxConstrainedOSQP(lhs, rhs, x);
            break;
#endif
        case EigenSolverType::QUADPROGPP:
            solveQuadprogBoxConstrainedQuadProgpp(lhs, rhs, x);
            break;
        case EigenSolverType::EIGEN_QP:
            solveQuadprogBoxConstrainedEigenQP(lhs, rhs, x);
            break;
    }
}
template
void solveLeastSquaresEigenSparseNormalEquations<float>(
        EigenSolverType eigenSolverType, float lambdaL,
        const Eigen::SparseMatrixXf& A, const Eigen::MatrixXf& b, Eigen::MatrixXf& x);
template
void solveLeastSquaresEigenSparseNormalEquations<double>(
        EigenSolverType eigenSolverType, double lambdaL,
        const Eigen::SparseMatrixXd& A, const Eigen::MatrixXd& b, Eigen::MatrixXd& x);
