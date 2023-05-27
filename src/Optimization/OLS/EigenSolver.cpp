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

#include "lsqr.hpp"
#include "cgls.hpp"
#include "EigenSolver.hpp"

void solveLeastSquaresEigenDense(
        EigenSolverType eigenSolverType, bool useRelaxation, const Real lambdaL,
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
        }
    } else {
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
        }
    }
}

void solveLinearSystemEigenSymmetric(
        EigenSolverType eigenSolverType, const Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    const auto c = A.cols();
    Eigen::MatrixXr M_I = Eigen::MatrixXr::Identity(c, c);

    Eigen::MatrixXr A_T = A.transpose();
    Eigen::MatrixXr lhs = A_T * A;
    if (lambdaL > Real(0)) {
        lhs += lambdaL * M_I;
    }
    Eigen::MatrixXr rhs = A_T * b;

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
    }
}

void solveLeastSquaresEigenSparse(
        EigenSparseSolverType solverType, const Real lambdaL,
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
        float tol = 1e-6f;
        int maxit = 100;
        float s = lambdaL;
        solveLeastSquaresEigenCGLS(A, b, x, s, tol, maxit, quiet);
    }
}

void solveLeastSquaresEigenSparseNormalEquations(
        EigenSolverType eigenSolverType, const Real lambdaL,
        const Eigen::SparseMatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x) {
    auto AT = A.adjoint();
    auto lhs = Eigen::MatrixXr(AT * A);
    auto rhs = Eigen::MatrixXr(AT * b);
    if (lambdaL > 0.0f) {
        const auto c = A.cols();
        Eigen::MatrixXr M_I = Eigen::MatrixXr::Identity(c, c);
        lhs += lambdaL * M_I;
    }
    x = lhs.householderQr().solve(rhs);
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
    }
}
