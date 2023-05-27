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

#ifndef SH_AUG_EIGENSOLVER_HPP
#define SH_AUG_EIGENSOLVER_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "PrecisionDefines.hpp"
#include "../OptDefines.hpp"

/**
 * Solves A*x = b for the vector x.
 * This solver uses the library Eigen. For more details see:
 * - https://eigen.tuxfamily.org/dox/group__LeastSquares.html
 * - https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
 * @param eigenSolverType The type of solver to use.
 * @param useRelaxation Whether to use relaxation/a regularizer.
 * @param lambdaL The relaxation/regularization factor.
 * @param A The system matrix A.
 * @param b The right-hand side vector.
 * @param x The left-hand side vector to solve for (output).
 */
void solveLeastSquaresEigenDense(
        EigenSolverType eigenSolverType, bool useRelaxation, const Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x);

/**
 * Solves A*x = b for the vector x for a symmetric matrix A.
 * This solver uses the library Eigen. For more details see:
 * - https://eigen.tuxfamily.org/dox/group__LeastSquares.html
 * - https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
 * @param eigenSolverType The type of solver to use.
 * @param lambdaL The relaxation/regularization factor.
 * @param A The system matrix A.
 * @param b The right-hand side vector.
 * @param x The left-hand side vector to solve for (output).
 */
void solveLinearSystemEigenSymmetric(
        EigenSolverType eigenSolverType, const Real lambdaL,
        const Eigen::MatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x);

/**
 * Solves A*x = b for the vector x.
 * This solver uses the library Eigen. For more details see:
 * - https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
 * @param solverType The type of solver to use.
 * @param lambdaL The relaxation/regularization factor (only for CGLS).
 * @param A The system matrix A.
 * @param b The right-hand side vector.
 * @param x The left-hand side vector to solve for (output).
 */
void solveLeastSquaresEigenSparse(
        EigenSparseSolverType solverType, const Real lambdaL,
        const Eigen::SparseMatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x);

/**
 * Solves A*x = b for the vector x using the normal equations, i.e., A^T A x = A^T b.
 * This solver uses the library Eigen. For more details see:
 * - https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
 * @param eigenSolverType The type of solver to use on A^T A.
 * @param lambdaL The relaxation/regularization factor.
 * @param A The system matrix A.
 * @param b The right-hand side vector.
 * @param x The left-hand side vector to solve for (output).
 */
void solveLeastSquaresEigenSparseNormalEquations(
        EigenSolverType eigenSolverType, const Real lambdaL,
        const Eigen::SparseMatrixXr& A, const Eigen::MatrixXr& b, Eigen::MatrixXr& x);

#endif //SH_AUG_EIGENSOLVER_HPP
