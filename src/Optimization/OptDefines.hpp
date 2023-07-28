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

#ifndef CORRERENDER_OPTDEFINES_HPP
#define CORRERENDER_OPTDEFINES_HPP

enum class TFOptimizerMethod {
    OLS, //< Ordinary least squares.
    GD, //< Gradient descent used for solving the ordinary least squares problem.
    DIFF_DVR //< Differentiable volume rendering.
};
const char* const TF_OPTIMIZER_METHOD_NAMES[] = {
        "Ordinary Least Squares (OLS)", "Gradient Descent", "Differentiable Volume Rendering"
};

// At the moment only supported in the OLS solver.
enum class FloatAccuracy {
    FLOAT, DOUBLE
};
const char* const FLOAT_ACCURACY_NAMES[] = {
        "Float (32-bit)", "Double (64-bit)"
};

/// For more details see: https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
enum class EigenSolverType {
    PartialPivLU, FullPivLU, HouseholderQR, ColPivHouseholderQR, FullPivHouseholderQR,
    CompleteOrthogonalDecomposition, LLT, LDLT, BDCSVD, JacobiSVD,
#ifdef SUPPORT_OSQP
    OSQP, // Uses OSQP, https://github.com/osqp/osqp
#endif
    QUADPROGPP, // Uses QuadProg++, https://github.com/osqp/osqp
    EIGEN_QP // Uses Eigen-QP, https://github.com/jarredbarber/eigen-QP/blob/master/eigen-qp.hpp
};
const char* const EIGEN_SOLVER_TYPE_NAMES[] = {
        "PartialPivLU", "FullPivLU", "HouseholderQR", "ColPivHouseholderQR", "FullPivHouseholderQR",
        "CompleteOrthogonalDecomposition", "LLT", "LDLT", "BDCSVD", "JacobiSVD",
#ifdef SUPPORT_OSQP
        "Quadratic Programming (OSQP)",
#endif
        "Quadratic Programming (QuadProg++)",
        "Quadratic Programming (Eigen-QP)"
};

/// For more details see: https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
enum class EigenSparseSolverType {
    QR, LEAST_SQUARES_CG, LEAST_SQUARES_CG_PRECONDITIONED, LSQR, CGLS
};
const char* const EIGEN_SPARSE_SOLVER_TYPE_NAMES[] = {
        "QR", "Least Squares CG", "Least Squares CG Preconditioned", "LSQR", "CGLS"
};

/// For more details see: https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
enum class CudaSolverType {
    LU, LU_PIVOT, QR, CHOL
};
const char* const CUDA_SOLVER_TYPE_NAMES[] = {
        "LU", "LU_PIVOT", "QR", "CHOL"
};

enum class CudaSparseSolverType {
    LSQR, CGLS, CUSOLVER_QR
};
const char* const CUDA_SPARSE_SOLVER_TYPE_NAMES[] = {
        "LSQR", "CGLS", "cuSOLVER QR"
};

// For gradient descent.
enum class OptimizerType {
    SGD, ADAM
};
const char* const OPTIMIZER_TYPE_NAMES[] = {
        "SGD", "Adam"
};
enum class LossType {
    L1, L2
};
const char* const LOSS_TYPE_NAMES[] = {
        "L1", "L2"
};

enum class OLSBackend {
    CPU, VULKAN, CUDA
};
const char* const OLS_BACKEND_NAMES[] = {
        "CPU", "Vulkan", "CUDA"
};

struct TFOptimizationWorkerSettings {
    int fieldIdxGT = -1;
    int fieldIdxOpt = -1;
    uint32_t tfSize = 0;
    TFOptimizerMethod optimizerMethod = TFOptimizerMethod::OLS;

    // For OLS.
    FloatAccuracy floatAccuracy = FloatAccuracy::FLOAT;
    EigenSolverType eigenSolverType = EigenSolverType::HouseholderQR;
    EigenSparseSolverType eigenSparseSolverType = EigenSparseSolverType::LEAST_SQUARES_CG_PRECONDITIONED;
    CudaSolverType cudaSolverType = CudaSolverType::QR;
    CudaSparseSolverType cudaSparseSolverType = CudaSparseSolverType::CGLS;
    bool useCudaMatrixSetup = true; //< Whether to set up the matrix on the GPU when using CUDA.
    OLSBackend backend = OLSBackend::VULKAN;
    bool useSparseSolve = true;
    bool useNormalEquations = true;
    float relaxationLambda = 1e-3f; //< Only used when useNormalEquations == true.

    // For DiffDVR and OLS_GRAD.
    OptimizerType optimizerType = OptimizerType::ADAM;
    LossType lossType = LossType::L2;
    int maxNumEpochs = 200;
    // SGD & Adam.
    float learningRate = 0.4f;
    // Adam.
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    // DVR.
    uint32_t imageWidth = 512;
    uint32_t imageHeight = 512;
    uint32_t batchSize = 8;
    float stepSize = 0.2f;
    float attenuationCoefficient = 10.0f;
    float lambdaSmoothingPrior = 0.4f;
    bool adjointDelayed = true;
};

#endif //CORRERENDER_OPTDEFINES_HPP
