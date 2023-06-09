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

#ifndef CORRERENDER_CUDAFORWARD_HPP
#define CORRERENDER_CUDAFORWARD_HPP

// LU decomposition
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRgetrf_bufferSize(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSgetrf_bufferSize(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDgetrf_bufferSize(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRgetrf(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSgetrf(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDgetrf(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRgetrs(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSgetrs(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDgetrs(std::forward<Params>(params)...);
}

// QR decomposition
template <typename Real, typename ...Params>
cublasStatus_t cublasRgemm(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cublasSgemm(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cublasDgemm(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cublasStatus_t cublasRtrsm(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cublasStrsm(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cublasDtrsm(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRgeqrf_bufferSize(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSgeqrf_bufferSize(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDgeqrf_bufferSize(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRormqr_bufferSize(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSormqr_bufferSize(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDormqr_bufferSize(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRgeqrf(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSgeqrf(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDgeqrf(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRormqr(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSormqr(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDormqr(std::forward<Params>(params)...);
}

// Cholesky decomposition
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRpotrf_bufferSize(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSpotrf_bufferSize(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDpotrf_bufferSize(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRpotrf(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSpotrf(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDpotrf(std::forward<Params>(params)...);
}
template <typename Real, typename ...Params>
cusolverStatus_t cusolverDnRpotrs(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverDnSpotrs(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverDnDpotrs(std::forward<Params>(params)...);
}

// Sparse QR decomposition
template <typename Real, typename ...Params>
cusolverStatus_t cusolverSpRcsrlsqvqrHost(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusolverSpScsrlsqvqrHost(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusolverSpDcsrlsqvqrHost(std::forward<Params>(params)...);
}

// Other sparse matrix functions
template <typename Real, typename ...Params>
cusparseStatus_t cusparseRcsrgemm2(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return cusparseScsrgemm2(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return cusparseDcsrgemm2(std::forward<Params>(params)...);
}

#endif //CORRERENDER_CUDAFORWARD_HPP
