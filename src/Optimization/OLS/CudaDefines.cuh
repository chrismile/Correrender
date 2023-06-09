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

#ifndef CORRERENDER_CUDADEFINES_CUH
#define CORRERENDER_CUDADEFINES_CUH

// See: https://stackoverflow.com/questions/17831218/how-can-i-use-templates-or-typedefs-to-select-between-float-double-vector-types

template <typename T, int cn> struct MakeVec;

template <> struct MakeVec<float, 2> {
    typedef float3 type;
};
template <> struct MakeVec<double, 2> {
    typedef double3 type;
};

template <> struct MakeVec<float, 3> {
    typedef float4 type;
};
template <> struct MakeVec<double, 3> {
    typedef double4 type;
};

template <> struct MakeVec<float, 4> {
    typedef float4 type;
};
template <> struct MakeVec<double, 4> {
    typedef double4 type;
};

/*template <typename Real, typename ...Params>
auto make_real4(Params&&... params) {
    if constexpr (std::is_same<Real, float>()) return make_float4(std::forward<Params>(params)...);
    if constexpr (std::is_same<Real, double>()) return make_double4(std::forward<Params>(params)...);
}*/

#endif //CORRERENDER_CUDADEFINES_CUH
