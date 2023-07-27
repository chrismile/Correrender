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

#ifndef CORRERENDER_CORRELATION_HPP
#define CORRERENDER_CORRELATION_HPP

#include <vector>
#include <cstdint>
#include <cstddef>

template<class T>
float computePearson1(
        const float* referenceValues, const std::vector<const float*>& ensembleFields, int es, size_t gridPointIdx);
extern template
float computePearson1<float>(
        const float* referenceValues, const std::vector<const float*>& ensembleFields, int es, size_t gridPointIdx);
extern template
float computePearson1<double>(
        const float* referenceValues, const std::vector<const float*>& ensembleFields, int es, size_t gridPointIdx);

template<class T>
float computePearson1(
        const float* referenceValues, const float* queryValues, int es);
extern template
float computePearson1<float>(
        const float* referenceValues, const float* queryValues, int es);
extern template
float computePearson1<double>(
        const float* referenceValues, const float* queryValues, int es);

template<class T>
float computePearson2(
        const float* referenceValues, const std::vector<const float*>& ensembleFields, int es, size_t gridPointIdx);
extern template
float computePearson2<float>(
        const float* referenceValues, const std::vector<const float*>& ensembleFields, int es, size_t gridPointIdx);
extern template
float computePearson2<double>(
        const float* referenceValues, const std::vector<const float*>& ensembleFields, int es, size_t gridPointIdx);

template<class T>
float computePearson2(
        const float* referenceValues, const float* queryValues, int es);
extern template
float computePearson2<float>(
        const float* referenceValues, const float* queryValues, int es);
extern template
float computePearson2<double>(
        const float* referenceValues, const float* queryValues, int es);

template<class T>
float computePearsonParallel(
        const float* referenceValues, const float* queryValues, int es);
extern template
float computePearsonParallel<float>(
        const float* referenceValues, const float* queryValues, int es);
extern template
float computePearsonParallel<double>(
        const float* referenceValues, const float* queryValues, int es);

void computeRanks(const float* values, float* ranks, std::vector<std::pair<float, int>>& ordinalRankArray, int es);

int computeTiesB(const float* values, std::vector<float>& ordinalRankArray, int es);

template<class IntType>
float computeKendall(
        const float* referenceValues, const float* queryValues, int es,
        std::vector<std::pair<float, float>>& jointArray, std::vector<float>& ordinalRankArray,
        std::vector<float>& y, std::vector<float>& sortArray, std::vector<std::pair<int, int>>& stack);
extern template
float computeKendall<int32_t>(
        const float* referenceValues, const float* queryValues, int es,
        std::vector<std::pair<float, float>>& jointArray, std::vector<float>& ordinalRankArray,
        std::vector<float>& y, std::vector<float>& sortArray, std::vector<std::pair<int, int>>& stack);
extern template
float computeKendall<int64_t>(
        const float* referenceValues, const float* queryValues, int es,
        std::vector<std::pair<float, float>>& jointArray, std::vector<float>& ordinalRankArray,
        std::vector<float>& y, std::vector<float>& sortArray, std::vector<std::pair<int, int>>& stack);

// O(n^2) implementation.
float computeKendallSlow(const float* referenceValues, const float* queryValues, int es);

#endif //CORRERENDER_CORRELATION_HPP
