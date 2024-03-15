/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#ifndef CORRERENDER_CORRELATIONCACHE_HPP
#define CORRERENDER_CORRELATIONCACHE_HPP

#define CORRELATION_CACHE                                         \
    std::vector<float> X(cs);                                     \
    std::vector<float> Y(cs);                                     \
                                                                  \
    std::vector<std::pair<float, int>> ordinalRankArraySpearman;  \
    std::vector<float> referenceRanks;                            \
    std::vector<float> gridPointRanks;                            \
    if (cmt == CorrelationMeasureType::SPEARMAN)                  \
    {                                                             \
        ordinalRankArraySpearman.reserve(cs);                     \
        referenceRanks.resize(cs);                                \
        gridPointRanks.resize(cs);                                \
    }                                                             \
                                                                  \
    std::vector<std::pair<float, float>> jointArray;              \
    std::vector<float> ordinalRankArray;                          \
    std::vector<std::pair<float, int>> ordinalRankArrayRef;       \
    std::vector<float> y;                                         \
    std::vector<float> sortArray;                                 \
    std::vector<std::pair<int, int>> stack;                       \
    if (cmt == CorrelationMeasureType::KENDALL)                   \
    {                                                             \
        jointArray.reserve(cs);                                   \
        ordinalRankArray.reserve(cs);                             \
        ordinalRankArrayRef.reserve(cs);                          \
        y.reserve(cs);                                            \
        sortArray.reserve(cs);                                    \
    }                                                             \
                                                                  \
    std::vector<double> histogram0;                               \
    std::vector<double> histogram1;                               \
    std::vector<double> histogram2d;                              \
    if (isMeasureBinnedMI(cmt))                                   \
    {                                                             \
        histogram0.reserve(numBins);                              \
        histogram1.reserve(numBins);                              \
        histogram2d.reserve(numBins *numBins);                    \
    }                                                             \
                                                                  \
    KraskovEstimatorCache<double> kraskovEstimatorCache;

#endif //CORRERENDER_CORRELATIONCACHE_HPP
