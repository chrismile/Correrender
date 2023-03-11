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

#include <algorithm>
#include <cmath>

#include "Correlation.hpp"

template<class T>
float computePearson1(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx) {
    auto n = T(es);
    T sumX = 0;
    T sumY = 0;
    T sumXY = 0;
    T sumXX = 0;
    T sumYY = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
        sumYY += y * y;
    }
    float pearsonCorrelation =
            (n * sumXY - sumX * sumY) / std::sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    return (float)pearsonCorrelation;
}
template
float computePearson1<float>(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx);
template
float computePearson1<double>(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx);

template<class T>
float computePearson1(
        const float* referenceValues, const float* queryValues, int es) {
    auto n = T(es);
    T sumX = 0;
    T sumY = 0;
    T sumXY = 0;
    T sumXX = 0;
    T sumYY = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
        sumYY += y * y;
    }
    float pearsonCorrelation =
            (n * sumXY - sumX * sumY) / std::sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    return (float)pearsonCorrelation;
}
template
float computePearson1<float>(
        const float* referenceValues, const float* queryValues, int es);
template
float computePearson1<double>(
        const float* referenceValues, const float* queryValues, int es);

template<class T>
float computePearson2(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx) {
    auto n = T(es);
    T meanX = 0;
    T meanY = 0;
    T invN = T(1) / n;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        meanX += invN * x;
        meanY += invN * y;
    }
    T varX = 0;
    T varY = 0;
    T invNm1 = T(1) / (n - T(1));
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        T diffX = x - meanX;
        T diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    T stdDevX = std::sqrt(varX);
    T stdDevY = std::sqrt(varY);
    T pearsonCorrelation = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }
    return (float)pearsonCorrelation;
}
template
float computePearson2<float>(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx);
template
float computePearson2<double>(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx);

template<class T>
float computePearson2(
        const float* referenceValues, const float* queryValues, int es) {
    auto n = T(es);
    T meanX = 0;
    T meanY = 0;
    T invN = T(1) / n;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        meanX += invN * x;
        meanY += invN * y;
    }
    T varX = 0;
    T varY = 0;
    T invNm1 = T(1) / (n - T(1));
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        T diffX = x - meanX;
        T diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    T stdDevX = std::sqrt(varX);
    T stdDevY = std::sqrt(varY);
    T pearsonCorrelation = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)queryValues[e];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }
    return (float)pearsonCorrelation;
}
template
float computePearson2<float>(
        const float* referenceValues, const float* queryValues, int es);
template
float computePearson2<double>(
        const float* referenceValues, const float* queryValues, int es);

void computeRanks(const float* values, float* ranks, std::vector<std::pair<float, int>>& ordinalRankArray, int es) {
    ordinalRankArray.clear();
    for (int i = 0; i < es; i++) {
        ordinalRankArray.emplace_back(values[i], i);
    }
    std::sort(ordinalRankArray.begin(), ordinalRankArray.end());

    // Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    float currentRank = 1.0f;
    int idx = 0;
    while (idx < es) {
        float value = ordinalRankArray.at(idx).first;
        int idxEqualEnd = idx + 1;
        while (idxEqualEnd < es && value == ordinalRankArray.at(idxEqualEnd).first) {
            idxEqualEnd++;
        }

        int numEqualValues = idxEqualEnd - idx;
        float meanRank = currentRank + float(numEqualValues - 1) * 0.5f;
        for (int offset = 0; offset < numEqualValues; offset++) {
            ranks[ordinalRankArray.at(idx + offset).second] = meanRank;
        }

        idx += numEqualValues;
        currentRank += float(numEqualValues);
    }
}

int computeTiesB(const float* values, std::vector<float>& ordinalRankArray, int es) {
    ordinalRankArray.clear();
    for (int i = 0; i < es; i++) {
        ordinalRankArray.emplace_back(values[i]);
    }
    std::sort(ordinalRankArray.begin(), ordinalRankArray.end());

    // Compute fractional ranking for ordinal ranking (see https://en.wikipedia.org/wiki/Ranking).
    int tiesB = 0;
    int idx = 0;
    while (idx < es) {
        float value = ordinalRankArray.at(idx);
        int idxEqualEnd = idx + 1;
        while (idxEqualEnd < es && value == ordinalRankArray.at(idxEqualEnd)) {
            idxEqualEnd++;
        }

        int numEqualValues = idxEqualEnd - idx;
        tiesB += numEqualValues * (numEqualValues - 1) / 2;
        idx += numEqualValues;
    }

    return tiesB;
}

// https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
int M(const std::vector<float>& L, const std::vector<float>& R) {
    int n = int(L.size());
    int m = int(R.size());
    int i = 0;
    int j = 0;
    int numSwaps = 0;
    while (i < n && j < m) {
        if (R[j] < L[i]) {
            numSwaps += n - i;
            j += 1;
        } else {
            i += 1;
        }
    }
    return numSwaps;
}

// https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
int S(const std::vector<float>& y) {
    int n = int(y.size());
    if (n <= 1) {
        return 0;
    }
    int s = n / 2;
    std::vector<float> y_l = std::vector<float>(y.begin(), y.begin() + s);
    std::vector<float> y_r = std::vector<float>(y.begin() + s, y.end());
    int S_y_l = S(y_l);
    int S_y_r = S(y_r);
    std::sort(y_l.begin(), y_l.end());
    std::sort(y_r.begin(), y_r.end());
    return S_y_l + S_y_r + M(y_l, y_r);
}

float computeKendall(
        const float* referenceValues, const float* queryValues, int es,
        std::vector<std::pair<float, float>>& jointArray, std::vector<float>& ordinalRankArray,
        std::vector<float>& y) {
    int n = es;
    for (int i = 0; i < es; i++) {
        jointArray.emplace_back(referenceValues[i], queryValues[i]);
    }
    std::sort(jointArray.begin(), jointArray.end());
    for (int i = 0; i < es; i++) {
        y.push_back(jointArray[i].second);
    }
    jointArray.clear();
    int S_y = S(y);
    y.clear();
    int n0 = (n * (n - 1)) / 2;
    int n1 = computeTiesB(referenceValues, ordinalRankArray, es);
    ordinalRankArray.clear();
    int n2 = computeTiesB(queryValues, ordinalRankArray, es);
    ordinalRankArray.clear();
    int n3 = 0; // Joint ties in ref and query, TODO.
    int numerator = n0 - n1 - n2 + n3 - 2 * S_y;
    //auto denominator = float(n0);  // Tau-a
    // The square root needs to be taken separately to avoid integer overflow.
    float denominator = std::sqrt(float(n0 - n1)) * std::sqrt(float(n0 - n2));
    return float(numerator) / denominator;
}

int sign(float value) {
    return value > 0.0f ? 1 : (value < 0.0f ? -1 : 0);
}

float computeKendallSlow(const float* referenceValues, const float* queryValues, int es) {
    int n = es;
    int numerator = 0;
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            numerator += sign(referenceValues[i] - referenceValues[j]) * sign(queryValues[i] - queryValues[j]);
        }
    }
    int n0 = (n * (n - 1)) / 2;
    int denominator = n0; // Tau-a
    return float(numerator) / float(denominator);
}
