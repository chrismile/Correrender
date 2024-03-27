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

#include <iostream>
#include <algorithm>
#include <cmath>

#include <boost/math/special_functions/digamma.hpp>
#include <Math/Math.hpp>

#include "DKL.hpp"

template<class Real>
float computeDKLBinned(float* valueArray, int numBins, int es, Real* histogram) {
    Real factor = Real(1) / Real(es);
    Real mean = 0;
    Real variance = 0;
    for (int c = 0; c < es; c++) {
        mean += factor * valueArray[c];
    }
    for (int c = 0; c < es; c++) {
        Real diff = mean - valueArray[c];
        variance += factor * diff * diff;
    }
    Real stdev = std::sqrt(variance);

    Real minVal = std::numeric_limits<Real>::max();
    Real maxVal = std::numeric_limits<Real>::lowest();
    for (int c = 0; c < es; c++) {
        Real val = (valueArray[c] - mean) / stdev;
        valueArray[c] = val;
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    minVal -= Real(0.01);
    maxVal += Real(0.01);
    Real binFactor = Real(numBins) / (maxVal - minVal);
    Real binFactorInv = (maxVal - minVal) / Real(numBins);

    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        histogram[binIdx] = Real(0);
    }

    Real dkl = 0;
    for (int c = 0; c < es; c++) {
        int binIdx = std::clamp(int((valueArray[c] - minVal) * binFactor), 0, numBins - 1);
        histogram[binIdx] += Real(1);
    }
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        if (histogram[binIdx] > 0) {
            Real px = histogram[binIdx] / Real(es);
            Real center = (Real(binIdx) + Real(0.5)) * binFactorInv + minVal;
            dkl += std::log(px * binFactor / (std::sqrt(Real(0.5) / Real(sgl::PI)) * std::exp(Real(-0.5) * sgl::sqr(center)))) * px;
        }
    }
    if (std::isinf(dkl)) {
        dkl = std::numeric_limits<float>::quiet_NaN();
    }
    return dkl;
}

template
float computeDKLBinned<float>(float* valueArray, int numBins, int es, float* histogram);
template
float computeDKLBinned<double>(float* valueArray, int numBins, int es, double* histogram);


template<class Real>
inline int isign(Real x) {
    return x > Real(0) ? 1 : (x < Real(0) ? -1 : 0);
}

template<class Real>
inline Real findKNearestNeighbors(const Real* data, int N, int k, int i) {
    int step = sgl::iceil(k, 2);
    int l = std::max(i - step, 0);
    int r = l + k;
    if (r >= N) {
        r = N - 1;
        l = N - k - 1;
    }
    Real vc = data[i];
    while (true) {
        Real vl = l >= 0 ? vc - data[l] : std::numeric_limits<Real>::max();
        Real vr = r < N ? data[r] - vc : std::numeric_limits<Real>::max();
        Real diff = std::max(vl, vr);
        Real vl0 = l - step <= i && l - step >= 0 ? vc - data[l - step] : std::numeric_limits<Real>::max();
        Real vr0 = r - step >= i && r - step < N ? data[r - step] - vc : std::numeric_limits<Real>::max();
        Real diff0 = std::max(vl0, vr0);
        Real vl1 = l + step <= i && l + step >= 0 ? vc - data[l + step] : std::numeric_limits<Real>::max();
        Real vr1 = r + step >= i && r + step < N ? data[r + step] - vc : std::numeric_limits<Real>::max();
        Real diff1 = std::max(vl1, vr1);
        int signDir = isign(diff0 - diff1);
        if (diff < diff0 && diff < diff1) {
            signDir = 0;
        }
        l += signDir * step;
        r += signDir * step;
        if (step == 1) {
            break;
        }
        step = sgl::iceil(step, 2);
    }
    return std::max(vc - data[l], data[r] - vc);
}

template<class Real>
float computeDKLKNNEstimate(float* valueArray, int k, int es) {
    Real factor = Real(1) / Real(es);
    Real mean = 0;
    Real variance = 0;
    for (int c = 0; c < es; c++) {
        mean += factor * valueArray[c];
    }
    for (int c = 0; c < es; c++) {
        Real diff = mean - valueArray[c];
        variance += factor * diff * diff;
    }
    Real stdev = std::sqrt(variance);
    for (int c = 0; c < es; c++) {
        valueArray[c] = (valueArray[c] - mean) / stdev;
    }

    std::sort(valueArray, valueArray + es);

    Real entropyEstimate = 0;
    Real secondMoment = 0;
    for (int c = 0; c < es; c++) {
        Real nnDist = findKNearestNeighbors(valueArray, es, k, c);
        entropyEstimate += factor * log(nnDist);
        Real value = valueArray[c];
        secondMoment += factor * value * value;
    }
    entropyEstimate += Real(boost::math::digamma(es)) - Real(boost::math::digamma(k)) + std::log(Real(2.0));

    auto dkl = float(-entropyEstimate + Real(0.5) * std::log(sgl::TWO_PI) + Real(0.5) * secondMoment);
    if (std::isinf(dkl)) {
        dkl = std::numeric_limits<float>::quiet_NaN();
    } else {
        dkl = std::max(dkl, 0.0f);
    }
    return dkl;
}

template
float computeDKLKNNEstimate<float>(float* valueArray, int k, int es);
template
float computeDKLKNNEstimate<double>(float* valueArray, int k, int es);
