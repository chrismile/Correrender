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

#ifndef CORRERENDER_DKL_HPP
#define CORRERENDER_DKL_HPP

/**
 * Estimates the Kullback-Leibler divergence (KL-divergence) of the distribution of ensemble samples at each grid point
 * (after normalization, i.e., (value - mean) / stddev), and the standard normal distribution.
 * Currently, two estimators are supported:
 * - An estimator based on binning.
 * - An estimator based on an estimation of the entropy of the ensemble distribution using a k-nearest neighbor search.
 *   This is based on the Kozachenko-Leonenko estimator of the Shannon entropy.
 *
 * Derivation for the Entropy-based KL-divergence estimator:
 * P := The normalized sample distribution
 * Q := N(0, 1)
 * H: X -> \mathbb{R}^+_0 is the Shannon entropy
 * PDF of Q, q(x) = 1 / sqrt(2 \pi) e^{-\frac{x^2}{2}}
 * \log q(x) = -\frac{1}{2} \log(2 \pi) - \frac{x^2}{2}
 * D_KL(P||Q) = \int_X p(x) \log \frac{p(x)}{q(x)} dx = \int_X p(x) \log p(x) dx - \int_X p(x) \log q(x) dx =
 * = -H(P) - \int_X p(x) \cdot \left( -\frac{1}{2} \log(2 \pi) - \frac{x^2}{2} \right) dx =
 * = -H(P) + \frac{1}{2} \log(2 \pi) \int_X p(x) dx + \frac{1}{2} \int_X x^2 p(x) dx =
 * = -H(P) + \frac{1}{2} \log(2 \pi) + \frac{1}{2} \mathbb{E}[P^2]
 * ... where \mathbb{E}[P^2] = \mu'_{2,P} is the second moment of P.
 */

template<class Real>
float computeDKLBinned(float* valueArray, int numBins, int es, Real* histogram);
extern template
float computeDKLBinned<float>(float* valueArray, int numBins, int es, float* histogram);
extern template
float computeDKLBinned<double>(float* valueArray, int numBins, int es, double* histogram);

template<class Real>
float computeDKLKNNEstimate(float* valueArray, int k, int es);
extern template
float computeDKLKNNEstimate<float>(float* valueArray, int k, int es);
extern template
float computeDKLKNNEstimate<double>(float* valueArray, int k, int es);

#endif //CORRERENDER_DKL_HPP
