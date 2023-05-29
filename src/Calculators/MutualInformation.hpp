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

#ifndef CORRERENDER_MUTUALINFORMATION_HPP
#define CORRERENDER_MUTUALINFORMATION_HPP

#include <Utils/SearchStructures/KdTreed.hpp>

template<class Real>
float computeMutualInformationBinned(
        const float* referenceValues, const float* queryValues, int numBins, int es,
        Real* histogram0, Real* histogram1, Real* histogram2d);
extern template
float computeMutualInformationBinned<float>(
        const float* referenceValues, const float* queryValues, int numBins, int es,
        float* histogram0, float* histogram1, float* histogram2d);
extern template
float computeMutualInformationBinned<double>(
        const float* referenceValues, const float* queryValues, int numBins, int es,
        double* histogram0, double* histogram1, double* histogram2d);

template<class Real>
struct KraskovEstimatorCache {
    // K-d-tree computations.
    std::vector<glm::vec<2, Real>> points;
    std::vector<glm::vec<2, Real>> pointsCopy;
    sgl::KdTreed<Real, 2, sgl::DistanceMeasure::CHEBYSHEV> kdTree2d;
    std::vector<Real> kthNeighborDistances;
    std::vector<Real> nearestNeighborDistances;

    // 2nd estimator.
    std::vector<Real> kthNeighborDistancesRef;
    std::vector<Real> kthNeighborDistancesQuery;
    std::vector<glm::vec<2, Real>> nearestNeighbors;

    // Average digamma computation.
    std::vector<Real> baseArray;
    std::vector<Real> sortedArray;
};

template<class Real>
float computeMutualInformationKraskov(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<Real>& cache);
template<class Real>
float computeMutualInformationKraskov2(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<Real>& cache);
extern template
float computeMutualInformationKraskov<float>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<float>& cache);
extern template
float computeMutualInformationKraskov<double>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<double>& cache);
extern template
float computeMutualInformationKraskov2<float>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<float>& cache);
extern template
float computeMutualInformationKraskov2<double>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<double>& cache);

float computeMaximumMutualInformationKraskov(int k, int es);

template<class Real>
float computeMutualInformationKraskovParallel(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<Real>& cache);
extern template
float computeMutualInformationKraskovParallel<float>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<float>& cache);
extern template
float computeMutualInformationKraskovParallel<double>(
        const float* referenceValues, const float* queryValues, int k, int es, KraskovEstimatorCache<double>& cache);

#endif //CORRERENDER_MUTUALINFORMATION_HPP
