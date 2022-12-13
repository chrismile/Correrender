/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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
#include <boost/math/special_functions/digamma.hpp>

#include <Utils/SearchStructures/KdTreed.hpp>
#include <Utils/Random/Xorshift.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "ReferencePointSelectionRenderer.hpp"
#include "SimilarityCalculator.hpp"

EnsembleSimilarityCalculator::EnsembleSimilarityCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
}

void EnsembleSimilarityCalculator::setViewManager(ViewManager* _viewManager) {
    viewManager = _viewManager;
    referencePointSelectionRenderer = new ReferencePointSelectionRenderer(viewManager);
    calculatorRenderer = RendererPtr(referencePointSelectionRenderer);
    referencePointSelectionRenderer->initialize();
}

void EnsembleSimilarityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    referencePointSelectionRenderer->setVolumeDataPtr(volumeData, isNewData);

    if (isNewData) {
        referencePointIndex.x = volumeData->getGridSizeX() / 2;
        referencePointIndex.y = volumeData->getGridSizeY() / 2;
        referencePointIndex.z = volumeData->getGridSizeZ() / 2;
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);

        fieldIndex = 0;
        fieldIndexGui = 0;
        scalarFieldNames = {};
        scalarFieldIndexArray = {};

        std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
        for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
            if (scalarFieldNamesNew.at(i) != getOutputFieldName()) {
                scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
                scalarFieldIndexArray.push_back(i);
            }
        }
    }
}

void EnsembleSimilarityCalculator::update(float dt) {
    // TODO: Use mouse for selection of reference point.

    if (continuousRecompute) {
        dirty = true;
    }
}

void EnsembleSimilarityCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addCombo(
            "Scalar Field", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
        dirty = true;
    }

    // TODO: Replace with referencePointSelectionRenderer.
    bool isRealtime = getIsRealtime();
    ImGui::EditMode editModes[3];
    editModes[0] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (X)", &referencePointIndex[0], 0, volumeData->getGridSizeX() - 1);
    editModes[1] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Y)", &referencePointIndex[1], 0, volumeData->getGridSizeY() - 1);
    editModes[2] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Z)", &referencePointIndex[2], 0, volumeData->getGridSizeZ() - 1);
    if ((isRealtime && (editModes[0] != ImGui::EditMode::NO_CHANGE
                || editModes[1] != ImGui::EditMode::NO_CHANGE
                || editModes[2] != ImGui::EditMode::NO_CHANGE))
            || (!isRealtime && (editModes[0] == ImGui::EditMode::INPUT_FINISHED
                || editModes[1] == ImGui::EditMode::INPUT_FINISHED
                || editModes[2] == ImGui::EditMode::INPUT_FINISHED))) {
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        dirty = true;
    }
}



PccCalculator::PccCalculator(sgl::vk::Renderer* renderer) : EnsembleSimilarityCalculator(renderer) {
    pccComputePass = std::make_shared<PccComputePass>(renderer);
    pccComputePass->setCorrelationMeasureType(correlationMeasureType);
    pccComputePass->setKraskovNumNeighbors(k);
}

void PccCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    EnsembleSimilarityCalculator::setVolumeData(_volumeData, isNewData);
    pccComputePass->setVolumeData(volumeData, isNewData);
}

FilterDevice PccCalculator::getFilterDevice() {
    return useGpu ? FilterDevice::VULKAN : FilterDevice::CPU;
}

void PccCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    EnsembleSimilarityCalculator::renderGuiImpl(propertyEditor);
    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        hasNameChanged = true;
        dirty = true;
        pccComputePass->setCorrelationMeasureType(correlationMeasureType);
        if (useGpu && (correlationMeasureType == CorrelationMeasureType::KENDALL
                || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED)) {
            hasFilterDeviceChanged = true;
            useGpu = false;
        }
    }
    if (correlationMeasureType != CorrelationMeasureType::KENDALL
            && correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
            && propertyEditor.addCheckbox("Use GPU", &useGpu)) {
        hasFilterDeviceChanged = true;
        dirty = true;
    }
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED && propertyEditor.addSliderIntEdit(
            "#Bins", &numBins, 2, 100) == ImGui::EditMode::INPUT_FINISHED) {
        dirty = true;
    }
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV && propertyEditor.addSliderIntEdit(
            "#Neighbors", &k, 1, 20) == ImGui::EditMode::INPUT_FINISHED) {
        pccComputePass->setKraskovNumNeighbors(k);
        dirty = true;
    }

#ifdef SHOW_DEBUG_OPTIONS
    if (propertyEditor.addCheckbox("Continuous Recompute", &continuousRecompute)) {
        dirty = true;
    }
#endif
}

template<class T>
inline float computePearson1(
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

template<class T>
inline float computePearson1(
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

template<class T>
inline float computePearson2(
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

template<class T>
inline float computePearson2(
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
    int num_swaps = 0;
    while (i < n and j < m) {
        if (R[j] < L[i]) {
            num_swaps += n - i;
            j += 1;
        } else {
            i += 1;
        }
    }
    return num_swaps;
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
    int n3 = 0;  // Joint ties in ref and query, TODO.
    int numerator = n0 - n1 - n2 + n3 - 2 * S_y;
    //auto denominator = float(n0);  // Tau-a
    float denominator = std::sqrt(float((n0 - n1) * (n0 - n2)));
    return float(numerator) / denominator;
}

template<class Real>
float computeMutualInformationBinned(
        const float* referenceValues, const float* queryValues, int numBins, int es,
        Real* histogram0, Real* histogram1, Real* histogram2d) {
    // Initialize the histograms with zeros.
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        histogram0[binIdx] = 0;
        histogram1[binIdx] = 0;
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] = 0;
        }
    }

    // Compute the 2D joint histogram.
    for (int idx0 = 0; idx0 < es; idx0++) {
        Real val0 = referenceValues[idx0];
        for (int idx1 = 0; idx1 < es; idx1++) {
            Real val1 = queryValues[idx1];
            if (!std::isnan(val0) && !std::isnan(val1)) {
                int binIdx0 = std::clamp(int(val0 * Real(numBins)), 0, numBins - 1);
                int binIdx1 = std::clamp(int(val1 * Real(numBins)), 0, numBins - 1);
                histogram2d[binIdx0 * numBins + binIdx1] += 1;
            }
        }
    }

    // Normalize the histograms.
    Real totalSum2d = 0;
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            totalSum2d += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] /= totalSum2d;
        }
    }

    // Regularize.
    const Real REG_FACTOR = 1e-7;
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] += REG_FACTOR;
        }
    }

    // Normalize again.
    totalSum2d = 0;
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            totalSum2d += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram2d[binIdx0 * numBins + binIdx1] /= totalSum2d;
        }
    }

    // Marginalization of joint distribution.
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            histogram0[binIdx0] += histogram2d[binIdx0 * numBins + binIdx1];
            histogram1[binIdx1] += histogram2d[binIdx0 * numBins + binIdx1];
        }
    }

    /*
     * Compute the mutual information metric. Two possible ways of calculation:
     * a) $MI = H(x) + H(y) - H(x, y)$
     * with the Shannon entropy $H(x) = -\sum_i p_x(i) \log p_x(i)$
     * and the joint entropy $H(x, y) = -\sum_i \sum_j p_{xy}(i, j) \log p_{xy}(i, j)$
     * b) $MI = \sum_i \sum_j p_{xy}(i, j) \log \frac{p_{xy}(i, j)}{p_x(i) p_y(j)}$
     */
    const Real EPSILON = Real(1) / Real(numBins * numBins) * 1e-3;
    Real mi = 0.0;
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        Real p_x = histogram0[binIdx];
        Real p_y = histogram1[binIdx];
        mi -= p_x * std::log(std::max(p_x, EPSILON));
        mi -= p_y * std::log(std::max(p_y, EPSILON));
    }
    for (int binIdx0 = 0; binIdx0 < numBins; binIdx0++) {
        for (int binIdx1 = 0; binIdx1 < numBins; binIdx1++) {
            Real p_xy = histogram2d[binIdx0 * numBins + binIdx1];
            mi += p_xy * std::log(std::max(p_xy, EPSILON));
        }
    }

    return float(mi);
}

#define KRASKOV_USE_RANDOM_NOISE
#define USE_1D_BINARY_SEARCH

template <typename FloatType> struct default_epsilon {};
template <> struct default_epsilon<float> { static const float value; static const float noise; };
template <> struct default_epsilon<double> { static const double value; static const double noise; };
const float default_epsilon<float>::value = 1e-6f;
const double default_epsilon<double>::value = 1e-15;
const float default_epsilon<float>::noise = 1e-5f;
const double default_epsilon<double>::noise = 1e-10;

template<class Real>
Real averageDigamma(const float* values, int es, const std::vector<Real>& distanceVec, bool isRef) {
#ifdef KRASKOV_USE_RANDOM_NOISE
    sgl::XorshiftRandomGenerator gen(isRef ? 617406168ul : 864730169ul);
    std::vector<Real> baseArray(es);
#endif
    Real factor = Real(1) / Real(es);
    Real meanDigammaValue = 0;
#ifdef USE_1D_BINARY_SEARCH
    std::vector<Real> sortedArray(es);
    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        sortedArray.at(e) = values[e] + gen.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise;
        baseArray.at(e) = sortedArray.at(e);
#else
        sortedArray.at(e) = values[e];
#endif
    }
    std::sort(sortedArray.begin(), sortedArray.end());

    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        Real currentValue = baseArray[e];
#else
        Real currentValue = values[e];
#endif
        Real kthDist = distanceVec[e] - default_epsilon<Real>::value;
        Real searchValueLower = currentValue - kthDist;
        Real searchValueUpper = currentValue + kthDist;
        int lower = 0;
        int upper = es;
        int middle = 0;
        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            Real middleValue = sortedArray[middle];
            if (middleValue < searchValueLower) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }

        int startRange = upper;
        lower = startRange;
        upper = es;

        // Binary search.
        while (lower < upper) {
            middle = (lower + upper) / 2;
            Real middleValue = sortedArray[middle];
            if (middleValue < searchValueUpper) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        int endRange = upper - 1;

        int numPoints = endRange + 1 - startRange;
        meanDigammaValue += factor * Real(boost::math::digamma(numPoints));
    }
#else
    sgl::KdTreed<Real, 1, sgl::DistanceMeasure::CHEBYSHEV> kdTree1d;
    std::vector<glm::vec<1, Real>> points;
    points.reserve(es);
    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        points.emplace_back(values[e] + gen.getRandomFloatBetween(-1.0f, 1.0f) * default_epsilon<Real>::noise);
#else
        points.emplace_back(values[e]);
#endif
    }
    kdTree1d.build(points);

    for (int e = 0; e < es; e++) {
        auto numPoints = int(kdTree1d.getNumPointsInSphere(points.at(e), distanceVec.at(e) - default_epsilon<Real>::value));
        meanDigammaValue += factor * Real(boost::math::digamma(numPoints));
    }
#endif
    return meanDigammaValue;
}

/**
 * For more details, please refer to:
 * - https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
 * - https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py
 */
template<class Real>
float computeMutualInformationKraskov(
        const float* referenceValues, const float* queryValues, int k, int es) {
    const int base = 2;

#ifdef KRASKOV_USE_RANDOM_NOISE
    sgl::XorshiftRandomGenerator genRef(617406168ul);
    sgl::XorshiftRandomGenerator genQuery(864730169ul);
#endif

    sgl::KdTreed<Real, 2, sgl::DistanceMeasure::CHEBYSHEV> kdTree2d;
    std::vector<glm::vec<2, Real>> points;
    points.reserve(es);
    for (int e = 0; e < es; e++) {
#ifdef KRASKOV_USE_RANDOM_NOISE
        points.emplace_back(
                referenceValues[e] + genRef.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise,
                queryValues[e] + genQuery.getRandomFloatBetween(0.0f, 1.0f) * default_epsilon<Real>::noise);
#else
        points.emplace_back(referenceValues[e], queryValues[e]);
#endif
    }
    kdTree2d.build(points);

    std::vector<Real> kthNeighborDistances;
    kthNeighborDistances.reserve(es);
    std::vector<Real> nearestNeighborDistances;
    nearestNeighborDistances.reserve(k + 1);
    for (int e = 0; e < es; e++) {
        nearestNeighborDistances.clear();
        kdTree2d.findKNearestNeighbors(points.at(e), k + 1, nearestNeighborDistances);
        kthNeighborDistances.emplace_back(nearestNeighborDistances.back());
    }

    FLT_MAX;
    auto a = averageDigamma<Real>(referenceValues, es, kthNeighborDistances, true);
    auto b = averageDigamma<Real>(queryValues, es, kthNeighborDistances, false);
    auto c = Real(boost::math::digamma(k));
    auto d = Real(boost::math::digamma(es));

    Real mi = (-a - b + c + d) / Real(std::log(base));
    return float(mi);
}

void PccCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

    std::vector<VolumeData::HostCacheEntry> ensembleEntryFields;
    std::vector<float*> ensembleFields;
    ensembleEntryFields.reserve(es);
    ensembleFields.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleFields.push_back(ensembleEntryField.get());
    }

    //size_t referencePointIdx =
    //        size_t(referencePointIndex.x) * size_t(referencePointIndex.y) * size_t(referencePointIndex.z);
    size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
    auto* referenceValues = new float[es];
    for (int e = 0; e < es; e++) {
        referenceValues[e] = ensembleFields.at(e)[referencePointIdx];
    }

    float minEnsembleVal = std::numeric_limits<float>::max();
    float maxEnsembleVal = std::numeric_limits<float>::lowest();
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
            auto [minVal, maxVal] = volumeData->getMinMaxScalarFieldValue(
                    scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
            minEnsembleVal = std::min(minEnsembleVal, minVal);
            maxEnsembleVal = std::max(maxEnsembleVal, maxVal);
        }
        for (int e = 0; e < es; e++) {
            referenceValues[e] = (referenceValues[e] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
            float val0 = referenceValues[e];
            int binIdx0 = std::clamp(int(val0 * float(numBins)), 0, numBins - 1);
            std::cout << binIdx0 << ", ";
        }
        std::cout << std::endl;
    }


#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    float* referenceRanks = nullptr;
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        referenceRanks = new float[es];
        std::vector<std::pair<float, int>> ordinalRankArray;
        ordinalRankArray.reserve(es);
        computeRanks(referenceValues, referenceRanks, ordinalRankArray, es);
    }

    size_t numGridPoints = size_t(xs) * size_t(ys) * size_t(zs);
    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* referenceValues = new float[es];
            for (int e = 0; e < es; e++) {
                referenceValues[e] = ensembleFields.at(e)[referencePointIdx];
            }
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel for shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none)
#endif
        for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
            if (es == 1) {
                buffer[gridPointIdx] = 1.0f;
                continue;
            }
            // See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
#define FORMULA_2_FLOAT
#ifdef FORMULA_1_FLOAT
            float pearsonCorrelation = computePearson1<float>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_1_DOUBLE)
            float pearsonCorrelation = computePearson1<double>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_2_FLOAT)
            float pearsonCorrelation = computePearson2<float>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_2_DOUBLE)
            float pearsonCorrelation = computePearson2<double>(referenceValues, ensembleFields, es, gridPointIdx);
#endif
            buffer[gridPointIdx] = pearsonCorrelation;
        }
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            auto* gridPointRanks = new float[es];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(es);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, referenceRanks, ensembleFields, buffer) default(none)
        {
            auto* gridPointValues = new float[es];
            auto* gridPointRanks = new float[es];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(es);
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                computeRanks(gridPointValues, gridPointRanks, ordinalRankArray, es);

#define FORMULA_2_FLOAT
#ifdef FORMULA_1_FLOAT
                float pearsonCorrelation = computePearson1<float>(referenceRanks, gridPointRanks, es);
#elif defined(FORMULA_1_DOUBLE)
                float pearsonCorrelation = computePearson1<double>(referenceRanks, gridPointRanks, es);
#elif defined(FORMULA_2_FLOAT)
                float pearsonCorrelation = computePearson2<float>(referenceRanks, gridPointRanks, es);
#elif defined(FORMULA_2_DOUBLE)
                float pearsonCorrelation = computePearson2<double>(referenceRanks, gridPointRanks, es);
#endif
                buffer[gridPointIdx] = pearsonCorrelation;
            }
            delete[] gridPointValues;
            delete[] gridPointRanks;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            jointArray.reserve(es);
            ordinalRankArray.reserve(es);
            y.reserve(es);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none)
        {
            auto* gridPointValues = new float[es];
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            jointArray.reserve(es);
            ordinalRankArray.reserve(es);
            y.reserve(es);
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float pearsonCorrelation = computeKendall(
                        referenceValues, gridPointValues, es, jointArray, ordinalRankArray, y);
                buffer[gridPointIdx] = pearsonCorrelation;
            }
            delete[] gridPointValues;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            auto* histogram0 = new float[numBins];
            auto* histogram1 = new float[numBins];
            auto* histogram2d = new float[numBins * numBins];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none) \
        shared(minEnsembleVal, maxEnsembleVal)
        {
            auto* gridPointValues = new float[es];
            auto* histogram0 = new double[numBins];
            auto* histogram1 = new double[numBins];
            auto* histogram2d = new double[numBins * numBins];
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                    gridPointValues[e] = (gridPointValues[e] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation = computeMutualInformationBinned<double>(
                        referenceValues, gridPointValues, numBins, es, histogram0, histogram1, histogram2d);
                buffer[gridPointIdx] = mutualInformation;
            }
            delete[] gridPointValues;
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[es];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, es, k, referenceValues, ensembleFields, buffer) default(none) \
        shared(minEnsembleVal, maxEnsembleVal)
        {
            auto* gridPointValues = new float[es];
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (es == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int e = 0; e < es; e++) {
                    gridPointValues[e] = ensembleFields.at(e)[gridPointIdx];
                    if (std::isnan(gridPointValues[e])) {
                        isNan = true;
                        break;
                    }
                    //gridPointValues[e] = (gridPointValues[e] - minEnsembleVal) / (maxEnsembleVal - minEnsembleVal);
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation = computeMutualInformationKraskov<double>(
                        referenceValues, gridPointValues, k, es);
                buffer[gridPointIdx] = mutualInformation;
            }
            delete[] gridPointValues;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    }

    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        delete[] referenceRanks;
    }
    delete[] referenceValues;

#ifdef TEST_INFERENCE_SPEED
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}

void PccCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    // We write to the descriptor set, so wait until the device is idle.
    renderer->getDevice()->waitIdle();

    int es = volumeData->getEnsembleMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

    std::vector<VolumeData::DeviceCacheEntry> ensembleEntryFields;
    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    std::vector<CUtexObject> ensembleTexturesCu;
    ensembleEntryFields.reserve(es);
    ensembleImageViews.reserve(es);
    ensembleTexturesCu.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleImageViews.push_back(ensembleEntryField->getVulkanImageView());
        ensembleTexturesCu.push_back(ensembleEntryField->getCudaTexture());
        if (ensembleEntryField->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            deviceCacheEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        }
    }

#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    pccComputePass->setOutputImage(deviceCacheEntry->getVulkanImageView());
    pccComputePass->setEnsembleImageViews(ensembleImageViews);

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);

    pccComputePass->buildIfNecessary();
    pccComputePass->setReferencePoint(referencePointIndex);
    pccComputePass->render();

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}



PccComputePass::PccComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    spearmanReferenceRankComputePass = std::make_shared<SpearmanReferenceRankComputePass>(renderer, uniformBuffer);
}

void PccComputePass::setVolumeData(VolumeData *_volumeData, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.es = uint32_t(volumeData->getEnsembleMemberCount());
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    spearmanReferenceRankComputePass->setVolumeData(_volumeData, isNewData);
    if (cachedEnsembleMemberCount != volumeData->getEnsembleMemberCount()) {
        cachedEnsembleMemberCount = volumeData->getEnsembleMemberCount();
        setShaderDirty();
        spearmanReferenceRankComputePass->setEnsembleMemberCount(cachedEnsembleMemberCount);
    }
}

void PccComputePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews == _ensembleImageViews) {
        return;
    }
    ensembleImageViews = _ensembleImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    }
    spearmanReferenceRankComputePass->setEnsembleImageViews(_ensembleImageViews);
}

void PccComputePass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    outputImage = _outputImage;
    if (computeData) {
        computeData->setStaticImageView(outputImage, "outputImage");
    }
}

void PccComputePass::setReferencePoint(const glm::ivec3& referencePointIndex) {
    if (correlationMeasureType == CorrelationMeasureType::PEARSON
            || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, referencePointIndex);
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        spearmanReferenceRankComputePass->buildIfNecessary();
        renderer->pushConstants(
                spearmanReferenceRankComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                0, referencePointIndex);
    }
}

void PccComputePass::setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType) {
    if (correlationMeasureType != _correlationMeasureType) {
        correlationMeasureType = _correlationMeasureType;
        setShaderDirty();
    }
}

void PccComputePass::setKraskovNumNeighbors(int _k) {
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV && k != _k) {
        setShaderDirty();
    }
    k = _k;
}

void PccComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(volumeData->getEnsembleMemberCount())));
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        auto maxBinaryTreeLevels = uint32_t(std::ceil(std::log2(volumeData->getEnsembleMemberCount() + 1)));
        preprocessorDefines.insert(std::make_pair(
                "MAX_STACK_SIZE_BUILD", std::to_string(2 * maxBinaryTreeLevels)));
        preprocessorDefines.insert(std::make_pair(
                "MAX_STACK_SIZE_KN", std::to_string(maxBinaryTreeLevels)));
        preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));
    }
    std::string shaderName;
    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
        shaderName = "PearsonCorrelation.Compute";
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        shaderName = "SpearmanRankCorrelation.Compute";
    } else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        shaderName = "MutualInformationKraskov.Compute";
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void PccComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticImageView(outputImage, "outputImage");
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        computeData->setStaticBuffer(
                spearmanReferenceRankComputePass->getReferenceRankBuffer(), "ReferenceRankBuffer");
    }
}

void PccComputePass::_render() {
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        spearmanReferenceRankComputePass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                spearmanReferenceRankComputePass->getReferenceRankBuffer());
    }

    uint32_t batchCount = 1;
    bool needsBatchedRendering = false;
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
        if (volumeData->getEnsembleMemberCount() > int(batchEnsembleCountThreshold)) {
            needsBatchedRendering = true;
            batchCount = sgl::uiceil(uint32_t(volumeData->getEnsembleMemberCount()), batchEnsembleCountThreshold);
        }
    }

    if (needsBatchedRendering) {
        auto blockSizeX = uint32_t(computeBlockSizeX);
        //auto blockSizeY = uint32_t(computeBlockSizeY);
        //auto blockSizeZ = uint32_t(computeBlockSizeZ);
        /*auto batchSizeX =
                sgl::uiceil(uint32_t(volumeData->getGridSizeX()), batchCount * blockSizeX) * blockSizeX;
        auto batchSizeY = uint32_t(volumeData->getGridSizeY());
        auto batchSizeZ = uint32_t(volumeData->getGridSizeZ());
        batchCount = sgl::uiceil(uint32_t(volumeData->getGridSizeX()), batchSizeX);*/
        auto batchSizeX = 2 * blockSizeX;
        auto batchSizeY = uint32_t(volumeData->getGridSizeY());
        auto batchSizeZ = uint32_t(volumeData->getGridSizeZ());
        batchCount = sgl::uiceil(uint32_t(volumeData->getGridSizeX()), batchSizeX);
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                renderer->pushConstants(
                        getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::uvec4),
                        glm::uvec3(batchSizeX * batchIdx, 0, 0));
            }
            if (batchIdx == batchCount - 1) {
                batchSizeX = uint32_t(volumeData->getGridSizeX()) - batchSizeX * batchIdx;
            }
            renderer->dispatch(
                    computeData,
                    sgl::uiceil(batchSizeX, uint32_t(computeBlockSizeX)),
                    sgl::uiceil(batchSizeY, uint32_t(computeBlockSizeY)),
                    sgl::uiceil(batchSizeZ, uint32_t(computeBlockSizeZ)));
            renderer->syncWithCpu();
        }
    } else {
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
            renderer->pushConstants(
                    getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::uvec4), glm::uvec3(0));
        }
        renderer->dispatch(
                computeData,
                sgl::iceil(volumeData->getGridSizeX(), computeBlockSizeX),
                sgl::iceil(volumeData->getGridSizeY(), computeBlockSizeY),
                sgl::iceil(volumeData->getGridSizeZ(), computeBlockSizeZ));
    }
}



SpearmanReferenceRankComputePass::SpearmanReferenceRankComputePass(
        sgl::vk::Renderer* renderer, sgl::vk::BufferPtr uniformBuffer)
        : ComputePass(renderer), uniformBuffer(std::move(uniformBuffer)) {
}

void SpearmanReferenceRankComputePass::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
}

void SpearmanReferenceRankComputePass::setEnsembleMemberCount(int ensembleMemberCount) {
    cachedEnsembleMemberCount = ensembleMemberCount;
    referenceRankBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(float) * cachedEnsembleMemberCount,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    setShaderDirty();
}

void SpearmanReferenceRankComputePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews == _ensembleImageViews) {
        return;
    }
    ensembleImageViews = _ensembleImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    }
}

void SpearmanReferenceRankComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(cachedEnsembleMemberCount)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "SpearmanRankCorrelation.Reference.Compute" }, preprocessorDefines);
}

void SpearmanReferenceRankComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticBuffer(referenceRankBuffer, "ReferenceRankBuffer");
}

void SpearmanReferenceRankComputePass::_render() {
    renderer->dispatch(
            computeData,
            sgl::iceil(volumeData->getGridSizeX(), computeBlockSizeX),
            sgl::iceil(volumeData->getGridSizeY(), computeBlockSizeY),
            sgl::iceil(volumeData->getGridSizeZ(), computeBlockSizeZ));
}
