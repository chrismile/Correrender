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

#include <Math/Math.hpp>

#include "Volume/VolumeData.hpp"
#include "Correlation.hpp"
#include "MutualInformation.hpp"
#include "Similarity.hpp"

template<class T>
float computeFieldSimilarity(
        VolumeData* volumeData, int similarityFieldIdx0, int similarityFieldIdx1, CorrelationMeasureType cmt,
        float& maxCorrelationValue, bool useAllTimeSteps, bool useAllEnsembleMembers) {
    size_t numVoxels =
            size_t(volumeData->getGridSizeX())
            * size_t(volumeData->getGridSizeY())
            * size_t(volumeData->getGridSizeZ());
    size_t numVoxelsTotal = numVoxels;
    const auto ts = size_t(volumeData->getTimeStepCount());
    const auto es = size_t(volumeData->getTimeStepCount());
    if (useAllTimeSteps) {
        numVoxelsTotal *= ts;
    }
    if (useAllEnsembleMembers) {
        numVoxelsTotal *= es;
    }

    auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    std::vector<float> X;
    std::vector<float> Y;
    X.reserve(numVoxelsTotal);
    Y.reserve(numVoxelsTotal);
    if (!useAllTimeSteps && !useAllEnsembleMembers) {
        VolumeData::HostCacheEntry fieldEntry0 = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, fieldNames.at(similarityFieldIdx0));
        VolumeData::HostCacheEntry fieldEntry1 = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, fieldNames.at(similarityFieldIdx1));
        const float* data0 = fieldEntry0->data<float>();
        const float* data1 = fieldEntry1->data<float>();
        for (size_t i = 0; i < numVoxels; i++) {
            float val0 = data0[i];
            float val1 = data1[i];
            if (!std::isnan(val0) && !std::isnan(val1)) {
                X.push_back(val0);
                Y.push_back(val1);
            }
        }
    } else if (useAllTimeSteps && !useAllEnsembleMembers) {
        for (size_t tidx = 0; tidx < ts; tidx++) {
            VolumeData::HostCacheEntry fieldEntry0 = volumeData->getFieldEntryCpu(
                    FieldType::SCALAR, fieldNames.at(similarityFieldIdx0), int(tidx), -1);
            VolumeData::HostCacheEntry fieldEntry1 = volumeData->getFieldEntryCpu(
                    FieldType::SCALAR, fieldNames.at(similarityFieldIdx1), int(tidx), -1);
            const float* data0 = fieldEntry0->data<float>();
            const float* data1 = fieldEntry1->data<float>();
            for (size_t i = 0; i < numVoxels; i++) {
                float val0 = data0[i];
                float val1 = data1[i];
                if (!std::isnan(val0) && !std::isnan(val1)) {
                    X.push_back(val0); // tidx * numVoxels + i
                    Y.push_back(val1); // tidx * numVoxels + i
                }
            }
        }
    } else if (!useAllTimeSteps) {
        for (size_t eidx = 0; eidx < ts; eidx++) {
            VolumeData::HostCacheEntry fieldEntry0 = volumeData->getFieldEntryCpu(
                    FieldType::SCALAR, fieldNames.at(similarityFieldIdx0), -1, int(eidx));
            VolumeData::HostCacheEntry fieldEntry1 = volumeData->getFieldEntryCpu(
                    FieldType::SCALAR, fieldNames.at(similarityFieldIdx1), -1, int(eidx));
            const float* data0 = fieldEntry0->data<float>();
            const float* data1 = fieldEntry1->data<float>();
            for (size_t i = 0; i < numVoxels; i++) {
                float val0 = data0[i];
                float val1 = data1[i];
                if (!std::isnan(val0) && !std::isnan(val1)) {
                    X.push_back(val0); // eidx * numVoxels + i
                    Y.push_back(val1); // eidx * numVoxels + i
                }
            }
        }
    } else {
        for (size_t tidx = 0; tidx < ts; tidx++) {
            for (size_t eidx = 0; eidx < ts; eidx++) {
                VolumeData::HostCacheEntry fieldEntry0 = volumeData->getFieldEntryCpu(
                        FieldType::SCALAR, fieldNames.at(similarityFieldIdx0), int(tidx), int(eidx));
                VolumeData::HostCacheEntry fieldEntry1 = volumeData->getFieldEntryCpu(
                        FieldType::SCALAR, fieldNames.at(similarityFieldIdx1), int(tidx), int(eidx));
                const float* data0 = fieldEntry0->data<float>();
                const float* data1 = fieldEntry1->data<float>();
                for (size_t i = 0; i < numVoxels; i++) {
                    float val0 = data0[i];
                    float val1 = data1[i];
                    if (!std::isnan(val0) && !std::isnan(val1)) {
                        X.push_back(val0); // (eidx + tidx * es) * numVoxels + i
                        Y.push_back(val1); // (eidx + tidx * es) * numVoxels + i
                    }
                }
            }
        }
    }
    auto cs = int(X.size());

    maxCorrelationValue = 1.0f;
    float correlationValue = 0.0f;
    if (cmt == CorrelationMeasureType::PEARSON) {
        correlationValue = computePearsonParallel<T>(X.data(), Y.data(), cs);
    } else if (cmt == CorrelationMeasureType::SPEARMAN) {
        std::vector<std::pair<float, int>> ordinalRankArraySpearman;
        std::vector<float> referenceRanks;
        std::vector<float> gridPointRanks;
        ordinalRankArraySpearman.reserve(cs);
        referenceRanks.resize(cs);
        gridPointRanks.resize(cs);
        computeRanks(X.data(), referenceRanks.data(), ordinalRankArraySpearman, cs);
        computeRanks(Y.data(), gridPointRanks.data(), ordinalRankArraySpearman, cs);
        correlationValue = computePearsonParallel<T>(referenceRanks.data(), gridPointRanks.data(), cs);
    } else if (cmt == CorrelationMeasureType::KENDALL) {
        std::vector<std::pair<float, float>> jointArray;
        std::vector<float> ordinalRankArray;
        std::vector<float> y;
        std::vector<float> sortArray;
        std::vector<std::pair<int, int>> stack;
        jointArray.reserve(cs);
        ordinalRankArray.reserve(cs);
        y.reserve(cs);
        correlationValue = computeKendall<int64_t>(
                X.data(), Y.data(), cs, jointArray, ordinalRankArray, y, sortArray, stack);
    } else if (isMeasureBinnedMI(cmt)) {
        int numBins = 80;
        std::vector<T> histogram0;
        std::vector<T> histogram1;
        std::vector<T> histogram2d;
        histogram0.reserve(numBins);
        histogram1.reserve(numBins);
        histogram2d.reserve(numBins *numBins);
        correlationValue = computeMutualInformationBinned<T>(
                X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
        if (cmt == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT) {
            correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
        }
    } else if (isMeasureKraskovMI(cmt)) {
        int k = std::clamp(sgl::iceil(3 * cs, 100), 1, 100);
        KraskovEstimatorCache<T> kraskovEstimatorCache;
        correlationValue = computeMutualInformationKraskovParallel<T>(
                X.data(), Y.data(), k, cs, kraskovEstimatorCache);
        if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
            maxCorrelationValue = computeMaximumMutualInformationKraskov(k, cs);
        } else {
            correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
        }
    }
    return correlationValue;
}
template
float computeFieldSimilarity<float>(
        VolumeData* volumeData, int similarityFieldIdx0, int similarityFieldIdx1, CorrelationMeasureType cmt,
        float& maxCorrelationValue, bool useAllTimeSteps, bool useAllEnsembleMembers);
template
float computeFieldSimilarity<double>(
        VolumeData* volumeData, int similarityFieldIdx0, int similarityFieldIdx1, CorrelationMeasureType cmt,
        float& maxCorrelationValue, bool useAllTimeSteps, bool useAllEnsembleMembers);
