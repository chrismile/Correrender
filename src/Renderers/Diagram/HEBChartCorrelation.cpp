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

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <thread>

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "Calculators/Correlation.hpp"
#include "Calculators/CorrelationCalculator.hpp"
#include "Calculators/MutualInformation.hpp"
#include "HEBChart.hpp"
#include "BayOpt.hpp"

// declaring static iterations variable
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::stop_maxiterations, iterations);

HEBChart::~HEBChart() {
    if (computeRenderer) {
        correlationComputePass = {};
        sgl::vk::Device* device = computeRenderer->getDevice();
        device->freeCommandBuffer(commandPool, commandBuffer);
        delete computeRenderer;
    }
}

void HEBChart::computeCorrelations(
        HEBChartFieldData* fieldData,
        const std::vector<float*>& downscaledFields0, const std::vector<float*>& downscaledFields1,
        std::vector<MIFieldEntry>& miFieldEntries) {
    std::chrono::time_point<std::chrono::system_clock> startTime;
    if (!isHeadlessMode) {
        startTime = std::chrono::system_clock::now();
    }

    if (samplingMethodType == SamplingMethodType::MEAN) {
        computeCorrelationsMean(fieldData, downscaledFields0, downscaledFields1, miFieldEntries);
    } else {
        if (useCorrelationComputationGpu) {
            computeCorrelationsSamplingGpu(fieldData, miFieldEntries);
        } else {
            computeCorrelationsSamplingCpu(fieldData, miFieldEntries);
        }
    }

    if (!isHeadlessMode) {
        auto endTime = std::chrono::system_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "Elapsed time correlations: " << elapsedTime.count() << "ms" << std::endl;
    } else {
        return; // No sorting.
    }

    //auto startTimeSort = std::chrono::system_clock::now();
    if (useAbsoluteCorrelationMeasure || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
            || correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
#ifdef USE_TBB
        tbb::parallel_sort(miFieldEntries.begin(), miFieldEntries.end());
//#elif __cpp_lib_parallel_algorithm >= 201603L
    //std::sort(std::execution::par_unseq, miFieldEntries.begin(), miFieldEntries.end());
#else
        std::sort(miFieldEntries.begin(), miFieldEntries.end());
#endif
    } else {
#ifdef USE_TBB
        tbb::parallel_sort(miFieldEntries.begin(), miFieldEntries.end(), [](const MIFieldEntry& x, const MIFieldEntry& y) {
            return std::abs(x.correlationValue) > std::abs(y.correlationValue);
        });
//#elif __cpp_lib_parallel_algorithm >= 201603L
    //std::sort(std::execution::par_unseq, miFieldEntries.begin(), miFieldEntries.end());
#else
        std::sort(miFieldEntries.begin(), miFieldEntries.end(), [](const MIFieldEntry& x, const MIFieldEntry& y) {
            return std::abs(x.correlationValue) > std::abs(y.correlationValue);
        });
#endif
    }
    //auto endTimeSort = std::chrono::system_clock::now();
    //auto elapsedTimeSort = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeSort - startTimeSort);
    //std::cout << "Elapsed time sort: " << elapsedTimeSort.count() << "ms" << std::endl;
}


#define CORRELATION_CACHE \
        std::vector<float> X(cs); \
        std::vector<float> Y(cs); \
        \
        std::vector<std::pair<float, int>> ordinalRankArraySpearman; \
        std::vector<float> referenceRanks; \
        std::vector<float> gridPointRanks; \
        if (cmt == CorrelationMeasureType::SPEARMAN) { \
            ordinalRankArraySpearman.reserve(cs); \
            referenceRanks.reserve(cs); \
            gridPointRanks.reserve(cs); \
        } \
        \
        std::vector<std::pair<float, float>> jointArray; \
        std::vector<float> ordinalRankArray; \
        std::vector<std::pair<float, int>> ordinalRankArrayRef; \
        std::vector<float> y; \
        if (cmt == CorrelationMeasureType::KENDALL) { \
            jointArray.reserve(cs); \
            ordinalRankArray.reserve(cs); \
            ordinalRankArrayRef.reserve(cs); \
            y.reserve(cs); \
        } \
        \
        std::vector<double> histogram0;  \
        std::vector<double> histogram1;  \
        std::vector<double> histogram2d; \
        if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) { \
            histogram0.reserve(numBins); \
            histogram1.reserve(numBins); \
            histogram2d.reserve(numBins * numBins); \
        } \
        \
        KraskovEstimatorCache<double> kraskovEstimatorCache;

void HEBChart::computeCorrelationsMean(
        HEBChartFieldData* fieldData,
        const std::vector<float*>& downscaledFields0, const std::vector<float*>& downscaledFields1,
        std::vector<MIFieldEntry>& miFieldEntries) {
    int cs = getCorrelationMemberCount();
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;
#ifdef USE_TBB
    miFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPoints0), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const& r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                const CorrelationMeasureType cmt = correlationMeasureType;
                CORRELATION_CACHE;
                float minFieldValRef, maxFieldValRef, minFieldVal, maxFieldVal;

                for (int i = r.begin(); i != r.end(); i++) {
#else
    miFieldEntries.reserve(regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1);
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries, numPoints0, numPoints1, cs, k, numBins) \
    shared(downscaledFields0, downscaledFields1)
#endif
    {
        const CorrelationMeasureType cmt = correlationMeasureType;
        std::vector<MIFieldEntry> miFieldEntriesThread;
        CORRELATION_CACHE;
        float minFieldValRef, maxFieldValRef, minFieldVal, maxFieldVal;

#if _OPENMP >= 201107
        #pragma omp for schedule(dynamic)
#endif
        for (int i = 0; i < numPoints0; i++) {
#endif
            bool isNan = false;
            for (int c = 0; c < cs; c++) {
                X[c] = downscaledFields0.at(c)[i];
                if (std::isnan(X[c])) {
                    isNan = true;
                    break;
                }
            }
            if (isNan) {
                continue;
            }
            if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
                computeRanks(X.data(), referenceRanks.data(), ordinalRankArrayRef, cs);
            }
            if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                minFieldValRef = std::numeric_limits<float>::max();
                maxFieldValRef = std::numeric_limits<float>::lowest();
                for (int c = 0; c < cs; c++) {
                    minFieldValRef = std::min(minFieldValRef, X[c]);
                    maxFieldValRef = std::max(maxFieldValRef, X[c]);
                }
                for (int c = 0; c < cs; c++) {
                    X[c] = (downscaledFields0.at(c)[i] - minFieldValRef) / (maxFieldValRef - minFieldValRef);
                }
            }

            int upperBounds = regionsEqual ? i : numPoints1;
            for (int j = 0; j < upperBounds; j++) {
                if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                    glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                    glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                    float cellDist = glm::length(pti - ptj);
                    if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                        continue;
                    }
                }
                if (regionsEqual && showCorrelationForClickedPoint
                        && clickedPointGridIdx != uint32_t(i) && clickedPointGridIdx != uint32_t(j)) {
                    continue;
                }
                for (int c = 0; c < cs; c++) {
                    Y[c] = downscaledFields1.at(c)[j];
                    if (std::isnan(Y[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (!isNan) {
                    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                        minFieldVal = minFieldValRef;
                        maxFieldVal = maxFieldValRef;
                        for (int c = 0; c < cs; c++) {
                            minFieldVal = std::min(minFieldVal, Y[c]);
                            maxFieldVal = std::max(maxFieldVal, Y[c]);
                        }
                        for (int c = 0; c < cs; c++) {
                            X[c] = (downscaledFields0.at(c)[i] - minFieldVal) / (maxFieldVal - minFieldVal);
                            Y[c] = (downscaledFields1.at(c)[j] - minFieldVal) / (maxFieldVal - minFieldVal);
                        }
                    }

                    float correlationValue = 0.0f;
                    if (cmt == CorrelationMeasureType::PEARSON) {
                        correlationValue = computePearson2<float>(X.data(), Y.data(), cs);
                    } else if (cmt == CorrelationMeasureType::SPEARMAN) {
                        computeRanks(Y.data(), gridPointRanks.data(), ordinalRankArraySpearman, cs);
                        correlationValue = computePearson2<float>(referenceRanks.data(), gridPointRanks.data(), cs);
                    } else if (cmt == CorrelationMeasureType::KENDALL) {
                        correlationValue = computeKendall(
                                X.data(), Y.data(), cs, jointArray, ordinalRankArray, y);
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                        correlationValue = computeMutualInformationBinned<double>(
                                X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                        correlationValue = computeMutualInformationKraskov<double>(
                                X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                    }
                    if (useAbsoluteCorrelationMeasure) {
                        correlationValue = std::abs(correlationValue);
                    }
                    if (correlationValue < correlationRange.x || correlationValue > correlationRange.y) {
                        continue;
                    }
                    miFieldEntriesThread.emplace_back(correlationValue, i, j);
                }
            }
        }

#ifdef USE_TBB
                return miFieldEntriesThread;
            },
            [&](std::vector<MIFieldEntry> lhs, std::vector<MIFieldEntry> rhs) -> std::vector<MIFieldEntry> {
                std::vector<MIFieldEntry> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
        #pragma omp for ordered schedule(static, 1)
        for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
            #pragma omp ordered
#else
            for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
            {
                miFieldEntries.insert(miFieldEntries.end(), miFieldEntriesThread.begin(), miFieldEntriesThread.end());
            }
        }
    }
#endif
}

void HEBChart::computeCorrelationsSamplingCpu(
        HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries) {
    int cs = getCorrelationMemberCount();

    float minFieldVal = std::numeric_limits<float>::max();
    float maxFieldVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(fieldData->selectedScalarFieldName, fieldIdx);
        const float* field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(fieldData->selectedScalarFieldName, fieldIdx);
            minFieldVal = std::min(minFieldVal, minVal);
            maxFieldVal = std::max(maxFieldVal, maxVal);
        }
    }

    if(samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION)
        correlationSamplingExecuteCpuBayesian(fieldData, miFieldEntries, fields, minFieldVal, maxFieldVal);
    else
        correlationSamplingExecuteCpuDefault(fieldData, miFieldEntries, fields, minFieldVal, maxFieldVal);
}

void HEBChart::correlationSamplingExecuteCpuDefault(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries, const std::vector<const float*>& fields, float minFieldVal, float maxFieldVal){

    const int cs = getCorrelationMemberCount();
    const int numPoints0 = xsd0 * ysd0 * zsd0;
    const int numPoints1 = xsd1 * ysd1 * zsd1;
     int numPairsDownsampled;
    if (isSubselection) 
        numPairsDownsampled = int(subselectionBlockPairs.size());
    else
        numPairsDownsampled = regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1;
    std::vector<float> samples(6 * numSamples);
    generateSamples(samples.data(), numSamples, samplingMethodType);

#ifdef USE_TBB
    miFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPairsDownsampled), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const& r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                const CorrelationMeasureType cmt = correlationMeasureType;
                CORRELATION_CACHE;

                for (int m = r.begin(); m != r.end(); m++) {
#else
    miFieldEntries.reserve(numPairsDownsampled);
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries, numPoints0, numPoints1, cs, k, numBins) \
    shared(numPairsDownsampled, minFieldVal, maxFieldVal, fields, numSamples, samples)
#endif
    {
        const CorrelationMeasureType cmt = correlationMeasureType;
        std::vector<MIFieldEntry> miFieldEntriesThread;
        CORRELATION_CACHE;

#if _OPENMP >= 201107
        #pragma omp for schedule(dynamic)
#endif
        for (int m = 0; m < numPairsDownsampled; m++) {
#endif
            uint32_t i, j;
            if (isSubselection) {
                std::tie(i, j) = subselectionBlockPairs.at(m);
            } else if (regionsEqual) {
                i = (1 + sgl::uisqrt(1 + 8 * uint32_t(m))) / 2;
                j = uint32_t(m) - i * (i - 1) / 2;
            } else {
                i = uint32_t(m / numPoints1);
                j = uint32_t(m % numPoints1);
            }
            if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                float cellDist = glm::length(pti - ptj);
                if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                    continue;
                }
            }
            if (regionsEqual && showCorrelationForClickedPoint
                    && clickedPointGridIdx != uint32_t(i) && clickedPointGridIdx != uint32_t(j)) {
                continue;
            }

            auto region0 = getGridRegionPointIdx(0, i);
            auto region1 = getGridRegionPointIdx(1, j);

            float correlationValueMax = 0.0f;
            bool isValidValue = false;
            for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
                int xi = std::clamp(int(std::round(samples[sampleIdx * 6 + 0] * float(region0.xsr) - 0.5f)), 0, region0.xsr - 1) + region0.xoff;
                int yi = std::clamp(int(std::round(samples[sampleIdx * 6 + 1] * float(region0.ysr) - 0.5f)), 0, region0.ysr - 1) + region0.yoff;
                int zi = std::clamp(int(std::round(samples[sampleIdx * 6 + 2] * float(region0.zsr) - 0.5f)), 0, region0.zsr - 1) + region0.zoff;
                int xj = std::clamp(int(std::round(samples[sampleIdx * 6 + 3] * float(region1.xsr) - 0.5f)), 0, region1.xsr - 1) + region1.xoff;
                int yj = std::clamp(int(std::round(samples[sampleIdx * 6 + 4] * float(region1.ysr) - 0.5f)), 0, region1.ysr - 1) + region1.yoff;
                int zj = std::clamp(int(std::round(samples[sampleIdx * 6 + 5] * float(region1.zsr) - 0.5f)), 0, region1.zsr - 1) + region1.zoff;
                int idxi = IDXS(xi, yi, zi);
                int idxj = IDXS(xj, yj, zj);
                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    X[c] = fields.at(c)[idxi];
                    Y[c] = fields.at(c)[idxj];
                    if (std::isnan(X[c]) || std::isnan(Y[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    continue;
                }

                float correlationValue = 0.0f;
                if (cmt == CorrelationMeasureType::PEARSON) {
                    correlationValue = computePearson2<float>(X.data(), Y.data(), cs);
                } else if (cmt == CorrelationMeasureType::SPEARMAN) {
                    computeRanks(Y.data(), referenceRanks.data(), ordinalRankArrayRef, cs);
                    computeRanks(Y.data(), gridPointRanks.data(), ordinalRankArraySpearman, cs);
                    correlationValue = computePearson2<float>(referenceRanks.data(), gridPointRanks.data(), cs);
                } else if (cmt == CorrelationMeasureType::KENDALL) {
                    correlationValue = computeKendall(
                            X.data(), Y.data(), cs, jointArray, ordinalRankArray, y);
                } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                    for (int c = 0; c < cs; c++) {
                        X[c] = (X[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                        Y[c] = (Y[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                    }
                    correlationValue = computeMutualInformationBinned<double>(
                            X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                    correlationValue = computeMutualInformationKraskov<double>(
                            X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                }
                if (std::abs(correlationValue) >= std::abs(correlationValueMax)) {
                    if (useAbsoluteCorrelationMeasure) {
                        correlationValueMax = std::abs(correlationValue);
                    } else {
                        correlationValueMax = correlationValue;
                    }
                    isValidValue = true;
                }
            }

            if (isValidValue && correlationValueMax >= correlationRange.x && correlationValueMax <= correlationRange.y) {
                miFieldEntriesThread.emplace_back(correlationValueMax, i, j);
            }
        }
#ifdef USE_TBB
        return miFieldEntriesThread;
            },
            [&](std::vector<MIFieldEntry> lhs, std::vector<MIFieldEntry> rhs) -> std::vector<MIFieldEntry> {
                std::vector<MIFieldEntry> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
        #pragma omp for ordered schedule(static, 1)
        for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
            #pragma omp ordered
#else
            for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
            {
                miFieldEntries.insert(miFieldEntries.end(), miFieldEntriesThread.begin(), miFieldEntriesThread.end());
            }
        }
    }
#endif
}
void HEBChart::correlationSamplingExecuteCpuBayesian(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries, const std::vector<const float*>& fields, float minFieldVal, float maxFieldVal){
    const int cs = getCorrelationMemberCount();
    const int numPoints0 = xsd0 * ysd0 * zsd0;
    const int numPoints1 = xsd1 * ysd1 * zsd1;
     int numPairsDownsampled;
    if (isSubselection) 
        numPairsDownsampled = int(subselectionBlockPairs.size());
    else
        numPairsDownsampled = regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1;

    miFieldEntries.reserve(numPairsDownsampled);

    std::atomic<int> cur_pair{};
    const auto correlationType = correlationMeasureType;
    const int bayOptIterationCount = std::max(0, numSamples - BayOpt::Params::init_randomsampling::samples());
    const int mutualInformationK = k;
    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    std::vector<std::vector<MIFieldEntry>> miFieldEntriesThread(threads.size());
    auto thread_func = [&](int id){
        for(int p = cur_pair++; p < numPairsDownsampled; p = cur_pair++){
            uint32_t i, j;
            if (isSubselection) {
                std::tie(i, j) = subselectionBlockPairs.at(p);
            } else if (regionsEqual) {
                i = (1 + sgl::uisqrt(1 + 8 * uint32_t(p))) / 2;
                j = uint32_t(p) - i * (i - 1) / 2;
            } else {
                i = uint32_t(p / numPoints1);
                j = uint32_t(p % numPoints1);
            }
            if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                float cellDist = glm::length(pti - ptj);
                if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                    continue;
                }
            }
            if (regionsEqual && showCorrelationForClickedPoint
                    && clickedPointGridIdx != uint32_t(i) && clickedPointGridIdx != uint32_t(j)) {
                continue;
            }

            auto region0 = getGridRegionPointIdx(0, i);
            auto region1 = getGridRegionPointIdx(1, j);
            const std::array<int, 6> region_min{region0.xmin, region0.ymin, region0.zmin, region1.xmin, region1.ymin, region1.zmin};
            const std::array<int, 6> region_max{region0.xmax, region0.ymax, region0.zmax, region1.xmax, region1.ymax, region1.zmax};

            limbo::bayes_opt::BOptimizer<BayOpt::Params> optimizer;
            BayOpt::Params::stop_maxiterations::set_iterations(bayOptIterationCount);
            switch(correlationType){
            case CorrelationMeasureType::PEARSON:
                optimizer.optimize(BayOpt::Eval<BayOpt::PearsonFunctor>{fields, region_min, region_max, cs, xs, ys});
                break;
            case CorrelationMeasureType::SPEARMAN:
                optimizer.optimize(BayOpt::Eval<BayOpt::SpearmanFunctor>{fields, region_min, region_max, cs, xs, ys});
                break;
            case CorrelationMeasureType::KENDALL: 
                optimizer.optimize(BayOpt::Eval<BayOpt::KendallFunctor>{fields, region_min, region_max, cs, xs, ys});
                break;
            case CorrelationMeasureType::MUTUAL_INFORMATION_BINNED: 
                optimizer.optimize(BayOpt::Eval<BayOpt::MutualBinnedFunctor>{fields, region_min, region_max, cs, xs, ys, BayOpt::MutualBinnedFunctor{minFieldVal, maxFieldVal, numBins}});
                break;
            case CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV: 
                optimizer.optimize(BayOpt::Eval<BayOpt::MutualFunctor>{fields, region_min, region_max, cs, xs, ys, {mutualInformationK}});
                break;
            default: assert(false && "Unimplemented Correlation measure type");
            }

            float correlationValueMax = optimizer.best_observation()(0);
            if (correlationValueMax >= correlationRange.x && correlationValueMax <= correlationRange.y) {
                miFieldEntriesThread[id].emplace_back(correlationValueMax, i, j);
            }
        }
    };
    int id{};
    for(auto& t: threads)
        t = std::thread(thread_func, id++);
    for(auto& t: threads)
        t.join();
    // merging threads
    for(const auto& fieldEntry: miFieldEntriesThread)
        miFieldEntries.insert(miFieldEntries.end(), fieldEntry.begin(), fieldEntry.end());
}

std::shared_ptr<HEBChartFieldCache> HEBChart::getFieldCache(HEBChartFieldData* fieldData) {
    int cs = getCorrelationMemberCount();
    auto fieldCache = std::make_shared<HEBChartFieldCache>();
    fieldCache->minFieldVal = std::numeric_limits<float>::max();
    fieldCache->maxFieldVal = std::numeric_limits<float>::lowest();
    fieldCache->fieldEntries.reserve(cs);
    fieldCache->fieldImageViews.reserve(cs);
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::DeviceCacheEntry fieldEntry = getFieldEntryDevice(fieldData->selectedScalarFieldName, fieldIdx);
        fieldCache->fieldEntries.push_back(fieldEntry);
        fieldCache->fieldImageViews.push_back(fieldEntry->getVulkanImageView());
        if (fieldEntry->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            fieldEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
        }
        if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(fieldData->selectedScalarFieldName, fieldIdx);
            fieldCache->minFieldVal = std::min(fieldCache->minFieldVal, minVal);
            fieldCache->maxFieldVal = std::max(fieldCache->maxFieldVal, maxVal);
        }
    }
    return fieldCache;
}

sgl::vk::BufferPtr HEBChart::computeCorrelationsForRequests(
        std::vector<CorrelationRequestData>& requests,
        std::shared_ptr<HEBChartFieldCache>& fieldCache, bool isFirstBatch) {
    int cs = getCorrelationMemberCount();

    computeRenderer->beginCommandBuffer();
    if (isFirstBatch) {
        correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
        correlationComputePass->setVolumeData(volumeData.get(), cs, false);
        correlationComputePass->setCorrelationMemberCount(cs);
        correlationComputePass->setNumBins(numBins);
        correlationComputePass->setKraskovNumNeighbors(k);
        correlationComputePass->setFieldImageViews(fieldCache->fieldImageViews);
        correlationComputePass->buildIfNecessary();
    }

    // Upload the requests to the GPU, compute the correlations, and download the correlations back to the CPU.
    requestsStagingBuffer->uploadData(sizeof(CorrelationRequestData) * requests.size(), requests.data());
    requestsStagingBuffer->copyDataTo(requestsBuffer, computeRenderer->getVkCommandBuffer());
    computeRenderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            requestsStagingBuffer);
    computeRenderer->pushConstants(
            correlationComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
            //glm::uvec2(batchSizeCells * uint32_t(numSamples), uint32_t(cs)));
            glm::uvec2(requests.size(), uint32_t(cs)));
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
        computeRenderer->pushConstants(
                correlationComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 2 * sizeof(float),
                glm::vec2(fieldCache->minFieldVal, fieldCache->maxFieldVal));
    }
    correlationComputePass->setNumRequests(uint32_t(requests.size()));
    correlationComputePass->render();
    computeRenderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            requestsStagingBuffer);
    correlationOutputBuffer->copyDataTo(correlationOutputStagingBuffer, computeRenderer->getVkCommandBuffer());
    computeRenderer->endCommandBuffer();
    computeRenderer->submitToQueue({}, {}, fence, VK_PIPELINE_STAGE_TRANSFER_BIT);
    fence->wait();
    fence->reset();


    return correlationOutputStagingBuffer;
}

void HEBChart::createBatchCacheData(uint32_t& batchSizeSamplesMax) {
    int cs = getCorrelationMemberCount();
    const uint32_t batchSizeSamplesMaxAllCs = 1 << 17; // Up to 131072 samples per batch.
    if(batchSizeSamplesMax == 0)
        batchSizeSamplesMax = batchSizeSamplesMaxAllCs;
    if (cs > 100) {
        double factorN = double(cs) / 100.0 * std::log2(double(cs) / 100.0 + 1.0);
        batchSizeSamplesMax = std::ceil(double(batchSizeSamplesMax) / factorN);
        batchSizeSamplesMax = uint32_t(sgl::nextPowerOfTwo(int(batchSizeSamplesMax)));
    }

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (!requestsBuffer) {
        computeRenderer = new sgl::vk::Renderer(device, 100);
        if (device->getGraphicsQueue() == device->getComputeQueue()) {
            supportsAsyncCompute = false;
        }
        fence = std::make_shared<sgl::vk::Fence>(device);

        sgl::vk::CommandPoolType commandPoolType{};
        commandPoolType.queueFamilyIndex = device->getComputeQueueIndex();
        commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandBuffer = device->allocateCommandBuffer(commandPoolType, &commandPool);
        computeRenderer->setCustomCommandBuffer(commandBuffer, false);

        requestsBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(CorrelationRequestData) * batchSizeSamplesMax,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        requestsStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(CorrelationRequestData) * batchSizeSamplesMax,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU);
        correlationOutputBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(float) * batchSizeSamplesMax,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        correlationOutputStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(float) * batchSizeSamplesMax,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_TO_CPU);

        correlationComputePass = std::make_shared<CorrelationComputePass>(computeRenderer);
        correlationComputePass->setUseRequestEvaluationMode(true);
        correlationComputePass->setRequestsBuffer(requestsBuffer);
        correlationComputePass->setOutputBuffer(correlationOutputBuffer);
    }
}

void HEBChart::computeCorrelationsSamplingGpu(
        HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries) {
    if(samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION)
        correlationSamplingExecuteGpuBayesian(fieldData, miFieldEntries);
    else
        correlationSamplingExecuteGpuDefault(fieldData, miFieldEntries);
}

void HEBChart::correlationSamplingExecuteGpuDefault(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries){
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;

    int numPairsDownsampled;
    if (isSubselection) {
        numPairsDownsampled = int(subselectionBlockPairs.size());
    } else {
        numPairsDownsampled = regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1;
    }

    std::vector<float> samples(6 * numSamples);
    generateSamples(samples.data(), numSamples, samplingMethodType);

    uint32_t batchSizeSamplesMax{};
    createBatchCacheData(batchSizeSamplesMax);
    uint32_t batchSizeCellsMax = std::max(batchSizeSamplesMax / numSamples, uint32_t(1));
    uint32_t numBatches = sgl::uiceil(uint32_t(numPairsDownsampled), batchSizeCellsMax);

    auto fieldCache = getFieldCache(fieldData);

    uint32_t cellIdxOffset = 0;
    auto numCellsLeft = uint32_t(numPairsDownsampled);
    std::vector<CorrelationRequestData> requests;
    requests.reserve(batchSizeCellsMax);
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchSizeCells;
        if (numCellsLeft > batchSizeCellsMax) {
            batchSizeCells = batchSizeCellsMax;
            numCellsLeft -= batchSizeCellsMax;
        } else if (numCellsLeft > 0) {
            batchSizeCells = numCellsLeft;
            numCellsLeft = 0;
        } else {
            break;
        }

        // Create the batch data to upload to the device.
        uint32_t loopMax = cellIdxOffset + batchSizeCells;
        requests.clear();
#ifdef USE_TBB
        requests = tbb::parallel_reduce(
            tbb::blocked_range<uint32_t>(cellIdxOffset, loopMax), std::vector<CorrelationRequestData>(),
            [&](tbb::blocked_range<int> const& r, std::vector<CorrelationRequestData> requestsThread) -> std::vector<CorrelationRequestData> {
                for (int m = r.begin(); m != r.end(); m++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(requests, numPoints0, numPoints1) \
        shared(cellIdxOffset, loopMax, numSamples, samples)
#endif
        {
            std::vector<CorrelationRequestData> requestsThread;

#if _OPENMP >= 201107
            #pragma omp for schedule(dynamic)
#endif
            for (uint32_t m = cellIdxOffset; m < loopMax; m++) {
#endif
                uint32_t i, j;
                if (isSubselection) {
                    std::tie(i, j) = subselectionBlockPairs.at(m);
                } else if (regionsEqual) {
                    i = (1 + sgl::uisqrt(1 + 8 * m)) / 2;
                    j = uint32_t(m) - i * (i - 1) / 2;
                } else {
                    i = m / uint32_t(numPoints1);
                    j = m % uint32_t(numPoints1);
                }
                if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                    glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                    glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                    float cellDist = glm::length(pti - ptj);
                    if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                        continue;
                    }
                }
                if (regionsEqual && showCorrelationForClickedPoint
                        && clickedPointGridIdx != i && clickedPointGridIdx != j) {
                    continue;
                }

                auto region0 = getGridRegionPointIdx(0, i);
                auto region1 = getGridRegionPointIdx(1, j);

                CorrelationRequestData request;
                for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
                    int xi = std::clamp(int(std::round(samples[sampleIdx * 6 + 0] * float(region0.xsr) - 0.5f)), 0, region0.xsr - 1) + region0.xoff;
                    int yi = std::clamp(int(std::round(samples[sampleIdx * 6 + 1] * float(region0.ysr) - 0.5f)), 0, region0.ysr - 1) + region0.yoff;
                    int zi = std::clamp(int(std::round(samples[sampleIdx * 6 + 2] * float(region0.zsr) - 0.5f)), 0, region0.zsr - 1) + region0.zoff;
                    int xj = std::clamp(int(std::round(samples[sampleIdx * 6 + 3] * float(region1.xsr) - 0.5f)), 0, region1.xsr - 1) + region1.xoff;
                    int yj = std::clamp(int(std::round(samples[sampleIdx * 6 + 4] * float(region1.ysr) - 0.5f)), 0, region1.ysr - 1) + region1.yoff;
                    int zj = std::clamp(int(std::round(samples[sampleIdx * 6 + 5] * float(region1.zsr) - 0.5f)), 0, region1.zsr - 1) + region1.zoff;
                    request.i = i;
                    request.j = j;
                    request.xi = uint32_t(xi);
                    request.yi = uint32_t(yi);
                    request.zi = uint32_t(zi);
                    request.xj = uint32_t(xj);
                    request.yj = uint32_t(yj);
                    request.zj = uint32_t(zj);
                    requestsThread.emplace_back(request);
                }
            }

#ifdef USE_TBB
                return requestsThread;
            },
            [&](std::vector<CorrelationRequestData> lhs, std::vector<CorrelationRequestData> rhs) -> std::vector<CorrelationRequestData> {
                std::vector<CorrelationRequestData> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
            #pragma omp for ordered schedule(static, 1)
            for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
                #pragma omp ordered
#else
                for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
                {
                    requests.insert(requests.end(), requestsThread.begin(), requestsThread.end());
                }
            }
        }
#endif
        cellIdxOffset += batchSizeCells;

        auto outputBuffer = computeCorrelationsForRequests(requests, fieldCache, batchIdx == 0);

        // Finally, insert the values of this batch.
        auto* correlationValues = static_cast<float*>(outputBuffer->mapMemory());
        auto numValidCells = int(requests.size() / size_t(numSamples));
#ifdef USE_TBB
        std::vector<MIFieldEntry> newMiFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPairsDownsampled), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const& r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                const CorrelationMeasureType cmt = correlationMeasureType;
                CORRELATION_CACHE;

                for (int m = r.begin(); m != r.end(); m++) {
#else
        miFieldEntries.reserve(numPairsDownsampled);
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(miFieldEntries, requests, correlationValues, numValidCells)
#endif
        {
            std::vector<MIFieldEntry> miFieldEntriesThread;
#if _OPENMP >= 201107
            #pragma omp for
#endif
            for (int cellIdx = 0; cellIdx < numValidCells; cellIdx++) {
#endif
                int requestIdx = cellIdx * numSamples;
                auto i = int(requests.at(requestIdx).i);
                auto j = int(requests.at(requestIdx).j);
                float correlationValueMax = 0.0f;
                bool isValidValue = false;
                for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {
                    float correlationValue = correlationValues[requestIdx];
                    if (std::abs(correlationValue) >= std::abs(correlationValueMax)) {
                        if (useAbsoluteCorrelationMeasure) {
                            correlationValueMax = std::abs(correlationValue);
                        } else {
                            correlationValueMax = correlationValue;
                        }
                        isValidValue = true;
                    }
                    requestIdx++;
                }
                if (isValidValue && correlationValueMax >= correlationRange.x && correlationValueMax <= correlationRange.y) {
                    miFieldEntriesThread.emplace_back(correlationValueMax, i, j);
                }
            }
#ifdef USE_TBB
            return miFieldEntriesThread;
            },
            [&](std::vector<MIFieldEntry> lhs, std::vector<MIFieldEntry> rhs) -> std::vector<MIFieldEntry> {
                std::vector<MIFieldEntry> listOut = std::move(lhs);
                listOut.insert(listOut.end(), std::make_move_iterator(rhs.begin()), std::make_move_iterator(rhs.end()));
                return listOut;
            });
#else

#if _OPENMP >= 201107
            #pragma omp for ordered schedule(static, 1)
            for (int threadIdx = 0; threadIdx < omp_get_num_threads(); ++threadIdx) {
                #pragma omp ordered
#else
                for (int threadIdx = 0; threadIdx < 1; ++threadIdx) {
#endif
                {
                    miFieldEntries.insert(miFieldEntries.end(), miFieldEntriesThread.begin(), miFieldEntriesThread.end());
                }
            }
        }
#endif
#ifdef USE_TBB
        miFieldEntries.insert(miFieldEntries.end(), newMiFieldEntries.begin(), newMiFieldEntries.end());
#endif
        outputBuffer->unmapMemory();
    }
}

void HEBChart::correlationSamplingExecuteGpuBayesian(HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries){
    // For bayesian gpu sampling the following steps are performed
    // 1. For each thread one optimizer per box pair is allocated
    // 2. Each thread draws batchSize box pairs for execution (thread safe), and executes them
    // 3. Init: BatchSize * InitSamples samples are generated randomly
    // 3.1      Evaluate all samples simultaneously and add them tho the optimizer
    // 4. Draw next sample point for each optimizer -> batchSize samples
    // 5. Evaluate samples and add one sample to each optimizer
    // 6. goto 4. until end
    // 7. Writeback of best sample
    // main algorithm is implemented in thread_func
    
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;
    const int bayOptIterationCount = std::max(0, numSamples - BayOpt::Params::init_randomsampling::samples());
    BayOpt::Params::stop_maxiterations::set_iterations(bayOptIterationCount);

    int numPairsDownsampled;
    if (isSubselection) {
        numPairsDownsampled = int(subselectionBlockPairs.size());
    } else {
        numPairsDownsampled = regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1;
    }
    constexpr int batchSize = 1000;
    uint32_t batchSizeNeeded = batchSize * BayOpt::Params::init_randomsampling::samples();
    createBatchCacheData(batchSizeNeeded);
    auto fieldCache = getFieldCache(fieldData);
    std::atomic<int> cur_pair{};
    bool gpu_first_call{true};
    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    std::vector<std::vector<MIFieldEntry>> threadFieldEntries(threads.size());
    std::mutex gpu_mutex;

    auto generate_requests = [&](const std::vector<float>& sample_positions, int start_pair_index, int end_pair_index, int samples_per_pair){
        std::vector<CorrelationRequestData> requests;
        for(int q: BayOpt::i_range(start_pair_index, end_pair_index)){
            uint32_t i,j;
            if (isSubselection) {
                std::tie(i, j) = subselectionBlockPairs.at(q);
            } else if (regionsEqual) {
                i = (1 + sgl::uisqrt(1 + 8 * q)) / 2;
                j = uint32_t(q) - i * (i - 1) / 2;
            } else {
                i = q / uint32_t(numPoints1);
                j = q % uint32_t(numPoints1);
            }
            
            if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                glm::vec3 pti(i % uint32_t(xsd0), (i / uint32_t(xsd0)) % uint32_t(ysd0), i / uint32_t(xsd0 * ysd0));
                glm::vec3 ptj(j % uint32_t(xsd1), (j / uint32_t(xsd1)) % uint32_t(ysd1), j / uint32_t(xsd1 * ysd1));
                float cellDist = glm::length(pti - ptj);
                if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                    continue;
                }
            }
            if (regionsEqual && showCorrelationForClickedPoint
                && clickedPointGridIdx != i && clickedPointGridIdx != j) 
                continue;
            
            auto region0 = getGridRegionPointIdx(0, i);
            auto region1 = getGridRegionPointIdx(1, j);

            CorrelationRequestData request;
            for(int sampleIdx: BayOpt::i_range(samples_per_pair)){
                int xi = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 0] * float(region0.xsr) - 0.5f)), 0, region0.xsr - 1) + region0.xoff;
                int yi = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 1] * float(region0.ysr) - 0.5f)), 0, region0.ysr - 1) + region0.yoff;
                int zi = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 2] * float(region0.zsr) - 0.5f)), 0, region0.zsr - 1) + region0.zoff;
                int xj = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 3] * float(region1.xsr) - 0.5f)), 0, region1.xsr - 1) + region1.xoff;
                int yj = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 4] * float(region1.ysr) - 0.5f)), 0, region1.ysr - 1) + region1.yoff;
                int zj = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 5] * float(region1.zsr) - 0.5f)), 0, region1.zsr - 1) + region1.zoff;
                request.i = i;
                request.j = j;
                request.xi = uint32_t(xi);
                request.yi = uint32_t(yi);
                request.zi = uint32_t(zi);
                request.xj = uint32_t(xj);
                request.yj = uint32_t(yj);
                request.zj = uint32_t(zj);
                requests.emplace_back(request);
            }
        }
        return requests;
    };
    auto thread_func = [&](int thread_id){
        using model_t = limbo::model::GP<BayOpt::Params>;
        limbo::opt::NLOptNoGrad<BayOpt::Params, nlopt::GN_DIRECT_L_RAND> acqui_optimizer;
        for(int p = cur_pair.fetch_add(batchSize); p < numPairsDownsampled; p = cur_pair.fetch_add(batchSize)){
            const int true_batch_size = std::min<int>(numPairsDownsampled - p, batchSize);
            // 1. optimizer allocation -----------------------------------------------------------------------
            std::vector<model_t> optimizers(true_batch_size);
            // 2. drawing a batch size already done in the for loop ------------------------------------------
            // 3. Init ---------------------------------------------------------------------------------------
            std::vector<float> sample_positions(true_batch_size * 6);
            generateSamples(sample_positions.data(), true_batch_size, SamplingMethodType::QUASIRANDOM_HALTON);
            // generate requests
            auto requests = generate_requests(sample_positions, p, p + true_batch_size, BayOpt::Params::init_randomsampling::samples());
            // evaluate requests and download scores (thread safe)
            {
                std::scoped_lock lock(gpu_mutex);
                auto outputBuffer = computeCorrelationsForRequests(requests, fieldCache, gpu_first_call);
                gpu_first_call = false;
                auto* correlationValues = static_cast<float*>(outputBuffer->mapMemory());
                Eigen::VectorXd p(6);
                Eigen::VectorXd v(1);
                assert(p.size() == 6 && v.size() == 1);
                for(int i: BayOpt::i_range(true_batch_size)){
                    for(int s: BayOpt::i_range(BayOpt::Params::init_randomsampling::samples())){
                        int lin_index = i * BayOpt::Params::init_randomsampling::samples() + s;
                        v(0) = correlationValues[lin_index];
                        if(std::isnan(v(0)) || std::isinf(v(0))) v(0) = -1;
                        std::copy(sample_positions.begin() + 6 * lin_index, sample_positions.begin() + 7 * lin_index, p.data());
                        optimizers[i].add_sample(p, v);
                    }
                }
                outputBuffer->unmapMemory();
            }

            for(int i: BayOpt::i_range(bayOptIterationCount)){
                // 4. Drawing new samples from the model -------------------------------------------
                for(int o: BayOpt::i_range(true_batch_size)){
                    limbo::acqui::UCB<BayOpt::Params, model_t> acqui_fun(optimizers[o], i);
                    
                    auto acqui_optimization = [&](const Eigen::VectorXd& x, bool g) {return acqui_fun(x, limbo::FirstElem{}, g);};
                    auto starting_point = limbo::tools::random_vector_bounded(6);
                    auto new_sample = acqui_optimizer(acqui_optimization, starting_point, true);

                    std::copy(new_sample.data(), new_sample.data() + 6, sample_positions.begin());
                }

                // 5. evaluating the samples and adding them to the model parameters
                requests = generate_requests(sample_positions, p, p + true_batch_size, 1);
                {
                    std::scoped_lock lock(gpu_mutex);
                    auto outputBuffer = computeCorrelationsForRequests(requests, fieldCache, gpu_first_call);
                    gpu_first_call = false;
                    auto* correlationValues = static_cast<float*>(outputBuffer->mapMemory());
                    Eigen::VectorXd p(6);
                    Eigen::VectorXd v(1);
                    assert(p.size() == 6 && v.size() == 1);
                    for(int o: BayOpt::i_range(true_batch_size)){
                        v(0) = correlationValues[o];
                        if(std::isnan(v(0)) || std::isinf(v(0))) v(0) = -1;
                        std::copy(sample_positions.begin() + 6 * o, sample_positions.begin() + 7 * o, p.data());
                        optimizers[o].add_sample(p, v);
                    }
                    outputBuffer->unmapMemory();
                }
                // 6. loop to 4.
            }
            // 7. Writing back the best results -------------------------------------------
            for(int o: BayOpt::i_range(true_batch_size)){
                double max_val{std::numeric_limits<float>::lowest()};
                for(auto& v: optimizers[o].observations_matrix().reshaped())
                    max_val = std::max(max_val, v);
                
                uint32_t i, j, q = p + o;
                if (isSubselection) {
                    std::tie(i, j) = subselectionBlockPairs.at(q);
                } else if (regionsEqual) {
                    i = (1 + sgl::uisqrt(1 + 8 * q)) / 2;
                    j = uint32_t(q) - i * (i - 1) / 2;
                } else {
                    i = q / uint32_t(numPoints1);
                    j = q % uint32_t(numPoints1);
                }
                threadFieldEntries[thread_id].emplace_back(float(max_val), i, j);
            }
        }

    };

    int thread_id{};
    for(auto& t: threads)
        t = std::thread(thread_func, thread_id++);
    for(auto& t: threads)
        t.join();

    // merging the outputs
    for(const auto& fieldEntry: threadFieldEntries)
        miFieldEntries.insert(miFieldEntries.end(), fieldEntry.begin(), fieldEntry.end());
}

HEBChart::PerfStatistics HEBChart::computeCorrelationsBlockPairs(
        const std::vector<std::pair<uint32_t, uint32_t>>& blockPairs) {
    HEBChart::PerfStatistics statistics{};
    isSubselection = true;
    subselectionBlockPairs = blockPairs;

    auto startTime = std::chrono::system_clock::now();
    std::vector<MIFieldEntry> miFieldEntries;
    computeCorrelations(fieldDataArray.front().get(), {}, {}, miFieldEntries);
    auto endTime = std::chrono::system_clock::now();
    auto elapsedTime = std::chrono::duration<double>(endTime - startTime);
    statistics.timeElapsedSeconds = elapsedTime.count();

    for (auto& entry : miFieldEntries) {
        statistics.maximumValues.push_back(entry.correlationValue);
    }
    isSubselection = false;
    subselectionBlockPairs.clear();

    return statistics;
}

void HEBChart::computeAllCorrelationsBlockPair(uint32_t i, uint32_t j, std::vector<float>& allValues) {
    ;
}
