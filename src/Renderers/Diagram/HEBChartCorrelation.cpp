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

#define NOMINMAX

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <thread>
#include <iomanip>

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Utils/DeviceThreadInfo.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Utils/Semaphore.hpp>

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "Calculators/Correlation.hpp"
#include "Calculators/CorrelationCalculator.hpp"
#include "Calculators/MutualInformation.hpp"
#include "Test/MultivariateGaussian.hpp"
#include "HEBChart.hpp"
#include "BayOpt.hpp"

// declaring static iterations variable
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::stop_maxiterations, iterations);
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::init_randomsampling, samples);
BO_DECLARE_DYN_PARAM(int, BayOpt::Params::opt_nloptnograd, iterations);

HEBChartFieldData::~HEBChartFieldData() {
    clearMemoryTokens();
}

void HEBChartFieldData::createFieldCache(
        VolumeData* _volumeData, bool regionsEqual, GridRegion r0, GridRegion r1, int mdfx, int mdfy, int mdfz,
        bool isEnsembleMode, CorrelationDataMode dataMode, bool useBufferTiling) {
    if (regionsEqual == cachedRegionsEqual && r0 == cachedR0 && r1 == cachedR1
            && mdfx == cachedMdfx && mdfy == cachedMdfy && mdfz == cachedMdfz
            && isEnsembleMode == cachedIsEnsembleMode && dataMode == cachedDataMode
            && useBufferTiling == cachedUseBufferTiling) {
        return;
    }
    volumeData = _volumeData;
    cachedRegionsEqual = regionsEqual;
    cachedR0 = r0;
    cachedR1 = r1;
    cachedMdfx = mdfx;
    cachedMdfy = mdfy;
    cachedMdfz = mdfz;
    cachedIsEnsembleMode = isEnsembleMode;
    cachedDataMode = dataMode;
    cachedUseBufferTiling = useBufferTiling;
    minFieldVal = std::numeric_limits<float>::max();
    maxFieldVal = std::numeric_limits<float>::lowest();
    minFieldVal2 = std::numeric_limits<float>::max();
    maxFieldVal2 = std::numeric_limits<float>::lowest();
    clearMemoryTokens();
    fieldImageViews.clear();
    fieldBuffers.clear();
    fieldImageViewsR1.clear();
    fieldBuffersR1.clear();
    if (useTwoFields) {
        fieldImageViews2.clear();
        fieldBuffers2.clear();
        fieldImageViewsR12.clear();
        fieldBuffersR12.clear();
    }

    int cs = getCorrelationMemberCount();
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        fieldImageViews.resize(cs);
        if (!regionsEqual) {
            fieldImageViewsR1.resize(cs);
        }
        if (useTwoFields) {
            fieldImageViews2.resize(cs);
            if (!regionsEqual) {
                fieldImageViewsR12.resize(cs);
            }
        }
    } else if (dataMode == CorrelationDataMode::BUFFER_ARRAY) {
        fieldBuffers.resize(cs);
        if (!regionsEqual) {
            fieldBuffersR1.resize(cs);
        }
        if (useTwoFields) {
            fieldBuffers2.resize(cs);
            if (!regionsEqual) {
                fieldBuffersR12.resize(cs);
            }
        }
    }

    deviceMemoryTokens.resize(cs);
    if (!regionsEqual) {
        deviceMemoryTokensR1.resize(cs);
    }
    if (useTwoFields) {
        deviceMemoryTokens2.resize(cs);
        if (!regionsEqual) {
            deviceMemoryTokensR12.resize(cs);
        }
    }

    elapsedTimeDownsampling = 0;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        computeDownscaledFields(0, 0, fieldIdx);
        if (!regionsEqual) {
            computeDownscaledFields(1, 0, fieldIdx);
        }
        auto [minVal, maxVal] = getMinMaxScalarFieldValue(selectedScalarFieldName1, fieldIdx);
        minFieldVal = std::min(minFieldVal, minVal);
        maxFieldVal = std::max(maxFieldVal, maxVal);

        if (useTwoFields) {
            computeDownscaledFields(0, 1, fieldIdx);
            if (!regionsEqual) {
                computeDownscaledFields(1, 1, fieldIdx);
            }
            auto [minVal2, maxVal2] = getMinMaxScalarFieldValue(selectedScalarFieldName2, fieldIdx);
            minFieldVal2 = std::min(minFieldVal2, minVal2);
            maxFieldVal2 = std::max(maxFieldVal2, maxVal2);
        }
    }
    std::cout << "Elapsed time downsampling: " << elapsedTimeDownsampling << "ms" << std::endl;
}

void HEBChartFieldData::clearMemoryTokens() {
    for (auto token : deviceMemoryTokens) {
        volumeData->popAuxiliaryMemoryDevice(token);
    }
    for (auto token : deviceMemoryTokensR1) {
        volumeData->popAuxiliaryMemoryDevice(token);
    }
    deviceMemoryTokens.clear();
    deviceMemoryTokensR1.clear();

    if (useTwoFields) {
        for (auto token : deviceMemoryTokens2) {
            volumeData->popAuxiliaryMemoryDevice(token);
        }
        for (auto token : deviceMemoryTokensR12) {
            volumeData->popAuxiliaryMemoryDevice(token);
        }
        deviceMemoryTokens2.clear();
        deviceMemoryTokensR12.clear();
    }
}

int HEBChartFieldData::getCorrelationMemberCount() {
    return cachedIsEnsembleMode ? volumeData->getEnsembleMemberCount() : volumeData->getTimeStepCount();
}

VolumeData::HostCacheEntry HEBChartFieldData::getFieldEntryCpu(const std::string& fieldName, int fieldIdx) {
    std::lock_guard<std::mutex> lock(volumeDataMutex);
    VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, fieldName, cachedIsEnsembleMode ? -1 : fieldIdx, cachedIsEnsembleMode ? fieldIdx : -1);
    return ensembleEntryField;
}

std::pair<float, float> HEBChartFieldData::getMinMaxScalarFieldValue(const std::string& fieldName, int fieldIdx) {
    std::lock_guard<std::mutex> lock(volumeDataMutex);
    return volumeData->getMinMaxScalarFieldValue(
            fieldName, cachedIsEnsembleMode ? -1 : fieldIdx, cachedIsEnsembleMode ? fieldIdx : -1);
}

void HEBChartFieldData::computeDownscaledFields(int idx, int varNum, int fieldIdx) {
    GridRegion r = idx == 0 ? cachedR0 : cachedR1;
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int dfx = cachedMdfx;
    int dfy = cachedMdfy;
    int dfz = cachedMdfz;
    int xsd = sgl::iceil(r.xsr, dfx);
    int ysd = sgl::iceil(r.ysr, dfy);
    int zsd = sgl::iceil(r.zsr, dfz);
    int numPoints = xsd * ysd * zsd;
    const int tileSizeX = 8;
    const int tileSizeY = 8;
    const int tileSizeZ = 4;
    std::string scalarFieldName;
    if (varNum == 0) {
        scalarFieldName = selectedScalarFieldName1;
    } else {
        scalarFieldName = selectedScalarFieldName2;
    }

    VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(scalarFieldName, fieldIdx);

    const float* field = fieldEntry->data<float>();
    auto* downscaledField = new float[numPoints];

//#define DOWNSAMPLING_MAX
    std::chrono::time_point<std::chrono::system_clock> startTime = std::chrono::system_clock::now();

    for (int zd = 0; zd < zsd; zd++) {
        for (int yd = 0; yd < ysd; yd++) {
            for (int xd = 0; xd < xsd; xd++) {
#ifdef DOWNSAMPLING_MAX
                float valueMean = 0.0f;
#else
                float valueMean = 0.0f;
#endif
                int numValid = 0;
                for (int zo = 0; zo < dfz; zo++) {
                    for (int yo = 0; yo < dfy; yo++) {
                        for (int xo = 0; xo < dfx; xo++) {
                            int x = r.xoff + xd * dfx + xo;
                            int y = r.yoff + yd * dfy + yo;
                            int z = r.zoff + zd * dfz + zo;
                            if (x <= r.xmax && y <= r.ymax && z <= r.zmax) {
                                float val = field[IDXS(x, y, z)];
                                if (!std::isnan(val)) {
#ifdef DOWNSAMPLING_MAX
                                    if (std::abs(val) > std::abs(valueMean)) {
                                        valueMean = val;
                                    }
#else
                                    valueMean += val;
#endif
                                    numValid++;
                                }
                            }
                        }
                    }
                }
                if (numValid > 0) {
#ifndef DOWNSAMPLING_MAX
                    valueMean = valueMean / float(numValid);
#endif
                } else {
                    valueMean = std::numeric_limits<float>::quiet_NaN();
                }
                downscaledField[IDXSD(xd, yd, zd)] = valueMean;
            }
        }
    }

    size_t sizeInBytes = size_t(numPoints) * sizeof(float);
    auto* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (cachedDataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        sgl::vk::ImageSettings imageSettings{};
        imageSettings.width = uint32_t(xsd);
        imageSettings.height = uint32_t(ysd);
        imageSettings.depth = uint32_t(zsd);
        imageSettings.imageType = VK_IMAGE_TYPE_3D;
        imageSettings.format = VK_FORMAT_R32_SFLOAT;
        imageSettings.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
        auto image = std::make_shared<sgl::vk::Image>(device, imageSettings);
        image->uploadData(sizeInBytes, downscaledField);
        sizeInBytes = image->getDeviceMemoryAllocationSize();
        auto imageView = std::make_shared<sgl::vk::ImageView>(image, VK_IMAGE_VIEW_TYPE_3D);
        if (varNum == 0) {
            if (idx == 0) {
                fieldImageViews.at(fieldIdx) = imageView;
            } else {
                fieldImageViewsR1.at(fieldIdx) = imageView;
            }
        } else {
            if (idx == 0) {
                fieldImageViews2.at(fieldIdx) = imageView;
            } else {
                fieldImageViewsR12.at(fieldIdx) = imageView;
            }
        }
    } else {
        if (cachedUseBufferTiling) {
            sizeInBytes =
                    sizeof(float) * size_t(sgl::uiceil(uint32_t(xsd), tileSizeX) * tileSizeX)
                    * size_t(sgl::uiceil(uint32_t(ysd), tileSizeY) * tileSizeY)
                    * size_t(sgl::uiceil(uint32_t(zsd), tileSizeZ) * tileSizeZ);
        }
        VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        auto buffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeInBytes, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);

        if (cachedUseBufferTiling) {
            size_t numEntries = sizeInBytes / sizeof(float);
            auto* linearBufferData = downscaledField;
            auto* tiledBufferData = new float[numEntries];
            auto tileNumVoxels = uint32_t(tileSizeX) * uint32_t(tileSizeY) * uint32_t(tileSizeZ);
            uint32_t xst = sgl::uiceil(xsd, tileSizeX);
            uint32_t yst = sgl::uiceil(ysd, tileSizeY);
            uint32_t zst = sgl::uiceil(zsd, tileSizeZ);
            uint32_t numTilesTotal = xst * yst * zst;
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<uint32_t>(0, numTilesTotal), [&](auto const& r) {
                for (auto tileIdx = r.begin(); tileIdx != r.end(); tileIdx++) {
#else
#if _OPENMP >= 200805
            #pragma omp parallel for default(none) shared(numTilesTotal, tileNumVoxels, xst, yst, zst) \
            shared(tileSizeX, tileSizeY, tileSizeZ, linearBufferData, tiledBufferData, xsd, ysd, zsd)
#endif
            for (uint32_t tileIdx = 0; tileIdx < numTilesTotal; tileIdx++) {
#endif
                uint32_t xt = tileIdx % xst;
                uint32_t yt = (tileIdx / xst) % yst;
                uint32_t zt = tileIdx / (xst * yst);
                for (uint32_t voxelIdx = 0; voxelIdx < tileNumVoxels; voxelIdx++) {
                    uint32_t vx = voxelIdx % tileSizeX;
                    uint32_t vy = (voxelIdx / tileSizeX) % tileSizeY;
                    uint32_t vz = voxelIdx / (tileSizeX * tileSizeY);
                    uint32_t x = vx + xt * tileSizeX;
                    uint32_t y = vy + yt * tileSizeY;
                    uint32_t z = vz + zt * tileSizeZ;
                    float value = 0.0f;
                    if (x < uint32_t(xsd) && y < uint32_t(ysd) && z < uint32_t(zsd)) {
                        value = linearBufferData[IDXSD(x, y, z)];
                    }
                    tiledBufferData[tileIdx * tileNumVoxels + voxelIdx] = value;
                }
            }
#ifdef USE_TBB
            });
#endif
            buffer->uploadData(sizeInBytes, tiledBufferData);
            delete[] tiledBufferData;
        } else {
            buffer->uploadData(sizeInBytes, downscaledField);
        }
        sizeInBytes = buffer->getDeviceMemoryAllocationSize();
        if (varNum == 0) {
            if (idx == 0) {
                fieldBuffers.at(fieldIdx) = buffer;
            } else {
                fieldBuffersR1.at(fieldIdx) = buffer;
            }
        } else {
            if (idx == 0) {
                fieldBuffers2.at(fieldIdx) = buffer;
            } else {
                fieldBuffersR12.at(fieldIdx) = buffer;
            }
        }
    }

    auto token = volumeData->pushAuxiliaryMemoryDevice(sizeInBytes);
    if (varNum == 0) {
        if (idx == 0) {
            deviceMemoryTokens.at(fieldIdx) = token;
        } else {
            deviceMemoryTokensR1.at(fieldIdx) = token;
        }
    } else {
        if (idx == 0) {
            deviceMemoryTokens2.at(fieldIdx) = token;
        } else {
            deviceMemoryTokensR12.at(fieldIdx) = token;
        }
    }

    auto endTime = std::chrono::system_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    elapsedTimeDownsampling += elapsedTime.count();

    delete[] downscaledField;
}



HEBChart::~HEBChart() {
    if (computeRenderer) {
        correlationComputePass = {};
        sgl::vk::Device* device = computeRenderer->getDevice();
        device->freeCommandBuffer(commandPool, commandBuffer);
        delete computeRenderer;
    }
}

void HEBChart::computeCorrelations(
    HEBChartFieldData *fieldData,
    const std::vector<float*> &downscaledFields0, const std::vector<float*> &downscaledFields1,
    std::vector<MIFieldEntry> &miFieldEntries) {
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
    if (useAbsoluteCorrelationMeasure || isMeasureMI(correlationMeasureType)) {
#ifdef USE_TBB
        tbb::parallel_sort(miFieldEntries.begin(), miFieldEntries.end());
//#elif __cpp_lib_parallel_algorithm >= 201603L
//std::sort(std::execution::par_unseq, miFieldEntries.begin(), miFieldEntries.end());
#else
        std::sort(miFieldEntries.begin(), miFieldEntries.end());
#endif
    } else {
#ifdef USE_TBB
        tbb::parallel_sort(miFieldEntries.begin(), miFieldEntries.end(), [](const MIFieldEntry &x, const MIFieldEntry &y)
                           { return std::abs(x.correlationValue) > std::abs(y.correlationValue); });
//#elif __cpp_lib_parallel_algorithm >= 201603L
//std::sort(std::execution::par_unseq, miFieldEntries.begin(), miFieldEntries.end());
#else
        std::sort(miFieldEntries.begin(), miFieldEntries.end(), [](const MIFieldEntry &x, const MIFieldEntry &y)
                  { return std::abs(x.correlationValue) > std::abs(y.correlationValue); });
#endif
    }
    //auto endTimeSort = std::chrono::system_clock::now();
    //auto elapsedTimeSort = std::chrono::duration_cast<std::chrono::milliseconds>(endTimeSort - startTimeSort);
    //std::cout << "Elapsed time sort: " << elapsedTimeSort.count() << "ms" << std::endl;
}

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

void HEBChart::computeCorrelationsMean(
    HEBChartFieldData *fieldData,
    const std::vector<float*> &downscaledFields0, const std::vector<float*> &downscaledFields1,
    std::vector<MIFieldEntry> &miFieldEntries) {
    int cs = getCorrelationMemberCount();
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;
    int numPairs = regionsEqual ? (numPoints0 * numPoints0 - numPoints0) / 2 : numPoints0 * numPoints1;
    if (isSubselection) {
        numPoints0 = numPairs = int(subselectionBlockPairs.size());
        numPoints1 = 1;
    }
#ifdef USE_TBB
    miFieldEntries = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, numPoints0), std::vector<MIFieldEntry>(),
        [&](tbb::blocked_range<int> const &r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
            const CorrelationMeasureType cmt = correlationMeasureType;
            CORRELATION_CACHE;
            float minFieldValRef = 0.0f, maxFieldValRef = 0.0f, minFieldVal = 0.0f, maxFieldVal = 0.0f;

            for (int i = r.begin(); i != r.end(); i++) {
#else
    miFieldEntries.reserve(numPairs);
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries, numPoints0, numPoints1, cs, k, numBins) \
    shared(downscaledFields0, downscaledFields1)
#endif
    {
        const CorrelationMeasureType cmt = correlationMeasureType;
        std::vector<MIFieldEntry> miFieldEntriesThread;
        CORRELATION_CACHE;
        float minFieldValRef = 0.0f, maxFieldValRef = 0.0f, minFieldVal = 0.0f, maxFieldVal = 0.0f;

#if _OPENMP >= 201107
        #pragma omp for schedule(dynamic)
#endif
        for (int i = 0; i < numPoints0; i++) {
#endif
                int upperBounds = regionsEqual ? i : numPoints1;
                int ir = i, jr;
                if (isSubselection) {
                    std::tie(ir, jr) = subselectionBlockPairs.at(i);
                    upperBounds = 1;
                }

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    X[c] = downscaledFields0.at(c)[ir];
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
                if (isMeasureBinnedMI(correlationMeasureType)) {
                    minFieldValRef = std::numeric_limits<float>::max();
                    maxFieldValRef = std::numeric_limits<float>::lowest();
                    for (int c = 0; c < cs; c++) {
                        minFieldValRef = std::min(minFieldValRef, X[c]);
                        maxFieldValRef = std::max(maxFieldValRef, X[c]);
                    }
                    for (int c = 0; c < cs; c++) {
                        X[c] = (downscaledFields0.at(c)[ir] - minFieldValRef) / (maxFieldValRef - minFieldValRef);
                    }
                }

                for (int j = 0; j < upperBounds; j++) {
                    if (!isSubselection) {
                        jr = j;
                    }
                    if (regionsEqual && (cellDistanceRange.x > 0 || cellDistanceRange.y < cellDistanceRangeTotal.y)) {
                        glm::vec3 pti(ir % uint32_t(xsd0), (ir / uint32_t(xsd0)) % uint32_t(ysd0), ir / uint32_t(xsd0 * ysd0));
                        glm::vec3 ptj(jr % uint32_t(xsd1), (jr / uint32_t(xsd1)) % uint32_t(ysd1), jr / uint32_t(xsd1 * ysd1));
                        float cellDist = glm::length(pti - ptj);
                        if (cellDist < float(cellDistanceRange.x) || cellDist > float(cellDistanceRange.y)) {
                            continue;
                        }
                    }
                    if (regionsEqual && showCorrelationForClickedPoint && clickedPointGridIdx != uint32_t(ir) && clickedPointGridIdx != uint32_t(jr)) {
                        continue;
                    }
                    for (int c = 0; c < cs; c++) {
                        Y[c] = downscaledFields1.at(c)[jr];
                        if (std::isnan(Y[c])) {
                            isNan = true;
                            break;
                        }
                    }
                    if (!isNan) {
                        if (isMeasureBinnedMI(correlationMeasureType)) {
                            minFieldVal = minFieldValRef;
                            maxFieldVal = maxFieldValRef;
                            for (int c = 0; c < cs; c++) {
                                minFieldVal = std::min(minFieldVal, Y[c]);
                                maxFieldVal = std::max(maxFieldVal, Y[c]);
                            }
                            for (int c = 0; c < cs; c++) {
                                X[c] = (downscaledFields0.at(c)[ir] - minFieldVal) / (maxFieldVal - minFieldVal);
                                Y[c] = (downscaledFields1.at(c)[jr] - minFieldVal) / (maxFieldVal - minFieldVal);
                            }
                        }

                        float correlationValue = 0.0f;
                        if (cmt == CorrelationMeasureType::PEARSON) {
                            correlationValue = computePearson2<float>(X.data(), Y.data(), cs);
                        } else if (cmt == CorrelationMeasureType::SPEARMAN) {
                            computeRanks(Y.data(), gridPointRanks.data(), ordinalRankArraySpearman, cs);
                            correlationValue = computePearson2<float>(referenceRanks.data(), gridPointRanks.data(), cs);
                        } else if (cmt == CorrelationMeasureType::KENDALL) {
                            correlationValue = computeKendall<int32_t>(
                                    X.data(), Y.data(), cs, jointArray, ordinalRankArray, y, sortArray, stack);
                        } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                            correlationValue = computeMutualInformationBinned<double>(
                                    X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                        } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                            correlationValue = computeMutualInformationKraskov<double>(
                                    X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                        } else if (cmt == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT) {
                            correlationValue = computeMutualInformationBinned<double>(
                                    X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                            correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
                        } else if (cmt == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT) {
                            correlationValue = computeMutualInformationKraskov<double>(
                                    X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                            correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
                        }
                        if (useAbsoluteCorrelationMeasure) {
                            correlationValue = std::abs(correlationValue);
                        }
                        if (correlationValue < correlationRange.x || correlationValue > correlationRange.y) {
                            continue;
                        }
                        miFieldEntriesThread.emplace_back(correlationValue, ir, jr);
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
    HEBChartFieldData *fieldData, std::vector<MIFieldEntry> &miFieldEntries) {
    int cs = getCorrelationMemberCount();

    const std::string& fieldName1 =
            fieldData->isSecondFieldMode
            ? fieldData->selectedScalarFieldName2 : fieldData->selectedScalarFieldName1;
    const std::string& fieldName2 =
            fieldData->isSecondFieldMode
            ? fieldData->selectedScalarFieldName1 : fieldData->selectedScalarFieldName2;

    float minFieldVal = std::numeric_limits<float>::max();
    float maxFieldVal = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(fieldName1, fieldIdx);
        const float *field = fieldEntry->data<float>();
        fieldEntries.push_back(fieldEntry);
        fields.push_back(field);
        if (isMeasureBinnedMI(correlationMeasureType)) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(fieldName1, fieldIdx);
            minFieldVal = std::min(minFieldVal, minVal);
            maxFieldVal = std::max(maxFieldVal, maxVal);
        }
    }

    float minFieldVal2 = std::numeric_limits<float>::max();
    float maxFieldVal2 = std::numeric_limits<float>::lowest();
    std::vector<VolumeData::HostCacheEntry> fieldEntries2;
    std::vector<const float*> fields2;
    if (fieldData->useTwoFields) {
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            VolumeData::HostCacheEntry fieldEntry2 = getFieldEntryCpu(fieldName2, fieldIdx);
            const float *field2 = fieldEntry2->data<float>();
            fieldEntries2.push_back(fieldEntry2);
            fields2.push_back(field2);
            if (isMeasureBinnedMI(correlationMeasureType)) {
                auto [minVal2, maxVal2] = getMinMaxScalarFieldValue(fieldName2, fieldIdx);
                minFieldVal2 = std::min(minFieldVal2, minVal2);
                maxFieldVal2 = std::max(maxFieldVal2, maxVal2);
            }
        }
    } else {
        minFieldVal2 = minFieldVal;
        maxFieldVal2 = maxFieldVal;
        fieldEntries2 = fieldEntries;
        fields2 = fields;
    }

    if (samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION) {
        correlationSamplingExecuteCpuBayesian(
                fieldData, miFieldEntries, fields, minFieldVal, maxFieldVal, fields2, minFieldVal2, maxFieldVal2);
    } else {
        correlationSamplingExecuteCpuDefault(
                fieldData, miFieldEntries, fields, minFieldVal, maxFieldVal, fields2, minFieldVal2, maxFieldVal2);
    }
}

void HEBChart::correlationSamplingExecuteCpuDefault(
        HEBChartFieldData* fieldData, std::vector<MIFieldEntry>& miFieldEntries,
        const std::vector<const float*>& fields, float minFieldVal, float maxFieldVal,
        const std::vector<const float*>& fields2, float minFieldVal2, float maxFieldVal2) {
    const int cs = getCorrelationMemberCount();
    const int numPoints0 = xsd0 * ysd0 * zsd0;
    const int numPoints1 = xsd1 * ysd1 * zsd1;
    int numPairsDownsampled;
    if (isSubselection) {
        numPairsDownsampled = int(subselectionBlockPairs.size());
    } else if (regionsEqual && fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 + numPoints0) / 2;
    } else if (regionsEqual && !fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 - numPoints0) / 2;
    } else {
        numPairsDownsampled = numPoints0 * numPoints1;
    }
    std::vector<float> samples(6 * numSamples);
    generateSamples(samples.data(), numSamples, samplingMethodType, isSubselection);

#ifdef USE_TBB
    miFieldEntries = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, numPairsDownsampled), std::vector<MIFieldEntry>(),
        [&](tbb::blocked_range<int> const &r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
            const CorrelationMeasureType cmt = correlationMeasureType;
            CORRELATION_CACHE;

            for (int m = r.begin(); m != r.end(); m++) {
#else
    miFieldEntries.reserve(numPairsDownsampled);
#if _OPENMP >= 201107
    #pragma omp parallel default(none) shared(miFieldEntries, numPoints0, numPoints1, cs, k, numBins, fieldData) \
    shared(numPairsDownsampled, minFieldVal, maxFieldVal, fields, minFieldVal2, maxFieldVal2, fields2, numSamples, samples)
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
                } else if (regionsEqual && fieldData->isSecondFieldMode) {
                    i = (-1 + sgl::uisqrt(1 + 8 * m)) / 2;
                    j = uint32_t(m) - i * (i + 1) / 2;
                } else if (regionsEqual && !fieldData->isSecondFieldMode) {
                    i = (1 + sgl::uisqrt(1 + 8 * m)) / 2;
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
                if (regionsEqual && showCorrelationForClickedPoint && clickedPointGridIdx != uint32_t(i) && clickedPointGridIdx != uint32_t(j)) {
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
                        Y[c] = fields2.at(c)[idxj];
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
                        computeRanks(X.data(), referenceRanks.data(), ordinalRankArrayRef, cs);
                        computeRanks(Y.data(), gridPointRanks.data(), ordinalRankArraySpearman, cs);
                        correlationValue = computePearson2<float>(referenceRanks.data(), gridPointRanks.data(), cs);
                    } else if (cmt == CorrelationMeasureType::KENDALL) {
                        correlationValue = computeKendall<int32_t>(
                            X.data(), Y.data(), cs, jointArray, ordinalRankArray, y, sortArray, stack);
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
                        for (int c = 0; c < cs; c++) {
                            X[c] = (X[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                            Y[c] = (Y[c] - minFieldVal2) / (maxFieldVal2 - minFieldVal2);
                        }
                        correlationValue = computeMutualInformationBinned<double>(
                            X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                    } else if (cmt == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
                        correlationValue = computeMutualInformationKraskov<double>(
                            X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                    } else if (cmt == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT) {
                        for (int c = 0; c < cs; c++) {
                            X[c] = (X[c] - minFieldVal) / (maxFieldVal - minFieldVal);
                            Y[c] = (Y[c] - minFieldVal2) / (maxFieldVal2 - minFieldVal2);
                        }
                        correlationValue = computeMutualInformationBinned<double>(
                                X.data(), Y.data(), numBins, cs, histogram0.data(), histogram1.data(), histogram2d.data());
                        correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
                    } else if (cmt == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT) {
                        correlationValue = computeMutualInformationKraskov<double>(
                                X.data(), Y.data(), k, cs, kraskovEstimatorCache);
                        correlationValue = std::sqrt(1.0f - std::exp(-2.0f * correlationValue));
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

void HEBChart::correlationSamplingExecuteCpuBayesian(
        HEBChartFieldData *fieldData, std::vector<MIFieldEntry> &miFieldEntries,
        const std::vector<const float*> &fields, float minFieldVal, float maxFieldVal,
        const std::vector<const float*>& fields2, float minFieldVal2, float maxFieldVal2) {
    const int cs = getCorrelationMemberCount();
    const int numPoints0 = xsd0 * ysd0 * zsd0;
    const int numPoints1 = xsd1 * ysd1 * zsd1;
    int numPairsDownsampled;
    if (isSubselection) {
        numPairsDownsampled = int(subselectionBlockPairs.size());
    } else if (regionsEqual && fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 + numPoints0) / 2;
    } else if (regionsEqual && !fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 - numPoints0) / 2;
    } else {
        numPairsDownsampled = numPoints0 * numPoints1;
    }

    miFieldEntries.reserve(numPairsDownsampled);

    std::atomic<int> cur_pair{};
    const auto correlationType = correlationMeasureType;
    const int bayOptIterationCount = std::max(0, numSamples - numInitSamples);
    const int mutualInformationK = k;
    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    std::vector<std::vector<MIFieldEntry>> miFieldEntriesThread(threads.size());
    auto thread_func = [&](int id) {
        for (int p = cur_pair++; p < numPairsDownsampled; p = cur_pair++) {
            uint32_t i, j;
            if (isSubselection) {
                std::tie(i, j) = subselectionBlockPairs.at(p);
            } else if (regionsEqual && fieldData->isSecondFieldMode) {
                i = (-1 + sgl::uisqrt(1 + 8 * p)) / 2;
                j = uint32_t(p) - i * (i + 1) / 2;
            } else if (regionsEqual && !fieldData->isSecondFieldMode) {
                i = (1 + sgl::uisqrt(1 + 8 * p)) / 2;
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
            if (regionsEqual && showCorrelationForClickedPoint && clickedPointGridIdx != uint32_t(i) && clickedPointGridIdx != uint32_t(j)) {
                continue;
            }

            auto region0 = getGridRegionPointIdx(0, i);
            auto region1 = getGridRegionPointIdx(1, j);
            const std::array<int, 6> region_min{region0.xmin, region0.ymin, region0.zmin, region1.xmin, region1.ymin, region1.zmin};
            const std::array<int, 6> region_max{region0.xmax, region0.ymax, region0.zmax, region1.xmax, region1.ymax, region1.zmax};

            limbo::bayes_opt::BOptimizer<BayOpt::Params> optimizer;
            BayOpt::Params::stop_maxiterations::set_iterations(bayOptIterationCount);
            BayOpt::Params::init_randomsampling::set_samples(numInitSamples);
            BayOpt::Params::opt_nloptnograd::set_iterations(numBOIterations);
            switch (correlationType) {
            case CorrelationMeasureType::PEARSON:
                optimizer.optimize(BayOpt::Eval<BayOpt::PearsonFunctor>{fields, fields2, region_min, region_max, cs, xs, ys, BayOpt::PearsonFunctor()});
                break;
            case CorrelationMeasureType::SPEARMAN:
                optimizer.optimize(BayOpt::Eval<BayOpt::SpearmanFunctor>{fields, fields2, region_min, region_max, cs, xs, ys});
                break;
            case CorrelationMeasureType::KENDALL:
                optimizer.optimize(BayOpt::Eval<BayOpt::KendallFunctor>{fields, fields2, region_min, region_max, cs, xs, ys});
                break;
            case CorrelationMeasureType::MUTUAL_INFORMATION_BINNED:
                optimizer.optimize(BayOpt::Eval<BayOpt::MutualBinnedFunctor>{
                    fields, fields2, region_min, region_max, cs, xs, ys,
                    BayOpt::MutualBinnedFunctor{minFieldVal, maxFieldVal, minFieldVal2, maxFieldVal2, numBins}});
                break;
            case CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV:
                optimizer.optimize(BayOpt::Eval<BayOpt::MutualFunctor>{fields, fields2, region_min, region_max, cs, xs, ys, {mutualInformationK}});
                break;
            case CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT:
                optimizer.optimize(BayOpt::Eval<BayOpt::MutualBinnedCCFunctor>{
                        fields, fields2, region_min, region_max, cs, xs, ys,
                        BayOpt::MutualBinnedCCFunctor{minFieldVal, maxFieldVal, minFieldVal2, maxFieldVal2, numBins}});
                break;
            case CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT:
                optimizer.optimize(BayOpt::Eval<BayOpt::MutualCCFunctor>{fields, fields2, region_min, region_max, cs, xs, ys, {mutualInformationK}});
                break;
            default:
                assert(false && "Unimplemented Correlation measure type");
            }

            auto correlationValueMax = float(optimizer.best_observation()(0));
            if (correlationValueMax >= correlationRange.x && correlationValueMax <= correlationRange.y) {
                miFieldEntriesThread[id].emplace_back(correlationValueMax, i, j);
            }
        }
    };
    int id{};
    for (auto &t : threads) {
        t = std::thread(thread_func, id++);
    }
    for (auto &t : threads) {
        t.join();
    }
    // merging threads
    for (const auto &fieldEntry : miFieldEntriesThread) {
        miFieldEntries.insert(miFieldEntries.end(), fieldEntry.begin(), fieldEntry.end());
    }
}

void HEBChart::clearFieldDeviceData() {
    if (correlationComputePass) {
        correlationComputePass->setFieldImageViews({});
        correlationComputePass->setFieldBuffers({});
    }
}

std::shared_ptr<HEBChartFieldCache> HEBChart::getFieldCache(HEBChartFieldData* fieldData) {
    int cs = getCorrelationMemberCount();
    auto fieldCache = std::make_shared<HEBChartFieldCache>();
    fieldCache->minFieldVal = std::numeric_limits<float>::max();
    fieldCache->maxFieldVal = std::numeric_limits<float>::lowest();
    fieldCache->fieldEntries.reserve(cs);
    bool useImageArray = dataMode == CorrelationDataMode::IMAGE_3D_ARRAY;
    if (useImageArray) {
        fieldCache->fieldImageViews.reserve(cs);
    } else {
        fieldCache->fieldBuffers.reserve(cs);
    }
    fieldCache->useTwoFields = fieldData->useTwoFields;
    fieldCache->isSecondFieldMode = fieldData->isSecondFieldMode;
    if (useMeanFields || fieldData->useTwoFields) {
        if (useImageArray) {
            fieldCache->fieldImageViewsR1.reserve(cs);
        } else {
            fieldCache->fieldBuffersR1.reserve(cs);
        }
    }
    if (fieldData->useTwoFields && !useMeanFields) {
        fieldCache->fieldEntries2.reserve(cs);
    }

    if (useMeanFields) {
        fieldData->createFieldCache(
                volumeData.get(), regionsEqual, r0, r1, mdfx, mdfy, mdfz, isEnsembleMode, dataMode, useBufferTiling);
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            if (useImageArray) {
                auto& fieldImageView =
                        fieldCache->isSecondFieldMode ? fieldData->fieldImageViews2.at(fieldIdx) : fieldData->fieldImageViews.at(fieldIdx);
                fieldCache->fieldImageViews.push_back(fieldImageView);
                if (fieldImageView->getImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    fieldImageView->getImage()->transitionImageLayout(
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
                }

                if (fieldData->useTwoFields) {
                    if (regionsEqual) {
                        auto& fieldImageViewR1 =
                                fieldCache->isSecondFieldMode ? fieldData->fieldImageViews.at(fieldIdx) : fieldData->fieldImageViews2.at(fieldIdx);
                        fieldCache->fieldImageViewsR1.push_back(fieldImageViewR1);
                        if (fieldImageViewR1->getImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                            fieldImageViewR1->getImage()->transitionImageLayout(
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
                        }
                    } else {
                        auto& fieldImageViewR1 =
                                fieldCache->isSecondFieldMode ? fieldData->fieldImageViewsR1.at(fieldIdx) : fieldData->fieldImageViewsR12.at(fieldIdx);
                        fieldCache->fieldImageViewsR1.push_back(fieldImageViewR1);
                        if (fieldImageViewR1->getImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                            fieldImageViewR1->getImage()->transitionImageLayout(
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
                        }
                    }
                } else if (!regionsEqual) {
                    auto& fieldImageViewR1 = fieldData->fieldImageViewsR1.at(fieldIdx);
                    fieldCache->fieldImageViewsR1.push_back(fieldImageViewR1);
                    if (fieldImageViewR1->getImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                        fieldImageViewR1->getImage()->transitionImageLayout(
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
                    }
                }
            } else {
                fieldCache->fieldBuffers.push_back(
                        fieldCache->isSecondFieldMode ? fieldData->fieldBuffers2.at(fieldIdx) : fieldData->fieldBuffers.at(fieldIdx));

                if (fieldData->useTwoFields) {
                    if (regionsEqual) {
                        fieldCache->fieldBuffersR1.push_back(
                                fieldCache->isSecondFieldMode ? fieldData->fieldBuffers.at(fieldIdx) : fieldData->fieldBuffers2.at(fieldIdx));
                    } else {
                        fieldCache->fieldBuffersR1.push_back(
                                fieldCache->isSecondFieldMode ? fieldData->fieldBuffersR1.at(fieldIdx) : fieldData->fieldBuffersR12.at(fieldIdx));
                    }
                } else if (!regionsEqual) {
                    fieldCache->fieldBuffersR1.push_back(fieldData->fieldBuffersR1.at(fieldIdx));
                }
            }
        }
        if (isMeasureBinnedMI(correlationMeasureType)) {
            fieldCache->minFieldVal = fieldCache->isSecondFieldMode ? fieldData->minFieldVal2 : fieldData->minFieldVal;
            fieldCache->maxFieldVal = fieldCache->isSecondFieldMode ? fieldData->maxFieldVal2 : fieldData->maxFieldVal;

            if (fieldData->useTwoFields) {
                fieldCache->minFieldVal2 = fieldCache->isSecondFieldMode ? fieldData->minFieldVal : fieldData->minFieldVal2;
                fieldCache->maxFieldVal2 = fieldCache->isSecondFieldMode ? fieldData->maxFieldVal : fieldData->maxFieldVal2;
            }
        }
    } else {
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            const std::string& fieldName1 =
                    fieldCache->isSecondFieldMode
                    ? fieldData->selectedScalarFieldName2 : fieldData->selectedScalarFieldName1;
            const std::string& fieldName2 =
                    fieldCache->isSecondFieldMode
                    ? fieldData->selectedScalarFieldName1 : fieldData->selectedScalarFieldName2;
            VolumeData::DeviceCacheEntry fieldEntry = getFieldEntryDevice(fieldName1, fieldIdx, useImageArray);
            fieldCache->fieldEntries.push_back(fieldEntry);
            if (useImageArray) {
                fieldCache->fieldImageViews.push_back(fieldEntry->getVulkanImageView());
                if (fieldEntry->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    fieldEntry->getVulkanImage()->transitionImageLayout(
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
                }
            } else {
                fieldCache->fieldBuffers.push_back(fieldEntry->getVulkanBuffer());
            }
            if (isMeasureBinnedMI(correlationMeasureType)) {
                auto [minVal, maxVal] = getMinMaxScalarFieldValue(fieldName1, fieldIdx);
                fieldCache->minFieldVal = std::min(fieldCache->minFieldVal, minVal);
                fieldCache->maxFieldVal = std::max(fieldCache->maxFieldVal, maxVal);
            }

            if (fieldData->useTwoFields) {
                VolumeData::DeviceCacheEntry fieldEntry2 = getFieldEntryDevice(fieldName2, fieldIdx, useImageArray);
                fieldCache->fieldEntries2.push_back(fieldEntry2);
                if (useImageArray) {
                    fieldCache->fieldImageViewsR1.push_back(fieldEntry2->getVulkanImageView());
                    if (fieldEntry2->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                        fieldEntry2->getVulkanImage()->transitionImageLayout(
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, computeRenderer->getVkCommandBuffer());
                    }
                } else {
                    fieldCache->fieldBuffersR1.push_back(fieldEntry2->getVulkanBuffer());
                }
                if (isMeasureBinnedMI(correlationMeasureType)) {
                    auto [minVal2, maxVal2] = getMinMaxScalarFieldValue(fieldName2, fieldIdx);
                    fieldCache->minFieldVal2 = std::min(fieldCache->minFieldVal2, minVal2);
                    fieldCache->maxFieldVal2 = std::max(fieldCache->maxFieldVal2, maxVal2);
                }
            }
        }
    }
    return fieldCache;
}

void HEBChart::setSyntheticTestCase(const std::vector<std::pair<uint32_t, uint32_t>>& blockPairs) {
    isSyntheticTestCase = true;
    std::mt19937 generator(17);
    for (const auto& blockPair : blockPairs) {
        auto data = std::make_shared<MultivariateGaussian>(dfx, dfy, dfz);
        data->initRandom(generator);
        syntheticFieldsMap.insert(std::make_pair(std::make_pair(blockPair.first, blockPair.second), data));
    }
}

sgl::vk::BufferPtr HEBChart::computeCorrelationsForRequests(
        std::vector<CorrelationRequestData>& requests,
        std::shared_ptr<HEBChartFieldCache>& fieldCache, bool isFirstBatch) {
    if (isSyntheticTestCase) {
        auto* values = static_cast<float*>(correlationOutputStagingBuffer->mapMemory());
        int requestIdx = 0;
        for (const auto& request : requests) {
            auto it = syntheticFieldsMap.find(std::make_pair(request.i, request.j));
            if (it == syntheticFieldsMap.end()) {
                sgl::Logfile::get()->throwError(
                        "Error in HEBChart::computeAllCorrelationsBlockPair: Invalid block pair.");
            }
            values[requestIdx] = it->second->eval(
                    int(request.xi) % dfx, int(request.yi) % dfy, int(request.zi) % dfz,
                    int(request.xj) % dfx, int(request.yj) % dfy, int(request.zj) % dfz);
            requestIdx++;
        }
        correlationOutputStagingBuffer->unmapMemory();
        return correlationOutputStagingBuffer;
    }

    computeRenderer->setCustomCommandBuffer(commandBuffer, false);
    computeRenderer->beginCommandBuffer();
    if (isFirstBatch) {
        int cs = getCorrelationMemberCount();
        correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
        correlationComputePass->setVolumeData(volumeData.get(), cs, false);
        correlationComputePass->setCorrelationMemberCount(cs);
        correlationComputePass->setNumBins(numBins);
        correlationComputePass->setKraskovNumNeighbors(k);
        correlationComputePass->setDataMode(dataMode);
        correlationComputePass->setUseBufferTiling(useBufferTiling);
        if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
            correlationComputePass->setFieldImageViews(fieldCache->fieldImageViews);
        } else {
            correlationComputePass->setFieldBuffers(fieldCache->fieldBuffers);
        }
        if (useMeanFields || fieldCache->useTwoFields) {
            if (useMeanFields) {
                correlationComputePass->overrideGridSize(
                        sgl::iceil(r0.xsr, mdfx), sgl::iceil(r0.ysr, mdfy), sgl::iceil(r0.zsr, mdfz),
                        sgl::iceil(r1.xsr, mdfx), sgl::iceil(r1.ysr, mdfy), sgl::iceil(r1.zsr, mdfz));
            } else {
                correlationComputePass->overrideGridSize(
                        volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ(),
                        volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ());
            }
            correlationComputePass->setUseSecondaryFields(!regionsEqual || fieldCache->useTwoFields);
            if (!regionsEqual || fieldCache->useTwoFields) {
                if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
                    correlationComputePass->setFieldImageViewsSecondary(fieldCache->fieldImageViewsR1);
                } else {
                    correlationComputePass->setFieldBuffersSecondary(fieldCache->fieldBuffersR1);
                }
            }
        } else {
            correlationComputePass->setUseSecondaryFields(false);
        }
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
            correlationComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, uint32_t(requests.size()));
    if (isMeasureBinnedMI(correlationMeasureType)) {
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
    computeRenderer->resetCustomCommandBuffer();
    fence->wait();
    fence->reset();

    return correlationOutputStagingBuffer;
}

void HEBChart::createBatchCacheData(uint32_t& batchSizeSamplesMax) {
    int cs = getCorrelationMemberCount();
    const uint32_t batchSizeSamplesMaxAllCs = 1 << 17; // Up to 131072 samples per batch.
    batchSizeSamplesMax = batchSizeSamplesMaxAllCs;
    if (cs > 100) {
        double factorN = double(cs) / 100.0 * std::log2(double(cs) / 100.0 + 1.0);
        batchSizeSamplesMax = uint32_t(std::ceil(double(batchSizeSamplesMax) / factorN));
        batchSizeSamplesMax = uint32_t(sgl::nextPowerOfTwo(int(batchSizeSamplesMax)));
    }

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    sgl::DeviceThreadInfo deviceCoresInfo = sgl::getDeviceThreadInfo(device);
    batchSizeSamplesMax = std::max(batchSizeSamplesMax, deviceCoresInfo.numCudaCoresEquivalent);

    if (!computeRenderer) {
        computeRenderer = new sgl::vk::Renderer(device, 100);
        if (device->getGraphicsQueue() == device->getComputeQueue()) {
            supportsAsyncCompute = false;
        }
        fence = std::make_shared<sgl::vk::Fence>(device);

        sgl::vk::CommandPoolType commandPoolType{};
        commandPoolType.queueFamilyIndex = device->getComputeQueueIndex();
        commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandBuffer = device->allocateCommandBuffer(commandPoolType, &commandPool);

        correlationComputePass = std::make_shared<CorrelationComputePass>(computeRenderer);
        correlationComputePass->setUseRequestEvaluationMode(true);
    }

    if (!requestsBuffer || cachedBatchSizeSamplesMax != batchSizeSamplesMax) {
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

        correlationComputePass->setRequestsBuffer(requestsBuffer);
        correlationComputePass->setOutputBuffer(correlationOutputBuffer);
    }
}

void HEBChart::computeCorrelationsSamplingGpu(
    HEBChartFieldData *fieldData, std::vector<MIFieldEntry> &miFieldEntries) {
    if (samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION) {
        correlationSamplingExecuteGpuBayesian(fieldData, miFieldEntries);
    } else {
        correlationSamplingExecuteGpuDefault(fieldData, miFieldEntries);
    }
}

void HEBChart::correlationSamplingExecuteGpuDefault(
        HEBChartFieldData *fieldData, std::vector<MIFieldEntry> &miFieldEntries) {
    //const auto function_start = std::chrono::system_clock::now();
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;

    int numPairsDownsampled;
    if (isSubselection) {
        numPairsDownsampled = int(subselectionBlockPairs.size());
    } else if (regionsEqual && fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 + numPoints0) / 2;
    } else if (regionsEqual && !fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 - numPoints0) / 2;
    } else {
        numPairsDownsampled = numPoints0 * numPoints1;
    }

    std::vector<float> samplesGlobal(6 * numSamples);
    if (!isSubselection) {
        generateSamples(samplesGlobal.data(), numSamples, samplingMethodType, false);
    }

    uint32_t batchSizeSamplesMax{};
    createBatchCacheData(batchSizeSamplesMax);
    uint32_t batchSizeCellsMax = std::max(batchSizeSamplesMax / numSamples, uint32_t(1));
    uint32_t numBatches = sgl::uiceil(uint32_t(numPairsDownsampled), batchSizeCellsMax);

    auto fieldCache = getFieldCache(fieldData);

    uint32_t cellIdxOffset = 0;
    auto numCellsLeft = uint32_t(numPairsDownsampled);
    std::vector<CorrelationRequestData> requests;
    requests.reserve(batchSizeSamplesMax);
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
            [&](tbb::blocked_range<uint32_t> const &r, std::vector<CorrelationRequestData> requestsThread) -> std::vector<CorrelationRequestData>
            {
                std::vector<float> samplesLocal;
                if (isSubselection) {
                    samplesLocal.resize(6 * numSamples);
                    generateSamples(samplesLocal.data(), numSamples, samplingMethodType, true);
                }
                std::vector<float>& samples = isSubselection ? samplesLocal : samplesGlobal;
                for (int m = r.begin(); m != r.end(); m++)
                {
#else
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(requests, numPoints0, numPoints1) \
        shared(cellIdxOffset, loopMax, numSamples, samplesGlobal, fieldData)
#endif
        {
            std::vector<CorrelationRequestData> requestsThread;
            std::vector<float> samplesLocal;
            if (isSubselection) {
                samplesLocal.resize(6 * numSamples);
                generateSamples(samplesLocal.data(), numSamples, samplingMethodType, true);
            }
            std::vector<float>& samples = isSubselection ? samplesLocal : samplesGlobal;

#if _OPENMP >= 201107
            #pragma omp for schedule(dynamic)
#endif
            for (uint32_t m = cellIdxOffset; m < loopMax; m++) {
#endif
                    uint32_t i, j;
                    if (isSubselection) {
                        std::tie(i, j) = subselectionBlockPairs.at(m);
                    } else if (regionsEqual && fieldData->isSecondFieldMode) {
                        i = (-1 + sgl::uisqrt(1 + 8 * m)) / 2;
                        j = uint32_t(m) - i * (i + 1) / 2;
                    } else if (regionsEqual && !fieldData->isSecondFieldMode) {
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
                    if (regionsEqual && showCorrelationForClickedPoint && clickedPointGridIdx != i && clickedPointGridIdx != j) {
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
                        if (useMeanFields) {
                            xi = (xi - r0.xoff) / mdfx;
                            yi = (yi - r0.yoff) / mdfy;
                            zi = (zi - r0.zoff) / mdfz;
                            xj = (xj - r1.xoff) / mdfx;
                            yj = (yj - r1.yoff) / mdfy;
                            zj = (zj - r1.zoff) / mdfz;
                        }
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

        //auto start = std::chrono::system_clock::now();
        auto outputBuffer = computeCorrelationsForRequests(requests, fieldCache, batchIdx == 0);

        // Finally, insert the values of this batch.
        auto *correlationValues = static_cast<float *>(outputBuffer->mapMemory());
        //auto end = std::chrono::system_clock::now();
        //std::cout << "Needed " << std::chrono::duration<double>(end - start).count() << " s to evaluate " << requests.size() << " requests on the gpu" << std::endl;
        auto numValidCells = int(requests.size() / size_t(numSamples));
#ifdef USE_TBB
        std::vector<MIFieldEntry> newMiFieldEntries = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, numPairsDownsampled), std::vector<MIFieldEntry>(),
            [&](tbb::blocked_range<int> const &r, std::vector<MIFieldEntry> miFieldEntriesThread) -> std::vector<MIFieldEntry> {
                const CorrelationMeasureType cmt = correlationMeasureType;
                for (int cellIdx = r.begin(); cellIdx != r.end(); cellIdx++) {
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
    //const auto function_end = std::chrono::system_clock::now();
    //std::cout << "In total took " << std::chrono::duration<double>(function_end - function_start).count() << "s" << std::endl;
}

void HEBChart::correlationSamplingExecuteGpuBayesian(HEBChartFieldData *fieldData, std::vector<MIFieldEntry> &miFieldEntries) {
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

    using model_t = limbo::model::GP<BayOpt::Params>;

    // TODO: Make 10 variable.
    const int numSamplesBos = numSamples - numInitSamples;
    const int NUM_SAMPLES_PER_IT = std::clamp(numSamplesBos, 1, 10);

    const auto function_start = std::chrono::system_clock::now();
    int numPoints0 = xsd0 * ysd0 * zsd0;
    int numPoints1 = xsd1 * ysd1 * zsd1;
    const int bayOptIterationCount = sgl::iceil(std::max(0, numSamples - numInitSamples), NUM_SAMPLES_PER_IT);
    BayOpt::Params::stop_maxiterations::set_iterations(bayOptIterationCount);
    BayOpt::Params::init_randomsampling::set_samples(numInitSamples);
    BayOpt::Params::opt_nloptnograd::set_iterations(numBOIterations);
    const int MAX_SAMPLE_COUNT = std::max(numInitSamples, NUM_SAMPLES_PER_IT);

    int numPairsDownsampled;
    if (isSubselection) {
        numPairsDownsampled = int(subselectionBlockPairs.size());
    } else if (regionsEqual && fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 + numPoints0) / 2;
    } else if (regionsEqual && !fieldData->isSecondFieldMode) {
        numPairsDownsampled = (numPoints0 * numPoints0 - numPoints0) / 2;
    } else {
        numPairsDownsampled = numPoints0 * numPoints1;
    }

    uint32_t gpu_max_sample_count{};
    createBatchCacheData(gpu_max_sample_count);
    uint32_t max_pairs_count = std::max(gpu_max_sample_count / std::max(numInitSamples, NUM_SAMPLES_PER_IT), uint32_t(1));
    //uint32_t numBatches = sgl::uiceil(uint32_t(numPairsDownsampled), max_pairs_count);

    auto fieldCache = getFieldCache(fieldData);
    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    std::vector<CorrelationRequestData> correlation_requests[2]; // front and back buffer for correlation requests to allow staggered evaluation
    std::atomic<int> correlation_request_main{};                 // int to indicate on which requests the main thread works on
    auto correlation_request_worker = [&correlation_request_main]()
    { return 1 - correlation_request_main; }; // int to indicate on which request the worker works on
    auto correlation_request_swap = [&correlation_request_main]
    { correlation_request_main ^= 1; }; // swaps front and back buffer
    std::vector<Semaphore> main_signal_workers(threads.size()), workers_signal_main(threads.size());
    std::atomic<bool> iterate{};
    std::atomic<bool> done{};
    std::vector<model_t> optimizers(max_pairs_count);
    auto setup_end = std::chrono::system_clock::now();

    std::mutex cout_mut;

    int cur_pair_offset{};
    int cur_pair_count{};
    int pairs_per_thread{};
    std::vector<float> correlationValues[2]{};

    double sample_gen_time{}, sample_gpu_calc_time{}, thread_time{}, setup_time{std::chrono::duration<double>(setup_end - function_start).count()};
    std::mutex sample_gen_mutex;

    auto generate_requests = [&](const float *sample_positions, int start_pair_index, int end_pair_index, int samples_per_pair) {
        std::vector<CorrelationRequestData> requests(samples_per_pair * (end_pair_index - start_pair_index));
        for (int q : BayOpt::i_range(start_pair_index, end_pair_index)) {
            uint32_t i, j;
            if (isSubselection) {
                std::tie(i, j) = subselectionBlockPairs.at(q);
            } else if (regionsEqual && fieldData->isSecondFieldMode) {
                i = (-1 + sgl::uisqrt(1 + 8 * q)) / 2;
                j = uint32_t(q) - i * (i + 1) / 2;
            } else if (regionsEqual && !fieldData->isSecondFieldMode) {
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
            if (regionsEqual && showCorrelationForClickedPoint && clickedPointGridIdx != i && clickedPointGridIdx != j)
                continue;

            auto region0 = getGridRegionPointIdx(0, i);
            auto region1 = getGridRegionPointIdx(1, j);

            CorrelationRequestData request;
            for (int sampleIdx : BayOpt::i_range(samples_per_pair)) {
                sampleIdx += (q - start_pair_index) * samples_per_pair;
                int xi = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 0] * float(region0.xsr))), 0, region0.xsr - 1) + region0.xoff;
                int yi = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 1] * float(region0.ysr))), 0, region0.ysr - 1) + region0.yoff;
                int zi = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 2] * float(region0.zsr))), 0, region0.zsr - 1) + region0.zoff;
                int xj = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 3] * float(region1.xsr))), 0, region1.xsr - 1) + region1.xoff;
                int yj = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 4] * float(region1.ysr))), 0, region1.ysr - 1) + region1.yoff;
                int zj = std::clamp(int(BayOpt::pr(sample_positions[sampleIdx * 6 + 5] * float(region1.zsr))), 0, region1.zsr - 1) + region1.zoff;
                if (useMeanFields) {
                    xi = (xi - r0.xoff) / mdfx;
                    yi = (yi - r0.yoff) / mdfy;
                    zi = (zi - r0.zoff) / mdfz;
                    xj = (xj - r1.xoff) / mdfx;
                    yj = (yj - r1.yoff) / mdfy;
                    zj = (zj - r1.zoff) / mdfz;
                }
                request.i = i;
                request.j = j;
                request.xi = uint32_t(xi);
                request.yi = uint32_t(yi);
                request.zi = uint32_t(zi);
                request.xj = uint32_t(xj);
                request.yj = uint32_t(yj);
                request.zj = uint32_t(zj);
                requests[sampleIdx] = request;
            }
        }
        return requests;
    };
    auto thread_func = [&](int thread_id)
    {
        BayOpt::AlgorithmNoGradVariants<BayOpt::Params> acqui_optimizer = BayOpt::getOptimizerAsVariant<BayOpt::Params>(algorithm);
        while (true) {
            main_signal_workers[thread_id].acquire();
            // stage 1 creating batch fo initial samples
            if (done)
                break;
            int global_pair_base_index = cur_pair_offset + thread_id * pairs_per_thread;
            int cur_thread_pair_count = std::max(std::min(pairs_per_thread, cur_pair_count - thread_id * pairs_per_thread), 0);
            auto start = std::chrono::system_clock::now();
            std::vector<float> samples_pos[2];
            samples_pos[correlation_request_worker()].resize(6 * MAX_SAMPLE_COUNT * cur_thread_pair_count);
            samples_pos[correlation_request_main].resize(6 * MAX_SAMPLE_COUNT * cur_thread_pair_count);
            // creating initial sample positions
            if (cur_thread_pair_count){
                assert(samples_pos[correlation_request_worker()].size() == 6 * numInitSamples * cur_thread_pair_count);
                generateSamples(samples_pos[correlation_request_worker()].data(), numInitSamples * cur_thread_pair_count, SamplingMethodType::RANDOM_UNIFORM, false);
                auto corr_requ = generate_requests(samples_pos[correlation_request_worker()].data(), global_pair_base_index, global_pair_base_index + cur_thread_pair_count, numInitSamples);
                int request_offset = thread_id * pairs_per_thread * numInitSamples;
                assert(request_offset + corr_requ.size() <= correlation_requests[correlation_request_worker()].size());
                assert(corr_requ.size() == cur_thread_pair_count * numInitSamples);
                std::copy(corr_requ.begin(), corr_requ.end(), correlation_requests[correlation_request_worker()].begin() + request_offset);
            }
            auto end = std::chrono::system_clock::now();
            {
                std::scoped_lock lock(sample_gen_mutex);
                sample_gen_time = sample_gen_time + std::chrono::duration<double>(end - start).count();
            }
            workers_signal_main[thread_id].release();
            main_signal_workers[thread_id].acquire();
            
            start = std::chrono::system_clock::now();
            // stage 2 create single samples
            Eigen::VectorXd p(6);
            Eigen::VectorXd v(1);
            for (int i : BayOpt::i_range(bayOptIterationCount + 1)) {
                //if(thread_id == 0)
                //    std::cout << "Worker are now in iteration " << i << std::endl;
                // Drawing new samples from the model and creating the requests for the samples
                if(i < bayOptIterationCount){
                    if (i == 0) {
                        generateSamples(samples_pos[correlation_request_worker()].data(), cur_thread_pair_count * NUM_SAMPLES_PER_IT, SamplingMethodType::RANDOM_UNIFORM, true);
                    } else {
                        for (int o : BayOpt::i_range(cur_thread_pair_count)) {
                            limbo::acqui::UCB<BayOpt::Params, model_t> acqui_fun(optimizers[thread_id * pairs_per_thread + o], i);

                            auto acqui_optimization = [&](const Eigen::VectorXd &x, bool g)
                            { return acqui_fun(x, limbo::FirstElem{}, g); };

                            for (int s = 0; s < NUM_SAMPLES_PER_IT; s++) {
                                auto starting_point = limbo::tools::random_vector_bounded(6);
                                auto new_sample = std::visit([&acqui_optimization, &starting_point](auto &&optimizer)
                                                             { return optimizer(acqui_optimization, starting_point, true); },
                                                             acqui_optimizer); //acqui_optimizer(acqui_optimization, starting_point, true);

                                std::copy(new_sample.data(), new_sample.data() + 6, samples_pos[correlation_request_worker()].begin() + o * 6 * NUM_SAMPLES_PER_IT + s * 6);
                            }
                        }
                    }
                    if (cur_thread_pair_count) {
                        assert(std::all_of(samples_pos[correlation_request_worker()].begin(), samples_pos[correlation_request_worker()].end(), [](float v)
                                           { return v >= 0.f && v <= 1.f; }));
                        auto corr_requ = generate_requests(
                                samples_pos[correlation_request_worker()].data(), global_pair_base_index,
                                global_pair_base_index + cur_thread_pair_count, NUM_SAMPLES_PER_IT);
                        assert(correlation_requests[correlation_request_worker()].size() > thread_id * pairs_per_thread);
                        std::copy(corr_requ.begin(), corr_requ.end(), correlation_requests[correlation_request_worker()].begin() + thread_id * pairs_per_thread * NUM_SAMPLES_PER_IT);
                    }
                }
                end = std::chrono::system_clock::now();
                {
                    std::scoped_lock lock(sample_gen_mutex);
                    sample_gen_time = sample_gen_time + std::chrono::duration<double>(end - start).count();
                }

                // synching with main, sample generation done, waiting for evaluation
                workers_signal_main[thread_id].release();
                main_signal_workers[thread_id].acquire();
                start = std::chrono::system_clock::now();

                // update optimizers
                const int samples = i == 0 ? numInitSamples : NUM_SAMPLES_PER_IT;
                for (int i : BayOpt::i_range(cur_thread_pair_count)) {
                    for(int s: BayOpt::i_range(samples)){
                        int lin_index = thread_id * pairs_per_thread * samples + i * samples + s;
                        assert(lin_index < correlation_requests[correlation_request_worker()].size());
                        v(0) = std::abs(correlationValues[correlation_request_worker()][lin_index]);
                        if (std::isnan(v(0)) || std::isinf(v(0)))
                            v(0) = 0;
                        int sample_index = i * samples + s;
                        assert(sample_index * 6 < samples_pos[correlation_request_worker()].size());
                        std::copy(samples_pos[correlation_request_worker()].begin() + 6 * sample_index, samples_pos[correlation_request_worker()].begin() + 6 * sample_index + 6, p.data());
                        optimizers[thread_id * pairs_per_thread + i].add_sample(p, v);
                    }
                }
            }
            //if(thread_id == 0)
            //    std::cout << "Worker iteration done" << std::endl;

            // writing the best result for each thread
            for(int p: BayOpt::i_range(cur_thread_pair_count)){
                const auto& cur_opt = optimizers[thread_id * pairs_per_thread + p].observations_matrix();
                double m = *std::max_element(cur_opt.data(), cur_opt.data() + cur_opt.size());

                uint32_t i, j, q = global_pair_base_index + p;
                if (isSubselection) {
                    std::tie(i, j) = subselectionBlockPairs.at(q);
                } else if (regionsEqual && fieldData->isSecondFieldMode) {
                    i = (-1 + sgl::uisqrt(1 + 8 * q)) / 2;
                    j = uint32_t(q) - i * (i + 1) / 2;
                } else if (regionsEqual && !fieldData->isSecondFieldMode) {
                    i = (1 + sgl::uisqrt(1 + 8 * q)) / 2;
                    j = uint32_t(q) - i * (i - 1) / 2;
                } else {
                    i = q / uint32_t(numPoints1);
                    j = q % uint32_t(numPoints1);
                }
                assert(global_pair_base_index + p < miFieldEntries.size());
                miFieldEntries[global_pair_base_index + p] = {float(m), i, j};
            }
            end = std::chrono::system_clock::now();
            {
                std::scoped_lock lock(sample_gen_mutex);
                sample_gen_time = sample_gen_time + std::chrono::duration<double>(end - start).count();
            }
            workers_signal_main[thread_id].release();
        }
    };
    auto release_all_workers = [&]()
    {
        for (auto &s : main_signal_workers)
            s.release();
    };
    auto acquire_all_workers = [&]()
    {
        for (auto &s : workers_signal_main)
            s.acquire();
    };

    auto thread_start = std::chrono::system_clock::now();
    int thread_id{};
    for (auto &t : threads)
        t = std::thread(thread_func, thread_id++);
    auto thread_end = std::chrono::system_clock::now();
    thread_time = std::chrono::duration<double>(thread_end - thread_start).count();

    bool isFirstBatch = true;
    miFieldEntries.resize(numPairsDownsampled);
    for (int base_pair_index : BayOpt::i_range<int>(0, numPairsDownsampled, max_pairs_count))
    {
        iterate = false;
        cur_pair_offset = base_pair_index;
        cur_pair_count = std::min<int>(numPairsDownsampled - base_pair_index, max_pairs_count);
        pairs_per_thread = (cur_pair_count + int(threads.size()) - 1) / int(threads.size());
        optimizers = std::vector<model_t>(cur_pair_count);
        //std::cout << "Base pair: " << base_pair_index << " with max pair index: " << numPairsDownsampled << " , and " << bayOptIterationCount << " refinement iterations" << std::endl;
        // create, evaluate and add the initial samples to models -----------------------------------------------------------------------
        correlation_requests[correlation_request_main].resize(cur_pair_count * NUM_SAMPLES_PER_IT);
        correlation_requests[correlation_request_worker()].resize(cur_pair_count * numInitSamples);
        // filling the back buffer of the correlation requests
        // this is stage 1, only workers run
        release_all_workers();
        acquire_all_workers();
        correlation_request_swap(); // swap front and back buffer
        // let the threads fill the back buffer
        // this is stage 2, main executes sample evaluation,
        //                  worker refill the back buffer with a single sample per thread
        release_all_workers();

        auto start = std::chrono::system_clock::now();
        auto end = start;
        // iteratively generating new good samples. Generation of sample positions is done multi threaded -------------------
        iterate = bayOptIterationCount;
        for (int i : BayOpt::i_range(bayOptIterationCount + 1)) {
            //std::cout << "Main in iteration " << i << std::endl;
            const int samples = i == 0 ? numInitSamples : NUM_SAMPLES_PER_IT;
            // stage 3, main executes sample evaluation
            //          worker add samples from the correlationValues to their model and creates a new sample after addition

            // evaluating the requests and updating the models
            auto outputBuffer = computeCorrelationsForRequests(
                    correlation_requests[correlation_request_main], fieldCache, isFirstBatch);
            isFirstBatch = false;
            correlation_requests[correlation_request_main].resize(cur_pair_count * NUM_SAMPLES_PER_IT);
            auto vals = static_cast<float *>(outputBuffer->mapMemory());
            correlationValues[correlation_request_main] = std::vector(vals, vals + cur_pair_count * samples);
            outputBuffer->unmapMemory();
            end = std::chrono::system_clock::now();
            sample_gpu_calc_time += std::chrono::duration<double>(end - start).count();

            acquire_all_workers();
            correlation_request_swap(); // swap back and front buffer

            release_all_workers();
            start = std::chrono::system_clock::now();
        }
        //std::cout << "Main iteration done" << std::endl;
        // adding the last samples and writing the result back to the field entries
        acquire_all_workers();

        start = std::chrono::system_clock::now();
    }

    // signaling the threads job is done
    thread_start = std::chrono::system_clock::now();
    done = true;
    release_all_workers();

    for (auto &t : threads)
        t.join();

    // Use filtering?
    if (correlationRange.x > correlationRangeTotal.x || correlationRange.y < correlationRangeTotal.y) {
        std::vector<MIFieldEntry> miFieldEntriesUnfiltered;
        std::swap(miFieldEntries, miFieldEntriesUnfiltered);
        miFieldEntries.reserve(miFieldEntriesUnfiltered.size());
        for (const auto& entry : miFieldEntriesUnfiltered) {
            if (entry.correlationValue >= correlationRange.x && entry.correlationValue <= correlationRange.y) {
                miFieldEntries.push_back(entry);
            }
        }
    }

    thread_end = std::chrono::system_clock::now();
    thread_time += std::chrono::duration<double>(thread_end - thread_start).count();

    auto function_end = std::chrono::system_clock::now();
    if (!isHeadlessMode)
        std::cout << "BayOpt details: sample_gen_time[" << sample_gen_time / threads.size() << "s] sample_eval_time[" << sample_gpu_calc_time << "s] threading_overhead[" << thread_time << "s] setup_time[" << setup_time << "s]  Total time: " << std::chrono::duration<double>(function_end - function_start).count() << "s" << std::endl;
}

void HEBChart::createFieldCacheForTests() {
    auto fieldCache = getFieldCache(fieldDataArray.front().get());
}

HEBChart::PerfStatistics HEBChart::computeCorrelationsBlockPairs(
    const std::vector<std::pair<uint32_t, uint32_t>> &blockPairs,
    const std::vector<float*> &downscaledFields0, const std::vector<float*> &downscaledFields1) {
    HEBChart::PerfStatistics statistics{};
    isSubselection = true;
    subselectionBlockPairs = blockPairs;

    auto startTime = std::chrono::system_clock::now();
    std::vector<MIFieldEntry> miFieldEntries;
    computeCorrelations(fieldDataArray.front().get(), downscaledFields0, downscaledFields1, miFieldEntries);
    auto endTime = std::chrono::system_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    statistics.elapsedTimeMicroseconds = double(elapsedTime.count());

    std::sort(miFieldEntries.begin(), miFieldEntries.end(), [](const MIFieldEntry &x, const MIFieldEntry &y) {
                  if (x.pointIndex0 != y.pointIndex0) {
                      return x.pointIndex0 < y.pointIndex0;
                  }
                  return x.pointIndex1 < y.pointIndex1;
              });
    int iPair = 0, iEntry = 0;
    while (true) {
        if (iEntry == int(miFieldEntries.size())) {
            while (iPair != int(blockPairs.size())) {
                statistics.maximumValues.push_back(0.0f);
                iPair++;
            }
            break;
        }
        MIFieldEntry entry = miFieldEntries.at(iEntry);
        std::pair<uint32_t, uint32_t> pair = blockPairs.at(iPair);
        while (entry.pointIndex0 != pair.first || entry.pointIndex1 != pair.second) {
            statistics.maximumValues.push_back(0.0f);
            iPair++;
            pair = blockPairs.at(iPair);
        }
        statistics.maximumValues.push_back(entry.correlationValue);
        iEntry++;
        iPair++;
    }
    isSubselection = false;
    subselectionBlockPairs.clear();

    return statistics;
}

void HEBChart::computeAllCorrelationsBlockPair(uint32_t i, uint32_t j, std::vector<float>& allValues) {
    if (isSyntheticTestCase) {
        auto it = syntheticFieldsMap.find(std::make_pair(int(i), int(j)));
        if (it == syntheticFieldsMap.end()) {
            sgl::Logfile::get()->throwError(
                    "Error in HEBChart::computeAllCorrelationsBlockPair: Invalid block pair.");
        }
        auto minMaxVal = it->second->getGlobalMinMax();
        allValues.push_back(minMaxVal.first);
        allValues.push_back(minMaxVal.second);
        return;
    }
    if (useMeanFields) {
        sgl::Logfile::get()->throwError(
                "Error in HEBChart::computeAllCorrelationsBlockPair: Mean field mode is currently not supported.");
    }

    uint32_t batchSizeSamplesMax;
    createBatchCacheData(batchSizeSamplesMax);

    auto region0 = getGridRegionPointIdx(0, i);
    auto region1 = getGridRegionPointIdx(1, j);
    uint32_t numPoints0 = uint32_t(region0.xsr) * uint32_t(region0.ysr) * uint32_t(region0.zsr);
    uint32_t numPoints1 = uint32_t(region1.xsr) * uint32_t(region1.ysr) * uint32_t(region1.zsr);
    uint32_t numSamplesTotal = numPoints0 * numPoints1;
    uint32_t numBatches = sgl::uiceil(numSamplesTotal, batchSizeSamplesMax);
    allValues.reserve(numBatches);

    HEBChartFieldData *fieldData = fieldDataArray.front().get();
    auto fieldCache = getFieldCache(fieldData);

    uint32_t sampleIdxOffset = 0;
    auto numSamplesLeft = uint32_t(numSamplesTotal);
    std::vector<CorrelationRequestData> requests;
    requests.reserve(batchSizeSamplesMax);
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchSizeSamples;
        if (numSamplesLeft > batchSizeSamplesMax) {
            batchSizeSamples = batchSizeSamplesMax;
            numSamplesLeft -= batchSizeSamplesMax;
        } else if (numSamplesLeft > 0) {
            batchSizeSamples = numSamplesLeft;
            numSamplesLeft = 0;
        } else {
            break;
        }

        // Create the batch data to upload to the device.
        uint32_t loopMax = sampleIdxOffset + batchSizeSamples;
        requests.clear();
#ifdef USE_TBB
        requests = tbb::parallel_reduce(
            tbb::blocked_range<uint32_t>(sampleIdxOffset, loopMax), std::vector<CorrelationRequestData>(),
            [&](tbb::blocked_range<uint32_t> const &r, std::vector<CorrelationRequestData> requestsThread) -> std::vector<CorrelationRequestData> {
                for (uint32_t m = r.begin(); m != r.end(); m++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel default(none) shared(requests, sampleIdxOffset, numPoints1, region0, region1, loopMax) \
        shared(i, j)
#endif
        {
            std::vector<CorrelationRequestData> requestsThread;

#if _OPENMP >= 201107
            #pragma omp for schedule(dynamic)
#endif
            for (uint32_t m = sampleIdxOffset; m < loopMax; m++) {
#endif
                    uint32_t is = m / uint32_t(numPoints1);
                    uint32_t js = m % uint32_t(numPoints1);

                    CorrelationRequestData request;
                    int xi = region0.xoff + int(is) % region0.xsr;
                    int yi = region0.yoff + (int(is) / region0.xsr) % region0.ysr;
                    int zi = region0.zoff + int(is) / (region0.xsr * region0.ysr);
                    int xj = region1.xoff + int(js) % region1.xsr;
                    int yj = region1.yoff + (int(js) / region1.xsr) % region1.ysr;
                    int zj = region1.zoff + int(js) / (region1.xsr * region1.ysr);
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
        sampleIdxOffset += batchSizeSamples;

        auto outputBuffer = computeCorrelationsForRequests(requests, fieldCache, batchIdx == 0);

        // Finally, insert the values of this batch.
        auto *correlationValues = static_cast<float *>(outputBuffer->mapMemory());
        auto numRequests = int(requests.size());
        for (int requestIdx = 0; requestIdx < numRequests; requestIdx++) {
            float correlationValue = correlationValues[requestIdx];
            if (!std::isnan(correlationValue)) {
                if (useAbsoluteCorrelationMeasure) {
                    correlationValue = std::abs(correlationValue);
                }
                allValues.push_back(correlationValue);
            }
        }

        outputBuffer->unmapMemory();
    }
}
