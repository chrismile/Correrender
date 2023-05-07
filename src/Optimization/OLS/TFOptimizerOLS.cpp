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

#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "EigenSolver.hpp"
#ifdef CUDA_ENABLED
#include "CudaSolver.hpp"
#endif

#include "Volume/VolumeData.hpp"
#include "TFOptimizerOLS.hpp"

struct TFOptimizerOLSCache {
    Eigen::MatrixXr A; ///< System matrix relating scalars and colors.
    Eigen::MatrixXr b; ///< Ground truth colors.
    Eigen::MatrixXr x; ///< Transfer function values.
    uint32_t cachedTfSize = 0;
    uint32_t cachedNumVoxels = 0;
    std::shared_ptr<HostCacheEntryType> fieldEntryGT, fieldEntryOpt;
    std::pair<float, float> minMaxGT, minMaxOpt;
    std::vector<glm::vec4> tfGT;
};

TFOptimizerOLS::TFOptimizerOLS(
        sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute)
        : TFOptimizer(renderer, parentRenderer, supportsAsyncCompute) {
    cache = new TFOptimizerOLSCache;
}

TFOptimizerOLS::~TFOptimizerOLS() {
    delete cache;
}

void TFOptimizerOLS::TFOptimizerOLS::onRequestQueued(VolumeData* volumeData) {
    auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
    std::string fieldNameGT = fieldNames.at(settings.fieldIdxGT);
    std::string fieldNameOpt = fieldNames.at(settings.fieldIdxOpt);
    cache->fieldEntryGT = volumeData->getFieldEntryCpu(FieldType::SCALAR, fieldNameGT);
    cache->fieldEntryOpt = volumeData->getFieldEntryCpu(FieldType::SCALAR, fieldNameOpt);
    cache->minMaxGT = tfWindow.getSelectedRangePair(settings.fieldIdxGT);
    cache->minMaxOpt = tfWindow.getSelectedRangePair(settings.fieldIdxGT);
    cache->tfGT = tfWindow.getTransferFunctionMap_sRGBDownscaled(settings.fieldIdxGT, int(settings.tfSize));
    uint32_t numVoxels =
            uint32_t(volumeData->getGridSizeX())
            * uint32_t(volumeData->getGridSizeY())
            * uint32_t(volumeData->getGridSizeZ());
    if (cache->cachedTfSize != settings.tfSize || cache->cachedNumVoxels != numVoxels) {
        cache->A = Eigen::MatrixXr(numVoxels * 4, settings.tfSize * 4);
    }
    if (cache->cachedNumVoxels != numVoxels) {
        cache->b = Eigen::VectorXr(numVoxels * 4);
    }
    if (cache->cachedTfSize != settings.tfSize) {
        cache->x = Eigen::VectorXr(settings.tfSize * 4);
    }
    cache->cachedTfSize = settings.tfSize;
    cache->cachedNumVoxels = numVoxels;
}

float TFOptimizerOLS::getProgress() {
    return 0.0f;
}

void TFOptimizerOLS::runOptimization(bool shallStop, bool& hasStopped) {
    // TODO: Move matrix setup to CUDA
    // Set up the system matrix A and the right hand side vector b.
    uint32_t tfNumEntries = cache->cachedTfSize * 4;
    uint32_t numMatrixRows = cache->cachedNumVoxels * 4;
    for (uint32_t i = 0; i < numMatrixRows; i++) {
        for (uint32_t j = 0; j < tfNumEntries; j++) {
            cache->A(i, j) = 0.0f;
        }
    }
    const float* fieldDataGT = cache->fieldEntryGT->data<float>();
    const float* fieldDataOpt = cache->fieldEntryOpt->data<float>();
    float minGT = cache->minMaxGT.first;
    float maxGT = cache->minMaxGT.second;
    float minOpt = cache->minMaxOpt.first;
    float maxOpt = cache->minMaxOpt.second;
    auto Nj = float(cache->cachedTfSize - 1);
    for (uint32_t voxelIdx = 0; voxelIdx < cache->cachedNumVoxels; voxelIdx++) {
        float scalarGT = fieldDataGT[voxelIdx];
        float scalarOpt = fieldDataOpt[voxelIdx];
        uint32_t i = voxelIdx * 4;

        if (std::isnan(scalarGT) || std::isnan(scalarOpt)) {
            for (int c = 0; c < 4; c++) {
                cache->b(i + c) = 0.0f;
            }
            continue;
        }

        float tGT = (scalarGT - minGT) / (maxGT - minGT);
        float tGT0 = std::clamp(std::floor(tGT * Nj), 0.0f, Nj);
        float tGT1 = std::clamp(std::ceil(tGT * Nj), 0.0f, Nj);
        float fGT = tGT * Nj - tGT0;
        int jGT0 = int(tGT0);
        int jGT1 = int(tGT1);
        glm::vec4 cGT0 = cache->tfGT.at(jGT0);
        glm::vec4 cGT1 = cache->tfGT.at(jGT1);
        glm::vec4 colorGT = glm::mix(cGT0, cGT1, fGT);
        for (int c = 0; c < 4; c++) {
            cache->b(i + c) = colorGT[c];
        }

        float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
        float tOpt0 = std::clamp(std::floor(tOpt * Nj), 0.0f, Nj);
        float tOpt1 = std::clamp(std::ceil(tOpt * Nj), 0.0f, Nj);
        float fOpt = tOpt * Nj - tOpt0;
        int jOpt0 = int(tOpt0);
        int jOpt1 = int(tOpt1);
        if (fOpt < 0.0f || fOpt > 1.0f) {
            std::cout << "ERROR" << std::endl;
        }
        for (int c = 0; c < 4; c++) {
            cache->A(i + c, jOpt0 * 4 + c) = 1.0f - fOpt;
        }
        if (jOpt0 != jOpt1) {
            for (int c = 0; c < 4; c++) {
                cache->A(i + c, jOpt1 * 4 + c) = fOpt;
            }
        }
    }
    cache->fieldEntryGT = {};
    cache->fieldEntryOpt = {};

    const Real lambdaL = settings.useRelaxation ? Real(settings.relaxationLambda) : Real(0.0);
#ifdef CUDA_ENABLED
    if (settings.useCuda) {
        solveSystemOfLinearEquationsCuda(
                settings.cudaSolverType, settings.useRelaxation, lambdaL, cache->A, cache->b, cache->x);
    } else {
#endif
    solveSystemOfLinearEquationsEigen(
            settings.eigenSolverType, settings.useRelaxation, lambdaL, cache->A, cache->b, cache->x);
#ifdef CUDA_ENABLED
    }
#endif

    // Debugging.
    std::cout << "A:" << std::endl;
    uint32_t numi = std::min(cache->cachedNumVoxels * 4u, 32u);
    uint32_t numj = std::min(cache->cachedTfSize * 4u, 20u);
    for (uint32_t i = 0; i < numi; i++) {
        for (uint32_t j = 0; j < numj; j++) {
            std::cout << cache->A(i, j);
            if (j != numj - 1) {
                std::cout << ", ";
            } else if (uint32_t(cache->cachedTfSize) * 4u > numj) {
                std::cout << ", ...";
            }
        }
        if (i != numi - 1) {
            std::cout << std::endl;
        } else if (uint32_t(cache->cachedNumVoxels) * 4u > numi) {
            std::cout << std::endl << "..." << std::endl;
        }
    }
    std::cout << std::endl << std::endl;

    std::cout << "b:" << std::endl;
    for (uint32_t i = 0; i < numi; i++) {
        std::cout << cache->b(i);
        if (i != numi - 1) {
            std::cout << std::endl;
        } else if (uint32_t(cache->cachedNumVoxels) * 4u > numi) {
            std::cout << std::endl << "..." << std::endl;
        }
    }
    std::cout << std::endl << std::endl;

    std::cout << "x:" << std::endl;
    for (uint32_t j = 0; j < numj; j++) {
        std::cout << cache->x(j);
        if (j != numj - 1) {
            std::cout << std::endl;
        } else if (uint32_t(cache->cachedTfSize) * 4u > numj) {
            std::cout << std::endl << "..." << std::endl;
        }
    }
    std::cout << std::endl << std::endl;

    // Clamp the transfer function values to [0, 1].
    for (uint32_t j = 0; j < tfNumEntries; j++) {
        cache->x(j) = std::clamp(cache->x(j), 0.0f, 1.0f);
    }

    tfArrayOpt.resize(settings.tfSize);
#ifdef USE_DOUBLE_PRECISION
    uint32_t numEntries = settings.tfSize * 4;
    for (uint32_t i = 0; i < numEntries; i += 4) {
        tfArrayOpt[i] = glm::vec4(cache->x(i), cache->x(i + 1), cache->x(i + 2), cache->x(i + 3));
    }
#else
    memcpy(tfArrayOpt.data(), cache->x.data(), sizeof(glm::vec4) * settings.tfSize);
#endif
}
