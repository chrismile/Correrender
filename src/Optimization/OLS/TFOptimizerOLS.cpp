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
}

void TFOptimizerOLS::TFOptimizerOLS::onRequestQueued(VolumeData* volumeData) {
    auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    // TODO
    std::string fieldNameGT = fieldNames.at(settings.fieldIdxGT);
    std::string fieldNameOpt = fieldNames.at(settings.fieldIdxOpt);
    cache->fieldEntryGT = volumeData->getFieldEntryCpu(FieldType::SCALAR, fieldNameGT);
    cache->fieldEntryOpt = volumeData->getFieldEntryCpu(FieldType::SCALAR, fieldNameOpt);
    cache->minMaxGT = volumeData->getMinMaxScalarFieldValue(fieldNameGT);
    cache->minMaxGT = volumeData->getMinMaxScalarFieldValue(fieldNameOpt);
    auto tfGTColor16 = volumeData->getMultiVarTransferFunctionWindow().getTransferFunctionMap_sRGB(settings.fieldIdxGT);
    cache->tfGT.clear();
    cache->tfGT.reserve(settings.tfSize);
    // TODO: Only settings.tfSize num entries
    for (auto& color16 : tfGTColor16) {
        glm::vec4 color;
        color.r = color16.getFloatR();
        color.g = color16.getFloatG();
        color.b = color16.getFloatB();
        color.a = color16.getFloatA();
        cache->tfGT.push_back(color);
    }
    cache->tfGT.shrink_to_fit();
    uint32_t cachedTfSize = 0;
    uint32_t numVoxels =
            uint32_t(volumeData->getGridSizeX())
            * uint32_t(volumeData->getGridSizeY())
            * uint32_t(volumeData->getGridSizeZ());
    if (cache->cachedTfSize != settings.tfSize || cache->cachedNumVoxels != numVoxels) {
        cache->A = Eigen::MatrixXr(numVoxels, settings.tfSize * 4);
    }
    if (cache->cachedNumVoxels != numVoxels) {
        cache->b = Eigen::VectorXr(numVoxels * 4);
    }
    if (cache->cachedTfSize != settings.tfSize) {
        cache->x = Eigen::VectorXr(settings.tfSize * 4);
        transferFunctionBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_CPU_ONLY, false);
    }
    cache->cachedTfSize = settings.tfSize;
    cache->cachedNumVoxels = numVoxels;
}

float TFOptimizerOLS::getProgress() {
    return 0.0f;
}

void TFOptimizerOLS::runOptimization(bool shallStop, bool& hasStopped) {
    // Set up the system matrix A and the right hand side vector b.
    uint32_t tfNumEntries = cache->cachedTfSize * 4;
    for (uint32_t i = 0; i < cache->cachedNumVoxels; i++) {
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
    for (uint32_t i = 0; i < cache->cachedNumVoxels; i++) {
        float scalarGT = fieldDataGT[i];
        cache->b(i) = fieldDataGT[i];

        float scalarOpt = fieldDataOpt[i];
        //cache->A(i, j0) = 0.0f;
    }
    cache->fieldEntryGT = {};
    cache->fieldEntryOpt = {};

    const Real lambdaL = settings.useRelaxation ? Real(settings.relaxationLambda) : Real(0.0);
#ifdef CUDA_ENABLED
    if (settings.useCuda) {
        solveSystemOfLinearEquationsCuda(
                settings.cudaSolverType, settings.useRelaxation, lambdaL, cache->A, cache->b, cache->x);
        return;
    }
#endif
    solveSystemOfLinearEquationsEigen(
            settings.eigenSolverType, settings.useRelaxation, lambdaL, cache->A, cache->b, cache->x);

    // TODO: Share buffers with CUDA.
    auto* tfEntries = reinterpret_cast<float*>(transferFunctionBuffer->mapMemory());
#ifdef USE_DOUBLE_PRECISION
    uint32_t numEntries = settings.tfSize * 4;
    for (uint32_t i = 0; i < numEntries; i++) {
        tfEntries[i] = float(cache->x(i));
    }
#else
    memcpy(tfEntries, cache->x.data(), sizeof(glm::vec4) * settings.tfSize);
#endif
    transferFunctionBuffer->unmapMemory();
}
