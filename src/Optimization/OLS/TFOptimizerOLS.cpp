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

#include <chrono>

#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "EigenSolver.hpp"
#ifdef CUDA_ENABLED
#include "CudaSolver.hpp"
#endif

#include "Volume/VolumeData.hpp"
#include "../CopyFieldImages.hpp"
#include "NormalEquations.hpp"
#include "TFOptimizerOLS.hpp"

template class TFOptimizerOLSTyped<float>;
template class TFOptimizerOLSTyped<double>;

template<class Real>
struct TFOptimizerOLSCacheTyped {
    uint32_t cachedTfSize = 0;
    uint32_t cachedNumVoxels = 0;
    uint32_t cachedXs = 0, cachedYs = 0, cachedZs = 0;
    std::shared_ptr<HostCacheEntryType> fieldEntryGT, fieldEntryOpt;
    std::pair<float, float> minMaxGT, minMaxOpt;
    std::vector<glm::vec4> tfGT;

    bool cachedUseSparseSolve = false;

    // Dense matrix.
    Eigen::MatrixXr A; ///< System matrix relating scalars and colors.
    Eigen::MatrixXr b; ///< Ground truth colors.
    Eigen::MatrixXr x; ///< Transfer function values.

    // Sparse matrix.
#ifdef SPARSE_ROW_MAJOR
    Eigen::SparseMatrixRowXr sparseA;
#else
    Eigen::SparseMatrixColXr sparseA;
#endif
    std::vector<Real> csrVals;
    std::vector<int> csrRowPtr;
    std::vector<int> csrColInd;
    std::vector<Real> bSparse;
#ifdef CUDA_ENABLED
    sgl::vk::TextureCudaExternalMemoryVkPtr cudaInputImageGT, cudaInputImageOpt;
    Real* cudaCsrVals = nullptr;
    int* cudaCsrRowPtr = nullptr;
    int* cudaCsrColInd = nullptr;
    Real* cudaBSparse = nullptr;
    int cudaNumNnz = 0;
    int cudaNumRows = 0;
#endif

    // Implicit matrix.
    sgl::vk::ImageViewPtr inputImageGT, inputImageOpt;
    sgl::vk::BufferPtr lhsBuffer, rhsBuffer;
    sgl::vk::BufferPtr lhsStagingBuffer, rhsStagingBuffer;
    sgl::vk::BufferPtr tfGTBuffer;
};

TFOptimizerOLS::TFOptimizerOLS(
        sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute)
        : TFOptimizer(renderer, parentRenderer, supportsAsyncCompute) {
    fopt = new TFOptimizerOLSTyped<float>(
            renderer, parentRenderer, supportsAsyncCompute, fence, commandPool, commandBuffer, settings, tfArrayOpt);
    dopt = new TFOptimizerOLSTyped<double>(
            renderer, parentRenderer, supportsAsyncCompute, fence, commandPool, commandBuffer, settings, tfArrayOpt);
}

TFOptimizerOLS::~TFOptimizerOLS() {
    delete fopt;
    delete dopt;
}

float TFOptimizerOLS::getProgress() {
    return 0.0f;
}

void TFOptimizerOLS::onRequestQueued(VolumeData* volumeData) {
    if (settings.floatAccuracy == FloatAccuracy::FLOAT) {
        dopt->clearCache();
        fopt->onRequestQueued(volumeData);
    } else {
        fopt->clearCache();
        dopt->onRequestQueued(volumeData);
    }
}

void TFOptimizerOLS::runOptimization(bool& shallStop, bool& hasStopped) {
    if (settings.floatAccuracy == FloatAccuracy::FLOAT) {
        return fopt->runOptimization(shallStop, hasStopped);
    } else {
        return dopt->runOptimization(shallStop, hasStopped);
    }
}

template<class Real>
TFOptimizerOLSTyped<Real>::TFOptimizerOLSTyped(
        sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute,
        sgl::vk::FencePtr fence, VkCommandPool commandPool, VkCommandBuffer commandBuffer,
        TFOptimizationWorkerSettings& settings, std::vector<glm::vec4>& tfArrayOpt)
        : renderer(renderer), parentRenderer(parentRenderer), supportsAsyncCompute(supportsAsyncCompute),
          fence(std::move(fence)), commandPool(commandPool), commandBuffer(commandBuffer),
          settings(settings), tfArrayOpt(tfArrayOpt) {
    cache = new TFOptimizerOLSCacheTyped<Real>;
    normalEquationsComputePass = std::make_shared<NormalEquationsComputePass>(renderer);
    normalEquationsCopySymmetricPass = std::make_shared<NormalEquationsCopySymmetricPass>(renderer);
    bool useDoublePrecision = std::is_same<Real, double>();
    normalEquationsComputePass->setUseDoublePrecision(useDoublePrecision);
    normalEquationsCopySymmetricPass->setUseDoublePrecision(useDoublePrecision);
}

template<class Real>
TFOptimizerOLSTyped<Real>::~TFOptimizerOLSTyped() {
    normalEquationsComputePass = {};
    normalEquationsCopySymmetricPass = {};
    delete cache;
}

template<class Real>
void TFOptimizerOLSTyped<Real>::clearCache() {
    cache->cachedTfSize = 0;
    cache->cachedNumVoxels = 0;
    cache->cachedXs = 0;
    cache->cachedYs = 0;
    cache->cachedZs = 0;
    cache->fieldEntryGT = {};
    cache->fieldEntryOpt = {};
    cache->tfGT = {};
    cache->cachedUseSparseSolve = 0;

    cache->lhsBuffer = {};
    cache->rhsBuffer = {};
    cache->lhsStagingBuffer = {};
    cache->rhsStagingBuffer = {};
    cache->tfGTBuffer = {};

    cache->A = {};
    cache->b = {};
    cache->x = {};
#ifdef CUDA_ENABLED
    cache->cudaInputImageGT = {};
    cache->cudaInputImageOpt = {};
#endif
    cache->inputImageGT = {};
    cache->inputImageOpt = {};
    cache->csrVals = {};
    cache->csrRowPtr = {};
    cache->csrColInd = {};
    cache->sparseA = {};
    cache->bSparse = {};
}

template<class Real>
void TFOptimizerOLSTyped<Real>::onRequestQueued(VolumeData* volumeData) {
    auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
    std::string fieldNameGT = fieldNames.at(settings.fieldIdxGT);
    std::string fieldNameOpt = fieldNames.at(settings.fieldIdxOpt);
    cache->fieldEntryGT = volumeData->getFieldEntryCpu(FieldType::SCALAR, fieldNameGT);
    cache->fieldEntryOpt = volumeData->getFieldEntryCpu(FieldType::SCALAR, fieldNameOpt);
    cache->minMaxGT = tfWindow.getSelectedRangePair(settings.fieldIdxGT);
    cache->minMaxOpt = tfWindow.getSelectedRangePair(settings.fieldIdxOpt);
    cache->tfGT = tfWindow.getTransferFunctionMap_sRGBDownscaled(settings.fieldIdxGT, int(settings.tfSize));
    uint32_t numVoxels =
            uint32_t(volumeData->getGridSizeX())
            * uint32_t(volumeData->getGridSizeY())
            * uint32_t(volumeData->getGridSizeZ());

    if (settings.backend == OLSBackend::VULKAN) {
        sgl::vk::Device* device = renderer->getDevice();
        if (!cache->lhsBuffer || cache->cachedTfSize != settings.tfSize) {
            cache->lhsBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, 16 * sizeof(Real) * settings.tfSize * settings.tfSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            cache->rhsBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, 4 * sizeof(Real) * settings.tfSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            cache->lhsStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, 16 * sizeof(Real) * settings.tfSize * settings.tfSize,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_TO_CPU);
            cache->rhsStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, 4 * sizeof(Real) * settings.tfSize,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_TO_CPU);
        }

        cache->tfGTBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        cache->tfGTBuffer->uploadData(sizeof(glm::vec4) * cache->tfGT.size(), cache->tfGT.data());

        auto fieldEntryGT = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNames.at(settings.fieldIdxGT));
        auto fieldEntryOpt = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNames.at(settings.fieldIdxOpt));
        CopyFieldImageDestinationData copyFieldImageDestinationData{};
        copyFieldImageDestinationData.inputImageGT = &cache->inputImageGT;
        copyFieldImageDestinationData.inputImageOpt = &cache->inputImageOpt;
        copyFieldImages(
                parentRenderer->getDevice(),
                uint32_t(volumeData->getGridSizeX()),
                uint32_t(volumeData->getGridSizeY()),
                uint32_t(volumeData->getGridSizeZ()),
                fieldEntryGT->getVulkanImage(), fieldEntryOpt->getVulkanImage(),
                copyFieldImageDestinationData,
                settings.fieldIdxGT, settings.fieldIdxOpt, false, false);
    } else {
        cache->lhsBuffer = {};
        cache->rhsBuffer = {};
        cache->lhsStagingBuffer = {};
        cache->rhsStagingBuffer = {};
        cache->tfGTBuffer = {};
    }

    if (settings.backend == OLSBackend::CUDA && settings.useCudaMatrixSetup) {
        auto fieldEntryGT = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNames.at(settings.fieldIdxGT));
        auto fieldEntryOpt = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNames.at(settings.fieldIdxOpt));
        CopyFieldImageDestinationData copyFieldImageDestinationData{};
        copyFieldImageDestinationData.inputImageGT = &cache->inputImageGT;
        copyFieldImageDestinationData.inputImageOpt = &cache->inputImageOpt;
#ifdef CUDA_ENABLED
        copyFieldImageDestinationData.cudaInputImageGT = &cache->cudaInputImageGT;
        copyFieldImageDestinationData.cudaInputImageOpt = &cache->cudaInputImageOpt;
#endif
        copyFieldImages(
                parentRenderer->getDevice(),
                uint32_t(volumeData->getGridSizeX()),
                uint32_t(volumeData->getGridSizeY()),
                uint32_t(volumeData->getGridSizeZ()),
                fieldEntryGT->getVulkanImage(), fieldEntryOpt->getVulkanImage(),
                copyFieldImageDestinationData,
                settings.fieldIdxGT, settings.fieldIdxOpt, true, true);

        //cache->cudaInputImageGT = fieldEntryGT->getTextureCudaExternalMemory();
        //cache->cudaInputImageOpt = fieldEntryOpt->getTextureCudaExternalMemory();
        //auto* device = renderer->getDevice();
        //auto commandBufferCompute = device->beginSingleTimeCommands(device->getGraphicsQueueIndex());
        //if (cache->cudaInputImageGT->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        //    cache->cudaInputImageGT->getVulkanImage()->transitionImageLayout(
        //            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, commandBufferCompute);
        //}
        //if (cache->cudaInputImageOpt->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        //    cache->cudaInputImageOpt->getVulkanImage()->transitionImageLayout(
        //            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, commandBufferCompute);
        //}
        //device->endSingleTimeCommands(commandBufferCompute, device->getGraphicsQueueIndex());
    }
#ifdef CUDA_ENABLED
    else {
        cache->cudaInputImageGT = {};
        cache->cudaInputImageOpt = {};
    }
#endif

    if (settings.backend == OLSBackend::CPU) {
        cache->inputImageGT = {};
        cache->inputImageOpt = {};
    }

    if (!settings.useSparseSolve || settings.backend != OLSBackend::CUDA) {
        cache->csrVals = {};
        cache->csrRowPtr = {};
        cache->csrColInd = {};
    }

    if (!settings.useSparseSolve || settings.backend != OLSBackend::CPU) {
        cache->sparseA = {};
    }

    if (settings.useSparseSolve || settings.backend == OLSBackend::VULKAN) {
        cache->A = {};
    }
    if (!settings.useSparseSolve || settings.backend != OLSBackend::CUDA) {
        cache->bSparse = {};
    }

    if (settings.useSparseSolve && settings.backend == OLSBackend::CPU) {
        cache->b = Eigen::VectorXr(numVoxels * 4);
    }
    if (!settings.useSparseSolve && settings.backend != OLSBackend::VULKAN) {
        if (cache->cachedTfSize != settings.tfSize || cache->cachedNumVoxels != numVoxels || cache->cachedUseSparseSolve) {
            cache->A = Eigen::MatrixXr(numVoxels * 4, settings.tfSize * 4);
        }
        if (cache->cachedNumVoxels != numVoxels || cache->cachedUseSparseSolve) {
            cache->b = Eigen::VectorXr(numVoxels * 4);
        }
    }

    if (cache->cachedTfSize != settings.tfSize) {
        cache->x = Eigen::VectorXr(settings.tfSize * 4);
    }

    cache->cachedTfSize = settings.tfSize;
    cache->cachedNumVoxels = numVoxels;
    cache->cachedXs = uint32_t(volumeData->getGridSizeX());
    cache->cachedYs = uint32_t(volumeData->getGridSizeY());
    cache->cachedZs = uint32_t(volumeData->getGridSizeZ());
    cache->cachedUseSparseSolve = settings.useSparseSolve;
}

template<class Real>
void TFOptimizerOLSTyped<Real>::buildSystemDense() {
    const float* fieldDataGT = cache->fieldEntryGT->template data<float>();
    const float* fieldDataOpt = cache->fieldEntryOpt->template data<float>();
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
        for (int c = 0; c < 4; c++) {
            cache->A(i + c, jOpt0 * 4 + c) = 1.0f - fOpt;
        }
        if (jOpt0 != jOpt1) {
            for (int c = 0; c < 4; c++) {
                cache->A(i + c, jOpt1 * 4 + c) = fOpt;
            }
        }
    }
}

template<class Real>
void TFOptimizerOLSTyped<Real>::buildSystemSparse() {
    float minGT = cache->minMaxGT.first;
    float maxGT = cache->minMaxGT.second;
    float minOpt = cache->minMaxOpt.first;
    float maxOpt = cache->minMaxOpt.second;

    if (settings.backend == OLSBackend::CUDA && settings.useCudaMatrixSetup) {
#ifdef CUDA_ENABLED
        createSystemMatrixCudaSparse(
                int(cache->cachedXs), int(cache->cachedYs), int(cache->cachedZs), int(cache->cachedTfSize),
                minGT, maxGT, minOpt, maxOpt,
                cache->cudaInputImageGT->getCudaTextureObject(), cache->cudaInputImageOpt->getCudaTextureObject(),
                reinterpret_cast<float*>(cache->tfGT.data()), cache->cudaNumRows, cache->cudaNumNnz,
                cache->cudaCsrVals, cache->cudaCsrRowPtr, cache->cudaCsrColInd, cache->cudaBSparse);
#endif
        return;
    }

    uint32_t numMatrixRows = cache->cachedNumVoxels * 4;
    uint32_t expectedNumNonZero = numMatrixRows * 2;
    auto Nj = float(cache->cachedTfSize - 1);
    const float* fieldDataGT = cache->fieldEntryGT->template data<float>();
    const float* fieldDataOpt = cache->fieldEntryOpt->template data<float>();
    if (settings.backend == OLSBackend::CUDA) {
        cache->csrVals.clear();
        cache->bSparse.clear();
        cache->csrColInd.clear();
        cache->csrRowPtr.clear();
        cache->csrVals.reserve(expectedNumNonZero);
        cache->bSparse.reserve(numMatrixRows);
        cache->csrColInd.reserve(expectedNumNonZero);
        cache->csrRowPtr.reserve(numMatrixRows + 1);
        cache->csrRowPtr.push_back(0);

        for (uint32_t voxelIdx = 0; voxelIdx < cache->cachedNumVoxels; voxelIdx++) {
            float scalarGT = fieldDataGT[voxelIdx];
            float scalarOpt = fieldDataOpt[voxelIdx];

            if (std::isnan(scalarGT) || std::isnan(scalarOpt)) {
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
                cache->bSparse.push_back(colorGT[c]);
            }

            float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
            float tOpt0 = std::clamp(std::floor(tOpt * Nj), 0.0f, Nj);
            float tOpt1 = std::clamp(std::ceil(tOpt * Nj), 0.0f, Nj);
            float fOpt = tOpt * Nj - tOpt0;
            int jOpt0 = int(tOpt0);
            int jOpt1 = int(tOpt1);
            for (int c = 0; c < 4; c++) {
                //cache->A(i + c, jOpt0 * 4 + c) = 1.0f - fOpt;
                cache->csrVals.push_back(1.0f - fOpt);
                cache->csrColInd.push_back(jOpt0 * 4 + c);
                if (jOpt0 != jOpt1) {
                    //cache->A(i + c, jOpt1 * 4 + c) = fOpt;
                    cache->csrVals.push_back(fOpt);
                    cache->csrColInd.push_back(jOpt1 * 4 + c);
                }
                cache->csrRowPtr.push_back(int(cache->csrVals.size()));
            }
        }
    } else {
#ifdef SPARSE_ROW_MAJOR
        uint32_t tfNumEntries = cache->cachedTfSize * 4;
        cache->sparseA = Eigen::SparseMatrixXr(numMatrixRows, tfNumEntries);
        cache->sparseA.reserve(Eigen::VectorXi::Constant(numMatrixRows, 2));

        int rowIdx = 0;
        for (uint32_t voxelIdx = 0; voxelIdx < cache->cachedNumVoxels; voxelIdx++) {
            float scalarGT = fieldDataGT[voxelIdx];
            float scalarOpt = fieldDataOpt[voxelIdx];

            if (std::isnan(scalarGT) || std::isnan(scalarOpt)) {
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
                cache->b(rowIdx + c) = colorGT[c];
            }

            float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
            float tOpt0 = std::clamp(std::floor(tOpt * Nj), 0.0f, Nj);
            float tOpt1 = std::clamp(std::ceil(tOpt * Nj), 0.0f, Nj);
            float fOpt = tOpt * Nj - tOpt0;
            int jOpt0 = int(tOpt0);
            int jOpt1 = int(tOpt1);
            for (int c = 0; c < 4; c++) {
                //cache->A(i + c, jOpt0 * 4 + c) = 1.0f - fOpt;
                cache->sparseA.insert(rowIdx, jOpt0 * 4 + c) = 1.0f - fOpt;
                if (jOpt0 != jOpt1) {
                    //cache->A(i + c, jOpt1 * 4 + c) = fOpt;
                    cache->sparseA.insert(rowIdx, jOpt1 * 4 + c) = fOpt;
                }
                rowIdx++;
            }
        }
        cache->sparseA.makeCompressed();
        cache->b.resize(numMatrixRows, 1);
#else
        uint32_t tfNumEntries = cache->cachedTfSize * 4;
        typedef Eigen::Triplet<Real> TripletReal;
        std::vector<TripletReal> tripletList;
        tripletList.reserve(numMatrixRows * 2);

        int rowIdx = 0;
        for (uint32_t voxelIdx = 0; voxelIdx < cache->cachedNumVoxels; voxelIdx++) {
            float scalarGT = fieldDataGT[voxelIdx];
            float scalarOpt = fieldDataOpt[voxelIdx];

            if (std::isnan(scalarGT) || std::isnan(scalarOpt)) {
                continue;
            }

            float tGT = (scalarGT - minGT) / (maxGT - minGT);
            float tGT0 = std::clamp(std::floor(tGT * Nj), 0.0f, Nj);
            float tGT1 = std::clamp(std::ceil(tGT * Nj), 0.0f, Nj);
            float fGT = tGT * Nj - tGT0;
            int jGT0 = int(tGT0);
            int jGT1 = int(tGT1);
            glm::vec4 cGT0 = tfGT.at(jGT0);
            glm::vec4 cGT1 = tfGT.at(jGT1);
            glm::vec4 colorGT = glm::mix(cGT0, cGT1, fGT);
            for (int c = 0; c < 4; c++) {
                cache->b(rowIdx + c) = colorGT[c];
            }

            float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
            float tOpt0 = std::clamp(std::floor(tOpt * Nj), 0.0f, Nj);
            float tOpt1 = std::clamp(std::ceil(tOpt * Nj), 0.0f, Nj);
            float fOpt = tOpt * Nj - tOpt0;
            int jOpt0 = int(tOpt0);
            int jOpt1 = int(tOpt1);
            for (int c = 0; c < 4; c++) {
                //cache->A(i + c, jOpt0 * 4 + c) = 1.0f - fOpt;
                tripletList.push_back(TripletReal(rowIdx, jOpt0 * 4 + c, 1.0f - fOpt));
                if (jOpt0 != jOpt1) {
                    //cache->A(i + c, jOpt1 * 4 + c) = fOpt;
                    tripletList.push_back(TripletReal(rowIdx, jOpt1 * 4 + c, fOpt));
                }
                rowIdx++;
            }
        }

        cache->sparseA = Eigen::SparseMatrixColXr(numMatrixRows, tfNumEntries);
        cache->sparseA.setFromTriplets(tripletList.begin(), tripletList.end());
        cache->b.resize(numMatrixRows, 1);
#endif
    }
}

template<class Real>
void TFOptimizerOLSTyped<Real>::runOptimization(bool& shallStop, bool& hasStopped) {
    uint32_t tfNumEntries = cache->cachedTfSize * 4;
    uint32_t numMatrixRows = cache->cachedNumVoxels * 4;

    if (settings.backend == OLSBackend::VULKAN) {
        float minGT = cache->minMaxGT.first;
        float maxGT = cache->minMaxGT.second;
        float minOpt = cache->minMaxOpt.first;
        float maxOpt = cache->minMaxOpt.second;
        normalEquationsComputePass->setBuffers(
                tfNumEntries, minGT, maxGT, minOpt, maxOpt, cache->lhsBuffer, cache->rhsBuffer, cache->tfGTBuffer);
        normalEquationsComputePass->setInputImages(
                cache->inputImageGT, cache->inputImageOpt);
        normalEquationsCopySymmetricPass->setBuffers(tfNumEntries, cache->lhsBuffer);

        renderer->setCustomCommandBuffer(commandBuffer, false);
        renderer->beginCommandBuffer();

        auto startSolve = std::chrono::system_clock::now();
        cache->lhsBuffer->fill(0, renderer->getVkCommandBuffer());
        cache->rhsBuffer->fill(0, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        normalEquationsComputePass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                cache->lhsBuffer);
        normalEquationsCopySymmetricPass->render();
        renderer->insertMemoryBarrier(
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        cache->lhsBuffer->copyDataTo(cache->lhsStagingBuffer, renderer->getVkCommandBuffer());
        cache->rhsBuffer->copyDataTo(cache->rhsStagingBuffer, renderer->getVkCommandBuffer());

        renderer->endCommandBuffer();
        renderer->submitToQueue({}, {}, fence, VK_PIPELINE_STAGE_TRANSFER_BIT);
        renderer->resetCustomCommandBuffer();
        fence->wait();
        fence->reset();

        /*auto* lhsData = reinterpret_cast<float*>(cache->lhsStagingBuffer->mapMemory());
        auto* rhsData = reinterpret_cast<float*>(cache->rhsStagingBuffer->mapMemory());
        Eigen::MatrixXr lhs = Eigen::MatrixXr(tfNumEntries, tfNumEntries);
        Eigen::MatrixXr rhs = Eigen::VectorXr(tfNumEntries);
        if constexpr (std::is_same<Real, double>()) {
            for (uint32_t i = 0; i < tfNumEntries; i++) {
                for (uint32_t j = 0; j < tfNumEntries; j++) {
                    lhs(i, j) = Real(lhsData[i + j * tfNumEntries]);
                }
            }
            for (uint32_t i = 0; i < tfNumEntries; i++) {
                rhs(i) = Real(rhsData[i]);
            }
        } else {
            memcpy(lhs.data(), lhsData, 4 * tfNumEntries * tfNumEntries);
            memcpy(rhs.data(), rhsData, 4 * tfNumEntries);
        }*/
        auto* lhsData = reinterpret_cast<Real*>(cache->lhsStagingBuffer->mapMemory());
        auto* rhsData = reinterpret_cast<Real*>(cache->rhsStagingBuffer->mapMemory());
        Eigen::MatrixXr lhs = Eigen::MatrixXr(tfNumEntries, tfNumEntries);
        Eigen::MatrixXr rhs = Eigen::VectorXr(tfNumEntries);
        memcpy(lhs.data(), lhsData, sizeof(Real) * tfNumEntries * tfNumEntries);
        memcpy(rhs.data(), rhsData, sizeof(Real) * tfNumEntries);
        cache->lhsStagingBuffer->unmapMemory();
        cache->rhsStagingBuffer->unmapMemory();

        uint32_t numi = std::min(cache->cachedTfSize * 4u, 32u);
        uint32_t numj = std::min(cache->cachedTfSize * 4u, 32u);
        std::cout << "lhs:" << std::endl;
        for (uint32_t i = 0; i < numi; i++) {
            for (uint32_t j = 0; j < numj; j++) {
                std::cout << lhs(i, j);
                if (j != numj - 1) {
                    std::cout << ", ";
                } else if (uint32_t(cache->cachedTfSize) * 4u > numj) {
                    std::cout << ", ...";
                }
            }
            if (i != numi - 1) {
                std::cout << std::endl;
            } else if (uint32_t(cache->cachedTfSize) * 4u > numi) {
                std::cout << std::endl << "..." << std::endl;
            }
        }
        std::cout << std::endl << std::endl;

        std::cout << "rhs:" << std::endl;
        for (uint32_t i = 0; i < numi; i++) {
            std::cout << rhs(i);
            if (i != numi - 1) {
                std::cout << std::endl;
            } else if (uint32_t(cache->cachedTfSize) * 4u > numi) {
                std::cout << std::endl << "..." << std::endl;
            }
        }
        std::cout << std::endl << std::endl;

        solveLinearSystemEigenSymmetric(
                settings.eigenSolverType, Real(settings.relaxationLambda), lhs, rhs, cache->x);
        auto endSolve = std::chrono::system_clock::now();
        std::cout << "Elapsed time solve: " << std::chrono::duration<double>(endSolve - startSolve).count() << "s" << std::endl;
    } else {
        // Set up the system matrix A and the right hand side vector b.
        auto startSetup = std::chrono::system_clock::now();
        if (!settings.useSparseSolve) {
            for (uint32_t i = 0; i < numMatrixRows; i++) {
                for (uint32_t j = 0; j < tfNumEntries; j++) {
                    cache->A(i, j) = 0.0f;
                }
            }
        }
        if (settings.useSparseSolve) {
            buildSystemSparse();
        } else {
            buildSystemDense();
        }
        cache->fieldEntryGT = {};
        cache->fieldEntryOpt = {};
        auto endSetup = std::chrono::system_clock::now();
        std::cout << "Elapsed time setup: " << std::chrono::duration<double>(endSetup - startSetup).count() << "s" << std::endl;

        auto startSolve = std::chrono::system_clock::now();
        if (settings.useSparseSolve) {
#ifdef CUDA_ENABLED
            if (settings.backend == OLSBackend::CUDA) {
                int numRows, nnz;
                Real* b, *csrVals;
                int* csrRowPtr, *csrColInd;
                if (settings.useCudaMatrixSetup) {
                    numRows = cache->cudaNumRows;
                    nnz = cache->cudaNumNnz;
                    b = cache->cudaBSparse;
                    csrVals = cache->cudaCsrVals;
                    csrRowPtr = cache->cudaCsrRowPtr;
                    csrColInd = cache->cudaCsrColInd;
                } else {
                    numRows = int(cache->csrRowPtr.size() - 1);
                    nnz = int(cache->csrVals.size());
                    b = cache->bSparse.data();
                    csrVals = cache->csrVals.data();
                    csrRowPtr = cache->csrRowPtr.data();
                    csrColInd = cache->csrColInd.data();
                }
                if (settings.useNormalEquations) {
                    solveLeastSquaresCudaSparseNormalEquations(
                            settings.cudaSparseSolverType, settings.eigenSolverType, settings.useNormalEquations,
                            Real(settings.relaxationLambda),
                            numRows, int(tfNumEntries), nnz,
                            csrVals, csrRowPtr, csrColInd, b, cache->x);
                } else {
                    solveLeastSquaresCudaSparse(
                            settings.cudaSparseSolverType, settings.useNormalEquations,
                            Real(settings.relaxationLambda),
                            numRows, int(tfNumEntries), nnz,
                            csrVals, csrRowPtr, csrColInd, b, cache->x);
                }
                if (cache->cudaCsrVals) {
                    freeSystemMatrixCudaSparse(cache->cudaCsrVals, cache->cudaCsrRowPtr, cache->cudaCsrColInd, cache->cudaBSparse);
                    cache->cudaCsrVals = {};
                    cache->cudaCsrRowPtr = {};
                    cache->cudaCsrColInd = {};
                    cache->cudaBSparse = {};
                };
            } else {
#endif
                if (settings.useNormalEquations) {
                    solveLeastSquaresEigenSparseNormalEquations(
                            settings.eigenSolverType, Real(settings.relaxationLambda),
                            cache->sparseA, cache->b, cache->x);
                } else {
                    solveLeastSquaresEigenSparse(
                            settings.eigenSparseSolverType, Real(settings.relaxationLambda),
                            cache->sparseA, cache->b, cache->x);
                }
#ifdef CUDA_ENABLED
            }
#endif
        } else {
            const Real lambdaL = settings.useNormalEquations ? Real(settings.relaxationLambda) : Real(0.0);
#ifdef CUDA_ENABLED
            if (settings.backend == OLSBackend::CUDA) {
                solveLeastSquaresCudaDense(
                        settings.cudaSolverType, settings.useNormalEquations, lambdaL, cache->A, cache->b, cache->x);
            } else {
#endif
                solveLeastSquaresEigenDense(
                        settings.eigenSolverType, settings.useNormalEquations, lambdaL, cache->A, cache->b, cache->x);
#ifdef CUDA_ENABLED
            }
#endif
        }
        auto endSolve = std::chrono::system_clock::now();
        std::cout << "Elapsed time solve: " << std::chrono::duration<double>(endSolve - startSolve).count() << "s" << std::endl;
    }

    // Debugging.
    uint32_t numi = std::min(cache->cachedNumVoxels * 4u, 32u);
    uint32_t numj = std::min(cache->cachedTfSize * 4u, 32u);
    if (!settings.useSparseSolve && settings.backend != OLSBackend::VULKAN) {
        std::cout << "A:" << std::endl;
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
    }

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
        Real value = cache->x(j);
        if (std::isnan(value)) {
            value = 0.0f;
        }
        cache->x(j) = std::clamp(value, Real(0), Real(1));
    }

    tfArrayOpt.resize(settings.tfSize);
    if constexpr (std::is_same<Real, double>()) {
        for (uint32_t i = 0; i < settings.tfSize; i++) {
            uint32_t j = i * 4;
            tfArrayOpt[i] = glm::vec4(cache->x(j), cache->x(j + 1), cache->x(j + 2), cache->x(j + 3));
        }
    } else {
        memcpy(tfArrayOpt.data(), cache->x.data(), sizeof(glm::vec4) * settings.tfSize);
    }
}
