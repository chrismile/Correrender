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
#include "NormalEquations.hpp"
#include "TFOptimizerOLS.hpp"

struct TFOptimizerOLSCache {
    uint32_t cachedTfSize = 0;
    uint32_t cachedNumVoxels = 0;
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

    // Implicit matrix.
    VkFormat cachedFormatGT{}, cachedFormatOpt{};
    sgl::vk::ImageViewPtr inputImageGT, inputImageOpt;
    sgl::vk::BufferPtr lhsBuffer, rhsBuffer;
    sgl::vk::BufferPtr lhsStagingBuffer, rhsStagingBuffer;
    sgl::vk::BufferPtr tfGTBuffer;
};

TFOptimizerOLS::TFOptimizerOLS(
        sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute)
        : TFOptimizer(renderer, parentRenderer, supportsAsyncCompute) {
    cache = new TFOptimizerOLSCache;
    normalEquationsComputePass = std::make_shared<NormalEquationsComputePass>(renderer);
    normalEquationsCopySymmetricPass = std::make_shared<NormalEquationsCopySymmetricPass>(renderer);
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
                    device, sizeof(glm::vec4) * sizeof(glm::vec4) * settings.tfSize * settings.tfSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            cache->rhsBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, sizeof(glm::vec4) * settings.tfSize,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            cache->lhsStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, sizeof(glm::vec4) * sizeof(glm::vec4) * settings.tfSize * settings.tfSize,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_GPU_TO_CPU);
            cache->rhsStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, sizeof(glm::vec4) * settings.tfSize,
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
        auto formatGT = fieldEntryGT->getVulkanImage()->getImageSettings().format;
        auto formatOpt = fieldEntryOpt->getVulkanImage()->getImageSettings().format;
        auto xs = uint32_t(volumeData->getGridSizeX());
        auto ys = uint32_t(volumeData->getGridSizeY());
        auto zs = uint32_t(volumeData->getGridSizeZ());
        if (!cache->inputImageGT || formatGT != cache->cachedFormatGT || formatOpt != cache->cachedFormatOpt
                || cache->inputImageGT->getImage()->getImageSettings().width != xs
                || cache->inputImageGT->getImage()->getImageSettings().height != ys
                || cache->inputImageGT->getImage()->getImageSettings().depth != zs) {
            sgl::vk::ImageSettings imageSettings;
            imageSettings.width = xs;
            imageSettings.height = ys;
            imageSettings.depth = zs;
            imageSettings.imageType = VK_IMAGE_TYPE_3D;
            imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
            imageSettings.format = formatGT;
            cache->inputImageGT = std::make_shared<sgl::vk::ImageView>(
                    std::make_shared<sgl::vk::Image>(device, imageSettings), VK_IMAGE_VIEW_TYPE_3D);
            imageSettings.format = formatOpt;
            cache->inputImageOpt = std::make_shared<sgl::vk::ImageView>(
                    std::make_shared<sgl::vk::Image>(device, imageSettings), VK_IMAGE_VIEW_TYPE_3D);
            cache->cachedFormatGT = formatGT;
            cache->cachedFormatOpt = formatOpt;
        }
        auto layoutOldGT = fieldEntryGT->getVulkanImage()->getVkImageLayout();
        auto layoutOldOpt = fieldEntryOpt->getVulkanImage()->getVkImageLayout();
        parentRenderer->getDevice()->waitIdle();

        auto commandBufferGraphics = device->beginSingleTimeCommands(device->getGraphicsQueueIndex());
        fieldEntryGT->getVulkanImage()->insertMemoryBarrier(
                commandBufferGraphics,
                layoutOldGT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
                device->getGraphicsQueueIndex(),
                device->getComputeQueueIndex());
        if (settings.fieldIdxGT != settings.fieldIdxOpt) {
            fieldEntryOpt->getVulkanImage()->insertMemoryBarrier(
                    commandBufferGraphics,
                    layoutOldOpt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
                    device->getGraphicsQueueIndex(),
                    device->getComputeQueueIndex());
        }
        device->endSingleTimeCommands(commandBufferGraphics, device->getGraphicsQueueIndex());

        auto commandBufferCompute = device->beginSingleTimeCommands(device->getComputeQueueIndex());
        fieldEntryGT->getVulkanImage()->insertMemoryBarrier(
                commandBufferCompute,
                layoutOldGT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
                device->getGraphicsQueueIndex(),
                device->getComputeQueueIndex());
        if (settings.fieldIdxGT != settings.fieldIdxOpt) {
            fieldEntryOpt->getVulkanImage()->insertMemoryBarrier(
                    commandBufferCompute,
                    layoutOldOpt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_READ_BIT,
                    device->getGraphicsQueueIndex(),
                    device->getComputeQueueIndex());
        }
        cache->inputImageGT->getImage()->insertMemoryBarrier(
                commandBufferCompute,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
        cache->inputImageOpt->getImage()->insertMemoryBarrier(
                commandBufferCompute,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
        fieldEntryGT->getVulkanImage()->copyToImage(
                cache->inputImageGT->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, commandBufferCompute);
        fieldEntryOpt->getVulkanImage()->copyToImage(
                cache->inputImageOpt->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, commandBufferCompute);
        fieldEntryGT->getVulkanImage()->insertMemoryBarrier(
                commandBufferCompute,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldGT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
                device->getComputeQueueIndex(),
                device->getGraphicsQueueIndex());
        if (settings.fieldIdxGT != settings.fieldIdxOpt) {
            fieldEntryOpt->getVulkanImage()->insertMemoryBarrier(
                    commandBufferCompute,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldOpt,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
                    device->getComputeQueueIndex(),
                    device->getGraphicsQueueIndex());
        }
        cache->inputImageGT->getImage()->insertMemoryBarrier(
                commandBufferCompute,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        cache->inputImageOpt->getImage()->insertMemoryBarrier(
                commandBufferCompute,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        device->endSingleTimeCommands(commandBufferCompute, device->getComputeQueueIndex());

        commandBufferGraphics = device->beginSingleTimeCommands(device->getGraphicsQueueIndex());
        fieldEntryGT->getVulkanImage()->insertMemoryBarrier(
                commandBufferGraphics,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldGT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
                device->getComputeQueueIndex(),
                device->getGraphicsQueueIndex());
        if (settings.fieldIdxGT != settings.fieldIdxOpt) {
            fieldEntryOpt->getVulkanImage()->insertMemoryBarrier(
                    commandBufferGraphics,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layoutOldOpt,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_NONE_KHR,
                    device->getComputeQueueIndex(),
                    device->getGraphicsQueueIndex());
        }
        device->endSingleTimeCommands(commandBufferGraphics, device->getGraphicsQueueIndex());
    } else {
        cache->inputImageGT = {};
        cache->inputImageOpt = {};
        cache->lhsBuffer = {};
        cache->rhsBuffer = {};
        cache->lhsStagingBuffer = {};
        cache->rhsStagingBuffer = {};
        cache->tfGTBuffer = {};
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
    cache->cachedUseSparseSolve = settings.useSparseSolve;
}

float TFOptimizerOLS::getProgress() {
    return 0.0f;
}

void TFOptimizerOLS::buildSystemDense() {
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

void TFOptimizerOLS::buildSystemSparse() {
    uint32_t numMatrixRows = cache->cachedNumVoxels * 4;
    uint32_t expectedNumNonZero = numMatrixRows * 2;
    const float* fieldDataGT = cache->fieldEntryGT->data<float>();
    const float* fieldDataOpt = cache->fieldEntryOpt->data<float>();
    float minGT = cache->minMaxGT.first;
    float maxGT = cache->minMaxGT.second;
    float minOpt = cache->minMaxOpt.first;
    float maxOpt = cache->minMaxOpt.second;
    auto Nj = float(cache->cachedTfSize - 1);

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

void TFOptimizerOLS::runOptimization(bool shallStop, bool& hasStopped) {
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

        auto* lhsData = reinterpret_cast<float*>(cache->lhsStagingBuffer->mapMemory());
        auto* rhsData = reinterpret_cast<float*>(cache->rhsStagingBuffer->mapMemory());
        Eigen::MatrixXr lhs = Eigen::MatrixXr(tfNumEntries, tfNumEntries);
        Eigen::MatrixXr rhs = Eigen::VectorXr(tfNumEntries);
#ifdef USE_DOUBLE_PRECISION
        for (uint32_t i = 0; i < tfNumEntries; i++) {
            rhs(i) = Real(lhsData[i]);
        }
        for (uint32_t i = 0; i < tfNumEntries; i++) {
            for (uint32_t j = 0; j < tfNumEntries; j++) {
                rhs(i, j) = Real(lhsData[i + j * tfNumEntries]);
            }
        }
#else
        memcpy(lhs.data(), lhsData, 4 * tfNumEntries * tfNumEntries);
        memcpy(rhs.data(), rhsData, 4 * tfNumEntries);
#endif
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
                settings.eigenSolverType, settings.relaxationLambda, lhs, rhs, cache->x);
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
                solveLeastSquaresCudaSparse(
                        int(numMatrixRows), int(tfNumEntries), int(cache->csrVals.size()),
                        cache->csrVals.data(), cache->csrRowPtr.data(), cache->csrColInd.data(),
                        cache->bSparse.data(), cache->x);
            } else {
#endif
                if (settings.useNormalEquations) {
                    solveLeastSquaresEigenSparseNormalEquations(
                            settings.eigenSolverType, Real(settings.relaxationLambda),
                            cache->sparseA, cache->b, cache->x);
                } else {
                    solveLeastSquaresEigenSparse(
                            settings.eigenSparseSolverType, cache->sparseA, cache->b, cache->x);
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
#ifdef USE_DOUBLE_PRECISION
    for (uint32_t i = 0; i < settings.tfSize; i++) {
        uint32_t j = i * 4;
        tfArrayOpt[i] = glm::vec4(cache->x(j), cache->x(j + 1), cache->x(j + 2), cache->x(j + 3));
    }
#else
    memcpy(tfArrayOpt.data(), cache->x.data(), sizeof(glm::vec4) * settings.tfSize);
#endif
}
