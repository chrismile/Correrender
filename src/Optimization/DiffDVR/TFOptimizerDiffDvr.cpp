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

#include <random>

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Volume/VolumeData.hpp"
#include "Optimization/DiffDVR/DvrForwardPass.hpp"
#include "Optimization/DiffDVR/DvrAdjointPass.hpp"
#include "Optimization/DiffDVR/SmoothingPriorLossPass.hpp"
#include "Optimization/DiffDVR/LossPass.hpp"
#include "Optimization/GD/OptimizerPass.hpp"
#include "../CopyFieldImages.hpp"
#include "TFOptimizerDiffDvr.hpp"

TFOptimizerDiffDvr::TFOptimizerDiffDvr(
        sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute)
        : TFOptimizer(renderer, parentRenderer, supportsAsyncCompute) {
    forwardGTPass = std::make_shared<DvrForwardPass>(renderer);
    forwardOptPass = std::make_shared<DvrForwardPass>(renderer);
    adjointPass = std::make_shared<DvrAdjointPass>(renderer);
    lossPass = std::make_shared<LossPass>(renderer);
    smoothingPriorLossPass = std::make_shared<SmoothingPriorLossPass>(renderer);
    optimizerPass = std::make_shared<OptimizerPass>(renderer);
}

TFOptimizerDiffDvr::~TFOptimizerDiffDvr() {
    forwardGTPass = {};
    forwardOptPass = {};
    adjointPass = {};
    lossPass = {};
    smoothingPriorLossPass = {};
    optimizerPass = {};
}

void TFOptimizerDiffDvr::onRequestQueued(VolumeData* volumeData) {
    maxNumEpochs = settings.maxNumEpochs;
    currentEpoch = 0;
    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    float voxelSizeX = aabb.getDimensions().x / float(volumeData->getGridSizeX());
    float voxelSizeY = aabb.getDimensions().y / float(volumeData->getGridSizeY());
    float voxelSizeZ = aabb.getDimensions().z / float(volumeData->getGridSizeZ());
    float voxelSize = std::min(voxelSizeX, std::min(voxelSizeY, voxelSizeZ));
    dvrSettings.minBoundingBox = aabb.min;
    dvrSettings.maxBoundingBox = aabb.max;
    dvrSettings.stepSize = voxelSize * settings.stepSize;
    dvrSettings.attenuationCoefficient = settings.attenuationCoefficient;
    dvrSettings.imageWidth = settings.imageWidth;
    dvrSettings.imageHeight = settings.imageHeight;
    dvrSettings.batchSize = settings.batchSize;

    auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
    std::string fieldNameGT = fieldNames.at(settings.fieldIdxGT);
    std::string fieldNameOpt = fieldNames.at(settings.fieldIdxOpt);
    auto fieldEntryGT = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNameGT);
    auto fieldEntryOpt = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNameOpt);
    auto minMaxGT = tfWindow.getSelectedRangePair(settings.fieldIdxGT);
    auto minMaxOpt = tfWindow.getSelectedRangePair(settings.fieldIdxOpt);

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    viewportWidth = settings.imageWidth;
    viewportHeight = settings.imageHeight;
    batchSize = settings.batchSize;
    uint32_t totalSize = batchSize * viewportHeight * viewportWidth;
    uint32_t cachedTotalSize = cachedBatchSize * cachedViewportHeight * cachedViewportWidth;
    batchSettingsArray.resize(settings.batchSize);

    forwardGTPass->setSettings(
            false, minMaxGT.first, minMaxGT.second, settings.tfSize,
            settings.imageWidth, settings.imageHeight, settings.batchSize);
    forwardOptPass->setSettings(
            true, minMaxOpt.first, minMaxOpt.second, settings.tfSize,
            settings.imageWidth, settings.imageHeight, settings.batchSize);
    adjointPass->setSettings(
            minMaxOpt.first, minMaxOpt.second, settings.tfSize,
            settings.imageWidth, settings.imageHeight, settings.batchSize, settings.adjointDelayed);
    smoothingPriorLossPass->setSettings(settings.lambdaSmoothingPrior, settings.tfSize);
    lossPass->setSettings(settings.lossType, settings.imageWidth, settings.imageHeight, settings.batchSize);
    optimizerPass->setOptimizerType(settings.optimizerType);
    optimizerPass->setSettings(
            settings.lossType, settings.tfSize * 4u,
            settings.learningRate, settings.beta1, settings.beta2, settings.epsilon);

    dvrSettingsBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(DvrSettingsBufferTf),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    if (cachedBatchSize != batchSize) {
        batchSettingsBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::mat4) * batchSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    }

    if (cachedTotalSize != totalSize) {
        finalColorsGTBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * totalSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        finalColorsOptBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * totalSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        adjointColorsBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * totalSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        terminationIndexBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(int32_t) * totalSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        lossPass->setBuffers(finalColorsGTBuffer, finalColorsOptBuffer, adjointColorsBuffer);
    }

    if (cachedTfSize != settings.tfSize) {
        tfGTBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        tfOptBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        tfDownloadStagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_GPU_TO_CPU);
        tfOptGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        smoothingPriorLossPass->setBuffers(tfOptBuffer, tfOptGradientBuffer);
        optimizerPass->setBuffers(tfOptBuffer, tfOptGradientBuffer);
    }

    if (cachedTotalSize != totalSize || cachedTfSize != settings.tfSize) {
        forwardGTPass->setBuffers(
                dvrSettingsBuffer, batchSettingsBuffer, tfGTBuffer, finalColorsGTBuffer, {});
        forwardOptPass->setBuffers(
                dvrSettingsBuffer, batchSettingsBuffer, tfOptBuffer, finalColorsOptBuffer, terminationIndexBuffer);
        adjointPass->setBuffers(
                dvrSettingsBuffer, batchSettingsBuffer, tfOptBuffer, finalColorsOptBuffer, terminationIndexBuffer,
                adjointColorsBuffer, tfOptGradientBuffer);
    }

    cachedBatchSize = batchSize;
    cachedViewportWidth = viewportWidth;
    cachedViewportHeight = viewportHeight;
    cachedTfSize = settings.tfSize;

    std::vector<glm::vec4> tfGT = volumeData->getMultiVarTransferFunctionWindow().getTransferFunctionMap_sRGBDownscaled(
            settings.fieldIdxGT, int(settings.tfSize));
    tfGTBuffer->uploadData(sizeof(glm::vec4) * tfGT.size(), tfGT.data());

    std::vector<glm::vec4> tfOpt = volumeData->getMultiVarTransferFunctionWindow().getTransferFunctionMap_sRGBDownscaled(
            settings.fieldIdxOpt, int(settings.tfSize));
    tfOptBuffer->uploadData(sizeof(glm::vec4) * tfOpt.size(), tfOpt.data());

    sgl::vk::ImageSamplerPtr cachedSampler;
    sgl::vk::ImageViewPtr cachedImageViewGT = imageViewFieldGT;
    sgl::vk::ImageViewPtr cachedImageViewOpt = imageViewFieldOpt;
    if (textureFieldGT) {
        cachedSampler = textureFieldGT->getImageSampler();
    }
    copyFieldImages(
            parentRenderer->getDevice(),
            uint32_t(volumeData->getGridSizeX()),
            uint32_t(volumeData->getGridSizeY()),
            uint32_t(volumeData->getGridSizeZ()),
            fieldEntryGT->getVulkanImage(), fieldEntryOpt->getVulkanImage(),
            imageViewFieldGT, imageViewFieldOpt,
            cachedFormatGT, cachedFormatOpt,
            settings.fieldIdxGT, settings.fieldIdxOpt, true);

    auto sampler = volumeData->getImageSampler();
    if (cachedSampler != sampler || cachedImageViewGT != imageViewFieldGT || cachedImageViewOpt != imageViewFieldOpt) {
        textureFieldGT = std::make_shared<sgl::vk::Texture>(imageViewFieldGT, sampler);
        textureFieldOpt = std::make_shared<sgl::vk::Texture>(imageViewFieldOpt, sampler);
    }

    forwardGTPass->setScalarFieldTexture(textureFieldGT);
    forwardOptPass->setScalarFieldTexture(textureFieldOpt);
    adjointPass->setScalarFieldTexture(textureFieldOpt);
}

float TFOptimizerDiffDvr::getProgress() {
    return float(currentEpoch) / float(maxNumEpochs);
}

void TFOptimizerDiffDvr::runOptimization(bool shallStop, bool& hasStopped) {
    if (currentEpoch == maxNumEpochs) {
        return;
    }

    // TODO: Add support for double buffering?
    for (; currentEpoch < maxNumEpochs; currentEpoch++) {
        if (shallStop) {
            hasStopped = true;
            break;
        }

        renderer->setCustomCommandBuffer(commandBuffer, false);
        renderer->beginCommandBuffer();

        if (currentEpoch == 0) {
            dvrSettingsBuffer->updateData(sizeof(DvrSettingsBufferTf), &dvrSettings, renderer->getVkCommandBuffer());
            smoothingPriorLossPass->updateUniformBuffer();
            lossPass->updateUniformBuffer();
            renderer->insertMemoryBarrier(
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

        runEpoch();

        renderer->endCommandBuffer();
        renderer->submitToQueue({}, {}, fence, VK_PIPELINE_STAGE_TRANSFER_BIT);
        renderer->resetCustomCommandBuffer();
        fence->wait();
        fence->reset();

        if (!supportsAsyncCompute) {
            break;
        }
    }

    if (currentEpoch == maxNumEpochs) {
        renderer->setCustomCommandBuffer(commandBuffer, false);
        renderer->beginCommandBuffer();

        tfArrayOpt.resize(settings.tfSize);
        tfOptBuffer->copyDataTo(tfDownloadStagingBuffer, commandBuffer);

        renderer->endCommandBuffer();
        renderer->submitToQueue({}, {}, fence, VK_PIPELINE_STAGE_TRANSFER_BIT);
        renderer->resetCustomCommandBuffer();
        fence->wait();
        fence->reset();

        auto* tfData = reinterpret_cast<glm::vec4*>(tfDownloadStagingBuffer->mapMemory());
        for (uint32_t i = 0; i < settings.tfSize; i++) {
            tfArrayOpt.at(i) = glm::clamp(tfData[i], 0.0f, 1.0f);
        }
        tfDownloadStagingBuffer->unmapMemory();
    }
}

void TFOptimizerDiffDvr::sampleCameraPoses() {
    const glm::vec3 globalUp(0.0f, 1.0f, 0.0f);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dist(0, 1);
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        float theta = sgl::TWO_PI * dist(generator);
        float phi = std::acos(1.0f - 2.0f * dist(generator));
        glm::vec3 cameraPosition(std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi));
        glm::vec3 cameraForward = -glm::normalize(cameraPosition);
        glm::vec3 cameraRight = glm::normalize(glm::cross(globalUp, cameraForward));
        glm::vec3 cameraUp = glm::normalize(glm::cross(cameraForward, cameraRight));
        glm::mat4 rotationMatrix;
        for (int i = 0; i < 4; i++) {
            rotationMatrix[i][0] = i < 3 ? cameraRight[0] : 0.0f;
            rotationMatrix[i][1] = i < 3 ? cameraUp[0] : 0.0f;
            rotationMatrix[i][2] = i < 3 ? cameraForward[0] : 0.0f;
            rotationMatrix[i][3] = i < 3 ? 0.0f : 1.0f;
        }
        glm::mat4 viewMatrix = rotationMatrix * sgl::matrixTranslation(-cameraPosition);
        glm::mat4 inverseViewMatrix = glm::inverse(viewMatrix);
        batchSettingsArray.at(batchIdx) = inverseViewMatrix;
    }
    batchSettingsBuffer->updateData(
            sizeof(glm::mat4) * batchSize, batchSettingsArray.data(), renderer->getVkCommandBuffer());
}

void TFOptimizerDiffDvr::runEpoch() {
    // Randomly sample camera poses.
    sampleCameraPoses();

    // Run the forward passes.
    forwardGTPass->render();
    forwardOptPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Compute the image loss.
    lossPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Clear the gradients.
    tfOptGradientBuffer->fill(0, renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            tfOptGradientBuffer);

    // Compute the gradients wrt. the transfer function entries for the smoothing prior loss.
    // This will also clear/overwrite the adjoint colors buffer.
    smoothingPriorLossPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Compute the gradients wrt. the transfer function entries for the image loss.
    adjointPass->render();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            tfOptGradientBuffer);

    // Run the optimizer.
    optimizerPass->setEpochIndex(currentEpoch);
    optimizerPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}
