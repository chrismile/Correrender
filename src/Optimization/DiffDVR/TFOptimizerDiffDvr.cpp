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
#include "Optimization/DiffDVR/SmoothingPriorLossPass.hpp"
#include "Optimization/DiffDVR/OptimizerPass.hpp"
#include "TFOptimizerDiffDvr.hpp"

TFOptimizerDiffDvr::TFOptimizerDiffDvr(
        sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute)
        : TFOptimizer(renderer, parentRenderer, supportsAsyncCompute) {
    // TODO
    //gtForwardPass = std::make_shared<ForwardPass>(renderer);
    //forwardPass = std::make_shared<ForwardPass>(renderer);
    //lossPass = std::make_shared<LossPass>(renderer);
    //adjointPass = std::make_shared<AdjointPass>(renderer);
    smoothingPriorLossPass = std::make_shared<SmoothingPriorLossPass>(renderer);
    optimizerPass = std::make_shared<OptimizerPass>(renderer);
}

TFOptimizerDiffDvr::~TFOptimizerDiffDvr() {
    if (renderer) {
        gtForwardPass = {};
        forwardPass = {};
        lossPass = {};
        adjointPass = {};
        smoothingPriorLossPass = {};
        optimizerPass = {};
    }
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
    auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    // TODO
    parentRenderer->getDevice()->waitIdle();
    auto fieldEntryGT = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNames.at(settings.fieldIdxGT));
    auto fieldEntryOpt = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNames.at(settings.fieldIdxOpt));
    recreateCache(
            fieldEntryGT->getVulkanImage()->getImageSettings().format,
            fieldEntryOpt->getVulkanImage()->getImageSettings().format,
            uint32_t(volumeData->getGridSizeX()),
            uint32_t(volumeData->getGridSizeY()),
            uint32_t(volumeData->getGridSizeZ()));
    fieldEntryGT->getVulkanImage()->copyToImage(
            imageViewFieldGT->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, commandBuffer);
    fieldEntryOpt->getVulkanImage()->copyToImage(
            imageViewFieldOpt->getImage(), VK_IMAGE_ASPECT_COLOR_BIT, commandBuffer);
}

void TFOptimizerDiffDvr::recreateCache(
        VkFormat formatGT, VkFormat formatOpt, uint32_t xs, uint32_t ys, uint32_t zs) {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    uint32_t totalSize = batchSize * viewportHeight * viewportWidth;
    uint32_t cachedTotalSize = cachedBatchSize * cachedViewportHeight * cachedViewportWidth;

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
        finalColorsBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * totalSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        gtFinalColorsBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * totalSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        terminationIndexBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(int32_t) * totalSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    }

    if (cachedTfSize != settings.tfSize) {
        transferFunctionBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY, false);
        transferFunctionGradientBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec4) * settings.tfSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    }

    if (!imageViewFieldGT || formatGT != cachedFormatGT || formatOpt != cachedFormatOpt
            || imageViewFieldGT->getImage()->getImageSettings().width != xs
            || imageViewFieldGT->getImage()->getImageSettings().height != ys
            || imageViewFieldGT->getImage()->getImageSettings().depth != zs) {
        sgl::vk::ImageSettings imageSettings;
        imageSettings.width = xs;
        imageSettings.height = ys;
        imageSettings.depth = zs;
        imageSettings.imageType = VK_IMAGE_TYPE_3D;
        imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageSettings.sharingMode = VK_SHARING_MODE_CONCURRENT;
        imageSettings.format = formatGT;
        imageViewFieldGT = std::make_shared<sgl::vk::ImageView>(
                std::make_shared<sgl::vk::Image>(device, imageSettings), VK_IMAGE_VIEW_TYPE_3D);
        imageSettings.format = formatOpt;
        imageViewFieldOpt = std::make_shared<sgl::vk::ImageView>(
                std::make_shared<sgl::vk::Image>(device, imageSettings), VK_IMAGE_VIEW_TYPE_3D);
    }

    cachedBatchSize = batchSize;
    cachedViewportWidth = viewportWidth;
    cachedViewportHeight = viewportHeight;
    cachedTfSize = settings.tfSize;
    cachedFormatGT = formatGT;
    cachedFormatOpt = formatOpt;
}

float TFOptimizerDiffDvr::getProgress() {
    return float(currentEpoch) / float(maxNumEpochs);
}

void TFOptimizerDiffDvr::runOptimization(bool shallStop, bool& hasStopped) {
    if (currentEpoch == 0) {
        dvrSettingsBuffer->updateData(sizeof(DvrSettingsBufferTf), &dvrSettings, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }

    // TODO: Add support for double buffering?
    for (; currentEpoch < maxNumEpochs; currentEpoch++) {
        if (shallStop) {
            hasStopped = true;
            break;
        }

        renderer->setCustomCommandBuffer(commandBuffer, false);
        renderer->beginCommandBuffer();

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

    //renderer->getDevice()->waitComputeQueueIdle();
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
    //gtForwardPass->render();
    //forwardPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Compute the image loss.
    //lossPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Clear the gradients.
    transferFunctionGradientBuffer->fill(0, renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            transferFunctionGradientBuffer);

    // Compute the gradients wrt. the transfer function entries for the image loss.
    //adjointPass->render();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            transferFunctionGradientBuffer);

    // Compute the gradients wrt. the transfer function entries for the smoothing prior loss.
    smoothingPriorLossPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Run the optimizer.
    optimizerPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}
