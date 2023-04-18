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

#include <Utils/AppSettings.hpp>
#include <Graphics/Scene/Camera.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/imgui.h>
#include <ImGui/imgui_stdlib.h>
#include <ImGui/imgui_custom.h>

#include "Volume/VolumeData.hpp"
#include "OptimizerPass.hpp"
#include "TFOptimization.hpp"

TFOptimization::TFOptimization(sgl::vk::Renderer* parentRenderer) : parentRenderer(parentRenderer) {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    uint32_t maxSharedMemorySize = device->getLimits().maxComputeSharedMemorySize;
    uint32_t subgroupSize = device->getPhysicalDeviceSubgroupProperties().subgroupSize;
    uint32_t maxTfSize = maxSharedMemorySize / (16 * subgroupSize);
    for (int tfSize : possibleTfSizes) {
        if (tfSize <= int(maxTfSize)) {
            tfSizes.push_back(tfSize);
            tfSizeStrings.push_back(std::to_string(tfSize));
        }
    }
    if (maxTfSize < 64) {
        tfSizeIdx = int(tfSizes.size()) - 1;
    } else {
        tfSizeIdx = int(std::find(tfSizes.begin(), tfSizes.end(), 64) - tfSizes.begin());
    }

    worker = new TFOptimizationWorker(parentRenderer);
}

TFOptimization::~TFOptimization() {
    worker->join();
}

void TFOptimization::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
    if (isNewData) {
        fieldIdxGT = 0;
        fieldIdxOpt = 0;
    }
}

void TFOptimization::openDialog() {
    ImGui::OpenPopup("Optimize Transfer Function");
    isOptimizationSettingsDialogOpen = true;
}

void TFOptimization::renderGuiDialog() {
    bool shallStartOptimization = false;
    bool workerHasReply = false;
    TFOptimizationWorkerReply reply;
    if (ImGui::BeginPopupModal(
            "Optimize Transfer Function", &isOptimizationSettingsDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        fieldIdxGT = std::min(fieldIdxGT, int(fieldNames.size()) - 1);
        fieldIdxOpt = std::min(fieldIdxOpt, int(fieldNames.size()) - 1);
        ImGui::Combo(
                "Field GT", &fieldIdxGT, fieldNames.data(), int(fieldNames.size()));
        ImGui::Combo(
                "Field Opt.", &fieldIdxOpt, fieldNames.data(), int(fieldNames.size()));
        ImGui::Combo(
                "TF Size", &tfSizeIdx, tfSizeStrings.data(), int(tfSizeStrings.size()));
        ImGui::Combo(
                "Optimizer", (int*)&optimizerType,
                OPTIMIZER_TYPE_NAMES, IM_ARRAYSIZE(OPTIMIZER_TYPE_NAMES));
        ImGui::SliderInt("Epochs", &maxNumEpochs, 1, 1000);
        ImGui::SliderFloat("alpha", &learningRate, 0.0f, 1.0f);
        if (optimizerType == OptimizerType::ADAM) {
            ImGui::SliderFloat("beta1", &beta1, 0.0f, 1.0f);
            ImGui::SliderFloat("beta2", &beta2, 0.0f, 1.0f);
        }

        if (ImGui::Button("OK", ImVec2(120, 0))) {
            shallStartOptimization = true;
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }

        if (shallStartOptimization) {
            ImGui::OpenPopup("Optimization Progress");
            isOptimizationProgressDialogOpen = true;

            // Initialize the transfer function buffer.
            auto tfBuffer = worker->getTFBuffer();
            auto& tfWidget = volumeData->getMultiVarTransferFunctionWindow();
            auto& tfImage = tfWidget.getTransferFunctionMapTextureVulkan()->getImage();
            sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
            device->waitIdle();
            VkCommandBuffer commandBuffer = device->beginSingleTimeCommands();
            tfImage->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, commandBuffer);
            tfImage->copyToBufferLayered(tfBuffer, uint32_t(fieldIdxOpt), commandBuffer);
            device->endSingleTimeCommands(commandBuffer);

            TFOptimizationWorkerSettings settings;
            settings.fieldIdxGT = fieldIdxGT;
            settings.fieldIdxOpt = fieldIdxOpt;
            settings.tfSize = uint32_t(tfSizes.at(tfSizeIdx));
            settings.optimizerType = optimizerType;
            settings.maxNumEpochs = maxNumEpochs;
            settings.learningRate = learningRate;
            settings.beta1 = beta1;
            settings.beta2 = beta2;
            settings.epsilon = epsilon;
            worker->queueRequest(settings, volumeData);
        }
        if (ImGui::BeginPopupModal(
                "Optimization Progress", &isOptimizationProgressDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Progress: Epoch %d of %d...", 1, maxNumEpochs);
            ImGui::ProgressSpinner(
                    "##progress-spinner-tfopt", -1.0f, -1.0f, 4.0f,
                    ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
            ImGui::SameLine();
            ImGui::ProgressBar(worker->getProgress(), ImVec2(300, 0));
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                worker->stop();
            }
            workerHasReply = worker->getReply(reply);
            if (workerHasReply && reply.hasStopped) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::EndPopup();
    }

    if (workerHasReply && !reply.hasStopped) {
        auto tfBuffer = worker->getTFBuffer();
        auto& tfWidget = volumeData->getMultiVarTransferFunctionWindow();
        auto& tfImage = tfWidget.getTransferFunctionMapTextureVulkan()->getImage();
        tfImage->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, parentRenderer->getVkCommandBuffer());
        tfImage->copyFromBufferLayered(tfBuffer, uint32_t(fieldIdxOpt), parentRenderer->getVkCommandBuffer());
    }
}



void TFOptimizationWorker::join() {
    if (supportsAsyncCompute && !programIsFinished) {
        {
            std::lock_guard<std::mutex> lockRequest(requestMutex);
            programIsFinished = true;
            hasRequest = true;
            {
                std::lock_guard<std::mutex> lockReply(replyMutex);
                this->hasReply = false;
            }
            hasReplyConditionVariable.notify_all();
        }
        hasRequestConditionVariable.notify_all();
        if (requesterThread.joinable()) {
            requesterThread.join();
        }
    }
}

void TFOptimizationWorker::queueRequest(const TFOptimizationWorkerSettings& newSettings, VolumeData* volumeData) {
    {
        std::lock_guard<std::mutex> lock(requestMutex);
        maxNumEpochs = settings.maxNumEpochs;
        currentEpoch = 0;
        settings = newSettings;
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
        hasRequest = true;
    }
    hasRequestConditionVariable.notify_all();
}

void TFOptimizationWorker::stop() {
    shallStop = true;
}

float TFOptimizationWorker::getProgress() const {
    return float(currentEpoch) / float(maxNumEpochs);
}

bool TFOptimizationWorker::getReply(TFOptimizationWorkerReply& reply) {
    if (!supportsAsyncCompute && hasRequest) {
        if (currentEpoch == 0) {
            shallStop = false;
        }
        runEpochs();
        if (currentEpoch == maxNumEpochs) {
            hasRequest = false;
            hasReply = true;
        }
    }

    bool hasReply;
    {
        std::lock_guard<std::mutex> lock(replyMutex);
        hasReply = this->hasReply;
        if (hasReply) {
            reply.hasStopped = hasStopped;
        }

        // Now, new requests can be worked on.
        this->hasReply = false;
        this->hasStopped = false;
    }
    hasReplyConditionVariable.notify_all();
    return hasReply;
}

const sgl::vk::BufferPtr& TFOptimizationWorker::getTFBuffer() {
    return transferFunctionBuffer;
}

void TFOptimizationWorker::mainLoop() {
#ifdef TRACY_ENABLE
    tracy::SetThreadName("TFOptimizationWorker");
#endif

    while (true) {
        std::unique_lock<std::mutex> requestLock(requestMutex);
        hasRequestConditionVariable.wait(requestLock, [this] { return hasRequest; });

        if (programIsFinished) {
            break;
        }

        if (hasRequest) {
            hasRequest = false;
            shallStop = false;
            requestLock.unlock();

            runEpochs();

            std::lock_guard<std::mutex> replyLock(replyMutex);
            hasReply = true;
        }
    }
}

TFOptimizationWorker::TFOptimizationWorker(sgl::vk::Renderer* parentRenderer) : parentRenderer(parentRenderer) {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    renderer = new sgl::vk::Renderer(device, 100);
    if (device->getGraphicsQueue() == device->getComputeQueue()) {
        supportsAsyncCompute = false;
    }
    fence = std::make_shared<sgl::vk::Fence>(device);

    sgl::vk::CommandPoolType commandPoolType{};
    commandPoolType.queueFamilyIndex = device->getComputeQueueIndex();
    commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandBuffer = device->allocateCommandBuffer(commandPoolType, &commandPool);

    // TODO
    //gtForwardPass = std::make_shared<ForwardPass>(renderer);
    //forwardPass = std::make_shared<ForwardPass>(renderer);
    //lossPass = std::make_shared<LossPass>(renderer);
    //adjointPass = std::make_shared<AdjointPass>(renderer);
    //smoothingPriorLossPass = std::make_shared<SmoothingPriorLossPass>(renderer);
    optimizerPass = std::make_shared<OptimizerPass>(renderer);

    if (supportsAsyncCompute) {
        requesterThread = std::thread(&TFOptimizationWorker::mainLoop, this);
    }
}

void TFOptimizationWorker::recreateCache(
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

TFOptimizationWorker::~TFOptimizationWorker() {
    if (renderer) {
        gtForwardPass = {};
        forwardPass = {};
        lossPass = {};
        adjointPass = {};
        smoothingPriorLossPass = {};
        optimizerPass = {};

        sgl::vk::Device *device = renderer->getDevice();
        device->freeCommandBuffer(commandPool, commandBuffer);
        delete renderer;
    }
}

void TFOptimizationWorker::runEpochs() {
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

void TFOptimizationWorker::sampleCameraPoses() {
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

void TFOptimizationWorker::runEpoch() {
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
    //smoothingPriorLossPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Run the optimizer.
    //optimizerPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}
