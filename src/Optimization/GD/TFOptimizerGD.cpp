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

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Volume/VolumeData.hpp"
#include "../CopyFieldImages.hpp"
#include "GradientPass.hpp"
#include "OptimizerPass.hpp"
#include "TFOptimizerGD.hpp"

TFOptimizerGD::TFOptimizerGD(
        sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute)
        : TFOptimizer(renderer, parentRenderer, supportsAsyncCompute) {
    settingsBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), sizeof(UniformSettings),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    gradientPass = std::make_shared<GradientPass>(renderer);
    optimizerPass = std::make_shared<OptimizerPass>(renderer);
}

TFOptimizerGD::~TFOptimizerGD() {
    if (renderer) {
        gradientPass = {};
        optimizerPass = {};
    }
}

void TFOptimizerGD::onRequestQueued(VolumeData* volumeData) {
    maxNumEpochs = settings.maxNumEpochs;
    currentEpoch = 0;

    auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
    std::string fieldNameGT = fieldNames.at(settings.fieldIdxGT);
    std::string fieldNameOpt = fieldNames.at(settings.fieldIdxOpt);
    auto fieldEntryGT = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNameGT);
    auto fieldEntryOpt = volumeData->getFieldEntryDevice(FieldType::SCALAR, fieldNameOpt);
    auto minMaxGT = tfWindow.getSelectedRangePair(settings.fieldIdxGT);
    auto minMaxOpt = tfWindow.getSelectedRangePair(settings.fieldIdxOpt);
    uniformSettings.minGT = minMaxGT.first;
    uniformSettings.maxGT = minMaxGT.second;
    uniformSettings.minOpt = minMaxOpt.first;
    uniformSettings.maxOpt = minMaxOpt.second;
    uniformSettings.xs = uint32_t(volumeData->getGridSizeX());
    uniformSettings.ys = uint32_t(volumeData->getGridSizeY());
    uniformSettings.zs = uint32_t(volumeData->getGridSizeZ());
    uniformSettings.tfNumEntries = settings.tfSize * 4u;

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    gradientPass->setSettings(settings.lossType);
    optimizerPass->setOptimizerType(settings.optimizerType);
    optimizerPass->setSettings(
            settings.lossType, settings.tfSize * 4u,
            settings.learningRate, settings.beta1, settings.beta2, settings.epsilon);

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
        gradientPass->setBuffers(
                settings.tfSize, settingsBuffer, tfGTBuffer, tfOptBuffer, tfOptGradientBuffer);
        optimizerPass->setBuffers(tfOptBuffer, tfOptGradientBuffer);
    }

    cachedTfSize = settings.tfSize;

    std::vector<glm::vec4> tfGT = volumeData->getMultiVarTransferFunctionWindow().getTransferFunctionMap_sRGBDownscaled(
            settings.fieldIdxGT, int(settings.tfSize));
    tfGTBuffer->uploadData(sizeof(glm::vec4) * tfGT.size(), tfGT.data());

    std::vector<glm::vec4> tfOpt = volumeData->getMultiVarTransferFunctionWindow().getTransferFunctionMap_sRGBDownscaled(
            settings.fieldIdxOpt, int(settings.tfSize));
    tfOptBuffer->uploadData(sizeof(glm::vec4) * tfOpt.size(), tfOpt.data());

    CopyFieldImageDestinationData copyFieldImageDestinationData{};
    copyFieldImageDestinationData.inputImageGT = &imageViewFieldGT;
    copyFieldImageDestinationData.inputImageOpt = &imageViewFieldOpt;
    copyFieldImages(
            parentRenderer->getDevice(),
            uint32_t(volumeData->getGridSizeX()),
            uint32_t(volumeData->getGridSizeY()),
            uint32_t(volumeData->getGridSizeZ()),
            fieldEntryGT->getVulkanImage(), fieldEntryOpt->getVulkanImage(),
            copyFieldImageDestinationData,
            settings.fieldIdxGT, settings.fieldIdxOpt, false, false);

    gradientPass->setInputImages(imageViewFieldGT, imageViewFieldOpt);
}

float TFOptimizerGD::getProgress() {
    return float(currentEpoch) / float(maxNumEpochs);
}

void TFOptimizerGD::runOptimization(bool& shallStop, bool& hasStopped) {
    if (currentEpoch == maxNumEpochs) {
        return;
    }

    gradientPass->setShaderDirty();

    // TODO: Add support for double buffering?
    auto startSolve = std::chrono::system_clock::now();
    for (; currentEpoch < maxNumEpochs; currentEpoch++) {
        if (shallStop) {
            hasStopped = true;
            break;
        }

        renderer->setCustomCommandBuffer(commandBuffer, false);
        renderer->beginCommandBuffer();

        if (currentEpoch == 0) {
            settingsBuffer->updateData(sizeof(UniformSettings), &uniformSettings, renderer->getVkCommandBuffer());
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

        //std::cout << "x:" << std::endl;
        auto* tfData = reinterpret_cast<glm::vec4*>(tfDownloadStagingBuffer->mapMemory());
        for (uint32_t i = 0; i < settings.tfSize; i++) {
            //std::cout << tfData[i][0] << ", " << tfData[i][1] << ", " << tfData[i][2] << ", " << tfData[i][3] << std::endl;
            tfArrayOpt.at(i) = glm::clamp(tfData[i], 0.0f, 1.0f);
        }
        tfDownloadStagingBuffer->unmapMemory();
        //std::cout << std::endl << std::endl;

        if (supportsAsyncCompute) {
            auto endSolve = std::chrono::system_clock::now();
            std::cout << "Elapsed time solve: " << std::chrono::duration<double>(endSolve - startSolve).count() << "s" << std::endl;
        }
    }
}

void TFOptimizerGD::runEpoch() {
    // Compute the gradients wrt. the transfer function entries.
    gradientPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Run the optimizer.
    optimizerPass->setEpochIndex(currentEpoch);
    optimizerPass->render();
    renderer->insertMemoryBarrier(
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}
