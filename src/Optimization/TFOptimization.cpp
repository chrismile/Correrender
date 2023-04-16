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

#include <Utils/AppSettings.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/imgui.h>
#include <ImGui/imgui_stdlib.h>
#include <ImGui/imgui_custom.h>

#include "Volume/VolumeData.hpp"
#include "TFOptimization.hpp"

TFOptimization::TFOptimization() {
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

    worker = new TFOptimizationWorker;
}

void TFOptimization::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
    if (isNewData) {
        fieldGTIdx = 0;
        fieldOptIdx = 0;
    }
}

void TFOptimization::openDialog() {
    ImGui::OpenPopup("Optimize Transfer Function");
    isOptimizationSettingsDialogOpen = true;
}

void TFOptimization::renderGuiDialog() {
    bool shallStartOptimization = false;
    if (ImGui::BeginPopupModal(
            "Optimize Transfer Function", &isOptimizationSettingsDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
        auto fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        fieldGTIdx = std::min(fieldGTIdx, int(fieldNames.size()) - 1);
        fieldOptIdx = std::min(fieldOptIdx, int(fieldNames.size()) - 1);
        ImGui::Combo(
                "Field GT", &fieldGTIdx, fieldNames.data(), int(fieldNames.size()));
        ImGui::Combo(
                "Field Opt.", &fieldOptIdx, fieldNames.data(), int(fieldNames.size()));
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
            time = 0.0f;

            // Initialize the transfer function buffer.
            auto tfBuffer = worker->getTFBuffer();
            auto& tfWidget = volumeData->getMultiVarTransferFunctionWindow();
            auto& tfImage = tfWidget.getTransferFunctionMapTextureVulkan()->getImage();
            tfImage->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
            tfImage->copyToBufferLayered(tfBuffer, uint32_t(fieldOptIdx), renderer->getVkCommandBuffer());
        }
        if (ImGui::BeginPopupModal(
                "Optimization Progress", &isOptimizationProgressDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
            time += 0.001f;
            time = std::min(time, 1.0f);
            ImGui::Text("Progress: Epoch %d of %d...", 1, maxNumEpochs);
            ImGui::ProgressSpinner(
                    "##progress-spinner-tfopt", -1.0f, -1.0f, 4.0f,
                    ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
            ImGui::SameLine();
            ImGui::ProgressBar(time, ImVec2(300, 0));
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::EndPopup();
    }

    if (worker->getIsResultAvailable()) {
        auto tfBuffer = worker->getTFBuffer();
        auto& tfWidget = volumeData->getMultiVarTransferFunctionWindow();
        auto& tfImage = tfWidget.getTransferFunctionMapTextureVulkan()->getImage();
        tfImage->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, renderer->getVkCommandBuffer());
        tfImage->copyFromBufferLayered(tfBuffer, uint32_t(fieldOptIdx), renderer->getVkCommandBuffer());
    }
}


const sgl::vk::BufferPtr& TFOptimizationWorker::getTFBuffer() {
    return transferFunctionBuffer;
}

bool TFOptimizationWorker::getIsResultAvailable() {
    return false;
}

void TFOptimizationWorker::runEpochs() {
    for (int epoch = 0; epoch < settings.maxNumEpochs; epoch++) {
        runEpoch();
    }
}

void TFOptimizationWorker::runEpoch() {
    // Run the forward pass.

    // Compute the image loss.

    // Clear the gradients.

    // Compute the gradients wrt. the transfer function entries for the image loss.

    // Compute the gradients wrt. the transfer function entries for the smoothing prior loss.

    // Run the optimizer.
}
