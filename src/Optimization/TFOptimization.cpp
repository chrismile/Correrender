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
#include "TFOptimizer.hpp"
#include "Optimization/OLS/TFOptimizerOLS.hpp"
#ifdef CUDA_ENABLED
#include "Optimization/OLS/CudaSolver.hpp"
#endif
#include "Optimization/GD/TFOptimizerGD.hpp"
#include "Optimization/DiffDVR/TFOptimizerDiffDvr.hpp"
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

    if (!device->getPhysicalDeviceShaderAtomicFloatFeatures().shaderBufferFloat32AtomicAdd) {
        settings.backend = OLSBackend::CPU;
    }
//#ifdef CUDA_ENABLED
//    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
//            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
//        settings.backend = OLSBackend::CUDA;
//        settings.useSparseSolve = true;
//    }
//#endif

    worker = new TFOptimizationWorker(parentRenderer);
}

void TFOptimization::initialize() {
    worker->initialize();
}

TFOptimization::~TFOptimization() {
    worker->join();
    delete worker;
}

void TFOptimization::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
    if (isNewData) {
        settings.fieldIdxGT = 0;
        settings.fieldIdxOpt = 0;
    }
}

#ifdef CUDA_ENABLED
void TFOptimization::setCudaContext(CUcontext context) {
    worker->setCudaContext(context);
}
#endif

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
        settings.fieldIdxGT = std::min(settings.fieldIdxGT, int(fieldNames.size()) - 1);
        settings.fieldIdxOpt = std::min(settings.fieldIdxOpt, int(fieldNames.size()) - 1);
        ImGui::Combo(
                "Field GT", &settings.fieldIdxGT, fieldNames.data(), int(fieldNames.size()));
        ImGui::Combo(
                "Field Opt.", &settings.fieldIdxOpt, fieldNames.data(), int(fieldNames.size()));
        ImGui::Combo(
                "TF Size", &tfSizeIdx, tfSizeStrings.data(), int(tfSizeStrings.size()));
        ImGui::Combo(
                "TF Optimizer", (int*)&settings.optimizerMethod,
                TF_OPTIMIZER_METHOD_NAMES, IM_ARRAYSIZE(TF_OPTIMIZER_METHOD_NAMES));

        if (ImGui::CollapsingHeader(
                "Optimizer Settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
            if (settings.optimizerMethod == TFOptimizerMethod::DIFF_DVR
                    || settings.optimizerMethod == TFOptimizerMethod::GD) {
                ImGui::Combo(
                        "Optimizer", (int*)&settings.optimizerType,
                        OPTIMIZER_TYPE_NAMES, IM_ARRAYSIZE(OPTIMIZER_TYPE_NAMES));
                ImGui::Combo(
                        "Loss", (int*)&settings.lossType,
                        LOSS_TYPE_NAMES, IM_ARRAYSIZE(LOSS_TYPE_NAMES));
                ImGui::SliderInt("Epochs", &settings.maxNumEpochs, 1, 1000);
                ImGui::SliderFloat("alpha", &settings.learningRate, 0.0f, 1.0f);
                if (settings.optimizerType == OptimizerType::ADAM) {
                    ImGui::SliderFloat("beta1", &settings.beta1, 0.0f, 1.0f);
                    ImGui::SliderFloat("beta2", &settings.beta2, 0.0f, 1.0f);
                }
            } else {
                auto* device = parentRenderer->getDevice();
                int numBackends = 2;
#ifdef CUDA_ENABLED
                if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
                        && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
                    numBackends++;
                }
#endif
                if (ImGui::Combo("Backend", (int*)&settings.backend, OLS_BACKEND_NAMES, numBackends)) {
                    if (settings.backend == OLSBackend::VULKAN) {
                        settings.useNormalEquations = true;
                    }
                }
                if (settings.optimizerMethod == TFOptimizerMethod::OLS
                        && settings.backend != OLSBackend::VULKAN) {
                    ImGui::Checkbox("Use Sparse Solve Step", &settings.useSparseSolve);
                }
                if (settings.backend == OLSBackend::CUDA && settings.useSparseSolve) {
                    ImGui::Checkbox("GPU Matrix Setup", &settings.useCudaMatrixSetup);
                }
                if (settings.backend == OLSBackend::CUDA && settings.useSparseSolve && !settings.useNormalEquations) {
                    ImGui::Combo(
                            "Solver", (int*)&settings.cudaSparseSolverType,
                            CUDA_SPARSE_SOLVER_TYPE_NAMES, IM_ARRAYSIZE(CUDA_SPARSE_SOLVER_TYPE_NAMES));
                }
                if (settings.backend == OLSBackend::CUDA && !settings.useSparseSolve) {
                    ImGui::Combo(
                            "Solver", (int*)&settings.cudaSolverType,
                            CUDA_SOLVER_TYPE_NAMES, IM_ARRAYSIZE(CUDA_SOLVER_TYPE_NAMES));
                }
                if (settings.backend != OLSBackend::CUDA || (settings.backend == OLSBackend::CUDA
                        && settings.useSparseSolve && settings.useNormalEquations)) {
                    if (settings.backend == OLSBackend::CPU && settings.useSparseSolve && !settings.useNormalEquations) {
                        ImGui::Combo(
                                "Solver", (int*)&settings.eigenSparseSolverType,
                                EIGEN_SPARSE_SOLVER_TYPE_NAMES, IM_ARRAYSIZE(EIGEN_SPARSE_SOLVER_TYPE_NAMES));
                    } else {
                        ImGui::Combo(
                                "Solver", (int*)&settings.eigenSolverType,
                                EIGEN_SOLVER_TYPE_NAMES, IM_ARRAYSIZE(EIGEN_SOLVER_TYPE_NAMES));
                    }
                }
                ImGui::Combo(
                        "Float Accuracy", (int*)&settings.floatAccuracy,
                        FLOAT_ACCURACY_NAMES, IM_ARRAYSIZE(FLOAT_ACCURACY_NAMES));
                if (settings.backend != OLSBackend::VULKAN) {
                    ImGui::Checkbox("Use Normal Equations", &settings.useNormalEquations);
                }
                bool supportsRelaxation = false;
                if (settings.backend == OLSBackend::CPU) {
                    supportsRelaxation = settings.useNormalEquations;
                } else if (settings.backend == OLSBackend::CUDA) {
                    supportsRelaxation = !settings.useSparseSolve || (settings.useSparseSolve
                            && settings.cudaSparseSolverType == CudaSparseSolverType::CGLS);
                } else if (settings.backend == OLSBackend::VULKAN) {
                    supportsRelaxation = true;
                }
                if (supportsRelaxation) {
                    ImGui::SliderFloat("Relaxation Lambda", &settings.relaxationLambda, 0.0f, 10.0f);
                }
            }
        }

        if (settings.optimizerMethod == TFOptimizerMethod::DIFF_DVR && ImGui::CollapsingHeader(
                "DVR Settings", nullptr, ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderIntPowerOfTwo(
                    "Image Width", (int*)&settings.imageWidth, 16, 4096);
            ImGui::SliderIntPowerOfTwo(
                    "Image Height", (int*)&settings.imageHeight, 16, 4096);
            ImGui::SliderIntPowerOfTwo(
                    "Batch Size", (int*)&settings.batchSize, 1, 64);
            ImGui::SliderFloat(
                    "Step Size", &settings.stepSize, 0.01f, 1.0f);
            ImGui::SliderFloat(
                    "Attenuation", &settings.attenuationCoefficient, 0.0f, 200.0f);
            ImGui::SliderFloat(
                    "Smoothing Factor", &settings.lambdaSmoothingPrior, 0.0f, 10.0f);
            ImGui::Checkbox("Adjoint Delayed", &settings.adjointDelayed);
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
            settings.tfSize = uint32_t(tfSizes.at(tfSizeIdx));
            worker->queueRequest(settings, volumeData);
        }
        if (ImGui::BeginPopupModal(
                "Optimization Progress", &isOptimizationProgressDialogOpen, ImGuiWindowFlags_AlwaysAutoResize)) {
            int maxNumEpochs = settings.optimizerMethod == TFOptimizerMethod::OLS ? 1 : settings.maxNumEpochs;
            float progress = worker->getProgress();
            ImGui::Text("Progress: Epoch %d of %d...", int(std::round(progress * float(maxNumEpochs))), maxNumEpochs);
            ImGui::ProgressSpinner(
                    "##progress-spinner-tfopt", -1.0f, -1.0f, 4.0f,
                    ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
            ImGui::SameLine();
            ImGui::ProgressBar(progress, ImVec2(300, 0));
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                worker->stop();
            }
            workerHasReply = worker->getReply(reply);
            if (workerHasReply) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::EndPopup();
    }

    if (workerHasReply && !reply.hasStopped) {
        std::vector<sgl::OpacityPoint> opacityPoints;
        std::vector<sgl::ColorPoint_sRGB> colorPoints;
        auto tfArrayOpt = worker->getTFArrayOpt();
        opacityPoints.reserve(tfArrayOpt.size());
        colorPoints.reserve(tfArrayOpt.size());
        for (size_t i = 0; i < tfArrayOpt.size(); i++) {
            float t = float(i) / float(tfArrayOpt.size() - 1);
            const glm::vec4& color = tfArrayOpt.at(i);
            opacityPoints.emplace_back(color.a, t);
            auto color16 = sgl::color16FromVec4(color);
            color16.setA(0xFFFFu);
            colorPoints.emplace_back(color16, t);
        }
        auto& tfWidget = volumeData->getMultiVarTransferFunctionWindow();
        int varIdx = settings.fieldIdxOpt;
        tfWidget.setTransferFunction(
                varIdx, opacityPoints, colorPoints, sgl::ColorSpace::COLOR_SPACE_SRGB);
        needsReRender = true;
    }
}



TFOptimizationWorker::TFOptimizationWorker(sgl::vk::Renderer* parentRenderer) : parentRenderer(parentRenderer) {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    renderer = new sgl::vk::Renderer(device, 100);
}

void TFOptimizationWorker::initialize() {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (device->getGraphicsQueue() == device->getComputeQueue()) {
        supportsAsyncCompute = false;
    }

    if (supportsAsyncCompute) {
        requesterThread = std::thread(&TFOptimizationWorker::mainLoop, this);
    }
#ifdef CUDA_ENABLED
    else if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
             && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        cudaInit(true, cudaContext);
    }
#endif
}

TFOptimizationWorker::~TFOptimizationWorker() {
#ifdef CUDA_ENABLED
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (!supportsAsyncCompute && device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        cudaRelease();
    }
#endif

    tfOptimizer = {};
    if (renderer) {
        delete renderer;
        renderer = nullptr;
    }
}

#ifdef CUDA_ENABLED
void TFOptimizationWorker::setCudaContext(CUcontext context) {
    cudaContext = context;
}
#endif

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
        if (!tfOptimizer || optimizerMethod != newSettings.optimizerMethod) {
            optimizerMethod = newSettings.optimizerMethod;
            if (optimizerMethod == TFOptimizerMethod::OLS) {
                tfOptimizer = std::make_shared<TFOptimizerOLS>(renderer, parentRenderer, supportsAsyncCompute);
            } else if (optimizerMethod == TFOptimizerMethod::GD) {
                tfOptimizer = std::make_shared<TFOptimizerGD>(renderer, parentRenderer, supportsAsyncCompute);
            } else if (optimizerMethod == TFOptimizerMethod::DIFF_DVR) {
                tfOptimizer = std::make_shared<TFOptimizerDiffDvr>(renderer, parentRenderer, supportsAsyncCompute);
            } else {
                sgl::Logfile::get()->throwError(
                        "Error in TFOptimizationWorker::queueRequest: Unsupported optimizer method");
            }
        }
        tfOptimizer->setSettings(newSettings);
        tfOptimizer->onRequestQueued(volumeData);
        hasRequest = true;
    }
    hasRequestConditionVariable.notify_all();
}

void TFOptimizationWorker::stop() {
    shallStop = true;
}

float TFOptimizationWorker::getProgress() const {
    return tfOptimizer->getProgress();
}

bool TFOptimizationWorker::getReply(TFOptimizationWorkerReply& reply) {
    if (!supportsAsyncCompute && hasRequest) {
        if (tfOptimizer->getCurrentIterationIndex() == 0) {
            shallStop = false;
        }
        tfOptimizer->runOptimization(shallStop, hasStopped);
        if (tfOptimizer->getCurrentIterationIndex() == tfOptimizer->getMaxNumIterations()) {
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

const std::vector<glm::vec4>& TFOptimizationWorker::getTFArrayOpt() const {
    return tfOptimizer->getTFArrayOpt();
}

void TFOptimizationWorker::mainLoop() {
#ifdef TRACY_ENABLE
    tracy::SetThreadName("TFOptimizationWorker");
#endif

#ifdef CUDA_ENABLED
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        cudaInit(false, cudaContext);
    }
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

            tfOptimizer->runOptimization(shallStop, hasStopped);

            std::lock_guard<std::mutex> replyLock(replyMutex);
            hasReply = true;
        }
    }

#ifdef CUDA_ENABLED
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        cudaRelease();
    }
#endif
}
