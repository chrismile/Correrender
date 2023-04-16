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

#ifndef CORRERENDER_TFOPTIMIZATION_HPP
#define CORRERENDER_TFOPTIMIZATION_HPP

#include <vector>

namespace sgl { namespace vk {
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
class Renderer;
}}

class VolumeData;
class TFOptimizationWorker;

enum class OptimizerType {
    SGD, ADAM
};
const char* const OPTIMIZER_TYPE_NAMES[] = {
        "SGD", "Adam"
};

enum class LossType {
    L1, L2
};
const char* const LOSS_TYPE_NAMES[] = {
        "L1", "L2"
};

const int possibleTfSizes[] = {
        3, 4, 5, 7, 8, 10, 16, 32, 64, 96, 128, 256
};

class TFOptimization {
public:
    TFOptimization();
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void onFieldRemoved(int fieldIdx);
    void openDialog();
    void renderGuiDialog();

private:
    sgl::vk::Renderer* renderer = nullptr;
    VolumeData* volumeData = nullptr;
    TFOptimizationWorker* worker = nullptr;

    void startOptimization();
    bool isOptimizationSettingsDialogOpen = false;
    bool isOptimizationProgressDialogOpen = false;
    float time = 0.0f;

    // Settings.
    int fieldGTIdx = 0, fieldOptIdx = 0;
    int tfSizeIdx = 0;
    std::vector<int> tfSizes;
    std::vector<std::string> tfSizeStrings;
    OptimizerType optimizerType = OptimizerType::ADAM;
    int maxNumEpochs = 200;
    // SGD & Adam.
    float learningRate = 0.8f;
    // Adam.
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

struct TFOptimizationWorkerSettings {
    int fieldIdxGT = -1;
    int fieldIdxOpt = -1;
    OptimizerType optimizerType = OptimizerType::ADAM;
    int maxNumEpochs = 200;

    // SGD & Adam.
    float learningRate = 0.8f;

    // Adam.
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

class TFOptimizationWorker {
public:
    void start(const TFOptimizationWorkerSettings& newSettings);
    void stop();
    float getProgress();
    const sgl::vk::BufferPtr& getTFBuffer();
    bool getIsResultAvailable();

private:
    void runEpochs();
    void runEpoch();

    TFOptimizationWorkerSettings settings;

    sgl::vk::BufferPtr gtFinalColorsBuffer;

    sgl::vk::BufferPtr batchSettingsBuffer;
    sgl::vk::BufferPtr dvrSettingsBuffer;
    sgl::vk::BufferPtr finalColorsBuffer;
    sgl::vk::BufferPtr terminationIndexBuffer;
    sgl::vk::BufferPtr transferFunctionBuffer;
    sgl::vk::BufferPtr transferFunctionGradientBuffer;

    // For Adam.
    sgl::vk::BufferPtr firstMomentEstimateBuffer;
    sgl::vk::BufferPtr secondMomentEstimateBuffer;
};

#endif //CORRERENDER_TFOPTIMIZATION_HPP
