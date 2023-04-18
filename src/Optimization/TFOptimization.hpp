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
#include <thread>
#include <mutex>
#include <condition_variable>

#include "OptDefines.hpp"

namespace sgl { namespace vk {
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
class Renderer;
}}

class VolumeData;
class TFOptimizationWorker;

const int possibleTfSizes[] = {
        3, 4, 5, 7, 8, 10, 16, 32, 64, 96, 128, 256
};

class TFOptimization {
public:
    explicit TFOptimization(sgl::vk::Renderer* parentRenderer);
    ~TFOptimization();
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void onFieldRemoved(int fieldIdx);
    void openDialog();
    void renderGuiDialog();

private:
    sgl::vk::Renderer* parentRenderer = nullptr;
    VolumeData* volumeData = nullptr;
    TFOptimizationWorker* worker = nullptr;

    void startOptimization();
    bool isOptimizationSettingsDialogOpen = false;
    bool isOptimizationProgressDialogOpen = false;

    // Settings.
    int fieldIdxGT = 0, fieldIdxOpt = 0;
    int tfSizeIdx = 0;
    std::vector<int> tfSizes;
    std::vector<std::string> tfSizeStrings;
    OptimizerType optimizerType = OptimizerType::ADAM;
    int maxNumEpochs = 200;
    // SGD & Adam.
    float learningRate = 0.4f;
    // Adam.
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

struct TFOptimizationWorkerSettings {
    int fieldIdxGT = -1;
    int fieldIdxOpt = -1;
    uint32_t tfSize = 0;
    OptimizerType optimizerType = OptimizerType::ADAM;
    int maxNumEpochs = 200;

    // DVR.
    float stepSize = 0.2f;
    float attenuationCoefficient = 100.0f;

    // SGD & Adam.
    float learningRate = 0.4f;

    // Adam.
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

struct TFOptimizationWorkerReply {
    bool hasStopped = false;
};

class ForwardPass;
class ForwardPass;
class LossPass;
class AdjointPass;
class SmoothingPriorLossPass;
class OptimizerPass;

class TFOptimizationWorker {
public:
    TFOptimizationWorker(sgl::vk::Renderer* parentRenderer);
    ~TFOptimizationWorker();

    /**
     * Starts a new request.
     * @param newSettings The settings of the new request.
     */
    void queueRequest(const TFOptimizationWorkerSettings& newSettings, VolumeData* volumeData);
    /// Queues a request for stopping the current computation.
    void stop();
    /// Waits for the requester thread to terminate.
    void join();
    /// Returns the progress of the current request.
    float getProgress() const;
    /**
     * Checks if a reply was received to a request.
     * @return Whether a reply was received.
     */
    bool getReply(TFOptimizationWorkerReply& reply);
    /// Returns the result buffer belonging to the last reply.
    const sgl::vk::BufferPtr& getTFBuffer();

private:
    void mainLoop();
    void recreateCache(
            VkFormat formatGT, VkFormat formatOpt, uint32_t xs, uint32_t ys, uint32_t zs);
    void runEpochs();
    void runEpoch();
    void sampleCameraPoses();

    // Multithreading.
    bool supportsAsyncCompute = true;
    std::thread requesterThread;
    std::condition_variable hasRequestConditionVariable;
    std::condition_variable hasReplyConditionVariable;
    std::mutex requestMutex;
    std::mutex replyMutex;
    bool programIsFinished = false;
    bool hasRequest = false;
    bool hasReply = false;
    bool shallStop = false, hasStopped = false;
    TFOptimizationWorkerReply reply;

    // Cached data.
    TFOptimizationWorkerSettings settings;

    sgl::vk::Renderer* parentRenderer = nullptr;
    sgl::vk::Renderer* renderer = nullptr;
    sgl::vk::FencePtr fence{};
    VkCommandPool commandPool{};
    VkCommandBuffer commandBuffer{};

    uint32_t batchSize = 8;
    uint32_t viewportWidth = 512;
    uint32_t viewportHeight = 512;
    int currentEpoch = 0;
    int maxNumEpochs = 0;

    struct DvrSettingsBufferTf {
        glm::mat4 inverseProjectionMatrix;
        glm::vec3 minBoundingBox;
        float attenuationCoefficient;
        glm::vec3 maxBoundingBox;
        float stepSize;
    };

    uint32_t cachedBatchSize = 0;
    uint32_t cachedViewportWidth = 0;
    uint32_t cachedViewportHeight = 0;
    uint32_t cachedTfSize = 0;
    VkFormat cachedFormatGT{}, cachedFormatOpt{};
    DvrSettingsBufferTf dvrSettings{};
    std::vector<glm::mat4> batchSettingsArray;
    sgl::vk::BufferPtr batchSettingsBuffer;
    sgl::vk::BufferPtr dvrSettingsBuffer;
    sgl::vk::BufferPtr gtFinalColorsBuffer;
    sgl::vk::BufferPtr finalColorsBuffer;
    sgl::vk::BufferPtr terminationIndexBuffer;
    sgl::vk::BufferPtr transferFunctionBuffer;
    sgl::vk::BufferPtr transferFunctionGradientBuffer;
    sgl::vk::ImageViewPtr imageViewFieldGT, imageViewFieldOpt;

    // For Adam.
    sgl::vk::BufferPtr firstMomentEstimateBuffer;
    sgl::vk::BufferPtr secondMomentEstimateBuffer;

    // Compute passes.
    std::shared_ptr<ForwardPass> gtForwardPass;
    std::shared_ptr<ForwardPass> forwardPass;
    std::shared_ptr<LossPass> lossPass;
    std::shared_ptr<AdjointPass> adjointPass;
    std::shared_ptr<SmoothingPriorLossPass> smoothingPriorLossPass;
    std::shared_ptr<OptimizerPass> optimizerPass;
};

#endif //CORRERENDER_TFOPTIMIZATION_HPP
