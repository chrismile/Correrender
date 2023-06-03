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
        2, 3, 4, 5, 7, 8, 10, 16, 32, 64, 96, 128, 256
};

class TFOptimization {
public:
    explicit TFOptimization(sgl::vk::Renderer* parentRenderer);
    void initialize();
    ~TFOptimization();
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    //void onFieldRemoved(int fieldIdx);
    void openDialog();
    void renderGuiDialog();

    inline bool getNeedsReRender() { bool tmp = needsReRender; needsReRender = false; return tmp; }

#ifdef CUDA_ENABLED
    void setCudaContext(CUcontext context);
#endif

private:
    sgl::vk::Renderer* parentRenderer = nullptr;
    VolumeData* volumeData = nullptr;
    TFOptimizationWorker* worker = nullptr;

    bool needsReRender = false;
    bool isOptimizationSettingsDialogOpen = false;
    bool isOptimizationProgressDialogOpen = false;

    // Settings.
    int tfSizeIdx = 0;
    std::vector<int> tfSizes;
    std::vector<std::string> tfSizeStrings;
    TFOptimizationWorkerSettings settings;
};

struct TFOptimizationWorkerReply {
    bool hasStopped = false;
};

class TFOptimizer;

class TFOptimizationWorker {
public:
    explicit TFOptimizationWorker(sgl::vk::Renderer* parentRenderer);
    void initialize();
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
    [[nodiscard]] float getProgress() const;
    /**
     * Checks if a reply was received to a request.
     * @return Whether a reply was received.
     */
    bool getReply(TFOptimizationWorkerReply& reply);
    /// Returns the result buffer belonging to the last reply.
    [[nodiscard]] const std::vector<glm::vec4>& getTFArrayOpt() const;

#ifdef CUDA_ENABLED
    void setCudaContext(CUcontext context);
#endif

protected:
    void mainLoop();

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
    TFOptimizerMethod optimizerMethod = TFOptimizerMethod::OLS;
    std::shared_ptr<TFOptimizer> tfOptimizer;

    sgl::vk::Renderer* parentRenderer = nullptr;
    sgl::vk::Renderer* renderer = nullptr;

#ifdef CUDA_ENABLED
    CUcontext cudaContext{};
#endif
};

#endif //CORRERENDER_TFOPTIMIZATION_HPP
