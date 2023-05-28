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

#ifndef CORRERENDER_TFOPTIMIZER_HPP
#define CORRERENDER_TFOPTIMIZER_HPP

#include <memory>

#include "OptDefines.hpp"

namespace sgl { namespace vk {
class Fence;
typedef std::shared_ptr<Fence> FencePtr;
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
class Renderer;
}}

class VolumeData;

class TFOptimizer {
public:
    TFOptimizer(sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute);
    virtual ~TFOptimizer();
    [[nodiscard]] inline const std::vector<glm::vec4>& getTFArrayOpt() const { return tfArrayOpt; }
    inline void setSettings(const TFOptimizationWorkerSettings& newSettings) { settings = newSettings; }
    virtual void onRequestQueued(VolumeData* volumeData)=0;
    virtual void runOptimization(bool& shallStop, bool& hasStopped)=0;
    virtual float getProgress()=0;
    virtual int getCurrentIterationIndex() { return 0; }
    virtual int getMaxNumIterations() { return 0; }

protected:
    sgl::vk::Renderer* renderer;
    sgl::vk::Renderer* parentRenderer;
    bool supportsAsyncCompute = false;

    sgl::vk::FencePtr fence{};
    VkCommandPool commandPool{};
    VkCommandBuffer commandBuffer{};

    TFOptimizationWorkerSettings settings;
    std::vector<glm::vec4> tfArrayOpt;
    //sgl::vk::BufferPtr transferFunctionBuffer;
};

#endif //CORRERENDER_TFOPTIMIZER_HPP
