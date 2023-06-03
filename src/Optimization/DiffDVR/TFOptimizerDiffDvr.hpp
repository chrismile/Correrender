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

#ifndef CORRERENDER_TFOPTIMIZERDIFFDVR_HPP
#define CORRERENDER_TFOPTIMIZERDIFFDVR_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>

#include "../TFOptimizer.hpp"

class DvrForwardPass;
class DvrAdjointPass;
class LossPass;
class SmoothingPriorLossPass;
class OptimizerPass;

class TFOptimizerDiffDvr : public TFOptimizer {
public:
    TFOptimizerDiffDvr(sgl::vk::Renderer* renderer, sgl::vk::Renderer* parentRenderer, bool supportsAsyncCompute);
    ~TFOptimizerDiffDvr() override;
    void onRequestQueued(VolumeData* volumeData) override;
    void runOptimization(bool& shallStop, bool& hasStopped) override;
    float getProgress() override;
    int getCurrentIterationIndex() override { return currentEpoch; }
    int getMaxNumIterations() override { return maxNumEpochs; }

private:
    void runEpoch();
    void sampleCameraPoses();

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
        uint32_t imageWidth, imageHeight, batchSize;
    };

    uint32_t cachedBatchSize = 0;
    uint32_t cachedViewportWidth = 0;
    uint32_t cachedViewportHeight = 0;
    uint32_t cachedTfSize = 0;
    DvrSettingsBufferTf dvrSettings{};
    std::vector<glm::mat4> batchSettingsArray;
    sgl::vk::BufferPtr batchSettingsBuffer;
    sgl::vk::BufferPtr dvrSettingsBuffer;
    sgl::vk::BufferPtr finalColorsGTBuffer, finalColorsOptBuffer, adjointColorsBuffer;
    sgl::vk::BufferPtr terminationIndexBuffer;
    sgl::vk::BufferPtr tfGTBuffer, tfOptBuffer, tfDownloadStagingBuffer;
    sgl::vk::BufferPtr tfOptGradientBuffer;
    sgl::vk::ImageViewPtr imageViewFieldGT, imageViewFieldOpt;
    sgl::vk::TexturePtr textureFieldGT, textureFieldOpt;

    // For Adam.
    sgl::vk::BufferPtr firstMomentEstimateBuffer;
    sgl::vk::BufferPtr secondMomentEstimateBuffer;

    // Compute passes.
    std::shared_ptr<DvrForwardPass> forwardGTPass;
    std::shared_ptr<DvrForwardPass> forwardOptPass;
    std::shared_ptr<DvrAdjointPass> adjointPass;
    std::shared_ptr<LossPass> lossPass;
    std::shared_ptr<SmoothingPriorLossPass> smoothingPriorLossPass;
    std::shared_ptr<OptimizerPass> optimizerPass;
};


#endif //CORRERENDER_TFOPTIMIZERDIFFDVR_HPP
