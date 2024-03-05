/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#ifndef CORRERENDER_COPYIMAGETOBUFFER_HPP
#define CORRERENDER_COPYIMAGETOBUFFER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include <glm/vec3.hpp>

class ImageToBufferCopyPass : public sgl::vk::ComputePass {
public:
    explicit ImageToBufferCopyPass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setData(
            uint32_t xs, uint32_t ys, uint32_t zs,
            const sgl::vk::ImageViewPtr& _inputImage, const sgl::vk::BufferPtr& _outputBuffer,
            const glm::uvec3& _tileSize);
    void resetData();

private:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    glm::uvec3 tileSize;
    std::string imageFormatString;
    struct UniformData {
        uint32_t xs = 0, ys = 0, zs = 0;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;
    sgl::vk::ImageViewPtr inputImage;
    sgl::vk::BufferPtr outputBuffer;
};

#endif //CORRERENDER_COPYIMAGETOBUFFER_HPP
