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

#ifndef CORRERENDER_SHADOWRASTERPASS_HPP
#define CORRERENDER_SHADOWRASTERPASS_HPP

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

struct SceneData;

class ShadowCircleRasterPass : public sgl::vk::RasterPass {
public:
    explicit ShadowCircleRasterPass(sgl::vk::Renderer* renderer, SceneData* camera);

    // Public interface.
    void setCenter(const glm::vec3& center);
    void setRadius(float radius);
    inline void setShadowColor(const glm::vec4& color) { uniformData.shadowColor = color; }
    void recreateSwapchain(uint32_t width, uint32_t height) override;

protected:
    void loadShader() override;
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override;
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override;
    void _render() override;

private:
    SceneData* sceneData;
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexBuffer;

    struct UniformData {
        glm::vec3 center{};
        float radius{};
        glm::vec4 shadowColor = glm::vec4(0.1f, 0.1f, 0.1f, 0.5f);
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformDataBuffer;
};

#endif //CORRERENDER_SHADOWRASTERPASS_HPP
