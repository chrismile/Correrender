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

#ifndef CORRERENDER_CONNECTINGLINERASTERPASS_HPP
#define CORRERENDER_CONNECTINGLINERASTERPASS_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "Renderers/SceneData.hpp"

class ConnectingLineRasterPass : public sgl::vk::RasterPass {
public:
    explicit ConnectingLineRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData);

    // Public interface.
    void setLineSettings(const std::pair<glm::vec3, glm::vec3>& points, float lineWidth);
    void setCustomColors(const glm::vec4& c0, const glm::vec4& c1);
    void resetCustomColors();
    void recreateSwapchain(uint32_t width, uint32_t height) override;

protected:
    void loadShader() override;
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override;
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override;
    void _render() override;

private:
    SceneData* sceneData;
    sgl::CameraPtr* camera;

    const int tubeNumSubdivisions = 16;
    bool useCustomColors = false;

    sgl::vk::BufferPtr indexBuffer;

    struct UniformData {
        glm::vec4 c0{};
        glm::vec4 c1{};
        glm::vec3 p0{};
        float lineWidth = 0.001f;
        glm::vec3 p1{};
        float padding{};
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformDataBuffer;
};

#endif //CORRERENDER_CONNECTINGLINERASTERPASS_HPP
