/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#ifndef CORRERENDER_DOMAINOUTLINERENDERER_HPP
#define CORRERENDER_DOMAINOUTLINERENDERER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "Renderer.hpp"

class DomainOutlineRasterPass;

class DomainOutlineRenderer : public Renderer {
public:
    explicit DomainOutlineRenderer(ViewManager* viewManager);
    ~DomainOutlineRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_DOMAIN_OUTLINE_RENDERER; }
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) override;
    void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void renderViewImpl(uint32_t viewIdx) override;
    void addViewImpl(uint32_t viewIdx) override;
    void removeViewImpl(uint32_t viewIdx) override;
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    VolumeDataPtr volumeData;
    std::vector<std::shared_ptr<DomainOutlineRasterPass>> domainOutlineRasterPasses;

    // UI renderer settings.
    float lineWidth = 0.001f;
    bool useDepthCues = true;

    void recreateBuffers();
    std::vector<uint32_t> triangleIndices;
    std::vector<glm::vec3> vertexPositions;
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
};

class DomainOutlineRasterPass : public sgl::vk::RasterPass {
public:
    explicit DomainOutlineRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData);

    // Public interface.
    void setRenderData(
            const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer);
    void setCustomColor(const glm::vec4& color);
    void resetCustomColor();
    void setUseDepthCues(bool _useDepthCues);
    void setAabb(const sgl::AABB3& aabb); //< Only needed for depth cues.
    void recreateSwapchain(uint32_t width, uint32_t height) override;

protected:
    void loadShader() override;
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override;
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override;
    void _render() override;

private:
    SceneData* sceneData;
    sgl::CameraPtr* camera;

    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    bool useCustomColor = false;
    bool useDepthCues = false;
    glm::vec3 aabbMin{};
    glm::vec3 aabbMax{};

    struct UniformData {
        glm::vec4 objectColor{};
        float minDepth{}, maxDepth{};
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformDataBuffer;
};

struct DomainOutlinePushConstants {
    glm::vec3 aabbMin{};
    float lineWidth{};
    glm::vec3 aabbMax{};
    float offset{};
};

class DomainOutlineComputePass : public sgl::vk::ComputePass {
public:
    explicit DomainOutlineComputePass(sgl::vk::Renderer* renderer);

    // Public interface.
    void setRenderData(
            const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer);
    void setOutlineSettings(const sgl::AABB3& aabb, float lineWidth, float offset);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    DomainOutlinePushConstants pushConstants;
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
};

#endif //CORRERENDER_DOMAINOUTLINERENDERER_HPP
