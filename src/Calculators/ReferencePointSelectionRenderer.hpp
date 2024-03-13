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

#ifndef CORRERENDER_REFERENCEPOINTSELECTIONRENDERER_HPP
#define CORRERENDER_REFERENCEPOINTSELECTIONRENDERER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "../Renderers/Renderer.hpp"

class ReferencePointSelectionRasterPass;
class ShadowCircleRasterPass;

class ReferencePointSelectionRenderer : public Renderer {
public:
    explicit ReferencePointSelectionRenderer(ViewManager* viewManager);
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_CUSTOM; }
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) override {}
    void setVolumeDataPtr(VolumeData* _volumeData, bool isNewData);
    void setReferencePosition(const glm::ivec3& referencePosition);
    void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) override;

protected:
    void renderViewImpl(uint32_t viewIdx) override;
    void addViewImpl(uint32_t viewIdx) override;
    void removeViewImpl(uint32_t viewIdx) override;
    void renderViewPostOpaqueImpl(uint32_t viewIdx) override;
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    VolumeData* volumeData;
    std::vector<std::shared_ptr<ReferencePointSelectionRasterPass>> referencePointSelectionRasterPasses;
    std::vector<std::shared_ptr<ShadowCircleRasterPass>> shadowCircleRasterPasses;

    glm::ivec3 referencePosition{};

    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;
};

class ReferencePointSelectionRasterPass : public sgl::vk::RasterPass {
public:
    explicit ReferencePointSelectionRasterPass(sgl::vk::Renderer* renderer, SceneData* camera);

    // Public interface.
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void setRenderData(
            const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
            const sgl::vk::BufferPtr& _vertexNormalBuffer);
    void setReferencePosition(const glm::ivec3& referencePosition);
    inline void setSphereRadius(float pointWidth) { uniformData.sphereRadius = pointWidth; }
    inline void setSphereColor(const glm::vec4& color) { uniformData.sphereColor = color; }
    void recreateSwapchain(uint32_t width, uint32_t height) override;

protected:
    void loadShader() override;
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override;
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override;
    void _render() override;

private:
    SceneData* sceneData;
    sgl::CameraPtr* camera;
    VolumeData* volumeData = nullptr;

    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;

    glm::ivec3 referencePosition{};

    struct UniformData {
        glm::vec3 cameraPosition;
        float padding0;
        glm::vec3 spherePosition;
        float sphereRadius;
        glm::vec4 sphereColor;
        glm::vec3 backgroundColor;
        float padding1;
        glm::vec3 foregroundColor;
        float padding2;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformDataBuffer;
};

void getSphereSurfaceRenderData(
        const glm::vec3& center, float radius, int sectorCount, int stackCount,
        std::vector<glm::vec3>& vertexPositions, std::vector<glm::vec3>& vertexNormals,
        std::vector<uint32_t>& triangleIndices);

#endif //CORRERENDER_REFERENCEPOINTSELECTIONRENDERER_HPP
