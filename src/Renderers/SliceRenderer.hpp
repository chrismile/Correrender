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

#ifndef CORRERENDER_SLICERENDERER_HPP
#define CORRERENDER_SLICERENDERER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "Renderer.hpp"

class SliceRasterPass;

class SliceRenderer : public Renderer {
public:
    explicit SliceRenderer(ViewManager* viewManager);
    ~SliceRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_SLICE_RENDERER; }
    [[nodiscard]] bool getIsOpaqueRenderer() const override { return true; }
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
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
    std::vector<std::shared_ptr<SliceRasterPass>> sliceRasterPasses;

    void createGeometryData(
            std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
            std::vector<glm::vec3>& vertexNormals);
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;

    // For scaling the transfer function to the visible range of values.
    void scaleTfToVisible();
    void computeVisibleRangeView(uint32_t viewIdx, float& minVal, float& maxVal);
    bool alwaysScaleTfToVisible = false;

    // UI renderer settings.
    int selectedFieldIdx = 0, oldSelectedFieldIdx = 0;
    std::string selectedScalarFieldName;
    glm::vec3 planeNormalUi = glm::vec3(0.0f, 0.0f, 1.0f);
    glm::vec3 planeNormal = glm::vec3(0.0f, 0.0f, 1.0f);
    float planeDist = 0.0f;
    float lightingFactor = 0.5f;
    bool fixOnGround = false; ///< Always show the plane on the ground, even if the plane distance is above?
    NaNHandling nanHandling = NaNHandling::IGNORE;
};

/**
 * Iso surface ray casting pass.
 */
class SliceRasterPass : public sgl::vk::RasterPass {
public:
    explicit SliceRasterPass(sgl::vk::Renderer* renderer, SceneData* camera);

    // Public interface.
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setSelectedScalarField(int selectedFieldIdx, const std::string& _scalarFieldName);
    void setRenderData(
            const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
            const sgl::vk::BufferPtr& _vertexNormalBuffer);
    inline void setNaNHandling(NaNHandling _nanHandling) { nanHandling = _nanHandling; shaderDirty = true; }
    inline void setLightingFactor(float factor) { renderSettingsData.lightingFactor = factor; }
    inline void setFixOnGround(bool fixOnGround) { renderSettingsData.fixOnGround = fixOnGround ? 1 : 0; }
    void recreateSwapchain(uint32_t width, uint32_t height) override;
    [[nodiscard]] inline sgl::CameraPtr& getCamera() { return *camera; }

protected:
    void loadShader() override;
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override;
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override;
    void _render() override;

private:
    SceneData* sceneData;
    sgl::CameraPtr* camera;
    VolumeDataPtr volumeData;
    NaNHandling nanHandling = NaNHandling::IGNORE;

    // Renderer settings.
    int selectedFieldIdx = 0;
    std::string selectedScalarFieldName;

    struct RenderSettingsData {
        glm::vec3 cameraPosition;
        uint32_t fieldIndex = 0;
        glm::vec3 minBoundingBox;
        float lightingFactor = 0.5f;
        glm::vec3 maxBoundingBox;
        uint32_t fixOnGround = 0;
    };
    RenderSettingsData renderSettingsData{};
    sgl::vk::BufferPtr rendererUniformDataBuffer;

    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;
};

#endif //CORRERENDER_SLICERENDERER_HPP
