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

#ifndef CORRERENDER_ISOSURFACERASTERIZER_HPP
#define CORRERENDER_ISOSURFACERASTERIZER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "Renderer.hpp"
#include "IsoSurfaces.hpp"

class IsoSurfaceRasterPass;

class IsoSurfaceRasterizer : public Renderer {
public:
    explicit IsoSurfaceRasterizer(ViewManager* viewManager);
    ~IsoSurfaceRasterizer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_ISOSURFACE_RASTERIZER; }
    [[nodiscard]] bool getIsOpaqueRenderer() const override { return isoSurfaceColor.a > 0.9999f; }
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;
    void reloadShaders() override;

protected:
    void renderViewImpl(uint32_t viewIdx) override;
    void addViewImpl(uint32_t viewIdx) override;
    void removeViewImpl(uint32_t viewIdx) override;
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    VolumeDataPtr volumeData;
    std::string exportFilePath;
    std::vector<std::shared_ptr<IsoSurfaceRasterPass>> isoSurfaceRasterPasses;

    void createIsoSurfaceData(
            std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
            std::vector<glm::vec3>& vertexNormals);
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;

    // UI renderer settings.
    int selectedFieldIdx = 0, oldSelectedFieldIdx = 0;
    std::string selectedScalarFieldName;
    std::pair<float, float> minMaxScalarFieldValue;
    float isoValue = 0.5f;
    float gammaSnapMC = 0.3f;
    glm::vec4 isoSurfaceColor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
    IsoSurfaceExtractionTechnique isoSurfaceExtractionTechnique = IsoSurfaceExtractionTechnique::SNAP_MC;
};

/**
 * Iso surface ray casting pass.
 */
class IsoSurfaceRasterPass : public sgl::vk::RasterPass {
public:
    explicit IsoSurfaceRasterPass(sgl::vk::Renderer* renderer, SceneData* camera);

    // Public interface.
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setRenderData(
            const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
            const sgl::vk::BufferPtr& _vertexNormalBuffer);
    void setRenderData(
            const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
            const sgl::vk::BufferPtr& _vertexNormalBuffer, const sgl::vk::BufferPtr& _vertexColorBuffer);
    inline void setIsoSurfaceColor(const glm::vec4& _color) { renderSettingsData.isoSurfaceColor = _color; }
    void recreateSwapchain(uint32_t width, uint32_t height) override;

protected:
    void loadShader() override;
    void setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) override;
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override;
    void _render() override;

private:
    SceneData* sceneData;
    sgl::CameraPtr* camera;
    VolumeDataPtr volumeData;

    // Renderer settings.
    struct RenderSettingsData {
        glm::vec3 cameraPosition;
        float padding;
        glm::vec4 isoSurfaceColor;
    };
    RenderSettingsData renderSettingsData{};
    sgl::vk::BufferPtr rendererUniformDataBuffer;

    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;
    sgl::vk::BufferPtr vertexColorBuffer; ///< Optional; if not present, last entry of uniform buffer is used.
};

#endif //CORRERENDER_ISOSURFACERASTERIZER_HPP
