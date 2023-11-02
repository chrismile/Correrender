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

#ifndef CORRERENDER_WORLDMAPRENDERER_HPP
#define CORRERENDER_WORLDMAPRENDERER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "Renderer.hpp"

class ShapefileRasterizer;
class WorldMapRasterPass;

enum class WorldMapSource {
    TIFF_FILE, SHAPEFILE_RASTERIZER
};
enum class WorldMapQuality {
    LOW, MEDIUM, HIGH
};

class WorldMapRenderer : public Renderer {
public:
    explicit WorldMapRenderer(ViewManager* viewManager);
    ~WorldMapRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_WORLD_MAP_RENDERER; }
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
    std::string worldMapFilePath;
    std::vector<std::shared_ptr<WorldMapRasterPass>> worldMapRasterPasses;

    void ensureWorldMapFileExistsTiff();
    void createGeometryData(
            std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
            std::vector<glm::vec3>& vertexNormals, std::vector<glm::vec2>& vertexTexCoords);
    void createWorldMapTexture();
    void createWorldMapTextureTiff();
    void createWorldMapTextureShapefile();
    WorldMapSource worldMapSource = WorldMapSource::TIFF_FILE;
    WorldMapQuality worldMapQuality = WorldMapQuality::MEDIUM;
    std::shared_ptr<ShapefileRasterizer> shapefileRasterizer;
    bool hasCheckedWorldMapExists = false;
    bool manuallySetRasterizer = false;
    float minNormX = 0.0f, minNormY = 0.0f, maxNormX = 0.0f, maxNormY = 0.0;
    uint32_t regionImageWidth = 0, regionImageHeight = 0;
    uint32_t xl = 0, yl = 0, xu = 0, yu = 0;
    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;
    sgl::vk::BufferPtr vertexTexCoordBuffer;
    sgl::vk::TexturePtr worldMapTexture;

    // UI renderer settings.
    float lightingFactor = 0.5f;
};

/**
 * Iso surface ray casting pass.
 */
class WorldMapRasterPass : public sgl::vk::RasterPass {
public:
    explicit WorldMapRasterPass(sgl::vk::Renderer* renderer, SceneData* camera);

    // Public interface.
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setRenderData(
            const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
            const sgl::vk::BufferPtr& _vertexNormalBuffer, const sgl::vk::BufferPtr& _vertexTexCoordBuffer,
            const sgl::vk::TexturePtr& _worldMapTexture);
    inline void setLightingFactor(float factor) { renderSettingsData.lightingFactor = factor; }
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
        uint32_t padding0 = 0;
        glm::vec3 minBoundingBox;
        float lightingFactor = 0.5f;
        glm::vec3 maxBoundingBox;
        float padding1 = 0.0f;
    };
    RenderSettingsData renderSettingsData{};
    sgl::vk::BufferPtr rendererUniformDataBuffer;

    sgl::vk::BufferPtr indexBuffer;
    sgl::vk::BufferPtr vertexPositionBuffer;
    sgl::vk::BufferPtr vertexNormalBuffer;
    sgl::vk::BufferPtr vertexTexCoordBuffer;
    sgl::vk::TexturePtr worldMapTexture;
};

#endif //CORRERENDER_WORLDMAPRENDERER_HPP
