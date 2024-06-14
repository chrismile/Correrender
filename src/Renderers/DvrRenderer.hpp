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

#ifndef CORRERENDER_DVRRENDERER_HPP
#define CORRERENDER_DVRRENDERER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "Renderer.hpp"

class DvrPass;

class DvrRenderer : public Renderer {
public:
    explicit DvrRenderer(ViewManager* viewManager);
    ~DvrRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_DIRECT_VOLUME_RENDERING; }
    [[nodiscard]] bool getIsOpaqueRenderer() const override { return false; }
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
    std::vector<std::shared_ptr<DvrPass>> dvrPasses;

    void createTetMeshData(
            std::vector<uint32_t>& cellIndices, std::vector<glm::vec3>& vertexPositions,
            std::vector<glm::vec4>& vertexColors);

    // UI renderer settings.
    int selectedFieldIdx = 0, oldSelectedFieldIdx = 0;
    std::string selectedScalarFieldName;
    float stepSize = 0.1f;
    float attenuationCoefficient = 100.0f;
    NaNHandling nanHandling = NaNHandling::IGNORE;
};

/**
 * Direct volume rendering (DVR) pass.
 */
class DvrPass : public sgl::vk::ComputePass {
public:
    explicit DvrPass(sgl::vk::Renderer* renderer, SceneData* camera);

    // Public interface.
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setSelectedScalarField(int selectedFieldIdx, const std::string& _scalarFieldName);
    inline void setStepSize(float _stepSize) { stepSize = _stepSize; }
    inline void setAttenuationCoefficient(float _coeff) { renderSettingsData.attenuationCoefficient = _coeff; }
    inline void setNaNHandling(NaNHandling _nanHandling) { nanHandling = _nanHandling; shaderDirty = true; }
    void recreateSwapchain(uint32_t width, uint32_t height) override;

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    SceneData* sceneData;
    sgl::CameraPtr* camera;
    VolumeDataPtr volumeData;
    sgl::vk::ImageViewPtr sceneImageView;
    float voxelSize = 1.0f;
    NaNHandling nanHandling = NaNHandling::IGNORE;
    bool isRgba = false;

    // Renderer settings.
    int selectedFieldIdx = 0;
    std::string selectedScalarFieldName;
    float stepSize = 0.1f;

    struct RenderSettingsData {
        glm::mat4 inverseViewMatrix;
        glm::mat4 inverseProjectionMatrix;
        float zNear;
        float zFar;
        uint32_t fieldIndex = 0;
        float padding1;
        glm::vec3 minBoundingBox;
        float attenuationCoefficient = 100.0f;
        glm::vec3 maxBoundingBox;
        float stepSize;
    };
    RenderSettingsData renderSettingsData{};
    sgl::vk::BufferPtr rendererUniformDataBuffer;
};

#endif //CORRERENDER_DVRRENDERER_HPP
