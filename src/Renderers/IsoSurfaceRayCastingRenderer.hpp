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

#ifndef CORRERENDER_ISOSURFACERAYCASTINGRENDERER_HPP
#define CORRERENDER_ISOSURFACERAYCASTINGRENDERER_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include "Renderer.hpp"

class IsoSurfaceRayCastingPass;

/// For an overview of different solvers, see: https://www.sci.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf
enum class IntersectionSolver {
    LINEAR_INTERPOLATION, NEUBAUER, MARMITT, SCHWARZE
};
const char* const INTERSECTION_SOLVER_NAMES[] = {
        "Linear Interpolation", "Neubauer", "Marmitt", "Schwarze"
};

class IsoSurfaceRayCastingRenderer : public Renderer {
public:
    explicit IsoSurfaceRayCastingRenderer(ViewManager* viewManager);
    ~IsoSurfaceRayCastingRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_ISOSURFACE_RAYCASTER; }
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
    std::vector<std::shared_ptr<IsoSurfaceRayCastingPass>> isoSurfaceRayCastingPasses;

    // UI renderer settings.
    int selectedFieldIdx = 0, oldSelectedFieldIdx = 0;
    std::string selectedScalarFieldName;
    std::pair<float, float> minMaxScalarFieldValue;
    float isoValue = 0.5f;
    bool analyticIntersections = false;
    float stepSize = 0.1f;
    glm::vec4 isoSurfaceColor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
    IntersectionSolver intersectionSolver = IntersectionSolver::MARMITT;
    bool closeIsoSurface = false; //< Only for ray marching at the moment.
};

/**
 * Iso surface ray casting pass.
 */
class IsoSurfaceRayCastingPass : public sgl::vk::ComputePass {
public:
    explicit IsoSurfaceRayCastingPass(sgl::vk::Renderer* renderer, SceneData* camera);

    // Public interface.
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setSelectedScalarFieldName(const std::string& _scalarFieldName);
    inline void setIsoValue(float _isoValue) { renderSettingsData.isoValue = _isoValue; }
    inline void setAnalyticIntersections(bool _analytic) { analyticIntersections = _analytic; setShaderDirty(); }
    inline void setStepSize(float _stepSize) { stepSize = _stepSize; }
    inline void setIsoSurfaceColor(const glm::vec4& _color) { renderSettingsData.isoSurfaceColor = _color; }
    inline void setIntersectionSolver(IntersectionSolver _solver) { intersectionSolver = _solver; setShaderDirty(); }
    inline void setCloseIsoSurface(bool _closeIsoSurface) { closeIsoSurface = _closeIsoSurface; setShaderDirty(); }
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
    bool useInterpolationNearestCached = false;

    // Renderer settings.
    std::string selectedScalarFieldName;
    bool analyticIntersections = false;
    float stepSize = 0.1f;
    IntersectionSolver intersectionSolver = IntersectionSolver::MARMITT;
    bool closeIsoSurface = false; //< Only for ray marching at the moment.

    struct RenderSettingsData {
        glm::mat4 inverseViewMatrix;
        glm::mat4 inverseProjectionMatrix;
        glm::vec3 cameraPosition;
        float dx;
        float dy;
        float dz;
        float zNear;
        float zFar;
        glm::vec3 minBoundingBox;
        float isoValue;
        glm::vec3 maxBoundingBox;
        float stepSize;
        glm::vec4 isoSurfaceColor;
        glm::vec3 voxelTexelSize;
    };
    RenderSettingsData renderSettingsData{};
    sgl::vk::BufferPtr rendererUniformDataBuffer;
};

#endif //CORRERENDER_ISOSURFACERAYCASTINGRENDERER_HPP
