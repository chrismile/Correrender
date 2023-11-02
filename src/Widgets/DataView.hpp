/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Christoph Neuhauser
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

#ifndef CORRERENDER_DATAVIEW_HPP
#define CORRERENDER_DATAVIEW_HPP

#include <set>

#include <Utils/SciVis/Navigation/CameraNavigator.hpp>
#include <Graphics/Color.hpp>
#include <Graphics/Scene/Camera.hpp>
#include <Graphics/Vulkan/Utils/ScreenshotReadbackHelper.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Shader/Shader.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitRenderPass.hpp>
#include <ImGui/imgui.h>

#include "Renderers/SceneData.hpp"

namespace sgl { namespace vk {
class Renderer;
}}

class Renderer;
class DataView;
typedef std::shared_ptr<DataView> DataViewPtr;

class DataView {
public:
    explicit DataView(SceneData* parentSceneData);
    ~DataView();
    virtual void resize(int newWidth, int newHeight);
    virtual void beginRender();
    virtual void endRender();
    void syncCamera();
    void saveScreenshot(const std::string& filename);
    void saveScreenshotDataIfAvailable();
    [[nodiscard]] ImTextureID getImGuiTextureId() const;
    void setClearColor(const sgl::Color& color);
    [[nodiscard]] inline int getSupersamplingFactor() const { return supersamplingFactor; }

    [[nodiscard]] inline std::string& getViewName() { return viewName; }
    [[nodiscard]] std::string getWindowNameImGui(const std::vector<DataViewPtr>& dataViews, int index) const;

    SceneData* parentSceneData = nullptr;
    bool showWindow = true;
    bool useLinearRGB = false;
    sgl::Color clearColor;
    std::string viewName = "Data View";
    int viewIdx;
    static std::set<int> usedViewIndices;
    static std::set<int> freeViewIndices;
    static std::vector<SceneData*> globalSceneData;

    bool reRender = false;

    sgl::vk::Renderer* renderer = nullptr;
    sgl::vk::Device* device = nullptr;
    VkFormat sceneDepthTextureVkFormat = VK_FORMAT_D32_SFLOAT;
    sgl::vk::TexturePtr sceneTextureVk;
    sgl::vk::TexturePtr sceneDepthTextureVk;
    sgl::vk::TexturePtr compositedTextureVk; ///< The final RGBA8 texture.
    sgl::vk::BlitRenderPassPtr sceneTextureBlitPass, sceneTextureBlitDownscalePass;
    sgl::vk::BlitRenderPassPtr sceneTextureGammaCorrectionPass, sceneTextureGammaCorrectionDownscalePass;

    sgl::vk::ScreenshotReadbackHelperPtr screenshotReadbackHelper; ///< For reading back screenshots from the GPU.

    VkDescriptorSet descriptorSetImGui{};

    /// Scene data (e.g., camera, main framebuffer, ...).
    bool syncWithParentCamera = true;
    sgl::CameraPtr camera;
    int32_t viewportPositionX = 0;
    int32_t viewportPositionY = 0;
    uint32_t viewportWidth = 0;
    uint32_t viewportHeight = 0;
    uint32_t viewportWidthVirtual = 0; //< Includes the supersampling factor.
    uint32_t viewportHeightVirtual = 0; //< Includes the supersampling factor.
    int supersamplingFactor = 1;
    SceneData sceneData;
};

#endif //CORRERENDER_DATAVIEW_HPP
