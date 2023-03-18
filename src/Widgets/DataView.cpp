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

#include <Utils/AppSettings.hpp>
#include <Utils/SciVis/Navigation/CameraNavigator2D.hpp>
#include <Input/Mouse.hpp>
#include <Graphics/Scene/RenderTarget.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/imgui_impl_vulkan.h>

#include "Volume/VolumeData.hpp"
#include "Renderers/Renderer.hpp"
#include "DataView.hpp"

std::set<int> DataView::usedViewIndices;
std::set<int> DataView::freeViewIndices;

std::string DataView::getWindowNameImGui(const std::vector<DataViewPtr>& dataViews, int index) const {
    bool foundDuplicateName = false;
    for (int i = 0; i < int(dataViews.size()); i++) {
        if (i == index) {
            continue;
        }
        if (dataViews.at(i)->viewName == viewName) {
            foundDuplicateName = true;
            break;
        }
    }

    if (foundDuplicateName) {
        int idx = 1;
        for (int i = 0; i < index; i++) {
            if (dataViews.at(i)->viewName == viewName) {
                idx++;
            }
        }
        return viewName + " (" + std::to_string(idx) + ")###data_view_" + std::to_string(viewIdx);
    } else {
        return viewName + "###data_view_" + std::to_string(viewIdx);
    }
}

DataView::DataView(SceneData* parentSceneData)
        : parentSceneData(parentSceneData), renderer(*parentSceneData->renderer), sceneData(*parentSceneData) {
    if (freeViewIndices.empty()) {
        viewIdx = int(usedViewIndices.size());
    } else {
        auto it = freeViewIndices.begin();
        viewIdx = *it;
        freeViewIndices.erase(it);
    }
    usedViewIndices.insert(viewIdx);

    device = renderer->getDevice();

    sceneData.sceneTexture = &sceneTextureVk;
    sceneData.sceneDepthTexture = &sceneDepthTextureVk;
    sceneData.viewportPositionX = &viewportPositionX;
    sceneData.viewportPositionY = &viewportPositionY;
    sceneData.viewportWidth = &viewportWidth;
    sceneData.viewportHeight = &viewportHeight;
    sceneData.viewportWidthVirtual = &viewportWidthVirtual;
    sceneData.viewportHeightVirtual = &viewportHeightVirtual;

    const sgl::CameraPtr& parentCamera = parentSceneData->camera;

    camera = std::make_shared<sgl::Camera>();
    camera->copyState(parentCamera);
    sceneData.camera = camera;

    sceneTextureBlitPass = std::make_shared<sgl::vk::BlitRenderPass>(renderer);
    sceneTextureBlitDownscalePass = sgl::vk::BlitRenderPassPtr(new sgl::vk::BlitRenderPass(
            renderer, { "Blit.Vertex", "Blit.FragmentDownscale" }));
    sceneTextureGammaCorrectionPass = sgl::vk::BlitRenderPassPtr(new sgl::vk::BlitRenderPass(
            renderer, { "GammaCorrection.Vertex", "GammaCorrection.Fragment" }));
    sceneTextureGammaCorrectionDownscalePass = sgl::vk::BlitRenderPassPtr(new sgl::vk::BlitRenderPass(
            renderer, { "GammaCorrection.Vertex", "GammaCorrection.FragmentDownscale" }));

    screenshotReadbackHelper = std::make_shared<sgl::vk::ScreenshotReadbackHelper>(renderer);
}

DataView::~DataView() {
    if (descriptorSetImGui) {
        sgl::ImGuiWrapper::get()->freeDescriptorSet(descriptorSetImGui);
        descriptorSetImGui = nullptr;
    }
    freeViewIndices.insert(viewIdx);
}

void DataView::resize(int newWidth, int newHeight) {
    viewportWidth = uint32_t(std::max(newWidth, 0));
    viewportHeight = uint32_t(std::max(newHeight, 0));
    viewportWidthVirtual = viewportWidth * uint32_t(supersamplingFactor);
    viewportHeightVirtual = viewportHeight * uint32_t(supersamplingFactor);

    if (viewportWidth == 0 || viewportHeight == 0) {
        sceneTextureVk = {};
        sceneDepthTextureVk = {};
        compositedTextureVk = {};
        return;
    }

    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = viewportWidthVirtual;
    imageSettings.height = viewportHeightVirtual;

    // Create scene texture.
    imageSettings.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    if (useLinearRGB) {
        imageSettings.format = VK_FORMAT_R16G16B16A16_UNORM;
    } else {
        imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    }
    sceneTextureVk = std::make_shared<sgl::vk::Texture>(
            device, imageSettings, sgl::vk::ImageSamplerSettings(),
            VK_IMAGE_ASPECT_COLOR_BIT);
    //sceneTextureVk->getImage()->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    renderer->transitionImageLayout(sceneTextureVk->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    if (*sceneData.screenshotTransparentBackground) {
        clearColor.setA(0);
    }
    sceneTextureVk->getImageView()->clearColor(
            clearColor.getFloatColorRGBA(), renderer->getVkCommandBuffer());
    if (*sceneData.screenshotTransparentBackground) {
        clearColor.setA(255);
    }

    // Create scene depth texture.
    imageSettings.usage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageSettings.format = sceneDepthTextureVkFormat;
    sceneDepthTextureVk = std::make_shared<sgl::vk::Texture>(
            device, imageSettings, sgl::vk::ImageSamplerSettings(),
            VK_IMAGE_ASPECT_DEPTH_BIT);
    sceneData.initDepthColor();

    // Create composited (gamma-resolved, if VK_FORMAT_R16G16B16A16_UNORM for scene texture) scene texture.
    imageSettings.width = viewportWidth;
    imageSettings.height = viewportHeight;
    imageSettings.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    compositedTextureVk = std::make_shared<sgl::vk::Texture>(
            device, imageSettings, sgl::vk::ImageSamplerSettings(),
            VK_IMAGE_ASPECT_COLOR_BIT);

    // Pass the textures to the render passes.
    sceneTextureBlitPass->setInputTexture(sceneTextureVk);
    sceneTextureBlitPass->setOutputImage(compositedTextureVk->getImageView());
    sceneTextureBlitPass->recreateSwapchain(viewportWidth, viewportHeight);
    sceneTextureBlitDownscalePass->setInputTexture(sceneTextureVk);
    sceneTextureBlitDownscalePass->setOutputImage(compositedTextureVk->getImageView());
    sceneTextureBlitDownscalePass->recreateSwapchain(viewportWidth, viewportHeight);
    sceneTextureGammaCorrectionPass->setInputTexture(sceneTextureVk);
    sceneTextureGammaCorrectionPass->setOutputImage(compositedTextureVk->getImageView());
    sceneTextureGammaCorrectionPass->recreateSwapchain(viewportWidth, viewportHeight);
    sceneTextureGammaCorrectionDownscalePass->setInputTexture(sceneTextureVk);
    sceneTextureGammaCorrectionDownscalePass->setOutputImage(compositedTextureVk->getImageView());
    sceneTextureGammaCorrectionDownscalePass->recreateSwapchain(viewportWidth, viewportHeight);

    screenshotReadbackHelper->onSwapchainRecreated(viewportWidth, viewportHeight);

    if (descriptorSetImGui) {
        sgl::ImGuiWrapper::get()->freeDescriptorSet(descriptorSetImGui);
        descriptorSetImGui = nullptr;
    }
    descriptorSetImGui = ImGui_ImplVulkan_AddTexture(
            compositedTextureVk->getImageSampler()->getVkSampler(),
            compositedTextureVk->getImageView()->getVkImageView(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    auto renderTarget = std::make_shared<sgl::RenderTarget>(
            int(viewportWidthVirtual), int(viewportHeightVirtual));
    camera->setRenderTarget(renderTarget, false);
    camera->onResolutionChanged({});
}

void DataView::beginRender() {
    if (syncWithParentCamera) {
        const sgl::CameraPtr& parentCamera = parentSceneData->camera;
        camera->copyState(parentCamera);
    }

    renderer->setProjectionMatrix(camera->getProjectionMatrix());
    renderer->setViewMatrix(camera->getViewMatrix());
    renderer->setModelMatrix(sgl::matrixIdentity());

    if (*sceneData.screenshotTransparentBackground) {
        clearColor.setA(0);
    }
    renderer->insertImageMemoryBarriers(
            { sceneTextureVk->getImage(), sceneDepthTextureVk->getImage() },
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
    sceneTextureVk->getImageView()->clearColor(clearColor.getFloatColorRGBA(), renderer->getVkCommandBuffer());
    sceneDepthTextureVk->getImageView()->clearDepthStencil(1.0f, 0, renderer->getVkCommandBuffer());
    sceneData.clearRenderTargetState();
    if (*sceneData.screenshotTransparentBackground) {
        clearColor.setA(255);
    }
}

void DataView::endRender() {
    sceneData.switchColorState(RenderTargetAccess::SAMPLED_FRAGMENT_SHADER);
    //renderer->transitionImageLayout(
    //        sceneTextureVk->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    sgl::vk::BlitRenderPass* renderPass;
    if (useLinearRGB) {
        if (supersamplingFactor > 1) {
            renderPass = sceneTextureGammaCorrectionDownscalePass.get();
        } else {
            renderPass = sceneTextureGammaCorrectionPass.get();
        }
    } else {
        if (supersamplingFactor > 1) {
            renderPass = sceneTextureBlitDownscalePass.get();
        } else {
            renderPass = sceneTextureBlitPass.get();
        }
    }
    if (supersamplingFactor > 1) {
        renderPass->buildIfNecessary();
        renderer->pushConstants(
                renderPass->getGraphicsPipeline(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, supersamplingFactor);
    }
    renderPass->render();
}

void DataView::syncCamera() {
    const sgl::CameraPtr& parentCamera = parentSceneData->camera;
    camera->copyState(parentCamera);
}

void DataView::saveScreenshot(const std::string& filename) {
    renderer->transitionImageLayout(
            compositedTextureVk->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    screenshotReadbackHelper->requestScreenshotReadback(compositedTextureVk->getImage(), filename);
}

void DataView::saveScreenshotDataIfAvailable() {
    if (viewportWidth == 0 || viewportHeight == 0) {
        return;
    }
    sgl::vk::Swapchain* swapchain = sgl::AppSettings::get()->getSwapchain();
    uint32_t imageIndex = swapchain ? swapchain->getImageIndex() : 0;
    screenshotReadbackHelper->saveDataIfAvailable(imageIndex);
}

ImTextureID DataView::getImGuiTextureId() const {
    compositedTextureVk->getImage()->transitionImageLayout(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
    return reinterpret_cast<ImTextureID>(descriptorSetImGui);
}

void DataView::setClearColor(const sgl::Color& color) {
    clearColor = color;
}
