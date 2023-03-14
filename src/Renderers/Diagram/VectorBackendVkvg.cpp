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

#include <glm/vec3.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Graphics/Color.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <Graphics/Vector/VectorWidget.hpp>

#include <vkvg.h>

#include "VectorBackendVkvg.hpp"

struct VkvgCache {
    VkvgDevice device{};
    VkvgSurface surface{};
    VkvgContext context{};
};

bool VectorBackendVkvg::checkIsSupported() {
    return true;
}

VectorBackendVkvg::VectorBackendVkvg(sgl::VectorWidget* vectorWidget) : sgl::VectorBackend(vectorWidget) {
    ;
}

void VectorBackendVkvg::initialize() {
    if (initialized) {
        return;
    }
    initialized = true;

    sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    /*
     * TODO for future:
     * - Support fDeviceFeatures2 / VkPhysicalDeviceFeatures2.
     */

    vkvgCache = new VkvgCache;

    VkPhysicalDeviceFeatures physicalDeviceFeatures{};
    auto* physicalDeviceVulkan12Features =
            (VkPhysicalDeviceVulkan12Features*)vkvg_get_device_requirements(&physicalDeviceFeatures);

    uint32_t instanceExtensionCount = 0;
    vkvg_get_required_instance_extensions(nullptr, &instanceExtensionCount);
    std::vector<const char*> instanceExtensions(instanceExtensionCount);
    vkvg_get_required_instance_extensions(instanceExtensions.data(), &instanceExtensionCount);

    uint32_t deviceExtensionCount = 0;
    vkvg_get_required_device_extensions(device->getVkPhysicalDevice(), nullptr, &deviceExtensionCount);
    std::vector<const char*> deviceExtensions(deviceExtensionCount);
    vkvg_get_required_device_extensions(device->getVkPhysicalDevice(), deviceExtensions.data(), &deviceExtensionCount);

    if (sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        vkvgCache->device = vkvg_device_create_from_vk_multisample(
                instance->getVkInstance(), device->getVkPhysicalDevice(), device->getVkDevice(),
                device->getGraphicsQueueIndex(), 0, sampleCount, true);
    } else {
        vkvgCache->device = vkvg_device_create_from_vk(
                instance->getVkInstance(), device->getVkPhysicalDevice(), device->getVkDevice(),
                device->getGraphicsQueueIndex(), 0);
    }
    if (!vkvgCache->device) {
        sgl::Logfile::get()->throwError("Error in VkvgRenderPass::VkvgRenderPass: VKVG device creation failed.");
    }

    auto dpi = int(std::round(96.0f * scaleFactor));
    vkvg_device_set_dpy(vkvgCache->device, dpi, dpi);

    std::string fontFilename = sgl::AppSettings::get()->getDataDirectory() + "Fonts/DroidSans.ttf";
    if (!sgl::loadFileFromSource(fontFilename, fontBuffer, fontBufferSize, true)) {
        sgl::Logfile::get()->throwError("Error in VectorBackendVkvg::initialize: Couldn't find the font file.");
    }
}

void VectorBackendVkvg::initializeFont() {
    //std::string fontFilename = sgl::AppSettings::get()->getDataDirectory() + "Fonts/DroidSans.ttf";
    // As of 2023-03-14, VKVG takes ownership of the memory and calls "free" on it.
    auto* fontBufferCopy = static_cast<uint8_t*>(malloc(fontBufferSize));
    memcpy(fontBufferCopy, fontBuffer, fontBufferSize);
    vkvg_load_font_from_memory(vkvgCache->context, fontBufferCopy, long(fontBufferSize), "sans");
    //vkvg_load_font_from_path(vkvgCache->context, fontFilename.c_str(), "sans");
    //vkvg_select_font_face(vkvgCache->context, "sans");
}

void VectorBackendVkvg::destroy() {
    if (!initialized) {
        return;
    }

    renderTargetTextureVk = {};
    renderTargetImageViewVk = {};

    if (vkvgCache->context) {
        vkvg_destroy(vkvgCache->context);
    }
    if (vkvgCache->surface) {
        vkvg_surface_destroy(vkvgCache->surface);
    }
    if (vkvgCache->device) {
        vkvg_device_destroy(vkvgCache->device);
    }
    if (fontBuffer) {
        delete[] fontBuffer;
        fontBuffer = nullptr;
        fontBufferSize = 0;
    }
    delete vkvgCache;
    vkvgCache = nullptr;

    initialized = false;
}

void VectorBackendVkvg::onResize() {
    renderTargetTextureVk = {};

    if (vkvgCache->context) {
        vkvg_destroy(vkvgCache->context);
    }
    if (vkvgCache->surface) {
        vkvg_surface_destroy(vkvgCache->surface);
    }

    vkvgCache->surface = vkvg_surface_create(vkvgCache->device, fboWidthInternal, fboHeightInternal);
    if (!vkvgCache->surface) {
        sgl::Logfile::get()->throwError(
                "Error in VkvgRenderPass::recreateSwapchain: vkvg_surface_create failed.");
    }

    vkvgCache->context = vkvg_create(vkvgCache->surface);
    initializeFont();
    if (!vkvgCache->surface) {
        sgl::Logfile::get()->throwError(
                "Error in VkvgRenderPass::recreateSwapchain: vkvg_create failed.");
    }
}

VkvgContext VectorBackendVkvg::getContext() {
    if (!initialized) {
        initialize();
    }

    return vkvgCache->context;
}

void VectorBackendVkvg::renderStart() {
    if (!initialized) {
        initialize();
    }

    vkvg_clear(vkvgCache->context);
    if (clearColor != glm::vec4(0.0f)) {
        //vkvg_rectangle(vkvgCache->context, 0, 0, float(imageSettings.width), float(imageSettings.height));
        vkvg_set_source_color(vkvgCache->context, sgl::colorFromVec4(clearColor).getColorRGBA());
        vkvg_paint(vkvgCache->context);
    }
}

void VectorBackendVkvg::renderEnd() {
    vkvg_flush(vkvgCache->context);
    if (sampleCount != VK_SAMPLE_COUNT_1_BIT) {
        vkvg_surface_resolve(vkvgCache->surface);
    }

    if (!renderTargetTextureVk) {
        VkFormat format = vkvg_surface_get_vk_format(vkvgCache->surface);
        VkImage image = vkvg_surface_get_vk_image(vkvgCache->surface);

        sgl::vk::ImageSettings imageSettings;
        imageSettings.width = fboWidthInternal;
        imageSettings.height = fboHeightInternal;
        imageSettings.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageSettings.format = format;
        imageSettings.usage =
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        sgl::vk::Device* device = rendererVk->getDevice();
        auto skiaImage = std::make_shared<sgl::vk::Image>(device, imageSettings, image, false);
        auto skiaImageView = std::make_shared<sgl::vk::ImageView>(skiaImage);
        renderTargetTextureVk = std::make_shared<sgl::vk::Texture>(
                skiaImageView, std::make_shared<sgl::vk::ImageSampler>(device, sgl::vk::ImageSamplerSettings()));
    }

    if (sampleCount == VK_SAMPLE_COUNT_1_BIT) {
        renderTargetTextureVk->getImage()->overwriteImageLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    } else {
        renderTargetTextureVk->getImage()->overwriteImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }
}

void VectorBackendVkvg::onRenderFinished() {
    if (sampleCount == VK_SAMPLE_COUNT_1_BIT) {
        renderTargetTextureVk->getImage()->transitionImageLayout(
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, rendererVk->getVkCommandBuffer());
    } else {
        renderTargetTextureVk->getImage()->transitionImageLayout(
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, rendererVk->getVkCommandBuffer());
    }
}

bool VectorBackendVkvg::renderGuiPropertyEditor(sgl::PropertyEditor& propertyEditor) {
    bool reRender = VectorBackend::renderGuiPropertyEditor(propertyEditor);
    bool recreate = false;

    int maxMsaaSamples = 32;
#ifdef SUPPORT_VULKAN
    if (renderBackend == sgl::RenderSystem::VULKAN) {
        maxMsaaSamples = int(rendererVk->getDevice()->getMaxUsableSampleCount());
    }
#endif
    if (propertyEditor.addSliderIntPowerOfTwo("#MSAA Samples", (int*)&sampleCount, 1, maxMsaaSamples)) {
        sampleCount = VkSampleCountFlagBits(glm::clamp(int(sampleCount), 1, maxMsaaSamples));
        recreate = true;
    }
    if (propertyEditor.addSliderIntPowerOfTwo("SSAA Factor", &supersamplingFactor, 1, 4)) {
        vectorWidget->setSupersamplingFactor(supersamplingFactor, false);
        recreate = true;
    }

    if (recreate) {
        destroy();
        initialize();
        vectorWidget->onWindowSizeChanged();
        reRender = true;
    }

    return reRender;
}

void VectorBackendVkvg::copyVectorBackendSettingsFrom(VectorBackend* backend) {
    if (getID() != backend->getID()) {
        sgl::Logfile::get()->throwError(
                "Error in VectorBackendVkvg::copyVectorBackendSettingsFrom: Vector backend ID mismatch.");
    }

    auto* vkvgBackend = static_cast<VectorBackendVkvg*>(backend);

    bool recreate = false;
    if (sampleCount != vkvgBackend->sampleCount) {
        sampleCount = vkvgBackend->sampleCount;
        recreate = true;
    }
    if (supersamplingFactor != vkvgBackend->supersamplingFactor) {
        supersamplingFactor = vkvgBackend->supersamplingFactor;
        vectorWidget->setSupersamplingFactor(supersamplingFactor, false);
        recreate = true;
    }

    if (recreate) {
        destroy();
        initialize();
        vectorWidget->onWindowSizeChanged();
    }
}
