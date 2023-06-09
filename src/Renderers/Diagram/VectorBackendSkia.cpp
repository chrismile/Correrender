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
#include <Graphics/Color.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <Graphics/Vector/VectorWidget.hpp>

#include <core/SkMilestone.h>
#include <core/SkCanvas.h>
#include <core/SkSurface.h>
#include <core/SkColorSpace.h>
#include <core/SkFont.h>
#include <gpu/GrDirectContext.h>
#include <gpu/vk/GrVkBackendContext.h>
#include <gpu/vk/VulkanExtensions.h>
#if SK_MILESTONE >= 115
#include <gpu/ganesh/SkSurfaceGanesh.h>
#include <gpu/GrBackendSurface.h>
#endif

#include "VectorBackendSkia.hpp"

struct SkiaCache {
    sk_sp<GrDirectContext> context{};
    sk_sp<SkColorSpace> colorSpace;
    sk_sp<SkSurface> surface{};
};

bool VectorBackendSkia::checkIsSupported() {
    return true;
}

VectorBackendSkia::VectorBackendSkia(sgl::VectorWidget* vectorWidget) : sgl::VectorBackend(vectorWidget) {
    ;
}

void VectorBackendSkia::initialize() {
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

    skiaCache = new SkiaCache;

    skiaCache->colorSpace = SkColorSpace::MakeSRGB();

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddrPtr = sgl::vk::Instance::getVkInstanceProcAddrFunctionPointer();
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddrPtr = sgl::vk::Device::getVkDeviceProcAddrFunctionPointer();
    auto getProcFun = [vkGetInstanceProcAddrPtr, vkGetDeviceProcAddrPtr](
            const char* proc_name, VkInstance instance, VkDevice device) {
        if (device != VK_NULL_HANDLE) {
            return vkGetDeviceProcAddrPtr(device, proc_name);
        }
        return vkGetInstanceProcAddrPtr(instance, proc_name);
    };

    const std::vector<const char*>& instanceExtensions = instance->getEnabledInstanceExtensionNames();
    const std::vector<const char*>& deviceExtensions = device->getEnabledDeviceExtensionNames();
    skgpu::VulkanExtensions skVkExtensions;
    skVkExtensions.init(
            getProcFun,
            instance->getVkInstance(), device->getVkPhysicalDevice(),
            uint32_t(instanceExtensions.size()), instanceExtensions.data(),
            uint32_t(deviceExtensions.size()), deviceExtensions.data());

    GrVkBackendContext skVkBackendContext;
    skVkBackendContext.fInstance = instance->getVkInstance();
    skVkBackendContext.fPhysicalDevice = device->getVkPhysicalDevice();
    skVkBackendContext.fDevice = device->getVkDevice();
    skVkBackendContext.fQueue = device->getGraphicsQueue();
    skVkBackendContext.fGraphicsQueueIndex = device->getGraphicsQueueIndex();
    skVkBackendContext.fMaxAPIVersion = instance->getApplicationInfo().apiVersion;
    skVkBackendContext.fVkExtensions = &skVkExtensions;
    skVkBackendContext.fDeviceFeatures = &device->getPhysicalDeviceFeatures();
    skVkBackendContext.fDeviceFeatures2 = nullptr;
    skVkBackendContext.fGetProc = getProcFun;

    //sk_sp<skgpu::VulkanMemoryAllocator> fMemoryAllocator;
    skiaCache->context = GrDirectContext::MakeVulkan(skVkBackendContext);
    if (!skiaCache->context) {
        sgl::Logfile::get()->writeError("Error in SkiaRenderPass::SkiaRenderPass: GrDirectContext::MakeVulkan failed.");
    }
}

sk_sp<SkTypeface> VectorBackendSkia::createDefaultTypeface() {
    std::string fontFilename = sgl::AppSettings::get()->getDataDirectory() + "Fonts/DroidSans.ttf";
    return SkTypeface::MakeFromFile(fontFilename.c_str());
    //return SkTypeface::MakeDefault();
}

void VectorBackendSkia::destroy() {
    if (!initialized) {
        return;
    }

    renderTargetTextureVk = {};
    renderTargetImageViewVk = {};

    if (skiaCache->surface) {
        skiaCache->surface.reset();
    }
    if (skiaCache->context) {
        skiaCache->context.reset();
    }
    skiaCache->colorSpace.reset();
    delete skiaCache;
    skiaCache = nullptr;

    initialized = false;
}

void VectorBackendSkia::onResize() {
    renderTargetTextureVk = {};

    if (skiaCache->surface) {
        skiaCache->surface.reset();
    }

    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = fboWidthInternal;
    imageSettings.height = fboHeightInternal;
    imageSettings.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    // Skia creates an intermediate MSAA VkImage and will then resolve into this VkImage.
    imageSettings.numSamples = VK_SAMPLE_COUNT_1_BIT;
    imageSettings.usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
            | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
            | VK_IMAGE_USAGE_SAMPLED_BIT;
    renderTargetTextureVk = std::make_shared<sgl::vk::Texture>(rendererVk->getDevice(), imageSettings);

    GrVkImageInfo skVkImageInfo;
    skVkImageInfo.fImage = renderTargetTextureVk->getImage()->getVkImage();
    skVkImageInfo.fImageTiling = imageSettings.tiling;
    skVkImageInfo.fImageLayout = renderTargetTextureVk->getImage()->getVkImageLayout();
    skVkImageInfo.fFormat = imageSettings.format;
    skVkImageInfo.fImageUsageFlags = imageSettings.usage;
    skVkImageInfo.fSampleCount = uint32_t(imageSettings.numSamples);
    skVkImageInfo.fLevelCount = imageSettings.mipLevels;
    skVkImageInfo.fSharingMode = imageSettings.sharingMode;

    // TODO: kDynamicMSAA_Flag: "Use internal MSAA to render to non-MSAA GPU surfaces."
    uint32_t flags = 0;
    if (useInternalAA) {
        flags |= SkSurfaceProps::kDynamicMSAA_Flag;
    }
    SkSurfaceProps skSurfaceProps(flags, kUnknown_SkPixelGeometry);

    GrBackendTexture backendTexture(int(fboWidthInternal), int(fboHeightInternal), skVkImageInfo);
#if SK_MILESTONE < 115
    skiaCache->surface = SkSurface::MakeFromBackendTexture(
#else
    skiaCache->surface = SkSurfaces::WrapBackendTexture(
#endif
            skiaCache->context.get(), backendTexture,
            kTopLeft_GrSurfaceOrigin, int(sampleCount), kRGBA_8888_SkColorType,
            nullptr, &skSurfaceProps);
            //skiaCache->colorSpace, &skSurfaceProps);
    if (!skiaCache->surface) {
        sgl::Logfile::get()->throwError(
                "Error in SkiaRenderPass::recreateSwapchain: SkSurface::MakeFromBackendTexture failed.");
    }
}

SkCanvas* VectorBackendSkia::getCanvas() {
    if (!initialized) {
        initialize();
    }

    return skiaCache->surface->getCanvas();
}

void VectorBackendSkia::initializePaint(SkPaint* paint) {
    paint->setAntiAlias(usePaintAA);
}

void VectorBackendSkia::renderStart() {
    if (!initialized) {
        initialize();
    }

    sgl::Color col = sgl::colorFromVec4(clearColor);
    auto skClearColor = SkColorSetARGB(col.getA(), col.getR(), col.getG(), col.getB());
    skiaCache->surface->getCanvas()->clear(skClearColor);
}

void VectorBackendSkia::renderEnd() {
    skiaCache->surface->flush();
    skiaCache->context->submit();

#if SK_MILESTONE < 115
    GrBackendTexture backendTexture = skiaCache->surface->getBackendTexture(SkSurface::kFlushRead_BackendHandleAccess);
#else
    GrBackendTexture backendTexture = SkSurfaces::GetBackendTexture(
            skiaCache->surface.get(), SkSurface::kFlushRead_BackendHandleAccess);
#endif
    if (!backendTexture.isValid()) {
        sgl::Logfile::get()->throwError("Error in SkiaRenderPass::recreateSwapchain: getBackendTexture failed.");
    }
    GrVkImageInfo skImageInfo;
    if (!backendTexture.getVkImageInfo(&skImageInfo)) {
        sgl::Logfile::get()->throwError("Error in SkiaRenderPass::recreateSwapchain: getVkImageInfo failed.");
    }

    renderTargetTextureVk->getImage()->overwriteImageLayout(skImageInfo.fImageLayout);
    renderTargetTextureVk->getImage()->transitionImageLayout(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rendererVk->getVkCommandBuffer());

    backendTexture.setVkImageLayout(renderTargetTextureVk->getImage()->getVkImageLayout());
}

bool VectorBackendSkia::renderGuiPropertyEditor(sgl::PropertyEditor& propertyEditor) {
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
    if (propertyEditor.addCheckbox("Paint AA", &usePaintAA)) {
        reRender = true;
    }
    if (propertyEditor.addCheckbox("Internal AA", &useInternalAA)) {
        reRender = true;
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

void VectorBackendSkia::copyVectorBackendSettingsFrom(VectorBackend* backend) {
    if (getID() != backend->getID()) {
        sgl::Logfile::get()->throwError(
                "Error in VectorBackendSkia::copyVectorBackendSettingsFrom: Vector backend ID mismatch.");
    }

    auto* skiaBackend = static_cast<VectorBackendSkia*>(backend);

    bool recreate = false;
    if (sampleCount != skiaBackend->sampleCount) {
        sampleCount = skiaBackend->sampleCount;
        recreate = true;
    }
    if (usePaintAA != skiaBackend->usePaintAA) {
        usePaintAA = skiaBackend->usePaintAA;
        recreate = true;
    }
    if (useInternalAA != skiaBackend->useInternalAA) {
        useInternalAA = skiaBackend->useInternalAA;
        recreate = true;
    }
    if (supersamplingFactor != skiaBackend->supersamplingFactor) {
        supersamplingFactor = skiaBackend->supersamplingFactor;
        vectorWidget->setSupersamplingFactor(supersamplingFactor, false);
        recreate = true;
    }

    if (recreate) {
        destroy();
        initialize();
        vectorWidget->onWindowSizeChanged();
    }
}

SkColor toSkColor(const sgl::Color& col) {
    return SkColorSetARGB(col.getA(), col.getR(), col.getG(), col.getB());
}
