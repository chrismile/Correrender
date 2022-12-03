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

#include <unordered_map>

#ifdef USE_PYTHON
#include <Utils/Python/PythonInit.hpp>
#endif

#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/AppLogic.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Shader/ShaderManager.hpp>

#include "MainApp.hpp"

int main(int argc, char *argv[]) {
    // Initialize the filesystem utilities.
    sgl::FileUtils::get()->initialize("Correrender", argc, argv);

#ifdef DATA_PATH
    if (!sgl::FileUtils::get()->directoryExists("Data") && !sgl::FileUtils::get()->directoryExists("../Data")) {
        sgl::AppSettings::get()->setDataDirectory(DATA_PATH);
    }
#endif
    sgl::AppSettings::get()->initializeDataDirectory();

    std::string iconPath = sgl::AppSettings::get()->getDataDirectory() + "Fonts/icon_256.png";
    sgl::AppSettings::get()->loadApplicationIconFromFile(iconPath);

    // Parse the arguments.
    bool usePerfMode = false;
    for (int i = 1; i < argc; i++) {
        std::string command = argv[i];
        if (command == "--perf") {
            usePerfMode = true;
        }
    }

    // Load the file containing the app settings
    std::string settingsFile = sgl::FileUtils::get()->getConfigDirectory() + "settings.txt";
    sgl::AppSettings::get()->loadSettings(settingsFile.c_str());
    sgl::AppSettings::get()->getSettings().addKeyValue("window-multisamples", 0);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);
    if (usePerfMode) {
        sgl::AppSettings::get()->getSettings().addKeyValue("window-vSync", false);
    } else {
        sgl::AppSettings::get()->getSettings().addKeyValue("window-vSync", true);
    }
    sgl::AppSettings::get()->getSettings().addKeyValue("window-resizable", true);
    sgl::AppSettings::get()->getSettings().addKeyValue("window-savePosition", true);
    //sgl::AppSettings::get()->setVulkanDebugPrintfEnabled();

    ImVector<ImWchar> fontRanges;
    ImFontGlyphRangesBuilder builder;
    builder.AddChar(L'\u03BB'); // lambda
    builder.BuildRanges(&fontRanges);
    bool useMultiViewport = false;
    if (sgl::AppSettings::get()->getSettings().getValueOpt("useDockSpaceMode", useMultiViewport)) {
        useMultiViewport = !useMultiViewport;
    }
    sgl::AppSettings::get()->setLoadGUI(fontRanges.Data, true, useMultiViewport);

    sgl::AppSettings::get()->setRenderSystem(sgl::RenderSystem::VULKAN);
    sgl::Window* window = sgl::AppSettings::get()->createWindow();

    std::vector<const char*> optionalDeviceExtensions;
#ifdef SUPPORT_CUDA_INTEROP
    optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
#endif
    std::vector<const char*> raytracingDeviceExtensions = {
    };
    optionalDeviceExtensions.insert(
            optionalDeviceExtensions.end(),
            raytracingDeviceExtensions.begin(), raytracingDeviceExtensions.end());
    //optionalDeviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);

    sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
    sgl::vk::Device* device = new sgl::vk::Device;
    sgl::vk::DeviceFeatures requestedDeviceFeatures{};
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.sampleRateShading = VK_TRUE; // For MSAA.
    requestedDeviceFeatures.optionalEnableShaderDrawParametersFeatures = true; // For deferred shading.
    requestedDeviceFeatures.requestedPhysicalDeviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
    // For ensemble combination when using Vulkan-CUDA interop with PyTorch.
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.runtimeDescriptorArray = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    device->createDeviceSwapchain(
            instance, window,
            {
                    VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
            },
            optionalDeviceExtensions, requestedDeviceFeatures);
    sgl::vk::Swapchain* swapchain = new sgl::vk::Swapchain(device);
    swapchain->create(window);
    sgl::AppSettings::get()->setPrimaryDevice(device);
    sgl::AppSettings::get()->setSwapchain(swapchain);
    sgl::AppSettings::get()->initializeSubsystems();

#ifdef USE_PYTHON
    sgl::pythonInit(argc, argv);
#endif

    auto app = new MainApp();
    app->run();
    delete app;

    sgl::AppSettings::get()->release();

#ifdef USE_PYTHON
    Py_Finalize();
#endif

    return 0;
}
