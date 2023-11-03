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

#include <Utils/StringUtils.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/AppLogic.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Shader/ShaderManager.hpp>

#ifdef SUPPORT_OPENGL
#include <Graphics/OpenGL/Context/OffscreenContextEGL.hpp>
#include <Graphics/OpenGL/Context/OffscreenContextGlfw.hpp>
#endif

#include "Renderers/Diagram/SamplingTest.hpp"
#include "MainApp.hpp"

void vulkanErrorCallbackHeadless() {
    SDL_CaptureMouse(SDL_FALSE);
    std::cerr << "Application callback" << std::endl;
}

#ifdef __linux__
#include <sys/resource.h>
// Some systems have 1024 open file descriptors as the maximum. Increase the soft limit of the process to the maximum.
void setFileDescriptorLimit() {
    rlimit lim{};
    getrlimit(RLIMIT_NOFILE, &lim);
    size_t rlimOld = lim.rlim_cur;
    if (lim.rlim_cur < lim.rlim_max) {
        lim.rlim_cur = lim.rlim_max;
        if (setrlimit(RLIMIT_NOFILE, &lim) == -1) {
            sgl::Logfile::get()->writeError("Error in setFileDescriptorLimit: setrlimit failed.");
            return;
        }
        getrlimit(RLIMIT_NOFILE, &lim);
        sgl::Logfile::get()->write(
                "File descriptor limit: " + std::to_string(lim.rlim_cur)
                + " (old: " + std::to_string(rlimOld) + ")", sgl::BLUE);
    } else {
        sgl::Logfile::get()->write("File descriptor limit: " + std::to_string(lim.rlim_cur), sgl::BLUE);
    }
}
#endif

int main(int argc, char *argv[]) {
    // Initialize the filesystem utilities.
    std::setlocale(LC_ALL, "en_US.UTF-8"); // For font rendering with VKVG.
    sgl::FileUtils::get()->initialize("Correrender", argc, argv);

    // Parse the arguments.
    bool usePerfMode = false, useSamplingMode = false, useReplicabilityStampMode = false;
    std::string dataSetPath;
    int testIdx = -1;
    bool useCustomShaderCompilerBackend = false;
    sgl::vk::ShaderCompilerBackend shaderCompilerBackend = sgl::vk::ShaderCompilerBackend::SHADERC;
    for (int i = 1; i < argc; i++) {
        std::string command = argv[i];
        if (command == "--perf") {
            usePerfMode = true;
        } else if (command == "--sampling") {
            useSamplingMode = true;
            if (i + 1 < argc && !sgl::startsWith(argv[i + 1], "-")) {
                i++;
                dataSetPath = argv[i];
            }
            if (i + 1 < argc && !sgl::startsWith(argv[i + 1], "-")) {
                i++;
                testIdx = sgl::fromString<int>(argv[i]);
            }
        } else if (command == "--replicability") {
            useReplicabilityStampMode = true;
        } else if (command == "--shader-backend") {
            i++;
            if (i >= argc) {
                sgl::Logfile::get()->throwError(
                        "Error: Command line argument '--shader-backend' expects a backend name.");
            }
            useCustomShaderCompilerBackend = true;
            std::string backendName = argv[i];
            if (backendName == "shaderc") {
                shaderCompilerBackend = sgl::vk::ShaderCompilerBackend::SHADERC;
            } else if (backendName == "glslang") {
                shaderCompilerBackend = sgl::vk::ShaderCompilerBackend::GLSLANG;
            }
        }
    }
    bool isHeadlessMode = useSamplingMode;

#ifdef DATA_PATH
    if (!sgl::FileUtils::get()->directoryExists("Data") && !sgl::FileUtils::get()->directoryExists("../Data")) {
        sgl::AppSettings::get()->setDataDirectory(DATA_PATH);
    }
#endif
    sgl::AppSettings::get()->initializeDataDirectory();

    std::string iconPath = sgl::AppSettings::get()->getDataDirectory() + "Fonts/icon_256.png";
    sgl::AppSettings::get()->loadApplicationIconFromFile(iconPath);

#ifdef __linux__
    setFileDescriptorLimit();
#endif

    // Load the file containing the app settings
    ImVector<ImWchar> fontRanges;
    if (isHeadlessMode) {
        sgl::AppSettings::get()->setSaveSettings(false);
        sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);
    } else {
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

        ImFontGlyphRangesBuilder builder;
        builder.AddChar(L'\u03BB'); // lambda
        builder.AddChar(L'\u03C3'); // sigma
        builder.BuildRanges(&fontRanges);
        bool useMultiViewport = false;
        if (sgl::AppSettings::get()->getSettings().getValueOpt("useDockSpaceMode", useMultiViewport)) {
            useMultiViewport = !useMultiViewport;
        }
        sgl::AppSettings::get()->setLoadGUI(fontRanges.Data, true, useMultiViewport);
    }

    sgl::AppSettings::get()->setRenderSystem(sgl::RenderSystem::VULKAN);

    sgl::Window* window = nullptr;
    std::vector<const char*> optionalDeviceExtensions;
    if (isHeadlessMode) {
        sgl::AppSettings::get()->createHeadless();
    } else {
#ifdef SUPPORT_OPENGL
        /*
         * OpenGL interop is optionally supported for rendering with NanoVG.
         * For this, we need to enable a few instance and device extensions.
         * We need to do this before we know whether we were able to successfully create the OpenGL context,
         * as we need a Vulkan device for matching an OpenGL context if EGL is supported.
         */
        sgl::AppSettings::get()->enableVulkanOffscreenOpenGLContextInteropSupport();
#endif

        window = sgl::AppSettings::get()->createWindow();

#ifdef SUPPORT_CUDA_INTEROP
        optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
#endif
#ifdef SUPPORT_OPENGL
        if (sgl::AppSettings::get()->getInstanceSupportsVulkanOpenGLInterop()) {
            std::vector<const char*> interopDeviceExtensions =
                    sgl::AppSettings::get()->getVulkanOpenGLInteropDeviceExtensions();
            for (const char* extensionName : interopDeviceExtensions) {
                bool foundExtension = false;
                for (size_t i = 0; i < optionalDeviceExtensions.size(); i++) {
                    if (strcmp(extensionName, optionalDeviceExtensions.at(i)) == 0) {
                        foundExtension = true;
                        break;
                    }
                }
                if (!foundExtension) {
                    optionalDeviceExtensions.push_back(extensionName);
                }
            }
        }
#endif
        //optionalDeviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
    }
    optionalDeviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

    sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
    if (isHeadlessMode) {
        sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallbackHeadless);
    }
    auto* device = new sgl::vk::Device;

    sgl::vk::DeviceFeatures requestedDeviceFeatures{};
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.sampleRateShading = VK_TRUE; // For MSAA.
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.geometryShader = VK_TRUE; // For Skia (if enabled).
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.dualSrcBlend = VK_TRUE; // For Skia (if enabled).
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.logicOp = VK_TRUE; // For VKVG (if enabled).
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.fillModeNonSolid = VK_TRUE; // For VKVG (if enabled).
    requestedDeviceFeatures.optionalEnableShaderDrawParametersFeatures = true; // For deferred shading.
    requestedDeviceFeatures.requestedPhysicalDeviceFeatures.fragmentStoresAndAtomics = VK_TRUE;
    // For transfer function optimization with 64-bit accuracy.
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderInt64 = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderFloat64 = VK_TRUE;
    // For ensemble combination when using Vulkan-CUDA interop with PyTorch.
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalPhysicalDeviceFeatures.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.runtimeDescriptorArray = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    // For VMLP.
    requestedDeviceFeatures.optionalVulkan12Features.shaderFloat16 = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan11Features.storageBuffer16BitAccess = VK_TRUE;
    requestedDeviceFeatures.optionalVulkan12Features.vulkanMemoryModel = VK_TRUE; // For cooperative matrices.
#ifdef VK_VERSION_1_3
    requestedDeviceFeatures.optionalVulkan13Features.subgroupSizeControl = VK_TRUE;
#endif
#ifdef VK_NV_cooperative_matrix
    optionalDeviceExtensions.push_back(VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME);
#endif
#ifdef VK_KHR_cooperative_matrix
    optionalDeviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#endif

    if (isHeadlessMode) {
        device->createDeviceHeadless(
                instance, {
                        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
                },
                optionalDeviceExtensions, requestedDeviceFeatures);
    } else {
        device->createDeviceSwapchain(
                instance, window, {
                        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
                },
                optionalDeviceExtensions, requestedDeviceFeatures);
    }

    sgl::OffscreenContext* offscreenContext = nullptr;
    if (!isHeadlessMode) {
#ifdef SUPPORT_OPENGL
        sgl::OffscreenContextParams params{};
#ifdef USE_ZINK
        params.tryUseZinkIfAvailable = true;
        setenv("__GLX_VENDOR_LIBRARY_NAME", "mesa", 0);
        setenv("MESA_LOADER_DRIVER_OVERRIDE", "zink", 0);
        setenv("GALLIUM_DRIVER", "zink", 0);
#endif
        offscreenContext = sgl::createOffscreenContext(device, params, false);
        if (offscreenContext && offscreenContext->getIsInitialized()) {
            //offscreenContext->makeCurrent(); //< This is called by createOffscreenContext to check interop extensions.
            sgl::AppSettings::get()->setOffscreenContext(offscreenContext);
        }
#endif
        sgl::vk::Swapchain* swapchain = new sgl::vk::Swapchain(device);
        swapchain->create(window);
        sgl::AppSettings::get()->setSwapchain(swapchain);
    }

    sgl::AppSettings::get()->setPrimaryDevice(device);
    sgl::AppSettings::get()->initializeSubsystems();
    if (useCustomShaderCompilerBackend) {
        sgl::vk::ShaderManager->setShaderCompilerBackend(shaderCompilerBackend);
    }

    if (!isHeadlessMode) {
        auto app = new MainApp();
        if (useReplicabilityStampMode) {
            app->setUseReplicabilityStampMode();
        }
        app->run();
        delete app;
    }

    if (useSamplingMode) {
        if (dataSetPath.empty()) {
            dataSetPath = sgl::FileUtils::get()->getUserDirectory() + "datasets/Necker/nc/necker_t5_tk_u.nc";
            //dataSetPath = sgl::FileUtils::get()->getUserDirectory() + "datasets/Necker/nc/necker_t5_e100_tk.nc";
        }
        if (testIdx < 0) {
            testIdx = 0;
        }
        runSamplingTests(dataSetPath, testIdx);
    }

    sgl::AppSettings::get()->release();

#ifdef SUPPORT_OPENGL
    if (offscreenContext) {
        sgl::destroyOffscreenContext(offscreenContext);
        offscreenContext = nullptr;
    }
#endif

    return 0;
}
