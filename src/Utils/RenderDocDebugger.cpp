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

#include <Utils/File/Logfile.hpp>
#include <Input/Keyboard.hpp>

#include "renderdoc_app.h"
#include "RenderDocDebugger.hpp"

#if defined(__linux__)
#include <dlfcn.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define dlsym GetProcAddress
#endif

RenderDocDebugger::RenderDocDebugger() {
#if defined(__linux__)
    renderdocHandle = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (!renderdocHandle) {
        return;
    }
#elif defined(_WIN32)
    renderdocHandle = GetModuleHandleA("renderdoc.dll");
    if (!renderdocHandle) {
        return;
    }
#endif

    auto RENDERDOC_GetAPI =
            (pRENDERDOC_GetAPI)dlsym(renderdocHandle, "RENDERDOC_GetAPI");
    int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void**)&renderdocApi);
    if (ret != 1) {
        sgl::Logfile::get()->writeWarning(
                "RenderDocDebugger::RenderDocDebugger: RenderDoc was found, but initialization failed.");
    }

    sgl::Logfile::get()->writeInfo(
            "RenderDocDebugger::RenderDocDebugger: Initialized RenderDoc debugger successfully.");
    isInitialized = true;
}

RenderDocDebugger::~RenderDocDebugger() {
    if (renderdocHandle) {
#if defined(__linux__)
        dlclose(renderdocHandle);
#elif defined(_WIN32)
        FreeLibrary(renderdocHandle);
#endif
        renderdocHandle = {};
    }
}

void RenderDocDebugger::update() {
    if (isInitialized) {
        if (sgl::Keyboard->keyPressed(SDLK_y) && (sgl::Keyboard->getModifier() & (KMOD_LCTRL | KMOD_RCTRL))) {
            shallCaptureFrame = true;
        }
    }
}

void RenderDocDebugger::startFrameCapture() {
    if (isInitialized && shallCaptureFrame) {
        reinterpret_cast<RENDERDOC_API_1_1_2*>(renderdocApi)->StartFrameCapture(nullptr, nullptr);
        shallCaptureFrame = false;
        isCapturingFrame = true;
    }
}

void RenderDocDebugger::endFrameCapture() {
    if (isInitialized && isCapturingFrame) {
        reinterpret_cast<RENDERDOC_API_1_1_2*>(renderdocApi)->EndFrameCapture(nullptr, nullptr);
        isCapturingFrame = false;
    }
}
