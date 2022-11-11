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

#include <ImGui/Widgets/PropertyEditor.hpp>
#include "Volume/VolumeData.hpp"
#include "Widgets/ViewManager.hpp"
#include "RenderingModes.hpp"
#include "Renderer.hpp"

Renderer::Renderer(
        std::string windowName, ViewManager* viewManager, sgl::TransferFunctionWindow& transferFunctionWindow)
        : windowName(std::move(windowName)), viewManager(viewManager), renderer(viewManager->getRenderer()),
          transferFunctionWindow(transferFunctionWindow) {
}

void Renderer::initialize() {
    // Add events for interaction with line data.
    onTransferFunctionMapRebuiltListenerToken = sgl::EventManager::get()->addListener(
            ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT, [this](const sgl::EventPtr&) {
                this->onTransferFunctionMapRebuilt();
            });

    isInitialized = true;
}

Renderer::~Renderer() {
    if (isInitialized) {
        sgl::EventManager::get()->removeListener(
                ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT, onTransferFunctionMapRebuiltListenerToken);
    }
}

bool Renderer::needsReRender() {
    bool tmp = reRender;
    reRender = false;
    return tmp;
}

bool Renderer::needsReRenderView(uint32_t viewIdx) {
    bool tmp = reRenderViewArray.at(viewIdx);
    reRenderViewArray.at(viewIdx) = false;
    return tmp;
}

bool Renderer::isVisibleInView(uint32_t viewIdx) {
    if (viewIdx >= viewVisibilityArray.size()) {
        return viewIdx == 0;
    }
    return viewVisibilityArray.at(viewIdx);
}

void Renderer::renderView(uint32_t viewIdx) {
    if (viewVisibilityArray.at(viewIdx)) {
        renderViewImpl(viewIdx);
    }
}

void Renderer::addView(uint32_t viewIdx) {
    viewVisibilityArray.resize(viewIdx + 1);
    if (viewIdx == 0) {
        viewVisibilityArray.at(0) = true;
    }
    reRenderViewArray.resize(viewIdx + 1);
    reRenderViewArray.at(viewIdx) = true;
    addViewImpl(viewIdx);
}

void Renderer::removeView(uint32_t viewIdx) {
    viewVisibilityArray.erase(viewVisibilityArray.begin() + viewIdx);
    reRenderViewArray.erase(reRenderViewArray.begin() + viewIdx);
    removeViewImpl(viewIdx);
}

void Renderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
}

void Renderer::renderGui(sgl::PropertyEditor& propertyEditor) {
    for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
        std::string text = "Show in View " + std::to_string(viewIdx + 1);
        bool showInView = viewVisibilityArray.at(viewIdx);
        if (propertyEditor.addCheckbox(text, &showInView)) {
            viewVisibilityArray.at(viewIdx) = showInView;
            reRenderViewArray.at(viewIdx) = true;
        }
    }
    renderGuiImpl(propertyEditor);
}

void Renderer::renderGuiOverlay(uint32_t viewIdx) {
    volumeData->renderGuiOverlay(viewIdx);
}
