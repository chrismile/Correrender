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
#include "Utils/InternalState.hpp"
#include "Volume/VolumeData.hpp"
#include "Widgets/ViewManager.hpp"
#include "RenderingModes.hpp"
#include "Renderer.hpp"

Renderer::Renderer(std::string windowName, ViewManager* viewManager)
        : windowName(std::move(windowName)), viewManager(viewManager), renderer(viewManager->getRenderer()) {
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

bool Renderer::isVisibleInAnyView() {
    for (bool visible : viewVisibilityArray) {
        if (visible) {
            return true;
        }
    }
    return false;
}

void Renderer::renderView(uint32_t viewIdx) {
    if (viewVisibilityArray.at(viewIdx)) {
        renderViewImpl(viewIdx);
    }
}

void Renderer::renderViewPre(uint32_t viewIdx) {
    if (viewVisibilityArray.at(viewIdx)) {
        renderViewPreImpl(viewIdx);
    }
}

void Renderer::renderViewPostOpaque(uint32_t viewIdx) {
    if (viewVisibilityArray.at(viewIdx)) {
        renderViewPostOpaqueImpl(viewIdx);
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
    updateViewComboSelection();
}

void Renderer::removeView(uint32_t viewIdx) {
    viewVisibilityArray.erase(viewVisibilityArray.begin() + viewIdx);
    reRenderViewArray.erase(reRenderViewArray.begin() + viewIdx);
    removeViewImpl(viewIdx);
    updateViewComboSelection();
}

void Renderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
}

void Renderer::updateViewComboSelection() {
    std::vector<std::string> comboSelVec(0);

    for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
        if (viewVisibilityArray.at(viewIdx)) {
            comboSelVec.push_back("View " + std::to_string(viewIdx + 1));
        }
    }

    showInViewComboValue = "";
    for (size_t v = 0; v < comboSelVec.size(); ++v) {
        showInViewComboValue += comboSelVec[v];
        if (comboSelVec.size() > 1 && v + 1 != comboSelVec.size()) {
            showInViewComboValue += ", ";
        }
    }
}

void Renderer::renderGui(sgl::PropertyEditor& propertyEditor) {
    if (viewVisibilityArray.size() <= 1) {
        for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
            std::string text = "Show in View " + std::to_string(viewIdx + 1);
            bool showInView = viewVisibilityArray.at(viewIdx);
            if (propertyEditor.addCheckbox(text, &showInView)) {
                viewVisibilityArray.at(viewIdx) = showInView;
                reRenderViewArray.at(viewIdx) = true;
                updateViewComboSelection();
            }
        }
    } else {
        if (propertyEditor.addBeginCombo(
                "Show in View", showInViewComboValue, ImGuiComboFlags_NoArrowButton)) {
            std::vector<std::string> comboSelVec(0);

            for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
                std::string text = "View " + std::to_string(viewIdx + 1);
                bool showInView = viewVisibilityArray.at(viewIdx);
                if (ImGui::Selectable(
                        text.c_str(), &showInView, ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups)) {
                    viewVisibilityArray.at(viewIdx) = showInView;
                    reRenderViewArray.at(viewIdx) = true;
                }

                if (showInView) {
                    ImGui::SetItemDefaultFocus();
                    comboSelVec.push_back(text);
                }
            }

            showInViewComboValue = "";
            for (size_t v = 0; v < comboSelVec.size(); ++v) {
                showInViewComboValue += comboSelVec[v];
                if (comboSelVec.size() > 1 && v + 1 != comboSelVec.size()) {
                    showInViewComboValue += ", ";
                }
            }

            propertyEditor.addEndCombo();
        }
    }

    renderGuiImpl(propertyEditor);
}

void Renderer::renderGuiOverlay(uint32_t viewIdx) {
}

void Renderer::setSettings(const SettingsMap& settings) {
    std::string viewVisibilityString;
    if (settings.getValueOpt("view_visibility", viewVisibilityString)) {
        for (size_t i = 0; i < viewVisibilityString.size(); i++) {
            bool visible = viewVisibilityString.at(i) != '0';
            if (visible != viewVisibilityArray.at(i)) {
                viewVisibilityArray.at(i) = visible;
                reRenderViewArray.at(i) = true;
            }
        }
        updateViewComboSelection();
    }
}

void Renderer::getSettings(SettingsMap& settings) {
    std::string viewVisibilityString;
    for (auto entry : viewVisibilityArray) {
        viewVisibilityString += entry ? '1' : '0';
    }
    settings.addKeyValue("view_visibility", viewVisibilityString);
}
