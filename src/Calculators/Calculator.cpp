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
#include "Renderers/Renderer.hpp"
#include "Calculator.hpp"

void Calculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    if (isNewData && volumeData) {
        dirty = true;
    }
    volumeData = _volumeData;
}

bool Calculator::getIsDirty() {
    bool tmp = dirty;
    dirty = false;
    return tmp;
}

bool Calculator::getIsDirtyDontReset() {
    return dirty;
}

bool Calculator::getHasNameChanged() {
    bool tmp = hasNameChanged;
    hasNameChanged = false;
    return tmp;
}

bool Calculator::getHasFilterDeviceChanged() {
    bool tmp = hasFilterDeviceChanged;
    hasFilterDeviceChanged = false;
    return tmp;
}

bool Calculator::getShallRemoveCalculator() {
    return shallRemoveCalculator;
}

void Calculator::renderGui(sgl::PropertyEditor& propertyEditor) {
    std::string nodeName = "Calculator (" + getOutputFieldName() + ")###calculator" + std::to_string(calculatorId);
    bool beginNode = propertyEditor.beginNode(nodeName);
    ImGui::SameLine();
    float indentWidth = ImGui::GetContentRegionAvail().x;
    ImGui::Indent(indentWidth);
    std::string buttonName = "X###x_calculator" + std::to_string(calculatorId);
    if (ImGui::Button(buttonName.c_str())) {
        shallRemoveCalculator = true;
    }
    ImGui::Unindent(indentWidth);
    if (beginNode) {
        const auto& calculatorRenderer = getCalculatorRenderer();
        renderGuiImpl(propertyEditor);
        if (calculatorRenderer) {
            calculatorRenderer->renderGui(propertyEditor);
        }
        propertyEditor.endNode();
    }
}
