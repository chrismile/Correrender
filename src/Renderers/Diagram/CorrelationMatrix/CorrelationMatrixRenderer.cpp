/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Widgets/DataView.hpp"
#include "Widgets/ViewManager.hpp"
#include "Utils/InternalState.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/Similarity.hpp"
#include "Volume/VolumeData.hpp"
#include "CorrelationMatrixChart.hpp"
#include "CorrelationMatrixRenderer.hpp"

CorrelationMatrixRenderer::CorrelationMatrixRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_CORRELATION_MATRIX)], viewManager) {
}

CorrelationMatrixRenderer::~CorrelationMatrixRenderer() {
    parentDiagram = {};
}

void CorrelationMatrixRenderer::initialize() {
    Renderer::initialize();

    parentDiagram = std::make_shared<CorrelationMatrixChart>();
    parentDiagram->setRendererVk(renderer);
    parentDiagram->initialize();
    if (volumeData) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
        parentDiagram->setColorMap(colorMap);
    }
}

void CorrelationMatrixRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;

    if (isNewData || _volumeData->getCurrentTimeStepIdx() != cachedTimeStepIdx
            || _volumeData->getCurrentEnsembleIdx() != cachedEnsembleIdx) {
        cachedTimeStepIdx = _volumeData->getCurrentTimeStepIdx();
        cachedEnsembleIdx = _volumeData->getCurrentEnsembleIdx();
        recomputeCorrelationMatrix();
    }

    if (isNewData) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
        parentDiagram->setColorMap(colorMap);
    }
}

void CorrelationMatrixRenderer::recomputeCorrelationMatrix() {
    const std::vector<std::string>& fieldNames = volumeData->getFieldNamesBase(FieldType::SCALAR);
    const auto numFields = int(fieldNames.size());
    std::shared_ptr<CorrelationMatrix> similarityMatrix = std::make_shared<SymmetricCorrelationMatrix>(numFields);

    float minCorrelationValueGlobal = 0.0f;
    float maxCorrelationValueGlobal = 0.0f;
    for (int fieldIdx0 = 0; fieldIdx0 < numFields; fieldIdx0++) {
        for (int fieldIdx1 = fieldIdx0 + 1; fieldIdx1 < numFields; fieldIdx1++) {
            float similarityMetricNumber;
            float maxCorrelationValueLocal = 0.0f;
            if (useFieldAccuracyDouble) {
                similarityMetricNumber = computeFieldSimilarity<double>(
                        volumeData.get(), fieldIdx0, fieldIdx1, correlationMeasureType, maxCorrelationValueLocal,
                        useAllTimeSteps, useAllEnsembleMembers);
            } else {
                similarityMetricNumber = computeFieldSimilarity<float>(
                        volumeData.get(), fieldIdx0, fieldIdx1, correlationMeasureType, maxCorrelationValueLocal,
                        useAllTimeSteps, useAllEnsembleMembers);
            }
            maxCorrelationValueGlobal = std::max(maxCorrelationValueGlobal, maxCorrelationValueLocal);
            similarityMatrix->set(fieldIdx0, fieldIdx1, similarityMetricNumber);
        }
    }

    //for (int fieldIdx = 0; fieldIdx < numFields; fieldIdx++) {
    //    similarityMatrix->set(fieldIdx, fieldIdx, maxCorrelationValueGlobal);
    //}

    if (correlationMeasureType == CorrelationMeasureType::PEARSON
            || correlationMeasureType == CorrelationMeasureType::SPEARMAN
            || correlationMeasureType == CorrelationMeasureType::KENDALL) {
        minCorrelationValueGlobal = -maxCorrelationValueGlobal;
    }

    parentDiagram->setMatrixData(
            correlationMeasureType, fieldNames, similarityMatrix,
            {minCorrelationValueGlobal, maxCorrelationValueGlobal});

    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void CorrelationMatrixRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        // TODO
    }
}

void CorrelationMatrixRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    if (viewIdx == diagramViewIdx) {
        recreateDiagramSwapchain();
    }
}

void CorrelationMatrixRenderer::recreateDiagramSwapchain(int diagramIdx) {
    SceneData* sceneData = viewManager->getViewSceneData(diagramViewIdx);
    if (!(*sceneData->sceneTexture)) {
        return;
    }
    parentDiagram->setBlitTargetVk(
            (*sceneData->sceneTexture)->getImageView(),
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    if (alignWithParentWindow) {
        parentDiagram->setBlitTargetSupersamplingFactor(
                viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
        parentDiagram->updateSizeByParent();
    }
    reRenderTriggeredByDiagram = true;
}

void CorrelationMatrixRenderer::update(float dt, bool isMouseGrabbed) {
    reRenderTriggeredByDiagram = false;
    parentDiagram->setIsMouseGrabbedByParent(isMouseGrabbed);
    parentDiagram->update(dt);
    if (parentDiagram->getNeedsReRender() && !reRenderViewArray.empty()) {
        for (int viewIdx = 0; viewIdx < int(viewVisibilityArray.size()); viewIdx++) {
            if (viewVisibilityArray.at(viewIdx)) {
                reRenderViewArray.at(viewIdx) = true;
            }
        }
        reRenderTriggeredByDiagram = true;
    }
    //isMouseGrabbed |= parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui();
}

void CorrelationMatrixRenderer::onHasMoved(uint32_t viewIdx) {
}

bool CorrelationMatrixRenderer::getHasGrabbedMouse() const {
    if (parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui()) {
        return true;
    }
    return false;
}

void CorrelationMatrixRenderer::setClearColor(const sgl::Color& clearColor) {
    parentDiagram->setClearColor(clearColor);
    reRenderTriggeredByDiagram = true;
}

void CorrelationMatrixRenderer::renderViewImpl(uint32_t viewIdx) {
    if (viewIdx != diagramViewIdx) {
        return;
    }

    SceneData* sceneData = viewManager->getViewSceneData(viewIdx);
    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);

    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        parentDiagram->setImGuiWindowOffset(int(pos.x), int(pos.y));
    } else {
        parentDiagram->setImGuiWindowOffset(0, 0);
    }
    if (reRenderTriggeredByDiagram || parentDiagram->getIsFirstRender()) {
        parentDiagram->render();
    }
    parentDiagram->setBlitTargetSupersamplingFactor(viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
    parentDiagram->blitToTargetVk();
}

void CorrelationMatrixRenderer::renderViewPreImpl(uint32_t viewIdx) {
    if ((viewIdx == diagramViewIdx) && alignWithParentWindow) {
        return;
    }
}

void CorrelationMatrixRenderer::renderViewPostOpaqueImpl(uint32_t viewIdx) {
    if (viewIdx == diagramViewIdx && alignWithParentWindow) {
        return;
    }
}

void CorrelationMatrixRenderer::addViewImpl(uint32_t viewIdx) {
}

bool CorrelationMatrixRenderer::adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx) {
    if (diagramViewIdx >= viewIdx && diagramViewIdx != 0) {
        if (diagramViewIdx != 0) {
            diagramViewIdx--;
            return true;
        } else if (viewManager->getNumViews() > 0) {
            diagramViewIdx++;
            return true;
        }
    }
    return false;
}

void CorrelationMatrixRenderer::removeViewImpl(uint32_t viewIdx) {
    bool diagramViewIdxChanged = false;
    diagramViewIdxChanged |= adaptIdxOnViewRemove(viewIdx, diagramViewIdx);
    if (diagramViewIdxChanged) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
        recreateDiagramSwapchain();
    }
}

void CorrelationMatrixRenderer::renderDiagramViewSelectionGui(
        sgl::PropertyEditor& propertyEditor, const std::string& name, uint32_t& diagramViewIdx) {
    std::string textDefault = "View " + std::to_string(diagramViewIdx + 1);
    if (propertyEditor.addBeginCombo(name, textDefault, ImGuiComboFlags_NoArrowButton)) {
        for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
            std::string text = "View " + std::to_string(viewIdx + 1);
            bool showInView = diagramViewIdx == uint32_t(viewIdx);
            if (ImGui::Selectable(
                    text.c_str(), &showInView, ImGuiSelectableFlags_::ImGuiSelectableFlags_None)) {
                diagramViewIdx = uint32_t(viewIdx);
                reRender = true;
                reRenderTriggeredByDiagram = true;
                recreateDiagramSwapchain();
            }
            if (showInView) {
                ImGui::SetItemDefaultFocus();
            }
        }
        propertyEditor.addEndCombo();
    }
}

void CorrelationMatrixRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    renderDiagramViewSelectionGui(propertyEditor, "Diagram View", diagramViewIdx);

    if (volumeData && volumeData->getTimeStepCount() > 1) {
        if (propertyEditor.addCheckbox("Use All Time Steps", &useAllTimeSteps)) {
            recomputeCorrelationMatrix();
        }
    }
    if (volumeData && volumeData->getEnsembleMemberCount() > 1) {
        if (propertyEditor.addCheckbox("Use All Ensemble Members", &useAllEnsembleMembers)) {
            recomputeCorrelationMatrix();
        }
    }

    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        recomputeCorrelationMatrix();
    }
    if (propertyEditor.addCombo(
            "Accuracy", (int*)&useFieldAccuracyDouble, FIELD_ACCURACY_NAMES, 2)) {
        recomputeCorrelationMatrix();
    }

    if (propertyEditor.addCombo(
            "Color Map", (int*)&colorMap, DIAGRAM_COLOR_MAP_NAMES,
            IM_ARRAYSIZE(DIAGRAM_COLOR_MAP_NAMES))) {
        parentDiagram->setColorMap(colorMap);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCheckbox("Align with Window", &alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (parentDiagram->renderGuiPropertyEditor(propertyEditor)) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
}

void CorrelationMatrixRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);

    bool diagramChanged = false;
    diagramChanged |= settings.getValueOpt("diagram_view", diagramViewIdx);
    if (diagramChanged) {
        recreateDiagramSwapchain();
    }
    if (settings.getValueOpt("align_with_parent_window", alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
    }

    if (settings.getValueOpt("use_all_time_steps", useAllTimeSteps)) {
        recomputeCorrelationMatrix();
    }
    if (settings.getValueOpt("use_all_ensemble_members", useAllEnsembleMembers)) {
        recomputeCorrelationMatrix();
    }

    std::string correlationMeasureTypeName;
    if (settings.getValueOpt("correlation_measure_type", correlationMeasureTypeName)) {
        for (int i = 0; i < IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_IDS); i++) {
            if (correlationMeasureTypeName == CORRELATION_MEASURE_TYPE_IDS[i]) {
                correlationMeasureType = CorrelationMeasureType(i);
                break;
            }
        }
        recomputeCorrelationMatrix();
    }
    if (settings.getValueOpt("use_field_accuracy_double", useFieldAccuracyDouble)) {
        recomputeCorrelationMatrix();
    }

    std::string colorMapName;
    std::string colorMapIdx = "color_map";
    if (settings.getValueOpt(colorMapIdx.c_str(), colorMapName)) {
        for (int i = 0; i < IM_ARRAYSIZE(DIAGRAM_COLOR_MAP_NAMES); i++) {
            if (colorMapName == DIAGRAM_COLOR_MAP_NAMES[i]) {
                colorMap = DiagramColorMap(i);
                parentDiagram->setColorMap(colorMap);
                break;
            }
        }
    }

    dirty = true;
    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void CorrelationMatrixRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);

    settings.addKeyValue("diagram_view", diagramViewIdx);
    settings.addKeyValue("align_with_parent_window", alignWithParentWindow);
    settings.addKeyValue("use_all_time_steps", useAllTimeSteps);
    settings.addKeyValue("use_all_ensemble_members", useAllEnsembleMembers);
    settings.addKeyValue(
            "correlation_measure_type", CORRELATION_MEASURE_TYPE_IDS[int(correlationMeasureType)]);
    settings.addKeyValue("use_field_accuracy_double", useFieldAccuracyDouble);
    settings.addKeyValue("color_map", DIAGRAM_COLOR_MAP_NAMES[int(colorMap)]);

    // No vector widget settings for now.
}
