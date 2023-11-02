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

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Input/Keyboard.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Widgets/DataView.hpp"
#include "Widgets/ViewManager.hpp"
#include "Utils/InternalState.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/PointPicker.hpp"
#include "Calculators/ReferencePointSelectionRenderer.hpp"
#include "Volume/VolumeData.hpp"
#include "ScatterPlotChart.hpp"
#include "ScatterPlotRenderer.hpp"

ScatterPlotRenderer::ScatterPlotRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_SCATTER_PLOT)], viewManager) {
    std::vector<glm::vec3> sphereVertexPositions;
    std::vector<glm::vec3> sphereVertexNormals;
    std::vector<uint32_t> sphereIndices;
    getSphereSurfaceRenderData(
            glm::vec3(0,0,0), 1.0f, 32, 32,
            sphereVertexPositions, sphereVertexNormals, sphereIndices);

    sgl::vk::Device* device = renderer->getDevice();
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * sphereVertexPositions.size(), sphereVertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexNormalBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * sphereVertexNormals.size(), sphereVertexNormals.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    indexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * sphereIndices.size(), sphereIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    for (int i = 0; i < 2; i++) {
        auto refPosSetter = [this, i](const glm::vec3& refPos) {
            this->setReferencePoint(i, refPos);
        };
        auto viewUsedIndexQuery = [this](int mouseHoverWindowIndex) -> bool {
            return mouseHoverWindowIndex == int(diagramViewIdx);
        };
        pointPicker[i] = std::make_shared<PointPicker>(
                viewManager, fixPickingZPlane, refPosSetter, viewUsedIndexQuery);
    }
    pointPicker[0]->setMouseButton(1);
    pointPicker[1]->setMouseButton(3);
}

ScatterPlotRenderer::~ScatterPlotRenderer() {
    volumeData->releaseScalarField(this, fieldIdx0);
    if (fieldIdx0 != fieldIdx1) {
        volumeData->releaseScalarField(this, fieldIdx1);
    }
    parentDiagram = {};
}

void ScatterPlotRenderer::initialize() {
    Renderer::initialize();

    parentDiagram = std::make_shared<ScatterPlotChart>();
    parentDiagram->setRendererVk(renderer);
    parentDiagram->initialize();
    if (volumeData) {
        parentDiagram->setVolumeData(volumeData, true);
        parentDiagram->setIsEnsembleMode(isEnsembleMode);
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
        parentDiagram->setUseGlobalMinMax(useGlobalMinMax);
    }
}

void ScatterPlotRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;

    int es = _volumeData->getEnsembleMemberCount();
    int ts = _volumeData->getTimeStepCount();
    if (isEnsembleMode && es <= 1 && ts > 1) {
        isEnsembleMode = false;
    } else if (!isEnsembleMode && ts <= 1 && es > 1) {
        isEnsembleMode = true;
    }

    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);

    if (isNewData) {
        reRenderTriggeredByDiagram = true;
        auto refPoint = glm::ivec3(
                volumeData->getGridSizeX() / 2, volumeData->getGridSizeY() / 2, volumeData->getGridSizeZ() / 2);
        setReferencePoint(0, refPoint);
        setReferencePoint(1, refPoint);
    }

    for (int i = 0; i < 2; i++) {
        for (auto& referencePointSelectionRasterPass : referencePointSelectionRasterPasses[i]) {
            referencePointSelectionRasterPass->setVolumeData(volumeData.get(), isNewData);
        }
    }

    for (int i = 0; i < 2; i++) {
        pointPicker[i]->setVolumeData(volumeData.get(), isNewData);
    }

    parentDiagram->setVolumeData(volumeData, isNewData);
    if (isNewData) {
        int standardFieldIdx = volumeData->getStandardScalarFieldIdx();
        fieldIdx0 = standardFieldIdx;
        fieldIdx1 = standardFieldIdx;
        fieldName0 = fieldNames.at(fieldIdx0);
        fieldName1 = fieldNames.at(fieldIdx1);
        volumeData->acquireScalarField(this, standardFieldIdx);
        parentDiagram->setField0(fieldIdx0, fieldName0);
        parentDiagram->setField1(fieldIdx1, fieldName1);
        parentDiagram->setIsEnsembleMode(isEnsembleMode);
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
    } else {
        parentDiagram->setVolumeData(volumeData, isNewData);
    }
}

void ScatterPlotRenderer::setReferencePoint(int idx, const glm::ivec3& referencePoint) {
    if (refPos[idx] != referencePoint) {
        glm::ivec3 maxCoord(
                volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
        refPos[idx] = referencePoint;
        refPos[idx] = glm::clamp(refPos[idx], glm::ivec3(0), maxCoord);
        for (auto& pass : referencePointSelectionRasterPasses[idx]) {
            pass->setReferencePosition(refPos[idx]);
        }
        parentDiagram->setReferencePoints(refPos[0], refPos[1]);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
}

void ScatterPlotRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (fieldIdx0 == fieldIdx) {
            fieldIdx0 = 0;
        } else if (fieldIdx0 > fieldIdx) {
            fieldIdx0--;
        }
    }
}

void ScatterPlotRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    if (viewIdx == diagramViewIdx) {
        recreateDiagramSwapchain();
    }

    for (int i = 0; i < 2; i++) {
        referencePointSelectionRasterPasses[i].at(viewIdx)->recreateSwapchain(width, height);
    }
    /*for (int idx = 0; idx < 2; idx++) {
         domainOutlineRasterPasses[idx].at(viewIdx)->recreateSwapchain(width, height);
         domainOutlineComputePasses[idx].at(viewIdx)->recreateSwapchain(width, height);
         selectionBoxRasterPasses[idx].at(viewIdx)->recreateSwapchain(width, height);
         shadowRectRasterPasses[idx].at(viewIdx)->recreateSwapchain(width, height);
     }
     connectingLineRasterPass.at(viewIdx)->recreateSwapchain(width, height);*/
}

void ScatterPlotRenderer::recreateDiagramSwapchain(int diagramIdx) {
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

void ScatterPlotRenderer::update(float dt, bool isMouseGrabbed) {
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
    isMouseGrabbed |= parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui();

    if (!isMouseGrabbed) {
        for (int i = 0; i < 2; i++) {
            pointPicker[i]->update(dt);
        }
    }
}

void ScatterPlotRenderer::onHasMoved(uint32_t viewIdx) {
}

bool ScatterPlotRenderer::getHasGrabbedMouse() const {
    if (parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui()) {
        return true;
    }
    return false;
}

void ScatterPlotRenderer::setClearColor(const sgl::Color& clearColor) {
    parentDiagram->setClearColor(clearColor);
    reRenderTriggeredByDiagram = true;
}

void ScatterPlotRenderer::renderViewImpl(uint32_t viewIdx) {
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

void ScatterPlotRenderer::renderViewPreImpl(uint32_t viewIdx) {
    if ((viewIdx == diagramViewIdx) && alignWithParentWindow) {
        return;
    }

    for (int i = 0; i < 2; i++) {
        referencePointSelectionRasterPasses[i].at(viewIdx)->render();
    }
}

void ScatterPlotRenderer::renderViewPostOpaqueImpl(uint32_t viewIdx) {
    if (viewIdx == diagramViewIdx && alignWithParentWindow) {
        return;
    }
}

void ScatterPlotRenderer::addViewImpl(uint32_t viewIdx) {
    for (int i = 0; i < 2; i++) {
        auto referencePointSelectionRasterPass = std::make_shared<ReferencePointSelectionRasterPass>(
                renderer, viewManager->getViewSceneData(viewIdx));
        if (volumeData) {
            referencePointSelectionRasterPass->setVolumeData(volumeData.get(), true);
        }
        referencePointSelectionRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer);
        referencePointSelectionRasterPass->setReferencePosition(refPos[i]);
        referencePointSelectionRasterPass->setSphereRadius(0.006f);
        referencePointSelectionRasterPass->setSphereColor(sgl::Color(255, 40, 0).getFloatColorRGBA());
        referencePointSelectionRasterPasses[i].push_back(referencePointSelectionRasterPass);
    }
}

bool ScatterPlotRenderer::adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx) {
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

void ScatterPlotRenderer::removeViewImpl(uint32_t viewIdx) {
    bool diagramViewIdxChanged = false;
    diagramViewIdxChanged |= adaptIdxOnViewRemove(viewIdx, diagramViewIdx);
    if (diagramViewIdxChanged) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
        recreateDiagramSwapchain();
    }

    for (int i = 0; i < 2; i++) {
        referencePointSelectionRasterPasses[i].erase(referencePointSelectionRasterPasses[i].begin() + viewIdx);
    }
}

void ScatterPlotRenderer::renderDiagramViewSelectionGui(
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

void ScatterPlotRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    renderDiagramViewSelectionGui(propertyEditor, "Diagram View", diagramViewIdx);

    if (volumeData && volumeData->getEnsembleMemberCount() > 1 && volumeData->getTimeStepCount() > 1) {
        int modeIdx = isEnsembleMode ? 0 : 1;
        if (propertyEditor.addCombo("Correlation Mode", &modeIdx, CORRELATION_MODE_NAMES, 2)) {
            isEnsembleMode = modeIdx == 0;
            parentDiagram->setIsEnsembleMode(isEnsembleMode);
            dirty = true;
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
    }

    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        int fieldIdxGui0 = fieldIdx0;
        std::string scalarFieldNameEntry = useSameField ? "Scalar Field" : "Scalar Field #1";
        if (propertyEditor.addCombo(scalarFieldNameEntry, &fieldIdxGui0, fieldNames.data(), int(fieldNames.size()))) {
            fieldIdx0 = fieldIdxGui0;
            fieldName0 = fieldNames.at(fieldIdx0);
            parentDiagram->setField0(fieldIdx0, fieldName0);
            if (useSameField) {
                fieldIdx1 = fieldIdx0;
                fieldName1 = fieldName0;
                parentDiagram->setField1(fieldIdx1, fieldName1);
            }
            dirty = true;
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
        int fieldIdxGui1 = fieldIdx1;
        if (!useSameField && propertyEditor.addCombo(
                "Scalar Field #2", &fieldIdxGui1, fieldNames.data(), int(fieldNames.size()))) {
            fieldIdx1 = fieldIdxGui1;
            fieldName1 = fieldNames.at(fieldIdx1);
            parentDiagram->setField1(fieldIdx1, fieldName1);
            dirty = true;
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
        if (propertyEditor.addCheckbox("Use Same Field", &useSameField)) {
            if (useSameField) {
                fieldIdx1 = fieldIdx0;
                fieldName1 = fieldName0;
                parentDiagram->setField1(fieldIdx1, fieldName1);
                dirty = true;
                reRender = true;
                reRenderTriggeredByDiagram = true;
            }
        }
    }

    if (propertyEditor.addSliderFloat("Point Size", &pointSize, 1.0f, 10.0f)) {
        parentDiagram->setPointRadius(pointSize);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    glm::vec4 pointColorVec = pointColor.getFloatColorRGBA();
    if (propertyEditor.addColorEdit4("Point Color", &pointColorVec.x)) {
        pointColor = sgl::colorFromVec4(pointColorVec);
        parentDiagram->setPointColor(pointColor);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCheckbox("Align with Window", &alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (!volumeData || volumeData->getGridSizeZ() > 1) {
        propertyEditor.addCheckbox("Fix Picking Z", &fixPickingZPlane);
    }

    if (propertyEditor.addCheckbox("Global Min/Max", &useGlobalMinMax)) {
        parentDiagram->setUseGlobalMinMax(useGlobalMinMax);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (parentDiagram->renderGuiPropertyEditor(propertyEditor)) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
}

void ScatterPlotRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);

    bool diagramChanged = false;
    diagramChanged |= settings.getValueOpt("diagram_view", diagramViewIdx);
    if (diagramChanged) {
        recreateDiagramSwapchain();
    }
    std::string ensembleModeName;
    if (settings.getValueOpt("correlation_mode", ensembleModeName)) {
        if (ensembleModeName == CORRELATION_MODE_NAMES[0]) {
            isEnsembleMode = true;
        } else {
            isEnsembleMode = false;
        }
        parentDiagram->setIsEnsembleMode(isEnsembleMode);
    }

    settings.getValueOpt("use_same_field", useSameField);

    std::string newFieldName0, newFieldName1;
    bool changedField0 = settings.getValueOpt("field0", newFieldName0);
    bool changedField1 = settings.getValueOpt("field1", newFieldName1);
    if (changedField0 || changedField1) {
        // Remove old selection.
        volumeData->releaseScalarField(this, fieldIdx0);
        if (fieldIdx0 != fieldIdx1) {
            volumeData->releaseScalarField(this, fieldIdx1);
        }

        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        int fieldIdxGui0 = fieldIdx0;
        if (changedField0) {
            fieldIdx0 = fieldIdxGui0;
            fieldName0 = fieldNames.at(fieldIdx0);
            parentDiagram->setField0(fieldIdx0, fieldName0);
            dirty = true;
            reRender = true;
        }
        int fieldIdxGui1 = fieldIdx1;
        if (changedField1) {
            fieldIdx1 = fieldIdxGui1;
            fieldName1 = fieldNames.at(fieldIdx1);
            parentDiagram->setField1(fieldIdx1, fieldName1);
            dirty = true;
            reRender = true;
        }
    }

    if (settings.getValueOpt("align_with_parent_window", alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
    }

    settings.getValueOpt("fix_picking_z", fixPickingZPlane);

    dirty = true;
    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void ScatterPlotRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);

    settings.addKeyValue("context_diagram_view", diagramViewIdx);
    settings.addKeyValue("correlation_mode", CORRELATION_MODE_NAMES[isEnsembleMode ? 0 : 1]);
    settings.addKeyValue("align_with_parent_window", alignWithParentWindow);
    settings.addKeyValue("use_same_field", useSameField);
    settings.addKeyValue("field0", fieldName0);
    settings.addKeyValue("field1", fieldName1);
    settings.addKeyValue("fix_picking_z", fixPickingZPlane);

    // No vector widget settings for now.
}
