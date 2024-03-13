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

#include <random>

#include <Utils/Parallel/Reduction.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "bhtsne/tsne.h"

#include "Widgets/DataView.hpp"
#include "Widgets/ViewManager.hpp"
#include "Utils/InternalState.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/Similarity.hpp"
#include "Volume/VolumeData.hpp"
#include "DistributionSimilarityChart.hpp"
#include "DistributionSimilarityRenderer.hpp"

DistributionSimilarityRenderer::DistributionSimilarityRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_DISTRIBUTION_SIMILARITY)], viewManager) {
}

DistributionSimilarityRenderer::~DistributionSimilarityRenderer() {
    parentDiagram = {};
}

void DistributionSimilarityRenderer::initialize() {
    Renderer::initialize();

    parentDiagram = std::make_shared<DistributionSimilarityChart>();
    parentDiagram->setRendererVk(renderer);
    parentDiagram->initialize();
    if (volumeData) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
        //parentDiagram->setColorMap(colorMap);
    }
}

void DistributionSimilarityRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;

    if (isNewData) {
        recomputeCorrelationMatrix();
    }

    if (isNewData) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setClearColor(viewManager->getClearColor());
        //parentDiagram->setColorMap(colorMap);
    }
}

void DistributionSimilarityRenderer::recomputeCorrelationMatrix() {
    //const std::vector<std::string>& fieldNames = volumeData->getFieldNamesBase(FieldType::SCALAR);
    //const auto numFields = int(fieldNames.size());

    std::mt19937 generator(17);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    int numVectors = 10000;
    //int numFeatures = 17 * 17 * 17;
    int numFeatures = 100;
    auto* featureVectorArray = new double[numFeatures * numVectors];
    auto* outputPoints = new double[2 * numVectors];
    for (int ptIdx = 0; ptIdx < numVectors; ptIdx++) {
        for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
            featureVectorArray[ptIdx * numFeatures + featureIdx] = distribution(generator);
        }
    }

    float perplexity = 30.0f;
    float theta = 0.5f;
    int randomSeed = 17; // -1 for pseudo-random
    int maxIter = 500;
    int stopLyingIter = 0;
    int momSwitchIter = 700;
    TSNE::run(
            featureVectorArray, numVectors, numFeatures, outputPoints, 2,
            double(perplexity), double(theta), randomSeed, false, maxIter, stopLyingIter, momSwitchIter);

    std::vector<glm::vec2> points(numVectors);
    for (int ptIdx = 0; ptIdx < numVectors; ptIdx++) {
        points.at(ptIdx) = glm::vec2(float(featureVectorArray[ptIdx * 2]), float(featureVectorArray[ptIdx * 2 + 1]));
    }
    parentDiagram->setPointData(points);
    auto bbData = sgl::reduceVec2ArrayAabb(points);
    //auto bbRender = bbData;
    //bbRender.min -= glm::vec2(bbData.getExtent() * 0.02f);
    //bbRender.min += glm::vec2(bbData.getExtent() * 0.02f);
    parentDiagram->setBoundingBox(bbData);
    delete[] featureVectorArray;
    delete[] outputPoints;

    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void DistributionSimilarityRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        // TODO
    }
}

void DistributionSimilarityRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    if (viewIdx == diagramViewIdx) {
        recreateDiagramSwapchain();
    }
}

void DistributionSimilarityRenderer::recreateDiagramSwapchain(int diagramIdx) {
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

void DistributionSimilarityRenderer::update(float dt, bool isMouseGrabbed) {
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

void DistributionSimilarityRenderer::onHasMoved(uint32_t viewIdx) {
}

bool DistributionSimilarityRenderer::getHasGrabbedMouse() const {
    if (parentDiagram->getIsMouseGrabbed() || parentDiagram->getIsMouseOverDiagramImGui()) {
        return true;
    }
    return false;
}

void DistributionSimilarityRenderer::setClearColor(const sgl::Color& clearColor) {
    parentDiagram->setClearColor(clearColor);
    reRenderTriggeredByDiagram = true;
}

void DistributionSimilarityRenderer::renderViewImpl(uint32_t viewIdx) {
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

void DistributionSimilarityRenderer::renderViewPreImpl(uint32_t viewIdx) {
    if ((viewIdx == diagramViewIdx) && alignWithParentWindow) {
        return;
    }
}

void DistributionSimilarityRenderer::renderViewPostOpaqueImpl(uint32_t viewIdx) {
    if (viewIdx == diagramViewIdx && alignWithParentWindow) {
        return;
    }
}

void DistributionSimilarityRenderer::addViewImpl(uint32_t viewIdx) {
}

bool DistributionSimilarityRenderer::adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx) {
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

void DistributionSimilarityRenderer::removeViewImpl(uint32_t viewIdx) {
    bool diagramViewIdxChanged = false;
    diagramViewIdxChanged |= adaptIdxOnViewRemove(viewIdx, diagramViewIdx);
    if (diagramViewIdxChanged) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
        recreateDiagramSwapchain();
    }
}

void DistributionSimilarityRenderer::renderDiagramViewSelectionGui(
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

void DistributionSimilarityRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    renderDiagramViewSelectionGui(propertyEditor, "Diagram View", diagramViewIdx);

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

    if (parentDiagram->renderGuiPropertyEditor(propertyEditor)) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
}

void DistributionSimilarityRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);

    bool diagramChanged = false;
    diagramChanged |= settings.getValueOpt("diagram_view", diagramViewIdx);
    if (diagramChanged) {
        recreateDiagramSwapchain();
    }
    if (settings.getValueOpt("align_with_parent_window", alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
    }

    if (settings.getValueOpt("point_size", pointSize)) {
        parentDiagram->setPointRadius(pointSize);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
    glm::vec4 pointColorVec = pointColor.getFloatColorRGBA();
    if (settings.getValueOpt("point_color", pointColorVec)) {
        pointColor = sgl::colorFromVec4(pointColorVec);
        parentDiagram->setPointColor(pointColor);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    dirty = true;
    reRender = true;
    reRenderTriggeredByDiagram = true;
}

void DistributionSimilarityRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);

    settings.addKeyValue("diagram_view", diagramViewIdx);
    settings.addKeyValue("align_with_parent_window", alignWithParentWindow);
    settings.addKeyValue("point_size", pointSize);
    settings.addKeyValue("point_color", pointColor.getFloatColorRGBA());

    // No vector widget settings for now.
}
