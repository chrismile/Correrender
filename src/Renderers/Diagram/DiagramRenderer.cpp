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

#include <iostream>
#include <random>

#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/imgui_custom.h>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Renderers/DomainOutlineRenderer.hpp"
#include "Calculators/CorrelationCalculator.hpp"
#include "Widgets/DataView.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "../RenderingModes.hpp"

#include "ConnectingLineRasterPass.hpp"
#include "RadarBarChart.hpp"
#include "HEBChart.hpp"

#include "DiagramRenderer.hpp"

DiagramRenderer::DiagramRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_DIAGRAM_RENDERER)], viewManager) {
}

DiagramRenderer::~DiagramRenderer() {
    for (auto& selectedScalarField : selectedScalarFields) {
        volumeData->releaseScalarField(this, selectedScalarField.first);
    }
}

void DiagramRenderer::initialize() {
    Renderer::initialize();

    variableNames = {
            "Pressure", "NI_OUT", "NR_OUT", "NS_OUT", "QC", "QG", "QG_OUT", "QH", "QH_OUT", "QI", "QI_OUT",
            "NCCLOUD", "QR", "QR_OUT", "QS", "QS_OUT", "QV", "S", "T", "artificial", "artificial (threshold)",
            "conv_400", "NCGRAUPEL", "conv_600", "dD_rainfrz_gh", "dD_rainfrz_ig", "dT_mult_max", "dT_mult_min",
            "da_HET", "da_ccn_1", "da_ccn_4", "db_HET", "db_ccn_1", "NCHAIL", "db_ccn_3", "db_ccn_4", "dc_ccn_1",
            "dc_ccn_4", "dcloud_c_z", "dd_ccn_1", "dd_ccn_2", "dd_ccn_3", "dd_ccn_4", "dgraupel_a_vel", "NCICE",
            "dgraupel_b_geo", "dgraupel_b_vel", "dgraupel_vsedi_max", "dhail_vsedi_max", "dice_a_f", "dice_a_geo",
            "dice_b_geo", "dice_b_vel", "dice_c_s", "dice_vsedi_max", "NCRAIN", "dinv_z", "dk_r",
            "dp_sat_ice_const_b", "dp_sat_melt", "drain_a_geo", "drain_a_vel", "drain_alpha", "drain_b_geo",
            "drain_b_vel", "drain_beta", "NCSNOW", "drain_c_z", "drain_g1", "drain_g2", "drain_gamma",
            "drain_min_x", "drain_min_x_freezing", "drain_mu", "drain_nu", "drho_vel", "dsnow_a_geo", "NG_OUT",
            "dsnow_b_geo", "dsnow_b_vel", "dsnow_vsedi_max", "mean of artificial", "mean of artificial (threshold)",
            "mean of physical", "mean of physical (high variability)", "physical", "physical (high variability)",
            "time_after_ascent", "NH_OUT", "w", "z"
    };

    {
        std::mt19937 generator(2);
        std::uniform_real_distribution<> dis(0, 1);

        int numVariables = int(variableNames.size());
        int numTimesteps = 10;
        variableValuesTimeDependent.resize(numTimesteps);
        for (int timeStepIdx = 0; timeStepIdx < numTimesteps; timeStepIdx++) {
            std::vector<float>& variableValuesAtTime = variableValuesTimeDependent.at(timeStepIdx);
            variableValuesAtTime.reserve(numVariables);
            for (int varIdx = 1; varIdx <= numVariables; varIdx++) {
                variableValuesAtTime.push_back(float(dis(generator)));
            }
        }
    }

    parentDiagram = std::make_shared<HEBChart>();
    //auto diagram = std::make_shared<RadarBarChart>(true);
    parentDiagram->setRendererVk(renderer);
    parentDiagram->initialize();
    //diagram->setDataTimeDependent(variableNames, variableValuesTimeDependent);
    //diagram->setUseEqualArea(true);
    if (volumeData) {
        parentDiagram->setVolumeData(volumeData, true);
        parentDiagram->setIsEnsembleMode(isEnsembleMode);
        parentDiagram->setCorrelationMeasureType(correlationMeasureType);
        parentDiagram->setBeta(beta);
        parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
        parentDiagram->setLineCountFactor(lineCountFactor);
        parentDiagram->setCurveThickness(curveThickness);
        parentDiagram->setCurveOpacity(curveOpacity);
        parentDiagram->setDiagramRadius(diagramRadius);
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setOpacityByValue(opacityByValue);
        parentDiagram->setColorByValue(colorByValue);
        parentDiagram->setUse2DField(use2dField);
        parentDiagram->setClearColor(viewManager->getClearColor());
    }
    diagrams.push_back(parentDiagram);
}

void DiagramRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
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

    if (isNewData) {
        int xs = volumeData->getGridSizeX();
        int ys = volumeData->getGridSizeY();
        int zs = volumeData->getGridSizeZ();

        float minDelta = std::min(volumeData->getDx(), std::min(volumeData->getDy(), volumeData->getDz()));
        int fx = int(std::round(volumeData->getDx() / minDelta));
        int fy = int(std::round(volumeData->getDy() / minDelta));
        int fz = int(std::round(volumeData->getDz() / minDelta));

        downscalingFactorUniform = fx == fy && fy == fz;

        int dsw = int(std::ceil(std::cbrt(float(xs * ys * zs) / 100.0f)));
        dsw = std::max(dsw, 1);
        if (!sgl::isPowerOfTwo(dsw)) {
            dsw = sgl::nextPowerOfTwo(dsw);
        }
        minDownscalingFactor = std::max(dsw / 2, 1);
        maxDownscalingFactor = dsw * 2;
        if (downscalingFactorUniform) {
            downscalingFactorX = downscalingFactorY = downscalingFactorZ = dsw;
        } else {
            downscalingFactorX = downscalingFactorY = downscalingFactorZ = dsw;
        }
    }

    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    if (fieldNames.size() > scalarFieldSelectionArray.size()) {
        scalarFieldSelectionArray.resize(fieldNames.size());
    }
    updateScalarFieldComboValue();
    for (int selectedFieldIdx = 0; selectedFieldIdx < int(selectedScalarFields.size()); selectedFieldIdx++) {
        auto& selectedScalarField = selectedScalarFields.at(selectedFieldIdx);
        selectedScalarField.second = fieldNames.at(selectedScalarField.first);
    }

    parentDiagram->setVolumeData(volumeData, isNewData);
    if (isNewData) {
        selectedScalarFields.clear();
        int standardFieldIdx = volumeData->getStandardScalarFieldIdx();
        selectedScalarFields.emplace_back(standardFieldIdx, fieldNames.at(standardFieldIdx), DiagramColorMap::VIRIDIS);
        scalarFieldSelectionArray.clear();
        scalarFieldSelectionArray.resize(fieldNames.size());
        scalarFieldSelectionArray.at(standardFieldIdx) = true;
        scalarFieldComboValue = fieldNames.at(standardFieldIdx);
        volumeData->acquireScalarField(this, standardFieldIdx);
        parentDiagram->clearScalarFields();
        for (int selectedFieldIdx = 0; selectedFieldIdx < int(selectedScalarFields.size()); selectedFieldIdx++) {
            auto& selectedScalarField = selectedScalarFields.at(selectedFieldIdx);
            parentDiagram->addScalarField(selectedScalarField.first, selectedScalarField.second);
            parentDiagram->setColorMap(selectedFieldIdx, selectedScalarField.third);
        }
        parentDiagram->setIsEnsembleMode(isEnsembleMode);
        parentDiagram->setCorrelationMeasureType(correlationMeasureType);
        parentDiagram->setBeta(beta);
        parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
        parentDiagram->setLineCountFactor(lineCountFactor);
        parentDiagram->setCurveThickness(curveThickness);
        parentDiagram->setCurveOpacity(curveOpacity);
        parentDiagram->setDiagramRadius(diagramRadius);
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setOpacityByValue(opacityByValue);
        parentDiagram->setColorByValue(colorByValue);
        parentDiagram->setUse2DField(use2dField);
        parentDiagram->setClearColor(viewManager->getClearColor());

        correlationRangeTotal = correlationRange = parentDiagram->getCorrelationRangeTotal();
        cellDistanceRange = cellDistanceRangeTotal = parentDiagram->getCellDistanceRangeTotal();
        parentDiagram->setCorrelationRange(correlationRangeTotal);
        parentDiagram->setCellDistanceRange(cellDistanceRange);
        resetSelections();
    } else {
        for (size_t i = 1; i < diagrams.size(); i++) {
            auto& diagram = diagrams.at(i);
            diagram->setVolumeData(volumeData, isNewData);
        }
    }
}

void DiagramRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        for (auto& diagram : diagrams) {
            diagram->removeScalarField(fieldIdx, true);
        }

        for (size_t i = 0; i < selectedScalarFields.size(); ) {
            if (selectedScalarFields.at(i).first == fieldIdx) {
                volumeData->releaseScalarField(this, selectedScalarFields.at(i).first);
                selectedScalarFields.erase(selectedScalarFields.begin() + ptrdiff_t(i));
                continue;
            } else if (selectedScalarFields.at(i).first > fieldIdx) {
                volumeData->releaseScalarField(this, selectedScalarFields.at(i).first);
                selectedScalarFields.at(i).first--;
                volumeData->acquireScalarField(this, selectedScalarFields.at(i).first);
            }
            i++;
        }
    }
}

void DiagramRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    if (viewIdx == diagramViewIdx) {
        recreateDiagramSwapchain();
    }
    for (int idx = 0; idx < 2; idx++) {
        domainOutlineRasterPasses[idx].at(viewIdx)->recreateSwapchain(width, height);
        domainOutlineComputePasses[idx].at(viewIdx)->recreateSwapchain(width, height);
    }
    connectingLineRasterPass.at(viewIdx)->recreateSwapchain(width, height);
}

void DiagramRenderer::recreateDiagramSwapchain(int diagramIdx) {
    SceneData* sceneData = viewManager->getViewSceneData(diagramViewIdx);
    for (size_t idx = 0; idx < diagrams.size(); idx++) {
        if (diagramIdx >= 0 && diagramIdx != int(idx)) {
            continue;
        }
        auto& diagram = diagrams.at(idx);
        diagram->setBlitTargetVk(
                (*sceneData->sceneTexture)->getImageView(),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (alignWithParentWindow) {
            diagram->updateSizeByParent();
        }
    }
}

void DiagramRenderer::resetSelections(int idx) {
    bool isNewDiagram = idx == int(diagrams.size()) && idx > 0;
    selectedRegionStack.resize(idx);
    if (idx > 0) {
        selectedRegionStack.at(idx - 1) = diagrams.at(idx - 1)->getFocusSelection();
    }
    diagrams.resize(idx + 1);
    if (isNewDiagram) {
        if (!diagrams.at(idx)) {
            auto diagram = std::make_shared<HEBChart>();
            diagram->setRendererVk(renderer);
            diagram->initialize();
            diagrams.at(idx) = diagram;
            recreateDiagramSwapchain(idx);
        }
    }
    if (idx > 0) {
        auto diagram = diagrams.at(idx);
        if (volumeData) {
            diagram->setVolumeData(volumeData, true);
            int dfx = sgl::iceil(downscalingFactorX, 1 << 2 * idx);
            int dfy = sgl::iceil(downscalingFactorY, 1 << 2 * idx);
            int dfz = sgl::iceil(downscalingFactorZ, 1 << 2 * idx);
            diagram->setDownscalingFactors(dfx, dfy, dfz);
            diagram->setRegions(selectedRegionStack.at(idx - 1));
            diagram->clearScalarFields();
            for (int selectedFieldIdx = 0; selectedFieldIdx < int(selectedScalarFields.size()); selectedFieldIdx++) {
                auto& selectedScalarField = selectedScalarFields.at(selectedFieldIdx);
                diagram->addScalarField(selectedScalarField.first, selectedScalarField.second);
                diagram->setColorMap(selectedFieldIdx, selectedScalarField.third);
            }
            diagram->setIsEnsembleMode(isEnsembleMode);
            diagram->setCorrelationMeasureType(correlationMeasureType);
            diagram->setBeta(beta);
            diagram->setLineCountFactor(lineCountFactor);
            diagram->setCurveThickness(curveThickness);
            diagram->setCurveOpacity(curveOpacity);
            diagram->setDiagramRadius(diagramRadius);
            diagram->setAlignWithParentWindow(alignWithParentWindow);
            diagram->setOpacityByValue(opacityByValue);
            diagram->setColorByValue(colorByValue);
            diagram->setUse2DField(use2dField);
            diagram->setClearColor(viewManager->getClearColor());
            diagram->getCorrelationRangeTotal();
            diagram->getCellDistanceRangeTotal();
            //diagram->setCorrelationRange(parentDiagram->getCorrelationRangeTotal());
            //diagram->setCellDistanceRange(parentDiagram->getCellDistanceRangeTotal());
            diagram->setCorrelationRange(diagram->getCorrelationRangeTotal());
            diagram->setCellDistanceRange(diagram->getCellDistanceRangeTotal());
        }
    }
}

void DiagramRenderer::update(float dt, bool isMouseGrabbed) {
    for (auto it = diagrams.rbegin(); it != diagrams.rend(); it++) {
        auto& diagram = *it;
        diagram->setIsMouseGrabbedByParent(isMouseGrabbed);
        diagram->update(dt);
        if (diagram->getNeedsReRender()) {
            reRenderViewArray.at(diagramViewIdx) = true;
        }
        isMouseGrabbed |= diagram->getIsMouseGrabbed() || diagram->getIsMouseOverDiagramImGui();
    }
    for (size_t i = 0; i < diagrams.size(); i++) {
        auto diagram = diagrams.at(i);
        bool isDeselection = false;
        if (diagram->getHasNewFocusSelection(isDeselection)) {
            renderer->getDevice()->waitIdle();
            resetSelections(int(i + 1));
        }
        if (isDeselection) {
            selectedRegionStack.resize(i);
            diagrams.resize(i + 1);
        }
    }

    HEBChart* diagram = nullptr;
    for (auto it = diagrams.rbegin(); it != diagrams.rend(); it++) {
        diagram = it->get();
        if (diagram->getIsRegionSelected(0)) {
            break;
        }
        diagram = nullptr;
    }
    if (diagram) {
        auto correlationCalculators = volumeData->getCorrelationCalculatorsUsed();
        auto selectedRegion0 = diagram->getSelectedRegion(0);
        if (diagram->getIsRegionSelected(1)) {
            if (useAlignmentRotation) {
                updateAlignmentRotation(dt, diagram);
            }

            std::vector<std::vector<ICorrelationCalculator*>> calculatorMap;
            calculatorMap.resize(int(lastCorrelationCalculatorType) - int(firstCorrelationCalculatorType) + 1);
            auto selectedRegion1 = diagram->getSelectedRegion(1);
            for (auto& calculator : correlationCalculators) {
                if (calculator->getIsEnsembleMode() != isEnsembleMode
                        || !std::any_of(
                                selectedScalarFields.begin(), selectedScalarFields.end(),
                                [&](DiagramSelectedFieldData& selectedField) {
                                    return calculator->getInputFieldIndex() == selectedField.first;
                                })) {
                    continue;
                }
                int idx = int(calculator->getCalculatorType()) - int(firstCorrelationCalculatorType);
                calculatorMap.at(idx).push_back(calculator.get());
            }
            for (auto& calculators : calculatorMap) {
                if (calculators.empty() || calculators.size() % 2 != 0) {
                    continue;
                }
                for (size_t calculatorIdx = 0; calculatorIdx < calculators.size(); calculatorIdx++) {
                    auto selectedRegion = calculatorIdx % 2 == 0 ? selectedRegion0 : selectedRegion1;
                    calculators.at(calculatorIdx)->setReferencePointFromWorld(selectedRegion.getCenter());
                }
            }
        } else {
            for (auto& calculator : correlationCalculators) {
                if (calculator->getIsEnsembleMode() != isEnsembleMode) {
                    continue;
                }
                calculator->setReferencePointFromWorld(selectedRegion0.getCenter());
            }
        }
    }
}

void DiagramRenderer::onHasMoved(uint32_t viewIdx) {
    cachedAlignmentRotationDiagram = nullptr;
}

void DiagramRenderer::updateAlignmentRotation(float dt, HEBChart* diagram) {
    sgl::Camera* camera;
    auto* dataView = viewManager->getDataView(diagramViewIdx);
    if (dataView->syncWithParentCamera) {
        camera = dataView->parentSceneData->camera.get();
    } else {
        camera = viewManager->getViewSceneData(diagramViewIdx)->camera.get();
    }

    bool needsRestart = false;
    if (diagram != cachedAlignmentRotationDiagram) {
        cachedAlignmentRotationDiagram = diagram;
        needsRestart = true;
    } else {
        uint32_t pointIndex0 = diagram->getSelectedPointIndexGrid(0);
        uint32_t pointIndex1 = diagram->getSelectedPointIndexGrid(1);
        if (pointIndex0 != cachedPointIdx0 || pointIndex1 != cachedPointIdx1) {
            cachedPointIdx0 = pointIndex0;
            cachedPointIdx1 = pointIndex1;
            needsRestart = true;
        }
    }

    if (needsRestart) {
        cameraUpStart = camera->getCameraUp();
        cameraLookAtStart = camera->getLookAtLocation();
        cameraPositionStart = camera->getPosition();
        alignmentRotationTime = 0.0f;

        auto lineDirection = diagram->getLineDirection();
        glm::vec3 lineDirProj = lineDirection - glm::dot(lineDirection, cameraUpStart) * cameraUpStart;
        float lineDirProjLength = glm::length(lineDirProj);
        if (lineDirProjLength < 1e-2f) {
            // All angles are perpendicular, so we can stop.
            alignmentRotationTime = alignmentRotationTotalTime;
        } else {
            lineDirProj /= lineDirProjLength;
            glm::vec3 lineDirProjNormal = glm::cross(cameraUpStart, lineDirProj);
            if (glm::dot(lineDirProjNormal, camera->getCameraFront()) < 0.0f) {
                lineDirProjNormal = -lineDirProjNormal;
            }
            float y = glm::dot(glm::cross(lineDirProjNormal, camera->getCameraFront()), camera->getCameraUp());
            float x = glm::dot(lineDirProjNormal, camera->getCameraFront());
            rotationAngleTotal = std::atan2(y, x);
        }
    }

    if (alignmentRotationTime >= alignmentRotationTotalTime) {
        return;
    }
    alignmentRotationTime += dt;
    alignmentRotationTime = std::min(alignmentRotationTime, alignmentRotationTotalTime);
    float t = alignmentRotationTime / alignmentRotationTotalTime;

    glm::vec3 cameraUp = cameraUpStart;
    glm::vec3 cameraLookAt = cameraLookAtStart;
    glm::vec3 cameraPosition = cameraPositionStart;

    float theta = t * rotationAngleTotal;
    glm::mat4 rotTheta = glm::rotate(glm::mat4(1.0f), -theta, {0.0f, 1.0f, 0.0f});
    cameraPosition = cameraPosition - cameraLookAt;
    cameraPosition = glm::vec3(rotTheta * glm::vec4(cameraPosition, 1.0f));
    cameraUp = glm::vec3(rotTheta * glm::vec4(cameraUp, 1.0f));
    cameraPosition = cameraPosition + cameraLookAt;

    camera->setLookAtViewMatrix(cameraPosition, cameraLookAt, cameraUp);
    reRender = true;
}

bool DiagramRenderer::getHasGrabbedMouse() const {
    for (auto& diagram : diagrams) {
        if (diagram->getIsMouseGrabbed() || diagram->getIsMouseOverDiagramImGui()) {
            return true;
        }
    }
    return false;
}

void DiagramRenderer::setClearColor(const sgl::Color& clearColor) {
    for (auto& diagram : diagrams) {
        diagram->setClearColor(clearColor);
    }
}

void DiagramRenderer::renderViewImpl(uint32_t viewIdx) {
    if (viewIdx != diagramViewIdx) {
        return;
    }

    SceneData* sceneData = viewManager->getViewSceneData(viewIdx);
    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);

    for (auto& diagram : diagrams) {
        if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
            ImVec2 pos = ImGui::GetCursorScreenPos();
            diagram->setImGuiWindowOffset(int(pos.x), int(pos.y));
        } else {
            diagram->setImGuiWindowOffset(0, 0);
        }
        diagram->render();
        diagram->setBlitTargetSupersamplingFactor(viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
        diagram->blitToTargetVk();
    }
}

void DiagramRenderer::renderViewPreImpl(uint32_t viewIdx) {
    HEBChart* diagram = nullptr;
    for (auto it = diagrams.rbegin(); it != diagrams.rend(); it++) {
        diagram = it->get();
        if (diagram->getIsRegionSelected(0)) {
            break;
        }
        diagram = nullptr;
    }
    if (!diagram) {
        return;
    }

    for (int idx = 0; idx < 2; idx++) {
        if (diagram->getIsRegionSelected(idx)) {
            domainOutlineComputePasses[idx].at(viewIdx)->setOutlineSettings(diagram->getSelectedRegion(idx), lineWidth);
            domainOutlineComputePasses[idx].at(viewIdx)->render();
            domainOutlineRasterPasses[idx].at(viewIdx)->render();
        }
    }
    if (diagram->getIsRegionSelected(0) && diagram->getIsRegionSelected(1)) {
        connectingLineRasterPass.at(viewIdx)->setLineSettings(diagram->getLinePositions(), lineWidth);
        connectingLineRasterPass.at(viewIdx)->render();
    }
}

void DiagramRenderer::addViewImpl(uint32_t viewIdx) {
    const size_t numEdges = 12;
    for (int idx = 0; idx < 2; idx++) {
        OutlineRenderData outlineRenderData{};
        sgl::vk::Device* device = renderer->getDevice();
        outlineRenderData.vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec3) * numEdges * 8,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        outlineRenderData.indexBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(uint32_t) * numEdges * 6 * 6,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        outlineRenderDataList[idx].push_back(outlineRenderData);

        auto domainOutlineRasterPass = std::make_shared<DomainOutlineRasterPass>(
                renderer, viewManager->getViewSceneData(viewIdx));
        domainOutlineRasterPass->setRenderData(outlineRenderData.indexBuffer, outlineRenderData.vertexPositionBuffer);
        domainOutlineRasterPasses[idx].push_back(domainOutlineRasterPass);

        auto domainOutlineComputePass = std::make_shared<DomainOutlineComputePass>(renderer);
        domainOutlineComputePass->setRenderData(outlineRenderData.indexBuffer, outlineRenderData.vertexPositionBuffer);
        domainOutlineComputePasses[idx].push_back(domainOutlineComputePass);
    }

    connectingLineRasterPass.push_back(std::make_shared<ConnectingLineRasterPass>(
            renderer, viewManager->getViewSceneData(viewIdx)));
}

void DiagramRenderer::removeViewImpl(uint32_t viewIdx) {
    if (diagramViewIdx >= viewIdx && diagramViewIdx != 0) {
        diagramViewIdx--;
    } else if (viewManager->getNumViews() > 0) {
        diagramViewIdx++;
    }

    connectingLineRasterPass.erase(connectingLineRasterPass.begin() + viewIdx);

    for (int idx = 0; idx < 2; idx++) {
        outlineRenderDataList[idx].erase(outlineRenderDataList[idx].begin() + viewIdx);
        domainOutlineRasterPasses[idx].erase(domainOutlineRasterPasses[idx].begin() + viewIdx);
        domainOutlineComputePasses[idx].erase(domainOutlineComputePasses[idx].begin() + viewIdx);
    }
}

void DiagramRenderer::updateScalarFieldComboValue() {
    std::vector<std::string> comboSelVec(0);
    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    for (size_t fieldIdx = 0; fieldIdx < fieldNames.size(); fieldIdx++) {
        if (scalarFieldSelectionArray.at(fieldIdx)) {
            ImGui::SetItemDefaultFocus();
            comboSelVec.push_back(fieldNames.at(fieldIdx));
        }
    }
    scalarFieldComboValue = "";
    for (size_t v = 0; v < comboSelVec.size(); ++v) {
        scalarFieldComboValue += comboSelVec[v];
        if (comboSelVec.size() > 1 && v + 1 != comboSelVec.size()) {
            scalarFieldComboValue += ", ";
        }
    }
}

void DiagramRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    std::string textDefault = "View " + std::to_string(diagramViewIdx + 1);
    if (propertyEditor.addBeginCombo(
            "Diagram View", textDefault, ImGuiComboFlags_NoArrowButton)) {
        for (size_t viewIdx = 0; viewIdx < viewVisibilityArray.size(); viewIdx++) {
            std::string text = "View " + std::to_string(viewIdx + 1);
            bool showInView = diagramViewIdx == uint32_t(viewIdx);
            if (ImGui::Selectable(
                    text.c_str(), &showInView, ImGuiSelectableFlags_::ImGuiSelectableFlags_None)) {
                diagramViewIdx = uint32_t(viewIdx);
                reRender = true;
                recreateDiagramSwapchain();
            }
            if (showInView) {
                ImGui::SetItemDefaultFocus();
            }
        }
        propertyEditor.addEndCombo();
    }

    if (volumeData->getEnsembleMemberCount() > 1 && volumeData->getTimeStepCount() > 1) {
        int modeIdx = isEnsembleMode ? 0 : 1;
        if (propertyEditor.addCombo("Correlation Mode", &modeIdx, CORRELATION_MODE_NAMES, 2)) {
            isEnsembleMode = modeIdx == 0;
            for (auto& diagram : diagrams) {
                diagram->setIsEnsembleMode(isEnsembleMode);
            }
            correlationRangeTotal = correlationRange = parentDiagram->getCorrelationRangeTotal();
            dirty = true;
            reRender = true;
        }
    }

    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        for (auto& diagram : diagrams) {
            diagram->setCorrelationMeasureType(correlationMeasureType);
        }
        correlationRangeTotal = correlationRange = parentDiagram->getCorrelationRangeTotal();
        for (auto& diagram : diagrams) {
            diagram->setCorrelationRange(correlationRangeTotal);
        }
        resetSelections();
        dirty = true;
        reRender = true;
    }

    if (volumeData) {
        if (propertyEditor.addBeginCombo(
                "Scalar Fields", scalarFieldComboValue, ImGuiComboFlags_NoArrowButton)) {
            const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
            for (size_t fieldIdx = 0; fieldIdx < fieldNames.size(); fieldIdx++) {
                std::string text = fieldNames.at(fieldIdx);
                bool useField = scalarFieldSelectionArray.at(fieldIdx);
                if (ImGui::Selectable(
                        text.c_str(), &useField, ImGuiSelectableFlags_::ImGuiSelectableFlags_DontClosePopups)) {
                    if (useField) {
                        parentDiagram->addScalarField(int(fieldIdx), text);

                        // Get the next free color map.
                        std::vector<int> colorMapUseCounter(NUM_COLOR_MAPS);
                        for (size_t i = 0; i < selectedScalarFields.size(); i++) {
                            colorMapUseCounter.at(int(selectedScalarFields.at(i).third)) += 1;
                        }
                        size_t minIndex = 0;
                        int minUseCount = std::numeric_limits<int>::max();
                        for (int i = 0; i < NUM_COLOR_MAPS; i++) {
                            if (colorMapUseCounter.at(i) < minUseCount) {
                                minUseCount = colorMapUseCounter.at(i);
                                minIndex = i;
                            }
                        }
                        DiagramSelectedFieldData newFieldData( int(fieldIdx), text, DiagramColorMap(minIndex));

                        bool foundInsertionPosition = false;
                        for (size_t i = 0; i < selectedScalarFields.size(); i++) {
                            if (selectedScalarFields.at(i).first > int(fieldIdx)) {
                                selectedScalarFields.insert(
                                        selectedScalarFields.begin() + ptrdiff_t(i), newFieldData);
                                parentDiagram->setColorMap(int(i), newFieldData.third);
                                foundInsertionPosition = true;
                                break;
                            }
                        }
                        if (!foundInsertionPosition) {
                            selectedScalarFields.push_back(newFieldData);
                            parentDiagram->setColorMap(int(selectedScalarFields.size() - 1), newFieldData.third);
                        }
                    } else {
                        parentDiagram->removeScalarField(int(fieldIdx), false);
                        for (size_t i = 0; i < selectedScalarFields.size(); i++) {
                            if (selectedScalarFields.at(i).first == int(fieldIdx)) {
                                volumeData->releaseScalarField(this, selectedScalarFields.at(i).first);
                                selectedScalarFields.erase(selectedScalarFields.begin() + ptrdiff_t(i));
                                break;
                            }
                        }
                    }
                    scalarFieldSelectionArray.at(fieldIdx) = useField;
                    dirty = true;
                    reRender = true;
                }
                if (useField) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            updateScalarFieldComboValue();
            propertyEditor.addEndCombo();
        }
    }

    for (int selectedFieldIdx = 0; selectedFieldIdx < int(selectedScalarFields.size()); selectedFieldIdx++) {
        ImGui::PushID(selectedFieldIdx);
        auto& selectedScalarField = selectedScalarFields.at(selectedFieldIdx);
        DiagramColorMap& colorMap = selectedScalarField.third;
        if (colorByValue && propertyEditor.addCombo(
                "Color Map (" + selectedScalarField.second + ")", (int*)&colorMap, DIAGRAM_COLOR_MAP_NAMES,
                IM_ARRAYSIZE(DIAGRAM_COLOR_MAP_NAMES))) {
            for (auto& diagram : diagrams) {
                diagram->setColorMap(selectedFieldIdx, colorMap);
            }
            reRender = true;
        }
        ImGui::PopID();
    }

    if (propertyEditor.addCheckbox("Value -> Opacity", &opacityByValue)) {
        for (auto& diagram : diagrams) {
            diagram->setOpacityByValue(opacityByValue);
        }
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Value -> Color", &colorByValue)) {
        for (auto& diagram : diagrams) {
            diagram->setColorByValue(colorByValue);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderFloatEdit("beta", &beta, 0.0f, 1.0f) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setBeta(beta);
        }
        reRender = true;
    }

    if (downscalingFactorUniform) {
        if (propertyEditor.addSliderIntPowerOfTwoEdit(
                "Downscaling", &downscalingFactorX, minDownscalingFactor, maxDownscalingFactor)
                    == ImGui::EditMode::INPUT_FINISHED) {
            downscalingFactorY = downscalingFactorX;
            downscalingFactorZ = downscalingFactorX;
            parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
            resetSelections();
            reRender = true;
        }
    } else {
        int dfs[3] = { downscalingFactorX, downscalingFactorY, downscalingFactorZ };
        auto editMode = propertyEditor.addSliderInt3PowerOfTwoEdit(
                "Downscaling", dfs, minDownscalingFactor, maxDownscalingFactor);
        downscalingFactorX = dfs[0];
        downscalingFactorY = dfs[1];
        downscalingFactorZ = dfs[2];
        if (editMode == ImGui::EditMode::INPUT_FINISHED) {
            parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
            resetSelections();
            reRender = true;
        }
    }

    if (propertyEditor.addSliderIntEdit(
            "#Line Factor", &lineCountFactor, 10, 1000) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setLineCountFactor(lineCountFactor);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderFloat("Curve Thickness", &curveThickness, 0.5f, 4.0f)) {
        for (auto& diagram : diagrams) {
            diagram->setCurveThickness(curveThickness);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderFloat("Opacity", &curveOpacity, 0.0f, 1.0f)) {
        for (auto& diagram : diagrams) {
            diagram->setCurveOpacity(curveOpacity);
        }
        reRender = true;
    }

    if (propertyEditor.addSliderFloat2Edit(
            "Correlation Range", &correlationRange.x,
            correlationRangeTotal.x, correlationRangeTotal.y) == ImGui::EditMode::INPUT_FINISHED) {
        parentDiagram->setCorrelationRange(correlationRange);
        resetSelections();
        reRender = true;
    }

    if (propertyEditor.addSliderInt2Edit(
            "Cell Dist. Range", &cellDistanceRange.x,
            cellDistanceRangeTotal.x, cellDistanceRangeTotal.y) == ImGui::EditMode::INPUT_FINISHED) {
        parentDiagram->setCellDistanceRange(cellDistanceRange);
        resetSelections();
        reRender = true;
    }

    if (propertyEditor.addSliderInt("Diagram Radius", &diagramRadius, 100, 400)) {
        parentDiagram->setDiagramRadius(diagramRadius);
        resetSelections();
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Align with Window", &alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        reRender = true;
    }

    /*if (propertyEditor.addCheckbox("Use 2D Field", &use2dField)) {
        for (auto& diagram : diagrams) {
            diagram->setUse2DField(use2dField);
        }
        reRender = true;
    }*/

    if (parentDiagram->renderGuiPropertyEditor(propertyEditor)) {
        reRender = true;
    }
}
