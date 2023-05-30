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
#include "SelectionBoxRasterPass.hpp"
#include "RadarBarChart.hpp"
#include "HEBChart.hpp"

#include "DiagramRenderer.hpp"
#include "NLOptDefines.hpp"

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

    parentDiagram = std::make_shared<HEBChart>();
    //auto diagram = std::make_shared<RadarBarChart>(true);
    parentDiagram->setRendererVk(renderer);
    parentDiagram->initialize();
    //diagram->setDataTimeDependent(variableNames, variableValuesTimeDependent);
    //diagram->setUseEqualArea(true);
    if (volumeData) {
        parentDiagram->setVolumeData(volumeData, true);
        parentDiagram->setIsEnsembleMode(isEnsembleMode);
        parentDiagram->setUseSeparateColorVarianceAndCorrelation(separateColorVarianceAndCorrelation);
        parentDiagram->setGlobalStdDevRangeQueryCallback([this](int idx) { return computeGlobalStdDevRange(idx); });
        parentDiagram->setColorMapVariance(colorMapVariance);
        parentDiagram->setCorrelationMeasureType(correlationMeasureType);
        parentDiagram->setUseAbsoluteCorrelationMeasure(useAbsoluteCorrelationMeasure);
        parentDiagram->setNumBins(numBins);
        parentDiagram->setKraskovNumNeighbors(k);
        parentDiagram->setSamplingMethodType(samplingMethodType);
        parentDiagram->setNumSamples(numSamples);
        parentDiagram->setBeta(beta);
        parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
        parentDiagram->setLineCountFactor(lineCountFactorContext);
        parentDiagram->setCurveThickness(curveThickness);
        parentDiagram->setCurveOpacity(curveOpacityContext);
        parentDiagram->setDiagramRadius(diagramRadius);
        parentDiagram->setOuterRingSizePercentage(outerRingSizePct);
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setOpacityByValue(opacityByValue);
        parentDiagram->setShowSelectedRegionsByColor(showSelectedRegionsByColor);
        parentDiagram->setDesaturateUnselectedRing(desaturateUnselectedRing);
        parentDiagram->setUseNeonSelectionColors(useNeonSelectionColors);
        parentDiagram->setUseGlobalStdDevRange(useGlobalStdDevRange);
        parentDiagram->setOctreeMethod(octreeMethod);
        parentDiagram->setColorByValue(colorByValue);
        parentDiagram->setUse2DField(use2dField);
        parentDiagram->setClearColor(viewManager->getClearColor());
        parentDiagram->setUseCorrelationComputationGpu(useCorrelationComputationGpu);
        parentDiagram->setDataMode(dataMode);
        parentDiagram->setUseBufferTiling(useBufferTiling);
    }
    diagrams.push_back(parentDiagram);
}

int DiagramRenderer::getCorrelationMemberCount() {
    return isEnsembleMode ? volumeData->getEnsembleMemberCount() : volumeData->getTimeStepCount();
}

bool DiagramRenderer::getSupportsBufferMode() {
    bool supportsBufferMode = true;
    for (const auto& selectedScalarField : selectedScalarFields) {
        if (!volumeData->getScalarFieldSupportsBufferMode(selectedScalarField.first)) {
            supportsBufferMode = false;
            break;
        }
    }
    if (!supportsBufferMode && dataMode == CorrelationDataMode::BUFFER_ARRAY) {
        dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        for (auto& diagram : diagrams) {
            diagram->setDataMode(dataMode);
        }
    }
    return supportsBufferMode;
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
        reRenderTriggeredByDiagram = true;
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

        if (!getSupportsBufferMode() || volumeData->getGridSizeZ() < 4) {
            dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        } else {
            dataMode = CorrelationDataMode::BUFFER_ARRAY;
        }
    }

    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    parentDiagram->setVolumeData(volumeData, isNewData);
    if (isNewData) {
        selectedScalarFields.clear();
        int standardFieldIdx = volumeData->getStandardScalarFieldIdx();
        selectedScalarFields.emplace_back(standardFieldIdx, fieldNames.at(standardFieldIdx), DiagramColorMap::WISTIA);
        scalarFieldSelectionArray.clear();
        scalarFieldSelectionArray.resize(fieldNames.size());
        scalarFieldSelectionArray.at(standardFieldIdx) = true;
        scalarFieldComboValue = fieldNames.at(standardFieldIdx);
        volumeData->acquireScalarField(this, standardFieldIdx);
        parentDiagram->clearScalarFields();
        parentDiagram->setUseSeparateColorVarianceAndCorrelation(separateColorVarianceAndCorrelation);
        for (int selectedFieldIdx = 0; selectedFieldIdx < int(selectedScalarFields.size()); selectedFieldIdx++) {
            auto& selectedScalarField = selectedScalarFields.at(selectedFieldIdx);
            parentDiagram->addScalarField(selectedScalarField.first, selectedScalarField.second);
            parentDiagram->setColorMap(selectedFieldIdx, selectedScalarField.third);
        }
        parentDiagram->setGlobalStdDevRangeQueryCallback([this](int idx) { return computeGlobalStdDevRange(idx); });
        parentDiagram->setColorMapVariance(colorMapVariance);
        parentDiagram->setIsEnsembleMode(isEnsembleMode);
        parentDiagram->setCorrelationMeasureType(correlationMeasureType);
        parentDiagram->setUseAbsoluteCorrelationMeasure(useAbsoluteCorrelationMeasure);
        parentDiagram->setNumBins(numBins);
        parentDiagram->setKraskovNumNeighbors(k);
        parentDiagram->setSamplingMethodType(samplingMethodType);
        parentDiagram->setNumSamples(numSamples);
        parentDiagram->setBeta(beta);
        parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
        parentDiagram->setLineCountFactor(lineCountFactorContext);
        parentDiagram->setCurveThickness(curveThickness);
        parentDiagram->setCurveOpacity(curveOpacityContext);
        parentDiagram->setDiagramRadius(diagramRadius);
        parentDiagram->setOuterRingSizePercentage(outerRingSizePct);
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        parentDiagram->setOpacityByValue(opacityByValue);
        parentDiagram->setShowSelectedRegionsByColor(showSelectedRegionsByColor);
        parentDiagram->setDesaturateUnselectedRing(desaturateUnselectedRing);
        parentDiagram->setUseNeonSelectionColors(useNeonSelectionColors);
        parentDiagram->setUseGlobalStdDevRange(useGlobalStdDevRange);
        parentDiagram->setOctreeMethod(octreeMethod);
        parentDiagram->setColorByValue(colorByValue);
        parentDiagram->setUse2DField(use2dField);
        parentDiagram->setClearColor(viewManager->getClearColor());
        parentDiagram->setUseCorrelationComputationGpu(useCorrelationComputationGpu);
        parentDiagram->setDataMode(dataMode);
        parentDiagram->setUseBufferTiling(useBufferTiling);

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
        if (fieldNames.size() > scalarFieldSelectionArray.size()) {
            scalarFieldSelectionArray.resize(fieldNames.size());
        }
    }

    if (isNewData || cachedMemberCount != getCorrelationMemberCount()) {
        onCorrelationMemberCountChanged();
    }

    updateScalarFieldComboValue();
    for (int selectedFieldIdx = 0; selectedFieldIdx < int(selectedScalarFields.size()); selectedFieldIdx++) {
        auto& selectedScalarField = selectedScalarFields.at(selectedFieldIdx);
        selectedScalarField.second = fieldNames.at(selectedScalarField.first);
    }
}

void DiagramRenderer::onCorrelationMemberCountChanged() {
    int cs = getCorrelationMemberCount();
    k = std::max(sgl::iceil(3 * cs, 100), 1);
    kMax = std::max(sgl::iceil(7 * cs, 100), 20);
    for (auto& diagram : diagrams) {
        diagram->setKraskovNumNeighbors(k);
    }
    if (parentDiagram) {
        correlationRangeTotal = correlationRange = parentDiagram->getCorrelationRangeTotal();
    }
    cachedMemberCount = cs;
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
    if (renderOnlyLastFocusDiagram) {
        if (viewIdx == contextDiagramViewIdx) {
            recreateDiagramSwapchain(0);
        }
        if (viewIdx == focusDiagramViewIdx) {
            for (size_t idx = 1; idx < diagrams.size(); idx++) {
                recreateDiagramSwapchain(int(idx));
            }
        }
    } else {
        if (viewIdx == contextDiagramViewIdx || viewIdx == focusDiagramViewIdx) {
            recreateDiagramSwapchain();
        }
    }
    for (int idx = 0; idx < 2; idx++) {
        domainOutlineRasterPasses[idx].at(viewIdx)->recreateSwapchain(width, height);
        domainOutlineComputePasses[idx].at(viewIdx)->recreateSwapchain(width, height);
        selectionBoxRasterPasses[idx].at(viewIdx)->recreateSwapchain(width, height);
    }
    connectingLineRasterPass.at(viewIdx)->recreateSwapchain(width, height);
}

void DiagramRenderer::recreateDiagramSwapchain(int diagramIdx) {
    for (size_t idx = 0; idx < diagrams.size(); idx++) {
        if (diagramIdx >= 0 && diagramIdx != int(idx)) {
            continue;
        }
        uint32_t diagramViewIdx = contextDiagramViewIdx;
        if (renderOnlyLastFocusDiagram && idx > 0) {
            diagramViewIdx = focusDiagramViewIdx;
        }
        SceneData* sceneData = viewManager->getViewSceneData(diagramViewIdx);
        auto& diagram = diagrams.at(idx);
        diagram->setBlitTargetVk(
                (*sceneData->sceneTexture)->getImageView(),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (alignWithParentWindow) {
            diagram->setBlitTargetSupersamplingFactor(
                    viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
            diagram->updateSizeByParent();
        }
    }
    reRenderTriggeredByDiagram = true;
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
            diagram->copyVectorWidgetSettingsFrom(parentDiagram.get());
            diagram->setIsFocusView(idx);
            diagrams.at(idx) = diagram;
            recreateDiagramSwapchain(idx);
        }
    }
    if (idx > 0) {
        auto diagram = diagrams.at(idx);
        if (volumeData) {
            diagram->setVolumeData(volumeData, true);
            // TODO
            //int dfx = sgl::iceil(downscalingFactorX, 1 << 2 * idx);
            //int dfy = sgl::iceil(downscalingFactorY, 1 << 2 * idx);
            //int dfz = sgl::iceil(downscalingFactorZ, 1 << 2 * idx);
            int dfx = sgl::iceil(downscalingFactorX, int(std::pow(downscalingFactorFocusX, idx)));
            int dfy = sgl::iceil(downscalingFactorY, int(std::pow(downscalingFactorFocusY, idx)));
            int dfz = sgl::iceil(downscalingFactorZ, int(std::pow(downscalingFactorFocusZ, idx)));
            diagram->setDownscalingFactors(dfx, dfy, dfz);
            diagram->setRegions(selectedRegionStack.at(idx - 1));
            diagram->clearScalarFields();
            if (showOnlySelectedVariableInFocusDiagrams && parentDiagram->getHasFocusSelectionField()) {
                // TODO: Also on update!
                diagram->setShowVariablesForFieldIdxOnly(parentDiagram->getFocusSelectionFieldIndex());
            } else {
                diagram->setShowVariablesForFieldIdxOnly(-1);
            }
            diagram->setUseSeparateColorVarianceAndCorrelation(separateColorVarianceAndCorrelation);
            for (int selectedFieldIdx = 0; selectedFieldIdx < int(selectedScalarFields.size()); selectedFieldIdx++) {
                auto& selectedScalarField = selectedScalarFields.at(selectedFieldIdx);
                diagram->addScalarField(selectedScalarField.first, selectedScalarField.second);
                diagram->setColorMap(selectedFieldIdx, selectedScalarField.third);
            }
            diagram->setGlobalStdDevRangeQueryCallback([this](int idx) { return computeGlobalStdDevRange(idx); });
            diagram->setColorMapVariance(colorMapVariance);
            diagram->setIsEnsembleMode(isEnsembleMode);
            diagram->setCorrelationMeasureType(correlationMeasureType);
            diagram->setUseAbsoluteCorrelationMeasure(useAbsoluteCorrelationMeasure);
            diagram->setNumBins(numBins);
            diagram->setKraskovNumNeighbors(k);
            diagram->setSamplingMethodType(samplingMethodType);
            diagram->setNumSamples(numSamples);
            diagram->setBeta(beta);
            diagram->setLineCountFactor(lineCountFactorFocus);
            diagram->setCurveThickness(curveThickness);
            diagram->setCurveOpacity(curveOpacityFocus);
            diagram->setDiagramRadius(diagramRadius);
            diagram->setOuterRingSizePercentage(outerRingSizePct);
            if (renderOnlyLastFocusDiagram) {
                diagram->setAlignWithParentWindow(alignWithParentWindow);
            }
            diagram->setOpacityByValue(opacityByValue);
            diagram->setShowSelectedRegionsByColor(showSelectedRegionsByColor);
            diagram->setDesaturateUnselectedRing(desaturateUnselectedRing);
            diagram->setUseNeonSelectionColors(useNeonSelectionColors);
            diagram->setUseGlobalStdDevRange(useGlobalStdDevRange);
            diagram->setOctreeMethod(octreeMethod);
            diagram->setColorByValue(colorByValue);
            diagram->setUse2DField(use2dField);
            diagram->setClearColor(viewManager->getClearColor());
            diagram->setUseCorrelationComputationGpu(useCorrelationComputationGpu);
            diagram->setDataMode(dataMode);
            diagram->setUseBufferTiling(useBufferTiling);
            diagram->getCorrelationRangeTotal();
            diagram->getCellDistanceRangeTotal();
            //diagram->setCorrelationRange(parentDiagram->getCorrelationRangeTotal());
            //diagram->setCellDistanceRange(parentDiagram->getCellDistanceRangeTotal());
            diagram->setCorrelationRange(diagram->getCorrelationRangeTotal());
            diagram->setCellDistanceRange(diagram->getCellDistanceRangeTotal());
        }
    }
}

std::pair<float, float> DiagramRenderer::computeGlobalStdDevRange(int fieldIdx) {
    std::pair<float, float> minMaxPair = parentDiagram->getLocalStdDevRange(fieldIdx);
    for (size_t i = 1; i < diagrams.size(); i++) {
        auto diagram = diagrams.at(i);
        auto newMinMax = diagram->getLocalStdDevRange(fieldIdx);
        if (newMinMax.first < minMaxPair.first) {
            newMinMax.first = newMinMax.first;
        }
        if (newMinMax.second > minMaxPair.second) {
            newMinMax.second = newMinMax.second;
        }
    }
    return minMaxPair;
}

void DiagramRenderer::update(float dt, bool isMouseGrabbed) {
    reRenderTriggeredByDiagram = false;
    for (auto it = diagrams.rbegin(); it != diagrams.rend(); it++) {
        auto& diagram = *it;
        diagram->setIsMouseGrabbedByParent(isMouseGrabbed);
        diagram->update(dt);
        if (diagram->getNeedsReRender() && !reRenderViewArray.empty()) {
            for (int viewIdx = 0; viewIdx < int(viewVisibilityArray.size()); viewIdx++) {
                if (viewVisibilityArray.at(viewIdx)) {
                    reRenderViewArray.at(viewIdx) = true;
                }
            }
            reRenderTriggeredByDiagram = true;
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
            renderer->getDevice()->waitIdle();
            selectedRegionStack.resize(i);
            diagrams.resize(i + 1);
        }

        int returnToViewIdx = diagram->getReturnToViewIdx();
        if (returnToViewIdx >= 0) {
            renderer->getDevice()->waitIdle();
            selectedRegionStack.resize(returnToViewIdx);
            diagrams.resize(returnToViewIdx + 1);
            //diagrams.at(returnToViewIdx)->resetSelectedPrimitives();
            diagrams.at(returnToViewIdx)->resetFocusSelection();
            for (int viewIdx = 0; viewIdx < int(viewVisibilityArray.size()); viewIdx++) {
                if (viewVisibilityArray.at(viewIdx)) {
                    reRenderViewArray.at(viewIdx) = true;
                }
            }
            reRenderTriggeredByDiagram = true;
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
    auto* dataView = viewManager->getDataView(contextDiagramViewIdx);
    if (dataView->syncWithParentCamera) {
        camera = dataView->parentSceneData->camera.get();
    } else {
        camera = viewManager->getViewSceneData(contextDiagramViewIdx)->camera.get();
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
        // Compute the vector perpendicular to both the up direction and the line direction.
        glm::vec3 normal = glm::cross(lineDirection, cameraUpStart);
        float normalLength = glm::length(normal);
        if (normalLength < 1e-2f) {
            // All angles are perpendicular, so we can stop.
            alignmentRotationTotalTime = alignmentRotationTime = 0.0f;
        } else {
            normal /= normalLength;
            // Take the shorter angle.
            if (glm::dot(normal, camera->getCameraFront()) < 0.0f) {
                normal = -normal;
            }
            // Compute the angle necessary to align the front direction with the normal vector.
            // Then, the viewing direction will be perpendicular to the line while pertaining the up direction.
            float y = glm::dot(glm::cross(normal, camera->getCameraFront()), camera->getCameraUp());
            float x = glm::dot(normal, camera->getCameraFront());
            rotationAngleTotal = std::atan2(y, x);
            alignmentRotationTotalTime = alignmentRotationTotalTimeMax * std::abs(rotationAngleTotal) / sgl::PI;
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
    reRenderTriggeredByDiagram = true;
}

void DiagramRenderer::renderViewImpl(uint32_t viewIdx) {
    if (viewIdx != contextDiagramViewIdx && viewIdx != focusDiagramViewIdx) {
        return;
    }

    SceneData* sceneData = viewManager->getViewSceneData(viewIdx);
    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);

    for (size_t i = 0; i < diagrams.size(); i++) {
        uint32_t diagramViewIdx = contextDiagramViewIdx;
        if (renderOnlyLastFocusDiagram) {
            // Only render the diagrams in the appropriate views.
            if ((viewIdx != contextDiagramViewIdx && i == 0) || (viewIdx != focusDiagramViewIdx && i > 0)) {
                continue;
            }
            // Only render the last view.
            if (alignWithParentWindow && i > 0 && i != diagrams.size() - 1) {
                continue;
            }
            if (i > 0) {
                diagramViewIdx = focusDiagramViewIdx;
            }
        }

        auto& diagram = diagrams.at(i);
        if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
            ImVec2 pos = ImGui::GetCursorScreenPos();
            diagram->setImGuiWindowOffset(int(pos.x), int(pos.y));
        } else {
            diagram->setImGuiWindowOffset(0, 0);
        }
        if (reRenderTriggeredByDiagram || diagram->getIsFirstRender()) {
            diagram->render();
        }
        diagram->setBlitTargetSupersamplingFactor(viewManager->getDataView(diagramViewIdx)->getSupersamplingFactor());
        diagram->blitToTargetVk();
    }
}

void DiagramRenderer::renderViewPreImpl(uint32_t viewIdx) {
    if ((viewIdx == contextDiagramViewIdx || viewIdx == focusDiagramViewIdx) && alignWithParentWindow) {
        return;
    }

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

    bool twoRegionsSelected = diagram->getIsRegionSelected(0) && diagram->getIsRegionSelected(1);

    for (int idx = 0; idx < 2; idx++) {
        if (diagram->getIsRegionSelected(idx)) {
            if (useOpaqueSelectionBoxes) {
                selectionBoxRasterPasses[idx].at(viewIdx)->setBoxAabb(diagram->getSelectedRegion(idx));
                selectionBoxRasterPasses[idx].at(viewIdx)->setColor(
                        diagram->getColorSelected(idx).getFloatColorRGBA());
                selectionBoxRasterPasses[idx].at(viewIdx)->render();
            } else {
                domainOutlineComputePasses[idx].at(viewIdx)->setOutlineSettings(
                        diagram->getSelectedRegion(idx), lineWidth, 1e-4f);
                domainOutlineComputePasses[idx].at(viewIdx)->render();
                if (twoRegionsSelected && diagram->getShowSelectedRegionsByColor()) {
                    domainOutlineRasterPasses[idx].at(viewIdx)->setCustomColor(
                            diagram->getColorSelected(idx).getFloatColorRGBA());
                } else {
                    domainOutlineRasterPasses[idx].at(viewIdx)->resetCustomColor();
                }
                domainOutlineRasterPasses[idx].at(viewIdx)->render();
            }
        }
    }
    if (twoRegionsSelected) {
        connectingLineRasterPass.at(viewIdx)->setLineSettings(diagram->getLinePositions(), lineWidth * 2.0f);
        if (diagram->getShowSelectedRegionsByColor()) {
            connectingLineRasterPass.at(viewIdx)->setCustomColors(
                    diagram->getColorSelected0().getFloatColorRGBA(),
                    diagram->getColorSelected1().getFloatColorRGBA());
        } else {
            connectingLineRasterPass.at(viewIdx)->resetCustomColors();
        }
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

        selectionBoxRasterPasses[idx].push_back(std::make_shared<SelectionBoxRasterPass>(
                renderer, viewManager->getViewSceneData(viewIdx)));
    }

    connectingLineRasterPass.push_back(std::make_shared<ConnectingLineRasterPass>(
            renderer, viewManager->getViewSceneData(viewIdx)));
}

bool DiagramRenderer::adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx) {
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

void DiagramRenderer::removeViewImpl(uint32_t viewIdx) {
    bool diagramViewIdxChanged = false;
    diagramViewIdxChanged |= adaptIdxOnViewRemove(viewIdx, contextDiagramViewIdx);
    if (renderOnlyLastFocusDiagram) {
        diagramViewIdxChanged |= adaptIdxOnViewRemove(viewIdx, focusDiagramViewIdx);
    }
    if (diagramViewIdxChanged) {
        reRender = true;
        reRenderTriggeredByDiagram = true;
        recreateDiagramSwapchain();
    }

    connectingLineRasterPass.erase(connectingLineRasterPass.begin() + viewIdx);

    for (int idx = 0; idx < 2; idx++) {
        outlineRenderDataList[idx].erase(outlineRenderDataList[idx].begin() + viewIdx);
        domainOutlineRasterPasses[idx].erase(domainOutlineRasterPasses[idx].begin() + viewIdx);
        domainOutlineComputePasses[idx].erase(domainOutlineComputePasses[idx].begin() + viewIdx);
        selectionBoxRasterPasses[idx].erase(selectionBoxRasterPasses[idx].begin() + viewIdx);
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

void DiagramRenderer::renderDiagramViewSelectionGui(
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

void DiagramRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (renderOnlyLastFocusDiagram) {
        renderDiagramViewSelectionGui(propertyEditor, "Context Diagram View", contextDiagramViewIdx);
        renderDiagramViewSelectionGui(propertyEditor, "Focus Diagram View", focusDiagramViewIdx);
    } else {
        renderDiagramViewSelectionGui(propertyEditor, "Diagram View", contextDiagramViewIdx);
        focusDiagramViewIdx = contextDiagramViewIdx;
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
            reRenderTriggeredByDiagram = true;
        }
    }

    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        for (auto& diagram : diagrams) {
            diagram->setCorrelationMeasureType(correlationMeasureType);
        }
        onCorrelationMemberCountChanged();
        for (auto& diagram : diagrams) {
            diagram->setCorrelationRange(correlationRangeTotal);
        }
        resetSelections();
        dirty = true;
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
            && correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
            && propertyEditor.addCheckbox("Absolute Correlations", &useAbsoluteCorrelationMeasure)) {
        for (auto& diagram : diagrams) {
            diagram->setUseAbsoluteCorrelationMeasure(useAbsoluteCorrelationMeasure);
        }
        correlationRangeTotal = correlationRange = parentDiagram->getCorrelationRangeTotal();
        for (auto& diagram : diagrams) {
            diagram->setCorrelationRange(correlationRangeTotal);
        }
        resetSelections();
        dirty = true;
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED && propertyEditor.addSliderIntEdit(
            "#Bins", &numBins, 10, 100) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setNumBins(numBins);
        }
        resetSelections();
        dirty = true;
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
    if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV && propertyEditor.addSliderIntEdit(
            "#Neighbors", &k, 1, kMax) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setKraskovNumNeighbors(k);
        }
        correlationRangeTotal = correlationRange = parentDiagram->getCorrelationRangeTotal();
        for (auto& diagram : diagrams) {
            diagram->setCorrelationRange(correlationRangeTotal);
        }
        resetSelections();
        dirty = true;
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCombo(
            "Sampling Method", (int*)&samplingMethodType,
            SAMPLING_METHOD_TYPE_NAMES, IM_ARRAYSIZE(SAMPLING_METHOD_TYPE_NAMES))) {
        for (auto& diagram : diagrams) {
            diagram->setSamplingMethodType(samplingMethodType);
        }
        correlationRangeTotal = correlationRange = parentDiagram->getCorrelationRangeTotal();
        for (auto& diagram : diagrams) {
            diagram->setCorrelationRange(correlationRangeTotal);
            diagram->setCorrelationRange(correlationRangeTotal);
        }
        resetSelections();
        dirty = true;
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (samplingMethodType != SamplingMethodType::MEAN && propertyEditor.addSliderIntEdit(
            "#Samples", &numSamples, 1, 1000) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setNumSamples(numSamples);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION && propertyEditor.addSliderIntEdit(
            "#numInitSamples", &numInitSamples, 1, 100) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setNumInitSamples(numInitSamples);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION && propertyEditor.addSliderIntEdit(
            "#numBOIterations", &numBOIterations, 1, 1000) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setNumBOIterations(numBOIterations);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
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
                        for (auto& diagram : diagrams) {
                            diagram->addScalarField(int(fieldIdx), text);
                        }

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
                        DiagramSelectedFieldData newFieldData(int(fieldIdx), text, DiagramColorMap(minIndex));

                        bool foundInsertionPosition = false;
                        for (size_t i = 0; i < selectedScalarFields.size(); i++) {
                            if (selectedScalarFields.at(i).first > int(fieldIdx)) {
                                selectedScalarFields.insert(
                                        selectedScalarFields.begin() + ptrdiff_t(i), newFieldData);
                                for (auto& diagram : diagrams) {
                                    diagram->setColorMap(int(i), newFieldData.third);
                                }
                                foundInsertionPosition = true;
                                break;
                            }
                        }
                        if (!foundInsertionPosition) {
                            selectedScalarFields.push_back(newFieldData);
                            for (auto& diagram : diagrams) {
                                diagram->setColorMap(int(selectedScalarFields.size() - 1), newFieldData.third);
                            }
                        }
                    } else {
                        for (auto& diagram : diagrams) {
                            diagram->removeScalarField(int(fieldIdx), false);
                        }
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
                    reRenderTriggeredByDiagram = true;
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
            reRenderTriggeredByDiagram = true;
        }
        ImGui::PopID();
    }

    if (propertyEditor.addCheckbox("Separate Color Rings", &separateColorVarianceAndCorrelation)) {
        for (auto& diagram : diagrams) {
            diagram->setUseSeparateColorVarianceAndCorrelation(separateColorVarianceAndCorrelation);
        }
    }

    if (separateColorVarianceAndCorrelation && propertyEditor.addCombo(
            "Color Ensemble Spread", (int*)&colorMapVariance, DIAGRAM_COLOR_MAP_NAMES,
            IM_ARRAYSIZE(DIAGRAM_COLOR_MAP_NAMES))) {
        for (auto& diagram : diagrams) {
            diagram->setColorMapVariance(colorMapVariance);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCheckbox("Value -> Opacity", &opacityByValue)) {
        for (auto& diagram : diagrams) {
            diagram->setOpacityByValue(opacityByValue);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCheckbox("Color Selections", &showSelectedRegionsByColor)) {
        for (auto& diagram : diagrams) {
            diagram->setShowSelectedRegionsByColor(showSelectedRegionsByColor);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderFloatEdit("beta", &beta, 0.0f, 1.0f) == ImGui::EditMode::INPUT_FINISHED) {
        for (auto& diagram : diagrams) {
            diagram->setBeta(beta);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (downscalingFactorUniform) {
        bool downscalingChanged;
        if (downscalingPowerOfTwo) {
            downscalingChanged = propertyEditor.addSliderIntPowerOfTwoEdit(
                    "Downscaling Context", &downscalingFactorX, minDownscalingFactor, maxDownscalingFactor)
                                    == ImGui::EditMode::INPUT_FINISHED;
        } else {
            downscalingChanged = propertyEditor.addSliderIntEdit(
                    "Downscaling Context", &downscalingFactorX, minDownscalingFactor, maxDownscalingFactor)
                                    == ImGui::EditMode::INPUT_FINISHED;
        }
        if (downscalingChanged) {
            downscalingFactorY = downscalingFactorX;
            downscalingFactorZ = downscalingFactorX;
            parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
            cellDistanceRange = cellDistanceRangeTotal = parentDiagram->getCellDistanceRangeTotal();
            parentDiagram->setCellDistanceRange(cellDistanceRange);
            resetSelections();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        if (downscalingPowerOfTwo) {
            downscalingChanged = propertyEditor.addSliderIntPowerOfTwoEdit(
                    "Downscaling Focus", &downscalingFactorFocusX, minDownscalingFactorFocus, maxDownscalingFactorFocus)
                                 == ImGui::EditMode::INPUT_FINISHED;
        } else {
            downscalingChanged = propertyEditor.addSliderIntEdit(
                    "Downscaling Focus", &downscalingFactorFocusX, minDownscalingFactorFocus, maxDownscalingFactorFocus)
                                 == ImGui::EditMode::INPUT_FINISHED;
        }
        if (downscalingChanged) {
            downscalingFactorFocusY = downscalingFactorFocusX;
            downscalingFactorFocusZ = downscalingFactorFocusX;
            resetSelections();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
    } else {
        ImGui::EditMode downscalingChanged;
        int dfs[3] = { downscalingFactorX, downscalingFactorY, downscalingFactorZ };
        if (downscalingPowerOfTwo) {
            downscalingChanged = propertyEditor.addSliderInt3PowerOfTwoEdit(
                    "Downscaling Context", dfs, minDownscalingFactor, maxDownscalingFactor);
        } else {
            downscalingChanged = propertyEditor.addSliderInt3Edit(
                    "Downscaling Context", dfs, minDownscalingFactor, maxDownscalingFactor);
        }
        downscalingFactorX = dfs[0];
        downscalingFactorY = dfs[1];
        downscalingFactorZ = dfs[2];
        if (downscalingChanged == ImGui::EditMode::INPUT_FINISHED) {
            parentDiagram->setDownscalingFactors(downscalingFactorX, downscalingFactorY, downscalingFactorZ);
            cellDistanceRange = cellDistanceRangeTotal = parentDiagram->getCellDistanceRangeTotal();
            parentDiagram->setCellDistanceRange(cellDistanceRange);
            resetSelections();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        int dfsf[3] = { downscalingFactorFocusX, downscalingFactorFocusY, downscalingFactorFocusZ };
        if (downscalingPowerOfTwo) {
            downscalingChanged = propertyEditor.addSliderInt3PowerOfTwoEdit(
                    "Downscaling Focus", dfsf, minDownscalingFactorFocus, maxDownscalingFactorFocus);
        } else {
            downscalingChanged = propertyEditor.addSliderInt3Edit(
                    "Downscaling Focus", dfsf, minDownscalingFactorFocus, maxDownscalingFactorFocus);
        }
        downscalingFactorFocusX = dfsf[0];
        downscalingFactorFocusY = dfsf[1];
        downscalingFactorFocusZ = dfsf[2];
        if (downscalingChanged == ImGui::EditMode::INPUT_FINISHED) {
            resetSelections();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }
    }
    propertyEditor.addCheckbox("Downscaling PoT", &downscalingPowerOfTwo);

    if (propertyEditor.addSliderIntEdit(
            "#Line Factor (Context)", &lineCountFactorContext, 10, 1000) == ImGui::EditMode::INPUT_FINISHED) {
        parentDiagram->setLineCountFactor(lineCountFactorContext);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderIntEdit(
            "#Line Factor (Focus)", &lineCountFactorFocus, 10, 1000) == ImGui::EditMode::INPUT_FINISHED) {
        for (size_t i = 1; i < diagrams.size(); i++) {
            diagrams.at(i)->setLineCountFactor(lineCountFactorFocus);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderFloat("Curve Thickness", &curveThickness, 0.5f, 4.0f)) {
        for (auto& diagram : diagrams) {
            diagram->setCurveThickness(curveThickness);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderFloat("Opacity Context", &curveOpacityContext, 0.0f, 1.0f)) {
        parentDiagram->setCurveOpacity(curveOpacityContext);
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
    if (propertyEditor.addSliderFloat("Opacity Focus", &curveOpacityFocus, 0.0f, 1.0f)) {
        for (size_t i = 1; i < diagrams.size(); i++) {
            diagrams.at(i)->setCurveOpacity(curveOpacityFocus);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderFloat2Edit(
            "Correlation Range", &correlationRange.x,
            correlationRangeTotal.x, correlationRangeTotal.y) == ImGui::EditMode::INPUT_FINISHED) {
        parentDiagram->setCorrelationRange(correlationRange);
        resetSelections();
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderInt2Edit(
            "Cell Dist. Range", &cellDistanceRange.x,
            cellDistanceRangeTotal.x, cellDistanceRangeTotal.y) == ImGui::EditMode::INPUT_FINISHED) {
        parentDiagram->setCellDistanceRange(cellDistanceRange);
        resetSelections();
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderInt("Diagram Radius", &diagramRadius, 100, 400)) {
        diagramRadius = std::max(diagramRadius, 1);
        for (auto& diagram : diagrams) {
            diagram->setDiagramRadius(diagramRadius);
        }
        resetSelections();
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addSliderFloat("Ring Size (%)", &outerRingSizePct, 0.0f, 1.0f)) {
        outerRingSizePct = std::clamp(outerRingSizePct, 0.0f, 1.0f);
        for (auto& diagram : diagrams) {
            diagram->setOuterRingSizePercentage(outerRingSizePct);
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.addCheckbox("Align with Window", &alignWithParentWindow)) {
        parentDiagram->setAlignWithParentWindow(alignWithParentWindow);
        if (renderOnlyLastFocusDiagram) {
            for (size_t i = 1; i < diagrams.size(); i++) {
                diagrams.at(i)->setAlignWithParentWindow(alignWithParentWindow);
            }
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }

    if (propertyEditor.beginNode("Advanced Settings")) {
        if (propertyEditor.addCheckbox("Use GPU Computations", &useCorrelationComputationGpu)) {
            for (auto& diagram : diagrams) {
                diagram->setUseCorrelationComputationGpu(useCorrelationComputationGpu);
            }
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        if (getSupportsBufferMode() && propertyEditor.addCombo(
                "Data Mode", (int*)&dataMode, DATA_MODE_NAMES, IM_ARRAYSIZE(DATA_MODE_NAMES))) {
            for (auto& diagram : diagrams) {
                diagram->setDataMode(dataMode);
            }
            dirty = true;
        }

        if (dataMode != CorrelationDataMode::IMAGE_3D_ARRAY && propertyEditor.addCheckbox(
                "Use Buffer Tiling", &useBufferTiling)) {
            for (auto& diagram : diagrams) {
                diagram->setUseBufferTiling(useBufferTiling);
            }
            dirty = true;
        }

        if (propertyEditor.addCheckbox("Focus: Only Selected Var.", &showOnlySelectedVariableInFocusDiagrams)) {
            resetSelections();
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        if (propertyEditor.addCheckbox("Focus+Context Mode", &renderOnlyLastFocusDiagram)) {
            reRender = true;
            reRenderTriggeredByDiagram = true;
            for (size_t i = 1; i < diagrams.size(); i++) {
                diagrams.at(i)->setAlignWithParentWindow(alignWithParentWindow && renderOnlyLastFocusDiagram);
            }
        }

        if (propertyEditor.addCheckbox("Alignment Rotation", &useAlignmentRotation)) {
            reRender = true;
        }

        if (propertyEditor.addCheckbox("Opaque Selection Boxes", &useOpaqueSelectionBoxes)) {
            reRender = true;
        }

        if (propertyEditor.addCheckbox("Desaturate Unselected Ring", &desaturateUnselectedRing)) {
            for (auto& diagram : diagrams) {
                diagram->setDesaturateUnselectedRing(desaturateUnselectedRing);
            }
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        if (propertyEditor.addCheckbox("Neon Selection Colors", &useNeonSelectionColors)) {
            for (auto& diagram : diagrams) {
                diagram->setUseNeonSelectionColors(useNeonSelectionColors);
            }
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        if (propertyEditor.addCheckbox("Global Ens. Spread Range", &useGlobalStdDevRange)) {
            for (auto& diagram : diagrams) {
                diagram->setUseGlobalStdDevRange(useGlobalStdDevRange);
            }
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        if (propertyEditor.addCombo(
                "Octree Method", (int*)&octreeMethod, OCTREE_METHOD_NAMES, IM_ARRAYSIZE(OCTREE_METHOD_NAMES))) {
            for (auto& diagram : diagrams) {
                diagram->setOctreeMethod(octreeMethod);
            }
            reRender = true;
            reRenderTriggeredByDiagram = true;
        }

        propertyEditor.endNode();
    }

    /*if (propertyEditor.addCheckbox("Use 2D Field", &use2dField)) {
         for (auto& diagram : diagrams) {
             diagram->setUse2DField(use2dField);
         }
         reRender = true;
         reRenderTriggeredByDiagram = true;
     }*/

    if (parentDiagram->renderGuiPropertyEditor(propertyEditor)) {
        for (size_t i = 1; i < diagrams.size(); i++) {
            diagrams.at(i)->copyVectorWidgetSettingsFrom(parentDiagram.get());
        }
        reRender = true;
        reRenderTriggeredByDiagram = true;
    }
}
