/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022-2023, Christoph Neuhauser
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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/AppSettings.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <Graphics/Vulkan/Utils/DeviceThreadInfo.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>
#include <Input/Mouse.hpp>
#include <Input/Keyboard.hpp>

#include "Utils/InternalState.hpp"
#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "Widgets/ViewManager.hpp"
#include "Correlation.hpp"
#include "MutualInformation.hpp"
#include "ReferencePointSelectionRenderer.hpp"
#include "PointPicker.hpp"
#include "CorrelationCalculator.hpp"

ICorrelationCalculator::ICorrelationCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
}

void ICorrelationCalculator::setViewManager(ViewManager* _viewManager) {
    viewManager = _viewManager;
    referencePointSelectionRenderer = new ReferencePointSelectionRenderer(viewManager);
    calculatorRenderer = RendererPtr(referencePointSelectionRenderer);
    referencePointSelectionRenderer->initialize();

    auto refPosSetter = [this](const glm::vec3& refPos) {
        this->setReferencePoint(refPos);
    };
    auto viewUsedIndexQuery = [this](int mouseHoverWindowIndex) -> bool {
        uint32_t varIdx = volumeData->getVarIdxForCalculator(this);
        return volumeData->getIsScalarFieldUsedInView(uint32_t(mouseHoverWindowIndex), varIdx, this);
    };
    pointPicker = std::make_shared<PointPicker>(
            viewManager, fixPickingZPlane, fixedZPlanePercentage, refPosSetter, viewUsedIndexQuery);
}

void ICorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (referencePointSelectionRenderer) {
        referencePointSelectionRenderer->setVolumeDataPtr(volumeData, isNewData);
    }
    pointPicker->setVolumeData(_volumeData, isNewData);

    int es = _volumeData->getEnsembleMemberCount();
    int ts = _volumeData->getTimeStepCount();
    if (isEnsembleMode && es <= 1 && ts > 1) {
        isEnsembleMode = false;
    } else if (!isEnsembleMode && ts <= 1 && es > 1) {
        isEnsembleMode = true;
    }

    scalarFieldNames = {};
    scalarFieldIndexArray = {};

    std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
    for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
        if (scalarFieldNamesNew.at(i) != getOutputFieldName()) {
            scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
            scalarFieldIndexArray.push_back(i);
        }
    }

    if (isNewData) {
        referencePointIndex.x = volumeData->getGridSizeX() / 2;
        referencePointIndex.y = volumeData->getGridSizeY() / 2;
        referencePointIndex.z = volumeData->getGridSizeZ() / 2;
        if (referencePointSelectionRenderer) {
            referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        }

        fieldIndex = fieldIndex2 = volumeData->getStandardScalarFieldIdx();
        fieldIndexGui = fieldIndex2Gui = volumeData->getStandardScalarFieldIdx();
        volumeData->acquireScalarField(this, fieldIndex);
        if (correlationFieldMode != CorrelationFieldMode::SINGLE) {
            volumeData->acquireScalarField(this, fieldIndex2);
        }

        if (!getSupportsBufferMode() || volumeData->getGridSizeZ() < 4) {
            dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        } else {
            dataMode = CorrelationDataMode::BUFFER_ARRAY;
        }
    }
}

int ICorrelationCalculator::getCorrelationMemberCount() const {
    return isEnsembleMode ? volumeData->getEnsembleMemberCount() : volumeData->getTimeStepCount();
}

VolumeData::HostCacheEntry ICorrelationCalculator::getFieldEntryCpu(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx) {
    VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx);
    return ensembleEntryField;
}

VolumeData::DeviceCacheEntry ICorrelationCalculator::getFieldEntryDevice(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx, bool wantsImageData) {
    VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx,
            wantsImageData, (!wantsImageData && useBufferTiling) ? glm::uvec3(8, 8, 4) : glm::uvec3(1, 1, 1));
    return ensembleEntryField;
}

std::pair<float, float> ICorrelationCalculator::getMinMaxScalarFieldValue(
        const std::string& fieldName, int fieldIdx, int timeStepIdx, int ensembleIdx) {
    return volumeData->getMinMaxScalarFieldValue(
            fieldName,
            isEnsembleMode ? timeStepIdx : fieldIdx,
            isEnsembleMode ? fieldIdx : ensembleIdx);
}

void ICorrelationCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (fieldIndex == fieldIdx) {
            fieldIndex = 0;
            volumeData->acquireScalarField(this, fieldIndex);
            dirty = true;
        } else if (fieldIndex > fieldIdx) {
            fieldIndex--;
        }
        fieldIndexGui = fieldIndex;

        if (fieldIndex2 == fieldIdx) {
            fieldIndex2 = 0;
            if (correlationFieldMode != CorrelationFieldMode::SINGLE) {
                volumeData->acquireScalarField(this, fieldIndex2);
                dirty = true;
            }
        } else if (fieldIndex2 > fieldIdx) {
            fieldIndex2--;
        }
        fieldIndex2Gui = fieldIndex2;
    }
}

void ICorrelationCalculator::update(float dt) {
    pointPicker->update(dt);

    if (continuousRecompute) {
        dirty = true;
    }
}

void ICorrelationCalculator::setReferencePoint(const glm::ivec3& referencePoint) {
    if (referencePointIndex != referencePoint) {
        glm::ivec3 maxCoord(
                volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
        referencePointIndex = referencePoint;
        referencePointIndex = glm::clamp(referencePointIndex, glm::ivec3(0), maxCoord);
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        if (useRenderRestriction) {
            setRenderRestrictionData();
        }
        dirty = true;
    }
}

void ICorrelationCalculator::setReferencePointFromWorld(const glm::vec3& worldPosition) {
    glm::ivec3 maxCoord(
            volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
    sgl::AABB3 gridAabb = volumeData->getBoundingBoxRendering();
    glm::vec3 position = (worldPosition - gridAabb.min) / (gridAabb.max - gridAabb.min);
    position *= glm::vec3(maxCoord);
    glm::ivec3 referencePointNew = glm::ivec3(glm::round(position));
    referencePointNew = glm::clamp(referencePointNew, glm::ivec3(0), maxCoord);
    if (referencePointIndex != referencePointNew) {
        referencePointIndex = referencePointNew;
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        dirty = true;
    }
}

void ICorrelationCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    bool needsScalarFieldData = getNeedsScalarFieldData();
    if (needsScalarFieldData && correlationFieldMode != CorrelationFieldMode::SINGLE) {
        if (propertyEditor.addCombo(
                "Scalar Field Reference", &fieldIndex2Gui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            clearFieldDeviceData();
            volumeData->releaseScalarField(this, fieldIndex2);
            fieldIndex2 = int(scalarFieldIndexArray.at(fieldIndex2Gui));
            volumeData->acquireScalarField(this, fieldIndex2);
            dirty = true;
        }
        if (propertyEditor.addCombo(
                "Scalar Field Query", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            clearFieldDeviceData();
            volumeData->releaseScalarField(this, fieldIndex);
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            volumeData->acquireScalarField(this, fieldIndex);
            dirty = true;
        }
    } else if (needsScalarFieldData && correlationFieldMode == CorrelationFieldMode::SINGLE) {
        if (propertyEditor.addCombo(
                "Scalar Field", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            clearFieldDeviceData();
            volumeData->releaseScalarField(this, fieldIndex);
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            volumeData->acquireScalarField(this, fieldIndex);
            dirty = true;
        }
    }

    if (volumeData->getEnsembleMemberCount() > 1 && volumeData->getTimeStepCount() > 1) {
        int modeIdx = isEnsembleMode ? 0 : 1;
        if (propertyEditor.addCombo("Correlation Mode", &modeIdx, CORRELATION_MODE_NAMES, 2)) {
            isEnsembleMode = modeIdx == 0;
            onCorrelationMemberCountChanged();
            clearFieldDeviceData();
            dirty = true;
        }
    }

    bool isRealtime = getIsRealtime();
    ImGui::EditMode editModes[3];
    editModes[0] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (X)", &referencePointIndex[0], 0, volumeData->getGridSizeX() - 1);
    editModes[1] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Y)", &referencePointIndex[1], 0, volumeData->getGridSizeY() - 1);
    editModes[2] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Z)", &referencePointIndex[2], 0, volumeData->getGridSizeZ() - 1);
    if ((isRealtime && (editModes[0] != ImGui::EditMode::NO_CHANGE
                || editModes[1] != ImGui::EditMode::NO_CHANGE
                || editModes[2] != ImGui::EditMode::NO_CHANGE))
            || (!isRealtime && (editModes[0] == ImGui::EditMode::INPUT_FINISHED
                || editModes[1] == ImGui::EditMode::INPUT_FINISHED
                || editModes[2] == ImGui::EditMode::INPUT_FINISHED))) {
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        dirty = true;
    }

    if (isEnsembleMode && useTimeLagCorrelations && volumeData->getTimeStepCount() > 1) {
        ImGui::EditMode editMode = propertyEditor.addSliderIntEdit(
                "Reference Time Step", &timeLagTimeStepIdx, 0, volumeData->getTimeStepCount() - 1);
        if ((isRealtime && editMode != ImGui::EditMode::NO_CHANGE)
                || (!isRealtime && editMode == ImGui::EditMode::INPUT_FINISHED)) {
            clearFieldDeviceData();
            dirty = true;
        }
    }

    if (scalarFieldNames.size() > 1 && getSupportsSeparateFields() && propertyEditor.addCombo(
            "Fields Mode", (int*)&correlationFieldMode,
            CORRELATION_FIELD_MODE_NAMES, IM_ARRAYSIZE(CORRELATION_FIELD_MODE_NAMES))) {
        if (correlationFieldMode != CorrelationFieldMode::SINGLE) {
            volumeData->acquireScalarField(this, fieldIndex2);
        } else {
            volumeData->releaseScalarField(this, fieldIndex2);
        }
        clearFieldDeviceData();
        dirty = true;
    }

    if (correlationFieldMode != CorrelationFieldMode::SINGLE && isEnsembleMode && volumeData->getTimeStepCount() > 1
            && propertyEditor.addCheckbox("Time Lag Correlations", &useTimeLagCorrelations)) {
        timeLagTimeStepIdx = volumeData->getCurrentTimeStepIdx();
        if (!useTimeLagCorrelations) {
            clearFieldDeviceData();
            dirty = true;
        }
    }

    renderGuiImplSub(propertyEditor);

    if (propertyEditor.beginNode("Advanced Settings")) {
        renderGuiImplAdvanced(propertyEditor);
        propertyEditor.endNode();
    }
}

void ICorrelationCalculator::renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) {
    if (!scalarFieldNames.empty() && getSupportsBufferMode() && propertyEditor.addCombo(
            "Data Mode", (int*)&dataMode, DATA_MODE_NAMES, IM_ARRAYSIZE(DATA_MODE_NAMES))) {
        clearFieldDeviceData();
        dirty = true;
    }

    if (dataMode != CorrelationDataMode::IMAGE_3D_ARRAY && propertyEditor.addCheckbox(
            "Use Buffer Tiling", &useBufferTiling)) {
        clearFieldDeviceData();
        dirty = true;
    }

    if (!volumeData || volumeData->getGridSizeZ() > 1) {
        propertyEditor.addCheckbox("Fix Picking Z", &fixPickingZPlane);
        auto fixedZPlane = int(fixedZPlanePercentage * float(volumeData->getGridSizeZ()));
        if (fixPickingZPlane && propertyEditor.addSliderInt(
                "Z Plane", &fixedZPlane, 0, volumeData->getGridSizeZ() - 1)) {
            fixedZPlanePercentage = float(fixedZPlane) / float(volumeData->getGridSizeZ());
            pointPicker->onUpdatePositionFixed();
        }
        if (fixPickingZPlane) {
            volumeData->displayLayerInfo(propertyEditor, fixedZPlane);
        }
    }

    bool useRenderRestrictionDirty = false;
    if (propertyEditor.addCheckbox("Restrict Rendering", &useRenderRestriction)) {
        useRenderRestrictionDirty = true;
    }
    if (useRenderRestriction) {
        if (propertyEditor.addSliderFloat("Rendering Radius", &renderRestrictionRadius, 0.01f, 0.5f)) {
            useRenderRestrictionDirty = true;
        }
        if (propertyEditor.addCombo(
                "Distance Metric", (int*)&renderRestrictionDistanceMetric,
                DISTANCE_METRIC_NAMES, IM_ARRAYSIZE(DISTANCE_METRIC_NAMES))) {
            useRenderRestrictionDirty = true;
        }
    }
    if (useRenderRestrictionDirty) {
        setRenderRestrictionData();
    }
}

void ICorrelationCalculator::setRenderRestrictionData() {
    if (useRenderRestriction) {
        glm::ivec3 maxCoord(
                volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
        glm::vec3 normalizedPosition = glm::vec3(referencePointIndex) / glm::vec3(maxCoord);
        sgl::AABB3 gridAabb = volumeData->getBoundingBoxRendering();
        auto position = normalizedPosition * (gridAabb.max - gridAabb.min) + gridAabb.min;
        volumeData->setRenderRestriction(
                this, renderRestrictionDistanceMetric, position, renderRestrictionRadius);
    } else {
        volumeData->resetRenderRestriction(this);
    }
}

bool ICorrelationCalculator::getSupportsBufferMode() {
    bool supportsBufferMode = true;
    if (!volumeData->getScalarFieldSupportsBufferMode(fieldIndex)) {
        supportsBufferMode = false;
    }
    if (correlationFieldMode != CorrelationFieldMode::SINGLE && !volumeData->getScalarFieldSupportsBufferMode(fieldIndex2)) {
        supportsBufferMode = false;
    }
    if (!supportsBufferMode && dataMode == CorrelationDataMode::BUFFER_ARRAY) {
        dataMode = CorrelationDataMode::IMAGE_3D_ARRAY;
        dirty = true;
    }
    return supportsBufferMode;
}

bool ICorrelationCalculator::getSupportsSeparateFields() {
    return true;
}

void ICorrelationCalculator::setSettings(const SettingsMap& settings) {
    Calculator::setSettings(settings);

    bool useSeparateFields = false;
    if (settings.getValueOpt("use_separate_fields", useSeparateFields)) {
        if (correlationFieldMode != CorrelationFieldMode::SINGLE) {
            volumeData->acquireScalarField(this, fieldIndex2);
        } else {
            volumeData->releaseScalarField(this, fieldIndex2);
        }
        clearFieldDeviceData();
        dirty = true;
    }

    std::string correlationFieldModeString;
    if (settings.getValueOpt("correlation_field_mode", correlationFieldModeString)) {
        for (int i = 0; i < IM_ARRAYSIZE(CORRELATION_FIELD_MODE_NAMES); i++) {
            if (correlationFieldModeString == CORRELATION_FIELD_MODE_NAMES[i]) {
                correlationFieldMode = CorrelationFieldMode(i);
                break;
            }
        }
        if (correlationFieldMode != CorrelationFieldMode::SINGLE) {
            volumeData->acquireScalarField(this, fieldIndex2);
        } else {
            volumeData->releaseScalarField(this, fieldIndex2);
        }
        clearFieldDeviceData();
        dirty = true;
    }

    bool needsScalarFieldData = getNeedsScalarFieldData();
    if (needsScalarFieldData && correlationFieldMode != CorrelationFieldMode::SINGLE) {
        if (settings.getValueOpt("scalar_field_idx_ref", fieldIndex2Gui)) {
            clearFieldDeviceData();
            volumeData->releaseScalarField(this, fieldIndex2);
            fieldIndex2 = int(scalarFieldIndexArray.at(fieldIndex2Gui));
            volumeData->acquireScalarField(this, fieldIndex2);
            dirty = true;
        }
        if (settings.getValueOpt("scalar_field_idx_query", fieldIndexGui)) {
            clearFieldDeviceData();
            volumeData->releaseScalarField(this, fieldIndex);
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            volumeData->acquireScalarField(this, fieldIndex);
            dirty = true;
        }
    } else {
        if (settings.getValueOpt("scalar_field_idx", fieldIndexGui)) {
            clearFieldDeviceData();
            volumeData->releaseScalarField(this, fieldIndex);
            fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
            volumeData->acquireScalarField(this, fieldIndex);
            dirty = true;
        }
    }

    std::string ensembleModeName;
    if (settings.getValueOpt("correlation_mode", ensembleModeName)) {
        if (ensembleModeName == CORRELATION_MODE_NAMES[0]) {
            isEnsembleMode = true;
        } else {
            isEnsembleMode = false;
        }
        onCorrelationMemberCountChanged();
        clearFieldDeviceData();
        dirty = true;
    }

    bool referencePointChanged = false;
    referencePointChanged |= settings.getValueOpt("reference_point_x", referencePointIndex[0]);
    referencePointChanged |= settings.getValueOpt("reference_point_y", referencePointIndex[1]);
    referencePointChanged |= settings.getValueOpt("reference_point_z", referencePointIndex[2]);
    if (referencePointChanged) {
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        pointPicker->overwriteFocusPointFromRefPoint(referencePointIndex);
        dirty = true;
    }

    // Advanced settings.
    std::string dataModeString;
    if (settings.getValueOpt("data_mode", dataModeString)) {
        for (int i = 0; i < IM_ARRAYSIZE(DATA_MODE_NAMES); i++) {
            if (dataModeString == DATA_MODE_NAMES[i]) {
                dataMode = CorrelationDataMode(i);
                break;
            }
        }
        clearFieldDeviceData();
        dirty = true;
    }
    if (settings.getValueOpt("use_buffer_tiling", useBufferTiling)) {
        clearFieldDeviceData();
        dirty = true;
    }
    settings.getValueOpt("fix_picking_z", fixPickingZPlane);
    if (settings.getValueOpt("fixed_z_plane_percentage", fixedZPlanePercentage)) {
        pointPicker->onUpdatePositionFixed();
    }

    if (settings.getValueOpt("use_time_lag_correlations", useTimeLagCorrelations)) {
        clearFieldDeviceData();
        dirty = true;
    }
    if (settings.getValueOpt("time_lag_time_step_idx", timeLagTimeStepIdx)) {
        clearFieldDeviceData();
        dirty = true;
    }

    bool useRenderRestrictionDirty = false;
    if (settings.getValueOpt("restrict_rendering", useRenderRestriction)) {
        useRenderRestrictionDirty = true;
    }
    if (useRenderRestriction) {
        if (settings.getValueOpt("render_restriction_radius", renderRestrictionRadius)) {
            useRenderRestrictionDirty = true;
        }
        std::string distanceMetricString;
        if (settings.getValueOpt("distance_metric", distanceMetricString)) {
            for (int i = 0; i < IM_ARRAYSIZE(DISTANCE_METRIC_NAMES); i++) {
                if (distanceMetricString == DISTANCE_METRIC_NAMES[i]) {
                    renderRestrictionDistanceMetric = DistanceMetric(i);
                    break;
                }
            }
            clearFieldDeviceData();
            dirty = true;
        }
    }
    if (useRenderRestrictionDirty) {
        setRenderRestrictionData();
    }
}

void ICorrelationCalculator::getSettings(SettingsMap& settings) {
    Calculator::getSettings(settings);

    //settings.addKeyValue("use_separate_fields", useSeparateFields);
    settings.addKeyValue("correlation_field_mode", CORRELATION_FIELD_MODE_NAMES[int(correlationFieldMode)]);
    bool needsScalarFieldData = getNeedsScalarFieldData();
    if (needsScalarFieldData && correlationFieldMode != CorrelationFieldMode::SINGLE) {
        settings.addKeyValue("scalar_field_idx_ref", fieldIndex2Gui);
        settings.addKeyValue("scalar_field_idx_query", fieldIndexGui);
    } else {
        settings.addKeyValue("scalar_field_idx", fieldIndexGui);
    }

    settings.addKeyValue("correlation_mode", CORRELATION_MODE_NAMES[isEnsembleMode ? 0 : 1]);
    settings.addKeyValue("reference_point_x", referencePointIndex[0]);
    settings.addKeyValue("reference_point_y", referencePointIndex[1]);
    settings.addKeyValue("reference_point_z", referencePointIndex[2]);

    // Advanced settings.
    settings.addKeyValue("data_mode", DATA_MODE_NAMES[int(dataMode)]);
    settings.addKeyValue("use_buffer_tiling", useBufferTiling);
    settings.addKeyValue("fix_picking_z", fixPickingZPlane);
    settings.addKeyValue("fixed_z_plane_percentage", fixedZPlanePercentage);

    if (getSupportsSeparateFields()) {
        settings.addKeyValue("use_time_lag_correlations", useTimeLagCorrelations);
        settings.addKeyValue("time_lag_time_step_idx", timeLagTimeStepIdx);
    }

    settings.addKeyValue("restrict_rendering", useRenderRestriction);
    if (useRenderRestriction) {
        settings.addKeyValue("render_restriction_radius", renderRestrictionRadius);
        settings.addKeyValue("distance_metric", DISTANCE_METRIC_NAMES[int(renderRestrictionDistanceMetric)]);
    }
}



CorrelationCalculator::CorrelationCalculator(sgl::vk::Renderer* renderer) : ICorrelationCalculator(renderer) {
#ifdef SUPPORT_CUDA_INTEROP
    useCuda = sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() && sgl::vk::getIsNvrtcFunctionTableInitialized();
#endif

    correlationComputePass = std::make_shared<CorrelationComputePass>(renderer);
    correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
    correlationComputePass->setNumBins(numBins);
    correlationComputePass->setKraskovNumNeighbors(k);
    correlationComputePass->setKraskovEstimatorIndex(kraskovEstimatorIndex);
}

void CorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    ICorrelationCalculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::CORRELATION);
    }
    correlationComputePass->setVolumeData(volumeData, getCorrelationMemberCount(), isNewData);
    if (isNewData || cachedMemberCount != getCorrelationMemberCount()) {
        onCorrelationMemberCountChanged();
    }
}

void CorrelationCalculator::onCorrelationMemberCountChanged() {
    int cs = getCorrelationMemberCount();
    k = std::max(sgl::iceil(3 * cs, 100), 1);
    kMax = std::max(sgl::iceil(7 * cs, 100), 20);
    correlationComputePass->setCorrelationMemberCount(cs);
    correlationComputePass->setKraskovNumNeighbors(k);
    cachedMemberCount = cs;

#ifdef SUPPORT_CUDA_INTEROP
    bool canUseCuda =
            sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() && sgl::vk::getIsNvrtcFunctionTableInitialized();
#else
    bool canUseCuda = false;
#endif
    referenceValuesBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), cs * sizeof(float),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY, true, canUseCuda, false);
#ifdef SUPPORT_CUDA_INTEROP
    if (canUseCuda) {
        referenceValuesCudaBuffer = std::make_shared<sgl::vk::BufferCudaExternalMemoryVk>(referenceValuesBuffer);
    }
#endif
    referenceValuesCpu.resize(cs);
}

void CorrelationCalculator::clearFieldDeviceData() {
    correlationComputePass->setFieldImageViews({});
    correlationComputePass->setFieldImageViewsSecondary({});
    correlationComputePass->setFieldBuffers({});
    correlationComputePass->setFieldBuffersSecondary({});
    correlationComputePass->setReferenceValuesBuffer({});
#ifdef SUPPORT_CUDA_INTEROP
    correlationComputePass->setReferenceValuesCudaBuffer({});
#endif
}

bool CorrelationCalculator::getIsRealtime() const {
    if (!useGpu) {
        return false;
    }
    return correlationMeasureType == CorrelationMeasureType::PEARSON || getCorrelationMemberCount() < 200;
}

FilterDevice CorrelationCalculator::getFilterDevice() {
    if (useGpu) {
        if (useCuda) {
            return FilterDevice::CUDA;
        } else {
            return FilterDevice::VULKAN;
        }
    }
    return FilterDevice::CPU;
}

void CorrelationCalculator::renderGuiImplSub(sgl::PropertyEditor& propertyEditor) {
    ICorrelationCalculator::renderGuiImplSub(propertyEditor);
    if (propertyEditor.addCombo(
            "Correlation Measure", (int*)&correlationMeasureType,
            CORRELATION_MEASURE_TYPE_NAMES, IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_NAMES))) {
        hasNameChanged = true;
        dirty = true;
        correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
    }

#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() && sgl::vk::getIsNvrtcFunctionTableInitialized()
            && isMeasureKraskovMI(correlationMeasureType)) {
        const char* const choices[] = {
                "CPU", "Vulkan", "CUDA"
        };
        int choice = !useGpu ? 0 : (!useCuda ? 1 : 2);
        if (propertyEditor.addCombo("Device", &choice, choices, 3)) {
            bool useGpuOld = useGpu;
            useGpu = choice != 0;
            useCuda = choice != 1;
            hasFilterDeviceChanged = useGpuOld != useGpu;
            dirty = true;
        }
    } else
#endif
    if (propertyEditor.addCheckbox("Use GPU", &useGpu)) {
        hasFilterDeviceChanged = true;
        dirty = true;
    }

    if (!isMeasureMI(correlationMeasureType) && propertyEditor.addCheckbox("Absolute Value", &calculateAbsoluteValue)) {
        correlationComputePass->setCalculateAbsoluteValue(calculateAbsoluteValue);
        dirty = true;
    }

    if (isMeasureBinnedMI(correlationMeasureType) && propertyEditor.addSliderIntEdit(
            "#Bins", &numBins, 10, 100) == ImGui::EditMode::INPUT_FINISHED) {
        correlationComputePass->setNumBins(numBins);
        dirty = true;
    }
    if (isMeasureKraskovMI(correlationMeasureType) && propertyEditor.addSliderIntEdit(
            "#Neighbors", &k, 1, kMax) == ImGui::EditMode::INPUT_FINISHED) {
        correlationComputePass->setKraskovNumNeighbors(k);
        dirty = true;
    }
    if (isMeasureKraskovMI(correlationMeasureType) && propertyEditor.addSliderIntEdit(
            "Kraskov Estimator", &kraskovEstimatorIndex, 1, 2) == ImGui::EditMode::INPUT_FINISHED) {
        kraskovEstimatorIndex = std::clamp(kraskovEstimatorIndex, 1, 2);
        correlationComputePass->setKraskovEstimatorIndex(kraskovEstimatorIndex);
        dirty = true;
    }

#ifdef SHOW_DEBUG_OPTIONS
    if (propertyEditor.addCheckbox("Continuous Recompute", &continuousRecompute)) {
        dirty = true;
    }
#endif
}

void CorrelationCalculator::setSettings(const SettingsMap& settings) {
    ICorrelationCalculator::setSettings(settings);

    std::string correlationMeasureTypeName;
    if (settings.getValueOpt("correlation_measure_type", correlationMeasureTypeName)) {
        for (int i = 0; i < IM_ARRAYSIZE(CORRELATION_MEASURE_TYPE_IDS); i++) {
            if (correlationMeasureTypeName == CORRELATION_MEASURE_TYPE_IDS[i]) {
                correlationMeasureType = CorrelationMeasureType(i);
                break;
            }
        }
        hasNameChanged = true;
        dirty = true;
        correlationComputePass->setCorrelationMeasureType(correlationMeasureType);
    }
    std::string deviceName;
    if (settings.getValueOpt("device", deviceName)) {
        const char* const choices[] = {
                "CPU", "Vulkan", "CUDA"
        };
        int choice = 1;
        for (int i = 0; i < 3; i++) {
            if (deviceName == choices[i]) {
                choice = i;
                break;
            }
        }
#ifndef SUPPORT_CUDA_INTEROP
        if (choice == 2) {
            choice = 1;
        }
#else
        if (choice == 2 && !sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            choice = 1;
        }
#endif
        bool useGpuOld = useGpu;
        useGpu = choice != 0;
        useCuda = choice != 1;
        hasFilterDeviceChanged = useGpuOld != useGpu;
        dirty = true;
    }
    if (settings.getValueOpt("calculate_absolute_value", calculateAbsoluteValue)) {
        correlationComputePass->setCalculateAbsoluteValue(calculateAbsoluteValue);
        dirty = true;
    }
    if (settings.getValueOpt("mi_bins", numBins)) {
        correlationComputePass->setNumBins(numBins);
        dirty = true;
    }
    if (settings.getValueOpt("kmi_neighbors", k)) {
        correlationComputePass->setKraskovNumNeighbors(k);
        dirty = true;
    }
    if (settings.getValueOpt("kraskov_estimator_index", kraskovEstimatorIndex)) {
        kraskovEstimatorIndex = std::clamp(kraskovEstimatorIndex, 1, 2);
        correlationComputePass->setKraskovEstimatorIndex(kraskovEstimatorIndex);
        dirty = true;
    }
}

void CorrelationCalculator::getSettings(SettingsMap& settings) {
    ICorrelationCalculator::getSettings(settings);

    settings.addKeyValue("correlation_measure_type", CORRELATION_MEASURE_TYPE_IDS[int(correlationMeasureType)]);
    const char* const choices[] = {
            "CPU", "Vulkan", "CUDA"
    };
    settings.addKeyValue("device", choices[!useGpu ? 0 : (!useCuda ? 1 : 2)]);
    settings.addKeyValue("calculate_absolute_value", calculateAbsoluteValue);
    settings.addKeyValue("mi_bins", numBins);
    settings.addKeyValue("kmi_neighbors", k);
    settings.addKeyValue("kraskov_estimator_index", kraskovEstimatorIndex);
}

void CorrelationCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int cs = getCorrelationMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

    std::vector<VolumeData::HostCacheEntry> fieldEntries;
    std::vector<const float*> fields;
    fieldEntries.reserve(cs);
    fields.reserve(cs);
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(
                scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
        fieldEntries.push_back(fieldEntry);
        fields.push_back(fieldEntry->data<float>());
    }

    size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
    auto* referenceValues = new float[cs];
    if (correlationFieldMode == CorrelationFieldMode::SEPARATE) {
        int timeStepIdxReference = timeStepIdx;
        if (useTimeLagCorrelations) {
            timeStepIdxReference = timeLagTimeStepIdx;
        }
        for (int c = 0; c < cs; c++) {
            VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(
                    scalarFieldNames.at(fieldIndex2Gui), c, timeStepIdxReference, ensembleIdx);
            referenceValues[c] = fieldEntry->dataAt<float>(referencePointIdx);
        }
    } else {
        for (int c = 0; c < cs; c++) {
            referenceValues[c] = fields.at(c)[referencePointIdx];
        }
    }

    float minFieldValRef = std::numeric_limits<float>::max();
    float maxFieldValRef = std::numeric_limits<float>::lowest();
    if (isMeasureBinnedMI(correlationMeasureType)) {
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(
                    scalarFieldNames.at(correlationFieldMode != CorrelationFieldMode::SINGLE ? fieldIndex2Gui : fieldIndexGui),
                    fieldIdx, timeStepIdx, ensembleIdx);
            minFieldValRef = std::min(minFieldValRef, minVal);
            maxFieldValRef = std::max(maxFieldValRef, maxVal);
        }
        for (int c = 0; c < cs; c++) {
            referenceValues[c] = (referenceValues[c] - minFieldValRef) / (maxFieldValRef - minFieldValRef);
        }
    }
    float minFieldValQuery = std::numeric_limits<float>::max();
    float maxFieldValQuery = std::numeric_limits<float>::lowest();
    if (correlationFieldMode != CorrelationFieldMode::SINGLE && isMeasureBinnedMI(correlationMeasureType)) {
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            auto [minVal, maxVal] = getMinMaxScalarFieldValue(
                    scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
            minFieldValQuery = std::min(minFieldValQuery, minVal);
            maxFieldValQuery = std::max(maxFieldValQuery, maxVal);
        }
    } else if (isMeasureBinnedMI(correlationMeasureType)) {
        minFieldValQuery = minFieldValRef;
        maxFieldValQuery = maxFieldValRef;
    }


#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    float* referenceRanks = nullptr;
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        referenceRanks = new float[cs];
        std::vector<std::pair<float, int>> ordinalRankArray;
        ordinalRankArray.reserve(cs);
        computeRanks(referenceValues, referenceRanks, ordinalRankArray, cs);
    }

    size_t numGridPoints = size_t(xs) * size_t(ys) * size_t(zs);
    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* referenceValues = new float[cs];
            for (int c = 0; c < cs; c++) {
                referenceValues[c] = fields.at(c)[referencePointIdx];
            }
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel for shared(numGridPoints, cs, referenceValues, fields, buffer) default(none)
#endif
        for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
            if (cs == 1) {
                buffer[gridPointIdx] = 1.0f;
                continue;
            }
            // See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
#define FORMULA_2_FLOAT
#ifdef FORMULA_1_FLOAT
            float pearsonCorrelation = computePearson1<float>(referenceValues, fields, cs, gridPointIdx);
#elif defined(FORMULA_1_DOUBLE)
            float pearsonCorrelation = computePearson1<double>(referenceValues, fields, cs, gridPointIdx);
#elif defined(FORMULA_2_FLOAT)
            float pearsonCorrelation = computePearson2<float>(referenceValues, fields, cs, gridPointIdx);
#elif defined(FORMULA_2_DOUBLE)
            float pearsonCorrelation = computePearson2<double>(referenceValues, fields, cs, gridPointIdx);
#endif
            buffer[gridPointIdx] = pearsonCorrelation;
        }
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[cs];
            auto* gridPointRanks = new float[cs];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(cs);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, referenceRanks, fields, buffer) default(none)
        {
#endif
            auto* gridPointValues = new float[cs];
            auto* gridPointRanks = new float[cs];
            std::vector<std::pair<float, int>> ordinalRankArray;
            ordinalRankArray.reserve(cs);
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    gridPointValues[c] = fields.at(c)[gridPointIdx];
                    if (std::isnan(gridPointValues[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                computeRanks(gridPointValues, gridPointRanks, ordinalRankArray, cs);

#define FORMULA_2_FLOAT
#ifdef FORMULA_1_FLOAT
                float pearsonCorrelation = computePearson1<float>(referenceRanks, gridPointRanks, cs);
#elif defined(FORMULA_1_DOUBLE)
                float pearsonCorrelation = computePearson1<double>(referenceRanks, gridPointRanks, cs);
#elif defined(FORMULA_2_FLOAT)
                float pearsonCorrelation = computePearson2<float>(referenceRanks, gridPointRanks, cs);
#elif defined(FORMULA_2_DOUBLE)
                float pearsonCorrelation = computePearson2<double>(referenceRanks, gridPointRanks, cs);
#endif
                buffer[gridPointIdx] = pearsonCorrelation;
            }
            delete[] gridPointValues;
            delete[] gridPointRanks;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[cs];
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            std::vector<float> sortArray;
            std::vector<std::pair<int, int>> stack;
            jointArray.reserve(cs);
            ordinalRankArray.reserve(cs);
            y.reserve(cs);
            sortArray.reserve(cs);
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, referenceValues, fields, buffer) default(none)
        {
#endif
            auto* gridPointValues = new float[cs];
            std::vector<std::pair<float, float>> jointArray;
            std::vector<float> ordinalRankArray;
            std::vector<float> y;
            std::vector<float> sortArray;
            std::vector<std::pair<int, int>> stack;
            jointArray.reserve(cs);
            ordinalRankArray.reserve(cs);
            y.reserve(cs);
            sortArray.reserve(cs);
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    gridPointValues[c] = fields.at(c)[gridPointIdx];
                    if (std::isnan(gridPointValues[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float pearsonCorrelation = computeKendall<int32_t>(
                        referenceValues, gridPointValues, cs, jointArray, ordinalRankArray, y, sortArray, stack);
                buffer[gridPointIdx] = pearsonCorrelation;
            }
            delete[] gridPointValues;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (isMeasureBinnedMI(correlationMeasureType)) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[cs];
            auto* histogram0 = new double[numBins];
            auto* histogram1 = new double[numBins];
            auto* histogram2d = new double[numBins * numBins];
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, referenceValues, fields, buffer) default(none) \
        shared(minFieldValQuery, maxFieldValQuery)
        {
#endif
            auto* gridPointValues = new float[cs];
            auto* histogram0 = new double[numBins];
            auto* histogram1 = new double[numBins];
            auto* histogram2d = new double[numBins * numBins];
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    gridPointValues[c] = fields.at(c)[gridPointIdx];
                    if (std::isnan(gridPointValues[c])) {
                        isNan = true;
                        break;
                    }
                    gridPointValues[c] =
                            (gridPointValues[c] - minFieldValQuery) / (maxFieldValQuery - minFieldValQuery);
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation = computeMutualInformationBinned<double>(
                        referenceValues, gridPointValues, numBins, cs, histogram0, histogram1, histogram2d);
                if (correlationMeasureType == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT) {
                    mutualInformation = std::sqrt(1.0f - std::exp(-2.0f * mutualInformation));
                }
                buffer[gridPointIdx] = mutualInformation;
            }
            delete[] gridPointValues;
            delete[] histogram0;
            delete[] histogram1;
            delete[] histogram2d;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    } else if (isMeasureKraskovMI(correlationMeasureType)) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
            auto* gridPointValues = new float[cs];
            KraskovEstimatorCache<double> kraskovEstimatorCache;
            for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel shared(numGridPoints, cs, k, referenceValues, fields, buffer) default(none)
        {
#endif
            auto* gridPointValues = new float[cs];
            KraskovEstimatorCache<double> kraskovEstimatorCache;
#if _OPENMP >= 200805
            #pragma omp for
#endif
            for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
                if (cs == 1) {
                    buffer[gridPointIdx] = 1.0f;
                    continue;
                }

                bool isNan = false;
                for (int c = 0; c < cs; c++) {
                    gridPointValues[c] = fields.at(c)[gridPointIdx];
                    if (std::isnan(gridPointValues[c])) {
                        isNan = true;
                        break;
                    }
                }
                if (isNan) {
                    buffer[gridPointIdx] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }

                float mutualInformation;
                if (kraskovEstimatorIndex == 1) {
                    mutualInformation = computeMutualInformationKraskov<double>(
                            referenceValues, gridPointValues, k, cs, kraskovEstimatorCache);
                } else {
                    mutualInformation = computeMutualInformationKraskov2<double>(
                            referenceValues, gridPointValues, k, cs, kraskovEstimatorCache);
                }
                if (correlationMeasureType == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT) {
                    mutualInformation = std::sqrt(1.0f - std::exp(-2.0f * mutualInformation));
                }
                buffer[gridPointIdx] = mutualInformation;
            }
            delete[] gridPointValues;
#if _OPENMP >= 200805
        }
#endif
#ifdef USE_TBB
        });
#endif
    }

    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        delete[] referenceRanks;
    }
    delete[] referenceValues;

#ifdef TEST_INFERENCE_SPEED
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}

void CorrelationCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    // We write to the descriptor set, so wait until the device is idle.
    renderer->getDevice()->waitIdle();

    int cs = getCorrelationMemberCount();

#ifdef TEST_INFERENCE_SPEED
    auto startLoad = std::chrono::system_clock::now();
#endif

    std::vector<VolumeData::DeviceCacheEntry> fieldEntries;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    std::vector<sgl::vk::BufferPtr> fieldBuffers;
    std::vector<VolumeData::DeviceCacheEntry> fieldEntriesSecondary;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViewsSecondary;
    std::vector<sgl::vk::BufferPtr> fieldBuffersSecondary;
    correlationComputePass->setDataMode(dataMode);
    correlationComputePass->setUseBufferTiling(useBufferTiling);
    correlationComputePass->setCorrelationFieldMode(correlationFieldMode);
    bool useImageArray = dataMode == CorrelationDataMode::IMAGE_3D_ARRAY;
    fieldEntries.reserve(cs);
    if (useImageArray) {
        fieldBuffers.reserve(cs);
    } else {
        fieldImageViews.reserve(cs);
    }
    if (correlationFieldMode == CorrelationFieldMode::SEPARATE_SYMMETRIC) {
        fieldEntriesSecondary.reserve(cs);
        if (useImageArray) {
            fieldBuffersSecondary.reserve(cs);
        } else {
            fieldImageViewsSecondary.reserve(cs);
        }
    }
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::DeviceCacheEntry fieldEntry = getFieldEntryDevice(
                scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx, useImageArray);
        fieldEntries.push_back(fieldEntry);
        if (useImageArray) {
            fieldImageViews.push_back(fieldEntry->getVulkanImageView());
            if (fieldEntry->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                fieldEntry->getVulkanImage()->transitionImageLayout(
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
            }
        } else {
            fieldBuffers.push_back(fieldEntry->getVulkanBuffer());
        }
        if (correlationFieldMode == CorrelationFieldMode::SEPARATE_SYMMETRIC) {
            VolumeData::DeviceCacheEntry fieldEntrySecondary = getFieldEntryDevice(
                    scalarFieldNames.at(fieldIndex2Gui), fieldIdx, timeStepIdx, ensembleIdx, useImageArray);
            fieldEntriesSecondary.push_back(fieldEntrySecondary);
            if (useImageArray) {
                fieldImageViewsSecondary.push_back(fieldEntrySecondary->getVulkanImageView());
                if (fieldEntrySecondary->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                    fieldEntrySecondary->getVulkanImage()->transitionImageLayout(
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
                }
            } else {
                fieldBuffersSecondary.push_back(fieldEntrySecondary->getVulkanBuffer());
            }
        }
    }
    correlationComputePass->setUseSecondaryFields(correlationFieldMode == CorrelationFieldMode::SEPARATE_SYMMETRIC);
    if (useImageArray) {
        correlationComputePass->setFieldImageViews(fieldImageViews);
    } else {
        correlationComputePass->setFieldBuffers(fieldBuffers);
    }
    if (correlationFieldMode == CorrelationFieldMode::SEPARATE_SYMMETRIC) {
        if (useImageArray) {
            correlationComputePass->setFieldImageViewsSecondary(fieldImageViewsSecondary);
        } else {
            correlationComputePass->setFieldBuffersSecondary(fieldBuffersSecondary);
        }
    }
    if (correlationFieldMode == CorrelationFieldMode::SEPARATE) {
        int xs = volumeData->getGridSizeX();
        int ys = volumeData->getGridSizeY();
        size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
        int timeStepIdxReference = timeStepIdx;
        if (useTimeLagCorrelations) {
            timeStepIdxReference = timeLagTimeStepIdx;
        }
        for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
            VolumeData::HostCacheEntry fieldEntry = getFieldEntryCpu(
                    scalarFieldNames.at(fieldIndex2Gui), fieldIdx, timeStepIdxReference, ensembleIdx);
            referenceValuesCpu[fieldIdx] = fieldEntry->dataAt<float>(referencePointIdx);
        }
        referenceValuesBuffer->updateData(
                sizeof(float) * cs, referenceValuesCpu.data(), renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                referenceValuesBuffer);
        correlationComputePass->setReferenceValuesBuffer(referenceValuesBuffer);
#ifdef SUPPORT_CUDA_INTEROP
        correlationComputePass->setReferenceValuesCudaBuffer(referenceValuesCudaBuffer);
#endif
    }

#ifdef TEST_INFERENCE_SPEED
    auto endLoad = std::chrono::system_clock::now();
    auto elapsedLoad = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);
    std::cout << "Elapsed time load: " << elapsedLoad.count() << "ms" << std::endl;
#endif

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    // TODO: Support CorrelationFieldMode::SEPARATE_SYMMETRIC with CUDA.
    if (!isMeasureKraskovMI(correlationMeasureType) || !useCuda
            || correlationFieldMode == CorrelationFieldMode::SEPARATE_SYMMETRIC) {
        correlationComputePass->setOutputImage(deviceCacheEntry->getVulkanImageView());

        renderer->insertImageMemoryBarrier(
                deviceCacheEntry->getVulkanImage(),
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);

        correlationComputePass->buildIfNecessary();
        correlationComputePass->setReferencePoint(referencePointIndex);
        if (isMeasureBinnedMI(correlationMeasureType)) {
            if (correlationFieldMode != CorrelationFieldMode::SINGLE) {
                float minFieldValRef = std::numeric_limits<float>::max();
                float maxFieldValRef = std::numeric_limits<float>::lowest();
                float minFieldValQuery = std::numeric_limits<float>::max();
                float maxFieldValQuery = std::numeric_limits<float>::lowest();
                if (isMeasureBinnedMI(correlationMeasureType)) {
                    int timeStepIdxReference = timeStepIdx;
                    if (useTimeLagCorrelations) {
                        timeStepIdxReference = timeLagTimeStepIdx;
                    }
                    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                        auto [minValRef, maxValRef] = getMinMaxScalarFieldValue(
                                scalarFieldNames.at(fieldIndex2Gui), fieldIdx, timeStepIdxReference, ensembleIdx);
                        minFieldValRef = std::min(minFieldValRef, minValRef);
                        maxFieldValRef = std::max(maxFieldValRef, maxValRef);
                        auto [minValQuery, maxValQuery] = getMinMaxScalarFieldValue(
                                scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
                        minFieldValQuery = std::min(minFieldValQuery, minValQuery);
                        maxFieldValQuery = std::max(maxFieldValQuery, maxValQuery);
                    }
                }
                renderer->pushConstants(
                        correlationComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 2 * sizeof(glm::vec4),
                        glm::vec4(minFieldValRef, maxFieldValRef, minFieldValQuery, maxFieldValQuery));
            } else {
                float minFieldVal = std::numeric_limits<float>::max();
                float maxFieldVal = std::numeric_limits<float>::lowest();
                if (isMeasureBinnedMI(correlationMeasureType)) {
                    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
                        auto [minVal, maxVal] = getMinMaxScalarFieldValue(
                                scalarFieldNames.at(fieldIndexGui), fieldIdx, timeStepIdx, ensembleIdx);
                        minFieldVal = std::min(minFieldVal, minVal);
                        maxFieldVal = std::max(maxFieldVal, maxVal);
                    }
                }
                renderer->pushConstants(
                        correlationComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 2 * sizeof(glm::vec4),
                        glm::vec4(minFieldVal, maxFieldVal, minFieldVal, maxFieldVal));
            }
        }
        correlationComputePass->render();
    } else {
#ifdef SUPPORT_CUDA_INTEROP
        correlationComputePass->computeCuda(
                this, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx, deviceCacheEntry,
                referencePointIndex);
#endif
    }

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}



CorrelationComputePass::CorrelationComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    spearmanReferenceRankComputePass = std::make_shared<SpearmanReferenceRankComputePass>(renderer, uniformBuffer);

#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamCreate(
                &stream, 0), "Error in cuStreamCreate: ");
    }
#endif
}

#ifdef SUPPORT_CUDA_INTEROP
struct CorrelationCalculatorKernelCache {
    ~CorrelationCalculatorKernelCache() {
        if (cumodule) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleUnload(
                    cumodule), "Error in cuModuleUnload: ");
        }
    }
    std::string kernelString;
    std::map<std::string, std::string> preprocessorDefines;
    CUmodule cumodule{};
    CUfunction kernel{};
};
#endif

CorrelationComputePass::~CorrelationComputePass() {
#ifdef SUPPORT_CUDA_INTEROP
    if (outputImageBufferCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                outputImageBufferCu), "Error in cuMemFree: ");
    }
    if (fieldTextureArrayCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                fieldTextureArrayCu), "Error in cuMemFree: ");
    }
    if (fieldBufferArrayCu != 0) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFree(
                fieldBufferArrayCu), "Error in cuMemFree: ");
    }
    if (kernelCache) {
        delete kernelCache;
        kernelCache = nullptr;
    }
    if (stream) {
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamDestroy(
                stream), "Error in cuStreamDestroy: ");
    }
#endif
}

void CorrelationComputePass::setVolumeData(VolumeData *_volumeData, int correlationMemberCount, bool isNewData) {
    volumeData = _volumeData;
    xs = volumeData->getGridSizeX();
    ys = volumeData->getGridSizeY();
    zs = volumeData->getGridSizeZ();
    uniformData.xs = uint32_t(xs);
    uniformData.ys = uint32_t(ys);
    uniformData.zs = uint32_t(zs);
    uniformData.cs = uint32_t(correlationMemberCount);
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    spearmanReferenceRankComputePass->setVolumeData(_volumeData, isNewData);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
        spearmanReferenceRankComputePass->setCorrelationMemberCount(cachedCorrelationMemberCount);
    }
}

void CorrelationComputePass::overrideGridSize(int _xsr, int _ysr, int _zsr, int _xsq, int _ysq, int _zsq) {
    uniformData.xs = _xsr;
    uniformData.ys = _ysr;
    uniformData.zs = _zsr;
    uniformData.xsr = _xsr;
    uniformData.ysr = _ysr;
    uniformData.zsr = _zsr;
    uniformData.xsq = _xsq;
    uniformData.ysq = _ysq;
    uniformData.zsq = _zsq;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
}

void CorrelationComputePass::setCorrelationMemberCount(int correlationMemberCount) {
    uniformData.cs = uint32_t(correlationMemberCount);
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedCorrelationMemberCount != correlationMemberCount) {
        cachedCorrelationMemberCount = correlationMemberCount;
        setShaderDirty();
        spearmanReferenceRankComputePass->setCorrelationMemberCount(cachedCorrelationMemberCount);
    }
}

void CorrelationComputePass::setDataMode(CorrelationDataMode _dataMode) {
    if (dataMode != _dataMode) {
        dataMode = _dataMode;
        setShaderDirty();
    }
    spearmanReferenceRankComputePass->setDataMode(_dataMode);
}

void CorrelationComputePass::setUseBufferTiling(bool _useBufferTiling) {
    if (useBufferTiling != _useBufferTiling) {
        useBufferTiling = _useBufferTiling;
        setShaderDirty();
    }
    spearmanReferenceRankComputePass->setUseBufferTiling(_useBufferTiling);
}

void CorrelationComputePass::setCorrelationFieldMode(CorrelationFieldMode _correlationFieldMode) {
    if (correlationFieldMode != _correlationFieldMode) {
        correlationFieldMode = _correlationFieldMode;
        setShaderDirty();
    }
    spearmanReferenceRankComputePass->setUseSeparateFields(correlationFieldMode == CorrelationFieldMode::SEPARATE);
}

void CorrelationComputePass::setFieldBuffers(const std::vector<sgl::vk::BufferPtr>& _fieldBuffers) {
    if (fieldBuffers == _fieldBuffers) {
        return;
    }
    fieldBuffers = _fieldBuffers;
    if (computeData) {
        computeData->setStaticBufferArrayOptional(fieldBuffers, "ScalarFieldBuffers");
    }
    spearmanReferenceRankComputePass->setFieldBuffers(fieldBuffers);
}

void CorrelationComputePass::setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews) {
    if (fieldImageViews == _fieldImageViews) {
        return;
    }
    fieldImageViews = _fieldImageViews;
    if (computeData) {
        computeData->setStaticImageViewArrayOptional(fieldImageViews, "scalarFields");
    }
    spearmanReferenceRankComputePass->setFieldImageViews(fieldImageViews);
}

void CorrelationComputePass::setUseSecondaryFields(bool _useSecondaryFields) {
    if (useSecondaryFields != _useSecondaryFields) {
        useSecondaryFields = _useSecondaryFields;
        setShaderDirty();
    }
}

void CorrelationComputePass::setFieldBuffersSecondary(const std::vector<sgl::vk::BufferPtr>& _fieldBuffers) {
    if (fieldBuffersSecondary == _fieldBuffers) {
        return;
    }
    fieldBuffersSecondary = _fieldBuffers;
    if (computeData) {
        computeData->setStaticBufferArrayOptional(fieldBuffersSecondary, "ScalarFieldBuffersSecondary");
    }
}

void CorrelationComputePass::setFieldImageViewsSecondary(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews) {
    if (fieldImageViewsSecondary == _fieldImageViews) {
        return;
    }
    fieldImageViewsSecondary = _fieldImageViews;
    if (computeData) {
        computeData->setStaticImageViewArrayOptional(fieldImageViewsSecondary, "scalarFieldsSecondary");
    }
}

void CorrelationComputePass::setReferenceValuesBuffer(const sgl::vk::BufferPtr& _referenceValuesBuffer) {
    if (referenceValuesBuffer == _referenceValuesBuffer) {
        return;
    }
    referenceValuesBuffer = _referenceValuesBuffer;
    if (computeData) {
        computeData->setStaticBufferOptional(referenceValuesBuffer, "ReferenceValuesBuffer");
    }
    spearmanReferenceRankComputePass->setReferenceValuesBuffer(_referenceValuesBuffer);
}

#ifdef SUPPORT_CUDA_INTEROP
void CorrelationComputePass::setReferenceValuesCudaBuffer(
        const sgl::vk::BufferCudaExternalMemoryVkPtr& _referenceValuesCudaBuffer) {
    referenceValuesCudaBuffer = _referenceValuesCudaBuffer;
}
#endif

void CorrelationComputePass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    outputImage = _outputImage;
    if (computeData) {
        computeData->setStaticImageView(outputImage, "outputImage");
    }
}

void CorrelationComputePass::setReferencePoint(const glm::ivec3& referencePointIndex) {
    if (correlationMeasureType == CorrelationMeasureType::PEARSON
            || correlationMeasureType == CorrelationMeasureType::KENDALL
            || isMeasureMI(correlationMeasureType)) {
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, referencePointIndex);
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        spearmanReferenceRankComputePass->buildIfNecessary();
        renderer->pushConstants(
                spearmanReferenceRankComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT,
                0, referencePointIndex);
    }
}

void CorrelationComputePass::setUseRequestEvaluationMode(bool _useRequestEvaluationMode) {
    useRequestEvaluationMode = _useRequestEvaluationMode;
}

void CorrelationComputePass::setNumRequests(uint32_t _numRequests) {
    numRequests = _numRequests;
}

void CorrelationComputePass::setRequestsBuffer(const sgl::vk::BufferPtr& _requestsBuffer) {
    requestsBuffer = _requestsBuffer;
    if (computeData) {
        computeData->setStaticBuffer(requestsBuffer, "RequestsBuffer");
    }
}

void CorrelationComputePass::setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer) {
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void CorrelationComputePass::setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType) {
    if (correlationMeasureType != _correlationMeasureType) {
        correlationMeasureType = _correlationMeasureType;
        setShaderDirty();
    }
}

void CorrelationComputePass::setCalculateAbsoluteValue(bool _calculateAbsoluteValue) {
    if (calculateAbsoluteValue != _calculateAbsoluteValue) {
        calculateAbsoluteValue = _calculateAbsoluteValue;
        setShaderDirty();
    }
}

void CorrelationComputePass::setNumBins(int _numBins) {
    if (isMeasureBinnedMI(correlationMeasureType) && numBins != _numBins) {
        setShaderDirty();
    }
    numBins = _numBins;
}

void CorrelationComputePass::setKraskovNumNeighbors(int _k) {
    if (isMeasureKraskovMI(correlationMeasureType) && k != _k) {
        setShaderDirty();
    }
    k = _k;
}

void CorrelationComputePass::setKraskovEstimatorIndex(int _kraskovEstimatorIndex) {
    if (isMeasureKraskovMI(correlationMeasureType) && kraskovEstimatorIndex != _kraskovEstimatorIndex) {
        setShaderDirty();
    }
    kraskovEstimatorIndex = _kraskovEstimatorIndex;
}

void CorrelationComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    if (useRequestEvaluationMode) {
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSize1D)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(1)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(1)));
    } else if (zs < 4) {
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSize2dX)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSize2dY)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", "1"));
    } else {
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    }
    preprocessorDefines.insert(std::make_pair(
            "MEMBER_COUNT", std::to_string(cachedCorrelationMemberCount)));
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        preprocessorDefines.insert(std::make_pair("USE_SCALAR_FIELD_IMAGES", ""));
    } else if (useBufferTiling) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_TILING", ""));
    }
    if (correlationFieldMode == CorrelationFieldMode::SEPARATE_SYMMETRIC) {
        preprocessorDefines.insert(std::make_pair("USE_SECONDARY_FIELDS", ""));
        preprocessorDefines.insert(std::make_pair("USE_SECONDARY_FIELDS_SYMMETRIC", ""));
    } else if (correlationFieldMode == CorrelationFieldMode::SEPARATE) {
        preprocessorDefines.insert(std::make_pair("SEPARATE_REFERENCE_AND_QUERY_FIELDS", ""));
    }
    if (useRequestEvaluationMode) {
        preprocessorDefines.insert(std::make_pair("USE_REQUESTS_BUFFER", ""));
        if (useSecondaryFields) {
            preprocessorDefines.insert(std::make_pair("USE_SECONDARY_FIELDS", ""));
        }
    }
    if (!isMeasureMI(correlationMeasureType) && calculateAbsoluteValue) {
        preprocessorDefines.insert(std::make_pair("CALCULATE_ABSOLUTE_VALUE", ""));
    }
    if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
        auto maxStackSize = uint32_t(std::ceil(std::log2(cachedCorrelationMemberCount))) + 1;
        preprocessorDefines.insert(std::make_pair(
                "MAX_STACK_SIZE", std::to_string(maxStackSize)));
        preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));
    } else if (isMeasureBinnedMI(correlationMeasureType)) {
        preprocessorDefines.insert(std::make_pair("numBins", std::to_string(numBins)));
    } else if (isMeasureKraskovMI(correlationMeasureType)) {
        auto maxBinaryTreeLevels = uint32_t(std::ceil(std::log2(cachedCorrelationMemberCount + 1)));
        preprocessorDefines.insert(std::make_pair(
                "MAX_STACK_SIZE_BUILD", std::to_string(2 * maxBinaryTreeLevels)));
        preprocessorDefines.insert(std::make_pair(
                "MAX_STACK_SIZE_KN", std::to_string(maxBinaryTreeLevels)));
        preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));
        preprocessorDefines.insert(std::make_pair("KRASKOV_ESTIMATOR_INDEX", std::to_string(kraskovEstimatorIndex)));
    }
    if (isMeasureCorrelationCoefficientMI(correlationMeasureType)) {
        preprocessorDefines.insert(std::make_pair("MI_CORRELATION_COEFFICIENT", ""));
    }
    std::string shaderName;
    if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
        shaderName = "PearsonCorrelation.Compute";
    } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        shaderName = "SpearmanRankCorrelation.Compute";
    } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
        shaderName = "KendallRankCorrelation.Compute";
    } else if (isMeasureBinnedMI(correlationMeasureType)) {
        shaderName = "MutualInformationBinned.Compute";
    } else if (isMeasureKraskovMI(correlationMeasureType)) {
        shaderName = "MutualInformationKraskov.Compute";
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void CorrelationComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
        computeData->setStaticImageViewArray(fieldImageViews, "scalarFields");
    } else {
        computeData->setStaticBufferArray(fieldBuffers, "ScalarFieldBuffers");
    }
    if (useSecondaryFields) {
        if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
            computeData->setStaticImageViewArray(fieldImageViewsSecondary, "scalarFieldsSecondary");
        } else {
            computeData->setStaticBufferArray(fieldBuffersSecondary, "ScalarFieldBuffersSecondary");
        }
    }
    if (correlationFieldMode == CorrelationFieldMode::SEPARATE) {
        computeData->setStaticBuffer(referenceValuesBuffer, "ReferenceValuesBuffer");
    }
    if (useRequestEvaluationMode) {
        computeData->setStaticBuffer(requestsBuffer, "RequestsBuffer");
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    } else {
        computeData->setStaticImageView(outputImage, "outputImage");
    }
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        computeData->setStaticBuffer(
                spearmanReferenceRankComputePass->getReferenceRankBuffer(), "ReferenceRankBuffer");
    }
}

void CorrelationComputePass::_render() {
    if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
        spearmanReferenceRankComputePass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                spearmanReferenceRankComputePass->getReferenceRankBuffer());
    }

    uint32_t batchCount = 1;
    bool needsBatchedRendering = false;
    bool supportsBatchedRendering =
            correlationMeasureType == CorrelationMeasureType::PEARSON
            || correlationMeasureType == CorrelationMeasureType::SPEARMAN
            || correlationMeasureType == CorrelationMeasureType::KENDALL
            || isMeasureMI(correlationMeasureType);
    supportsBatchedRendering = supportsBatchedRendering && !useRequestEvaluationMode;
    if (supportsBatchedRendering) {
        /*
         * M = #cells, N = Members.
         * Estimated time: M * N * log2(N).
         * On a RTX 3090, M = 1.76 * 10^6 and N = 100 takes approx. 1s.
         */
        sgl::DeviceThreadInfo deviceCoresInfo = sgl::getDeviceThreadInfo(device);
        const uint32_t numCudaCoresRtx3090 = 10496;
        int M = xs * ys * zs;
        int N = cachedCorrelationMemberCount;
        double factorM = double(M) / (1.76 * 1e6);
        double factorN = double(N) / 100.0 * std::log2(double(N) / 100.0 + 1.0);
        double factorCores = double(numCudaCoresRtx3090) / double(deviceCoresInfo.numCudaCoresEquivalent);
        batchCount = uint32_t(std::ceil(factorM * factorN * factorCores));

        uint32_t batchCorrelationMemberCountThreshold = 10;
        if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
            batchCorrelationMemberCountThreshold = batchCorrelationMemberCountThresholdPearson;
        } else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
            batchCorrelationMemberCountThreshold = batchCorrelationMemberCountThresholdSpearman;
        } else if (correlationMeasureType == CorrelationMeasureType::KENDALL) {
            batchCorrelationMemberCountThreshold = batchCorrelationMemberCountThresholdKendall;
        } else if (isMeasureBinnedMI(correlationMeasureType)) {
            batchCorrelationMemberCountThreshold = batchCorrelationMemberCountThresholdMiBinned;
        } else if (isMeasureKraskovMI(correlationMeasureType)) {
            batchCorrelationMemberCountThreshold = batchCorrelationMemberCountThresholdKraskov;
        }
        if (cachedCorrelationMemberCount > int(batchCorrelationMemberCountThreshold)) {
            needsBatchedRendering = true;
            batchCount = sgl::uiceil(uint32_t(cachedCorrelationMemberCount), batchCorrelationMemberCountThreshold);
        }
    }

    int blockSizeX = zs < 4 ? computeBlockSize2dX : computeBlockSizeX;
    int blockSizeY = zs < 4 ? computeBlockSize2dY : computeBlockSizeY;
    int blockSizeZ = zs < 4 ? 1 : computeBlockSizeZ;
    if (needsBatchedRendering) {
        auto blockSizeXUint = uint32_t(blockSizeX);
        //auto blockSizeY = uint32_t(computeBlockSizeY);
        //auto blockSizeZ = uint32_t(computeBlockSizeZ);
        /*auto batchSizeX =
                sgl::uiceil(uint32_t(xs), batchCount * blockSizeXUint) * blockSizeXUint;
        auto batchSizeY = uint32_t(ys);
        auto batchSizeZ = uint32_t(zs);
        batchCount = sgl::uiceil(uint32_t(xs), batchSizeX);*/
        auto batchSizeX = 2 * blockSizeXUint;
        auto batchSizeY = uint32_t(ys);
        auto batchSizeZ = uint32_t(zs);
        batchCount = sgl::uiceil(uint32_t(xs), batchSizeX);
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            if (supportsBatchedRendering) {
                renderer->pushConstants(
                        getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::uvec4),
                        glm::uvec3(batchSizeX * batchIdx, 0, 0));
            }
            if (batchIdx == batchCount - 1) {
                batchSizeX = uint32_t(xs) - batchSizeX * batchIdx;
            }
            renderer->dispatch(
                    computeData,
                    sgl::uiceil(batchSizeX, uint32_t(blockSizeX)),
                    sgl::uiceil(batchSizeY, uint32_t(blockSizeY)),
                    sgl::uiceil(batchSizeZ, uint32_t(blockSizeZ)));
            renderer->syncWithCpu();
        }
    } else {
        if (supportsBatchedRendering) {
            renderer->pushConstants(
                    getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(glm::uvec4), glm::uvec3(0));
        }
        if (useRequestEvaluationMode) {
            renderer->dispatch(computeData, sgl::uiceil(numRequests, uint32_t(computeBlockSize1D)), 1, 1);
        } else {
            renderer->dispatch(
                    computeData, sgl::iceil(xs, blockSizeX), sgl::iceil(ys, blockSizeY), sgl::iceil(zs, blockSizeZ));
        }
    }
}

#ifdef SUPPORT_CUDA_INTEROP
void CorrelationComputePass::computeCuda(
        CorrelationCalculator* correlationCalculator,
        const std::string& fieldName, int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry,
        glm::ivec3& referencePointIndex) {
    int cs = cachedCorrelationMemberCount;
    int N = cs;
    int M = xs * ys * zs;

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);

    size_t volumeDataSlice3dSize = volumeData->getSlice3dSizeInBytes(FieldType::SCALAR);
    if (cachedVolumeDataSlice3dSize != volumeDataSlice3dSize) {
        if (outputImageBufferCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    outputImageBufferCu, stream), "Error in cuMemFreeAsync: ");
        }
        cachedVolumeDataSlice3dSize = volumeDataSlice3dSize;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &outputImageBufferCu, volumeDataSlice3dSize, stream), "Error in cuMemAllocAsync: ");
    }

    if (cachedCorrelationMemberCountDevice != size_t(cs)) {
        if (fieldTextureArrayCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    fieldTextureArrayCu, stream), "Error in cuMemFreeAsync: ");
        }
        if (fieldBufferArrayCu != 0) {
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemFreeAsync(
                    fieldBufferArrayCu, stream), "Error in cuMemFreeAsync: ");
        }
        cachedCorrelationMemberCountDevice = size_t(cs);
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &fieldTextureArrayCu, cs * sizeof(CUtexObject), stream), "Error in cuMemAllocAsync: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemAllocAsync(
                &fieldBufferArrayCu, cs * sizeof(CUdeviceptr), stream), "Error in cuMemAllocAsync: ");
    }

    sgl::vk::Swapchain* swapchain = sgl::AppSettings::get()->getSwapchain();
    uint32_t frameIndex = swapchain ? swapchain->getImageIndex() : 0;
    size_t numSwapchainImages = swapchain ? swapchain->getNumImages() : 1;
    if (numSwapchainImages != cachedNumSwapchainImages) {
        cachedNumSwapchainImages = numSwapchainImages;
        sgl::vk::Device* device = renderer->getDevice();
        timelineValue = 0;
        postRenderCommandBuffers.clear();
        sgl::vk::CommandPoolType commandPoolType;
        commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        for (size_t frameIdx = 0; frameIdx < numSwapchainImages; frameIdx++) {
            postRenderCommandBuffers.push_back(std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType));
        }
        vulkanFinishedSemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE, timelineValue);
        cudaFinishedSemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE, timelineValue);
    }
    timelineValue++;

    bool useImageArray = dataMode == CorrelationDataMode::IMAGE_3D_ARRAY;
    std::vector<VolumeData::DeviceCacheEntry> fieldEntries;
    std::vector<CUtexObject> fieldTexturesCu;
    std::vector<CUdeviceptr> fieldBuffersCu;
    fieldEntries.reserve(cs);
    if (useImageArray) {
        fieldTexturesCu.reserve(cs);
    } else {
        fieldBuffersCu.reserve(cs);
    }
    for (int fieldIdx = 0; fieldIdx < cs; fieldIdx++) {
        VolumeData::DeviceCacheEntry fieldEntry = correlationCalculator->getFieldEntryDevice(
                fieldName, fieldIdx, timeStepIdx, ensembleIdx, useImageArray);
        fieldEntries.push_back(fieldEntry);
        if (useImageArray) {
            fieldTexturesCu.push_back(fieldEntry->getCudaTexture());
            if (fieldEntry->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                deviceCacheEntry->getVulkanImage()->transitionImageLayout(
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
            }
        } else {
            fieldBuffersCu.push_back(fieldEntry->getCudaBuffer());
        }
    }

    if (useImageArray && cachedFieldTexturesCu != fieldTexturesCu) {
        cachedFieldTexturesCu = fieldTexturesCu;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                stream), "Error in cuStreamSynchronize: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                fieldTextureArrayCu, fieldTexturesCu.data(), sizeof(CUtexObject) * cs), "Error in cuMemcpyHtoD: ");
    }
    if (!useImageArray && cachedFieldBuffersCu != fieldBuffersCu) {
        cachedFieldBuffersCu = fieldBuffersCu;
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(
                stream), "Error in cuStreamSynchronize: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuMemcpyHtoD(
                fieldBufferArrayCu, fieldBuffersCu.data(), sizeof(CUdeviceptr) * cs), "Error in cuMemcpyHtoD: ");
    }

    sgl::vk::CommandBufferPtr commandBufferRender = renderer->getCommandBuffer();
    vulkanFinishedSemaphore->setSignalSemaphoreValue(timelineValue);
    commandBufferRender->pushSignalSemaphore(vulkanFinishedSemaphore);
    renderer->endCommandBuffer();

    renderer->submitToQueue();

    vulkanFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);

    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair(
            "MEMBER_COUNT", std::to_string(N)));
    auto maxBinaryTreeLevels = uint32_t(std::ceil(std::log2(N + 1)));
    preprocessorDefines.insert(std::make_pair(
            "MAX_STACK_SIZE_BUILD", std::to_string(2 * maxBinaryTreeLevels)));
    preprocessorDefines.insert(std::make_pair(
            "MAX_STACK_SIZE_KN", std::to_string(maxBinaryTreeLevels)));
    preprocessorDefines.insert(std::make_pair("k", std::to_string(k)));
    if (isMeasureCorrelationCoefficientMI(correlationMeasureType)) {
        preprocessorDefines.insert(std::make_pair("MI_CORRELATION_COEFFICIENT", ""));
    }
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        preprocessorDefines.insert(std::make_pair("USE_SCALAR_FIELD_IMAGES", ""));
    } else if (useBufferTiling) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_TILING", ""));
    }
    if (correlationFieldMode == CorrelationFieldMode::SEPARATE_SYMMETRIC) {
        preprocessorDefines.insert(std::make_pair("USE_SECONDARY_FIELDS", ""));
        preprocessorDefines.insert(std::make_pair("USE_SECONDARY_FIELDS_SYMMETRIC", ""));
    } else if (correlationFieldMode == CorrelationFieldMode::SEPARATE) {
        preprocessorDefines.insert(std::make_pair("SEPARATE_REFERENCE_AND_QUERY_FIELDS", ""));
    }

    if (!kernelCache || kernelCache->preprocessorDefines != preprocessorDefines) {
        if (kernelCache) {
            delete kernelCache;
        }
        kernelCache = new CorrelationCalculatorKernelCache;

        if (kernelCache->kernelString.empty()) {
            std::ifstream inFile(
                    sgl::AppSettings::get()->getDataDirectory() + "Shaders/Correlation/MutualInformationKraskov.cu",
                    std::ios::binary);
            if (!inFile.is_open()) {
                sgl::Logfile::get()->throwError(
                        "Error in CorrelationComputePass::computeCuda: Could not open MutualInformationKraskov.cu.");
            }
            inFile.seekg(0, std::ios::end);
            auto fileSize = inFile.tellg();
            inFile.seekg(0, std::ios::beg);
            kernelCache->kernelString.resize(fileSize);
            inFile.read(kernelCache->kernelString.data(), fileSize);
            inFile.close();
        }

        std::string code;
        for (const auto& entry : preprocessorDefines) {
            code += std::string("#define ") + entry.first + " " + entry.second + "\n";
        }
        code += "#line 1\n";
        code += kernelCache->kernelString;

        nvrtcProgram prog;
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcCreateProgram(
                &prog, code.c_str(), "MutualInformationKraskov.cu", 0, nullptr, nullptr), "Error in nvrtcCreateProgram: ");
        auto retVal = sgl::vk::g_nvrtcFunctionTable.nvrtcCompileProgram(prog, 0, nullptr);
        if (retVal == NVRTC_ERROR_COMPILATION) {
            size_t logSize = 0;
            sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetProgramLogSize(
                    prog, &logSize), "Error in nvrtcGetProgramLogSize: ");
            char* log = new char[logSize];
            sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetProgramLog(
                    prog, log), "Error in nvrtcGetProgramLog: ");
            std::cerr << "NVRTC log:" << std::endl << log << std::endl;
            delete[] log;
            sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcDestroyProgram(
                    &prog), "Error in nvrtcDestroyProgram: ");
            exit(1);
        }

        size_t ptxSize = 0;
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetPTXSize(
                prog, &ptxSize), "Error in nvrtcGetPTXSize: ");
        char* ptx = new char[ptxSize];
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcGetPTX(
                prog, ptx), "Error in nvrtcGetPTX: ");
        sgl::vk::checkNvrtcResult(sgl::vk::g_nvrtcFunctionTable.nvrtcDestroyProgram(
                &prog), "Error in nvrtcDestroyProgram: ");

        kernelCache->preprocessorDefines = preprocessorDefines;

        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleLoadDataEx(
                &kernelCache->cumodule, ptx, 0, nullptr, nullptr), "Error in cuModuleLoadDataEx: ");
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuModuleGetFunction(
                &kernelCache->kernel, kernelCache->cumodule, "mutualInformationKraskov"), "Error in cuModuleGetFunction: ");
        delete[] ptx;
    }

    int minGridSize = 0;
    int bestBlockSize = 32;
    sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuOccupancyMaxPotentialBlockSize(
            &minGridSize, &bestBlockSize, kernelCache->kernel, nullptr, 0, 0), "Error in cuOccupancyMaxPotentialBlockSize: ");

    sgl::DeviceThreadInfo deviceCoresInfo = sgl::getDeviceThreadInfo(device);
    int BLOCK_SIZE = sgl::lastPowerOfTwo(bestBlockSize);
    while (BLOCK_SIZE > int(deviceCoresInfo.warpSize)
           && sgl::iceil(int(M), BLOCK_SIZE) * 2 < int(deviceCoresInfo.numMultiprocessors)) {
        BLOCK_SIZE /= 2;
    }

    /*
     * Estimated time: M * N * log2(N).
     * On a RTX 3090, M = 1.76 * 10^6 and N = 100 takes approx. 1s.
     */
    const uint32_t numCudaCoresRtx3090 = 10496;
    double factorM = double(M) / (1.76 * 1e6);
    double factorN = double(N) / 100.0 * std::log2(double(N) / 100.0 + 1.0);
    double factorCores = double(numCudaCoresRtx3090) / double(deviceCoresInfo.numCudaCoresEquivalent);
    auto batchCount = uint32_t(std::ceil(factorM * factorN * factorCores));

    CUdeviceptr scalarFields =
            dataMode == CorrelationDataMode::IMAGE_3D_ARRAY ? fieldTextureArrayCu : fieldBufferArrayCu;
    CUdeviceptr miArray = outputImageBufferCu;
    CUdeviceptr referenceValuesPtr =
            correlationFieldMode == CorrelationFieldMode::SEPARATE ? referenceValuesCudaBuffer->getCudaDevicePtr() : 0;
    auto batchSize = uint32_t(M);
    if (batchCount == 1) {
        auto batchOffset = uint32_t(0);
        void* kernelParameters[] = {
                &scalarFields, &referenceValuesPtr, &miArray, &xs, &ys, &zs, &referencePointIndex.x,
                &batchOffset, &batchSize
        };
        sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                kernelCache->kernel, sgl::iceil(int(M), BLOCK_SIZE), 1, 1, //< Grid size.
                BLOCK_SIZE, 1, 1, //< Block size.
                0, //< Dynamic shared memory size.
                stream,
                kernelParameters, //< Kernel parameters.
                nullptr
        ), "Error in cuLaunchKernel: "); //< Extra (empty).
    } else {
        auto batchSizeLocal = sgl::uiceil(uint32_t(M), batchCount);
        for (uint32_t batchIdx = 0; batchIdx < batchCount; batchIdx++) {
            auto batchOffset = batchSizeLocal * batchIdx;
            if (batchOffset + batchSizeLocal > uint32_t(M)) {
                batchSizeLocal = uint32_t(M) - batchSizeLocal;
            }
            void* kernelParameters[] = {
                    &scalarFields, &referenceValuesPtr, &miArray, &xs, &ys, &zs, &referencePointIndex.x,
                    &batchOffset, &batchSize
            };
            sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuLaunchKernel(
                    kernelCache->kernel, sgl::iceil(int(batchSizeLocal), BLOCK_SIZE), 1, 1, //< Grid size.
                    BLOCK_SIZE, 1, 1, //< Block size.
                    0, //< Dynamic shared memory size.
                    stream,
                    kernelParameters, //< Kernel parameters.
                    nullptr
            ), "Error in cuLaunchKernel: "); //< Extra (empty).
        }
    }

    deviceCacheEntry->getImageCudaExternalMemory()->memcpyCudaDtoA3DAsync(outputImageBufferCu, stream);

    cudaFinishedSemaphore->signalSemaphoreCuda(stream, timelineValue);
    cudaFinishedSemaphore->setWaitSemaphoreValue(timelineValue);
    sgl::vk::CommandBufferPtr postRenderCommandBuffer = postRenderCommandBuffers.at(frameIndex);
    renderer->pushCommandBuffer(postRenderCommandBuffer);
    renderer->beginCommandBuffer();
    postRenderCommandBuffer->pushWaitSemaphore(
            cudaFinishedSemaphore, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
}
#endif



SpearmanReferenceRankComputePass::SpearmanReferenceRankComputePass(
        sgl::vk::Renderer* renderer, sgl::vk::BufferPtr uniformBuffer)
        : ComputePass(renderer), uniformBuffer(std::move(uniformBuffer)) {
}

void SpearmanReferenceRankComputePass::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
}

void SpearmanReferenceRankComputePass::setCorrelationMemberCount(int correlationMemberCount) {
    cachedCorrelationMemberCount = correlationMemberCount;
    referenceRankBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(float) * cachedCorrelationMemberCount,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    setShaderDirty();
}

void SpearmanReferenceRankComputePass::setDataMode(CorrelationDataMode _dataMode) {
    if (dataMode != _dataMode) {
        dataMode = _dataMode;
        setShaderDirty();
    }
}

void SpearmanReferenceRankComputePass::setUseBufferTiling(bool _useBufferTiling) {
    if (useBufferTiling != _useBufferTiling) {
        useBufferTiling = _useBufferTiling;
        setShaderDirty();
    }
}

void SpearmanReferenceRankComputePass::setUseSeparateFields(bool _useSeparateFields) {
    if (useSeparateFields != _useSeparateFields) {
        useSeparateFields = _useSeparateFields;
        setShaderDirty();
    }
}

void SpearmanReferenceRankComputePass::setFieldBuffers(const std::vector<sgl::vk::BufferPtr>& _fieldBuffers) {
    if (fieldBuffers == _fieldBuffers) {
        return;
    }
    fieldBuffers = _fieldBuffers;
    if (computeData) {
        computeData->setStaticBufferArrayOptional(fieldBuffers, "ScalarFieldBuffers");
    }
}

void SpearmanReferenceRankComputePass::setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews) {
    if (fieldImageViews == _fieldImageViews) {
        return;
    }
    fieldImageViews = _fieldImageViews;
    if (computeData) {
        computeData->setStaticImageViewArrayOptional(fieldImageViews, "scalarFields");
    }
}

void SpearmanReferenceRankComputePass::setReferenceValuesBuffer(const sgl::vk::BufferPtr& _referenceValuesBuffer) {
    if (referenceValuesBuffer == _referenceValuesBuffer) {
        return;
    }
    referenceValuesBuffer = _referenceValuesBuffer;
    if (computeData) {
        computeData->setStaticBufferOptional(referenceValuesBuffer, "ReferenceValuesBuffer");
    }
}

void SpearmanReferenceRankComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair(
            "MEMBER_COUNT", std::to_string(cachedCorrelationMemberCount)));
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        preprocessorDefines.insert(std::make_pair("USE_SCALAR_FIELD_IMAGES", ""));
    } else if (useBufferTiling) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_TILING", ""));
    }
    if (useSeparateFields) {
        preprocessorDefines.insert(std::make_pair("SEPARATE_REFERENCE_AND_QUERY_FIELDS", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "SpearmanRankCorrelation.Reference.Compute" }, preprocessorDefines);
}

void SpearmanReferenceRankComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    if (dataMode == CorrelationDataMode::IMAGE_3D_ARRAY) {
        computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
        computeData->setStaticImageViewArray(fieldImageViews, "scalarFields");
    } else {
        computeData->setStaticBufferArray(fieldBuffers, "ScalarFieldBuffers");
    }
    if (useSeparateFields) {
        computeData->setStaticBuffer(referenceValuesBuffer, "ReferenceValuesBuffer");
    }
    computeData->setStaticBuffer(referenceRankBuffer, "ReferenceRankBuffer");
}

void SpearmanReferenceRankComputePass::_render() {
    renderer->dispatch(computeData, 1, 1, 1);
}
