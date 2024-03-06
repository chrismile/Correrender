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

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Utils/Normalization.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "Export/WriteTetMesh.hpp"
#include "RenderingModes.hpp"
#include "DvrRenderer.hpp"

DvrRenderer::DvrRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_DIRECT_VOLUME_RENDERING)], viewManager) {
    ;
}

DvrRenderer::~DvrRenderer() {
    if (!selectedScalarFieldName.empty()) {
        volumeData->releaseTf(this, selectedFieldIdx);
        volumeData->releaseScalarField(this, selectedFieldIdx);
    }
}

void DvrRenderer::initialize() {
    Renderer::initialize();
}

void DvrRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;
    if (!selectedScalarFieldName.empty()) {
        volumeData->releaseTf(this, oldSelectedFieldIdx);
        volumeData->releaseScalarField(this, oldSelectedFieldIdx);
    }
    const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
    if (isNewData) {
        selectedFieldIdx = volumeData->getStandardScalarFieldIdx();

        std::string standardExportDirectory = sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/preshaded";
        exportFilePath = standardExportDirectory + "/" + volumeData->getDataSetInformation().name + ".bintet";
        if (!sgl::FileUtils::get()->directoryExists(standardExportDirectory)) {
            sgl::FileUtils::get()->ensureDirectoryExists(standardExportDirectory);
        }
    }
    selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
    volumeData->acquireTf(this, selectedFieldIdx);
    volumeData->acquireScalarField(this, selectedFieldIdx);
    oldSelectedFieldIdx = selectedFieldIdx;

    for (auto& dvrPass : dvrPasses) {
        dvrPass->setVolumeData(volumeData, isNewData);
        dvrPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        dvrPass->setStepSize(stepSize);
    }
}

void DvrRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (selectedFieldIdx == fieldIdx) {
            selectedFieldIdx = 0;
        } else if (selectedFieldIdx > fieldIdx) {
            selectedFieldIdx--;
        }
        if (oldSelectedFieldIdx == fieldIdx) {
            oldSelectedFieldIdx = -1;
            selectedScalarFieldName.clear();
        } else if (oldSelectedFieldIdx > fieldIdx) {
            oldSelectedFieldIdx--;
        }
    }
}

/*
 * Vertex and edge IDs:
 *
 *      3 +----------------+ 2
 *       /|               /|
 *      / |              / |
 *     /  |             /  |
 *    /   |            /   |
 * 7 +----------------+ 6  |
 *   |    |           |    |
 *   |    |           |    |
 *   |    |           |    |
 *   |  0 +-----------|----+ 1
 *   |   /            |   /
 *   |  /             |  /
 *   | /              | /
 *   |/               |/
 * 4 +----------------+ 5
 *
 * Tet mapping: https://cs.stackexchange.com/questions/89910/how-to-decompose-a-unit-cube-into-tetrahedra
 */
const int HEX_TO_TET_TABLE[6][4] = {
        // { 0, 4, 7, 6 }, -1
        // { 0, 4, 5, 6 }, +1
        // { 0, 3, 7, 6 }, +1
        // { 0, 3, 2, 6 }, -1
        // { 0, 1, 5, 6 }, -1
        // { 0, 1, 2, 6 }, +1
        { 0, 4, 7, 6 },
        { 0, 4, 6, 5 },
        { 0, 3, 6, 7 },
        { 0, 3, 2, 6 },
        { 0, 1, 5, 6 },
        { 0, 1, 6, 2 },
};

void DvrRenderer::createTetMeshData(
        std::vector<uint32_t>& cellIndices, std::vector<glm::vec3>& vertexPositions,
        std::vector<glm::vec4>& vertexColors) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    sgl::AABB3 gridAabb;
    gridAabb.min = glm::vec3(-0.5f, -0.5f, -0.5f);
    gridAabb.max = glm::vec3(volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ()) - glm::vec3(0.5f, 0.5f, 0.5f);
    gridAabb.min *= glm::vec3(volumeData->getDx(), volumeData->getDy(), volumeData->getDz());
    gridAabb.max *= glm::vec3(volumeData->getDx(), volumeData->getDy(), volumeData->getDz());

    // Add all vertices.
    auto& tfWindow = volumeData->getMultiVarTransferFunctionWindow();
    auto tf = tfWindow.getTransferFunctionMap_sRGB(selectedFieldIdx);
    auto minVal = tfWindow.getSelectedRangeMin(selectedFieldIdx);
    auto maxVal = tfWindow.getSelectedRangeMax(selectedFieldIdx);
    auto field = volumeData->getFieldEntryCpu(FieldType::SCALAR, selectedScalarFieldName);
    auto Nm1 = float(tf.size() - 1);
    for (int iz = 0; iz < zs; iz++) {
        for (int iy = 0; iy < ys; iy++) {
            for (int ix = 0; ix < xs; ix++) {
                // Add vertex position.
                glm::vec3 p;
                p.x = gridAabb.min.x + (float(ix) / float(xs - 1)) * (gridAabb.max.x - gridAabb.min.x);
                p.y = gridAabb.min.y + (float(iy) / float(ys - 1)) * (gridAabb.max.y - gridAabb.min.y);
                p.z = gridAabb.min.z + (float(iz) / float(zs - 1)) * (gridAabb.max.z - gridAabb.min.z);
                vertexPositions.emplace_back(p);

                // Add vertex color.
                float t = (field->getDataFloatAt(IDXS(ix, iy, iz)) - minVal) / (maxVal - minVal);
                float t0 = std::clamp(std::floor(t * Nm1), 0.0f, Nm1);
                float t1 = std::clamp(std::ceil(t * Nm1), 0.0f, Nm1);
                float f = t * Nm1 - t0;
                int idx0 = int(t0);
                int idx1 = int(t1);
                auto color0 = sgl::color16ToVec4(tf.at(idx0));
                auto color1 = sgl::color16ToVec4(tf.at(idx1));
                vertexColors.emplace_back(glm::mix(color0, color1, f));
            }
        }
    }
    normalizeVertexPositions(vertexPositions, gridAabb, nullptr);

    // Add all tet indices.
    int hex[8];
    for (int iz = 0; iz < zs - 1; iz++) {
        for (int iy = 0; iy < ys - 1; iy++) {
            for (int ix = 0; ix < xs - 1; ix++) {
                hex[0] = IDXS(ix,     iy,     iz    );
                hex[1] = IDXS(ix + 1, iy,     iz    );
                hex[2] = IDXS(ix + 1, iy + 1, iz    );
                hex[3] = IDXS(ix,     iy + 1, iz    );
                hex[4] = IDXS(ix,     iy,     iz + 1);
                hex[5] = IDXS(ix + 1, iy,     iz + 1);
                hex[6] = IDXS(ix + 1, iy + 1, iz + 1);
                hex[7] = IDXS(ix,     iy + 1, iz + 1);
                for (int tet = 0; tet < 6; tet++) {
                    for (int idx = 0; idx < 4; idx++) {
                        cellIndices.push_back(hex[HEX_TO_TET_TABLE[tet][idx]]);
                    }
                }
            }
        }
    }
}

void DvrRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    dvrPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void DvrRenderer::renderViewImpl(uint32_t viewIdx) {
    dvrPasses.at(viewIdx)->render();
}

void DvrRenderer::addViewImpl(uint32_t viewIdx) {
    auto dvrPass = std::make_shared<DvrPass>(renderer, viewManager->getViewSceneData(viewIdx));
    if (volumeData) {
        dvrPass->setVolumeData(volumeData, true);
        dvrPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
    }
    dvrPass->setStepSize(stepSize);
    dvrPass->setAttenuationCoefficient(attenuationCoefficient);
    dvrPass->setNaNHandling(nanHandling);
    dvrPasses.push_back(dvrPass);
}

void DvrRenderer::removeViewImpl(uint32_t viewIdx) {
    dvrPasses.erase(dvrPasses.begin() + viewIdx);
}

void DvrRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        int selectedFieldIdxGui = selectedFieldIdx;
        if (propertyEditor.addCombo("Scalar Field", &selectedFieldIdxGui, fieldNames.data(), int(fieldNames.size()))) {
            selectedFieldIdx = selectedFieldIdxGui;
            selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
            for (auto& dvrPass : dvrPasses) {
                dvrPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
            }
            dirty = true;
            reRender = true;
        }
    }
    if (propertyEditor.addSliderFloat("Step Size (Voxel)", &stepSize, 0.01f, 1.0f)) {
        stepSize = std::clamp(stepSize, 0.01f, 1.0f);
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setStepSize(stepSize);
        }
        reRender = true;
    }
    if (propertyEditor.addSliderFloat("Attenuation Coefficient", &attenuationCoefficient, 0.0f, 500.0f)) {
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setAttenuationCoefficient(attenuationCoefficient);
        }
        reRender = true;
    }
    if (propertyEditor.addCombo(
            "NaN Handling", (int*)&nanHandling, NAN_HANDLING_NAMES, IM_ARRAYSIZE(NAN_HANDLING_NAMES))) {
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setNaNHandling(nanHandling);
        }
        reRender = true;
    }

    if (propertyEditor.beginNode("Advanced Settings")) {
        propertyEditor.addInputAction("File Path", &exportFilePath);
        if (propertyEditor.addButton("", "Export Tet Mesh")) {
            std::vector<uint32_t> cellIndices;
            std::vector<glm::vec3> vertexPositions;
            std::vector<glm::vec4> vertexColors;
            createTetMeshData(cellIndices, vertexPositions, vertexColors);
            std::string meshExtension = sgl::FileUtils::get()->getFileExtensionLower(exportFilePath);
            if (meshExtension == "bintet") {
                saveBinTet(exportFilePath, cellIndices, vertexPositions, vertexColors);
            } else if (meshExtension == "txt") {
                saveTxtTet(exportFilePath, cellIndices, vertexPositions, vertexColors);
            } else {
                sgl::Logfile::get()->throwError(
                        "Error in DvrRenderer::renderGuiImpl: Unknown tet mesh file extension.");
            }
        }
        propertyEditor.endNode();
    }
}

void DvrRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);
    if (settings.getValueOpt("selected_field_idx", selectedFieldIdx)) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        selectedFieldIdx = std::clamp(selectedFieldIdx, 0, int(fieldNames.size()) - 1);
        selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        }
        dirty = true;
        reRender = true;
    }
    if (settings.getValueOpt("step_size", stepSize)) {
        stepSize = std::clamp(stepSize, 0.01f, 1.0f);
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setStepSize(stepSize);
        }
        reRender = true;
    }
    if (settings.getValueOpt("attenuation_coefficient", attenuationCoefficient)) {
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setAttenuationCoefficient(attenuationCoefficient);
        }
        reRender = true;
    }
    std::string nanHandlingId;
    if (settings.getValueOpt("nan_handling", nanHandlingId)) {
        for (int i = 0; i < IM_ARRAYSIZE(NAN_HANDLING_IDS); i++) {
            if (nanHandlingId == NAN_HANDLING_IDS[i]) {
                nanHandling = NaNHandling(i);
                break;
            }
        }
        for (auto& dvrPass : dvrPasses) {
            dvrPass->setNaNHandling(nanHandling);
        }
        reRender = true;
    }

    settings.getValueOpt("export_file_path", exportFilePath);
}

void DvrRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);
    settings.addKeyValue("selected_field_idx", selectedFieldIdx);
    settings.addKeyValue("step_size", stepSize);
    settings.addKeyValue("attenuation_coefficient", attenuationCoefficient);
    settings.addKeyValue("nan_handling", NAN_HANDLING_IDS[int(nanHandling)]);
    settings.addKeyValue("export_file_path", exportFilePath);
}

void DvrRenderer::reloadShaders() {
    for (auto& dvrPass : dvrPasses) {
        dvrPass->setShaderDirty();
    }
}



DvrPass::DvrPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : ComputePass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    rendererUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(RenderSettingsData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void DvrPass::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;

    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    float voxelSizeX = aabb.getDimensions().x / float(volumeData->getGridSizeX());
    float voxelSizeY = aabb.getDimensions().y / float(volumeData->getGridSizeY());
    float voxelSizeZ = aabb.getDimensions().z / float(volumeData->getGridSizeZ());
    voxelSize = std::min(voxelSizeX, std::min(voxelSizeY, voxelSizeZ));
    renderSettingsData.minBoundingBox = aabb.min;
    renderSettingsData.maxBoundingBox = aabb.max;
    renderSettingsData.stepSize = voxelSize * stepSize;

    dataDirty = true;
}

void DvrPass::setSelectedScalarField(int _selectedFieldIdx, const std::string& _scalarFieldName) {
    selectedFieldIdx = _selectedFieldIdx;
    selectedScalarFieldName = _scalarFieldName;
    if (volumeData && computeData) {
        auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
        computeData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    }
}

void DvrPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    volumeData->getPreprocessorDefines(preprocessorDefines);
    if (sceneData->useDepthBuffer) {
        preprocessorDefines.insert(std::make_pair("SUPPORT_DEPTH_BUFFER", ""));
    }
    if (nanHandling == NaNHandling::IGNORE) {
        preprocessorDefines.insert(std::make_pair("IGNORE_NAN", ""));
    } else if (nanHandling == NaNHandling::SHOW_AS_YELLOW) {
        preprocessorDefines.insert(std::make_pair("NAN_YELLOW", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({"DvrShader.Compute"}, preprocessorDefines);
}

void DvrPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    volumeData->setRenderDataBindings(computeData);
    computeData->setStaticBuffer(rendererUniformDataBuffer, "RendererUniformDataBuffer");
    computeData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    computeData->setStaticImageView(sceneImageView, "outputImage");
    if (sceneData->useDepthBuffer) {
        computeData->setStaticImageView(sceneData->sceneDepthColorImage, "depthBuffer");
    }
}

void DvrPass::recreateSwapchain(uint32_t width, uint32_t height) {
    sceneImageView = (*sceneData->sceneTexture)->getImageView();

    if (computeData) {
        computeData->setStaticImageView(sceneImageView, "outputImage");
        if (sceneData->useDepthBuffer) {
            computeData->setStaticImageView(sceneData->sceneDepthColorImage, "depthBuffer");
        }
    }
}

void DvrPass::_render() {
    renderSettingsData.stepSize = voxelSize * stepSize;
    renderSettingsData.inverseViewMatrix = glm::inverse((*camera)->getViewMatrix());
    renderSettingsData.inverseProjectionMatrix = glm::inverse((*camera)->getProjectionMatrix());
    renderSettingsData.zNear = (*camera)->getNearClipDistance();
    renderSettingsData.zFar = (*camera)->getFarClipDistance();
    renderSettingsData.fieldIndex = uint32_t(selectedFieldIdx);
    rendererUniformDataBuffer->updateData(
            sizeof(RenderSettingsData), &renderSettingsData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    //renderer->insertImageMemoryBarrier(
    //       sceneImageView->getImage(),
    //       sceneImageView->getImage()->getVkImageLayout(), VK_IMAGE_LAYOUT_GENERAL,
    //       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    //       VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
    sceneData->switchColorState(RenderTargetAccess::COMPUTE);
    if (sceneData->useDepthBuffer) {
        sceneData->switchDepthState(RenderTargetAccess::COMPUTE);
    }

    auto scalarField = computeData->getImageView("scalarField")->getImage();
    if (scalarField->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        scalarField->transitionImageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
    }

    int width = int(sceneImageView->getImage()->getImageSettings().width);
    int height = int(sceneImageView->getImage()->getImageSettings().height);
    int groupCountX = sgl::iceil(width, 16);
    int groupCountY = sgl::iceil(height, 16);
    renderer->dispatch(computeData, groupCountX, groupCountY, 1);

    //renderer->insertImageMemoryBarrier(
    //        sceneImageView->getImage(),
    //        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    //        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    //        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}
