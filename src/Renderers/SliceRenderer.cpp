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

#include <Utils/AppSettings.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include "Utils/InternalState.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "RenderingModes.hpp"
#include "SliceRenderer.hpp"

SliceRenderer::SliceRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_SLICE_RENDERER)], viewManager) {
}

SliceRenderer::~SliceRenderer() {
    if (!selectedScalarFieldName.empty()) {
        volumeData->releaseScalarField(this, selectedFieldIdx);
    }
}

void SliceRenderer::initialize() {
    Renderer::initialize();
}

void SliceRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
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
    }
    selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
    volumeData->acquireTf(this, selectedFieldIdx);
    volumeData->acquireScalarField(this, selectedFieldIdx);
    oldSelectedFieldIdx = selectedFieldIdx;

    indexBuffer = {};
    vertexPositionBuffer = {};
    vertexNormalBuffer = {};

    for (auto& sliceRasterPass : sliceRasterPasses) {
        sliceRasterPass->setVolumeData(volumeData, isNewData);
        sliceRasterPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        sliceRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer);
    }

    std::vector<uint32_t> triangleIndices;
    std::vector<glm::vec3> vertexPositions;
    std::vector<glm::vec3> vertexNormals;
    createGeometryData(triangleIndices, vertexPositions, vertexNormals);

    if (triangleIndices.empty()) {
        return;
    }

    sgl::vk::Device* device = renderer->getDevice();
    indexBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t) * triangleIndices.size(), triangleIndices.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexPositionBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexPositions.size(), vertexPositions.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    vertexNormalBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(glm::vec3) * vertexNormals.size(), vertexNormals.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    for (auto& sliceRasterPass : sliceRasterPasses) {
        sliceRasterPass->setVolumeData(volumeData, isNewData);
        sliceRasterPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        sliceRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer);
    }
}

void SliceRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
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

static void testRayPlaneIntersection(
        const glm::vec3& origin, const glm::vec3& dir, const sgl::Plane& plane,
        std::vector<glm::vec3>& polygonPoints) {
    float vd = plane.a * dir.x + plane.b * dir.y + plane.c * dir.z;
    if (std::abs(vd) < 1e-6f) {
        return;
    }
    float t = -(plane.a * origin.x + plane.b * origin.y + plane.c * origin.z + plane.d) / vd;
    if (t >= 0.0f && t <= 1.0f) {
        polygonPoints.push_back(origin + dir * t);
    }
}

void SliceRenderer::createGeometryData(
        std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
        std::vector<glm::vec3>& vertexNormals) {
    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    sgl::Plane plane(planeNormal, planeDist);
    std::vector<glm::vec3> polygonPoints;

    // Special case: The volume is already a slice.
    if (volumeData->getGridSizeZ() == 1) {
        vertexPositions.emplace_back(aabb.min.x, aabb.min.y, aabb.max.z);
        vertexPositions.emplace_back(aabb.max.x, aabb.min.y, aabb.max.z);
        vertexPositions.emplace_back(aabb.max.x, aabb.max.y, aabb.max.z);
        vertexPositions.emplace_back(aabb.min.x, aabb.max.y, aabb.max.z);

        triangleIndices.reserve(6);
        for (uint32_t i = 2; i < 4; i++) {
            triangleIndices.push_back(0);
            triangleIndices.push_back(i - 1);
            triangleIndices.push_back(i);
        }

        vertexNormals.reserve(4);
        for (uint32_t i = 0; i < 4; i++) {
            vertexNormals.emplace_back(0.0f, 0.0f, 1.0f);
        }

        return;
    }

    /*
     * Idea for sorting points from: https://asawicki.info/news_1428_finding_polygon_of_plane-aabb_intersection
     */

    glm::vec3 dir = glm::vec3(aabb.max.x - aabb.min.x, 0.0f, 0.0f);
    glm::vec3 origin = aabb.min;
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.min.x, aabb.max.y, aabb.min.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.min.x, aabb.min.y, aabb.max.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.min.x, aabb.max.y, aabb.max.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);

    dir = glm::vec3(0.0f, aabb.max.y - aabb.min.y, 0.0f);
    origin = glm::vec3(aabb.min.x, aabb.min.y, aabb.min.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.max.x, aabb.min.y, aabb.min.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.min.x, aabb.min.y, aabb.max.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.max.x, aabb.min.y, aabb.max.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);

    dir = glm::vec3(0.0f, 0.0f, aabb.max.z - aabb.min.z);
    origin = glm::vec3(aabb.min.x, aabb.min.y, aabb.min.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.max.x, aabb.min.y, aabb.min.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.min.x, aabb.max.y, aabb.min.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);
    origin = glm::vec3(aabb.max.x, aabb.max.y, aabb.min.z);
    testRayPlaneIntersection(origin, dir, plane, polygonPoints);

    if (polygonPoints.size() < 3) {
        return;
    }

    glm::vec3 startPoint = polygonPoints.front();
    std::sort(polygonPoints.begin(), polygonPoints.end(), [this, startPoint](const glm::vec3& p0, const glm::vec3& p1) {
        glm::vec3 crossProdDirs = glm::cross(p0 - startPoint, p1 - startPoint);
        return glm::dot(crossProdDirs, planeNormal) < 0.0f;
    });

    vertexPositions = polygonPoints;
    auto numPoints = uint32_t(polygonPoints.size());
    triangleIndices.reserve((polygonPoints.size() - 2) * 3);
    for (uint32_t i = 2; i < numPoints; i++) {
        triangleIndices.push_back(0);
        triangleIndices.push_back(i - 1);
        triangleIndices.push_back(i);
    }

    vertexNormals.reserve(polygonPoints.size());
    for (uint32_t i = 0; i < numPoints; i++) {
        vertexNormals.push_back(planeNormal);
    }
}

void SliceRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    sliceRasterPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void SliceRenderer::renderViewImpl(uint32_t viewIdx) {
    if (indexBuffer) {
        sliceRasterPasses.at(viewIdx)->render();
    }
}

void SliceRenderer::addViewImpl(uint32_t viewIdx) {
    auto sliceRasterPass = std::make_shared<SliceRasterPass>(renderer, viewManager->getViewSceneData(viewIdx));
    if (volumeData) {
        sliceRasterPass->setVolumeData(volumeData, true);
        sliceRasterPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        sliceRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer);
    }
    sliceRasterPass->setLightingFactor(lightingFactor);
    sliceRasterPass->setNaNHandling(nanHandling);
    sliceRasterPasses.push_back(sliceRasterPass);
}

void SliceRenderer::removeViewImpl(uint32_t viewIdx) {
    sliceRasterPasses.erase(sliceRasterPasses.begin() + viewIdx);
}

void SliceRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (volumeData) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        int selectedFieldIdxGui = selectedFieldIdx;
        if (propertyEditor.addCombo("Scalar Field", &selectedFieldIdxGui, fieldNames.data(), int(fieldNames.size()))) {
            selectedFieldIdx = selectedFieldIdxGui;
            selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
            for (auto& sliceRasterPass : sliceRasterPasses) {
                sliceRasterPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
            }
            dirty = true;
            reRender = true;
        }
    }

    if (!volumeData || volumeData->getGridSizeZ() > 1) {
        if (propertyEditor.addSliderFloat3("Normal", &planeNormalUi.x, -1.0f, 1.0f)) {
            planeNormal = glm::normalize(planeNormalUi);
            dirty = true;
            reRender = true;
        }
        if (propertyEditor.addDragFloat("Distance", &planeDist, 0.001f)) {
            dirty = true;
            reRender = true;
        }
    }

    if (propertyEditor.addSliderFloat("Lighting Factor", &lightingFactor, 0.0f, 1.0f)) {
        for (auto& sliceRasterPass : sliceRasterPasses) {
            sliceRasterPass->setLightingFactor(lightingFactor);
        }
        reRender = true;
    }
    if (propertyEditor.addCombo(
            "NaN Handling", (int*)&nanHandling, NAN_HANDLING_NAMES, IM_ARRAYSIZE(NAN_HANDLING_NAMES))) {
        for (auto& sliceRasterPass : sliceRasterPasses) {
            sliceRasterPass->setNaNHandling(nanHandling);
        }
        reRender = true;
    }
}

void SliceRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);
    if (settings.getValueOpt("selected_field_idx", selectedFieldIdx)) {
        const std::vector<std::string>& fieldNames = volumeData->getFieldNames(FieldType::SCALAR);
        selectedFieldIdx = std::clamp(selectedFieldIdx, 0, int(fieldNames.size()) - 1);
        selectedScalarFieldName = fieldNames.at(selectedFieldIdx);
        for (auto& sliceRasterPass : sliceRasterPasses) {
            sliceRasterPass->setSelectedScalarField(selectedFieldIdx, selectedScalarFieldName);
        }
        dirty = true;
        reRender = true;
    }
    bool normalChanged = false;
    normalChanged |= settings.getValueOpt("normal_x", planeNormalUi.x);
    normalChanged |= settings.getValueOpt("normal_y", planeNormalUi.y);
    normalChanged |= settings.getValueOpt("normal_z", planeNormalUi.z);
    if (normalChanged) {
        planeNormal = glm::normalize(planeNormalUi);
        dirty = true;
        reRender = true;
    }
    if (settings.getValueOpt("plane_dist", planeDist)) {
        dirty = true;
        reRender = true;
    }
    if (settings.getValueOpt("lighting_factor", lightingFactor)) {
        for (auto& sliceRasterPass : sliceRasterPasses) {
            sliceRasterPass->setLightingFactor(lightingFactor);
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
        for (auto& sliceRasterPass : sliceRasterPasses) {
            sliceRasterPass->setNaNHandling(nanHandling);
        }
        reRender = true;
    }
}

void SliceRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);
    settings.addKeyValue("selected_field_idx", selectedFieldIdx);
    settings.addKeyValue("normal_x", planeNormalUi.x);
    settings.addKeyValue("normal_y", planeNormalUi.y);
    settings.addKeyValue("normal_z", planeNormalUi.z);
    settings.addKeyValue("plane_dist", planeDist);
    settings.addKeyValue("lighting_factor", lightingFactor);
    settings.addKeyValue("nan_handling", NAN_HANDLING_IDS[int(nanHandling)]);
}



SliceRasterPass::SliceRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : RasterPass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    rendererUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(RenderSettingsData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void SliceRasterPass::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;

    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    renderSettingsData.minBoundingBox = aabb.min;
    renderSettingsData.maxBoundingBox = aabb.max;

    dataDirty = true;
}

void SliceRasterPass::setSelectedScalarField(int _selectedFieldIdx, const std::string& _scalarFieldName) {
    selectedFieldIdx = _selectedFieldIdx;
    selectedScalarFieldName = _scalarFieldName;
    if (volumeData && rasterData) {
        auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
        rasterData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
    }
}

void SliceRasterPass::setRenderData(
        const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
        const sgl::vk::BufferPtr& _vertexNormalBuffer) {
    indexBuffer = _indexBuffer;
    vertexPositionBuffer = _vertexPositionBuffer;
    vertexNormalBuffer = _vertexNormalBuffer;

    setDataDirty();
    //if (rasterData) {
    //    rasterData->setIndexBuffer(indexBuffer);
    //    rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
    //    rasterData->setVertexBuffer(vertexNormalBuffer, "vertexNormal");
    //}
}

void SliceRasterPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    volumeData->getPreprocessorDefines(preprocessorDefines);
    if (nanHandling == NaNHandling::IGNORE) {
        preprocessorDefines.insert(std::make_pair("IGNORE_NAN", ""));
    } else if (nanHandling == NaNHandling::SHOW_AS_YELLOW) {
        preprocessorDefines.insert(std::make_pair("NAN_YELLOW", ""));
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            {"Slice.Vertex", "Slice.Fragment"}, preprocessorDefines);
}

void SliceRasterPass::setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
    pipelineInfo.setInputAssemblyTopology(sgl::vk::PrimitiveTopology::TRIANGLE_LIST);
    pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexNormal", sizeof(glm::vec3));
}

void SliceRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    auto scalarFieldData = volumeData->getFieldEntryDevice(FieldType::SCALAR, selectedScalarFieldName);
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    volumeData->setRenderDataBindings(rasterData);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
    rasterData->setVertexBuffer(vertexNormalBuffer, "vertexNormal");
    rasterData->setStaticBuffer(rendererUniformDataBuffer, "RendererUniformDataBuffer");
    rasterData->setStaticTexture(scalarFieldData->getVulkanTexture(), "scalarField");
}

void SliceRasterPass::recreateSwapchain(uint32_t width, uint32_t height) {
    framebuffer = std::make_shared<sgl::vk::Framebuffer>(device, width, height);

    sgl::vk::AttachmentState attachmentState;
    attachmentState.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachmentState.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachmentState.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    framebuffer->setColorAttachment(
            (*sceneData->sceneTexture)->getImageView(), 0, attachmentState,
            sceneData->clearColor->getFloatColorRGBA());

    sgl::vk::AttachmentState depthAttachmentState;
    depthAttachmentState.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachmentState.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachmentState.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    framebuffer->setDepthStencilAttachment(
            (*sceneData->sceneDepthTexture)->getImageView(), depthAttachmentState, 1.0f);

    framebufferDirty = true;
    dataDirty = true;
}

void SliceRasterPass::_render() {
    renderSettingsData.cameraPosition = (*camera)->getPosition();
    renderSettingsData.fieldIndex = uint32_t(selectedFieldIdx);
    rendererUniformDataBuffer->updateData(
            sizeof(RenderSettingsData), &renderSettingsData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    sceneData->switchColorState(RenderTargetAccess::RASTERIZER);
    if (sceneData->useDepthBuffer) {
        sceneData->switchDepthState(RenderTargetAccess::RASTERIZER);
    }

    RasterPass::_render();
}
