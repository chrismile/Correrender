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

#include <boost/algorithm/string/replace.hpp>
#include <tiffio.h>
#include <curl/curl.h>

#include <Utils/AppSettings.hpp>
#include <Utils/File/Archive.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "RenderingModes.hpp"
#include "WorldMapRenderer.hpp"

static size_t writeDataCallbackCurl(void *pointer, size_t size, size_t numMembers, void *stream) {
    size_t written = fwrite(pointer, size, numMembers, (FILE*)stream);
    return written;
}

bool downloadFile(const std::string &url, const std::string &localFileName) {
    CURL* curlHandle = curl_easy_init();
    if (!curlHandle) {
        return false;
    }
    CURLcode curlErrorCode = CURLE_OK;

    char* compressedUrl = curl_easy_escape(curlHandle, url.c_str(), url.size());
    std::string fixedUrl = compressedUrl;
    boost::replace_all(fixedUrl, "%3A", ":");
    boost::replace_all(fixedUrl, "%2F", "/");
    std::cout << "Starting to download \"" << fixedUrl << "\"..." << std::endl;

    curl_easy_setopt(curlHandle, CURLOPT_URL, fixedUrl.c_str());
    //curl_easy_setopt(curlHandle, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curlHandle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curlHandle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, writeDataCallbackCurl);
    FILE* pagefile = fopen(localFileName.c_str(), "wb");
    if (pagefile) {
        curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, pagefile);
        curlErrorCode = curl_easy_perform(curlHandle);
        if (curlErrorCode != CURLE_OK) {
            fclose(pagefile);
            curl_free(compressedUrl);
            curl_easy_cleanup(curlHandle);
            return false;
        }
        fclose(pagefile);
    }

    curl_free(compressedUrl);
    curl_easy_cleanup(curlHandle);
    return true;
}

WorldMapRenderer::WorldMapRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_WORLD_MAP_RENDERER)], viewManager) {
    ensureWorldMapFileExists();
}

WorldMapRenderer::~WorldMapRenderer() {
}

void WorldMapRenderer::initialize() {
    Renderer::initialize();
}

void WorldMapRenderer::ensureWorldMapFileExists() {
    const std::string mapsDirectory = sgl::AppSettings::get()->getDataDirectory() + "Maps/";
    worldMapFilePath = mapsDirectory + "HYP_50M_SR_W.tif";
    const std::string worldMapArchivePath = mapsDirectory + "HYP_50M_SR_W.zip";
    if (sgl::FileUtils::get()->exists(worldMapFilePath)) {
        return;
    }

    if (!sgl::FileUtils::get()->directoryExists(mapsDirectory)) {
        sgl::FileUtils::get()->ensureDirectoryExists(mapsDirectory);
    }
    std::string worldMapUrl =
            "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/raster/HYP_50M_SR_W.zip";
    if (!downloadFile(worldMapUrl, worldMapArchivePath)) {
        sgl::Logfile::get()->writeWarning("Warning: Downloading HYP_50M_SR_W.zip with CURL failed.", true);
    }

    std::unordered_map<std::string, sgl::ArchiveEntry> files;
    auto retType = sgl::loadAllFilesFromArchive(worldMapArchivePath, files, false);
    if (retType != sgl::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeWarning("Warning: HYP_50M_SR_W.zip could not be opened.", true);
        return;
    }
    auto it = files.find("HYP_50M_SR_W.tif");
    if (it == files.end()) {
        sgl::Logfile::get()->writeWarning(
                "Warning: HYP_50M_SR_W.zip does not contain HYP_50M_SR_W/HYP_50M_SR_W.tif.", true);
        sgl::FileUtils::get()->removeFile(worldMapArchivePath);
        return;
    }

    auto mapArchiveEntry = it->second;
    FILE* mapFile = fopen(worldMapFilePath.c_str(), "wb");
    if (!mapFile) {
        sgl::Logfile::get()->writeWarning("Warning: Could not create file HYP_50M_SR_W.tif.", true);
        sgl::FileUtils::get()->removeFile(worldMapArchivePath);
        return;
    }
    fwrite(mapArchiveEntry.bufferData.get(), 1, mapArchiveEntry.bufferSize, mapFile);
    fclose(mapFile);

    sgl::FileUtils::get()->removeFile(worldMapArchivePath);
}

void WorldMapRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;

    indexBuffer = {};
    vertexPositionBuffer = {};
    vertexNormalBuffer = {};
    worldMapTexture = {};

    for (auto& worldMapRasterPass : worldMapRasterPasses) {
        worldMapRasterPass->setVolumeData(volumeData, isNewData);
        worldMapRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer, worldMapTexture);
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

    if (isNewData) {
        createWorldMapTexture();
    }

    for (auto& worldMapRasterPass : worldMapRasterPasses) {
        worldMapRasterPass->setVolumeData(volumeData, isNewData);
        worldMapRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer, worldMapTexture);
    }
}

void WorldMapRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
}

void WorldMapRenderer::createGeometryData(
        std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
        std::vector<glm::vec3>& vertexNormals) {
    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    std::vector<glm::vec3> polygonPoints;

    // Special case: The volume is already a slice.
    vertexPositions.emplace_back(aabb.min.x, aabb.min.y, aabb.min.z);
    vertexPositions.emplace_back(aabb.max.x, aabb.min.y, aabb.min.z);
    vertexPositions.emplace_back(aabb.max.x, aabb.max.y, aabb.min.z);
    vertexPositions.emplace_back(aabb.min.x, aabb.max.y, aabb.min.z);

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
}

void WorldMapRenderer::createWorldMapTexture() {
    if (!volumeData->getHasLatLonData()) {
        sgl::Logfile::get()->writeWarning(
                "Warning in WorldMapRenderer::createWorldMapTexture: The volume has no lat/lon data.");
        return;
    }
    if (!sgl::FileUtils::get()->exists(worldMapFilePath)) {
        return;
    }

    const float* latData = nullptr;
    const float* lonData = nullptr;
    volumeData->getLatLonData(latData, lonData);
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = uint32_t(xs);
    imageSettings.height = uint32_t(ys);
    sgl::vk::ImageSamplerSettings imageSamplerSettings{};
    worldMapTexture = std::make_shared<sgl::vk::Texture>(
            renderer->getDevice(), imageSettings, imageSamplerSettings);

    const auto worldMapSizeBytes = size_t(4 * xs * ys);
    auto* worldMapData = new uint8_t[worldMapSizeBytes];

    TIFF* tif = TIFFOpen(worldMapFilePath.c_str(), "r");

    uint16_t tiffEncoding = 0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &tiffEncoding);
    if (tiffEncoding == COMPRESSION_JBIG) {
        sgl::Logfile::get()->throwError("Error: JBIG encoding is disabled to comply with the license of JBIG-KIT.");
    }

    uint32 imageWidth, imageHeight;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imageWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imageHeight);
    auto* imageDataRgba = reinterpret_cast<uint32*>(_TIFFmalloc(imageWidth * imageHeight * sizeof(uint32)));
    if (imageDataRgba == nullptr) {
        sgl::Logfile::get()->throwError("Error: _TIFFmalloc failed!");
    }
    if (!TIFFReadRGBAImage(tif, imageWidth, imageHeight, imageDataRgba, 0)) {
        _TIFFfree(imageDataRgba);
        sgl::Logfile::get()->throwError("Error: TIFFReadRGBAImage failed!");
    }

    for (int y = 0; y < ys; y++) {
        for (int x = 0; x < xs; x++) {
            float x_norm = lonData[x + y * xs] / 360.0f + 0.5f;
            float y_norm = latData[x + y * xs] / 180.0f + 0.5f;
            auto x_in = uint32_t(std::clamp(int(std::round(x_norm * float(imageWidth))), 0, int(imageWidth) - 1));
            auto y_in = uint32_t(std::clamp(int(std::round(y_norm * float(imageHeight))), 0, int(imageHeight) - 1));
            uint32_t color = imageDataRgba[x_in + y_in * imageWidth];
            int r = TIFFGetR(color);
            int g = TIFFGetG(color);
            int b = TIFFGetB(color);
            int writeIdx = (x + y * xs) * 4;
            worldMapData[writeIdx] = r;
            worldMapData[writeIdx + 1] = g;
            worldMapData[writeIdx + 2] = b;
            worldMapData[writeIdx + 3] = 255;
        }
    }

    _TIFFfree(imageDataRgba);
    TIFFClose(tif);

    worldMapTexture->getImage()->uploadData(worldMapSizeBytes, worldMapData);
    delete[] worldMapData;
}

void WorldMapRenderer::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    worldMapRasterPasses.at(viewIdx)->recreateSwapchain(width, height);
}

void WorldMapRenderer::renderViewImpl(uint32_t viewIdx) {
    if (indexBuffer && worldMapTexture) {
        worldMapRasterPasses.at(viewIdx)->render();
    }
}

void WorldMapRenderer::addViewImpl(uint32_t viewIdx) {
    auto worldMapRasterPass = std::make_shared<WorldMapRasterPass>(renderer, viewManager->getViewSceneData(viewIdx));
    if (volumeData) {
        worldMapRasterPass->setVolumeData(volumeData, true);
        worldMapRasterPass->setRenderData(indexBuffer, vertexPositionBuffer, vertexNormalBuffer, worldMapTexture);
    }
    worldMapRasterPass->setLightingFactor(lightingFactor);
    worldMapRasterPasses.push_back(worldMapRasterPass);
}

void WorldMapRenderer::removeViewImpl(uint32_t viewIdx) {
    worldMapRasterPasses.erase(worldMapRasterPasses.begin() + viewIdx);
}

void WorldMapRenderer::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addSliderFloat("Lighting Factor", &lightingFactor, 0.0f, 1.0f)) {
        for (auto& worldMapRasterPass : worldMapRasterPasses) {
            worldMapRasterPass->setLightingFactor(lightingFactor);
        }
        reRender = true;
    }
}



WorldMapRasterPass::WorldMapRasterPass(sgl::vk::Renderer* renderer, SceneData* sceneData)
        : RasterPass(renderer), sceneData(sceneData), camera(&sceneData->camera) {
    rendererUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(RenderSettingsData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void WorldMapRasterPass::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    this->volumeData = _volumeData;

    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    renderSettingsData.minBoundingBox = aabb.min;
    renderSettingsData.maxBoundingBox = aabb.max;

    dataDirty = true;
}

void WorldMapRasterPass::setRenderData(
        const sgl::vk::BufferPtr& _indexBuffer, const sgl::vk::BufferPtr& _vertexPositionBuffer,
        const sgl::vk::BufferPtr& _vertexNormalBuffer, const sgl::vk::TexturePtr& _worldMapTexture) {
    indexBuffer = _indexBuffer;
    vertexPositionBuffer = _vertexPositionBuffer;
    vertexNormalBuffer = _vertexNormalBuffer;
    worldMapTexture = _worldMapTexture;

    setDataDirty();
}

void WorldMapRasterPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    volumeData->getPreprocessorDefines(preprocessorDefines);
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            {"WorldMap.Vertex", "WorldMap.Fragment"}, preprocessorDefines);
}

void WorldMapRasterPass::setGraphicsPipelineInfo(sgl::vk::GraphicsPipelineInfo& pipelineInfo) {
    pipelineInfo.setInputAssemblyTopology(sgl::vk::PrimitiveTopology::TRIANGLE_LIST);
    pipelineInfo.setCullMode(sgl::vk::CullMode::CULL_NONE);
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexPosition", sizeof(glm::vec3));
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexNormal", sizeof(glm::vec3));
}

void WorldMapRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    volumeData->setRenderDataBindings(rasterData);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
    rasterData->setVertexBuffer(vertexNormalBuffer, "vertexNormal");
    rasterData->setStaticBuffer(rendererUniformDataBuffer, "RendererUniformDataBuffer");
    rasterData->setStaticTexture(worldMapTexture, "worldMapTexture");
}

void WorldMapRasterPass::recreateSwapchain(uint32_t width, uint32_t height) {
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

void WorldMapRasterPass::_render() {
    renderSettingsData.cameraPosition = (*camera)->getPosition();
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
