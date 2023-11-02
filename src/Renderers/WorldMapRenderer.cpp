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

#include <tiffio.h>

#include <Utils/AppSettings.hpp>
#include <Utils/File/Archive.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include "Utils/InternalState.hpp"
#include "Utils/Download.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "Raster/ShapefileRasterizer.hpp"
#include "RenderingModes.hpp"
#include "WorldMapRenderer.hpp"

const char* const WORLD_MAP_SOURCE_NAMES[] = {
        "TIFF File", "Shapefile Rasterizer"
};
const char* const WORLD_MAP_QUALITY_NAMES[] = {
        "Low", "Medium", "High"
};
const char* const WORLD_MAP_NAMES[] = {
        "HYP_50M_SR_W", "HYP_LR_SR_OB_DR", "HYP_HR_SR_OB_DR"
};
const char* const WORLD_MAP_URLS[] = {
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/raster/HYP_50M_SR_W.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/HYP_LR_SR_OB_DR.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/raster/HYP_HR_SR_OB_DR.zip"
};
const uint32_t WORLD_MAP_QUALITY_RES[] = {
        512, 1024, 2048
};

WorldMapRenderer::WorldMapRenderer(ViewManager* viewManager)
        : Renderer(RENDERING_MODE_NAMES[int(RENDERING_MODE_WORLD_MAP_RENDERER)], viewManager) {
    shapefileRasterizer = std::make_shared<ShapefileRasterizer>(renderer);
}

WorldMapRenderer::~WorldMapRenderer() {
    shapefileRasterizer = {};
}

void WorldMapRenderer::initialize() {
    Renderer::initialize();
}

void WorldMapRenderer::ensureWorldMapFileExistsTiff() {
    const std::string mapsDirectory = sgl::AppSettings::get()->getDataDirectory() + "Maps/";
    const std::string mapName = std::string(WORLD_MAP_NAMES[int(worldMapQuality)]);
    worldMapFilePath = mapsDirectory + mapName + ".tif";
    const std::string worldMapArchivePath = mapsDirectory + mapName + ".zip";
    if (sgl::FileUtils::get()->exists(worldMapFilePath)) {
        return;
    }

    if (!sgl::FileUtils::get()->directoryExists(mapsDirectory)) {
        sgl::FileUtils::get()->ensureDirectoryExists(mapsDirectory);
    }
    std::string worldMapUrl = WORLD_MAP_URLS[int(worldMapQuality)];
    if (!downloadFile(worldMapUrl, worldMapArchivePath)) {
        sgl::Logfile::get()->writeWarning("Warning: Downloading " + mapName + ".zip with CURL failed.", true);
    }

    std::unordered_map<std::string, sgl::ArchiveEntry> files;
    auto retType = sgl::loadAllFilesFromArchive(worldMapArchivePath, files, false);
    if (retType != sgl::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeWarning("Warning: " + mapName + ".zip could not be opened.", true);
        return;
    }
    auto it = files.find(mapName + ".tif");
    if (it == files.end()) {
        sgl::Logfile::get()->writeWarning(
                "Warning: " + mapName + ".zip does not contain " + mapName + ".tif.", true);
        sgl::FileUtils::get()->removeFile(worldMapArchivePath);
        return;
    }

    auto mapArchiveEntry = it->second;
    FILE* mapFile = fopen(worldMapFilePath.c_str(), "wb");
    if (!mapFile) {
        sgl::Logfile::get()->writeWarning("Warning: Could not create file " + mapName + ".tif.", true);
        sgl::FileUtils::get()->removeFile(worldMapArchivePath);
        return;
    }
    fwrite(mapArchiveEntry.bufferData.get(), 1, mapArchiveEntry.bufferSize, mapFile);
    fclose(mapFile);

    sgl::FileUtils::get()->removeFile(worldMapArchivePath);
    hasCheckedWorldMapExists = true;
}

void WorldMapRenderer::setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) {
    if (!volumeData) {
        isNewData = true;
    }
    volumeData = _volumeData;

    if (isNewData) {
        // Free old data first.
        indexBuffer = {};
        vertexPositionBuffer = {};
        vertexNormalBuffer = {};
        worldMapTexture = {};
        for (auto& worldMapRasterPass : worldMapRasterPasses) {
            worldMapRasterPass->setVolumeData(volumeData, isNewData);
            worldMapRasterPass->setRenderData(
                    indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
        }

        if (manuallySetRasterizer) {
            worldMapSource = WorldMapSource::TIFF_FILE;
            manuallySetRasterizer = false;
        }
        const float* latData = nullptr;
        const float* lonData = nullptr;
        volumeData->getLatLonData(latData, lonData);
        if (worldMapSource == WorldMapSource::TIFF_FILE && lonData && latData) {
            int xs = volumeData->getGridSizeX();
            int ys = volumeData->getGridSizeY();
            float minLon = std::numeric_limits<float>::max();
            float maxLon = std::numeric_limits<float>::lowest();
            float minLat = std::numeric_limits<float>::max();
            float maxLat = std::numeric_limits<float>::lowest();
            for (int y = 0; y < ys; y++) {
                for (int x = 0; x < xs; x++) {
                    float x_norm = lonData[x + y * xs];
                    float y_norm = latData[x + y * xs];
                    minLon = std::min(minLon, x_norm);
                    maxLon = std::max(maxLon, x_norm);
                    minLat = std::min(minLat, y_norm);
                    maxLat = std::max(maxLat, y_norm);
                }
            }
            std::cout << "dlon: " << maxLon - minLon << ", dlat: " << maxLat - minLat << std::endl;
            // Tokyo: (1.4, 1.2); Germany (Necker): (11.7, 9.6); Central Europe (Matsunobu): (23.6, 14.6)
            if (maxLon - minLon < 4.0f || maxLat - minLat < 4.0f) {
                worldMapSource = WorldMapSource::SHAPEFILE_RASTERIZER;
                manuallySetRasterizer = true;
            }
        }

        std::vector<uint32_t> triangleIndices;
        std::vector<glm::vec3> vertexPositions;
        std::vector<glm::vec3> vertexNormals;
        std::vector<glm::vec2> vertexTexCoords;
        createWorldMapTexture();
        createGeometryData(triangleIndices, vertexPositions, vertexNormals, vertexTexCoords);

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
        vertexTexCoordBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(glm::vec2) * vertexTexCoords.size(), vertexTexCoords.data(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    }

    for (auto& worldMapRasterPass : worldMapRasterPasses) {
        worldMapRasterPass->setVolumeData(volumeData, isNewData);
        worldMapRasterPass->setRenderData(
                indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
    }
}

void WorldMapRenderer::onFieldRemoved(FieldType fieldType, int fieldIdx) {
}

void WorldMapRenderer::createGeometryData(
        std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
        std::vector<glm::vec3>& vertexNormals, std::vector<glm::vec2>& vertexTexCoords) {
    sgl::AABB3 aabb = volumeData->getBoundingBoxRendering();
    std::vector<glm::vec3> polygonPoints;

    const float* latData = nullptr;
    const float* lonData = nullptr;
    volumeData->getLatLonData(latData, lonData);
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int numVertices = xs * ys;

    vertexPositions.reserve(numVertices);
    vertexNormals.reserve(numVertices);
    vertexTexCoords.reserve(numVertices);
    for (int y = 0; y < ys; y++) {
        for (int x = 0; x < xs; x++) {
            vertexPositions.emplace_back(
                    aabb.min.x + (aabb.max.x - aabb.min.x) * float(x) / float(xs - 1),
                    aabb.min.y + (aabb.max.y - aabb.min.y) * float(y) / float(ys - 1),
                    aabb.min.z);
            vertexNormals.emplace_back(0.0f, 0.0f, 1.0f);
            float x_norm = (lonData ? lonData[x + y * xs] : 0.0f) / 360.0f + 0.5f;
            float y_norm = (latData ? latData[x + y * xs] : 0.0f) / 180.0f + 0.5f;
            float texCoordX = (x_norm - minNormX) / (maxNormX - minNormX);
            float texCoordY = (y_norm - minNormY) / (maxNormY - minNormY);
            vertexTexCoords.emplace_back(texCoordX, texCoordY);
        }
    }

    triangleIndices.reserve(6 * (xs - 1) * (ys - 1));
    for (int y = 0; y < ys - 1; y++) {
        for (int x = 0; x < xs - 1; x++) {
            triangleIndices.push_back(x     + y       * xs);
            triangleIndices.push_back(x + 1 + y       * xs);
            triangleIndices.push_back(x     + (y + 1) * xs);
            triangleIndices.push_back(x + 1 + y       * xs);
            triangleIndices.push_back(x + 1 + (y + 1) * xs);
            triangleIndices.push_back(x     + (y + 1) * xs);
        }
    }
}

void WorldMapRenderer::createWorldMapTexture() {
    if (worldMapSource == WorldMapSource::TIFF_FILE) {
        createWorldMapTextureTiff();
    } else if (worldMapSource == WorldMapSource::SHAPEFILE_RASTERIZER) {
        createWorldMapTextureShapefile();
    }
}

void WorldMapRenderer::createWorldMapTextureTiff() {
    if (!hasCheckedWorldMapExists) {
        ensureWorldMapFileExistsTiff();
    }

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

    TIFF* tif = TIFFOpen(worldMapFilePath.c_str(), "r");

    uint16_t tiffEncoding = 0;
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &tiffEncoding);
    if (tiffEncoding == COMPRESSION_JBIG) {
        sgl::Logfile::get()->throwError("Error: JBIG encoding is disabled to comply with the license of JBIG-KIT.");
    }

    uint32_t imageWidth, imageHeight;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imageWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imageHeight);

    minNormX = std::numeric_limits<float>::max();
    maxNormX = std::numeric_limits<float>::lowest();
    minNormY = std::numeric_limits<float>::max();
    maxNormY = std::numeric_limits<float>::lowest();
    for (int y = 0; y < ys; y++) {
        for (int x = 0; x < xs; x++) {
            float x_norm = lonData[x + y * xs] / 360.0f + 0.5f;
            float y_norm = latData[x + y * xs] / 180.0f + 0.5f;
            minNormX = std::min(minNormX, x_norm);
            maxNormX = std::max(maxNormX, x_norm);
            minNormY = std::min(minNormY, y_norm);
            maxNormY = std::max(maxNormY, y_norm);
        }
    }
    xl = uint32_t(std::floor(float(imageWidth - 1) * minNormX));
    yl = uint32_t(std::floor(float(imageHeight - 1) * minNormY));
    xu = uint32_t(std::ceil(float(imageWidth - 1) * maxNormX));
    yu = uint32_t(std::ceil(float(imageHeight - 1) * maxNormY));
    minNormX = float(xl) / float(imageWidth - 1);
    minNormY = float(yl) / float(imageHeight - 1);
    maxNormX = float(xu) / float(imageWidth - 1);
    maxNormY = float(yu) / float(imageHeight - 1);
    regionImageWidth = xu - xl + 1;
    regionImageHeight = yu - yl + 1;

    auto* imageDataRgba = reinterpret_cast<uint32_t*>(_TIFFmalloc(imageWidth * imageHeight * sizeof(uint32_t)));
    if (imageDataRgba == nullptr) {
        sgl::Logfile::get()->throwError("Error: _TIFFmalloc failed!");
    }
    if (!TIFFReadRGBAImage(tif, imageWidth, imageHeight, imageDataRgba, 0)) {
        _TIFFfree(imageDataRgba);
        sgl::Logfile::get()->throwError("Error: TIFFReadRGBAImage failed!");
    }

    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = uint32_t(regionImageWidth);
    imageSettings.height = uint32_t(regionImageHeight);
    sgl::vk::ImageSamplerSettings imageSamplerSettings{};
    worldMapTexture = std::make_shared<sgl::vk::Texture>(
            renderer->getDevice(), imageSettings, imageSamplerSettings);

    const auto worldMapSizeBytes = size_t(4) * size_t(regionImageWidth) * size_t(regionImageHeight);
    auto* worldMapData = new uint8_t[worldMapSizeBytes];

    for (uint32_t y = 0; y < regionImageHeight; y++) {
        for (uint32_t x = 0; x < regionImageWidth; x++) {
            uint32_t color = imageDataRgba[xl + x + (yl + y) * imageWidth];
            int r = TIFFGetR(color);
            int g = TIFFGetG(color);
            int b = TIFFGetB(color);
            auto writeIdx = (x + y * regionImageWidth) * 4;
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

    /*int xs = volumeData->getGridSizeX();
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

    uint32_t imageWidth, imageHeight;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imageWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imageHeight);
    auto* imageDataRgba = reinterpret_cast<uint32_t*>(_TIFFmalloc(imageWidth * imageHeight * sizeof(uint32_t)));
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
    delete[] worldMapData;*/
}

void WorldMapRenderer::createWorldMapTextureShapefile() {
    if (!volumeData->getHasLatLonData()) {
        sgl::Logfile::get()->writeWarning(
                "Warning in WorldMapRenderer::createWorldMapTexture: The volume has no lat/lon data.");
        return;
    }

    if (!hasCheckedWorldMapExists) {
        shapefileRasterizer->checkTempFiles();
        hasCheckedWorldMapExists = true;
    }

    // TODO: Use worldMapFilePath to cache image for future runs?

    const float* latData = nullptr;
    const float* lonData = nullptr;
    volumeData->getLatLonData(latData, lonData);
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();

    float minLon = std::numeric_limits<float>::max();
    float maxLon = std::numeric_limits<float>::lowest();
    float minLat = std::numeric_limits<float>::max();
    float maxLat = std::numeric_limits<float>::lowest();
    for (int y = 0; y < ys; y++) {
        for (int x = 0; x < xs; x++) {
            float x_norm = lonData[x + y * xs];
            float y_norm = latData[x + y * xs];
            minLon = std::min(minLon, x_norm);
            maxLon = std::max(maxLon, x_norm);
            minLat = std::min(minLat, y_norm);
            maxLat = std::max(maxLat, y_norm);
        }
    }
    minNormX = minLon / 360.0f + 0.5f;
    maxNormX = maxLon / 360.0f + 0.5f;
    minNormY = minLat / 180.0f + 0.5f;
    maxNormY = maxLat / 180.0f + 0.5f;

    regionImageWidth = WORLD_MAP_QUALITY_RES[int(worldMapQuality)];
    regionImageHeight = WORLD_MAP_QUALITY_RES[int(worldMapQuality)];
    if (xs > ys) {
        regionImageHeight = uint32_t(std::ceil(float(regionImageWidth) * float(ys) / float(xs)));
    } else if (xs < ys) {
        regionImageWidth = uint32_t(std::ceil(float(regionImageHeight) * float(xs) / float(ys)));
    }

    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = uint32_t(regionImageWidth);
    imageSettings.height = uint32_t(regionImageHeight);
    sgl::vk::ImageSamplerSettings imageSamplerSettings{};
    worldMapTexture = std::make_shared<sgl::vk::Texture>(
            renderer->getDevice(), imageSettings, imageSamplerSettings);

    const auto worldMapSizeBytes = size_t(4) * size_t(regionImageWidth) * size_t(regionImageHeight);
    auto* worldMapData = new uint8_t[worldMapSizeBytes];
    shapefileRasterizer->rasterize(minLon, maxLon, minLat, maxLat, regionImageWidth, regionImageHeight, worldMapData);

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
        worldMapRasterPass->setRenderData(
                indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
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
    if (propertyEditor.addCombo(
            "Source", (int*)&worldMapSource, WORLD_MAP_SOURCE_NAMES, IM_ARRAYSIZE(WORLD_MAP_SOURCE_NAMES))) {
        hasCheckedWorldMapExists = false;
        manuallySetRasterizer = false;
        createWorldMapTexture();
        for (auto& worldMapRasterPass : worldMapRasterPasses) {
            worldMapRasterPass->setRenderData(
                    indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
        }
        reRender = true;
    }
    if (propertyEditor.addCombo(
            "Resolution", (int*)&worldMapQuality, WORLD_MAP_QUALITY_NAMES, IM_ARRAYSIZE(WORLD_MAP_QUALITY_NAMES))) {
        hasCheckedWorldMapExists = false;
        createWorldMapTexture();
        for (auto& worldMapRasterPass : worldMapRasterPasses) {
            worldMapRasterPass->setRenderData(
                    indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
        }
        reRender = true;
    }
    if (worldMapSource == WorldMapSource::SHAPEFILE_RASTERIZER) {
        if (shapefileRasterizer->renderGuiPropertyEditor(propertyEditor)) {
            hasCheckedWorldMapExists = false;
            createWorldMapTexture();
            for (auto& worldMapRasterPass : worldMapRasterPasses) {
                worldMapRasterPass->setRenderData(
                        indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
            }
            reRender = true;
        }

    }
}

void WorldMapRenderer::setSettings(const SettingsMap& settings) {
    Renderer::setSettings(settings);
    if (settings.getValueOpt("lighting_factor", lightingFactor)) {
        for (auto& worldMapRasterPass : worldMapRasterPasses) {
            worldMapRasterPass->setLightingFactor(lightingFactor);
        }
        reRender = true;
    }
    std::string worldMapSourceString;
    if (settings.getValueOpt("world_map_source", worldMapSourceString)) {
        manuallySetRasterizer = false;
        for (int i = 0; i < IM_ARRAYSIZE(WORLD_MAP_SOURCE_NAMES); i++) {
            if (worldMapSourceString == WORLD_MAP_SOURCE_NAMES[i]) {
                worldMapSource = WorldMapSource(i);
                break;
            }
        }
        hasCheckedWorldMapExists = false;
        createWorldMapTexture();
        for (auto& worldMapRasterPass : worldMapRasterPasses) {
            worldMapRasterPass->setRenderData(
                    indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
        }
        reRender = true;
    } else {
        worldMapSource = WorldMapSource::TIFF_FILE; //< Old default.
    }
    std::string worldMapQualityString;
    if (settings.getValueOpt("world_map_quality", worldMapQualityString)) {
        for (int i = 0; i < IM_ARRAYSIZE(WORLD_MAP_QUALITY_NAMES); i++) {
            if (worldMapQualityString == WORLD_MAP_QUALITY_NAMES[i]) {
                worldMapQuality = WorldMapQuality(i);
                break;
            }
        }
        hasCheckedWorldMapExists = false;
        createWorldMapTexture();
        for (auto& worldMapRasterPass : worldMapRasterPasses) {
            worldMapRasterPass->setRenderData(
                    indexBuffer, vertexPositionBuffer, vertexNormalBuffer, vertexTexCoordBuffer, worldMapTexture);
        }
        reRender = true;
    }
}

void WorldMapRenderer::getSettings(SettingsMap& settings) {
    Renderer::getSettings(settings);
    settings.addKeyValue("lighting_factor", lightingFactor);
    settings.addKeyValue("world_map_source", WORLD_MAP_SOURCE_NAMES[int(worldMapSource)]);
    settings.addKeyValue("world_map_quality", WORLD_MAP_QUALITY_NAMES[int(worldMapQuality)]);
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
        const sgl::vk::BufferPtr& _vertexNormalBuffer, const sgl::vk::BufferPtr& _vertexTexCoordBuffer,
        const sgl::vk::TexturePtr& _worldMapTexture) {
    indexBuffer = _indexBuffer;
    vertexPositionBuffer = _vertexPositionBuffer;
    vertexNormalBuffer = _vertexNormalBuffer;
    vertexTexCoordBuffer = _vertexTexCoordBuffer;
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
    pipelineInfo.setVertexBufferBindingByLocationIndex("vertexTexCoord", sizeof(glm::vec2));
}

void WorldMapRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    volumeData->setRenderDataBindings(rasterData);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexPositionBuffer, "vertexPosition");
    rasterData->setVertexBuffer(vertexNormalBuffer, "vertexNormal");
    rasterData->setVertexBuffer(vertexTexCoordBuffer, "vertexTexCoord");
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
