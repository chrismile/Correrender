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
#include <Graphics/Color.hpp>

#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Archive.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>
#include <Graphics/Vector/nanovg/nanovg.h>

#ifdef SUPPORT_SKIA
#include <core/SkCanvas.h>
#include <core/SkPaint.h>
#include <core/SkPath.h>
#include <core/SkShader.h>
#include <core/SkFont.h>
#include <core/SkFontMetrics.h>
#include <effects/SkGradientShader.h>
#endif
#ifdef SUPPORT_VKVG
#include <vkvg.h>
#endif

#ifdef SUPPORT_SKIA
#include "../Diagram/VectorBackendSkia.hpp"
#endif
#ifdef SUPPORT_VKVG
#include "../Diagram/VectorBackendVkvg.hpp"
#endif

#include "Utils/Download.hpp"
#include "Loaders/LoadersUtil.hpp"
#include "ShapefileRasterizer.hpp"

enum class ShapeType : uint32_t {
    NULL_SHAPE = 0,
    POINT = 1, // x, y
    POLYLINE = 3, // AABB, #parts, #points, parts, points
    POLYGON = 5, // AABB, #parts, #points, parts, points
    MULTI_POINT = 8, // AABB, #points, points
};

/*
 * https://en.wikipedia.org/wiki/Shapefile
 * https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf
 */
#pragma pack(4)
struct ShapefileHeader {
    uint32_t fileCode; // Big-endian; must be 0x0000270a.
    uint32_t unused[5]; // Big-endian.
    uint32_t fileLength; // Big-endian; total length, including header, as multiple of 2 bytes.
    uint32_t version;
    ShapeType shapeType;
    double minX, minY, maxX, maxY;
    double minZ, maxZ;
    double minM, maxM;
};

struct RecordHeader {
    uint32_t recordNumber; // Big-endian
    uint32_t recordLength; // Big-endian; as multiple of 2 bytes.
    ShapeType shapeType;
};

struct ShapefilePoint {
    double x, y;
};

struct ShapefileAabb {
    double minX, minY, maxX, maxY;
};

ShapefileRasterizer::ShapefileRasterizer(sgl::vk::Renderer* renderer) : renderer(renderer) {
    setRendererVk(renderer);
    windowOffsetX = 0;
    windowOffsetY = 0;
    customScaleFactor = 1.0f; // Overwrite automatic UI scale.

    sgl::NanoVGSettings nanoVgSettings{};
#ifdef SUPPORT_OPENGL
    if (sgl::AppSettings::get()->getOffscreenContext()) {
        nanoVgSettings.renderBackend = sgl::RenderSystem::OPENGL;
    } else {
#endif
        nanoVgSettings.renderBackend = sgl::RenderSystem::VULKAN;
#ifdef SUPPORT_OPENGL
    }
#endif

    registerRenderBackendIfSupported<sgl::VectorBackendNanoVG>([this]() { this->renderBaseNanoVG(); }, nanoVgSettings);
#ifdef SUPPORT_SKIA
    registerRenderBackendIfSupported<VectorBackendSkia>([this]() { this->renderBaseSkia(); });
#endif
#ifdef SUPPORT_VKVG
    registerRenderBackendIfSupported<VectorBackendVkvg>([this]() { this->renderBaseVkvg(); });
#endif

    std::string defaultBackendId = sgl::VectorBackendNanoVG::getClassID();
#if defined(SUPPORT_SKIA) || defined(SUPPORT_VKVG)
    // NanoVG Vulkan port is broken at the moment, so use Skia or VKVG if OpenGL NanoVG cannot be used.
    if (!sgl::AppSettings::get()->getOffscreenContext()) {
#if defined(SUPPORT_SKIA)
        defaultBackendId = VectorBackendSkia::getClassID();
#elif defined(SUPPORT_VKVG)
        defaultBackendId = VectorBackendVkvg::getClassID();
#endif
    }
#endif
#if defined(__linux__) && defined(SUPPORT_VKVG)
    // OpenGL interop seems to results in kernel soft lockups as of 2023-07-06 on NVIDIA hardware.
    //sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    //if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
    //    defaultBackendId = VectorBackendVkvg::getClassID();
    //}
#endif
    setDefaultBackendId(defaultBackendId);

    _initialize();
}

const char* const SHAPEFILE_ARCHIVE_URLS[] = {
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_bathymetry_L_0.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_bathymetry_K_200.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_bathymetry_J_1000.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_land.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_rivers_lake_centerlines.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_lakes.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip",
        //"https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_bathymetry_all.zip"
};
const char* const SHAPEFILE_FILENAMES[] = {
        "ne_10m_bathymetry_L_0.shp",
        "ne_10m_bathymetry_K_200.shp",
        "ne_10m_bathymetry_J_1000.shp",
        "ne_10m_land.shp",
        "ne_10m_rivers_lake_centerlines.shp",
        "ne_10m_lakes.shp",
        "ne_10m_coastline.shp",
};
const sgl::Color SHAPEFILE_FILL_COLORS[] = {
        sgl::Color(177, 215, 239, 255), // bright: sgl::Color(208, 231, 245, 255)
        sgl::Color(135, 189, 223, 255),
        sgl::Color(127, 183, 218, 255),
        sgl::Color(114, 158, 145, 255), // brown: sgl::Color(200, 110, 55, 255)
        sgl::Color(0, 0, 0, 0),
        sgl::Color(200, 228, 245, 255),
        sgl::Color(0, 0, 0, 0),
};
const sgl::Color SHAPEFILE_STROKE_COLORS[] = {
        sgl::Color(0, 0, 0, 0),
        sgl::Color(0, 0, 0, 0),
        sgl::Color(0, 0, 0, 0),
        sgl::Color(0, 0, 0, 0),
        sgl::Color(53, 146, 173, 255),
        sgl::Color(53, 146, 173, 255),
        sgl::Color(53, 146, 173, 255),
};

#define CUSTOM_ARRAYSIZE(_ARR) ((int)(sizeof(_ARR) / sizeof(*(_ARR))))

void ShapefileRasterizer::checkTempFiles() {
    const std::string mapsDirectory = sgl::AppSettings::get()->getDataDirectory() + "Maps/";
    for (int i = 0; i < CUSTOM_ARRAYSIZE(SHAPEFILE_FILENAMES); i++) {
        checkCached(mapsDirectory, SHAPEFILE_ARCHIVE_URLS[i], SHAPEFILE_FILENAMES[i]);
    }
}

void ShapefileRasterizer::checkCached(
        const std::string& mapsDirectory, const std::string& archiveUrl, const std::string& filename) {
    std::string targetPath = mapsDirectory + filename;
    if (sgl::FileUtils::get()->exists(targetPath)) {
        return;
    }

    if (!sgl::FileUtils::get()->directoryExists(mapsDirectory)) {
        sgl::FileUtils::get()->ensureDirectoryExists(mapsDirectory);
    }
    std::string archiveName = archiveUrl.substr(archiveUrl.find_last_of('/') + 1);
    std::string archivePath = mapsDirectory + archiveName;
    if (!sgl::FileUtils::get()->exists(archivePath)) {
        if (!downloadFile(archiveUrl, archivePath)) {
            sgl::Logfile::get()->writeWarning("Warning: Downloading " + archiveName + " with CURL failed.", true);
        }
    }

    std::unordered_map<std::string, sgl::ArchiveEntry> files;
    auto retType = sgl::loadAllFilesFromArchive(archivePath, files, false);
    if (retType != sgl::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeWarning("Warning: " + archiveName + " could not be opened.", true);
        return;
    }
    auto it = files.find(filename);
    if (it == files.end()) {
        sgl::Logfile::get()->writeWarning("Warning: " + archiveName + " does not contain " + filename + ".", true);
        return;
    }

    auto mapArchiveEntry = it->second;
    FILE* outputFile = fopen(targetPath.c_str(), "wb");
    if (!outputFile) {
        sgl::Logfile::get()->writeWarning("Warning: Could not create file " + filename + ".", true);
        return;
    }
    fwrite(mapArchiveEntry.bufferData.get(), 1, mapArchiveEntry.bufferSize, outputFile);
    fclose(outputFile);
}

void ShapefileRasterizer::rasterize(
        float _minLon, float _maxLon, float _minLat, float _maxLat,
        uint32_t regionImageWidth, uint32_t regionImageHeight, uint8_t* worldMapData) {
    minLon = double(_minLon);
    maxLon = double(_maxLon);
    minLat = double(_minLat);
    maxLat = double(_maxLat);

    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = uint32_t(regionImageWidth);
    imageSettings.height = uint32_t(regionImageHeight);
    imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
    auto imageViewTarget = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(
            renderer->getDevice(), imageSettings));
    imageSettings.tiling = VK_IMAGE_TILING_LINEAR;
    imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    auto imageStaging = std::make_shared<sgl::vk::Image>(renderer->getDevice(), imageSettings);
    imageStaging->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, renderer->getVkCommandBuffer());

    windowWidth = float(regionImageWidth);
    windowHeight = float(regionImageHeight);
    setBlitTargetVk(imageViewTarget, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    onWindowSizeChanged();
    render();
    blitToTargetVk();
    imageViewTarget->getImage()->copyToImage(imageStaging, VK_IMAGE_ASPECT_COLOR_BIT, renderer->getVkCommandBuffer());
    renderer->syncWithCpu();

    auto subresourceLayout = imageStaging->getSubresourceLayout(VK_IMAGE_ASPECT_COLOR_BIT);
    auto* mappedData = reinterpret_cast<uint8_t*>(imageStaging->mapMemory());
    for (uint32_t y = 0; y < regionImageHeight; y++) {
        for (uint32_t x = 0; x < regionImageWidth; x++) {
            uint32_t readOffset =
                    uint32_t(subresourceLayout.offset) + x * 4
                    + uint32_t(subresourceLayout.rowPitch) * (regionImageHeight - y - 1);
            uint32_t writeOffset = (x + y * regionImageWidth) * 4;
            worldMapData[writeOffset] = mappedData[readOffset];
            worldMapData[writeOffset + 1] = mappedData[readOffset + 1];
            worldMapData[writeOffset + 2] = mappedData[readOffset + 2];
            worldMapData[writeOffset + 3] = 255;
        }
    }
    imageStaging->unmapMemory();
}

void ShapefileRasterizer::rasterizeShapefiles() {
#ifdef SUPPORT_SKIA
    if (canvas) {
        paint = new SkPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(paint);
    }
#endif

    // Clear the canvas.
    if (vg) {
        NVGcolor backgroundFillColorNvg = nvgRGBA(
                backgroundFillColor.getR(), backgroundFillColor.getG(), backgroundFillColor.getB(), 255);
        nvgBeginPath(vg);
        nvgRect(vg, 0, 0, windowWidth, windowHeight);
        nvgFillColor(vg, backgroundFillColorNvg);
        nvgFill(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setColor(toSkColor(backgroundFillColor));
        paint->setStroke(false);
        canvas->drawRect(SkRect{0 * s, 0 * s, windowWidth * s, windowHeight * s}, *paint);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_rectangle(context, 0 * s, 0 * s, windowWidth * s, windowHeight * s);
        vkvg_set_opacity(context, 1.0f);
        vkvg_set_source_color(context, backgroundFillColor.getColorRGBA());
    }
#endif

    for (int i = 0; i < CUSTOM_ARRAYSIZE(SHAPEFILE_FILENAMES); i++) {
        const std::string mapsDirectory = sgl::AppSettings::get()->getDataDirectory() + "Maps/";
        std::string shapefilePath = mapsDirectory + SHAPEFILE_FILENAMES[i];
        uint8_t* buffer = nullptr;
        size_t length = 0;
        bool loaded = sgl::loadFileFromSource(shapefilePath, buffer, length, true);
        if (!loaded) {
            continue;
        }
        currentFillColor = SHAPEFILE_FILL_COLORS[i];
        currentStrokeColor = SHAPEFILE_STROKE_COLORS[i];
        rasterizeShapefile(buffer, length);
        delete[] buffer;
    }

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
    }
#endif
}

void ShapefileRasterizer::rasterizeShapefile(uint8_t* buffer, size_t length) {
    uint8_t* endPtr = buffer + length;
    auto header = *reinterpret_cast<ShapefileHeader*>(buffer);
    swapEndianness(reinterpret_cast<uint8_t*>(&header), 7 * 4, 4);
    if (header.fileCode != 0x0000270au) {
        sgl::Logfile::get()->writeError("Error in ShapefileRasterizer::rasterizeShapefile: File code mismatch.");
        return;
    }
    if (length != size_t(header.fileLength) * 2) {
        sgl::Logfile::get()->writeError("Error in ShapefileRasterizer::rasterizeShapefile: File length mismatch.");
    }

    uint8_t* filePtr = buffer + sizeof(ShapefileHeader);
    while (filePtr < endPtr) {
        auto recordHeader = *reinterpret_cast<RecordHeader*>(filePtr);
        swapEndianness(reinterpret_cast<uint8_t*>(&recordHeader), 2 * 4, 4);
        auto* data = filePtr + 12;
        filePtr += recordHeader.recordLength * 2 + 8;

        if (recordHeader.shapeType == ShapeType::POINT) {
            auto* pointData = reinterpret_cast<ShapefilePoint*>(data);
            if (isPointVisible(pointData->x, pointData->y)) {
                rasterizePoint(pointData->x, pointData->y);
            }
        } else if (recordHeader.shapeType == ShapeType::MULTI_POINT) {
            auto* aabb = reinterpret_cast<ShapefileAabb*>(data);
            uint32_t numPoints = *reinterpret_cast<uint32_t*>(data + 32);
            auto* pointData = reinterpret_cast<ShapefilePoint*>(data + 36);
            if (isAabbVisible(aabb)) {
                rasterizeMultiPoint(numPoints, pointData);
            }
        } else if (recordHeader.shapeType == ShapeType::POLYLINE) {
            auto* aabb = reinterpret_cast<ShapefileAabb*>(data);
            uint32_t numParts = *reinterpret_cast<uint32_t*>(data + 32);
            uint32_t numPoints = *reinterpret_cast<uint32_t*>(data + 36);
            auto* partsData = reinterpret_cast<uint32_t*>(data + 40);
            auto* pointData = reinterpret_cast<ShapefilePoint*>(data + 40 + 4 * numParts);
            if (isAabbVisible(aabb)) {
                rasterizePolyline(numParts, numPoints, partsData, pointData);
            }
        } else if (recordHeader.shapeType == ShapeType::POLYGON) {
            auto* aabb = reinterpret_cast<ShapefileAabb*>(data);
            uint32_t numParts = *reinterpret_cast<uint32_t*>(data + 32);
            uint32_t numPoints = *reinterpret_cast<uint32_t*>(data + 36);
            auto* partsData = reinterpret_cast<uint32_t*>(data + 40);
            auto* pointData = reinterpret_cast<ShapefilePoint*>(data + 40 + 4 * numParts);
            if (isAabbVisible(aabb)) {
                rasterizePolygon(numParts, numPoints, partsData, pointData);
            }
            // Polygon data: Clockwise -> solid, counterclockwise -> hole.
            // First point is also stored as last point (i.e., explicit closing).
        }
    }
}

#define CONV_X(x) float(((x) - minLon) / (maxLon - minLon) * double(windowWidth))
#define CONV_Y(y) float((1.0 - ((y) - minLat) / (maxLat - minLat)) * double(windowHeight))

float ShapefileRasterizer::getStrokeWidth() {
    float windowScaleFactor = std::max(windowWidth, windowHeight) / 1024.0f;
    float strokeWidth = float(10.0 / std::max(maxLon - minLon, maxLat - minLat));
    strokeWidth = std::clamp(strokeWidth, 0.5f, 8.0f);
    return strokeWidth * windowScaleFactor;
}

void ShapefileRasterizer::rasterizePoint(double x, double y) {
    // Not implemented so far, as world map doesn't have point data at the moment.
}

void ShapefileRasterizer::rasterizeMultiPoint(uint32_t numPoints, ShapefilePoint* pointData) {
    // Not implemented so far, as world map doesn't have point data at the moment.
}

void ShapefileRasterizer::rasterizePolyline(
        uint32_t numParts, uint32_t numPoints, uint32_t* partsData, ShapefilePoint* pointData) {
    if (vg) {
        nvgBeginPath(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        path = new SkPath;
    }
#endif

    for (uint32_t partIdx = 0; partIdx < numParts; partIdx++) {
        uint32_t firstPoint = partsData[partIdx];
        uint32_t lastPoint = partIdx == numParts - 1 ? numPoints : partsData[partIdx + 1];

        for (uint32_t pointIdx = firstPoint; pointIdx < lastPoint; pointIdx++) {
            ShapefilePoint& point = pointData[pointIdx];
            auto px = CONV_X(point.x);
            auto py = CONV_Y(point.y);

            if (vg) {
                if (pointIdx == firstPoint) {
                    nvgMoveTo(vg, px, py);
                } else {
                    nvgLineTo(vg, px, py);
                }
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                if (pointIdx == firstPoint) {
                    path->moveTo(px * s, py * s);
                } else {
                    path->lineTo(px * s, py * s);
                }
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                if (pointIdx == firstPoint) {
                    vkvg_move_to(context, px * s, py * s);
                } else {
                    vkvg_line_to(context, px * s, py * s);
                }
            }
#endif
        }
    }

    float strokeWidth = getStrokeWidth();

    if (vg) {
        NVGcolor fillColorNvg = nvgRGBA(
                currentStrokeColor.getR(), currentStrokeColor.getG(), currentStrokeColor.getB(), 255);
        nvgStrokeWidth(vg, strokeWidth);
        nvgStrokeColor(vg, fillColorNvg);
        nvgStroke(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setStroke(true);
        paint->setStrokeWidth(strokeWidth * s);
        paint->setColor(toSkColor(currentStrokeColor));
        canvas->drawPath(*path, *paint);
        delete path;
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_line_width(context, strokeWidth * s);
        vkvg_set_source_color(context, currentStrokeColor.getColorRGBA());
        vkvg_stroke(context);
    }
#endif
}

void ShapefileRasterizer::rasterizePolygon(
        uint32_t numParts, uint32_t numPoints, uint32_t* partsData, ShapefilePoint* pointData) {
    if (vg) {
        nvgBeginPath(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        path = new SkPath;
    }
#endif

    for (uint32_t partIdx = 0; partIdx < numParts; partIdx++) {
        uint32_t firstPoint = partsData[partIdx];
        uint32_t lastPoint = partIdx == numParts - 1 ? numPoints : partsData[partIdx + 1];

        for (uint32_t pointIdx = firstPoint; pointIdx < lastPoint; pointIdx++) {
            ShapefilePoint& point = pointData[pointIdx];
            auto px = CONV_X(point.x);
            auto py = CONV_Y(point.y);

            if (vg) {
                if (pointIdx == firstPoint) {
                    nvgMoveTo(vg, px, py);
                } else {
                    nvgLineTo(vg, px, py);
                }
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                if (pointIdx == firstPoint) {
                    path->moveTo(px * s, py * s);
                } else {
                    path->lineTo(px * s, py * s);
                }
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                if (pointIdx == firstPoint) {
                    vkvg_move_to(context, px * s, py * s);
                } else {
                    vkvg_line_to(context, px * s, py * s);
                }
            }
#endif
        }

        if (vg) {
            nvgClosePath(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            path->close();
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            vkvg_close_path(context);
        }
#endif
    }

    float strokeWidth = getStrokeWidth();

    if (vg) {
        NVGcolor fillColorNvg = nvgRGBA(currentFillColor.getR(), currentFillColor.getG(), currentFillColor.getB(), 255);
        nvgFillColor(vg, fillColorNvg);
        nvgFill(vg);
        if (currentStrokeColor.getA() != 0) {
            NVGcolor strokeColorNvg = nvgRGBA(
                    currentStrokeColor.getR(), currentStrokeColor.getG(), currentStrokeColor.getB(), 255);
            nvgStrokeWidth(vg, strokeWidth);
            nvgStrokeColor(vg, strokeColorNvg);
            nvgStroke(vg);
        }
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setStroke(false);
        paint->setColor(toSkColor(currentFillColor));
        canvas->drawPath(*path, *paint);
        if (currentStrokeColor.getA() != 0) {
            paint->setStroke(true);
            paint->setStrokeWidth(strokeWidth * s);
            paint->setColor(toSkColor(currentStrokeColor));
            canvas->drawPath(*path, *paint);
        }
        delete path;
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_source_color(context, currentFillColor.getColorRGBA());
        if (true) {
            vkvg_fill_preserve(context);
            vkvg_set_line_width(context, strokeWidth * s);
            vkvg_set_source_color(context, currentStrokeColor.getColorRGBA());
            vkvg_stroke(context);
        } else {
            vkvg_fill(context);
        }
    }
#endif
}

bool ShapefileRasterizer::isPointVisible(double x, double y) {
    return x >= minLon && x <= maxLon && y >= minLat && y <= maxLat;
}

bool ShapefileRasterizer::isAabbVisible(const ShapefileAabb* aabb) {
    if (maxLon < aabb->minX || minLon > aabb->maxX || maxLat < aabb->minY || minLat > aabb->maxY) {
        return false;
    }
    return true;
}

void ShapefileRasterizer::getNanoVGContext() {
    vg = static_cast<sgl::VectorBackendNanoVG*>(vectorBackend)->getContext();
#ifdef SUPPORT_SKIA
    canvas = nullptr;
#endif
#ifdef SUPPORT_VKVG
    context = nullptr;
#endif
}

void ShapefileRasterizer::renderBaseNanoVG() {
    getNanoVGContext();

    rasterizeShapefiles();
}


#ifdef SUPPORT_SKIA
void ShapefileRasterizer::getSkiaCanvas() {
    vg = nullptr;
    canvas = static_cast<VectorBackendSkia*>(vectorBackend)->getCanvas();
#ifdef SUPPORT_VKVG
    context = nullptr;
#endif
}

void ShapefileRasterizer::renderBaseSkia() {
    getSkiaCanvas();
    s = scaleFactor * float(supersamplingFactor);

    rasterizeShapefiles();
}
#endif


#ifdef SUPPORT_VKVG
void ShapefileRasterizer::getVkvgContext() {
    vg = nullptr;
#ifdef SUPPORT_SKIA
    canvas = nullptr;
#endif
    context = static_cast<VectorBackendVkvg*>(vectorBackend)->getContext();
}

void ShapefileRasterizer::renderBaseVkvg() {
    getVkvgContext();
    s = scaleFactor * float(supersamplingFactor);

    rasterizeShapefiles();
}
#endif
