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

#ifndef CORRERENDER_SHAPEFILERASTERIZER_HPP
#define CORRERENDER_SHAPEFILERASTERIZER_HPP

#include <cstdint>

#include <Graphics/Vector/VectorWidget.hpp>

namespace sgl { namespace vk {
class Renderer;
}}

struct NVGcontext;
typedef struct NVGcontext NVGcontext;
struct NVGcolor;
class SkCanvas;
class SkPaint;
class SkPath;
#ifdef SUPPORT_VKVG
struct _vkvg_context_t;
typedef struct _vkvg_context_t* VkvgContext;
#endif

struct ShapefilePoint;
struct ShapefileAabb;

/**
 * Shapefile rasterizer for world maps. For more details, please refer to:
 * - https://en.wikipedia.org/wiki/Shapefile
 * - https://www.esri.com/content/dam/esrisites/sitecore-archive/Files/Pdfs/library/whitepapers/pdfs/shapefile.pdf
 * Public domain shapefiles are automatically downloaded from naturalearthdata.com.
 *
 * Currently, three rasterizers are supported (with individual pros and cons):
 *
 * (1) NanoVG using OpenGL.
 * - Pros: Works on all builds, fast.
 * - Cons: Needs Vulkan-OpenGL interop, holes are incorrect (https://github.com/memononen/nanovg/issues/598).
 * TODO: Can this be fixed with manually specifying nvgPathWinding?
 * (2) Skia using Vulkan.
 * - Pros: Holes are correct, fast.
 * - Cons: Fails to free some resources, which causes an error message when quitting the application,
 *         currently is not built with Windows MSYS2 build.
 * (3) VKVG using Vulkan:
 * - Pros: Works on all builds.
 * - Cons: Incorrect and broken output, slow.
 */
class ShapefileRasterizer : public sgl::VectorWidget {
public:
    explicit ShapefileRasterizer(sgl::vk::Renderer* renderer);
    void checkTempFiles();
    void rasterize(
            float _minLon, float _maxLon, float _minLat, float _maxLat,
            uint32_t regionImageWidth, uint32_t regionImageHeight, uint8_t* worldMapData);

private:
    void checkCached(const std::string& mapsDirectory, const std::string& archiveUrl, const std::string& filename);
    void rasterizeShapefiles();
    void rasterizeShapefile(uint8_t* buffer, size_t length);
    void rasterizePoint(double x, double y);
    void rasterizeMultiPoint(uint32_t numPoints, ShapefilePoint* pointData);
    void rasterizePolyline(
            uint32_t numParts, uint32_t numPoints, uint32_t* partsData, ShapefilePoint* pointData);
    void rasterizePolygon(
            uint32_t numParts, uint32_t numPoints, uint32_t* partsData, ShapefilePoint* pointData);
    bool isPointVisible(double x, double y);
    bool isAabbVisible(const ShapefileAabb* aabb);
    float getStrokeWidth();
    double minLon = 0, maxLon = 0, minLat = 0, maxLat = 0;
    sgl::Color currentFillColor = sgl::Color(200, 110, 55, 255);
    sgl::Color currentStrokeColor = sgl::Color(200, 110, 55, 255);

protected:
    sgl::vk::Renderer* renderer;

    // Color scheme.
    sgl::Color backgroundFillColor = sgl::Color(127, 183, 218, 255);

    // NanoVG backend.
    virtual void renderBaseNanoVG();
    void getNanoVGContext();
    NVGcontext* vg = nullptr;

    // Scale factor used for rendering.
    float s = 1.0f;

    // Skia backend.
#ifdef SUPPORT_SKIA
    virtual void renderBaseSkia();
    void getSkiaCanvas();
    SkCanvas* canvas = nullptr;
    SkPaint* paint = nullptr;
    SkPath* path = nullptr;
#endif

    // VKVG backend.
#ifdef SUPPORT_VKVG
    virtual void renderBaseVkvg();
    void getVkvgContext();
    VkvgContext context = nullptr;
#endif
};

#endif //CORRERENDER_SHAPEFILERASTERIZER_HPP
