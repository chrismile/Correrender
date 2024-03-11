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

#include <Math/Math.hpp>
#include <Input/Mouse.hpp>
#include <Graphics/Vector/nanovg/nanovg.h>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitRenderPass.hpp>

#ifdef SUPPORT_OPENGL
#include <Graphics/Vulkan/Utils/Interop.hpp>
#include <Graphics/OpenGL/Texture.hpp>
#endif

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
#include "../VectorBackendSkia.hpp"
#endif
#ifdef SUPPORT_VKVG
#include "../VectorBackendVkvg.hpp"
#endif

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "TimeSeriesCorrelationChart.hpp"

class TimeSeriesRasterPass : public sgl::vk::BlitRenderPass {
public:
    explicit TimeSeriesRasterPass(sgl::vk::Renderer* renderer);
    void setCorrelationDataBuffer(const sgl::vk::BufferPtr& _correlationDataBuffer);
    void setSettings(int samples, int numWindows, float minAttributeValue, float maxAttributeValue);
    void setColorPoints(const std::vector<glm::vec3>& colorPoints);

protected:
    void loadShader() override;
    void createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) override;
    void _render() override;

private:
    struct UniformData {
        int samples;
        int numWindows;
        float minAttributeValue;
        float maxAttributeValue;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr correlationDataBuffer;
    sgl::vk::TexturePtr transferFunctionTexture;
};

TimeSeriesRasterPass::TimeSeriesRasterPass(sgl::vk::Renderer* renderer)
        : sgl::vk::BlitRenderPass(renderer, {"TimeSeriesBlit.Vertex", "TimeSeriesBlit.Fragment"}) {
    this->setAttachmentLoadOp(VK_ATTACHMENT_LOAD_OP_CLEAR);
    this->setOutputImageFinalLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    this->setBlendMode(sgl::vk::BlendMode::OVERWRITE);
}

void TimeSeriesRasterPass::setSettings(int samples, int numWindows, float minAttributeValue, float maxAttributeValue) {
    uniformData.samples = samples;
    uniformData.numWindows = numWindows;
    uniformData.minAttributeValue = minAttributeValue;
    uniformData.maxAttributeValue = maxAttributeValue;
}

void TimeSeriesRasterPass::setColorPoints(const std::vector<glm::vec3>& colorPoints) {
    const int TFRES = 256;
    const auto numColorPoints = int(colorPoints.size());
    std::vector<sgl::Color16> colorPointsRgba16;
    colorPointsRgba16.reserve(TFRES);
    for (int i = 0; i < TFRES; i++) {
        float t = static_cast<float>(i) / float(TFRES - 1) * float(numColorPoints - 1);
        auto t0 = sgl::clamp(int(std::floor(t)), 0, numColorPoints - 1);
        auto t1 = sgl::clamp(t0 + 1, 0, numColorPoints - 1);
        auto factor = t - float(t0);
        glm::vec3 color0 = colorPoints.at(t0);
        glm::vec3 color1 = colorPoints.at(t1);
        auto color = glm::mix(color0, color1, factor);
        colorPointsRgba16.push_back(sgl::color16FromVec3(color));
    }
    renderer->getDevice()->waitGraphicsQueueIdle();
    sgl::vk::ImageSettings imageSettings{};
    imageSettings.imageType = VK_IMAGE_TYPE_1D;
    imageSettings.format = VK_FORMAT_R16G16B16A16_UNORM;
    imageSettings.width = uint32_t(colorPointsRgba16.size());
    transferFunctionTexture = std::make_shared<sgl::vk::Texture>(renderer->getDevice(), imageSettings);
    transferFunctionTexture->getImage()->uploadData(colorPointsRgba16.size() * 8, colorPointsRgba16.data());
    if (rasterData) {
        rasterData->setStaticTexture(transferFunctionTexture, "transferFunctionTexture");
    }
}

void TimeSeriesRasterPass::setCorrelationDataBuffer(const sgl::vk::BufferPtr& _correlationDataBuffer) {
    if (correlationDataBuffer != _correlationDataBuffer) {
        correlationDataBuffer = _correlationDataBuffer;
        if (rasterData) {
            rasterData->setStaticBuffer(correlationDataBuffer, "CorrelationDataBuffer");
        }
    }
}

void TimeSeriesRasterPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    shaderStages = sgl::vk::ShaderManager->getShaderStages(shaderIds, preprocessorDefines);
}

void TimeSeriesRasterPass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    rasterData = std::make_shared<sgl::vk::RasterData>(renderer, graphicsPipeline);
    rasterData->setIndexBuffer(indexBuffer);
    rasterData->setVertexBuffer(vertexBuffer, 0);
    rasterData->setStaticBuffer(correlationDataBuffer, "CorrelationDataBuffer");
    rasterData->setStaticTexture(transferFunctionTexture, "transferFunctionTexture");
}

void TimeSeriesRasterPass::_render() {
    renderer->pushConstants(rasterData->getGraphicsPipeline(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, uniformData);
    sgl::vk::BlitRenderPass::_render();
}



TimeSeriesCorrelationChart::TimeSeriesCorrelationChart() {
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
}

TimeSeriesCorrelationChart::~TimeSeriesCorrelationChart() = default;

void TimeSeriesCorrelationChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    windowWidth = (200 + borderSizeX) * 2.0f;
    windowHeight = (200 + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void TimeSeriesCorrelationChart::onUpdatedWindowSize() {
    borderSizeX = borderSizeY = 10.0f;
    ox = borderSizeX;
    oy = borderSizeY;
    dw = windowWidth - borderSizeX * 7 - colorLegendWidth;
    dh = windowHeight - borderSizeY * 2;
    if (useScrollBar) {
        dw -= scrollBarWidth;
        recomputeScrollThumbHeight();
    }
    dw = std::max(dw, 1.0f);
    dh = std::max(dh, 1.0f);
    computeColorLegendHeight();
}

void TimeSeriesCorrelationChart::updateSizeByParent() {
    auto [parentWidth, parentHeight] = getBlitTargetSize();
    auto ssf = float(blitTargetSupersamplingFactor);
    windowOffsetX = 0;
    windowOffsetY = 0;
    windowWidth = float(parentWidth) / (scaleFactor * float(ssf));
    windowHeight = float(parentHeight) / (scaleFactor * float(ssf));
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void TimeSeriesCorrelationChart::setAlignWithParentWindow(bool _align) {
    alignWithParentWindow = _align;
    renderBackgroundStroke = !_align;
    isWindowFixed = alignWithParentWindow;
    if (alignWithParentWindow) {
        updateSizeByParent();
    }
}

void TimeSeriesCorrelationChart::setColorMap(DiagramColorMap _colorMap) {
    if (colorMap != _colorMap) {
        colorMap = _colorMap;
        needsReRender = true;
    }
    colorPoints = getColorPoints(colorMap);
    colorMapChanged = true;
}

void TimeSeriesCorrelationChart::setDiagramSelectionCallback(std::function<void(int series, int time)> callback) {
    diagramSelectionCallback = callback;
}

glm::vec4 TimeSeriesCorrelationChart::evalColorMapVec4(float t) {
    if (std::isnan(t)) {
        if (colorMap == DiagramColorMap::VIRIDIS) {
            return glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
        } else {
            return glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
        }
    }
    t = glm::clamp(t, 0.0f, 1.0f);
    auto N = int(colorPoints.size());
    float arrayPosFlt = t * float(N - 1);
    int lastIdx = std::min(int(arrayPosFlt), N - 1);
    int nextIdx = std::min(lastIdx + 1, N - 1);
    float f1 = arrayPosFlt - float(lastIdx);
    const glm::vec3& c0 = colorPoints.at(lastIdx);
    const glm::vec3& c1 = colorPoints.at(nextIdx);
    return glm::vec4(glm::mix(c0, c1, f1), 1.0f);
}

sgl::Color TimeSeriesCorrelationChart::evalColorMap(float t) {
    return sgl::colorFromVec4(evalColorMapVec4(t));
}

void TimeSeriesCorrelationChart::update(float dt) {
    glm::vec2 mousePosition(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePosition -= glm::vec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();

    float startX = ox;
    float startY = oy;
    float matrixWidth = dw;
    float matrixHeight = dh;
    glm::vec2 pctMousePos = (mousePosition - glm::vec2(startX, startY)) / glm::vec2(matrixWidth, matrixHeight);

    // Check whether the mouse is hovering a button. In this case, window move and resize events should be disabled.
    //bool isMatrixHovered =
    //        pctMousePos.x >= 0.0f && pctMousePos.y >= 0.0f
    //        && pctMousePos.x < 1.0f && pctMousePos.y < 1.0f;
    //isMouseGrabbedByParent = isMouseGrabbedByParent || isMatrixHovered;

    DiagramBase::update(dt);

    glm::vec2 gridMousePos = pctMousePos * glm::vec2(numWindows, samples);
    auto gridPosition = glm::ivec2(gridMousePos);
    int time = gridPosition.x;
    int series = gridPosition.y;

    if (pctMousePos.x >= 0.0f && pctMousePos.y >= 0.0f && pctMousePos.x < 1.0f && pctMousePos.y < 1.0f
            && sgl::Mouse->buttonReleased(1) && !isMouseGrabbedByParent) {
        diagramSelectionCallback(series, time);
    }
}

void TimeSeriesCorrelationChart::setCorrelationDataBuffer(
        int _samples, int _numWindows, const sgl::vk::BufferPtr& _correlationDataBuffer) {
    samples = _samples;
    numWindows = _numWindows;
    if (correlationDataBuffer != _correlationDataBuffer) {
        correlationDataBuffer = _correlationDataBuffer;
        if (!timeSeriesRasterPass) {
            timeSeriesRasterPass = std::make_shared<TimeSeriesRasterPass>(rendererVk);
        }
        timeSeriesRasterPass->setCorrelationDataBuffer(correlationDataBuffer);
    }
    dataDirty = true;
}

void TimeSeriesCorrelationChart::onCorrelationDataRecalculated(
        CorrelationMeasureType _correlationMeasureType,
        const std::pair<float, float>& _minMaxCorrelationValue, bool _isNetworkData) {
    correlationMeasureType = _correlationMeasureType;
    minMaxCorrelationValue = _minMaxCorrelationValue;
    isNetworkData = _isNetworkData;
}

void TimeSeriesCorrelationChart::onWindowSizeChanged() {
    DiagramBase::onWindowSizeChanged();

    auto width = int(std::ceil(float(dw) * scaleFactor)) * supersamplingFactor;
    auto height = int(std::ceil(float(dh) * scaleFactor)) * supersamplingFactor;

    auto* device = sgl::AppSettings::get()->getPrimaryDevice();
    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = uint32_t(width);
    imageSettings.height = uint32_t(height);
    imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageSettings.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageSettings.exportMemory = true;
    sgl::vk::ImageSamplerSettings samplerSettings;
    correlationTextureVk = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
#ifdef SUPPORT_OPENGL
    correlationTextureGl = sgl::TexturePtr(new sgl::TextureGLExternalMemoryVk(correlationTextureVk));
#endif
    correlationImageViewVk = correlationTextureVk->getImageView();
    if (!timeSeriesRasterPass) {
        timeSeriesRasterPass = std::make_shared<TimeSeriesRasterPass>(rendererVk);
    }
    timeSeriesRasterPass->setOutputImage(correlationImageViewVk);
    timeSeriesRasterPass->recreateSwapchain(uint32_t(width), uint32_t(height));
    imageHandleDirty = true;
}

void TimeSeriesCorrelationChart::renderPrepare() {
    if (!correlationDataBuffer) {
        return;
    }

    if (colorMapChanged) {
        timeSeriesRasterPass->setColorPoints(colorPoints);
        colorMapChanged = false;
    }
    timeSeriesRasterPass->setSettings(
            samples, numWindows, minMaxCorrelationValue.first, minMaxCorrelationValue.second);
    timeSeriesRasterPass->render();

#ifdef SUPPORT_OPENGL
    if (vectorBackend) {
        vectorBackend->addImageGl(
                correlationTextureGl,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
#endif
}

void TimeSeriesCorrelationChart::onBackendDestroyed() {
    DiagramBase::onBackendDestroyed();
    if (vg && imageHandleNvg >= 0) {
        nvgDeleteImage(vg, imageHandleNvg);
        imageHandleNvg = -1;
        vg = nullptr;
    }
}

void TimeSeriesCorrelationChart::updateData() {
    ;
}

void TimeSeriesCorrelationChart::renderTimeSeries() {
    if (!correlationDataBuffer) {
        return;
    }

    sgl::Color textColor = isDarkMode ? textColorDark : textColorBright;
    NVGcolor textColorNvg;
    if (vg) {
        textColorNvg = nvgRGBA(textColor.getR(), textColor.getG(), textColor.getB(), 255);
    }

#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr, *gradientPaint = nullptr;
    SkFont* font = nullptr;
    SkFontMetrics metrics{};
    if (canvas) {
        paint = new SkPaint;
        gradientPaint = new SkPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(paint);
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(gradientPaint);
        font = new SkFont(typeface, textSizeLegend * s);
        font->getMetrics(&metrics);
    }
#endif

    auto x = ox;
    auto y = oy;
    auto w = dw;
    auto h = dh;

    if (vg) {
        if (useScrollBar) {
            nvgSave(vg);
            nvgScissor(
                    vg, borderWidth, borderWidth,
                    windowWidth - 2.0f * borderWidth, windowHeight - 2.0f * borderWidth);
            nvgTranslate(vg, 0.0f, -scrollTranslationY);
        }

#ifdef SUPPORT_OPENGL
        if (imageHandleDirty) {
            int imageHandleGl = static_cast<sgl::TextureGL*>(correlationTextureGl.get())->getTexture();
            imageHandleNvg = nvgImportImage(
                    vg, correlationTextureGl->getW(), correlationTextureGl->getH(), NVG_IMAGE_NEAREST, &imageHandleGl);
            imageHandleDirty = false;
        }
#endif
        NVGpaint nvgPaint = nvgImagePattern(vg, x, y, w, h, 0.0f, imageHandleNvg, 1.0f);
        nvgBeginPath(vg);
        nvgRect(vg, x, y, w, h);
        nvgFillPaint(vg, nvgPaint);
        nvgFill(vg);

        if (useScrollBar) {
            nvgRestore(vg);
        }

        if (useScrollBar) {
            drawScrollBar();
        }
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        sgl::Color color(255, 255, 0, 255);
        paint->setStroke(false);
        paint->setColor(toSkColor(color));
        canvas->drawRect(SkRect{x * s, y * s, (x + w) * s, (y + h) * s}, *paint);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        sgl::Color color(255, 255, 0, 255);
        vkvg_rectangle(context, x * s, y * s, w * s, h * s);
        vkvg_set_source_color(context, color.getColorRGBA());
        vkvg_fill(context);
    }
#endif

    // Draw color legend.
    if (shallDrawColorLegend) {
        drawColorLegends();
    }

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
        delete gradientPaint;
        delete font;
    }
#endif
}

void TimeSeriesCorrelationChart::drawScrollBar() {
    NVGcolor scrollBarColor = nvgRGBA(120, 120, 120, 120);
    nvgBeginPath(vg);
    nvgRect(vg, windowWidth - scrollBarWidth, borderWidth, 1.0f, windowHeight - 2.0f * borderWidth);
    nvgFillColor(vg, scrollBarColor);
    nvgFill(vg);

    NVGcolor scrollThumbColor;
    if (scrollThumbHover) {
        scrollThumbColor = nvgRGBA(90, 90, 90, 120);
    } else {
        scrollThumbColor = nvgRGBA(160, 160, 160, 120);
    }
    nvgBeginPath(vg);
    nvgRoundedRectVarying(
            vg, windowWidth - scrollBarWidth, scrollThumbPosition, scrollBarWidth - borderWidth, scrollThumbHeight,
            0.0f, borderRoundingRadius, borderRoundingRadius, 0.0f);
    nvgFillColor(vg, scrollThumbColor);
    nvgFill(vg);

    NVGcolor scrollBarGripColor = nvgRGBA(60, 60, 60, 120);
    float scrollBarMiddle = scrollThumbPosition + scrollThumbHeight / 2.0f;
    float scrollBarLeft = windowWidth - scrollBarWidth;
    float scrollBarWidthReal = scrollBarWidth - borderWidth;
    float gripLeft = scrollBarLeft + 0.2f * scrollBarWidthReal;
    float gripWidth = scrollBarWidthReal * 0.6f;
    nvgBeginPath(vg);
    nvgRect(vg, gripLeft, scrollBarMiddle - 3.0f, gripWidth, 1.0f);
    nvgRect(vg, gripLeft, scrollBarMiddle + 0.0f, gripWidth, 1.0f);
    nvgRect(vg, gripLeft, scrollBarMiddle + 3.0f, gripWidth, 1.0f);
    nvgFillColor(vg, scrollBarGripColor);
    nvgFill(vg);
}

void TimeSeriesCorrelationChart::recomputeScrollThumbHeight() {
    scrollThumbHeight = 1.0f / zoomFactorVertical;
}

void TimeSeriesCorrelationChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderTimeSeries();
}

#ifdef SUPPORT_SKIA
void TimeSeriesCorrelationChart::renderBaseSkia() {
    DiagramBase::renderBaseSkia();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderTimeSeries();
}
#endif

#ifdef SUPPORT_VKVG
void TimeSeriesCorrelationChart::renderBaseVkvg() {
    DiagramBase::renderBaseVkvg();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderTimeSeries();
}
#endif

void TimeSeriesCorrelationChart::drawColorLegends() {
#ifdef __APPLE__
    auto minMaxMi = minMaxCorrelationValue;
    float minMi = minMaxMi.first;
    float maxMi = minMaxMi.second;
#else
    auto [minMi, maxMi] = minMaxCorrelationValue;
#endif
    std::string variableName;
    if (isNetworkData) {
        variableName =
                correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
                ? "MI" : "Pearson";
    } else {
        variableName = CORRELATION_MEASURE_TYPE_SHORT_NAMES[int(correlationMeasureType)];
    }
    std::function<glm::vec4(float)> colorMap;
    std::function<std::string(float)> labelMap;
    colorMap = [this](float t) {
        return evalColorMapVec4(t);
    };
    labelMap = [minMi, maxMi](float t) {
        return getNiceNumberString((1.0f - t) * minMi + t * maxMi, 2);
    };

    int numLabels = 5;
    int numTicks = 5;
    if (colorLegendHeight < textSizeLegend * 0.625f) {
        numLabels = 0;
        numTicks = 0;
    } else if (colorLegendHeight < textSizeLegend * 2.0f) {
        numLabels = 2;
        numTicks = 2;
    } else if (colorLegendHeight < textSizeLegend * 4.0f) {
        numLabels = 3;
        numTicks = 3;
    }

    //float posX =
    //        windowWidth - borderSizeX
    //        - float(1) * (colorLegendWidth + textWidthMax)
    //        - float(0) * colorLegendSpacing;
    float posX = windowWidth - 4.0f * borderSizeX - colorLegendWidth - (useScrollBar ? scrollBarWidth : 0);

    float posY = windowHeight - borderSizeY - colorLegendHeight;
    drawColorLegend(
            posX, posY, colorLegendWidth, colorLegendHeight, numLabels, numTicks, labelMap, colorMap, variableName);
}

void TimeSeriesCorrelationChart::drawColorLegend(
        float x, float y, float w, float h, int numLabels, int numTicks,
        const std::function<std::string(float)>& labelMap, const std::function<glm::vec4(float)>& colorMap,
        const std::string& textTop) {
    sgl::Color textColor = isDarkMode ? textColorDark : textColorBright;
    NVGcolor textColorNvg;
    if (vg) {
        textColorNvg = nvgRGBA(textColor.getR(), textColor.getG(), textColor.getB(), 255);
    }
#ifdef SUPPORT_SKIA
    SkPaint* paint = nullptr, *gradientPaint = nullptr;
    SkFont* font = nullptr;
    SkFontMetrics metrics{};
    if (canvas) {
        paint = new SkPaint;
        gradientPaint = new SkPaint;
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(paint);
        static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(gradientPaint);
        font = new SkFont(typeface, textSizeLegend * s);
        font->getMetrics(&metrics);
    }
#endif

    const int numPoints = 17; // 9
    const int numSubdivisions = numPoints - 1;

    // Draw color bar.
    for (int i = 0; i < numSubdivisions; i++) {
        float t0 = 1.0f - float(i) / float(numSubdivisions);
        float t1 = 1.0f - float(i + 1) / float(numSubdivisions);

        glm::vec4 fillColorVec0 = colorMap(t0);
        glm::vec4 fillColorVec1 = colorMap(t1);
        fillColorVec0.w = 1.0f;
        fillColorVec1.w = 1.0f;
        if (vg) {
            auto fillColor0 = nvgRGBAf(fillColorVec0.x, fillColorVec0.y, fillColorVec0.z, 1.0f);
            auto fillColor1 = nvgRGBAf(fillColorVec1.x, fillColorVec1.y, fillColorVec1.z, 1.0f);
            nvgBeginPath(vg);
            nvgRect(vg, x, y + h * float(i) / float(numSubdivisions), w, h / float(numSubdivisions) + 1e-1f);
            NVGpaint paint = nvgLinearGradient(
                    vg, x, y + h * float(i) / float(numSubdivisions),
                    x, y + h * float(i+1) / float(numSubdivisions),
                    fillColor0, fillColor1);
            nvgFillPaint(vg, paint);
            nvgFill(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            auto fillColor0 = toSkColor(sgl::colorFromVec4(fillColorVec0));
            auto fillColor1 = toSkColor(sgl::colorFromVec4(fillColorVec1));
            SkPoint linearPoints[2] = {
                    { x * s, (y + h * float(i) / float(numSubdivisions)) * s },
                    { x * s, (y + h * float(i + 1) / float(numSubdivisions)) * s }
            };
            SkColor linearColors[2] = { fillColor0, fillColor1 };
            gradientPaint->setShader(SkGradientShader::MakeLinear(
                    linearPoints, linearColors, nullptr, 2, SkTileMode::kClamp));
            canvas->drawRect(
                    SkRect{
                            x * s, (y + h * float(i) / float(numSubdivisions)) * s,
                            (x + w) * s, (y + h * float(i + 1) / float(numSubdivisions) + 1e-1f) * s}, *gradientPaint);
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            auto pattern = vkvg_pattern_create_linear(
                    x * s, (y + h * float(i) / float(numSubdivisions)) * s,
                    x * s, (y + h * float(i + 1) / float(numSubdivisions)) * s);
            vkvg_pattern_add_color_stop(pattern, 0.0f, fillColorVec0.x, fillColorVec0.y, fillColorVec0.z, 1.0f);
            vkvg_pattern_add_color_stop(pattern, 1.0f, fillColorVec1.x, fillColorVec1.y, fillColorVec1.z, 1.0f);
            vkvg_set_source(context, pattern);
            vkvg_pattern_destroy(pattern);
            vkvg_rectangle(
                    context, x * s, (y + h * float(i) / float(numSubdivisions)) * s,
                    w * s, (h / float(numSubdivisions) + 1e-1f) * s);
            vkvg_fill(context);
        }
#endif
    }

    // Draw ticks.
    const float tickWidth = 4.0f;
    const float tickHeight = 1.0f;
    if (vg) {
        nvgBeginPath(vg);
        for (int tickIdx = 0; tickIdx < numTicks; tickIdx++) {
            float centerY = y + float(tickIdx) / float(numTicks - 1) * h;
            nvgRect(vg, x + w, centerY - tickHeight / 2.0f, tickWidth, tickHeight);
        }
        nvgFillColor(vg, textColorNvg);
        nvgFill(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setColor(toSkColor(textColor));
        paint->setStroke(false);
        for (int tickIdx = 0; tickIdx < numTicks; tickIdx++) {
            float centerY = y + float(tickIdx) / float(numTicks - 1) * h;
            canvas->drawRect(
                    SkRect{
                            (x + w) * s, (centerY - tickHeight / 2.0f) * s,
                            (x + w + tickWidth) * s, (centerY + tickHeight / 2.0f) * s}, *paint);
        }
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_source_color(context, textColor.getColorRGBA());
        for (int tickIdx = 0; tickIdx < numTicks; tickIdx++) {
            float centerY = y + float(tickIdx) / float(numTicks - 1) * h;
            vkvg_rectangle(
                    context, (x + w) * s, (centerY - tickHeight / 2.0f) * s, tickWidth * s, tickHeight * s);
        }
        vkvg_fill(context);
    }
#endif

    // Draw on the right.
    if (vg) {
        nvgFontSize(vg, textSizeLegend);
        nvgFontFace(vg, "sans");
        for (int tickIdx = 0; tickIdx < numLabels; tickIdx++) {
            float t = 1.0f - float(tickIdx) / float(numLabels - 1);
            float centerY = y + float(tickIdx) / float(numLabels - 1) * h;
            std::string labelText = labelMap(t);
            nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
            nvgFillColor(vg, textColorNvg);
            nvgText(vg, x + w + 2.0f * tickWidth, centerY, labelText.c_str(), nullptr);
        }
        nvgFillColor(vg, textColorNvg);
        nvgFill(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setColor(toSkColor(textColor));
        for (int tickIdx = 0; tickIdx < numLabels; tickIdx++) {
            float t = 1.0f - float(tickIdx) / float(numLabels - 1);
            float centerY = y + float(tickIdx) / float(numLabels - 1) * h;
            std::string labelText = labelMap(t);
            SkRect bounds{};
            font->measureText(labelText.c_str(), labelText.size(), SkTextEncoding::kUTF8, &bounds);
            canvas->drawString(
                    labelText.c_str(),
                    (x + w + 2.0f * tickWidth) * s, centerY * s + 0.5f * (bounds.height() - metrics.fDescent),
                    *font, *paint);
        }
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_font_size(context, uint32_t(std::round(textSizeLegend * s * 0.75f)));
        //vkvg_select_font_face(context, "sans");
        vkvg_set_source_color(context, textColor.getColorRGBA());
        for (int tickIdx = 0; tickIdx < numLabels; tickIdx++) {
            float t = 1.0f - float(tickIdx) / float(numLabels - 1);
            float centerY = y + float(tickIdx) / float(numLabels - 1) * h;
            std::string labelText = labelMap(t);
            vkvg_text_extents_t te{};
            vkvg_text_extents(context, labelText.c_str(), &te);
            vkvg_font_extents_t fe{};
            vkvg_font_extents(context, &fe);
            vkvg_move_to(context, (x + w + 2.0f * tickWidth) * s, centerY * s + 0.5f * te.height - fe.descent);
            vkvg_show_text(context, labelText.c_str());
        }
    }
#endif

    // Draw text on the top.
    if (vg) {
        nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_BOTTOM);
        nvgFillColor(vg, textColorNvg);
        nvgText(vg, x + w * 0.5f, y - 4, textTop.c_str(), nullptr);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        SkRect bounds{};
        font->measureText(textTop.c_str(), textTop.size(), SkTextEncoding::kUTF8, &bounds);
        paint->setColor(toSkColor(textColor));
        canvas->drawString(
                textTop.c_str(), (x + w * 0.5f) * s - 0.5f * bounds.width(), (y - 4) * s - metrics.fDescent,
                *font, *paint);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_text_extents_t te{};
        vkvg_text_extents(context, textTop.c_str(), &te);
        vkvg_font_extents_t fe{};
        vkvg_font_extents(context, &fe);
        vkvg_move_to(context, (x + w * 0.5f) * s - 0.5f * te.width, (y - 4) * s - fe.descent);
        vkvg_show_text(context, textTop.c_str());
    }
#endif

    // Draw box outline.
    if (vg) {
        nvgBeginPath(vg);
        nvgRect(vg, x, y, w, h);
        nvgStrokeWidth(vg, 0.75f);
        nvgStrokeColor(vg, textColorNvg);
        nvgStroke(vg);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        paint->setStroke(true);
        paint->setStrokeWidth(0.75f * s);
        paint->setColor(toSkColor(textColor));
        canvas->drawRect(SkRect{x * s, y * s, (x + w) * s, (y + h) * s}, *paint);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_set_line_width(context, 0.75f * s);
        vkvg_set_source_color(context, textColor.getColorRGBA());
        vkvg_rectangle(context, x * s, y * s, w * s, h * s);
        vkvg_stroke(context);
    }
#endif

#ifdef SUPPORT_SKIA
    if (canvas) {
        delete paint;
        delete gradientPaint;
        delete font;
    }
#endif
}

template<typename T> inline T sqr(T val) { return val * val; }

void TimeSeriesCorrelationChart::computeColorLegendHeight() {
    textWidthMax = textWidthMaxBase * textSize / 8.0f;
    const float maxHeight = std::min(maxColorLegendHeight, dh * 0.5f);
    colorLegendHeight = maxHeight;
}
