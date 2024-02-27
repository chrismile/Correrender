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
#include <Graphics/Vector/nanovg/nanovg.h>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>

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
#include "CorrelationMatrixChart.hpp"

CorrelationMatrixChart::CorrelationMatrixChart() {
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

CorrelationMatrixChart::~CorrelationMatrixChart() = default;

void CorrelationMatrixChart::initialize() {
    borderSizeX = 10;
    borderSizeY = 10;
    windowWidth = (200 + borderSizeX) * 2.0f;
    windowHeight = (200 + borderSizeY) * 2.0f;

    DiagramBase::initialize();
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void CorrelationMatrixChart::onUpdatedWindowSize() {
    //if (windowWidth < 360.0f || windowHeight < 360.0f) {
    //    borderSizeX = borderSizeY = 10.0f;
    //} else {
    //    borderSizeX = borderSizeY = std::min(windowWidth, windowHeight) / 36.0f;
    //}
    if (windowWidth < 360.0f || windowHeight < 360.0f) {
        borderSizeX = borderSizeY = 60.0f;
    } else {
        borderSizeX = borderSizeY = std::min(windowWidth, windowHeight) / 6.0f;
    }
    float minDim = std::min(windowWidth - 2.0f * borderSizeX, windowHeight - 2.0f * borderSizeY);
    totalRadius = std::round(0.5f * minDim);
    computeColorLegendHeight();
}

void CorrelationMatrixChart::updateSizeByParent() {
    auto [parentWidth, parentHeight] = getBlitTargetSize();
    auto ssf = float(blitTargetSupersamplingFactor);
    windowOffsetX = 0;
    windowOffsetY = 0;
    windowWidth = float(parentWidth) / (scaleFactor * float(ssf));
    windowHeight = float(parentHeight) / (scaleFactor * float(ssf));
    onUpdatedWindowSize();
    onWindowSizeChanged();
}

void CorrelationMatrixChart::setAlignWithParentWindow(bool _align) {
    alignWithParentWindow = _align;
    renderBackgroundStroke = !_align;
    isWindowFixed = alignWithParentWindow;
    if (alignWithParentWindow) {
        updateSizeByParent();
    }
}

void CorrelationMatrixChart::setColorMap(DiagramColorMap _colorMap) {
    if (colorMap != _colorMap) {
        colorMap = _colorMap;
        needsReRender = true;
    }
    colorPoints = getColorPoints(colorMap);
}

glm::vec4 CorrelationMatrixChart::evalColorMapVec4(float t) {
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

sgl::Color CorrelationMatrixChart::evalColorMap(float t) {
    return sgl::colorFromVec4(evalColorMapVec4(t));
}

void CorrelationMatrixChart::update(float dt) {
    DiagramBase::update(dt);

    /*float startX = windowWidth / 2.0f - chartRadius;
    float startY = windowHeight / 2.0f - chartRadius;
    float matrixWidth = 2.0f * chartRadius;
    float matrixHeight = 2.0f * chartRadius;
    glm::vec2 pctMousePos = (mousePosition - glm::vec2(startX, startY)) / glm::vec2(matrixWidth, matrixHeight);
    glm::vec2 gridMousePos = pctMousePos * glm::vec2(correlationMatrix->getNumRows(), correlationMatrix->getNumColumns());
    auto gridPosition = glm::ivec2(gridMousePos);
    int i = gridPosition.x;
    int j = gridPosition.y;

    if (pctMousePos.x >= 0.0f && pctMousePos.y >= 0.0f && pctMousePos.x < 1.0f && pctMousePos.y < 1.0f
        && (!correlationMatrix->getIsSymmetric() || i > j)) {
        if (!regionsEqual) {
            gridPosition.y += int(leafIdxOffset1 - leafIdxOffset);
        }
        hoveredGridIdx = gridPosition;
    } else {
        hoveredGridIdx = {};
    }

    hoveredPointIdx = -1;
    hoveredLineIdx = -1;
    selectedLineIdx = -1;*/
}

void CorrelationMatrixChart::setMatrixData(
        CorrelationMeasureType _correlationMeasureType,
        const std::vector<std::string>& _fieldNames,
        const std::shared_ptr<CorrelationMatrix>& _similarityMatrix,
        const std::pair<float, float>& _minMaxCorrelationValue) {
    correlationMeasureType = _correlationMeasureType;
    fieldNames = _fieldNames;
    correlationMatrix = _similarityMatrix;
    minMaxCorrelationValue = _minMaxCorrelationValue;
    dataDirty = true;
}

void CorrelationMatrixChart::updateData() {
    ;
}

void CorrelationMatrixChart::renderScatterPlot() {
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

    const float chartRadius = totalRadius;

    auto [minMi, maxMi] = minMaxCorrelationValue;
    float startX = windowWidth / 2.0f - chartRadius;
    float startY = windowHeight / 2.0f - chartRadius;
    float matrixWidth = 2.0f * chartRadius;
    float matrixHeight = 2.0f * chartRadius;

    bool isSymmetric = correlationMatrix->getIsSymmetric();
    int numRows = correlationMatrix->getNumRows();
    int numColumns = correlationMatrix->getNumRows();
    float wp  = matrixWidth * float(1) / float(numRows);
    float hp = matrixHeight * float(1) / float(numColumns);
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numColumns; j++) {
            if (isSymmetric && i <= j) {
                continue;
            }
            float correlationValue = correlationMatrix->get(i, j);
            if (std::isnan(correlationValue)) {
                continue;
            }
            float factor = 1.0f;
            if (minMi < maxMi) {
                factor = (correlationValue - minMi) / (maxMi - minMi);
            }

            float w = wp;
            float h = hp;
            //float x = startX + float(numRows - i - 1) * w;
            float x = startX + float(i) * w;
            //float y = startY + float(j) * h;
            float y = startY + float(numRows - j - 1) * h;
            w += 1e-1f;
            h += 1e-1f;

            if (vg) {
                glm::vec4 color = evalColorMapVec4(factor);
                nvgBeginPath(vg);
                nvgRect(vg, x, y, w, h);
                nvgFillColor(vg, nvgRGBAf(color.x, color.y, color.z, 1.0f));
                nvgFill(vg);
            }
#ifdef SUPPORT_SKIA
            else if (canvas) {
                sgl::Color color = evalColorMap(factor);
                color.setA(255);
                paint->setStroke(false);
                paint->setColor(toSkColor(color));
                canvas->drawRect(SkRect{x * s, y * s, (x + w) * s, (y + h) * s}, *paint);
            }
#endif
#ifdef SUPPORT_VKVG
            else if (context) {
                sgl::Color color = evalColorMap(factor);
                color.setA(255);
                vkvg_rectangle(context, x * s, y * s, w * s, h * s);
                vkvg_set_source_color(context, color.getColorRGBA());
                vkvg_fill(context);
            }
#endif
        }
    }

    // Draw field names.
    for (int i = 0; i < numRows; i++) {
        // Draw on the right.
        const float w = wp;
        const float h = hp;
        //float x0 = startX + float(numRows) * w;
        const float x0 = startX;
        //const float y0 = startY + (float(i) + 0.5f) * h;
        const float y0 = startY + (float(numRows - i - 1) + 0.5f) * h;
        const float x1 = startX + (float(i) + 0.5f) * w;
        const float y1 = startY + float(numRows) * h;
        const float angleCenter = float(M_PI) / 4.0f;
        std::string labelText = fieldNames.at(i);
        if (vg) {
            // Text side
            nvgBeginPath(vg);
            nvgFontSize(vg, textSizeLegend);
            nvgFontFace(vg, "sans");
            //nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
            nvgTextAlign(vg, NVG_ALIGN_RIGHT | NVG_ALIGN_MIDDLE);
            nvgFillColor(vg, textColorNvg);
            nvgText(vg, x0, y0, labelText.c_str(), nullptr);
            nvgFill(vg);

            // Text top
            const glm::vec2 textPosition(x1, y1 + textSizeLegend / 2.0f);
            nvgSave(vg);
            nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
            glm::vec2 bounds[2];
            nvgTextBounds(vg, textPosition.x, textPosition.y, labelText.c_str(), nullptr, &bounds[0].x);
            nvgTranslate(vg, textPosition.x, textPosition.y);
            nvgRotate(vg, angleCenter);
            nvgTranslate(vg, -textPosition.x, -textPosition.y);
            nvgFillColor(vg, textColorNvg);
            nvgText(vg, textPosition.x, textPosition.y, labelText.c_str(), nullptr);
            nvgRestore(vg);
        }
#ifdef SUPPORT_SKIA
        else if (canvas) {
            // Text side
            paint->setColor(toSkColor(textColor));
            SkRect bounds{};
            font->measureText(labelText.c_str(), labelText.size(), SkTextEncoding::kUTF8, &bounds);
            canvas->drawString(
                    labelText.c_str(),
                    x0 * s - bounds.width(), y0 * s + 0.5f * (bounds.height() - metrics.fDescent),
                    *font, *paint);

            // Text top
            const glm::vec2 textPosition(
                    x1 * s, (y1 + textSizeLegend / 2.0f) * s + 0.5f * (bounds.height() - metrics.fDescent));
            canvas->save();
            //canvas->rotate(angleCenter / sgl::PI * 180.0f, textPosition.x, textPosition.y);
            canvas->translate(textPosition.x, textPosition.y);
            canvas->rotate(angleCenter / sgl::PI * 180.0f);
            canvas->translate(-textPosition.x, -textPosition.y);
            canvas->drawString(labelText.c_str(), textPosition.x, textPosition.y, *font, *paint);
            canvas->restore();
        }
#endif
#ifdef SUPPORT_VKVG
        else if (context) {
            // Text side
            vkvg_set_font_size(context, uint32_t(std::round(textSizeLegend * s * 0.75f)));
            //vkvg_select_font_face(context, "sans");
            vkvg_set_source_color(context, textColor.getColorRGBA());
            vkvg_text_extents_t te{};
            vkvg_text_extents(context, labelText.c_str(), &te);
            vkvg_font_extents_t fe{};
            vkvg_font_extents(context, &fe);
            vkvg_move_to(context, x0 * s - te.width, y0 * s + 0.5f * te.height - fe.descent);
            vkvg_show_text(context, labelText.c_str());

            // Text top
            const glm::vec2 textPosition(x1 * s, (y1 + textSizeLegend / 2.0f) * s + 0.5f * te.height - fe.descent);
            vkvg_save(context);
            vkvg_translate(context, textPosition.x, textPosition.y);
            vkvg_rotate(context, angleCenter);
            vkvg_translate(context, -textPosition.x, -textPosition.y);
            vkvg_move_to(context, textPosition.x, textPosition.y);
            vkvg_show_text(context, labelText.c_str());
            vkvg_restore(context);
        }
#endif
    }

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

void CorrelationMatrixChart::renderBaseNanoVG() {
    DiagramBase::renderBaseNanoVG();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}

#ifdef SUPPORT_SKIA
void CorrelationMatrixChart::renderBaseSkia() {
    DiagramBase::renderBaseSkia();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}
#endif

#ifdef SUPPORT_VKVG
void CorrelationMatrixChart::renderBaseVkvg() {
    DiagramBase::renderBaseVkvg();

    if (dataDirty) {
        updateData();
        dataDirty = false;
    }

    renderScatterPlot();
}
#endif

void CorrelationMatrixChart::drawColorLegends() {
#ifdef __APPLE__
    auto minMaxMi = minMaxCorrelationValue;
    float minMi = minMaxMi.first;
    float maxMi = minMaxMi.second;
#else
    auto [minMi, maxMi] = minMaxCorrelationValue;
#endif
    std::string variableName = CORRELATION_MEASURE_TYPE_NAMES[int(correlationMeasureType)];
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
    float posX = windowWidth / 2.0f + totalRadius + borderSizeX * 0.5f;

    float posY = windowHeight - borderSizeY - colorLegendHeight;
    drawColorLegend(
            posX, posY, colorLegendWidth, colorLegendHeight, numLabels, numTicks, labelMap, colorMap, variableName);
}

void CorrelationMatrixChart::drawColorLegend(
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

float CorrelationMatrixChart::computeColorLegendHeightForNumFields(int numFields, float maxHeight) {
    return maxHeight;
}

void CorrelationMatrixChart::computeColorLegendHeight() {
    textWidthMax = textWidthMaxBase * textSize / 8.0f;
    auto numFieldsSize = int(fieldNames.size());
    const float maxHeight = std::min(maxColorLegendHeight, totalRadius * 0.5f);
    colorLegendHeight = computeColorLegendHeightForNumFields(numFieldsSize, maxHeight);
}
