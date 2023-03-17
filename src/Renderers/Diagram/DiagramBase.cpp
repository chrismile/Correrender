/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2023, Christoph Neuhauser
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

#ifdef SUPPORT_SKIA
#include <core/SkCanvas.h>
#include <core/SkPaint.h>
#endif
#ifdef SUPPORT_VKVG
#include <vkvg.h>
#endif

#include <Math/Geometry/AABB2.hpp>
#include <Utils/AppSettings.hpp>
#include <Input/Mouse.hpp>
#include <Math/Math.hpp>
#include <Graphics/Vector/VectorBackendNanoVG.hpp>
#include <Graphics/Vector/nanovg/nanovg.h>
#include <ImGui/ImGuiWrapper.hpp>

#ifdef SUPPORT_SKIA
#include "VectorBackendSkia.hpp"
#endif
#ifdef SUPPORT_VKVG
#include "VectorBackendVkvg.hpp"
#endif
#include "DiagramBase.hpp"

DiagramBase::DiagramBase() {
    sgl::NanoVGSettings nanoVgSettings{};
    if (sgl::AppSettings::get()->getOffscreenContext()) {
        nanoVgSettings.renderBackend = sgl::RenderSystem::OPENGL;
    } else {
        nanoVgSettings.renderBackend = sgl::RenderSystem::VULKAN;
    }

    registerRenderBackendIfSupported<sgl::VectorBackendNanoVG>([this]() { this->renderBaseNanoVG(); }, nanoVgSettings);
}

void DiagramBase::initialize() {
    _initialize();
}

void DiagramBase::onBackendCreated() {
#ifdef SUPPORT_SKIA
    if (strcmp(vectorBackend->getID(), VectorBackendSkia::getClassID()) == 0) {
        auto* skiaBackend = static_cast<VectorBackendSkia*>(vectorBackend);
        typeface = skiaBackend->createDefaultTypeface();
    }
#endif
}

void DiagramBase::onBackendDestroyed() {
    vg = nullptr;
#ifdef SUPPORT_SKIA
    canvas = nullptr;
#endif
#ifdef SUPPORT_VKVG
    context = nullptr;
#endif
}

void DiagramBase::setImGuiWindowOffset(int offsetX, int offsetY) {
    imGuiWindowOffsetX = offsetX;
    imGuiWindowOffsetY = offsetY;
}

void DiagramBase::setClearColor(const sgl::Color& clearColor) {
    float r = clearColor.getFloatR();
    float g = clearColor.getFloatG();
    float b = clearColor.getFloatB();
    float clearColorLuminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    isDarkMode = clearColorLuminance <= 0.5f;
}

void DiagramBase::setIsMouseGrabbedByParent(bool _isMouseGrabbedByParent) {
    isMouseGrabbedByParent = _isMouseGrabbedByParent;
}

void DiagramBase::update(float dt) {
    glm::ivec2 mousePositionPx(sgl::Mouse->getX(), sgl::Mouse->getY());
    glm::vec2 mousePosition(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePosition -= glm::vec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
        mousePositionPx -= glm::ivec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    mousePosition -= glm::vec2(getWindowOffsetX(), getWindowOffsetY());
    mousePosition /= getScaleFactor();

    bool isMouseOverDiagram = getIsMouseOverDiagram(mousePositionPx) && !isMouseGrabbedByParent;
    windowMoveOrResizeJustFinished = false;

    // Mouse press event.
    if (isMouseOverDiagram && !isWindowFixed) {
        if (sgl::Mouse->buttonPressed(1)) {
            isMouseGrabbed = true;
        }
        mousePressEventResizeWindow(mousePositionPx, mousePosition);
        mousePressEventMoveWindow(mousePositionPx, mousePosition);
    }

    // Mouse move event.
    if (sgl::Mouse->mouseMoved()) {
        if (isMouseOverDiagram || isMouseGrabbed) {
            mouseMoveEvent(mousePositionPx, mousePosition);
        } else {
            mouseMoveEventParent(mousePositionPx, mousePosition);
        }
    }

    // Mouse release event.
    if (sgl::Mouse->buttonReleased(1)) {
        checkWindowMoveOrResizeJustFinished(mousePositionPx);
        resizeDirection = ResizeDirection::NONE;
        isDraggingWindow = false;
        isResizingWindow = false;
        isMouseGrabbed =  false;
    }
}

void DiagramBase::checkWindowMoveOrResizeJustFinished(const glm::ivec2& mousePositionPx) {
    bool dragFinished =
            isDraggingWindow && (mousePositionPx.x - mouseDragStartPosX || mousePositionPx.y - mouseDragStartPosY);
    bool resizeFinished = isResizingWindow;
    if (dragFinished || resizeFinished) {
        windowMoveOrResizeJustFinished = true;
    }
}

bool DiagramBase::getIsMouseOverDiagramImGui() const {
    glm::ivec2 mousePositionPx(sgl::Mouse->getX(), sgl::Mouse->getY());
    if (sgl::ImGuiWrapper::get()->getUseDockSpaceMode()) {
        mousePositionPx -= glm::ivec2(imGuiWindowOffsetX, imGuiWindowOffsetY);
    }
    return getIsMouseOverDiagram(mousePositionPx);
}

void DiagramBase::mouseMoveEvent(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled) {
    if (sgl::Mouse->buttonReleased(1)) {
        checkWindowMoveOrResizeJustFinished(mousePositionPx);
        resizeDirection = ResizeDirection::NONE;
        isDraggingWindow = false;
        isResizingWindow = false;
    }

    if (resizeDirection != ResizeDirection::NONE) {
        auto diffX = float(mousePositionPx.x - lastResizeMouseX);
        auto diffY = float(mousePositionPx.y - lastResizeMouseY);
        if ((resizeDirection & ResizeDirection::LEFT) != 0) {
            windowOffsetX += diffX;
            windowWidth -= diffX / scaleFactor;
        }
        if ((resizeDirection & ResizeDirection::RIGHT) != 0) {
            windowWidth += diffX / scaleFactor;
        }
        if ((resizeDirection & ResizeDirection::BOTTOM) != 0) {
            windowOffsetY += diffY;
            windowHeight -= diffY / scaleFactor;
        }
        if ((resizeDirection & ResizeDirection::TOP) != 0) {
            windowHeight += diffY / scaleFactor;
        }
        lastResizeMouseX = mousePositionPx.x;
        lastResizeMouseY = mousePositionPx.y;
        needsReRender = true;
        syncRendererWithCpu();
        onWindowSizeChanged();
        onUpdatedWindowSize();
    } else {
        glm::vec2 mousePosition(float(mousePositionPx.x), float(mousePositionPx.y));

        sgl::AABB2 leftAabb;
        leftAabb.min = glm::vec2(windowOffsetX, windowOffsetY);
        leftAabb.max = glm::vec2(windowOffsetX + resizeMargin, windowOffsetY + float(fboHeightDisplay));
        sgl::AABB2 rightAabb;
        rightAabb.min = glm::vec2(windowOffsetX + float(fboWidthDisplay) - resizeMargin, windowOffsetY);
        rightAabb.max = glm::vec2(windowOffsetX + float(fboWidthDisplay), windowOffsetY + float(fboHeightDisplay));
        sgl::AABB2 bottomAabb;
        bottomAabb.min = glm::vec2(windowOffsetX, windowOffsetY);
        bottomAabb.max = glm::vec2(windowOffsetX + float(fboWidthDisplay), windowOffsetY + resizeMargin);
        sgl::AABB2 topAabb;
        topAabb.min = glm::vec2(windowOffsetX, windowOffsetY + float(fboHeightDisplay) - resizeMargin);
        topAabb.max = glm::vec2(windowOffsetX + float(fboWidthDisplay), windowOffsetY + float(fboHeightDisplay));

        ResizeDirection resizeDirectionCurr = ResizeDirection::NONE;
        if (leftAabb.contains(mousePosition)) {
            resizeDirectionCurr = ResizeDirection(resizeDirectionCurr | ResizeDirection::LEFT);
        }
        if (rightAabb.contains(mousePosition)) {
            resizeDirectionCurr = ResizeDirection(resizeDirectionCurr | ResizeDirection::RIGHT);
        }
        if (bottomAabb.contains(mousePosition)) {
            resizeDirectionCurr = ResizeDirection(resizeDirectionCurr | ResizeDirection::BOTTOM);
        }
        if (topAabb.contains(mousePosition)) {
            resizeDirectionCurr = ResizeDirection(resizeDirectionCurr | ResizeDirection::TOP);
        }

        sgl::CursorType newCursorShape = sgl::CursorType::DEFAULT;
        if (resizeDirectionCurr == ResizeDirection::LEFT
                || resizeDirectionCurr == ResizeDirection::RIGHT) {
            newCursorShape = sgl::CursorType::SIZEWE;
        } else if (resizeDirectionCurr == ResizeDirection::BOTTOM
                || resizeDirectionCurr == ResizeDirection::TOP) {
            newCursorShape = sgl::CursorType::SIZENS;
        } else if (resizeDirectionCurr == ResizeDirection::BOTTOM_LEFT
                || resizeDirectionCurr == ResizeDirection::TOP_RIGHT) {
            newCursorShape = sgl::CursorType::SIZENESW;
        } else if (resizeDirectionCurr == ResizeDirection::TOP_LEFT
                || resizeDirectionCurr == ResizeDirection::BOTTOM_RIGHT) {
            newCursorShape = sgl::CursorType::SIZENWSE;
        } else {
            newCursorShape = sgl::CursorType::DEFAULT;
        }

        if (newCursorShape != cursorShape) {
            sgl::Window* window = sgl::AppSettings::get()->getMainWindow();
            cursorShape = newCursorShape;
            window->setCursorType(cursorShape);
        }
    }

    if (isDraggingWindow) {
        windowOffsetX = windowOffsetXBase + float(mousePositionPx.x - mouseDragStartPosX);
        windowOffsetY = windowOffsetYBase + float(mousePositionPx.y - mouseDragStartPosY);
        needsReRender = true;
    }
}

void DiagramBase::mouseMoveEventParent(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled) {
    if (sgl::Mouse->isButtonUp(1)) {
        checkWindowMoveOrResizeJustFinished(mousePositionPx);
        resizeDirection = ResizeDirection::NONE;
        isDraggingWindow = false;
        isResizingWindow = false;
    }

    if (resizeDirection != ResizeDirection::NONE) {
        float diffX = float(mousePositionPx.x - lastResizeMouseX);
        float diffY = float(mousePositionPx.y - lastResizeMouseY);
        if ((resizeDirection & ResizeDirection::LEFT) != 0) {
            windowOffsetX += diffX;
            windowWidth -= diffX / scaleFactor;
        }
        if ((resizeDirection & ResizeDirection::RIGHT) != 0) {
            windowWidth += diffX / scaleFactor;
        }
        if ((resizeDirection & ResizeDirection::BOTTOM) != 0) {
            windowOffsetY += diffY;
            windowHeight -= diffY / scaleFactor;
        }
        if ((resizeDirection & ResizeDirection::TOP) != 0) {
            windowHeight += diffY / scaleFactor;
        }
        lastResizeMouseX = mousePositionPx.x;
        lastResizeMouseY = mousePositionPx.y;
        needsReRender = true;
        syncRendererWithCpu();
        onWindowSizeChanged();
        onUpdatedWindowSize();
    } else {
        if (cursorShape != sgl::CursorType::DEFAULT) {
            sgl::Window* window = sgl::AppSettings::get()->getMainWindow();
            cursorShape = sgl::CursorType::DEFAULT;
            window->setCursorType(cursorShape);
        }
    }

    if (isDraggingWindow) {
        windowOffsetX = windowOffsetXBase + float(mousePositionPx.x - mouseDragStartPosX);
        windowOffsetY = windowOffsetYBase + float(mousePositionPx.y - mouseDragStartPosY);
        needsReRender = true;
    }
}

void DiagramBase::mousePressEventResizeWindow(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled) {
    if (sgl::Mouse->buttonPressed(1)) {
        // First, check if a resize event was started.
        glm::vec2 mousePosition(float(mousePositionPx.x), float(mousePositionPx.y));

        sgl::AABB2 leftAabb;
        leftAabb.min = glm::vec2(windowOffsetX, windowOffsetY);
        leftAabb.max = glm::vec2(windowOffsetX + resizeMargin, windowOffsetY + float(fboHeightDisplay));
        sgl::AABB2 rightAabb;
        rightAabb.min = glm::vec2(windowOffsetX + float(fboWidthDisplay) - resizeMargin, windowOffsetY);
        rightAabb.max = glm::vec2(windowOffsetX + float(fboWidthDisplay), windowOffsetY + float(fboHeightDisplay));
        sgl::AABB2 bottomAabb;
        bottomAabb.min = glm::vec2(windowOffsetX, windowOffsetY);
        bottomAabb.max = glm::vec2(windowOffsetX + float(fboWidthDisplay), windowOffsetY + resizeMargin);
        sgl::AABB2 topAabb;
        topAabb.min = glm::vec2(windowOffsetX, windowOffsetY + float(fboHeightDisplay) - resizeMargin);
        topAabb.max = glm::vec2(windowOffsetX + float(fboWidthDisplay), windowOffsetY + float(fboHeightDisplay));

        resizeDirection = ResizeDirection::NONE;
        if (leftAabb.contains(mousePosition)) {
            resizeDirection = ResizeDirection(resizeDirection | ResizeDirection::LEFT);
        }
        if (rightAabb.contains(mousePosition)) {
            resizeDirection = ResizeDirection(resizeDirection | ResizeDirection::RIGHT);
        }
        if (bottomAabb.contains(mousePosition)) {
            resizeDirection = ResizeDirection(resizeDirection | ResizeDirection::BOTTOM);
        }
        if (topAabb.contains(mousePosition)) {
            resizeDirection = ResizeDirection(resizeDirection | ResizeDirection::TOP);
        }

        if (resizeDirection != ResizeDirection::NONE) {
            isResizingWindow = true;
            lastResizeMouseX = mousePositionPx.x;
            lastResizeMouseY = mousePositionPx.y;
        }
    }
}

void DiagramBase::mousePressEventMoveWindow(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled) {
    if (resizeDirection == ResizeDirection::NONE && sgl::Mouse->buttonPressed(1)) {
        isDraggingWindow = true;
        windowOffsetXBase = windowOffsetX;
        windowOffsetYBase = windowOffsetY;
        mouseDragStartPosX = mousePositionPx.x;
        mouseDragStartPosY = mousePositionPx.y;
    }
}


void DiagramBase::getNanoVGContext() {
    vg = static_cast<sgl::VectorBackendNanoVG*>(vectorBackend)->getContext();
#ifdef SUPPORT_SKIA
    canvas = nullptr;
#endif
#ifdef SUPPORT_VKVG
    context = nullptr;
#endif
}

void DiagramBase::renderBaseNanoVG() {
    getNanoVGContext();

    sgl::Color backgroundFillColor = isDarkMode ? backgroundFillColorDark : backgroundFillColorBright;
    sgl::Color backgroundStrokeColor = isDarkMode ? backgroundStrokeColorDark : backgroundStrokeColorBright;
    NVGcolor backgroundFillColorNvg = nvgRGBA(
            backgroundFillColor.getR(), backgroundFillColor.getG(),
            backgroundFillColor.getB(), std::clamp(int(backgroundOpacity * 255), 0, 255));
    NVGcolor backgroundStrokeColorNvg = nvgRGBA(
            backgroundStrokeColor.getR(), backgroundStrokeColor.getG(),
            backgroundStrokeColor.getB(), std::clamp(int(backgroundOpacity * 255), 0, 255));

    // Render the render target-filling widget rectangle.
    nvgBeginPath(vg);
    nvgRoundedRect(
            vg, borderWidth, borderWidth, windowWidth - 2.0f * borderWidth, windowHeight - 2.0f * borderWidth,
            borderRoundingRadius);
    nvgFillColor(vg, backgroundFillColorNvg);
    nvgFill(vg);
    if (renderBackgroundStroke) {
        nvgStrokeColor(vg, backgroundStrokeColorNvg);
        nvgStroke(vg);
    }
}


#ifdef SUPPORT_SKIA
void DiagramBase::getSkiaCanvas() {
    vg = nullptr;
    canvas = static_cast<VectorBackendSkia*>(vectorBackend)->getCanvas();
#ifdef SUPPORT_VKVG
    context = nullptr;
#endif
}

void DiagramBase::renderBaseSkia() {
    getSkiaCanvas();
    s = scaleFactor * float(supersamplingFactor);

    sgl::Color backgroundFillColor = isDarkMode ? backgroundFillColorDark : backgroundFillColorBright;
    sgl::Color backgroundStrokeColor = isDarkMode ? backgroundStrokeColorDark : backgroundStrokeColorBright;
    backgroundFillColor.setA(std::clamp(int(backgroundOpacity * 255), 0, 255));
    backgroundStrokeColor.setA(std::clamp(int(backgroundOpacity * 255), 0, 255));

    // Render the render target-filling widget rectangle.
    SkPaint paint;
    static_cast<VectorBackendSkia*>(vectorBackend)->initializePaint(&paint);

    paint.setColor(toSkColor(backgroundFillColor));
    paint.setStroke(false);
    canvas->drawRoundRect(
            SkRect{borderWidth * s, borderWidth * s, (windowWidth - borderWidth) * s, (windowHeight - borderWidth) * s},
            borderRoundingRadius * s, borderRoundingRadius * s, paint);

    if (renderBackgroundStroke) {
        paint.setColor(toSkColor(backgroundStrokeColor));
        paint.setStroke(true);
        paint.setStrokeWidth(1.0f * s);
        canvas->drawRoundRect(
                SkRect{borderWidth * s, borderWidth * s, (windowWidth - borderWidth) * s, (windowHeight - borderWidth) * s},
                borderRoundingRadius * s, borderRoundingRadius * s, paint);
    }
}
#endif


#ifdef SUPPORT_VKVG
void DiagramBase::getVkvgContext() {
    vg = nullptr;
#ifdef SUPPORT_SKIA
    canvas = nullptr;
#endif
    context = static_cast<VectorBackendVkvg*>(vectorBackend)->getContext();
}

void DiagramBase::renderBaseVkvg() {
    getVkvgContext();
    s = scaleFactor * float(supersamplingFactor);

    sgl::Color backgroundFillColor = isDarkMode ? backgroundFillColorDark : backgroundFillColorBright;
    sgl::Color backgroundStrokeColor = isDarkMode ? backgroundStrokeColorDark : backgroundStrokeColorBright;

    // Render the render target-filling widget rectangle.
    vkvg_rounded_rectangle(
            context, borderWidth * s, borderWidth * s,
            (windowWidth - 2.0f * borderWidth) * s, (windowHeight - 2.0f * borderWidth) * s,
            borderRoundingRadius * s);
    vkvg_set_opacity(context, backgroundOpacity);
    vkvg_set_source_color(context, backgroundFillColor.getColorRGBA());
    if (renderBackgroundStroke) {
        vkvg_fill_preserve(context);
        vkvg_set_source_color(context, backgroundStrokeColor.getColorRGBA());
        vkvg_set_line_width(context, 1.0f * s);
        vkvg_stroke(context);
    } else {
        vkvg_fill(context);
    }

    vkvg_set_opacity(context, 1.0f);
}
#endif


/// Removes trailing zeros and unnecessary decimal points.
std::string removeTrailingZeros(const std::string& numberString) {
    size_t lastPos = numberString.size();
    for (int i = int(numberString.size()) - 1; i > 0; i--) {
        char c = numberString.at(i);
        if (c == '.') {
            lastPos--;
            break;
        }
        if (c != '0') {
            break;
        }
        lastPos--;
    }
    return numberString.substr(0, lastPos);
}

/// Removes decimal points if more than maxDigits digits are used.
std::string DiagramBase::getNiceNumberString(float number, int digits) {
    int maxDigits = digits + 2; // Add 2 digits for '.' and one digit afterwards.
    std::string outString = removeTrailingZeros(sgl::toString(number, digits, true));

    // Can we remove digits after the decimal point?
    size_t dotPos = outString.find('.');
    if (int(outString.size()) > maxDigits && dotPos != std::string::npos) {
        size_t substrSize = dotPos;
        if (int(dotPos) < maxDigits - 1) {
            substrSize = maxDigits;
        }
        outString = outString.substr(0, substrSize);
    }

    // Still too large?
    if (int(outString.size()) > maxDigits || (outString == "0" && number > std::numeric_limits<float>::epsilon())) {
        outString = sgl::toString(number, std::max(digits - 2, 1), false, false, true);
    }
    return outString;
}

void DiagramBase::drawColorLegend(
        const NVGcolor& textColor, float x, float y, float w, float h, int numLabels, size_t numTicks,
        const std::function<std::string(float)>& labelMap, const std::function<NVGcolor(float)>& colorMap,
        const std::string& textTop) {
    const int numPoints = 9;
    const int numSubdivisions = numPoints - 1;

    // Draw color bar.
    for (int i = 0; i < numSubdivisions; i++) {
        float t0 = 1.0f - float(i) / float(numSubdivisions);
        float t1 = 1.0f - float(i + 1) / float(numSubdivisions);
        NVGcolor fillColor0 = colorMap(t0);
        NVGcolor fillColor1 = colorMap(t1);
        nvgBeginPath(vg);
        nvgRect(vg, x, y + h * float(i) / float(numSubdivisions), w, h / float(numSubdivisions));
        NVGpaint paint = nvgLinearGradient(
                vg, x, y + h * float(i) / float(numSubdivisions), x, y + h * float(i+1) / float(numSubdivisions),
                fillColor0, fillColor1);
        nvgFillPaint(vg, paint);
        nvgFill(vg);
    }

    // Draw ticks.
    const float tickWidth = 4.0f;
    const float tickHeight = 1.0f;
    nvgBeginPath(vg);
    for (size_t tickIdx = 0; tickIdx < numTicks; tickIdx++) {
        //float t = 1.0f - float(tickIdx) / float(int(numTicks) - 1);
        float centerY = y + float(tickIdx) / float(int(numTicks) - 1) * h;
        nvgRect(vg, x + w, centerY - tickHeight / 2.0f, tickWidth, tickHeight);
    }
    nvgFillColor(vg, textColor);
    nvgFill(vg);

    // Draw on the right.
    nvgFontSize(vg, textSizeLegend);
    nvgFontFace(vg, "sans");
    for (size_t tickIdx = 0; tickIdx < numTicks; tickIdx++) {
        float t = 1.0f - float(tickIdx) / float(int(numTicks) - 1);
        float centerY = y + float(tickIdx) / float(int(numTicks) - 1) * h;
        std::string labelText = labelMap(t);
        nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_MIDDLE);
        nvgFillColor(vg, textColor);
        nvgText(vg, x + w + 2.0f * tickWidth, centerY, labelText.c_str(), nullptr);
    }
    nvgFillColor(vg, textColor);
    nvgFill(vg);

    // Draw text on the top.
    nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_BOTTOM);
    nvgFillColor(vg, textColor);
    nvgText(vg, x + w / 2.0f, y - 4, textTop.c_str(), nullptr);

    // Draw box outline.
    nvgBeginPath(vg);
    nvgRect(vg, x, y, w, h);
    nvgStrokeWidth(vg, 0.75f);
    nvgStrokeColor(vg, textColor);
    nvgStroke(vg);
}
