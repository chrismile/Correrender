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

#ifndef CORRERENDER_DIAGRAMBASE_HPP
#define CORRERENDER_DIAGRAMBASE_HPP

#include <set>
#include <sstream>
#include <functional>

#ifdef SUPPORT_SKIA
#include <core/SkRefCnt.h>
#include <core/SkTypeface.h>
#endif

#include <Graphics/Window.hpp>
#include <Graphics/Vector/VectorWidget.hpp>

struct NVGcontext;
typedef struct NVGcontext NVGcontext;
struct NVGcolor;
class SkCanvas;
#ifdef SUPPORT_VKVG
struct _vkvg_context_t;
typedef struct _vkvg_context_t* VkvgContext;
#endif

enum class DiagramType {
    RADAR_BAR_CHART, HEB_CHART, SCATTER_PLOT, CORRELATION_MATRIX
};

class DiagramBase : public sgl::VectorWidget {
public:
    DiagramBase();
    virtual void initialize();
    void update(float dt) override;
    [[nodiscard]] bool getIsMouseOverDiagramImGui() const;
    void setIsMouseGrabbedByParent(bool _isMouseGrabbedByParent);
    virtual void updateSizeByParent() {}
    virtual DiagramType getDiagramType()=0;
    void setImGuiWindowOffset(int offsetX, int offsetY);
    void setClearColor(const sgl::Color& clearColor);
    [[nodiscard]] inline bool getNeedsReRender() { bool tmp = needsReRender; needsReRender = false; return tmp; }
    [[nodiscard]] inline bool getIsMouseGrabbed() const { return isMouseGrabbed; }

    [[nodiscard]] inline bool getSelectedVariablesChanged() const { return selectedVariablesChanged; };
    [[nodiscard]] inline const std::set<size_t>& getSelectedVariableIndices() const { return selectedVariableIndices; };
    inline void getSelectedVariableIndices(const std::set<size_t>& newSelectedVariableIndices) {
        selectedVariableIndices = newSelectedVariableIndices;
    };

protected:
    virtual bool hasData()=0;
    void onBackendCreated() override;
    void onBackendDestroyed() override;

    // Widget move/resize events.
    void mouseMoveEvent(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled);
    void mouseMoveEventParent(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled);
    void mousePressEventResizeWindow(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled);
    void mousePressEventMoveWindow(const glm::ivec2& mousePositionPx, const glm::vec2& mousePositionScaled);
    virtual void onUpdatedWindowSize() {}

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
#endif

    // VKVG backend.
#ifdef SUPPORT_VKVG
    virtual void renderBaseVkvg();
    void getVkvgContext();
    VkvgContext context = nullptr;
#endif

    // Utility functions.
    void drawColorLegend(
            const NVGcolor& textColor, float x, float y, float w, float h, int numLabels, size_t numTicks,
            const std::function<std::string(float)>& labelMap, const std::function<NVGcolor(float)>& colorMap,
            const std::string& textTop = "");

    /// Removes decimal points if more than maxDigits digits are used.
    static std::string getNiceNumberString(float number, int digits);
    /// Conversion to and from string
    template <class T>
    static std::string toString(
            T obj, int precision, bool fixed = true, bool noshowpoint = false, bool scientific = false) {
        std::ostringstream ostr;
        ostr.precision(precision);
        if (fixed) {
            ostr << std::fixed;
        }
        if (noshowpoint) {
            ostr << std::noshowpoint;
        }
        if (scientific) {
            ostr << std::scientific;
        }
        ostr << obj;
        return ostr.str();
    }

    float textSize = 8.0f;

    bool needsReRender = false;
    float borderSizeX = 0, borderSizeY = 0;
    const float borderWidth = 1.0f;
    const float borderRoundingRadius = 4.0f;
    float backgroundOpacity = 1.0f;
    float textSizeLegend = 12.0f;

#ifdef SUPPORT_SKIA
    sk_sp<SkTypeface> typeface;
#endif

    // Color palette.
    bool isDarkMode = true;
    sgl::Color backgroundFillColorDark = sgl::Color(20, 20, 20, 255);
    //sgl::Color backgroundFillColorBright = sgl::Color(230, 230, 230, 255);
    sgl::Color backgroundFillColorBright = sgl::Color(245, 245, 245, 255);
    sgl::Color backgroundStrokeColorDark = sgl::Color(60, 60, 60, 255);
    sgl::Color backgroundStrokeColorBright = sgl::Color(190, 190, 190, 255);
    bool renderBackgroundStroke = true;

    enum ResizeDirection {
        NONE = 0, LEFT = 1, RIGHT = 2, BOTTOM = 4, TOP = 8,
        BOTTOM_LEFT = BOTTOM | LEFT, BOTTOM_RIGHT = BOTTOM | RIGHT, TOP_LEFT = TOP | LEFT, TOP_RIGHT = TOP | RIGHT
    };
    [[nodiscard]] inline ResizeDirection getResizeDirection() const { return resizeDirection; }

    // Dragging the window.
    bool isDraggingWindow = false;
    int mouseDragStartPosX = 0;
    int mouseDragStartPosY = 0;
    float windowOffsetXBase = 0.0f;
    float windowOffsetYBase = 0.0f;

    // Resizing the window.
    bool isResizingWindow = false;
    ResizeDirection resizeDirection = ResizeDirection::NONE;
    const float resizeMarginBase = 4;
    float resizeMargin = resizeMarginBase; // including scale factor
    int lastResizeMouseX = 0;
    int lastResizeMouseY = 0;
    sgl::CursorType cursorShape = sgl::CursorType::DEFAULT;

    // Offset for deducing mouse position.
    void checkWindowMoveOrResizeJustFinished(const glm::ivec2& mousePositionPx);
    int imGuiWindowOffsetX = 0, imGuiWindowOffsetY = 0;
    bool isMouseGrabbedByParent = false;
    bool isMouseGrabbed = false;
    bool windowMoveOrResizeJustFinished = false;
    bool isWindowFixed = false; //< Is resize and grabbing disabled?

    // Variables can be selected by clicking on them.
    size_t numVariables = 0;
    std::set<size_t> selectedVariableIndices;
    bool selectedVariablesChanged = false;
};

#endif //CORRERENDER_DIAGRAMBASE_HPP
