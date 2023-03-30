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

#include <Math/Geometry/AABB2.hpp>
#include <Input/Mouse.hpp>
#include <Graphics/Color.hpp>

#include <Graphics/Vector/nanovg/nanovg.h>

#ifdef SUPPORT_SKIA
#include <core/SkCanvas.h>
#include <core/SkPaint.h>
#include <core/SkPath.h>
#endif
#ifdef SUPPORT_VKVG
#include <vkvg.h>
#endif

#ifdef SUPPORT_SKIA
#include "VectorBackendSkia.hpp"
#endif
#include "DiagramButton.hpp"

DiagramButton::DiagramButton(ButtonType buttonType) : buttonType(buttonType) {}

void DiagramButton::setPosition(float _x, float _y) {
    x = _x;
    y = _y;
}

void DiagramButton::setSize(float _buttonSize) {
    buttonSize = _buttonSize;
}

void DiagramButton::render(
        NVGcontext* vg,
#ifdef SUPPORT_SKIA
        SkCanvas* canvas, SkPaint* defaultPaint,
#endif
#ifdef SUPPORT_VKVG
        VkvgContext context,
#endif
        bool isDarkMode, float s) {
    const float strokeWidth = std::max(3.0f * buttonSize / 40.0f, 1.0f);
    const float buttonEdgeRadius = buttonSize * 0.2f;

    sgl::Color textColor = isDarkMode ? textColorDark : textColorBright;
    if (buttonState == ButtonState::NORMAL) {
        textColor = isDarkMode ? textColorUnselectedDark : textColorUnselectedBright;
    }
    sgl::Color fillColorNormal = isDarkMode ? buttonFillColorNormalDark : buttonFillColorNormalBright;
    sgl::Color fillColorHovered = isDarkMode ? buttonFillColorHoveredDark : buttonFillColorHoveredBright;
    sgl::Color fillColorClicked = isDarkMode ? buttonFillColorClickedDark : buttonFillColorClickedBright;
    sgl::Color fillColor = fillColorNormal;
    if (buttonState == ButtonState::HOVERED) {
        fillColor = fillColorHovered;
    } else if (buttonState == ButtonState::CLICKED) {
        fillColor = fillColorClicked;
    }

    NVGcolor textColorNvg, fillColorNvg;
    if (vg) {
        textColorNvg = nvgRGBA(textColor.getR(), textColor.getG(), textColor.getB(), 255);
        fillColorNvg = nvgRGBA(fillColor.getR(), fillColor.getG(), fillColor.getB(), 255);
    }
#ifdef SUPPORT_SKIA
    SkPaint fillPaint, strokePaint;
    if (canvas) {
        fillPaint = *defaultPaint;
        fillPaint.setStroke(false);
        fillPaint.setColor(toSkColor(fillColor));
        strokePaint = *defaultPaint;
        strokePaint.setStroke(true);
        strokePaint.setStrokeWidth(strokeWidth * s);
        strokePaint.setColor(toSkColor(textColor));
    }
#endif

    // Points close button.
    glm::vec2 pc0(x + 0.25f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 pc1(x + 0.75f * buttonSize, y + 0.75f * buttonSize);
    glm::vec2 pc2(x + 0.75f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 pc3(x + 0.25f * buttonSize, y + 0.75f * buttonSize);
    // Points back 1 button.
    glm::vec2 p1b0(x + 0.68f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p1b1(x + 0.30f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p1b2(x + 0.68f * buttonSize, y + 0.75f * buttonSize);
    // Points back 2 button.
    glm::vec2 p2b0(x + 0.51f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p2b1(x + 0.19f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p2b2(x + 0.51f * buttonSize, y + 0.75f * buttonSize);
    glm::vec2 p2b3(x + 0.76f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p2b4(x + 0.44f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p2b5(x + 0.76f * buttonSize, y + 0.75f * buttonSize);
    // Points back 3 button.
    glm::vec2 p3b0(x + 0.43f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p3b1(x + 0.16f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p3b2(x + 0.43f * buttonSize, y + 0.75f * buttonSize);
    glm::vec2 p3b3(x + 0.63f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p3b4(x + 0.35f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p3b5(x + 0.63f * buttonSize, y + 0.75f * buttonSize);
    glm::vec2 p3b6(x + 0.82f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p3b7(x + 0.55f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p3b8(x + 0.82f * buttonSize, y + 0.75f * buttonSize);
    // Points back 4 button.
    glm::vec2 p4b0(x + 0.37f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p4b1(x + 0.16f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p4b2(x + 0.37f * buttonSize, y + 0.75f * buttonSize);
    glm::vec2 p4b3(x + 0.525f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p4b4(x + 0.31f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p4b5(x + 0.525f * buttonSize, y + 0.75f * buttonSize);
    glm::vec2 p4b6(x + 0.68f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p4b7(x + 0.47f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p4b8(x + 0.68f * buttonSize, y + 0.75f * buttonSize);
    glm::vec2 p4b9(x + 0.835f * buttonSize, y + 0.25f * buttonSize);
    glm::vec2 p4b10(x + 0.625f * buttonSize, y + 0.50f * buttonSize);
    glm::vec2 p4b11(x + 0.835f * buttonSize, y + 0.75f * buttonSize);

    if (vg) {
        nvgBeginPath(vg);
        nvgRoundedRect(vg, x, y, buttonSize, buttonSize, buttonEdgeRadius);
        nvgFillColor(vg, fillColorNvg);
        nvgFill(vg);
        nvgStrokeWidth(vg, strokeWidth);
        nvgStrokeColor(vg, textColorNvg);
        nvgStroke(vg);

        nvgLineCap(vg, NVG_ROUND);
        if (buttonType == ButtonType::CLOSE) {
            nvgBeginPath(vg);
            nvgMoveTo(vg, pc0.x, pc0.y);
            nvgLineTo(vg, pc1.x, pc1.y);
            nvgMoveTo(vg, pc2.x, pc2.y);
            nvgLineTo(vg, pc3.x, pc3.y);
            nvgStroke(vg);
        } else if (buttonType == ButtonType::BACK) {
            nvgBeginPath(vg);
            nvgMoveTo(vg, p1b0.x, p1b0.y);
            nvgLineTo(vg, p1b1.x, p1b1.y);
            nvgLineTo(vg, p1b2.x, p1b2.y);
            nvgStroke(vg);
        } else if (buttonType == ButtonType::BACK_TWO) {
            nvgBeginPath(vg);
            nvgMoveTo(vg, p2b0.x, p2b0.y);
            nvgLineTo(vg, p2b1.x, p2b1.y);
            nvgLineTo(vg, p2b2.x, p2b2.y);
            nvgMoveTo(vg, p2b3.x, p2b3.y);
            nvgLineTo(vg, p2b4.x, p2b4.y);
            nvgLineTo(vg, p2b5.x, p2b5.y);
            nvgStroke(vg);
        } else if (buttonType == ButtonType::BACK_THREE) {
            nvgBeginPath(vg);
            nvgMoveTo(vg, p3b0.x, p3b0.y);
            nvgLineTo(vg, p3b1.x, p3b1.y);
            nvgLineTo(vg, p3b2.x, p3b2.y);
            nvgMoveTo(vg, p3b3.x, p3b3.y);
            nvgLineTo(vg, p3b4.x, p3b4.y);
            nvgLineTo(vg, p3b5.x, p3b5.y);
            nvgMoveTo(vg, p3b6.x, p3b6.y);
            nvgLineTo(vg, p3b7.x, p3b7.y);
            nvgLineTo(vg, p3b8.x, p3b8.y);
            nvgStroke(vg);
        } else if (buttonType == ButtonType::BACK_FOUR) {
            nvgBeginPath(vg);
            nvgMoveTo(vg, p4b0.x, p4b0.y);
            nvgLineTo(vg, p4b1.x, p4b1.y);
            nvgLineTo(vg, p4b2.x, p4b2.y);
            nvgMoveTo(vg, p4b3.x, p4b3.y);
            nvgLineTo(vg, p4b4.x, p4b4.y);
            nvgLineTo(vg, p4b5.x, p4b5.y);
            nvgMoveTo(vg, p4b6.x, p4b6.y);
            nvgLineTo(vg, p4b7.x, p4b7.y);
            nvgLineTo(vg, p4b8.x, p4b8.y);
            nvgMoveTo(vg, p4b9.x, p4b9.y);
            nvgLineTo(vg, p4b10.x, p4b10.y);
            nvgLineTo(vg, p4b11.x, p4b11.y);
            nvgStroke(vg);
        }
        nvgLineCap(vg, NVG_BUTT);
    }
#ifdef SUPPORT_SKIA
    else if (canvas) {
        canvas->drawRoundRect(
                SkRect{x * s, y * s, (x + buttonSize) * s, (y + buttonSize) * s},
                buttonEdgeRadius * s, buttonEdgeRadius * s, fillPaint);
        canvas->drawRoundRect(
                SkRect{x * s, y * s, (x + buttonSize) * s, (y + buttonSize) * s},
                buttonEdgeRadius * s, buttonEdgeRadius * s, strokePaint);

        strokePaint.setStrokeCap(SkPaint::kRound_Cap);
        SkPath path;
        if (buttonType == ButtonType::CLOSE) {
            path.moveTo(pc0.x * s, pc0.y * s);
            path.lineTo(pc1.x * s, pc1.y * s);
            path.moveTo(pc2.x * s, pc2.y * s);
            path.lineTo(pc3.x * s, pc3.y * s);
        } else if (buttonType == ButtonType::BACK) {
            path.moveTo(p1b0.x * s, p1b0.y * s);
            path.lineTo(p1b1.x * s, p1b1.y * s);
            path.lineTo(p1b2.x * s, p1b2.y * s);
        } else if (buttonType == ButtonType::BACK_TWO) {
            path.moveTo(p2b0.x * s, p2b0.y * s);
            path.lineTo(p2b1.x * s, p2b1.y * s);
            path.lineTo(p2b2.x * s, p2b2.y * s);
            path.moveTo(p2b3.x * s, p2b3.y * s);
            path.lineTo(p2b4.x * s, p2b4.y * s);
            path.lineTo(p2b5.x * s, p2b5.y * s);
        } else if (buttonType == ButtonType::BACK_THREE) {
            path.moveTo(p3b0.x * s, p3b0.y * s);
            path.lineTo(p3b1.x * s, p3b1.y * s);
            path.lineTo(p3b2.x * s, p3b2.y * s);
            path.moveTo(p3b3.x * s, p3b3.y * s);
            path.lineTo(p3b4.x * s, p3b4.y * s);
            path.lineTo(p3b5.x * s, p3b5.y * s);
            path.moveTo(p3b6.x * s, p3b6.y * s);
            path.lineTo(p3b7.x * s, p3b7.y * s);
            path.lineTo(p3b8.x * s, p3b8.y * s);
        } else if (buttonType == ButtonType::BACK_FOUR) {
            path.moveTo(p4b0.x * s, p4b0.y * s);
            path.lineTo(p4b1.x * s, p4b1.y * s);
            path.lineTo(p4b2.x * s, p4b2.y * s);
            path.moveTo(p4b3.x * s, p4b3.y * s);
            path.lineTo(p4b4.x * s, p4b4.y * s);
            path.lineTo(p4b5.x * s, p4b5.y * s);
            path.moveTo(p4b6.x * s, p4b6.y * s);
            path.lineTo(p4b7.x * s, p4b7.y * s);
            path.lineTo(p4b8.x * s, p4b8.y * s);
            path.moveTo(p4b9.x * s, p4b9.y * s);
            path.lineTo(p4b10.x * s, p4b10.y * s);
            path.lineTo(p4b11.x * s, p4b11.y * s);
        }
        canvas->drawPath(path, strokePaint);
        strokePaint.setStrokeCap(SkPaint::kButt_Cap);
    }
#endif
#ifdef SUPPORT_VKVG
    else if (context) {
        vkvg_rounded_rectangle(context, x * s, y * s, buttonSize * s, buttonSize * s, buttonEdgeRadius * s);
        vkvg_set_source_color(context, fillColor.getColorRGBA());
        vkvg_fill_preserve(context);
        vkvg_set_line_width(context, strokeWidth * s);
        vkvg_set_source_color(context, textColor.getColorRGBA());
        vkvg_stroke(context);

        vkvg_set_line_cap(context, VKVG_LINE_CAP_ROUND);
        if (buttonType == ButtonType::CLOSE) {
            vkvg_move_to(context, pc0.x * s, pc0.y * s);
            vkvg_line_to(context, pc1.x * s, pc1.y * s);
            vkvg_move_to(context, pc2.x * s, pc2.y * s);
            vkvg_line_to(context, pc3.x * s, pc3.y * s);
            vkvg_stroke(context);
        } else if (buttonType == ButtonType::BACK) {
            vkvg_move_to(context, p1b0.x * s, p1b0.y * s);
            vkvg_line_to(context, p1b1.x * s, p1b1.y * s);
            vkvg_line_to(context, p1b2.x * s, p1b2.y * s);
            vkvg_stroke(context);
        } else if (buttonType == ButtonType::BACK_TWO) {
            vkvg_move_to(context, p2b0.x * s, p2b0.y * s);
            vkvg_line_to(context, p2b1.x * s, p2b1.y * s);
            vkvg_line_to(context, p2b2.x * s, p2b2.y * s);
            vkvg_move_to(context, p2b3.x * s, p2b3.y * s);
            vkvg_line_to(context, p2b4.x * s, p2b4.y * s);
            vkvg_line_to(context, p2b5.x * s, p2b5.y * s);
            vkvg_stroke(context);
        } else if (buttonType == ButtonType::BACK_THREE) {
            vkvg_move_to(context, p3b0.x * s, p3b0.y * s);
            vkvg_line_to(context, p3b1.x * s, p3b1.y * s);
            vkvg_line_to(context, p3b2.x * s, p3b2.y * s);
            vkvg_move_to(context, p3b3.x * s, p3b3.y * s);
            vkvg_line_to(context, p3b4.x * s, p3b4.y * s);
            vkvg_line_to(context, p3b5.x * s, p3b5.y * s);
            vkvg_move_to(context, p3b6.x * s, p3b6.y * s);
            vkvg_line_to(context, p3b7.x * s, p3b7.y * s);
            vkvg_line_to(context, p3b8.x * s, p3b8.y * s);
            vkvg_stroke(context);
        } else if (buttonType == ButtonType::BACK_FOUR) {
            vkvg_move_to(context, p4b0.x * s, p4b0.y * s);
            vkvg_line_to(context, p4b1.x * s, p4b1.y * s);
            vkvg_line_to(context, p4b2.x * s, p4b2.y * s);
            vkvg_move_to(context, p4b3.x * s, p4b3.y * s);
            vkvg_line_to(context, p4b4.x * s, p4b4.y * s);
            vkvg_line_to(context, p4b5.x * s, p4b5.y * s);
            vkvg_move_to(context, p4b6.x * s, p4b6.y * s);
            vkvg_line_to(context, p4b7.x * s, p4b7.y * s);
            vkvg_line_to(context, p4b8.x * s, p4b8.y * s);
            vkvg_move_to(context, p4b9.x * s, p4b9.y * s);
            vkvg_line_to(context, p4b10.x * s, p4b10.y * s);
            vkvg_line_to(context, p4b11.x * s, p4b11.y * s);
            vkvg_stroke(context);
        }
        vkvg_set_line_cap(context, VKVG_LINE_CAP_BUTT);
    }
#endif
}

bool DiagramButton::update(float dt, const glm::vec2& mousePosition) {
    bool isHovered = getIsHovered(mousePosition);
    ButtonState oldButtonState = buttonState;
    if (isHovered && sgl::Mouse->isButtonDown(1)) {
        buttonState = ButtonState::CLICKED;
    } else if (isHovered) {
        buttonState = ButtonState::HOVERED;
    } else {
        buttonState = ButtonState::NORMAL;
    }
    if (isHovered && sgl::Mouse->buttonReleased(1)) {
        buttonTriggered = true;
    }
    return buttonState != oldButtonState;
}

bool DiagramButton::getIsHovered(const glm::vec2& mousePosition) const {
    sgl::AABB2 buttonAabb(glm::vec2(x, y), glm::vec2(x + buttonSize, y + buttonSize));
    return buttonAabb.contains(mousePosition);
}
