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

#ifndef CORRERENDER_DIAGRAMBUTTON_HPP
#define CORRERENDER_DIAGRAMBUTTON_HPP

#include <glm/vec2.hpp>

struct NVGcontext;
typedef struct NVGcontext NVGcontext;
#ifdef SUPPORT_SKIA
class SkCanvas;
class SkPaint;
#endif
#ifdef SUPPORT_VKVG
struct _vkvg_context_t;
typedef struct _vkvg_context_t* VkvgContext;
#endif

enum class ButtonType {
    CLOSE, BACK, BACK_TWO, BACK_THREE, BACK_FOUR
};
inline bool isAnyBackButton(ButtonType bt) {
    return bt == ButtonType::BACK || bt == ButtonType::BACK_TWO
            || bt == ButtonType::BACK_THREE || bt == ButtonType::BACK_FOUR;
}

enum class ButtonState {
    NORMAL, HOVERED, CLICKED
};

class DiagramButton {
public:
    explicit DiagramButton(ButtonType buttonType);
    void setPosition(float _x, float _y);
    void setSize(float _buttonSize);

    void render(
            NVGcontext* vg,
#ifdef SUPPORT_SKIA
            SkCanvas* canvas, SkPaint* defaultPaint,
#endif
#ifdef SUPPORT_VKVG
            VkvgContext context,
#endif
            bool isDarkMode, float s);
    bool update(float dt, const glm::vec2& mousePosition);

    [[nodiscard]] bool getIsHovered(const glm::vec2& mousePosition) const;
    [[nodiscard]] inline ButtonType getButtonType() const { return buttonType; }
    [[nodiscard]] inline ButtonState getButtonState() const { return buttonState; }
    [[nodiscard]] inline bool getButtonTriggered() { bool tmp = buttonTriggered; buttonTriggered = false; return tmp; }

private:
    ButtonType buttonType = ButtonType::CLOSE;
    ButtonState buttonState = ButtonState::NORMAL;
    float x = 0.0f, y = 0.0f;
    float buttonSize = 20.0f;
    bool buttonTriggered = false;
    sgl::Color textColorUnselectedDark = sgl::Color(127, 127, 127, 255);
    sgl::Color textColorUnselectedBright = sgl::Color(127, 127, 127, 255);
    sgl::Color textColorDark = sgl::Color(255, 255, 255, 255);
    sgl::Color textColorBright = sgl::Color(0, 0, 0, 255);
    sgl::Color buttonFillColorNormalDark = sgl::Color(35, 35, 35, 255);
    sgl::Color buttonFillColorNormalBright = sgl::Color(235, 235, 235, 255);
    sgl::Color buttonFillColorHoveredDark = sgl::Color(63, 63, 63, 255);
    sgl::Color buttonFillColorHoveredBright = sgl::Color(210, 210, 210, 255);
    sgl::Color buttonFillColorClickedDark = sgl::Color(111, 111, 111, 255);
    sgl::Color buttonFillColorClickedBright = sgl::Color(174, 174, 174, 255);
};

#endif //CORRERENDER_DIAGRAMBUTTON_HPP
