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

#include <cmath>
#include <algorithm>

#include <glm/glm.hpp>

#include "ColorSpace.hpp"

/*
 * Formulas from:
 * https://en.wikipedia.org/wiki/HSL_and_HSV
 * https://de.wikipedia.org/wiki/HSV-Farbraum
 */

glm::vec3 rgbToHsv(const glm::vec3& color) {
    float minValue = std::min(color.r, std::min(color.g, color.b));
    float maxValue = std::max(color.r, std::max(color.g, color.b));

    float C = maxValue - minValue;

    // Compute the hue H.
    float H = 0.0f;
    if (maxValue == minValue) {
        H = 0.0f; //< Undefined.
    } else if (maxValue == color.r) {
        H = std::fmod((color.g - color.b) / C, 6.0f);
    } else if (maxValue == color.g) {
        H = (color.b - color.r) / C + 2.0f;
    } else if (maxValue == color.b) {
        H = (color.r - color.g) / C + 4.0f;
    }
    H *= 60.0f; //< Hue is in degrees.

    // Compute the saturation S.
    float S = 0.0f;
    if (maxValue != 0.0f) {
        S = C / maxValue;
    }

    // Compute the value V.
    float V = maxValue;

    return glm::vec3(H, S, V);
}

glm::vec3 rgbToHsl(const glm::vec3& color) {
    float minValue = std::min(color.r, std::min(color.g, color.b));
    float maxValue = std::max(color.r, std::max(color.g, color.b));

    float C = maxValue - minValue;

    // Compute the hue H.
    float H = 0.0f;
    if (maxValue == minValue) {
        H = 0.0f; //< Undefined.
    } else if (maxValue == color.r) {
        H = std::fmod((color.g - color.b) / C, 6.0f);
    } else if (maxValue == color.g) {
        H = (color.b - color.r) / C + 2.0f;
    } else if (maxValue == color.b) {
        H = (color.r - color.g) / C + 4.0f;
    }
    H *= 60.0f; //< Hue is in degrees.

    // Compute the saturation S.
    float S = 0.0f;
    if (maxValue != 0.0f) {
        S = C / (1.0f - std::abs(maxValue + minValue - 1.0f));
    }

    // Compute the lightness L.
    float L = (maxValue + minValue) * 0.5f;

    return glm::vec3(H, S, L);
}

/*
 * Formula from: https://de.wikipedia.org/wiki/HSV-Farbraum
 */
glm::vec3 hsvToRgb(const glm::vec3& color) {
    const float H = color.r;
    const float S = color.g;
    const float V = color.b;

    float h = H / 60.0f;

    int hi = int(glm::floor(h));
    float f = (h - float(hi));

    float p = V * (1.0f - S);
    float q = V * (1.0f - S * f);
    float t = V * (1.0f - S * (1.0f - f));

    if (hi == 1) {
        return glm::vec3(q, V, p);
    } else if (hi == 2) {
        return glm::vec3(p, V, t);
    } else if (hi == 3) {
        return glm::vec3(p, q, V);
    } else if (hi == 4) {
        return glm::vec3(t, p, V);
    } else if (hi == 5) {
        return glm::vec3(V, p, q);
    } else {
        return glm::vec3(V, t, p);
    }
}

/*
 * Alternative formula ("HSL to RGB alternative") from https://en.wikipedia.org/wiki/HSL_and_HSV
 */
static inline float f(const glm::vec3& color, float n) {
    float k = std::fmod(n + color.r / 30.0f, 12.0f);
    float a = color.g * std::min(color.b, 1.0f - color.b);
    return color.b - a * std::max(-1.0f, std::min(k - 3.0f, std::min(9.0f - k, 1.0f)));
}

glm::vec3 hslToRgb(const glm::vec3& color) {
    return glm::vec3(f(color, 0.0f), f(color, 8.0f), f(color, 4.0f));
}
