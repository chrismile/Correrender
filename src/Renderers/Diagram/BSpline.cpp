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

#include "BSpline.hpp"

static float B(int i, int k, float x, const std::vector<float>& t) {
    if (k == 1) {
        if (t[i] <= x && x < t[i + 1]) {
            return 1;
        } else {
            return 0;
        }
    } else {
        float left;
        float denomLeft = t[i + k - 1] - t[i];
        if (denomLeft > 1e-6f) {
            left = (x - t[i]) / denomLeft * B(i, k - 1, x, t);
        } else {
            left = 0.0f;
        }
        float right;
        float denomRight = t[i + k] - t[i + 1];
        if (denomRight > 1e-6f) {
            right = (t[i + k] - x) / denomRight * B(i + 1, k - 1, x, t);
        } else {
            right = 0.0f;
        }
        return left + right;
    }
}

glm::vec2 evaluateBSpline(float x, int k, const std::vector<glm::vec2>& controlPoints) {
    auto numControlPoints = int(controlPoints.size());
    std::vector<float> t(k + numControlPoints);
    for (int i = 0; i < k - 1; i++) {
        t[i] = 0.0f;
        t[int(t.size()) - i - 1] = 1.0f;
    }
    int numMiddle = numControlPoints - k + 2;
    for (int i = 0; i < numMiddle; i++) {
        t[i + k - 1] = float(i) / float(numMiddle - 1);
    }
    if (x == 1) {
        x -= 1e-5f;
    }
    glm::vec2 sum(0.0f);
    for (int i = 0; i < numControlPoints; i++) {
        float B_val = B(i, k, x, t);
        sum.x += B_val * controlPoints[i].x;
        sum.y += B_val * controlPoints[i].y;
    }
    return sum;
}
