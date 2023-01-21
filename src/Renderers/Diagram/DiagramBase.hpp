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
#include <functional>

#include <Graphics/Vulkan/libs/nanovg/NanoVGWidget.hpp>

struct NVGcolor;

enum class DiagramType {
    RADAR_CHART, RADAR_BAR_CHART, HORIZON_GRAPH
};

class DiagramBase : public sgl::NanoVGWidget {
public:
    DiagramBase();
    virtual void initialize();
    virtual DiagramType getDiagramType()=0;
    virtual void renderImGui() {}

    [[nodiscard]] inline bool getSelectedVariablesChanged() const { return selectedVariablesChanged; };
    [[nodiscard]] inline const std::set<size_t>& getSelectedVariableIndices() const { return selectedVariableIndices; };
    inline void getSelectedVariableIndices(const std::set<size_t>& newSelectedVariableIndices) {
        selectedVariableIndices = newSelectedVariableIndices;
    };

protected:
    virtual bool hasData()=0;

    void renderBase() override;

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

    float borderSizeX = 0, borderSizeY = 0;
    const float borderWidth = 1.0f;
    const float borderRoundingRadius = 4.0f;
    float backgroundOpacity = 1.0f;
    float textSizeLegend = 12.0f;

    // Variables can be selected by clicking on them.
    size_t numVariables = 0;
    std::set<size_t> selectedVariableIndices;
    bool selectedVariablesChanged = false;
};

#endif //CORRERENDER_DIAGRAMBASE_HPP