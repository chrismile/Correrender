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

#ifndef CORRERENDER_RADARBARCHART_HPP
#define CORRERENDER_RADARBARCHART_HPP

#include <string>
#include <vector>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include "DiagramBase.hpp"

struct NVGcolor;

class RadarBarChart : public DiagramBase {
public:
    explicit RadarBarChart(bool equalArea = true);
    DiagramType getDiagramType() override { return DiagramType::RADAR_BAR_CHART; }
    void initialize() override;
    void update(float dt) override;

    /**
     * Sets time independent data.
     * @param variableNames The names of the variables to be displayed.
     * @param variableValues An array with dimensions: Variable.
     */
    void setDataTimeIndependent(
            const std::vector<std::string>& variableNames,
            const std::vector<float>& variableValues);

    /**
     * Sets time dependent data.
     * @param variableNames The names of the variables to be displayed.
     * @param variableValuesTimeDependent An array with dimensions: Trajectory - Variable.
     */
    void setDataTimeDependent(
            const std::vector<std::string>& variableNames,
            const std::vector<std::vector<float>>& variableValuesTimeDependent);

    inline void setUseEqualArea(bool useEqualArea) { equalArea = useEqualArea; }

protected:
    bool hasData() override {
        return useTimeDependentData ? !variableValuesTimeDependent.empty() : !variableValues.empty();
    }
    void renderBaseNanoVG() override;

private:
    glm::vec3 transferFunction(float value);
    void drawPieSlice(const glm::vec2& center, int varIdx);
    void drawEqualAreaPieSlices(const glm::vec2& center, int varIdx);
    void drawEqualAreaPieSlicesWithLabels(const glm::vec2& center);
    void drawPieSliceTextHorizontal(const NVGcolor& textColor, const glm::vec2& center, int varIdx);
    void drawPieSliceTextRotated(const NVGcolor& textColor, const glm::vec2& center, int indvarIdxex);
    void drawDashedCircle(
            const NVGcolor& circleColor, const glm::vec2& center, float radius,
            int numDashes, float dashSpaceRatio, float thickness);
    /**
     * Maps the variable index to the corresponding angle.
     * @param varIdxFloat The variable index.
     */
    float mapVarIdxToAngle(float varIdxFloat);

    enum class TextMode {
        HORIZONTAL, ROTATED
    };
    TextMode textMode = TextMode::ROTATED;
    bool useTimeDependentData = true;
    bool equalArea = true;
    bool timeStepColorMode = true;

    float chartRadius;
    float chartHoleRadius;

    // Color legend.
    const float colorLegendWidth = 16;
    const float colorLegendHeight = 160;
    const float textWidthMax = 26;

    std::vector<std::string> variableNames;
    std::vector<float> variableValues;
    std::vector<std::vector<float>> variableValuesTimeDependent;
};

#endif //CORRERENDER_RADARBARCHART_HPP
