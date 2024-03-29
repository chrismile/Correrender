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

#ifndef CORRERENDER_DISTRIBUTIONSIMILARITYCHART_HPP
#define CORRERENDER_DISTRIBUTIONSIMILARITYCHART_HPP

#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/CorrelationMatrix.hpp"
#include "Renderers/Diagram/DiagramColorMap.hpp"
#include "../DiagramBase.hpp"

class HostCacheEntryType;
typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
class VolumeData;
typedef std::shared_ptr<VolumeData> VolumeDataPtr;

class DistributionSimilarityChart : public DiagramBase {
public:
    DistributionSimilarityChart();
    ~DistributionSimilarityChart() override;
    DiagramType getDiagramType() override { return DiagramType::CORRELATION_MATRIX; }
    void initialize() override;
    void update(float dt) override;
    void updateSizeByParent() override;
    void setAlignWithParentWindow(bool _align);
    void setPointColor(const sgl::Color& _pointColor);
    void setPointRadius(float _pointRadius);
    void setPointData(const std::vector<glm::vec2>& _pointData);
    void setClusterData(const std::vector<std::vector<size_t>>& _clusterData);
    void setBoundingBox(const sgl::AABB2& _bb);
    [[nodiscard]] inline int getSelectedPointIdx() const { return selectedPointIdx; }

protected:
    bool hasData() override {
        return true;
    }
    void renderBaseNanoVG() override;
#ifdef SUPPORT_SKIA
    void renderBaseSkia() override;
#endif
#ifdef SUPPORT_VKVG
    void renderBaseVkvg() override;
#endif

    void renderScatterPlot();
    void onUpdatedWindowSize() override;

private:
    bool dataDirty = true;
    void updateData();

    // GUI data.
    bool alignWithParentWindow = false;
    float ox = 0, oy = 0, dw = 0, dh = 0;
    sgl::Color pointColor = sgl::Color(31, 119, 180);
    sgl::Color pointColorGreyDark = sgl::Color(115, 115, 115);
    sgl::Color pointColorGreyBright = sgl::Color(175, 175, 175);
    sgl::Color hoveredPointColor = sgl::Color(240, 40, 10);
    float pointRadius = 5.0f;
    float strokeWidth = 1.5f;
    std::vector<glm::vec2> pointData;
    std::vector<std::vector<size_t>> clusterData;
    std::vector<int> pointToClusterArray;
    sgl::AABB2 bb;
    int hoveredPointIdx = -1;
    int clickedPointIdx = -1;
    int selectedPointIdx = -1;

    sgl::Color textColorDark = sgl::Color(255, 255, 255, 255);
    sgl::Color textColorBright = sgl::Color(0, 0, 0, 255);
};

#endif //CORRERENDER_DISTRIBUTIONSIMILARITYCHART_HPP
