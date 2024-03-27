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

#ifndef CORRERENDER_SCATTERPLOTCHART_HPP
#define CORRERENDER_SCATTERPLOTCHART_HPP

#include "../DiagramBase.hpp"

class HostCacheEntryType;
typedef std::shared_ptr<HostCacheEntryType> HostCacheEntry;
class VolumeData;
typedef std::shared_ptr<VolumeData> VolumeDataPtr;

class ScatterPlotChart : public DiagramBase {
public:
    ScatterPlotChart();
    ~ScatterPlotChart() override;
    DiagramType getDiagramType() override { return DiagramType::SCATTER_PLOT; }
    void initialize() override;
    void update(float dt) override;
    void updateSizeByParent() override;
    void setAlignWithParentWindow(bool _align);
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData);
    void setIsEnsembleMode(bool _isEnsembleMode);
    void setField0(int _fieldIdx0, const std::string& _fieldName0);
    void setField1(int _fieldIdx1, const std::string& _fieldName1);
    void setReferencePoints(const glm::ivec3& pt0, const glm::ivec3& pt1);
    void setPointColor(const sgl::Color& _pointColor);
    void setPointRadius(float _pointRadius);
    void setUseGlobalMinMax(bool _useGlobalMinMax);

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
    VolumeDataPtr volumeData;
    bool dataDirty = true;

    int getCorrelationMemberCount();
    HostCacheEntry getFieldEntryCpu(const std::string& fieldName, int fieldIdx);
    std::pair<float, float> getMinMaxScalarFieldValue(const std::string& fieldName, int fieldIdx);
    bool isEnsembleMode = true; //< Ensemble or time mode?

    void updateData();

    // GUI data.
    int cachedTimeStepIdx = -1, cachedEnsembleIdx = -1;
    bool alignWithParentWindow = false;
    int fieldIdx0 = 0, fieldIdx1 = 1;
    std::string fieldName0, fieldName1;
    float totalRadius = 0.0f;
    sgl::Color pointColor = sgl::Color(31, 119, 180);
    float pointRadius = 5.0f;
    float strokeWidth = 1.5f;
    glm::ivec3 refPoint0{}, refPoint1{};
    std::vector<float> values0, values1;
    bool useGlobalMinMax = false;
    bool isMinMaxDirty = true;
    std::pair<float, float> minMax0{}, minMax1{};
    float offsetPct = 0.01f;
    std::pair<float, float> minMaxOff0{}, minMaxOff1{};

    sgl::Color textColorDark = sgl::Color(255, 255, 255, 255);
    sgl::Color textColorBright = sgl::Color(0, 0, 0, 255);
};

#endif //CORRERENDER_SCATTERPLOTCHART_HPP
