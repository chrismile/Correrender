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

#ifndef CORRERENDER_TIMESERIESCORRELATIONRENDERER_HPP
#define CORRERENDER_TIMESERIESCORRELATIONRENDERER_HPP

#include <Graphics/Color.hpp>

#include "../../Renderer.hpp"
#include "Calculators/CorrelationDefines.hpp"
#include "Calculators/CorrelationMatrix.hpp"
#include "TimeSeries.hpp"

class TimeSeriesCorrelationChart;

class TimeSeriesCorrelationRenderer : public Renderer {
public:
    explicit TimeSeriesCorrelationRenderer(ViewManager* viewManager);
    ~TimeSeriesCorrelationRenderer() override;
    void initialize() override;
    [[nodiscard]] RenderingMode getRenderingMode() const override { return RENDERING_MODE_TIME_SERIES_CORRELATION; }
    [[nodiscard]] bool getIsOpaqueRenderer() const override { return false; }
    [[nodiscard]] bool getIsOverlayRenderer() const override { return true; }
    [[nodiscard]] bool getShallRenderWithoutData() const override { return true; }
    void setVolumeData(VolumeDataPtr& _volumeData, bool isNewData) override;
    void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) override;
    void update(float dt, bool isMouseGrabbed) override;
    void onHasMoved(uint32_t viewIdx) override;
    void setClearColor(const sgl::Color& clearColor) override;
    [[nodiscard]] bool getHasGrabbedMouse() const override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void renderViewImpl(uint32_t viewIdx) override;
    void renderViewPreImpl(uint32_t viewIdx) override;
    void renderViewPostOpaqueImpl(uint32_t viewIdx) override;
    void addViewImpl(uint32_t viewIdx) override;
    void removeViewImpl(uint32_t viewIdx) override;
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    void recreateDiagramSwapchain(int diagramIdx = -1);
    void renderDiagramViewSelectionGui(
            sgl::PropertyEditor& propertyEditor, const std::string& name, uint32_t& diagramViewIdx);
    bool adaptIdxOnViewRemove(uint32_t viewIdx, uint32_t& diagramViewIdx);

    void recomputeCorrelationMatrix();
    VolumeDataPtr volumeData;
    uint32_t diagramViewIdx = 0;
    bool reRenderTriggeredByDiagram = false;
    std::shared_ptr<TimeSeriesCorrelationChart> parentDiagram; //< Parent diagram.
    bool alignWithParentWindow = false;
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;
    float minCorrelationValue = 0.0f;
    float maxCorrelationValue = 1.0f;
    DiagramColorMap colorMap = DiagramColorMap::COOL_TO_WARM;

    // Time series data presets.
    void parsePresetsFile(const std::string& filename);
    std::vector<std::string> timeSeriesNames;
    std::vector<std::string> timeSeriesDataPaths;
    std::vector<std::string> timeSeriesModelPaths;
    std::string timeSeriesFilePath, modelFilePath;
    int presetIndex = 0;

    // Time series data.
    void loadTimeSeriesFromFile(const std::string& filePath);
    void loadModelFromFile(const std::string& filePath);
    TimeSeriesMetadata timeSeriesMetadata;
    TimeSeriesDataPtr timeSeriesData;
    int numWindows = -1;
    int windowLength = 32; ///< May be restricted by the metadata entry "window".
    sgl::vk::BufferPtr correlationDataBuffer; ///< Stores the calculated correlation values.
    sgl::vk::BufferPtr correlationDataStagingBuffer; ///< Stores the calculated correlation values.
    int cachedNumSamples = -1, cachedNumWindows = -1;
    int sidxRef = 0; ///< Index of reference time series.
    bool diagramDataDirty = false;
};

#endif //CORRERENDER_TIMESERIESCORRELATIONRENDERER_HPP
