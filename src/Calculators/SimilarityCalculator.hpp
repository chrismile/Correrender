/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#ifndef CORRERENDER_SIMILARITYCALCULATOR_HPP
#define CORRERENDER_SIMILARITYCALCULATOR_HPP

#include <vector>
#include <glm/vec3.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Calculator.hpp"

class ReferencePointSelectionRenderer;

class EnsembleSimilarityCalculator : public Calculator {
public:
    explicit EnsembleSimilarityCalculator(sgl::vk::Renderer* renderer);
    void setViewManager(ViewManager* _viewManager) override;
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    [[nodiscard]] virtual bool getIsRealtime() const { return false; }
    [[nodiscard]] bool getShouldRenderGui() const override { return true; }
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    [[nodiscard]] bool getHasFixedRange() const override { return false; }
    RendererPtr getCalculatorRenderer() override { return calculatorRenderer; }
    void update(float dt) override;

protected:
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

    ViewManager* viewManager;
    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    int fieldIndex = 0, fieldIndexGui = 0;
    glm::ivec3 referencePointIndex{};
    RendererPtr calculatorRenderer;
    ReferencePointSelectionRenderer* referencePointSelectionRenderer;
};


class PccComputePass;

enum class CorrelationMeasureType {
    PEARSON, SPEARMAN, KENDALL, MUTUAL_INFORMATION_BINNED, MUTUAL_INFORMATION_KRASKOV
};
const char* const CORRELATION_MEASURE_TYPE_NAMES[] = {
        "Pearson", "Spearman", "Kendall", "Mutual Information (Binned)", "Mutual Information (Kraskov)"
};

/**
 * Pearson correlation coefficient (PCC) calculator.
 */
class PccCalculator : public EnsembleSimilarityCalculator {
public:
    explicit PccCalculator(sgl::vk::Renderer* renderer);
    std::string getOutputFieldName() override {
        return std::string(CORRELATION_MEASURE_TYPE_NAMES[int(correlationMeasureType)]) + " Correlation";
    }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    [[nodiscard]] bool getIsRealtime() const override { return useGpu; }
    FilterDevice getFilterDevice() override;
    [[nodiscard]] bool getHasFixedRange() const override {
        return correlationMeasureType != CorrelationMeasureType::MUTUAL_INFORMATION_BINNED;
    }
    [[nodiscard]] std::pair<float, float> getFixedRange() const override { return std::make_pair(-1.0f, 1.0f); }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

protected:
    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    std::shared_ptr<PccComputePass> pccComputePass;
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;
    bool useGpu = true;
    int numBins = 40; ///< For CorrelationMeasureType::MUTUAL_INFORMATION_BINNED.
};

class SpearmanReferenceRankComputePass;

class PccComputePass : public sgl::vk::ComputePass {
public:
    explicit PccComputePass(sgl::vk::Renderer* renderer);
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews);
    void setOutputImage(const sgl::vk::ImageViewPtr& _outputImage);
    void setReferencePoint(const glm::ivec3& referencePointIndex);
    void setCorrelationMeasureType(CorrelationMeasureType _correlationMeasureType);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    VolumeData* volumeData = nullptr;
    int cachedEnsembleMemberCount = 0;
    CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;

    const int computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    struct UniformData {
        uint32_t xs, ys, zs, es;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    sgl::vk::ImageViewPtr outputImage;

    // For Spearman correlation.
    std::shared_ptr<SpearmanReferenceRankComputePass> spearmanReferenceRankComputePass;
    sgl::vk::BufferPtr referenceRankBuffer;
};

class SpearmanReferenceRankComputePass : public sgl::vk::ComputePass {
public:
    SpearmanReferenceRankComputePass(sgl::vk::Renderer* renderer, sgl::vk::BufferPtr uniformBuffer);
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void setEnsembleMemberCount(int ensembleMemberCount);
    void setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews);
    inline const sgl::vk::BufferPtr& getReferenceRankBuffer() { return referenceRankBuffer; }

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    VolumeData* volumeData = nullptr;
    int cachedEnsembleMemberCount = 0;

    const int computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    sgl::vk::BufferPtr uniformBuffer;

    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    sgl::vk::BufferPtr referenceRankBuffer;
};

#endif //CORRERENDER_SIMILARITYCALCULATOR_HPP
