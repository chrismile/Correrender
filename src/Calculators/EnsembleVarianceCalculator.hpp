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

#ifndef CORRERENDER_ENSEMBLEVARIANCECALCULATOR_HPP
#define CORRERENDER_ENSEMBLEVARIANCECALCULATOR_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Calculator.hpp"

class EnsembleVarianceComputePass;

class EnsembleVarianceCalculator : public Calculator {
public:
    explicit EnsembleVarianceCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::ENSEMBLE_VARIANCE; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    [[nodiscard]] bool getShouldRenderGui() const override { return true; }
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    std::string getOutputFieldName() override;
    FilterDevice getFilterDevice() override { return FilterDevice::VULKAN; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    int scalarFieldIndex = 0;
    int scalarFieldIndexGui = 0;
    std::shared_ptr<EnsembleVarianceComputePass> ensembleVarianceComputePass;
};

class EnsembleVarianceComputePass : public sgl::vk::ComputePass {
public:
    explicit EnsembleVarianceComputePass(sgl::vk::Renderer* renderer);
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews);
    void setOutputImage(const sgl::vk::ImageViewPtr& _outputImage);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    VolumeData* volumeData = nullptr;
    int cachedEnsembleMemberCount = 0;

    const uint32_t computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    struct UniformData {
        uint32_t xs, ys, zs, es;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    sgl::vk::ImageViewPtr outputImage;
};

#endif //CORRERENDER_ENSEMBLEVARIANCECALCULATOR_HPP
