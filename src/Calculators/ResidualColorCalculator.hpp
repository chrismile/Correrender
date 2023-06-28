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

#ifndef CORRERENDER_RESIDUALCOLORCALCULATOR_HPP
#define CORRERENDER_RESIDUALCOLORCALCULATOR_HPP

#include <vector>

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Calculator.hpp"

class ResidualColorComputePass;

class ResidualColorCalculator : public Calculator {
public:
    explicit ResidualColorCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::RESIDUAL_COLOR; }
    [[nodiscard]] bool getUseTransferFunction() const override { return true; }
    [[nodiscard]] bool getUsesScalarFieldIdx(int fieldIdx) const override {
        return fieldIdx == scalarFieldIndices[0] || fieldIdx == scalarFieldIndices[1];
    }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    [[nodiscard]] bool getShouldRenderGui() const override { return true; }
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    std::string getOutputFieldName() override;
    FilterDevice getFilterDevice() override { return FilterDevice::VULKAN; }
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    int scalarFieldIndices[2] = { 0, 0 };
    int scalarFieldIndicesGui[2] = { 0, 0 };
    std::shared_ptr<ResidualColorComputePass> residualColorComputePass;
};

class ResidualColorComputePass : public sgl::vk::ComputePass {
public:
    explicit ResidualColorComputePass(sgl::vk::Renderer* renderer);
    void setVolumeData(VolumeData* _volumeData);
    void setInputOutputImages(
            int fieldIndex0, int fieldIndex1,
            const sgl::vk::ImageViewPtr& _inputImage0,
            const sgl::vk::ImageViewPtr& _inputImage1,
            const sgl::vk::ImageViewPtr& _outputImage);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    struct UniformData {
        uint32_t xs, ys, zs, es;
        uint32_t fieldIndex0, fieldIndex1, padding0, padding1;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    VolumeData* volumeData = nullptr;
    sgl::vk::ImageViewPtr inputImage0, inputImage1, outputImage;
};

#endif //CORRERENDER_RESIDUALCOLORCALCULATOR_HPP
