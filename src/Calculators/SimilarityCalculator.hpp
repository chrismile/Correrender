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
#include "Calculator.hpp"

class EnsembleSimilarityCalculator : public Calculator {
public:
    explicit EnsembleSimilarityCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {}
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    [[nodiscard]] bool getShouldRenderGui() const override { return true; }
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    RendererPtr getCalculatorRenderer() override { return referencePointSelectionRenderer; }
    void update(float dt) override;

protected:
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    int fieldIndex = 0, fieldIndexGui = 0;
    glm::ivec3 referencePointIndex{};
    RendererPtr referencePointSelectionRenderer;
};

/**
 * Pearson correlation coefficient (PCC) calculator.
 */
class PccCalculator : public EnsembleSimilarityCalculator {
public:
    explicit PccCalculator(sgl::vk::Renderer* renderer) : EnsembleSimilarityCalculator(renderer) {}
    std::string getOutputFieldName() override { return "Pearson Correlation"; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
};

#endif //CORRERENDER_SIMILARITYCALCULATOR_HPP
