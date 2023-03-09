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

#ifndef CORRERENDER_VELOCITYCALCULATOR_HPP
#define CORRERENDER_VELOCITYCALCULATOR_HPP

#include "Calculator.hpp"

class VelocityCalculator : public Calculator {
public:
    explicit VelocityCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::VELOCITY; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    FieldType getOutputFieldType() override { return FieldType::VECTOR; }
    std::string getOutputFieldName() override { return "Velocity"; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
};

class VectorMagnitudeCalculator : public Calculator {
public:
    explicit VectorMagnitudeCalculator(sgl::vk::Renderer* renderer, const std::string& vectorFieldName);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::VECTOR_MAGNITUDE; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    std::string getOutputFieldName() override { return magnitudeFieldName; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;

private:
    std::string vectorFieldName, magnitudeFieldName;
};

class VorticityCalculator : public Calculator {
public:
    explicit VorticityCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::VORTICITY; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    FieldType getOutputFieldType() override { return FieldType::VECTOR; }
    std::string getOutputFieldName() override { return "Vorticity"; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
};

class HelicityCalculator : public Calculator {
public:
    explicit HelicityCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::HELICITY; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    std::string getOutputFieldName() override { return "Helicity"; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
};

#endif //CORRERENDER_VELOCITYCALCULATOR_HPP
