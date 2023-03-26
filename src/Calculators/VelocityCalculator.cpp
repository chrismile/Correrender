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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include "Loaders/DataSet.hpp"
#include "Loaders/LoadersUtil.hpp"
#include "Volume/VolumeData.hpp"
#include "VelocityCalculator.hpp"

VelocityCalculator::VelocityCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
}

void VelocityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::VELOCITY);
    }
}

void VelocityCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    bool uLowerCaseVariableExists = volumeData->getFieldExists(FieldType::SCALAR, "u");
    bool vLowerCaseVariableExists = volumeData->getFieldExists(FieldType::SCALAR, "v");
    bool wLowerCaseVariableExists = volumeData->getFieldExists(FieldType::SCALAR, "w");
    bool uUpperCaseVariableExists = volumeData->getFieldExists(FieldType::SCALAR, "U");
    bool vUpperCaseVariableExists = volumeData->getFieldExists(FieldType::SCALAR, "V");
    bool wUpperCaseVariableExists = volumeData->getFieldExists(FieldType::SCALAR, "W");

    std::string velocityFieldNameX, velocityFieldNameY, velocityFieldNameZ;
    if (uLowerCaseVariableExists && vLowerCaseVariableExists && wLowerCaseVariableExists) {
        velocityFieldNameX = "u";
        velocityFieldNameY = "v";
        velocityFieldNameZ = "w";
    } else if (uUpperCaseVariableExists && vUpperCaseVariableExists && wUpperCaseVariableExists) {
        velocityFieldNameX = "U";
        velocityFieldNameY = "V";
        velocityFieldNameZ = "W";
    } else {
        sgl::Logfile::get()->throwError(
                "Error in VelocityCalculator::calculateCpu: Could not find u, v, w (or U, V, W) wind speeds.");
    }

    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();

    VolumeData::HostCacheEntry entryVelocityX = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, velocityFieldNameX, timeStepIdx, ensembleIdx);
    const float* velocityX = entryVelocityX->data<float>();
    VolumeData::HostCacheEntry entryVelocityY = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, velocityFieldNameY, timeStepIdx, ensembleIdx);
    const float* velocityY = entryVelocityY->data<float>();
    VolumeData::HostCacheEntry entryVelocityZ = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, velocityFieldNameZ, timeStepIdx, ensembleIdx);
    const float* velocityZ = entryVelocityZ->data<float>();

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
        for (auto z = r.begin(); z != r.end(); z++) {
#else
    #pragma omp parallel for shared(xs, ys, zs, velocityX, velocityY, velocityZ, buffer) default(none)
    for (int z = 0; z < zs; z++) {
#endif
        for (int y = 0; y < ys; y++) {
            for (int x = 0; x < xs; x++) {
                buffer[IDXV(x, y, z, 0)] = velocityX[IDXS(x, y, z)];
                buffer[IDXV(x, y, z, 1)] = velocityY[IDXS(x, y, z)];
                buffer[IDXV(x, y, z, 2)] = velocityZ[IDXS(x, y, z)];
            }
        }
    }
#ifdef USE_TBB
    });
#endif
}

VectorMagnitudeCalculator::VectorMagnitudeCalculator(sgl::vk::Renderer* renderer, const std::string& vectorFieldName)
        : Calculator(renderer), vectorFieldName(vectorFieldName),
          magnitudeFieldName(vectorFieldName + " Magnitude") {
}

void VectorMagnitudeCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::VECTOR_MAGNITUDE);
    }
}

void VectorMagnitudeCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    VolumeData::HostCacheEntry entryVectorField = volumeData->getFieldEntryCpu(
            FieldType::VECTOR, vectorFieldName, timeStepIdx, ensembleIdx);
    computeVectorMagnitudeField(
            entryVectorField->data<float>(), buffer,
            volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ());
}

VorticityCalculator::VorticityCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
}

void VorticityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::VORTICITY);
    }
}

void VorticityCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    VolumeData::HostCacheEntry entryVelocity = volumeData->getFieldEntryCpu(
            FieldType::VECTOR, "Velocity", timeStepIdx, ensembleIdx);
    computeVorticityField(
            entryVelocity->data<float>(), buffer,
            volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ(),
            volumeData->getDx(), volumeData->getDy(), volumeData->getDz());
}

HelicityCalculator::HelicityCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
}

void HelicityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::HELICITY);
    }
}

void HelicityCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    VolumeData::HostCacheEntry entryVelocity = volumeData->getFieldEntryCpu(
            FieldType::VECTOR, "Velocity", timeStepIdx, ensembleIdx);
    VolumeData::HostCacheEntry entryVorticity = volumeData->getFieldEntryCpu(
            FieldType::VECTOR, "Vorticity", timeStepIdx, ensembleIdx);
    computeHelicityField(
            entryVelocity->data<float>(), entryVorticity->data<float>(), buffer,
            volumeData->getGridSizeX(), volumeData->getGridSizeY(), volumeData->getGridSizeZ());
}
