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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/Parallel/Reduction.hpp>
#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include "Utils/InternalState.hpp"
#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "NoiseReductionCalculator.hpp"

NoiseReductionCalculator::NoiseReductionCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
    smoothingComputePass = std::make_shared<SmoothingComputePass>(renderer);
    smoothingComputePass->setSigma(sigma);
    smoothingComputePass->setKernelSize(kernelSize);
}

std::string NoiseReductionCalculator::getOutputFieldName() {
    std::string outputFieldName = NOISE_REDUCTION_TYPE_NAMES[int(noiseReductionType)];
    if (calculatorConstructorUseCount > 1) {
        outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
    }
    return outputFieldName;
}

void NoiseReductionCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::NOISE_REDUCTION);
    }

    scalarFieldNames = {};
    scalarFieldIndexArray = {};

    std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
    for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
        if (scalarFieldNamesNew.at(i) != getOutputFieldName()) {
            scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
            scalarFieldIndexArray.push_back(i);
        }
    }

    if (isNewData) {
        scalarFieldIndex = volumeData->getStandardScalarFieldIdx();
        scalarFieldIndexGui = volumeData->getStandardScalarFieldIdx();
        volumeData->acquireScalarField(this, scalarFieldIndex);
    }
}

void NoiseReductionCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        if (scalarFieldIndex == fieldIdx) {
            scalarFieldIndex = 0;
            scalarFieldIndexGui = 0;
            volumeData->acquireScalarField(this, scalarFieldIndex);
            dirty = true;
        } else if (scalarFieldIndex > fieldIdx) {
            scalarFieldIndex--;
        }
        scalarFieldIndexGui = scalarFieldIndex;
    }
}

#define IDXK(x,y,z) ((z)*kernelSize*kernelSize + (y)*kernelSize + (x))

void NoiseReductionCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();

    VolumeData::HostCacheEntry entryScalarField = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndex), timeStepIdx, ensembleIdx);
    const float* scalarField = entryScalarField->data<float>();

    int kernelSizeHalf = kernelSize / 2;

    auto* kernelWeights = new float[kernelSize * kernelSize * kernelSize];
    for (int zi = 0; zi < kernelSize; zi++) {
        for (int yi = 0; yi < kernelSize; yi++) {
            for (int xi = 0; xi < kernelSize; xi++) {
                int x = xi - kernelSizeHalf;
                int y = yi - kernelSizeHalf;
                int z = zi - kernelSizeHalf;
                float value =
                        1.0f / (sgl::TWO_PI * sigma * sigma)
                        * std::exp(-float(x * x + y * y + z * z) / (2.0f * sigma * sigma));
                kernelWeights[IDXK(xi, yi, zi)] = value;
            }
        }
    }

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
        for (auto z = r.begin(); z != r.end(); z++) {
#else
#pragma omp parallel for shared(xs, ys, zs, kernelSizeHalf, kernelSize, kernelWeights, scalarField, buffer) default(none)
    for (int z = 0; z < zs; z++) {
#endif
        for (int y = 0; y < ys; y++) {
            for (int x = 0; x < xs; x++) {
                float sum = 0.0f;
                float weightSum = 0.0f;
                for (int zi = 0; zi < kernelSize; zi++) {
                    for (int yi = 0; yi < kernelSize; yi++) {
                        for (int xi = 0; xi < kernelSize; xi++) {
                            int xr = x + xi - kernelSizeHalf;
                            int yr = y + yi - kernelSizeHalf;
                            int zr = z + zi - kernelSizeHalf;
                            float value = std::numeric_limits<float>::quiet_NaN();
                            if (xr >= 0 && yr >= 0 && zr >= 0 && xr < xs && yr < ys && zr < zs) {
                                value = scalarField[IDXS(xr, yr, zr)];
                            }
                            if (!std::isnan(value)) {
                                float weight = kernelWeights[IDXK(xi, yi, zi)];
                                weightSum += weight;
                                sum += value * weight;
                            }
                        }
                    }
                }
                float valueOut = std::numeric_limits<float>::quiet_NaN();
                if (weightSum > 1e-5f) {
                    valueOut = sum / weightSum;
                }
                buffer[IDXS(x, y, z)] = valueOut;
            }
        }
    }
#ifdef USE_TBB
    });
#endif

    delete[] kernelWeights;
}

void NoiseReductionCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    VolumeData::DeviceCacheEntry entryScalarField = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndex), timeStepIdx, ensembleIdx);
    smoothingComputePass->setInputOutputImages(
            entryScalarField->getVulkanImageView(),
            deviceCacheEntry->getVulkanImageView());

    entryScalarField->getVulkanImage()->transitionImageLayout(VK_IMAGE_LAYOUT_GENERAL, renderer->getVkCommandBuffer());
    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
    smoothingComputePass->render();
}

void NoiseReductionCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    std::string comboName = "Scalar Field";
    if (propertyEditor.addCombo(
            comboName, &scalarFieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }

    if (noiseReductionType == NoiseReductionType::GAUSSIAN_BLUR) {
        if (propertyEditor.addSliderFloatEdit("Std. Dev.", &sigma, 0.25f, 8.0f) == ImGui::EditMode::INPUT_FINISHED) {
            smoothingComputePass->setSigma(sigma);
            dirty = true;
        }
        if (propertyEditor.addSliderIntEdit("Kernel Size", &kernelSize, 1, 7) == ImGui::EditMode::INPUT_FINISHED) {
            smoothingComputePass->setKernelSize(kernelSize);
            dirty = true;
        }
    }

    /*if (propertyEditor.addCombo(
            "Operator", (int*)&noiseReductionType,
            NOISE_REDUCTION_TYPE_NAMES, IM_ARRAYSIZE(NOISE_REDUCTION_TYPE_NAMES))) {
        hasNameChanged = true;
        dirty = true;
    }*/
}

void NoiseReductionCalculator::setSettings(const SettingsMap& settings) {
    Calculator::setSettings(settings);
    if (settings.getValueOpt("scalar_field_idx", scalarFieldIndexGui)) {
        volumeData->releaseScalarField(this, scalarFieldIndex);
        scalarFieldIndex = int(scalarFieldIndexArray.at(scalarFieldIndexGui));
        volumeData->acquireScalarField(this, scalarFieldIndex);
        dirty = true;
    }
    std::string noiseReductionTypeString;
    if (settings.getValueOpt("noise_reduction_type", noiseReductionTypeString)) {
        for (int i = 0; i < IM_ARRAYSIZE(NOISE_REDUCTION_TYPE_NAMES); i++) {
            if (noiseReductionTypeString == NOISE_REDUCTION_TYPE_NAMES[i]) {
                noiseReductionType = NoiseReductionType(i);
                break;
            }
        }
        hasNameChanged = true;
        dirty = true;
    }
    if (settings.getValueOpt("sigma", sigma)) {
        smoothingComputePass->setSigma(sigma);
        dirty = true;
    }
    if (settings.getValueOpt("kernel_size", kernelSize)) {
        smoothingComputePass->setKernelSize(kernelSize);
        dirty = true;
    }
}

void NoiseReductionCalculator::getSettings(SettingsMap& settings) {
    Calculator::getSettings(settings);
    settings.addKeyValue("scalar_field_idx", scalarFieldIndexGui);
    settings.addKeyValue("noise_reduction_type", NOISE_REDUCTION_TYPE_NAMES[int(noiseReductionType)]);
    settings.addKeyValue("sigma", sigma);
    settings.addKeyValue("kernel_size", kernelSize);
}



SmoothingComputePass::SmoothingComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void SmoothingComputePass::setInputOutputImages(
        const sgl::vk::ImageViewPtr& _inputImage,
        const sgl::vk::ImageViewPtr& _outputImage) {
    if (inputImage != _inputImage) {
        bool formatMatches = true;
        if (inputImage) {
            formatMatches =
                    getImageFormatGlslString(inputImage->getImage())
                    == getImageFormatGlslString(_inputImage->getImage());
        }
        inputImage = _inputImage;
        if (formatMatches && computeData) {
            computeData->setStaticImageView(inputImage, "inputImage");
        }
        if (!formatMatches) {
            setShaderDirty();
        }
    }
    if (outputImage != _outputImage) {
        outputImage = _outputImage;
        if (computeData) {
            computeData->setStaticImageView(outputImage, "outputImage");
        }
    }
}

void SmoothingComputePass::setSigma(float _sigma) {
    if (sigma != _sigma) {
        sigma = _sigma;
        kernelDirty = true;
    }
}

void SmoothingComputePass::setKernelSize(int _kernelSize) {
    if (kernelSize != _kernelSize) {
        kernelSize = _kernelSize;
        kernelBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeof(float) * kernelSize * kernelSize * kernelSize,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        setShaderDirty();
        kernelDirty = true;
    }
}

void SmoothingComputePass::createKernel() {
    auto* kernelWeights = new float[kernelSize * kernelSize * kernelSize];
    int kernelSizeHalf = kernelSize / 2;
    for (int zi = 0; zi < kernelSize; zi++) {
        for (int yi = 0; yi < kernelSize; yi++) {
            for (int xi = 0; xi < kernelSize; xi++) {
                int x = xi - kernelSizeHalf;
                int y = yi - kernelSizeHalf;
                int z = zi - kernelSizeHalf;
                float value =
                        1.0f / (sgl::TWO_PI * sigma * sigma)
                        * std::exp(-float(x * x + y * y + z * z) / (2.0f * sigma * sigma));
                kernelWeights[IDXK(xi, yi, zi)] = value;
            }
        }
    }
    kernelBuffer->updateData(
            sizeof(float) * kernelSize * kernelSize * kernelSize, kernelWeights,
            renderer->getVkCommandBuffer());
    delete[] kernelWeights;
    kernelDirty = false;
}

void SmoothingComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair(
            "INPUT_IMAGE_FORMAT", getImageFormatGlslString(inputImage->getImage())));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair("KERNEL_SIZE", std::to_string(kernelSize)));
    std::string shaderName = "GaussianBlur3D.Compute";
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void SmoothingComputePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticBuffer(kernelBuffer, "KernelBuffer");
    computeData->setStaticImageView(inputImage, "inputImage");
    computeData->setStaticImageView(outputImage, "outputImage");
}

void SmoothingComputePass::_render() {
    if (kernelDirty) {
        createKernel();
    }

    uniformData.xs = int(outputImage->getImage()->getImageSettings().width);
    uniformData.ys = int(outputImage->getImage()->getImageSettings().height);
    uniformData.zs = int(outputImage->getImage()->getImageSettings().depth);
    uniformData.nanValue = std::numeric_limits<float>::quiet_NaN();
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    renderer->dispatch(
            computeData,
            sgl::uiceil(uniformData.xs, computeBlockSizeX),
            sgl::uiceil(uniformData.ys, computeBlockSizeY),
            sgl::uiceil(uniformData.zs, computeBlockSizeZ));
}
