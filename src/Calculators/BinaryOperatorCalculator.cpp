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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "BinaryOperatorCalculator.hpp"

BinaryOperatorCalculator::BinaryOperatorCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
    binaryOperatorComputePass = std::make_shared<BinaryOperatorComputePass>(renderer);
    binaryOperatorComputePass->setBinaryOperatorType(binaryOperatorType);
}

std::string BinaryOperatorCalculator::getOutputFieldName() {
    std::string outputFieldName = BINARY_OPERATOR_NAMES[int(binaryOperatorType)];
    if (calculatorConstructorUseCount > 1) {
        outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
    }
    return outputFieldName;
}

void BinaryOperatorCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::BINARY_OPERATOR);

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
        for (int i = 0; i < 2; i++) {
            scalarFieldIndices[i] = volumeData->getStandardScalarFieldIdx();
            scalarFieldIndicesGui[i] = volumeData->getStandardScalarFieldIdx();
            volumeData->acquireScalarField(this, scalarFieldIndices[i]);
        }
    }
}

void BinaryOperatorCalculator::onFieldRemoved(FieldType fieldType, int fieldIdx) {
    if (fieldType == FieldType::SCALAR) {
        for (int i = 0; i < 2; i++) {
            if (scalarFieldIndices[i] == fieldIdx) {
                scalarFieldIndices[i] = 0;
                scalarFieldIndicesGui[i] = 0;
                volumeData->acquireScalarField(this, scalarFieldIndices[i]);
                dirty = true;
            } else if (scalarFieldIndices[i] > fieldIdx) {
                scalarFieldIndices[i]--;
            }
            scalarFieldIndicesGui[i] = scalarFieldIndices[i];
        }
    }
}

void BinaryOperatorCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();

    VolumeData::HostCacheEntry entryScalarField0 = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndices[0]), timeStepIdx, ensembleIdx);
    float* scalarField0 = entryScalarField0.get();
    VolumeData::HostCacheEntry entryScalarField1 = volumeData->getFieldEntryCpu(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndices[1]), timeStepIdx, ensembleIdx);
    float* scalarField1 = entryScalarField1.get();

    std::function<float(float val0, float val1)> binaryOperator;
    if (binaryOperatorType == BinaryOperatorType::IDENTITY_1) {
        binaryOperator = [](float val0, float val1) { return val0; };
    } else if (binaryOperatorType == BinaryOperatorType::IDENTITY_2) {
        binaryOperator = [](float val0, float val1) { return val1; };
    } else if (binaryOperatorType == BinaryOperatorType::SUM) {
        binaryOperator = [](float val0, float val1) { return val0 + val1; };
    } else if (binaryOperatorType == BinaryOperatorType::DIFFERENCE) {
        binaryOperator = [](float val0, float val1) { return val0 - val1; };
    } else if (binaryOperatorType == BinaryOperatorType::ABSOLUTE_DIFFERENCE) {
        binaryOperator = [](float val0, float val1) { return std::abs(val0 - val1); };
    } else if (binaryOperatorType == BinaryOperatorType::PRODUCT) {
        binaryOperator = [](float val0, float val1) { return val0 * val1; };
    } else if (binaryOperatorType == BinaryOperatorType::MAX) {
        binaryOperator = [](float val0, float val1) { return std::max(val0, val1); };
    } else if (binaryOperatorType == BinaryOperatorType::MIN) {
        binaryOperator = [](float val0, float val1) { return std::min(val0, val1); };
    }

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
        for (auto z = r.begin(); z != r.end(); z++) {
#else
    #pragma omp parallel for shared(xs, ys, zs, scalarField0, scalarField1, binaryOperator, buffer) default(none)
    for (int z = 0; z < zs; z++) {
#endif
        for (int y = 0; y < ys; y++) {
            for (int x = 0; x < xs; x++) {
                buffer[IDXS(x, y, z)] = binaryOperator(scalarField0[IDXS(x, y, z)], scalarField1[IDXS(x, y, z)]);
            }
        }
    }
#ifdef USE_TBB
    });
#endif
}

void BinaryOperatorCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    VolumeData::DeviceCacheEntry entryScalarField0 = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndices[0]), timeStepIdx, ensembleIdx);
    VolumeData::DeviceCacheEntry entryScalarField1 = volumeData->getFieldEntryDevice(
            FieldType::SCALAR, scalarFieldNames.at(scalarFieldIndices[1]), timeStepIdx, ensembleIdx);
    binaryOperatorComputePass->setInputOutputImages(
            entryScalarField0->getVulkanImageView(),
            entryScalarField1->getVulkanImageView(),
            deviceCacheEntry->getVulkanImageView());

    entryScalarField0->getVulkanImage()->transitionImageLayout(VK_IMAGE_LAYOUT_GENERAL, renderer->getVkCommandBuffer());
    entryScalarField1->getVulkanImage()->transitionImageLayout(VK_IMAGE_LAYOUT_GENERAL, renderer->getVkCommandBuffer());
    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
    binaryOperatorComputePass->render();
}

void BinaryOperatorCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    for (int i = 0; i < 2; i++) {
        std::string comboName = "Scalar Field #" + std::to_string(i);
        if (propertyEditor.addCombo(
                comboName, &scalarFieldIndicesGui[i], scalarFieldNames.data(), int(scalarFieldNames.size()))) {
            volumeData->releaseScalarField(this, scalarFieldIndices[i]);
            scalarFieldIndices[i] = int(scalarFieldIndexArray.at(scalarFieldIndicesGui[i]));
            volumeData->acquireScalarField(this, scalarFieldIndices[i]);
            dirty = true;
        }
    }

    if (propertyEditor.addCombo(
            "Operator", (int*)&binaryOperatorType, BINARY_OPERATOR_NAMES, IM_ARRAYSIZE(BINARY_OPERATOR_NAMES))) {
        binaryOperatorComputePass->setBinaryOperatorType(binaryOperatorType);
        hasNameChanged = true;
        dirty = true;
    }
}



BinaryOperatorComputePass::BinaryOperatorComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void BinaryOperatorComputePass::setInputOutputImages(
        const sgl::vk::ImageViewPtr& _inputImage0,
        const sgl::vk::ImageViewPtr& _inputImage1,
        const sgl::vk::ImageViewPtr& _outputImage) {
    if (inputImage0 != _inputImage0) {
        inputImage0 = _inputImage0;
        if (computeData) {
            computeData->setStaticImageView(inputImage0, "inputImage0");
        }
    }
    if (inputImage1 != _inputImage1) {
        inputImage1 = _inputImage1;
        if (computeData) {
            computeData->setStaticImageView(inputImage1, "inputImage1");
        }
    }
    if (outputImage != _outputImage) {
        outputImage = _outputImage;
        if (computeData) {
            computeData->setStaticImageView(outputImage, "outputImage");
        }
    }
}

void BinaryOperatorComputePass::setBinaryOperatorType(BinaryOperatorType _binaryOperatorType) {
    if (binaryOperatorType != _binaryOperatorType) {
        binaryOperatorType = _binaryOperatorType;
        setShaderDirty();
    }
}

void BinaryOperatorComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    if (binaryOperatorType == BinaryOperatorType::IDENTITY_1) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "(x)"));
    } else if (binaryOperatorType == BinaryOperatorType::IDENTITY_2) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "(y)"));
    } else if (binaryOperatorType == BinaryOperatorType::SUM) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "((x) + (y))"));
    } else if (binaryOperatorType == BinaryOperatorType::DIFFERENCE) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "((x) - (y))"));
    } else if (binaryOperatorType == BinaryOperatorType::ABSOLUTE_DIFFERENCE) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "(abs((x) - (y)))"));
    } else if (binaryOperatorType == BinaryOperatorType::PRODUCT) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "((x) * (y))"));
    } else if (binaryOperatorType == BinaryOperatorType::MAX) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "(max((x), (y)))"));
    } else if (binaryOperatorType == BinaryOperatorType::MIN) {
        preprocessorDefines.insert(std::make_pair("binaryOperator(x, y)", "(min((x), (y)))"));
    }
    std::string shaderName = "BinaryOperatorCalculator.Compute";
    //if (correlationMeasureType == CorrelationMeasureType::PEARSON) {
    //    shaderName = "PearsonCorrelation.Compute";
    //} else if (correlationMeasureType == CorrelationMeasureType::SPEARMAN) {
    //    shaderName = "SpearmanRankCorrelation.Compute";
    //} else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED) {
    //    shaderName = "MutualInformationBinned.Compute";
    //} else if (correlationMeasureType == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV) {
    //    shaderName = "MutualInformationKraskov.Compute";
    //}
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void BinaryOperatorComputePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticImageView(inputImage0, "inputImage0");
    computeData->setStaticImageView(inputImage1, "inputImage1");
    computeData->setStaticImageView(outputImage, "outputImage");
}

void BinaryOperatorComputePass::_render() {
    uniformData.xs = outputImage->getImage()->getImageSettings().width;
    uniformData.ys = outputImage->getImage()->getImageSettings().height;
    uniformData.zs = outputImage->getImage()->getImageSettings().depth;
    uniformData.es = 0;
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