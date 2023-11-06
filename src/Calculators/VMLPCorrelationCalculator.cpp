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

#include <utility>
#include <json/json.h>

#include <Math/Math.hpp>
#include <Utils/File/Archive.hpp>
#include <Graphics/Vulkan/Utils/DeviceThreadInfo.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "Utils/InternalState.hpp"
#include "Volume/VolumeData.hpp"
#include "VMLP/Network.hpp"
#include "VMLP/Symmetrizer.hpp"
#include "VMLPCorrelationCalculator.hpp"

struct VMLPModuleWrapper {
    Json::Value configGeneral;
    Json::Value configEncoder;
    Json::Value configDecoder;
    vmlp::ModulePtr networkEncoder;
    vmlp::ModulePtr networkDecoder;
    std::shared_ptr<vmlp::Symmetrizer> symmetrizerModule;
};

struct VMLPCacheWrapper {
    vmlp::Matrix networkInput;
    vmlp::Matrix referenceEncoded;
    vmlp::Matrix queryEncoded;
    vmlp::Matrix symmetrizedInput;
    vmlp::Matrix queryDecoded;
    AuxiliaryMemoryToken auxMemoryToken{};
};

class WriteGridPositionsBasePass : public sgl::vk::ComputePass {
public:
    explicit WriteGridPositionsBasePass(sgl::vk::Renderer* renderer);
    void setDataSize(uint32_t xs, uint32_t ys, uint32_t zs, uint32_t stride);
    void setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer);
    inline void setBatchInformation(uint32_t _batchOffset, uint32_t _batchSize) {
        batchOffset = _batchOffset;
        batchSize = _batchSize;
    }

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    std::string shaderName;

    const uint32_t BLOCK_SIZE = 256;
    uint32_t batchOffset = 0, batchSize = 0;
    struct UniformData {
        uint32_t xs, ys, zs, stride;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;
    sgl::vk::BufferPtr outputBuffer;
};

WriteGridPositionsBasePass::WriteGridPositionsBasePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void WriteGridPositionsBasePass::setDataSize(
        uint32_t xs, uint32_t ys, uint32_t zs, uint32_t stride) {
    uniformData.xs = xs;
    uniformData.ys = ys;
    uniformData.zs = zs;
    uniformData.stride = stride;
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            uniformBuffer);
}

void WriteGridPositionsBasePass::setOutputBuffer(const sgl::vk::BufferPtr& _outputBuffer) {
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void WriteGridPositionsBasePass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void WriteGridPositionsBasePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}


struct WriteGridPositionsPassPC {
    uint32_t batchOffset;
    uint32_t batchSize;
};

class WriteGridPositionsPass : public WriteGridPositionsBasePass {
public:
    explicit WriteGridPositionsPass(
            sgl::vk::Renderer* renderer) : WriteGridPositionsBasePass(renderer) {
        shaderName = "WriteGridPositions.WriteDefault.Compute";
    }

protected:
    void _render() override {
        WriteGridPositionsPassPC pc{};
        pc.batchOffset = batchOffset;
        pc.batchSize = batchSize;
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
        renderer->dispatch(computeData, sgl::uiceil(batchSize, BLOCK_SIZE), 1, 1);
    }
};

class WriteGridPositionsStencilPass : public WriteGridPositionsBasePass {
public:
    explicit WriteGridPositionsStencilPass(
            sgl::vk::Renderer* renderer) : WriteGridPositionsBasePass(renderer) {
        shaderName = "WriteGridPositions.WriteStencil.Compute";
    }
    void setNonNanIndexBuffer(const sgl::vk::BufferPtr& buffer) {
        nonNanIndexBuffer = buffer;
        if (computeData) {
            computeData->setStaticBuffer(nonNanIndexBuffer, "NonNanIndexBuffer");
        }
    }

protected:
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
        computeData->setStaticBuffer(nonNanIndexBuffer, "NonNanIndexBuffer");
    }
    void _render() override {
        WriteGridPositionsPassPC pc{};
        pc.batchOffset = batchOffset;
        pc.batchSize = batchSize;
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
        renderer->dispatch(computeData, sgl::uiceil(batchSize, BLOCK_SIZE), 1, 1);
    }

    sgl::vk::BufferPtr nonNanIndexBuffer;
};

class UnpackStencilValuesPass : public sgl::vk::ComputePass {
public:
    explicit UnpackStencilValuesPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}
    void setNonNanIndexBuffer(const sgl::vk::BufferPtr& buffer, uint32_t _numNonNanValues) {
        nonNanIndexBuffer = buffer;
        numNonNanValues = _numNonNanValues;
        if (computeData) {
            computeData->setStaticBuffer(nonNanIndexBuffer, "NonNanIndexBuffer");
        }
    }
    void setOutputBuffers(
            const sgl::vk::BufferPtr& _outputBufferStenciled, const sgl::vk::BufferPtr& _outputBuffer) {
        outputBufferStenciled = _outputBufferStenciled;
        outputBuffer = _outputBuffer;
        if (computeData) {
            computeData->setStaticBuffer(outputBufferStenciled, "OutputBufferStenciled");
            computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
        }
    }

protected:
    void loadShader() override {
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
        shaderStages = sgl::vk::ShaderManager->getShaderStages({ "UnpackStencil.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticBuffer(outputBufferStenciled, "OutputBufferStenciled");
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
        computeData->setStaticBuffer(nonNanIndexBuffer, "NonNanIndexBuffer");
    }
    void _render() override {
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, numNonNanValues);
        renderer->dispatch(computeData, sgl::uiceil(numNonNanValues, BLOCK_SIZE), 1, 1);
    }

    const uint32_t BLOCK_SIZE = 256;
    uint32_t numNonNanValues{};
    sgl::vk::BufferPtr outputBufferStenciled;
    sgl::vk::BufferPtr outputBuffer;
    sgl::vk::BufferPtr nonNanIndexBuffer;
};

class WriteGridPositionReferencePass : public WriteGridPositionsBasePass {
public:
    explicit WriteGridPositionReferencePass(
            sgl::vk::Renderer* renderer) : WriteGridPositionsBasePass(renderer) {
        shaderName = "WriteGridPositions.WriteReference.Compute";
    }
    inline void setReferencePointIndex(const glm::ivec3& _referencePointIndex) {
        referencePointIndex = _referencePointIndex;
    }

protected:
    void _render() override {
        renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, referencePointIndex);
        renderer->dispatch(computeData, 1, 1, 1);
    }
    glm::ivec3 referencePointIndex{};
};


class CopyDecoderOutputPass : public sgl::vk::ComputePass {
public:
    explicit CopyDecoderOutputPass(sgl::vk::Renderer* renderer, std::string shaderName);
    void setBuffers(const sgl::vk::BufferPtr& _inputBuffer, const sgl::vk::BufferPtr& _outputBuffer);
    inline void setBatchInformation(uint32_t _batchOffset, uint32_t _batchSize) {
        batchOffset = _batchOffset;
        batchSize = _batchSize;
    }
    void setPaddingFactor(uint32_t _paddingFactor) {
        paddingFactor = _paddingFactor;
        uniformBuffer->updateData(sizeof(uint32_t), &paddingFactor, renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                uniformBuffer);
    }
    void setFloatFormat(vmlp::FloatFormat _format);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;
    std::string shaderName;

    const uint32_t BLOCK_SIZE = 256;
    vmlp::FloatFormat format = vmlp::FloatFormat::FLOAT32;
    uint32_t batchOffset = 0, batchSize = 0;
    uint32_t paddingFactor = 0;
    sgl::vk::BufferPtr uniformBuffer;
    sgl::vk::BufferPtr inputBuffer, outputBuffer;
};

CopyDecoderOutputPass::CopyDecoderOutputPass(sgl::vk::Renderer* renderer, std::string shaderName)
        : ComputePass(renderer), shaderName(std::move(shaderName)) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(uint32_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void CopyDecoderOutputPass::setBuffers(
        const sgl::vk::BufferPtr& _inputBuffer, const sgl::vk::BufferPtr& _outputBuffer) {
    inputBuffer = _inputBuffer;
    outputBuffer = _outputBuffer;
    if (computeData) {
        computeData->setStaticBuffer(inputBuffer, "InputBuffer");
        computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
    }
}

void CopyDecoderOutputPass::setFloatFormat(vmlp::FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void CopyDecoderOutputPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void CopyDecoderOutputPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setStaticBuffer(inputBuffer, "InputBuffer");
    computeData->setStaticBuffer(outputBuffer, "OutputBuffer");
}

void CopyDecoderOutputPass::_render() {
    WriteGridPositionsPassPC pc{};
    pc.batchOffset = batchOffset;
    pc.batchSize = batchSize;
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
    renderer->dispatch(computeData, sgl::uiceil(batchSize, BLOCK_SIZE), 1, 1);
}


VMLPCorrelationCalculator::VMLPCorrelationCalculator(sgl::vk::Renderer* renderer)
        : DeepLearningCorrelationCalculator("VMLP", "vmlp", renderer) {
    sgl::vk::Device* device = renderer->getDevice();

    // e.g., 131072 for RTX 3090 (rounded up from 83968).
    auto deviceThreadInfo = sgl::getDeviceThreadInfo(device);
    srnGpuBatchSize1DBase = int(deviceThreadInfo.numCoresTotal) * 8;
    if (!sgl::isPowerOfTwo(srnGpuBatchSize1DBase)) {
        srnGpuBatchSize1DBase = sgl::nextPowerOfTwo(srnGpuBatchSize1DBase);
    }
    srnGpuBatchSize1DBase = std::clamp(srnGpuBatchSize1DBase, 256, 131072);

    bool shaderFloat16 =
            device->getPhysicalDeviceShaderFloat16Int8Features().shaderFloat16
            || device->getPhysicalDeviceVulkan12Features().shaderFloat16;
    bool storageBufferFloat16 =
            device->getPhysicalDevice16BitStorageFeatures().storageBuffer16BitAccess
            || device->getPhysicalDeviceVulkan11Features().storageBuffer16BitAccess;
    deviceSupporsFp16 = shaderFloat16 && storageBufferFloat16;
    if (deviceSupporsFp16) {
        floatFormat = vmlp::FloatFormat::FLOAT16;
    }

#ifdef VK_NV_cooperative_matrix
    if (device->getCooperativeMatrixFeaturesNV().cooperativeMatrix) {
        const auto& cooperativeMatrixProperties = device->getSupportedCooperativeMatrixPropertiesNV();
        //std::cout << "Supported modes (NV):\n" << std::endl;
        for (size_t i = 0; i < cooperativeMatrixProperties.size(); i++) {
            auto& props = cooperativeMatrixProperties[i];
            if (props.scope == VK_SCOPE_SUBGROUP_NV
                    && props.AType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && props.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && props.CType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && props.DType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && props.MSize == props.NSize && props.NSize == props.KSize
                    && props.MSize <= 16) {
                // For now, only up to 16x16 matrices are supported due to 32 byte padding.
                matrixBlockSizesNV.push_back(props.MSize);
            }
            //std::cout
            //        << "MSize: " << props.MSize
            //        << "\nNSize: " << props.NSize
            //        << "\nKSize: " << props.KSize
            //        << "\nAType: " << COMPONENT_TYPE_NAMES[int(props.AType)]
            //        << "\nBType: " << COMPONENT_TYPE_NAMES[int(props.BType)]
            //        << "\nCType: " << COMPONENT_TYPE_NAMES[int(props.CType)]
            //        << "\nDType: " << COMPONENT_TYPE_NAMES[int(props.DType)]
            //        << "\nscope: " << SCOPE_NAMES[int(props.scope)]
            //        << "\n" << std::endl;
        }
        std::sort(matrixBlockSizesNV.begin(), matrixBlockSizesNV.end());
        if (!matrixBlockSizesNV.empty()) {
            formatIdxNV = int(matrixBlockSizesNV.size()) - 1;
        }
        for (auto matrixBlockSize : matrixBlockSizesNV) {
            matrixBlockSizesStringNV.push_back(std::to_string(matrixBlockSize));
        }
    }
#endif
#ifdef VK_KHR_cooperative_matrix
    if (device->getCooperativeMatrixFeaturesKHR().cooperativeMatrix) {
        const auto& cooperativeMatrixProperties = device->getSupportedCooperativeMatrixPropertiesKHR();
        //std::cout << "Supported modes (KHR):\n" << std::endl;
        for (size_t i = 0; i < cooperativeMatrixProperties.size(); i++) {
            auto& props = cooperativeMatrixProperties[i];
            if (props.scope == VK_SCOPE_SUBGROUP_KHR
                    && props.AType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && props.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && props.CType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && props.ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && props.MSize == props.NSize && props.NSize == props.KSize
                    && props.MSize <= 16) {
                // For now, only up to 16x16 matrices are supported due to 32 byte padding.
                matrixBlockSizesKHR.push_back(props.MSize);
            }
            //std::cout
            //        << "\nMSize: " << props.MSize
            //        << "\nNSize: " << props.NSize
            //        << "\nKSize: " << props.KSize
            //        << "\nAType: " << COMPONENT_TYPE_NAMES[int(props.AType)]
            //        << "\nBType: " << COMPONENT_TYPE_NAMES[int(props.BType)]
            //        << "\nCType: " << COMPONENT_TYPE_NAMES[int(props.CType)]
            //        << "\nResultType: " << COMPONENT_TYPE_NAMES[int(props.ResultType)]
            //        << "\nsaturatingAccumulation: " << props.saturatingAccumulation
            //        << "\nscope: " << SCOPE_NAMES[int(props.scope)]
            //        << "\n" << std::endl;
        }
        std::sort(matrixBlockSizesKHR.begin(), matrixBlockSizesKHR.end());
        if (!matrixBlockSizesKHR.empty()) {
            formatIdxKHR = int(matrixBlockSizesKHR.size()) - 1;
        }
        for (auto matrixBlockSize : matrixBlockSizesKHR) {
            matrixBlockSizesStringKHR.push_back(std::to_string(matrixBlockSize));
        }
    }
#endif

    subgroupSizes.push_back(device->getPhysicalDeviceSubgroupProperties().subgroupSize);
#ifdef VK_VERSION_1_3
    if (device->getPhysicalDeviceVulkan13Features().subgroupSizeControl) {
        auto minSubgroupSize = device->getPhysicalDeviceVulkan13Properties().minSubgroupSize;
        auto maxSubgroupSize = device->getPhysicalDeviceVulkan13Properties().maxSubgroupSize;
        for (uint32_t subgroupSize = minSubgroupSize; subgroupSize < maxSubgroupSize; subgroupSize++) {
            if (subgroupSize != device->getPhysicalDeviceSubgroupProperties().subgroupSize) {
                subgroupSizes.push_back(subgroupSize);
            }
        }
    }
#endif
    for (auto subgroupSize : subgroupSizes) {
        subgroupSizesString.push_back(std::to_string(subgroupSize));
    }

    if (!matrixBlockSizesNV.empty() || !matrixBlockSizesKHR.empty()) {
        supportsFusedMlp = true;
        useFusedMlp = true;
        useKhrExtension = !matrixBlockSizesKHR.empty();
    }

    writeGridPositionsPass = std::make_shared<WriteGridPositionsPass>(renderer);
    writeGridPositionsStencilPass = std::make_shared<WriteGridPositionsStencilPass>(renderer);
    writeGridPositionReferencePass = std::make_shared<WriteGridPositionReferencePass>(renderer);
    unpackStencilValuesPass = std::make_shared<UnpackStencilValuesPass>(renderer);

    copyDecoderOutputMutualInformationPass = std::make_shared<CopyDecoderOutputPass>(
            renderer, "CopyDecoderOutput.MutualInformation.Compute");
    copyDecoderOutputCorrelationCoefficientPass = std::make_shared<CopyDecoderOutputPass>(
            renderer, "CopyDecoderOutput.CorrelationCoefficient.Compute");
    copyDecoderOutputCorrelationCoefficientAbsPass = std::make_shared<CopyDecoderOutputPass>(
            renderer, "CopyDecoderOutput.CorrelationCoefficientAbs.Compute");

    onFloatFormatChanged();
}

void VMLPCorrelationCalculator::initialize() {
    DeepLearningCorrelationCalculator::initialize();
}

VMLPCorrelationCalculator::~VMLPCorrelationCalculator() = default;

uint32_t VMLPCorrelationCalculator::getInputChannelAlignment() {
    return 1;
}

uint32_t VMLPCorrelationCalculator::getSrnStride() {
    return 3;
}

void VMLPCorrelationCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    DeepLearningCorrelationCalculator::setVolumeData(_volumeData, isNewData);
    auto xs = uint32_t(_volumeData->getGridSizeX());
    auto ys = uint32_t(_volumeData->getGridSizeY());
    auto zs = uint32_t(_volumeData->getGridSizeZ());
    auto stride = getSrnStride();
    writeGridPositionsPass->setDataSize(xs, ys, zs, stride);
    writeGridPositionsStencilPass->setDataSize(xs, ys, zs, stride);
    writeGridPositionReferencePass->setDataSize(xs, ys, zs, stride);
    if (isNewData) {
        calculatorConstructorUseCount = volumeData->getNewCalculatorUseCount(CalculatorType::VMLP);
    }
}

void VMLPCorrelationCalculator::onFloatFormatChanged() {
    if (moduleWrapper) {
        moduleWrapper->networkEncoder->setFloatFormat(floatFormat);
        moduleWrapper->networkDecoder->setFloatFormat(floatFormat);
        moduleWrapper->symmetrizerModule->setFloatFormat(floatFormat);
    }
    copyDecoderOutputMutualInformationPass->setFloatFormat(floatFormat);
    copyDecoderOutputCorrelationCoefficientPass->setFloatFormat(floatFormat);
    copyDecoderOutputCorrelationCoefficientAbsPass->setFloatFormat(floatFormat);
    //if (sgl::FileUtils::get()->exists(modelFilePath) && !sgl::FileUtils::get()->isDirectory(modelFilePath)) {
    //    loadModelFromFile(modelFilePath);
    //}
    cacheNeedsRecreate = true;
    dirty = true;
}

void VMLPCorrelationCalculator::updateMlpSettings() {
    auto networkWithEncoding = reinterpret_cast<vmlp::NetworkWithInputEncoding*>(moduleWrapper->networkEncoder.get());
    updateMlpSettingsNetwork(networkWithEncoding->getNetwork());
    updateMlpSettingsNetwork(moduleWrapper->networkDecoder);
    numLayersOutDecoder = uint32_t(moduleWrapper->networkDecoder->getNumChannelsOutPadded());
    cacheNeedsRecreate = true;
    dirty = true;
}

void VMLPCorrelationCalculator::updateMlpSettingsNetwork(const std::shared_ptr<vmlp::Module>& module) {
    auto* network = reinterpret_cast<vmlp::MlpNetwork*>(module.get());
    auto& matrixBlockSizes = useKhrExtension ? matrixBlockSizesKHR : matrixBlockSizesNV;
    auto& formatIdx = useKhrExtension ? formatIdxKHR : formatIdxNV;
    network->setUseFusedMlp(useFusedMlp);
    network->setFusedMlpExtension(useKhrExtension);
    network->setFusedMlpMatrixBlockSize(matrixBlockSizes.at(formatIdx));
    network->setFusedMlpSubgroupSize(subgroupSizes.at(subgroupSizeIdx));
    network->setFusedMlpSharedMemoryType(memoryType);
    network->checkRecreateFusedPass();
}

void VMLPCorrelationCalculator::renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) {
    DeepLearningCorrelationCalculator::renderGuiImplAdvanced(propertyEditor);

    if (supportsFusedMlp && propertyEditor.addCheckbox("Use Fused MLP", &useFusedMlp)) {
        if (floatFormat == vmlp::FloatFormat::FLOAT32) {
            floatFormat = vmlp::FloatFormat::FLOAT16;
            onFloatFormatChanged();
        }
        updateMlpSettings();
    }
    if (useFusedMlp) {
        if (!matrixBlockSizesNV.empty() && !matrixBlockSizesKHR.empty()) {
            if (propertyEditor.addCheckbox("Use KHR Extension", &useKhrExtension)) {
                updateMlpSettings();
            }
        }
        if (subgroupSizes.size() > 1 && propertyEditor.addCombo(
                "Subgroup Size", &subgroupSizeIdx, subgroupSizesString.data(), int(subgroupSizesString.size()))) {
            updateMlpSettings();
        }
        auto& matrixBlockSizesString = useKhrExtension ? matrixBlockSizesStringKHR : matrixBlockSizesStringNV;
        auto& formatIdx = useKhrExtension ? formatIdxKHR : formatIdxNV;
        if (matrixBlockSizesString.size() > 1 && propertyEditor.addCombo(
                "Matrix Block Size", &formatIdx,
                matrixBlockSizesString.data(), int(matrixBlockSizesString.size()))) {
            updateMlpSettings();
        }
        if (propertyEditor.addCombo(
                "Shared Memory Type", (int*)&memoryType,
                vmlp::FUSED_MLP_MEMORY_TYPE_NAME, IM_ARRAYSIZE(vmlp::FUSED_MLP_MEMORY_TYPE_NAME))) {
            updateMlpSettings();
        }
    }

    if (!useFusedMlp && deviceSupporsFp16 && propertyEditor.addCombo(
            "Float Format", (int*)&floatFormat,
            vmlp::FLOAT_FORMAT_UI_NAMES, IM_ARRAYSIZE(vmlp::FLOAT_FORMAT_UI_NAMES))) {
        onFloatFormatChanged();
    }
}

void VMLPCorrelationCalculator::setSettings(const SettingsMap& settings) {
    std::string floatFormatString;
    if (!useFusedMlp && deviceSupporsFp16 && settings.getValueOpt("float_format", floatFormatString)) {
        for (int i = 0; i < IM_ARRAYSIZE(vmlp::FLOAT_FORMAT_UI_NAMES); i++) {
            if (floatFormatString == vmlp::FLOAT_FORMAT_UI_NAMES[i]) {
                floatFormat = vmlp::FloatFormat(i);
                onFloatFormatChanged();
                break;
            }
        }
    }
    DeepLearningCorrelationCalculator::setSettings(settings);
}

void VMLPCorrelationCalculator::getSettings(SettingsMap& settings) {
    DeepLearningCorrelationCalculator::getSettings(settings);
    settings.addKeyValue("float_format", vmlp::FLOAT_FORMAT_UI_NAMES[int(floatFormat)]);
}

static void loadNetworkVMLP(
        sgl::vk::Renderer* renderer, std::shared_ptr<vmlp::Module>& network, const std::string& modelPath,
        vmlp::FloatFormat floatFormat, const Json::Value& config, const sgl::ArchiveEntry& entry) {
    auto* header = reinterpret_cast<NetworkParametersHeader*>(entry.bufferData.get());
    uint8_t* paramsData = entry.bufferData.get() + sizeof(NetworkParametersHeader);
    uint32_t numParams = header->numParams;

    size_t sizePerEntry = header->format == NETWORK_PARAMS_FORMAT_FLOAT ? 4 : 2;
    if (numParams * sizePerEntry + sizeof(NetworkParametersHeader) != entry.bufferSize) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetworkVMLP: Invalid number of parameters for file size.");
    }

    bool hasInputEncoding = config.isMember("encoding");
    bool isInputEncodingIdentity = false;
    auto encodingOpts = config.get("encoding", Json::Value());
    auto lossOpts = config.get("loss", Json::Value());
    auto optimizerOpts = config.get("optimizer", Json::Value());
    auto networkOpts = config.get("network", Json::Value());
    if (hasInputEncoding) {
        if (encodingOpts.get("otype", "Identity").asString() == "Identity"
                && encodingOpts.get("scale", 1.0f).asFloat() == 1.0f
                && encodingOpts.get("offset", 0.0f).asFloat() == 0.0f) {
            isInputEncodingIdentity = true;
        }
    }

    //uint32_t numInputDims = networkOpts["n_input_dims"].asUInt();
    //uint32_t numOutputDims = networkOpts["n_output_dims"].asUInt();
    if (hasInputEncoding && !isInputEncodingIdentity) {
        network = vmlp::createNetworkWithInputEncoding(renderer, encodingOpts, networkOpts);
    } else {
        network = vmlp::createNetwork(renderer, networkOpts);
    }

    // Do we need padding because the output width is not a multiple of 16?
    /*if (network->output_width() != network->padded_output_width() && network->n_params() != numParams) {
        uint32_t numNeurons = networkOpts["n_neurons"];
        uint32_t paddingSize = numNeurons * (network->padded_output_width() - network->output_width());
        size_t numParamsOld = numParams;
        numParams += paddingSize;
        const uint8_t* paramsDataOld = paramsData;
        paramsData = new uint8_t[numParams * sizePerEntry];
        memcpy(paramsData, paramsDataOld, numParamsOld * sizePerEntry);
        memset(paramsData + numParamsOld * sizePerEntry, 0, paddingSize * sizePerEntry);
    }*/

    if (network->getNumParameters() != numParams) {
        sgl::Logfile::get()->throwError(
                "Error in loadNetworkVMLP: Mismatching network parameter count (" + std::to_string(numParams)
                + " vs. " + std::to_string(network->getNumParameters()) + ") for \"" + modelPath + "\".");
    }

    if (header->format == NETWORK_PARAMS_FORMAT_FLOAT) {
        network->setParametersCpu(reinterpret_cast<float*>(paramsData), numParams);
    } else {
        sgl::Logfile::get()->throwError(
                "Error in VMLPCorrelationCalculator::loadNetworkVMLP: Half precision weights not supported so far.");
    }
    network->setFloatFormat(floatFormat);

    /*if (network->output_width() != network->padded_output_width() && network->n_params() != numParams) {
        delete[] paramsData;
    }*/
}

void VMLPCorrelationCalculator::loadModelFromFile(const std::string& modelPath) {
    moduleWrapper = std::make_shared<VMLPModuleWrapper>();
    cacheWrapper = std::make_shared<VMLPCacheWrapper>();

    std::unordered_map<std::string, sgl::ArchiveEntry> archiveFiles;
    sgl::ArchiveFileLoadReturnType retVal = sgl::loadAllFilesFromArchive(modelPath, archiveFiles, true);
    if (retVal != sgl::ArchiveFileLoadReturnType::ARCHIVE_FILE_LOAD_SUCCESSFUL) {
        sgl::Logfile::get()->writeError(
                "Error in VMLPCorrelationCalculator::loadModelFromFile: Could not load data from model \""
                + modelPath + "\".");
        return;
    }

    // A global configuration file is optional.
    auto itConfig = archiveFiles.find("config.json");
    if (itConfig != archiveFiles.end()) {
        const auto& entry = itConfig->second;

        char* bufferData = reinterpret_cast<char*>(entry.bufferData.get());
        Json::CharReaderBuilder builder;
        std::unique_ptr<Json::CharReader> const charReader(builder.newCharReader());
        JSONCPP_STRING errorString;
        if (!charReader->parse(
                bufferData, bufferData + entry.bufferSize, &moduleWrapper->configGeneral, &errorString)) {
            sgl::Logfile::get()->writeError("Error in VMLPCorrelationCalculator::loadModelFromFile: " + errorString);
            moduleWrapper = {};
            cacheWrapper = {};
            return;
        }

        auto symmetrizerTypeName = moduleWrapper->configGeneral.get(
                "symmetrizer_type", SYMMETRIZER_TYPE_SHORT_NAMES[0]).asString();
        bool foundSymmetrizerType = false;
        for (int i = 0; i < IM_ARRAYSIZE(SYMMETRIZER_TYPE_SHORT_NAMES); i++) {
            if (SYMMETRIZER_TYPE_SHORT_NAMES[i] == symmetrizerTypeName) {
                symmetrizerType = SymmetrizerType(i);
                foundSymmetrizerType = true;
                break;
            }
        }
        if (!foundSymmetrizerType) {
            sgl::Logfile::get()->writeError(
                    "Error in VMLPCorrelationCalculator::loadModelFromFile: Invalid symmetrizer type \""
                    + symmetrizerTypeName + "\".");
            return;
        }

        auto networkTypeName = moduleWrapper->configGeneral.get(
                "network_type", NETWORK_TYPE_SHORT_NAMES[0]).asString();
        bool foundNetworkType = false;
        for (int i = 0; i < IM_ARRAYSIZE(NETWORK_TYPE_SHORT_NAMES); i++) {
            if (NETWORK_TYPE_SHORT_NAMES[i] == networkTypeName) {
                networkType = NetworkType(i);
                foundNetworkType = true;
                break;
            }
        }
        if (!foundNetworkType && networkTypeName == "MINE_SRN") {
            networkType = NetworkType::SRN_MINE;
            foundNetworkType = true;
        }
        if (!foundNetworkType) {
            sgl::Logfile::get()->writeError(
                    "Error in VMLPCorrelationCalculator::loadModelFromFile: Invalid network type \""
                    + networkTypeName + "\".");
            return;
        }

        isMutualInformationData = moduleWrapper->configGeneral.get("is_mutual_information", true).asBool();
    }

    // Encoder and decoder configuration files are mandatory.
    auto itConfigEncoder = archiveFiles.find("config_encoder.json");
    auto itConfigDecoder = archiveFiles.find("config_decoder.json");
    if (itConfigEncoder != archiveFiles.end() && itConfigDecoder != archiveFiles.end()) {
        const auto& entryEncoder = itConfigEncoder->second;
        const auto& entryDecoder = itConfigDecoder->second;

        char* bufferDataEncoder = reinterpret_cast<char*>(entryEncoder.bufferData.get());
        char* bufferDataDecoder = reinterpret_cast<char*>(entryDecoder.bufferData.get());
        Json::CharReaderBuilder builder;
        std::unique_ptr<Json::CharReader> const charReader(builder.newCharReader());
        JSONCPP_STRING errorString;
        if (!charReader->parse(
                bufferDataEncoder, bufferDataEncoder + entryEncoder.bufferSize,
                &moduleWrapper->configEncoder, &errorString)) {
            sgl::Logfile::get()->writeError("Error in VMLPCorrelationCalculator::loadModelFromFile: " + errorString);
            moduleWrapper = {};
            cacheWrapper = {};
            return;
        }
        if (!charReader->parse(
                bufferDataDecoder, bufferDataDecoder + entryDecoder.bufferSize,
                &moduleWrapper->configDecoder, &errorString)) {
            sgl::Logfile::get()->writeError("Error in VMLPCorrelationCalculator::loadModelFromFile: " + errorString);
            moduleWrapper = {};
            cacheWrapper = {};
            return;
        }
    } else {
        sgl::Logfile::get()->writeError(
                "Error in VMLPCorrelationCalculator::loadModelFromFile: Could not load encoder or decoder "
                "configuration from model \"" + modelPath + "\".");
        return;
    }

    // Set input/output layer configurations for both networks.
    auto encoderNetworkOpts = moduleWrapper->configEncoder.get("network", Json::Value());
    auto decoderNetworkOpts = moduleWrapper->configDecoder.get("network", Json::Value());
    // mlp_fused_forward needs multiple of 16 for number of input layers.
    const int numInputLayers = 3;
    moduleWrapper->configEncoder["network"]["n_input_dims"] = numInputLayers;
    moduleWrapper->configDecoder["network"]["n_output_dims"] = 1;
    if (!encoderNetworkOpts.isMember("n_output_dims")) {
        moduleWrapper->configEncoder["network"]["n_output_dims"] = moduleWrapper->configEncoder["network"]["n_neurons"];
    }
    uint32_t symmetrizerFactor = symmetrizerType == SymmetrizerType::AddDiff ? 2 : 1;
    if (!decoderNetworkOpts.isMember("n_input_dims")) {
        uint32_t encoderOutputDims = moduleWrapper->configEncoder["network"].get("n_output_dims", 0).asUInt();
        moduleWrapper->configDecoder["network"]["n_input_dims"] = encoderOutputDims * symmetrizerFactor;
    }

    //const char* networkTypeName = VMLP_NETWORK_IMPLEMENTATION_NAMES[int(networkImplementation)];
    //moduleWrapper->configEncoder["network"]["otype"] = networkTypeName;
    //moduleWrapper->configDecoder["network"]["otype"] = networkTypeName;

    auto itNetworkEncoder = archiveFiles.find("network_encoder.bin");
    auto itNetworkDecoder = archiveFiles.find("network_decoder.bin");
    if (itNetworkEncoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in VMLPCorrelationCalculator::loadModelFromFile: Missing network_encoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    if (itNetworkDecoder == archiveFiles.end()) {
        sgl::Logfile::get()->writeError(
                "Error in VMLPCorrelationCalculator::loadModelFromFile: Missing network_decoder.bin in file \""
                + modelPath + "\".");
        return;
    }
    moduleWrapper->networkEncoder = {};
    moduleWrapper->networkDecoder = {};
    moduleWrapper->symmetrizerModule = {};
#if TCNN_HALF_PRECISION
    if (hasInputEncoding && !isInputEncodingIdentity) {
#endif
    loadNetworkVMLP(
            renderer, moduleWrapper->networkEncoder, modelPath, floatFormat,
            moduleWrapper->configEncoder, itNetworkEncoder->second);
    loadNetworkVMLP(
            renderer, moduleWrapper->networkDecoder, modelPath, floatFormat,
            moduleWrapper->configDecoder, itNetworkDecoder->second);
    moduleWrapper->symmetrizerModule = std::make_shared<vmlp::Symmetrizer>(renderer, symmetrizerType);
    moduleWrapper->symmetrizerModule->setFloatFormat(floatFormat);
    updateMlpSettings();

    // numLayersOutEncoder == numLayersInDecoder when symmetrizer is sum operation.
    numLayersInEncoder = uint32_t(moduleWrapper->networkEncoder->getNumChannelsIn());
    numLayersOutEncoder = uint32_t(moduleWrapper->networkEncoder->getNumChannelsOut());

    numLayersInDecoder = uint32_t(moduleWrapper->networkDecoder->getNumChannelsIn());
    numLayersOutDecoder = uint32_t(moduleWrapper->networkDecoder->getNumChannelsOutPadded());

    if (numLayersOutEncoder * symmetrizerFactor != numLayersInDecoder) {
        sgl::Logfile::get()->throwError(
                "Error in VMLPCorrelationCalculator::loadModelFromFile: Mismatch between encoder output and "
                "decoder input dimensions.");
    }

    cacheNeedsRecreate = true;
}

vmlp::Matrix VMLPCorrelationCalculator::createMatrix(size_t dim0, size_t dim1, VkBufferUsageFlags usage) {
    size_t entrySize = vmlp::FLOAT_FORMAT_SIZES_IN_BYTE[int(floatFormat)];
    auto buffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), entrySize * dim0 * dim1, usage, VMA_MEMORY_USAGE_GPU_ONLY);
    return { buffer, uint32_t(entrySize), uint32_t(dim0), uint32_t(dim1) };
}

vmlp::Matrix VMLPCorrelationCalculator::createMatrixFloat(size_t dim0, size_t dim1, VkBufferUsageFlags usage) {
    constexpr size_t entrySize = sizeof(float);
    auto buffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), entrySize * dim0 * dim1, usage, VMA_MEMORY_USAGE_GPU_ONLY);
    return { buffer, uint32_t(entrySize), uint32_t(dim0), uint32_t(dim1) };
}

void VMLPCorrelationCalculator::recreateCache(int batchSize) {
    cacheWrapper->networkInput = {};
    cacheWrapper->referenceEncoded = {};
    cacheWrapper->queryEncoded = {};
    cacheWrapper->symmetrizedInput = {};
    cacheWrapper->queryDecoded = {};
    if (cacheWrapper->auxMemoryToken) {
        volumeData->popAuxiliaryMemoryDevice(cacheWrapper->auxMemoryToken);
    }

    const uint32_t numInputLayers = 3;
    // For debug purposes: Add ", VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT".
    cacheWrapper->networkInput = createMatrixFloat(numInputLayers, batchSize);
    cacheWrapper->referenceEncoded = createMatrix(
            numLayersOutEncoder, 1, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    cacheWrapper->queryEncoded = createMatrix(
            numLayersOutEncoder, batchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    cacheWrapper->symmetrizedInput = createMatrix(numLayersInDecoder, batchSize);
    cacheWrapper->queryDecoded = createMatrix(numLayersOutDecoder, batchSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    size_t auxBuffersSizeInBytes = 0;
    auxBuffersSizeInBytes += size_t(cacheWrapper->networkInput.getBuffer()->getSizeInBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->referenceEncoded.getBuffer()->getSizeInBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryEncoded.getBuffer()->getSizeInBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->symmetrizedInput.getBuffer()->getSizeInBytes());
    auxBuffersSizeInBytes += size_t(cacheWrapper->queryDecoded.getBuffer()->getSizeInBytes());
    cacheWrapper->auxMemoryToken = volumeData->pushAuxiliaryMemoryDevice(auxBuffersSizeInBytes);

    moduleWrapper->networkEncoder->setInputOutputMatrices(cacheWrapper->networkInput, cacheWrapper->queryEncoded);
    moduleWrapper->symmetrizerModule->setInputOutputMatrices(
            cacheWrapper->referenceEncoded, cacheWrapper->queryEncoded, cacheWrapper->symmetrizedInput);
    moduleWrapper->networkDecoder->setInputOutputMatrices(cacheWrapper->symmetrizedInput, cacheWrapper->queryDecoded);

    writeGridPositionsPass->setOutputBuffer(cacheWrapper->networkInput.getBuffer());
    writeGridPositionsStencilPass->setOutputBuffer(cacheWrapper->networkInput.getBuffer());
    writeGridPositionReferencePass->setOutputBuffer(cacheWrapper->networkInput.getBuffer());
    copyDecoderOutputMutualInformationPass->setBuffers(
            cacheWrapper->queryDecoded.getBuffer(), outputImageBuffer);
    copyDecoderOutputCorrelationCoefficientPass->setBuffers(
            cacheWrapper->queryDecoded.getBuffer(), outputImageBuffer);
    copyDecoderOutputCorrelationCoefficientAbsPass->setBuffers(
            cacheWrapper->queryDecoded.getBuffer(), outputImageBuffer);

    copyDecoderOutputMutualInformationPass->setPaddingFactor(numLayersOutDecoder);
    copyDecoderOutputCorrelationCoefficientPass->setPaddingFactor(numLayersOutDecoder);
    copyDecoderOutputCorrelationCoefficientAbsPass->setPaddingFactor(numLayersOutDecoder);
}

void VMLPCorrelationCalculator::computeNanStencilBuffer() {
    std::vector<uint32_t> nonNanIndexBufferHost = DeepLearningCorrelationCalculator::computeNanStencilBufferHost();
    nonNanIndexBuffer = {};
    nonNanIndexBuffer = std::make_shared<sgl::vk::Buffer>(
            renderer->getDevice(), sizeof(uint32_t) * numNonNanValues, nonNanIndexBufferHost.data(),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
}

void VMLPCorrelationCalculator::runInferenceReference() {
    moduleWrapper->networkEncoder->setBatchSize(1);
    moduleWrapper->networkEncoder->runInference();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            cacheWrapper->queryEncoded.getBuffer());

    cacheWrapper->queryEncoded.getBuffer()->copyDataTo(
            cacheWrapper->referenceEncoded.getBuffer(), 0, 0,
            cacheWrapper->referenceEncoded.getBuffer()->getSizeInBytes(), renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            cacheWrapper->queryEncoded.getBuffer());

    // TODO
    //debugPrintBuffer(cacheWrapper->queryEncoded.getBuffer(), floatFormat, 128);
    //std::cout << std::endl;
}

void VMLPCorrelationCalculator::runInferenceBatch(uint32_t batchOffset, uint32_t batchSize) {
    moduleWrapper->networkEncoder->setBatchSize(batchSize);
    moduleWrapper->networkEncoder->runInference();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            cacheWrapper->queryEncoded.getBuffer());

    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            cacheWrapper->symmetrizedInput.getBuffer());

    //renderer->syncWithCpu();
    //auto beginNetwork = std::chrono::system_clock::now();
    moduleWrapper->symmetrizerModule->setBatchSize(batchSize);
    moduleWrapper->symmetrizerModule->runInference();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            cacheWrapper->symmetrizedInput.getBuffer());

    moduleWrapper->networkDecoder->setBatchSize(batchSize);
    moduleWrapper->networkDecoder->runInference();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            cacheWrapper->queryDecoded.getBuffer());
    //renderer->syncWithCpu();
    //auto endNetwork = std::chrono::system_clock::now();
    //auto elapsedNetwork = std::chrono::duration_cast<std::chrono::microseconds>(endNetwork - beginNetwork);
    //std::cout << "Elapsed time decoder: " << (elapsedNetwork.count() * 1e-3) << "ms" << std::endl;

    //debugPrintBuffer(cacheWrapper->networkInput.getBuffer(), vmlp::FloatFormat::FLOAT32, 16);
    //debugPrintBuffer(cacheWrapper->referenceEncoded.getBuffer(), floatFormat, 16);
    //debugPrintBuffer(cacheWrapper->queryEncoded.getBuffer(), floatFormat, 128);
    //debugPrintBuffer(cacheWrapper->symmetrizedInput.getBuffer(), floatFormat, 16);
    //debugPrintBuffer(cacheWrapper->queryDecoded.getBuffer(), floatFormat, 128);
    //std::cout << std::endl;

    std::shared_ptr<CopyDecoderOutputPass> copyDecoderOutputPass;
    if (isMutualInformationData) {
        copyDecoderOutputPass = copyDecoderOutputMutualInformationPass;
    } else if (calculateAbsoluteValue) {
        copyDecoderOutputPass = copyDecoderOutputCorrelationCoefficientAbsPass;
    } else {
        copyDecoderOutputPass = copyDecoderOutputCorrelationCoefficientPass;
    }
    copyDecoderOutputPass->setBatchInformation(batchOffset, batchSize);
    copyDecoderOutputPass->render();
}

void VMLPCorrelationCalculator::calculateDevice(
        int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    DeepLearningCorrelationCalculator::calculateDevice(timeStepIdx, ensembleIdx, deviceCacheEntry);
    if (!getIsModuleLoaded()) {
        return;
    }

    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();

    int gpuBatchSize1D = srnGpuBatchSize1DBase;

    size_t volumeDataSlice3dSize = volumeData->getSlice3dSizeInBytes(FieldType::SCALAR);
    if (cachedVolumeDataSlice3dSize != volumeDataSlice3dSize) {
        cachedVolumeDataSlice3dSize = volumeDataSlice3dSize;
        outputImageBuffer = {};
        outputImageBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), volumeDataSlice3dSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
    }
    bool recreatedUnpackBuffer = false;
    if (useDataNanStencil && cachedVolumeDataSlice3dSizeUnpacked != volumeDataSlice3dSize) {
        cachedVolumeDataSlice3dSizeUnpacked = volumeDataSlice3dSize;
        outputImageBufferUnpacked = {};
        outputImageBufferUnpacked = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), volumeDataSlice3dSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        unpackStencilValuesPass->setOutputBuffers(outputImageBuffer, outputImageBufferUnpacked);
        recreatedUnpackBuffer = true;
    }
    if (!useDataNanStencil && outputImageBufferUnpacked) {
        cachedVolumeDataSlice3dSizeUnpacked = 0;
        outputImageBufferUnpacked = {};
    }
    if (useDataNanStencil && (recreatedUnpackBuffer || !isNanStencilInitialized)) {
        const uint32_t fillValueUint = sgl::convertBitRepresentationFloatToUint32(
                std::numeric_limits<float>::quiet_NaN());
        outputImageBufferUnpacked->fill(fillValueUint, renderer->getVkCommandBuffer());
    }

    if (cacheNeedsRecreate) {
        cacheNeedsRecreate = false;
        recreateCache(srnGpuBatchSize1DBase);
    }

#ifdef TEST_INFERENCE_SPEED
    auto startInference = std::chrono::system_clock::now();
#endif

    if (useDataNanStencil && !isNanStencilInitialized) {
        computeNanStencilBuffer();
        unpackStencilValuesPass->setNonNanIndexBuffer(nonNanIndexBuffer, numNonNanValues);
        writeGridPositionsStencilPass->setNonNanIndexBuffer(nonNanIndexBuffer);
        isNanStencilInitialized = true;
    }

    writeGridPositionReferencePass->setReferencePointIndex(referencePointIndex);
    writeGridPositionReferencePass->render();
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            cacheWrapper->networkInput.getBuffer());

    runInferenceReference();

    uint32_t numSliceEntries = uint32_t(xs) * uint32_t(ys) * uint32_t(zs);
    if (useDataNanStencil) {
        numSliceEntries = numNonNanValues;
    }
    uint32_t numBatches = sgl::uiceil(numSliceEntries, uint32_t(gpuBatchSize1D));
    for (uint32_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
        uint32_t batchOffset = batchIdx * gpuBatchSize1D;
        uint32_t batchSize = std::min(uint32_t(gpuBatchSize1D), numSliceEntries - batchOffset);

        if (useDataNanStencil) {
            writeGridPositionsStencilPass->setBatchInformation(batchOffset, batchSize);
            writeGridPositionsStencilPass->render();
        } else {
            writeGridPositionsPass->setBatchInformation(batchOffset, batchSize);
            writeGridPositionsPass->render();
        }
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                cacheWrapper->networkInput.getBuffer());

        runInferenceBatch(batchOffset, batchSize);
    }

    if (useDataNanStencil) {
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                outputImageBuffer);
        unpackStencilValuesPass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                outputImageBufferUnpacked);
        deviceCacheEntry->getVulkanImage()->copyFromBuffer(outputImageBufferUnpacked, renderer->getVkCommandBuffer());
    } else {
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                outputImageBuffer);
        deviceCacheEntry->getVulkanImage()->copyFromBuffer(outputImageBuffer, renderer->getVkCommandBuffer());
    }

#ifdef TEST_INFERENCE_SPEED
    renderer->syncWithCpu();
    auto endInference = std::chrono::system_clock::now();
    auto elapsedInference = std::chrono::duration_cast<std::chrono::milliseconds>(endInference - startInference);
    std::cout << "Elapsed time inference: " << elapsedInference.count() << "ms" << std::endl;
#endif
}

