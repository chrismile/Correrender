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

#ifndef CORRERENDER_VMLPCORRELATIONCALCULATOR_HPP
#define CORRERENDER_VMLPCORRELATIONCALCULATOR_HPP

#include "Volume/Cache/AuxiliaryMemoryToken.hpp"
#include "VMLP/Format.hpp"
#include "SymmetrizerType.hpp"
#include "DeepLearningCorrelationCalculator.hpp"

struct VMLPModuleWrapper;
struct VMLPCacheWrapper;

class WriteGridPositionsPass;
class WriteGridPositionsStencilPass;
class WriteGridPositionReferencePass;
class UnpackStencilValuesPass;
class CopyDecoderOutputPass;

namespace vmlp {
class Matrix;
class Module;
}

class VMLPCorrelationCalculator : public DeepLearningCorrelationCalculator {
public:
    explicit VMLPCorrelationCalculator(sgl::vk::Renderer* renderer);
    void initialize() override;
    ~VMLPCorrelationCalculator() override;
    FilterDevice getFilterDevice() override { return FilterDevice::VULKAN; }
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::VMLP; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void computeNanStencilBuffer();
    void clearFieldDeviceData() override {}
    [[nodiscard]] bool getNeedsScalarFieldData() const override { return false; }

    void loadModelFromFile(const std::string& modelPath) override;

    // Inference steps to be implemented by subclasses.
    bool getIsModuleLoaded() override { return moduleWrapper != nullptr; }
    vmlp::Matrix createMatrix(size_t dim0, size_t dim1, VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    vmlp::Matrix createMatrixFloat(
            size_t dim0, size_t dim1, VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    void recreateCache(int batchSize);
    void runInferenceReference();
    void runInferenceBatch(uint32_t batchOffset, uint32_t batchSize);
    uint32_t getInputChannelAlignment();
    uint32_t getSrnStride();

    void renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) override;

    SymmetrizerType symmetrizerType = SymmetrizerType::Add;

    // Internal settings.
    void onFloatFormatChanged();
    bool deviceSupporsFp16 = false;
    vmlp::FloatFormat floatFormat = vmlp::FloatFormat::FLOAT32;

    // Fused MLP/cooperative matrices.
    void updateMlpSettings();
    void updateMlpSettingsNetwork(const std::shared_ptr<vmlp::Module>& module);
    bool supportsFusedMlp = false;
    bool useFusedMlp = false;
    bool useKhrExtension = false;
    std::vector<uint32_t> subgroupSizes;
    std::vector<std::string> subgroupSizesString;
    int subgroupSizeIdx = 0;
    vmlp::FusedMlpMemoryType memoryType = vmlp::FusedMlpMemoryType::FLOAT16_NO_PADDING;
    bool fusedMlpDirectLoad = true;
    bool useSharedMemoryBankSkew = true;
    int formatIdxNV = -1, formatIdxKHR = -1;
    std::vector<uint32_t> matrixBlockSizesNV, matrixBlockSizesKHR;
    std::vector<std::string> matrixBlockSizesStringNV, matrixBlockSizesStringKHR;

    /// For networkType == NetworkType::SRN_MINE.
    int srnGpuBatchSize1DBase = 131072;
    size_t cachedVolumeDataSlice3dSize = 0;

    /// NaN stencil for networkType == NetworkType::SRN_MINE.
    sgl::vk::BufferPtr nonNanIndexBuffer{};
    sgl::vk::BufferPtr outputImageBufferUnpacked{};
    size_t cachedVolumeDataSlice3dSizeUnpacked = 0;

    bool cacheNeedsRecreate = true;

    // Network & cache.
    std::shared_ptr<VMLPModuleWrapper> moduleWrapper;
    std::shared_ptr<VMLPCacheWrapper> cacheWrapper;
    uint32_t numLayersInEncoder = 0, numLayersOutEncoder = 0, numLayersInDecoder = 0, numLayersOutDecoder = 0;

    sgl::vk::BufferPtr outputImageBuffer;
    // For networkType == NetworkType::SRN_MINE.
    std::shared_ptr<WriteGridPositionsPass> writeGridPositionsPass;
    std::shared_ptr<WriteGridPositionsStencilPass> writeGridPositionsStencilPass;
    std::shared_ptr<WriteGridPositionReferencePass> writeGridPositionReferencePass;
    std::shared_ptr<UnpackStencilValuesPass> unpackStencilValuesPass;
    std::shared_ptr<CopyDecoderOutputPass> copyDecoderOutputMutualInformationPass;
    std::shared_ptr<CopyDecoderOutputPass> copyDecoderOutputCorrelationCoefficientPass;
    std::shared_ptr<CopyDecoderOutputPass> copyDecoderOutputCorrelationCoefficientAbsPass;
};

#endif //CORRERENDER_VMLPCORRELATIONCALCULATOR_HPP
