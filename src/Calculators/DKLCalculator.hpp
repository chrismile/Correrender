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

#ifndef CORRERENDER_DKLCALCULATOR_HPP
#define CORRERENDER_DKLCALCULATOR_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Calculator.hpp"
#include "CorrelationDefines.hpp"

class DKLComputePass;

enum class DKLEstimatorType {
    BINNED, ENTROPY_KNN
};
const char* const DKL_ESTIMATOR_TYPE_NAMES[] = {
        "Binning", "Entropy k-NN"
};

/**
 * Estimates the Kullback-Leibler divergence (KL-divergence) of the distribution of ensemble samples at each grid point
 * (after normalization, i.e., (value - mean) / stddev), and the standard normal distribution.
 * Currently, two estimators are supported:
 * - An estimator based on binning.
 * - An estimator based on an estimation of the entropy of the ensemble distribution using a k-nearest neighbor search.
 *   This is based on the Kozachenko-Leonenko estimator of the Shannon entropy.
 *
 * Derivation for the Entropy-based KL-divergence estimator:
 * P := The normalized sample distribution
 * Q := N(0, 1)
 * H: X -> \mathbb{R}^+_0 is the Shannon entropy
 * PDF of Q, q(x) = 1 / sqrt(2 \pi) e^{-\frac{x^2}{2}}
 * \log q(x) = -\frac{1}{2} \log(2 \pi) - \frac{x^2}{2}
 * D_KL(P||Q) = \int_X p(x) \log \frac{p(x)}{q(x)} dx = \int_X p(x) \log p(x) dx - \int_X p(x) \log q(x) dx =
 * = -H(P) - \int_X p(x) \cdot \left( -\frac{1}{2} \log(2 \pi) - \frac{x^2}{2} \right) dx =
 * = -H(P) + \frac{1}{2} \log(2 \pi) \int_X p(x) dx + \frac{1}{2} \int_X x^2 p(x) dx =
 * = -H(P) + \frac{1}{2} \log(2 \pi) + \frac{1}{2} \mathbb{E}[P^2]
 * ... where \mathbb{E}[P^2] = \mu'_{2,P} is the second moment of P.
 */
class DKLCalculator : public Calculator {
public:
    explicit DKLCalculator(sgl::vk::Renderer* renderer);
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::DKL_CALCULATOR; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    void onFieldRemoved(FieldType fieldType, int fieldIdx) override;
    [[nodiscard]] bool getShouldRenderGui() const override { return true; }
    FieldType getOutputFieldType() override { return FieldType::SCALAR; }
    std::string getOutputFieldName() override;
    FilterDevice getFilterDevice() override { return useGpu ? FilterDevice::VULKAN : FilterDevice::CPU; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;
    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    void onMemberCountChanged();
    void clearFieldDeviceData();
    bool getSupportsBufferMode();

    std::vector<std::string> scalarFieldNames;
    std::vector<size_t> scalarFieldIndexArray;
    int scalarFieldIndex = 0;
    int scalarFieldIndexGui = 0;
    std::shared_ptr<DKLComputePass> dklComputePass;
    bool useGpu = true;
    int cachedMemberCount = 0;
    DKLEstimatorType estimatorType = DKLEstimatorType::ENTROPY_KNN;
    CorrelationDataMode dataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool useBufferTiling = true;
    int numBins = 80; ///< For DKLEstimatorType::BINNED.
    int k = 3; ///< For DKLEstimatorType::ENTROPY_KNN.
    int kMax = 20; ///< For DKLEstimatorType::ENTROPY_KNN.
};

class DKLComputePass : public sgl::vk::ComputePass {
public:
    explicit DKLComputePass(sgl::vk::Renderer* renderer);
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    void setDataMode(CorrelationDataMode _dataMode);
    void setUseBufferTiling(bool _useBufferTiling);
    void setFieldBuffers(const std::vector<sgl::vk::BufferPtr>& _fieldBuffers);
    void setFieldImageViews(const std::vector<sgl::vk::ImageViewPtr>& _fieldImageViews);
    void setOutputImage(const sgl::vk::ImageViewPtr& _outputImage);
    void setEstimatorType(DKLEstimatorType _estimatorType);
    void setNumBins(int _numBins);
    void setNumNeighbors(int _k);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    VolumeData* volumeData = nullptr;
    int cachedEnsembleMemberCount = 0;

    const int computeBlockSizeX = 8, computeBlockSizeY = 8, computeBlockSizeZ = 4;
    const int computeBlockSize2dX = 8, computeBlockSize2dY = 8;
    struct UniformData {
        uint32_t xs, ys, zs, es;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;

    CorrelationDataMode dataMode = CorrelationDataMode::BUFFER_ARRAY;
    bool useBufferTiling = true;
    std::vector<sgl::vk::BufferPtr> fieldBuffers;
    std::vector<sgl::vk::ImageViewPtr> fieldImageViews;
    sgl::vk::ImageViewPtr outputImage;

    bool useGpu = true;
    DKLEstimatorType estimatorType = DKLEstimatorType::ENTROPY_KNN;
    int numBins = 16; ///< For DKLEstimatorType::BINNED.
    int k = 3; ///< For DKLEstimatorType::ENTROPY_KNN.
    int kMax = 20; ///< For DKLEstimatorType::ENTROPY_KNN.
};

#endif //CORRERENDER_DKLCALCULATOR_HPP
