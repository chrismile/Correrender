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

#ifndef CORRERENDER_TINYCUDANNCORRELATIONCALCULATOR_HPP
#define CORRERENDER_TINYCUDANNCORRELATIONCALCULATOR_HPP

#include "TinyCudaNNCorrelationDefines.hpp"
#include "DeepLearningCudaCorrelationCalculator.hpp"

struct TinyCudaNNModuleWrapper;
struct TinyCudaNNCacheWrapper;

/**
 * Uses a fork of tiny-cuda-nn for the implementation of the network architecture described in the following publication:
 *
 * Farokhmanesh, F., K. Höhlein, C. Neuhauser, T. Necker, M. Weissmann, T. Miyoshi, and R. Westermann (2023).
 * "Neural Fields for Interactive Visualization of Statistical Dependencies in 3D Simulation Ensembles".
 * In: Vision, Modeling, and Visualization. The Eurographics Association. ISBN: 978-3-03868-232-5.
 * DOI: https://doi.org/10.2312/vmv.20231229.
 *
 * For more details see:
 * - Müller, T. (2021). tiny-cuda-nn. Version 1.7. URL: https://github.com/NVlabs/tiny-cuda-nn.
 * - Neuhauser, C. and T. Müller (2024). tiny-cuda-nn (fork).
 *   URL: https://github.com/chrismile/tiny-cuda-nn/tree/activations.
 * - Müller, T., A. Evans, C. Schied, and A. Keller (2022). "Instant Neural Graphics Primitives with a Multiresolution
 *   Hash Encoding". In: ACM Trans. Graph. 41.4. ISSN: 0730-0301. DOI: https://doi.org/10.1145/3528223.3530127.
 * - Müller, T., F. Rousselle, J. Novák, and A. Keller (2021). "Real-Time Neural Radiance Caching for Path Tracing".
 *   In: ACM Trans. Graph. 40.4. ISSN: 0730-0301. DOI: https://doi.org/10.1145/3450626.3459812.
 */
class TinyCudaNNCorrelationCalculator : public DeepLearningCudaCorrelationCalculator {
public:
    explicit TinyCudaNNCorrelationCalculator(sgl::vk::Renderer* renderer);
    ~TinyCudaNNCorrelationCalculator() override;
    [[nodiscard]] CalculatorType getCalculatorType() const override { return CalculatorType::TINY_CUDA_NN; }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;

    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    void loadModelFromFile(const std::string& modelPath) override;

    void callbackBeginCompute() override;
    void callbackEndCompute() override;
    bool getIsModuleLoaded() override { return moduleWrapper != nullptr; }
    void recreateCache(int batchSize) override;
    CUdeviceptr getReferenceInputPointer() override;
    CUdeviceptr getQueryInputPointer() override;
    void runInferenceReference() override;
    void runInferenceBatch(uint32_t batchOffset, uint32_t batchSize) override;
    uint32_t getInputChannelAlignment() override { return isInputEncodingIdentity ? 16 : 4; }
    uint32_t getSrnStride() override { return isInputEncodingIdentity ? 16 : 3; }

    void renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) override;

private:
    uint32_t numLayersInEncoder = 0, numLayersOutEncoder = 0, numLayersInDecoder = 0, numLayersOutDecoder = 0;
    bool isInputEncodingIdentity = false;
    bool deviceSupporsFullyFusedMlp = false;
    TinyCudaNNNetworkImplementation networkImplementation = TinyCudaNNNetworkImplementation::FULLY_FUSED_MLP;
    std::shared_ptr<TinyCudaNNModuleWrapper> moduleWrapper;
    std::shared_ptr<TinyCudaNNCacheWrapper> cacheWrapper;
};

#endif //CORRERENDER_TINYCUDANNCORRELATIONCALCULATOR_HPP
