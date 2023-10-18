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

#ifndef CORRERENDER_DEEPLEARNINGCORRELATIONCALCULATOR_HPP
#define CORRERENDER_DEEPLEARNINGCORRELATIONCALCULATOR_HPP

#include "CorrelationCalculator.hpp"

class DeepLearningCorrelationCalculator : public ICorrelationCalculator {
public:
    /**
     * @param implName E.g., "tiny-cuda-nn" or "QuickMLP".
     * @param implNameKey E.g., "tinyCudaNN" or "quickMLP".
     * @param renderer The renderer object.
     */
    explicit DeepLearningCorrelationCalculator(
            const std::string& implName, const std::string& implNameKey, sgl::vk::Renderer* renderer);
    void initialize() override;
    ~DeepLearningCorrelationCalculator() override;
    std::string getOutputFieldName() override {
        std::string outputFieldName = "Correlation " + implName;
        if (calculatorConstructorUseCount > 1) {
            outputFieldName += " (" + std::to_string(calculatorConstructorUseCount) + ")";
        }
        return outputFieldName;
    }
    [[nodiscard]] bool getHasFixedRange() const override {
        return !isMutualInformationData;
    }
    [[nodiscard]] std::pair<float, float> getFixedRange() const override {
        if (calculateAbsoluteValue) {
            return std::make_pair(0.0f, 1.0f);
        } else {
            return std::make_pair(-1.0f, 1.0f);
        }
    }
    void setVolumeData(VolumeData* _volumeData, bool isNewData) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

    void setSettings(const SettingsMap& settings) override;
    void getSettings(SettingsMap& settings) override;

protected:
    std::vector<uint32_t> computeNanStencilBufferHost();
    virtual void loadModelFromFile(const std::string& modelPath) = 0;
    bool getSupportsSeparateFields() override;

    // Inference steps to be implemented by subclasses.
    virtual bool getIsModuleLoaded() = 0;

    void renderGuiImplSub(sgl::PropertyEditor& propertyEditor) override;
    void renderGuiImplAdvanced(sgl::PropertyEditor& propertyEditor) override;

    std::string modelFilePath;
    std::string fileDialogDirectory;

    std::vector<std::string> modelPresets;
    std::vector<std::string> modelPresetFilenames;
    int modelPresetIndex = 0;

    NetworkType networkType = NetworkType::SRN_MINE;

    /// For networkType == NetworkType::SRN_MINE.
    bool isMutualInformationData = true;
    bool calculateAbsoluteValue = false;

    /// NaN stencil for networkType == NetworkType::SRN_MINE.
    bool useDataNanStencil = true;
    bool isNanStencilInitialized = false;
    uint32_t numNonNanValues = 0;

private:
    void parseModelPresetsFile(const std::string& filename);
    std::string implName, implNameKey, implNameKeyUpper, fileDialogKey, fileDialogDescription, modelFilePathSettingsKey;
};

#endif //CORRERENDER_DEEPLEARNINGCORRELATIONCALCULATOR_HPP
