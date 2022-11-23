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

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/imgui_custom.h>

#include "Loaders/DataSet.hpp"
#include "Volume/VolumeData.hpp"
#include "ReferencePointSelectionRenderer.hpp"
#include "SimilarityCalculator.hpp"

EnsembleSimilarityCalculator::EnsembleSimilarityCalculator(sgl::vk::Renderer* renderer) : Calculator(renderer) {
}

void EnsembleSimilarityCalculator::setViewManager(ViewManager* _viewManager) {
    viewManager = _viewManager;
    referencePointSelectionRenderer = new ReferencePointSelectionRenderer(viewManager);
    calculatorRenderer = RendererPtr(referencePointSelectionRenderer);
    referencePointSelectionRenderer->initialize();
}

void EnsembleSimilarityCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    Calculator::setVolumeData(_volumeData, isNewData);
    referencePointSelectionRenderer->setVolumeDataPtr(volumeData, isNewData);

    if (isNewData) {
        referencePointIndex.x = volumeData->getGridSizeX() / 2;
        referencePointIndex.y = volumeData->getGridSizeY() / 2;
        referencePointIndex.z = volumeData->getGridSizeZ() / 2;
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);

        fieldIndex = 0;
        fieldIndexGui = 0;
        scalarFieldNames = {};
        scalarFieldIndexArray = {};

        std::vector<std::string> scalarFieldNamesNew = volumeData->getFieldNames(FieldType::SCALAR);
        for (size_t i = 0; i < scalarFieldNamesNew.size(); i++) {
            if (scalarFieldNamesNew.at(i) != getOutputFieldName()) {
                scalarFieldNames.push_back(scalarFieldNamesNew.at(i));
                scalarFieldIndexArray.push_back(i);
            }
        }
    }
}

void EnsembleSimilarityCalculator::update(float dt) {
    // TODO: Use mouse for selection of reference point.
}

void EnsembleSimilarityCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.addCombo(
            "Scalar Field", &fieldIndexGui, scalarFieldNames.data(), int(scalarFieldNames.size()))) {
        fieldIndex = int(scalarFieldIndexArray.at(fieldIndexGui));
        dirty = true;
    }

    // TODO: Replace with referencePointSelectionRenderer.
    bool inputFinished[3];
    inputFinished[0] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (X)", &referencePointIndex[0], 0, volumeData->getGridSizeX() - 1)
            == ImGui::EditMode::INPUT_FINISHED;
    inputFinished[1] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Y)", &referencePointIndex[1], 0, volumeData->getGridSizeY() - 1)
            == ImGui::EditMode::INPUT_FINISHED;
    inputFinished[2] =
            propertyEditor.addSliderIntEdit(
                    "Reference Point (Z)", &referencePointIndex[2], 0, volumeData->getGridSizeZ() - 1)
            == ImGui::EditMode::INPUT_FINISHED;
    if (inputFinished[0] || inputFinished[1] || inputFinished[2]) {
        referencePointSelectionRenderer->setReferencePosition(referencePointIndex);
        dirty = true;
    }
}



PccCalculator::PccCalculator(sgl::vk::Renderer* renderer) : EnsembleSimilarityCalculator(renderer) {
    pccComputePass = std::make_shared<PccComputePass>(renderer);
}

void PccCalculator::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    EnsembleSimilarityCalculator::setVolumeData(_volumeData, isNewData);
    pccComputePass->setVolumeData(volumeData, isNewData);
}

FilterDevice PccCalculator::getFilterDevice() {
    return useGpu ? FilterDevice::VULKAN : FilterDevice::CPU;
}

void PccCalculator::renderGuiImpl(sgl::PropertyEditor& propertyEditor) {
     EnsembleSimilarityCalculator::renderGuiImpl(propertyEditor);
   if (propertyEditor.addCheckbox("Use GPU", &useGpu)) {
        hasFilterDeviceChanged = true;
        dirty = true;
    }
}

template<class T>
inline float computePearson1(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx) {
    auto n = T(es);
    T sumX = 0;
    T sumY = 0;
    T sumXY = 0;
    T sumXX = 0;
    T sumYY = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        sumX += x;
        sumY += y;
        sumXY += x * y;
        sumXX += x * x;
        sumYY += y * y;
    }
    float pearsonCorrelation =
            (n * sumXY - sumX * sumY) / std::sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    return (float)pearsonCorrelation;
}

template<class T>
inline float computePearson2(
        const float* referenceValues, const std::vector<float*>& ensembleFields, int es, size_t gridPointIdx) {
    auto n = T(es);
    T meanX = 0;
    T meanY = 0;
    T invN = T(1) / n;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        meanX += invN * x;
        meanY += invN * y;
    }
    T varX = 0;
    T varY = 0;
    T invNm1 = T(1) / (n - T(1));
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        T diffX = x - meanX;
        T diffY = y - meanY;
        varX += invNm1 * diffX * diffX;
        varY += invNm1 * diffY * diffY;
    }
    T stdDevX = std::sqrt(varX);
    T stdDevY = std::sqrt(varY);
    T pearsonCorrelation = 0;
    for (int e = 0; e < es; e++) {
        T x = (T)referenceValues[e];
        T y = (T)ensembleFields.at(e)[gridPointIdx];
        pearsonCorrelation += invNm1 * ((x - meanX) / stdDevX) * ((y - meanY) / stdDevY);
    }
    return (float)pearsonCorrelation;
}

void PccCalculator::calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();

    std::vector<VolumeData::HostCacheEntry> ensembleEntryFields;
    std::vector<float*> ensembleFields;
    ensembleEntryFields.reserve(es);
    ensembleFields.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::HostCacheEntry ensembleEntryField = volumeData->getFieldEntryCpu(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleFields.push_back(ensembleEntryField.get());
    }

    //size_t referencePointIdx =
    //        size_t(referencePointIndex.x) * size_t(referencePointIndex.y) * size_t(referencePointIndex.z);
    size_t referencePointIdx = IDXS(referencePointIndex.x, referencePointIndex.y, referencePointIndex.z);
    auto* referenceValues = new float[es];
    for (int e = 0; e < es; e++) {
        referenceValues[e] = ensembleFields.at(e)[referencePointIdx];
    }

    // See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    size_t numGridPoints = size_t(xs) * size_t(ys) * size_t(zs);
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridPoints), [&](auto const& r) {
        for (auto gridPointIdx = r.begin(); gridPointIdx != r.end(); gridPointIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel for shared(numGridPoints, es, referenceValues, ensembleFields, buffer) default(none)
#endif
    for (size_t gridPointIdx = 0; gridPointIdx < numGridPoints; gridPointIdx++) {
#endif
        if (es == 1) {
            buffer[gridPointIdx] = 1.0f;
            continue;
        }
#define FORMULA_2_DOUBLE
#ifdef FORMULA_1_FLOAT
        float pearsonCorrelation = computePearson1<float>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_1_DOUBLE)
        float pearsonCorrelation = computePearson1<double>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_2_FLOAT)
        float pearsonCorrelation = computePearson2<float>(referenceValues, ensembleFields, es, gridPointIdx);
#elif defined(FORMULA_2_DOUBLE)
        float pearsonCorrelation = computePearson2<double>(referenceValues, ensembleFields, es, gridPointIdx);
#endif
        buffer[gridPointIdx] = pearsonCorrelation;
    }
#ifdef USE_TBB
    });
#endif

    delete[] referenceValues;
}

void PccCalculator::calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {
    // We write to the descriptor set, so wait until the device is idle.
    renderer->getDevice()->waitIdle();

    int es = volumeData->getEnsembleMemberCount();

    std::vector<VolumeData::DeviceCacheEntry> ensembleEntryFields;
    std::vector<sgl::vk::ImageViewPtr> ensembleImageViews;
    std::vector<CUtexObject> ensembleTexturesCu;
    ensembleEntryFields.reserve(es);
    ensembleImageViews.reserve(es);
    ensembleTexturesCu.reserve(es);
    for (ensembleIdx = 0; ensembleIdx < es; ensembleIdx++) {
        VolumeData::DeviceCacheEntry ensembleEntryField = volumeData->getFieldEntryDevice(
                FieldType::SCALAR, scalarFieldNames.at(fieldIndexGui), timeStepIdx, ensembleIdx);
        ensembleEntryFields.push_back(ensembleEntryField);
        ensembleImageViews.push_back(ensembleEntryField->getVulkanImageView());
        ensembleTexturesCu.push_back(ensembleEntryField->getCudaTexture());
        if (ensembleEntryField->getVulkanImage()->getVkImageLayout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            deviceCacheEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        }
    }

    pccComputePass->setOutputImage(deviceCacheEntry->getVulkanImageView());
    pccComputePass->setEnsembleImageViews(ensembleImageViews);

    renderer->insertImageMemoryBarrier(
            deviceCacheEntry->getVulkanImage(),
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);

    pccComputePass->buildIfNecessary();
    renderer->pushConstants(pccComputePass->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, referencePointIndex);
    pccComputePass->render();
}



PccComputePass::PccComputePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

void PccComputePass::setVolumeData(VolumeData *_volumeData, bool isNewData) {
    volumeData = _volumeData;
    uniformData.xs = uint32_t(volumeData->getGridSizeX());
    uniformData.ys = uint32_t(volumeData->getGridSizeY());
    uniformData.zs = uint32_t(volumeData->getGridSizeZ());
    uniformData.es = uint32_t(volumeData->getEnsembleMemberCount());
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (cachedEnsembleMemberCount != volumeData->getEnsembleMemberCount()) {
        cachedEnsembleMemberCount = volumeData->getEnsembleMemberCount();
        setShaderDirty();
    }
}

void PccComputePass::setEnsembleImageViews(const std::vector<sgl::vk::ImageViewPtr>& _ensembleImageViews) {
    if (ensembleImageViews == _ensembleImageViews) {
        return;
    }
    ensembleImageViews = _ensembleImageViews;
    if (computeData) {
        computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    }
}

void PccComputePass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    outputImage = _outputImage;
    if (computeData) {
        computeData->setStaticImageView(outputImage, "outputImage");
    }
}

void PccComputePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(computeBlockSizeX)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(computeBlockSizeY)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(computeBlockSizeZ)));
    preprocessorDefines.insert(std::make_pair(
            "ENSEMBLE_MEMBER_COUNT", std::to_string(volumeData->getEnsembleMemberCount())));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "PearsonCorrelation.Compute" }, preprocessorDefines);
}

void PccComputePass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");
    computeData->setImageSampler(volumeData->getImageSampler(), "scalarFieldSampler");
    computeData->setStaticImageViewArray(ensembleImageViews, "scalarFieldEnsembles");
    computeData->setStaticImageView(outputImage, "outputImage");
}

void PccComputePass::_render() {
    renderer->dispatch(
            computeData,
            sgl::iceil(volumeData->getGridSizeX(), computeBlockSizeX),
            sgl::iceil(volumeData->getGridSizeY(), computeBlockSizeY),
            sgl::iceil(volumeData->getGridSizeZ(), computeBlockSizeZ));
}
