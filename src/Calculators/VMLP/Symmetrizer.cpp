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

#include <Math/Math.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "Symmetrizer.hpp"

namespace vmlp {

struct PushConstantsSymmetrizer {
    uint32_t numElements, numChannels;
};

class SymmetrizerPass : public sgl::vk::ComputePass {
public:
    explicit SymmetrizerPass(sgl::vk::Renderer* renderer, SymmetrizerType symmetrizerType, int passIndex)
            : ComputePass(renderer), symmetrizerType(symmetrizerType), passIndex(passIndex) {}

    void setBuffersInOut(const Matrix& _matrixReference, const Matrix& _matrixQuery, const Matrix& _matrixOut);
    void setBatchSize(uint32_t _batchSize);
    void setFloatFormat(FloatFormat _format);

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t BLOCK_SIZE = 256;
    SymmetrizerType symmetrizerType;
    int passIndex;
    FloatFormat format = FloatFormat::FLOAT32;
    Matrix matrixReference, matrixQuery, matrixOut;
    uint32_t batchSize = 0;
};

void SymmetrizerPass::setBuffersInOut(
        const Matrix& _matrixReference, const Matrix& _matrixQuery, const Matrix& _matrixOut) {
    matrixReference = _matrixReference;
    matrixQuery = _matrixQuery;
    matrixOut = _matrixOut;
    dataDirty = true;
}

void SymmetrizerPass::setBatchSize(uint32_t _batchSize) {
    batchSize = _batchSize;
}

void SymmetrizerPass::setFloatFormat(FloatFormat _format) {
    format = _format;
    shaderDirty = true;
}

void SymmetrizerPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    addPreprocessorDefinesFormat(preprocessorDefines, format);
    std::string shaderName;
    if (symmetrizerType == SymmetrizerType::Add) {
        shaderName = "Symmetrizer.SymmetrizerAdd.Compute";
    } if (symmetrizerType == SymmetrizerType::Mul) {
        shaderName = "Symmetrizer.SymmetrizerMul.Compute";
    } if (symmetrizerType == SymmetrizerType::AddDiff && passIndex == 0) {
        shaderName = "Symmetrizer.SymmetrizerAddDiff_Add.Compute";
    } if (symmetrizerType == SymmetrizerType::AddDiff && passIndex == 1) {
        shaderName = "Symmetrizer.SymmetrizerAddDiff_Diff.Compute";
    }
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void SymmetrizerPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticBuffer(matrixReference.getBuffer(), "ReferenceBuffer");
    computeData->setStaticBuffer(matrixQuery.getBuffer(), "QueryBuffer");
    computeData->setStaticBuffer(matrixOut.getBuffer(), "OutputBuffer");
}

void SymmetrizerPass::_render() {
    const uint32_t numThreads = matrixQuery.getNumChannels() * batchSize;
    PushConstantsSymmetrizer pc{ numThreads, matrixQuery.getNumChannels() };
    renderer->pushConstants(getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
    renderer->dispatch(computeData, sgl::uiceil(numThreads, BLOCK_SIZE), 1, 1);
}

Symmetrizer::Symmetrizer(sgl::vk::Renderer* renderer, SymmetrizerType symmetrizerType)
        : symmetrizerType(symmetrizerType) {
    symmetrizerPass0 = std::make_shared<SymmetrizerPass>(renderer, symmetrizerType, 0);
    if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerPass1 = std::make_shared<SymmetrizerPass>(renderer, symmetrizerType, 1);
    }
}

void Symmetrizer::setInputOutputMatrices(
        const Matrix& _matrixReference, const Matrix& _matrixQuery, const Matrix& _matrixOutput) {
    symmetrizerPass0->setBuffersInOut(_matrixReference, _matrixQuery, _matrixOutput);
    if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerPass1->setBuffersInOut(_matrixReference, _matrixQuery, _matrixOutput);
    }
}

void Symmetrizer::setBatchSize(uint32_t _batchSize) {
    symmetrizerPass0->setBatchSize(_batchSize);
    if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerPass1->setBatchSize(_batchSize);
    }
}

void Symmetrizer::setFloatFormat(vmlp::FloatFormat _format) {
    format = _format;
    symmetrizerPass0->setFloatFormat(_format);
    if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerPass1->setFloatFormat(_format);
    }
}

void Symmetrizer::runInference() {
    symmetrizerPass0->render();
    if (symmetrizerType == SymmetrizerType::AddDiff) {
        symmetrizerPass1->render();
    }
}

}
