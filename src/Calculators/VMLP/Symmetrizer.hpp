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

#ifndef CORRERENDER_SYMMETRIZER_HPP
#define CORRERENDER_SYMMETRIZER_HPP

#include "../SymmetrizerType.hpp"
#include "Network.hpp"

namespace vmlp {

class SymmetrizerPass;

class Symmetrizer : public Module {
public:
    Symmetrizer(sgl::vk::Renderer* renderer, SymmetrizerType symmetrizerType);
    uint32_t getNumChannelsIn() override {
        return matrixQuery.getNumChannels();
    }
    uint32_t getNumChannelsOut() override {
        return matrixOutput.getNumChannels();
    }
    void setInputOutputMatrices(
            const Matrix& input, const Matrix& output) override {
        sgl::Logfile::get()->throwError("Error in Symmetrizer::setInputOutputMatrices: Expected binary input.");
    }
    void setInputOutputMatrices(
            const Matrix& _matrixReference, const Matrix& _matrixQuery, const Matrix& _matrixOutput);
    void setBatchSize(uint32_t _batchSize) override;
    void setFloatFormat(FloatFormat _format) override;

    uint32_t getNumParameters() override {
        return 0;
    }
    void setParametersCpu(float* weights, uint32_t _numParameters) override {}

    void runInference() override;

private:
    SymmetrizerType symmetrizerType;
    Matrix matrixReference, matrixQuery, matrixOutput;
    std::shared_ptr<SymmetrizerPass> symmetrizerPass0, symmetrizerPass1;
};

}

#endif //CORRERENDER_SYMMETRIZER_HPP
