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

#ifndef CORRERENDER_SAMPLING_HPP
#define CORRERENDER_SAMPLING_HPP

#include <glm/vec3.hpp>

enum class SamplingMethodType {
    MEAN, RANDOM_UNIFORM, QUASIRANDOM_HALTON, QUASIRANDOM_PLASTIC, BAYESIAN_OPTIMIZATION
};

const char* const SAMPLING_METHOD_TYPE_NAMES[] = {
        "Mean", "Random Uniform", "Quasirandom Halton", "Quasirandom Plastic", "Bayesian Optimization"
};

void generateSamples(
        float* samples, int numSamples, SamplingMethodType samplingMethodType, bool useRandomSeed);

void generateSamples3D(
        float* samples, int numSamples, SamplingMethodType samplingMethodType, bool useRandomSeed);

class SampleGenerator3D {
public:
    virtual ~SampleGenerator3D() = default;
    virtual glm::vec3 next() = 0;
};
SampleGenerator3D* createSampleGenerator3D(SamplingMethodType samplingMethodType, bool useRandomSeed);

#endif //CORRERENDER_SAMPLING_HPP
