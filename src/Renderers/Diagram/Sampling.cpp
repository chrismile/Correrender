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

#include <random>

#include <Utils/File/Logfile.hpp>

#include "Sampling.hpp"

/**
 * Generates a sequence of random samples uniformly distributed in [0, 1).
 * @param samples An array of numSamples * 6 random samples in [0, 1).
 * @param numSamples The number of samples.
 */
void generateSamplesRandomUniform(float* samples, int numSamples) {
    constexpr int d = 6;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    for (int i = 0; i < d * numSamples; i++) {
        samples[i] = dis(generator);
    }
}

/**
 * For more details on this quasi-random sequence, please refer to: https://en.wikipedia.org/wiki/Halton_sequence
 * @param samples An array of numSamples * 6 quasi-random samples in [0, 1).
 * @param numSamples The number of samples.
 */
void generateSamplesQuasirandomHalton(float* samples, int numSamples, bool useRandomSeed) {
    double seed = 0.0f;
    if  (useRandomSeed) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<float> dis(0, 1);
        seed = dis(generator);
    }

    // Use the first 6 prime numbers as bases.
    constexpr int d = 6;
    const int bases[] = { 2, 3, 5, 7, 11, 13 };
    for (int j = 0; j < d; j++) {
        int b = bases[j];
        int n = 0;
        int m = 1;
        for (int i = 0; i < numSamples; i++) {
            int x = m - n;
            if (x == 1) {
                n = 1;
                m *= b;
            } else {
                int y = m / b;
                while (x <= y) {
                    y /= b;
                }
                n = (b + 1) * y - x;
            }
            samples[i * d + j] = float(std::fmod(seed + float(n) / float(m), 1.0));
        }
    }
}

/**
 * For more details on this quasi-random sequence, please refer to:
 * http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
 * @param samples An array of numSamples * 6 quasi-random samples in [0, 1).
 * @param numSamples The number of samples.
 * @param useRandomSeed Whether to use 0.5 as the seed or a random seed in [0, 1).
 */
void generateSamplesQuasirandomPlastic(float* samples, int numSamples, bool useRandomSeed) {
    double seed = 0.5f;
    if  (useRandomSeed) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<float> dis(0, 1);
        seed = dis(generator);
    }

    // Compute phi(6).
    constexpr int d = 6;
    double g = 2.0;
    for (int i = 0; i < 10; i++) {
        g = std::pow(1.0 + g, 1.0 / (d + 1.0));
    }

    double alpha[6];
    for (int j = 0; j < d; j++) {
        alpha[j] = std::fmod(std::pow(1.0 / g, double(j + 1)), 1.0);
    }

    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < d; j++) {
            samples[i * d + j] = float(std::fmod(seed + alpha[j] * double(i + 1), 1.0));
        }
    }
}

void generateSamples(
        float* samples, int numSamples, SamplingMethodType samplingMethodType, bool useRandomSeed) {
    if (samplingMethodType == SamplingMethodType::RANDOM_UNIFORM) {
        generateSamplesRandomUniform(samples, numSamples);
    } else if (samplingMethodType == SamplingMethodType::QUASIRANDOM_HALTON) {
        generateSamplesQuasirandomHalton(samples, numSamples, useRandomSeed);
    } else if (samplingMethodType == SamplingMethodType::QUASIRANDOM_PLASTIC) {
        generateSamplesQuasirandomPlastic(samples, numSamples, useRandomSeed);
    } else {
        sgl::Logfile::get()->throwError("Error in generateSamples: Unsupported sampling method type.");
    }
}
