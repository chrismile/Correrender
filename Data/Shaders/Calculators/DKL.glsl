/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

-- Compute

#version 450 core
#extension GL_EXT_nonuniform_qualifier : require

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout(push_constant) uniform PushConstants {
    float quietNan;
};

/**
 * Kullback-Leibler Divergence (D_KL) calculator.
 * This calculator estimates the KL-Divergence between the distribution of the ensemble member samples after
 * normalization and the standard normal distribution N(0, 1).
 * A binning-based version and a nearest neighbor-based version are available.
 */

#include "ScalarFields.glsl"

layout (binding = 1, r32f) uniform writeonly image3D outputImage;

float valueArray[MEMBER_COUNT];

#define FLT_MAX 3.402823466e+38


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

#ifdef ENTROPY_KNN_ESTIMATOR

/**
 * Lanczos approximation of digamma function using weights by Viktor T. Toth.
 * - digamma = d/dx ln(Gamma(x)) = Gamma'(x) / Gamma(x) (https://en.wikipedia.org/wiki/Digamma_function)
 * - Lanczos approximation: https://www.rskey.org/CMS/index.php/the-library/11
 * - Weights: https://www.rskey.org/CMS/index.php/the-library/11
 *
 * This function could be extended for values < 1 by:
 * - float z = 1 - iz;
 * - return digammaValue - M_PI * cos(M_PI * iz) / sin(M_PI * iz);
 */
#define G (5.15)
#define P0 (2.50662827563479526904)
#define P1 (225.525584619175212544)
#define P2 (-268.295973841304927459)
#define P3 (80.9030806934622512966)
#define P4 (-5.00757863970517583837)
#define P5 (0.0114684895434781459556)
float digamma(uint iz) {
    if (iz == 1u) {
        return -0.57721566490153287;
    }
    float z = float(iz);
    float zh = z - 0.5;
    float z1 = z + 1.0;
    float z2 = z + 2.0;
    float z3 = z + 3.0;
    float z4 = z + 4.0;
    float ZP = P0 + P1 / z + P2 / z1 + P3 / z2 + P4 / z3 + P5 / z4;
    float dZP = P1 / (z * z) + P2 / (z1 * z1) + P3 / (z2 * z2) + P4 / (z3 * z3) + P5 / (z4 * z4);
    float digammaValue = log(zh + G) + zh / (zh + G) - dZP / ZP - 1.0;

    /*
     * Alternative formulation:
     * float zh = z + 0.5f;
     * float z1 = z + 1.0f;
     * float z2 = z + 2.0f;
     * float z3 = z + 3.0f;
     * float z4 = z + 4.0f;
     * float z5 = z + 5.0f;
     * float ZP = P0 + P1 / z1 + P2 / z2 + P3 / z3 + P4 / z4 + P5 / z5;
     * float dZP = P1 / (z1 * z1) + P2 / (z2 * z2) + P3 / (z3 * z3) + P4 / (z4 * z4) + P5 / (z5 * z5);
     * float digammaValue = log(zh + G) - (zh + G + G * z)/(z * (zh + G)) - dZP / ZP;
     */

    return digammaValue;
}

void swapElements(uint i, uint j) {
    float temp = valueArray[i];
    valueArray[i] = valueArray[j];
    valueArray[j] = temp;
}

void heapify(uint i, uint numElements) {
    uint child;
    float childValue0, childValue1;
    while ((child = 2 * i + 1) < numElements) {
        // Is left or right child larger?
        childValue0 = valueArray[child];
        childValue1 = valueArray[child + 1];
        if (child + 1 < numElements && childValue0 < childValue1) {
            childValue0 = childValue1;
            child++;
        }
        // Swap with child if it is larger than the parent.
        if (valueArray[i] >= childValue0) {
            break;
        }
        swapElements(i, child);
        i = child;
    }
}

void heapSort() {
    // We can't use "i >= 0" with uint, thus adapt range and subtract 1 from i.
    uint i;
    for (i = MEMBER_COUNT / 2; i > 0; i--) {
        heapify(i - 1, MEMBER_COUNT);
    }
    // Largest element is at index 0. Swap it to the end of the processed array portion iteratively.
    for (i = 1; i < MEMBER_COUNT; i++) {
        swapElements(0, MEMBER_COUNT - i);
        heapify(0, MEMBER_COUNT - i);
    }
}

int iceil(int x, int y) { return (x - 1) / y + 1; }

int isign(float x) {
    return x > 0.0 ? 1 : (x < 0.0 ? -1 : 0);
}

float findKNearestNeighbors(uint c) {
    int i = int(c);
    int step = iceil(k, 2);
    int l = max(i - step, 0);
    int r = l + k;
    if (r >= MEMBER_COUNT) {
        r = MEMBER_COUNT - 1;
        l = MEMBER_COUNT - k - 1;
    }
    float vc = valueArray[i];
    while (true) {
        float vl = l >= 0 ? vc - valueArray[l] : FLT_MAX;
        float vr = r < MEMBER_COUNT ? valueArray[r] - vc : FLT_MAX;
        float diff = max(vl, vr);
        float vl0 = l - step <= i && l - step >= 0 ? vc - valueArray[l - step] : FLT_MAX;
        float vr0 = r - step >= i && r - step < MEMBER_COUNT ? valueArray[r - step] - vc : FLT_MAX;
        float diff0 = max(vl0, vr0);
        float vl1 = l + step <= i && l + step >= 0 ? vc - valueArray[l + step] : FLT_MAX;
        float vr1 = r + step >= i && r + step < MEMBER_COUNT ? valueArray[r + step] - vc : FLT_MAX;
        float diff1 = max(vl1, vr1);
        int signDir = isign(diff0 - diff1);
        if (diff < diff0 && diff < diff1) {
            signDir = 0;
        }
        l += signDir * step;
        r += signDir * step;
        if (step == 1) {
            break;
        }
        step = iceil(step, 2);
    }
    return max(vc - valueArray[l], valueArray[r] - vc);
}

// TODO: Do we want to add noise like for the MI estimator?
const float EPSILON_NOISE = 1e-5;
#define TWO_PI 6.2831853071795864

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }
#ifndef USE_SCALAR_FIELD_IMAGES
    uint currentIdx = IDXS(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

    float factor = 1.0 / float(MEMBER_COUNT);
    float nanValue = 0.0;
    float mean = 0.0;
    for (int c = 0; c < MEMBER_COUNT; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
        float val = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
#else
        float val = scalarFields[nonuniformEXT(c)].values[currentIdx];
#endif
        if (isnan(val)) {
            nanValue = val;
        }
        valueArray[c] = val;
        mean += factor * val;
    }

    float variance = 0.0;
    for (int c = 0; c < MEMBER_COUNT; c++) {
        float diff = mean - valueArray[c];
        variance += factor * diff * diff;
    }
    float stdev = sqrt(variance);
    for (int c = 0; c < MEMBER_COUNT; c++) {
        valueArray[c] = (valueArray[c] - mean) / stdev;
    }

    heapSort();

    float entropyEstimate = 0.0;
    float secondMoment = 0.0;
    for (uint c = 0; c < MEMBER_COUNT; c++) {
        float nnDist = findKNearestNeighbors(c);
        entropyEstimate += factor * log(nnDist);
        float value = valueArray[c];
        secondMoment += factor * value * value;
    }
    entropyEstimate += digamma(MEMBER_COUNT) - digamma(k) + log(2.0);

    float dkl = -entropyEstimate + 0.5 * log(TWO_PI) + 0.5 * secondMoment;
    if (isinf(dkl)) {
        dkl = quietNan;
    } else {
        dkl = max(dkl, 0.0);
    }

    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : dkl));
}

#elif defined(BINNED_ESTIMATOR)

#define PI 3.1415926535897932
float sqr(float val) { return val * val; }

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }
#ifndef USE_SCALAR_FIELD_IMAGES
    uint currentIdx = IDXS(currentPointIdx.x, currentPointIdx.y, currentPointIdx.z);
#endif

    float nanValue = 0.0;
    for (int c = 0; c < MEMBER_COUNT; c++) {
#ifdef USE_SCALAR_FIELD_IMAGES
        float val = texelFetch(sampler3D(scalarFields[nonuniformEXT(c)], scalarFieldSampler), currentPointIdx, 0).r;
#else
        float val = scalarFields[nonuniformEXT(c)].values[currentIdx];
#endif
        if (isnan(val)) {
            nanValue = val;
        }
        valueArray[c] = val;
    }

    float factor = 1.0 / float(MEMBER_COUNT);
    float mean = 0;
    float variance = 0;
    for (int c = 0; c < MEMBER_COUNT; c++) {
        mean += factor * valueArray[c];
    }
    for (int c = 0; c < MEMBER_COUNT; c++) {
        float diff = mean - valueArray[c];
        variance += factor * diff * diff;
    }
    float stdev = sqrt(variance);

    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;
    for (int c = 0; c < MEMBER_COUNT; c++) {
        float val = (valueArray[c] - mean) / stdev;
        valueArray[c] = val;
        minVal = min(minVal, val);
        maxVal = max(maxVal, val);
    }
    minVal -= 0.01;
    maxVal += 0.01;
    float binFactor = float(numBins) / (maxVal - minVal);
    float binFactorInv = (maxVal - minVal) / float(numBins);

    float histogram[numBins];
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        histogram[binIdx] = 0.0;
    }

    float dkl = 0;
    for (int c = 0; c < MEMBER_COUNT; c++) {
        int binIdx = clamp(int((valueArray[c] - minVal) * binFactor), 0, numBins - 1);
        histogram[binIdx] += 1;
    }
    for (int binIdx = 0; binIdx < numBins; binIdx++) {
        if (histogram[binIdx] > 0) {
            float px = histogram[binIdx] / float(MEMBER_COUNT);
            float center = (float(binIdx) + 0.5) * binFactorInv + minVal;
            dkl += log(px * binFactor / (sqrt(0.5 / PI) * exp(-0.5 * sqr(center)))) * px;
        }
    }
    if (isinf(dkl)) {
        dkl = quietNan;
    }

    imageStore(outputImage, currentPointIdx, vec4(isnan(nanValue) ? nanValue : dkl));
}

#endif
