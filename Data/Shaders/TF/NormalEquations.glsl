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

-- Compute

#version 450 core

#extension GL_EXT_shader_atomic_float : require

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;
//layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, tfNumEntries;
    float minGT, maxGT, minOpt, maxOpt;
};

layout(push_constant) uniform PushConstants {
    float Nj;
};

layout (binding = 1, INPUT_IMAGE_0_FORMAT) uniform readonly image3D inputImageGT;
layout (binding = 2, INPUT_IMAGE_1_FORMAT) uniform readonly image3D inputImageOpt;

layout(binding = 3) coherent buffer LhsBuffer {
    float lhs[];
};

layout(binding = 4) coherent buffer RhsBuffer {
    float rhs[];
};

layout(binding = 5) readonly buffer TransferFunctionGTBuffer {
    vec4 tfGT[];
};

// i: row, j: column. Assumes column-major storage format.
#define IDXM(i, j) ((i) + (j) * tfNumEntries)

void main() {
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    float scalarGT = imageLoad(inputImageGT, currentPointIdx).x;
    float scalarOpt = imageLoad(inputImageOpt, currentPointIdx).x;
    if (isnan(scalarGT) || isnan(scalarOpt)) {
        return;
    }

    float tGT = (scalarGT - minGT) / (maxGT - minGT);
    float tGT0 = clamp(floor(tGT * Nj), 0.0f, Nj);
    float tGT1 = clamp(ceil(tGT * Nj), 0.0f, Nj);
    float fGT = tGT * Nj - tGT0;
    int jGT0 = int(tGT0);
    int jGT1 = int(tGT1);
    vec4 cGT0 = tfGT[jGT0];
    vec4 cGT1 = tfGT[jGT1];
    vec4 colorGT = mix(cGT0, cGT1, fGT);

    float tOpt = (scalarOpt - minOpt) / (maxOpt - minOpt);
    float tOpt0 = clamp(floor(tOpt * Nj), 0.0f, Nj);
    float tOpt1 = clamp(ceil(tOpt * Nj), 0.0f, Nj);
    float fOpt = tOpt * Nj - tOpt0;
    int jOpt0 = int(tOpt0);
    int jOpt1 = int(tOpt1);
    for (int c = 0; c < 4; c++) {
        uint i = jOpt0 * 4 + c;
        atomicAdd(rhs[i], fOpt * colorGT[c]);
        atomicAdd(lhs[IDXM(i, i)], fOpt * fOpt);
        if (jOpt0 != jOpt1) {
            uint j = jOpt1 * 4 + c;
            float fOpt1 = 1.0f - fOpt;
            atomicAdd(rhs[j], fOpt1 * colorGT[c]);
            atomicAdd(lhs[IDXM(i, j)], fOpt * fOpt1);
            atomicAdd(lhs[IDXM(j, j)], fOpt1 * fOpt1);
        }
    }
}

-- ComputeSymmetrization

#version 450 core

layout(local_size_x = BLOCK_SIZE) in;

layout(push_constant) uniform PushConstants {
    uint tfNumEntries;
    uint workSize;
};

layout(binding = 0) coherent buffer LhsBuffer {
    float lhs[];
};

// Fast integer square root, i.e., floor(sqrt(s)), see https://en.wikipedia.org/wiki/Integer_square_root
uint uisqrt(uint s) {
    if (s <= 1) {
        return s;
    }

    /*
     * Initial estimate should be pow2(floor(log2(n)/2)+1) if this can be estimated cheaply.
     * Otherwise, n/2 is used as the initial estimate. The estimate MUST always be larger than the result.
     * std::bit_width(s) == floor(log2(n)) + 1
     * NOTE: pow2(floor(log2(n)/2)+1) == pow2(floor(log2(n))/2+1)
     */
    uint x0 = 1 << ((findMSB(s) >> 1u) + 1u);
    uint x1 = (x0 + s / x0) / 2;
    while (x1 < x0) {
        x0 = x1;
        x1 = (x0 + s / x0) / 2;
    }
    return x0;
}

// i: row, j: column. Assumes column-major storage format.
#define IDXM(i, j) ((i) + (j) * tfNumEntries)

void main() {
    uint q = gl_GlobalInvocationID.x;
    if (q > workSize) {
        return;
    }
    uint i = (1 + uisqrt(1 + 8 * q)) / 2;
    uint j = uint(q) - i * (i - 1) / 2;
    lhs[IDXM(i, j)] = lhs[IDXM(j, i)];
}
