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

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform SmoothingPriorSettingsBuffer {
    float lambda; ///< Smoothing rate.
    uint R; ///< Number of TF entries in the value axis.
};

layout(binding = 1) readonly buffer TransferFunctionBuffer {
    float tfEntries[NUM_TF_ENTRIES];
};

layout(binding = 2) readonly buffer TransferFunctionGradientBuffer {
    float g[NUM_TF_ENTRIES];
};

#define IDXTF(c, r) ((c) + (r) * 4u)

void main() {
    uint c = gl_GlobalInvocationID.x % 4u;
    uint r = gl_GlobalInvocationID.x / 4u;

    float gradVal = 0.0;
    float centerVal = tfEntries[IDXTF(c, r)];
    if (r > 0) {
        gradVal += centerVal - tfEntries[IDXTF(c, r - 1)];
    }
    if (r < R - 1) {
        gradVal += centerVal - tfEntries[IDXTF(c, r + 1)];
    }

    gradVal /= 2.0 * (R - 1);
    g[gl_GlobalInvocationID.x] += lambda * gradVal;
}
