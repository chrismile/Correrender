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

-- Compute.SGD

#version 450 core

/*
 * Stochastic gradient descent optimizer.
 */

//#extension GL_EXT_debug_printf : enable

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform OptimizerSettingsBuffer {
    float alpha; ///< Learning rate.
};

layout(binding = 1, std430) buffer TfOptBuffer {
    float tfOpt[NUM_TF_ENTRIES];
};

layout(binding = 2, std430) readonly buffer TfOptGradientBuffer {
    float g[NUM_TF_ENTRIES];
};

void main() {
    uint globalThreadIdx = gl_GlobalInvocationID.x;
    if (globalThreadIdx >= NUM_TF_ENTRIES) {
        return;
    }

    // Update the parameters.
    //if (globalThreadIdx >= 252) {
    //    debugPrintfEXT("g: %u, %f, %f", globalThreadIdx, tfOpt[globalThreadIdx], g[globalThreadIdx]);
    //}
    //debugPrintfEXT("g: %u %f, %f", globalThreadIdx, tfOpt[globalThreadIdx], g[globalThreadIdx]);
    //tfOpt[globalThreadIdx] -= alpha * g[globalThreadIdx];
    tfOpt[globalThreadIdx] = clamp(tfOpt[globalThreadIdx] - alpha * g[globalThreadIdx], 0.0, 1.0);
}


-- Compute.Adam

#version 450 core

/*
 * Implementation of the optimizer introduced in:
 * Adam: A Method for Stochastic Optimization. Diederik P. Kingma, Jimmy Ba (2015).
 * https://arxiv.org/abs/1412.6980
 */

//#extension GL_EXT_debug_printf : enable

layout(local_size_x = BLOCK_SIZE) in;

layout(binding = 0) uniform OptimizerSettingsBuffer {
    float alpha; ///< Learning rate.
    float beta1; ///< First moment update rate.
    float beta2; ///< Second moment update rate.
    float epsilon; ///< Small epsilon value used to avoid division by zero.
};

layout(push_constant) uniform PushConstants {
    float t; ///< Time step.
};

layout(binding = 1, std430) buffer TfOptBuffer {
    float tfOpt[NUM_TF_ENTRIES];
};

layout(binding = 2, std430) readonly buffer TfOptGradientBuffer {
    float g[NUM_TF_ENTRIES];
};

layout(binding = 3, std430) buffer FirstMomentEstimateBuffer {
    float m[NUM_TF_ENTRIES];
};

layout(binding = 4, std430) buffer SecondMomentEstimateBuffer {
    float v[NUM_TF_ENTRIES];
};

void main() {
    uint globalThreadIdx = gl_GlobalInvocationID.x;
    if (globalThreadIdx >= NUM_TF_ENTRIES) {
        return;
    }

    // Update biased first and second moment estimate.
    float gt = g[globalThreadIdx];
    float mt = beta1 * m[globalThreadIdx] + (1.0 - beta1) * gt;
    m[globalThreadIdx] = mt;
    float vt = beta2 * v[globalThreadIdx] + (1.0 - beta1) * gt * gt;
    v[globalThreadIdx] = vt;

    // Compute bias-corrected first and second moment estimate.
    float mht = mt / (1.0 - pow(beta1, t));
    float vht = vt / (1.0 - pow(beta2, t));

    // Update the parameters.
    //if (globalThreadIdx == NUM_TF_ENTRIES / 2) {
    //    debugPrintfEXT("%f, %f, %f, %f, %f, %f, %f, %f", tfOpt[globalThreadIdx], alpha, beta1, beta2, gt, mht, vht, epsilon);
    //}
    //tfOpt[globalThreadIdx] -= alpha * mht / (sqrt(vht) + epsilon);
    tfOpt[globalThreadIdx] = clamp(tfOpt[globalThreadIdx] - alpha * mht / (sqrt(vht) + epsilon), 0.0, 1.0);
}
