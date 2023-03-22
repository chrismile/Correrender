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

-- Compute

#version 450 core

layout(local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout (binding = 0) uniform UniformBuffer {
    int xs, ys, zs;
    float nanValue;
};

layout (binding = 1) readonly buffer KernelBuffer {
    float kernelWeights[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE];
};

layout (binding = 2, INPUT_IMAGE_FORMAT) uniform readonly image3D inputImage;
layout (binding = 3, r32f) uniform writeonly image3D outputImage;

#define IDXK(x,y,z) ((z)*KERNEL_SIZE*KERNEL_SIZE + (y)*KERNEL_SIZE + (x))

void main() {
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }

    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz);
    int x = currentPointIdx.x;
    int y = currentPointIdx.y;
    int z = currentPointIdx.z;

    const int kernelSizeHalf = KERNEL_SIZE / 2;
    float sum = 0.0f;
    float weightSum = 0.0f;
    for (int zi = 0; zi < KERNEL_SIZE; zi++) {
        for (int yi = 0; yi < KERNEL_SIZE; yi++) {
            for (int xi = 0; xi < KERNEL_SIZE; xi++) {
                int xr = x + xi - kernelSizeHalf;
                int yr = y + yi - kernelSizeHalf;
                int zr = z + zi - kernelSizeHalf;
                float value = nanValue;
                if (xr >= 0 && yr >= 0 && zr >= 0 && xr < xs && yr < ys && zr < zs) {
                    value = imageLoad(inputImage, ivec3(xr, yr, zr)).x;
                }
                if (!isnan(value)) {
                    float weight = kernelWeights[IDXK(xi, yi, zi)];
                    weightSum += weight;
                    sum += value * weight;
                }
            }
        }
    }
    float valueOut = nanValue;
    if (weightSum > 1e-5f) {
        valueOut = sum / weightSum;
    }

    imageStore(outputImage, currentPointIdx, vec4(valueOut));
}
