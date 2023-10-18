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

-- Header

layout(local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;

//layout (binding = 0) uniform UniformBuffer {
layout(push_constant) uniform PushConstants {
    uint numElements, numChannels;
};

layout(binding = 1, std430) readonly buffer ReferenceBuffer {
    real referenceValues[];
};

layout(binding = 2, std430) readonly buffer QueryBuffer {
    real queryValues[];
};

layout(binding = 3, std430) writeonly buffer OutputBuffer {
    real outputValues[];
};


-- SymmetrizerAdd.Compute

#version 450 core

#import ".Header"

void main() {
    uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numElements) {
        return;
    }
    uint readOffset = threadIdx;
    uint readOffsetRef = threadIdx % numChannels;
    outputValues[readOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}


-- SymmetrizerMul.Compute

#version 450 core

#import ".Header"

void main() {
    uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numElements) {
        return;
    }
    uint readOffset = threadIdx;
    uint readOffsetRef = threadIdx % numChannels;
    outputValues[readOffset] = referenceValues[readOffsetRef] * queryValues[readOffset];
}


-- SymmetrizerAddDiff_Add.Compute

#version 450 core

#import ".Header"

void main() {
    uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numElements) {
        return;
    }
    uint readOffset = threadIdx;
    uint channelIdx = threadIdx % numChannels;
    uint readOffsetRef = channelIdx;
    uint batchEnsembleIdx = threadIdx / numChannels;
    uint writeOffset = batchEnsembleIdx * numChannels * 2 + channelIdx;
    outputValues[writeOffset] = referenceValues[readOffsetRef] + queryValues[readOffset];
}


-- SymmetrizerAddDiff_Diff.Compute

#version 450 core

#import ".Header"

void main() {
    uint threadIdx = gl_GlobalInvocationID.x;
    if (threadIdx >= numElements) {
        return;
    }
    uint readOffset = globalThreadIdx;
    uint channelIdx = globalThreadIdx % numChannels;
    uint readOffsetRef = channelIdx;
    uint batchEnsembleIdx = globalThreadIdx / numChannels;
    uint writeOffset = batchEnsembleIdx * numChannels * 2 + numChannels + channelIdx;
    outputValues[writeOffset] = absT(referenceValues[readOffsetRef] - queryValues[readOffset]);
}
