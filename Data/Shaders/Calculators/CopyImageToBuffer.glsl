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

layout(local_size_x = TILE_SIZE_X, local_size_y = TILE_SIZE_Y, local_size_z = TILE_SIZE_Z) in;

layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs;
};

layout (binding = 1, INPUT_IMAGE_FORMAT) uniform readonly image3D inputImage;

layout (binding = 2) writeonly buffer OutputBuffer {
    float outputBuffer[];
};

uint IDXS(uint x, uint y, uint z) {
    uint xst = (xs - 1) / TILE_SIZE_X + 1;
    uint yst = (ys - 1) / TILE_SIZE_Y + 1;
    //uint zst = (zs - 1) / TILE_SIZE_Z + 1;
    uint xt = x / TILE_SIZE_X;
    uint yt = y / TILE_SIZE_Y;
    uint zt = z / TILE_SIZE_Z;
    uint tileAddressLinear = (xt + yt * xst + zt * xst * yst) * (TILE_SIZE_X * TILE_SIZE_Y * TILE_SIZE_Z);
    uint vx = x & (TILE_SIZE_X - 1u);
    uint vy = y & (TILE_SIZE_Y - 1u);
    uint vz = z & (TILE_SIZE_Z - 1u);
    uint voxelAddressLinear = vx + vy * TILE_SIZE_X + vz * TILE_SIZE_X * TILE_SIZE_Y;
    return tileAddressLinear | voxelAddressLinear;
}

void main() {
    if (gl_GlobalInvocationID.x >= xs || gl_GlobalInvocationID.y >= ys || gl_GlobalInvocationID.z >= zs) {
        return;
    }
    float val = imageLoad(inputImage, ivec3(gl_GlobalInvocationID.xyz)).x;
    outputBuffer[IDXS(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z)] = val;
}
