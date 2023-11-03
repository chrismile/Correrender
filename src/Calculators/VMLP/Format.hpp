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

#ifndef CORRERENDER_FORMAT_HPP
#define CORRERENDER_FORMAT_HPP

#include <cstdint>

namespace vmlp {

enum class FloatFormat {
    FLOAT32 = 0, FLOAT16 = 1
};
const uint32_t FLOAT_FORMAT_SIZES_IN_BYTE[] = {
        4, 2
};
const char* const FLOAT_FORMAT_UI_NAMES[] = { "Float", "Half" };
const char* const FLOAT_FORMAT_GLSL_NAMES[] = { "float", "float16_t" };
const char* const FLOAT2_FORMAT_GLSL_NAMES[] = { "vec2", "f16vec2" };
const char* const FLOAT3_FORMAT_GLSL_NAMES[] = { "vec3", "f16vec3" };
const char* const FLOAT4_FORMAT_GLSL_NAMES[] = { "vec4", "f16vec4" };

enum class FusedMlpMemoryType {
    FLOAT16_NO_PADDING, FLOAT16_PADDING, UINT, UVEC2, UVEC4
};
const char* const FUSED_MLP_MEMORY_TYPE_NAME[] {
        "float16_t (no pad)", "float16_t (pad)", "uint", "uvec2", "uvec4"
};

}

const uint32_t NETWORK_PARAMS_FORMAT_FLOAT = 0;
const uint32_t NETWORK_PARAMS_FORMAT_HALF = 1;
struct NetworkParametersHeader {
    uint32_t format = 0;
    uint32_t numParams = 0;
};

#endif //CORRERENDER_FORMAT_HPP
