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

#ifndef CORRERENDER_FIELDACCESS_HPP
#define CORRERENDER_FIELDACCESS_HPP

#include <string>
#include <glm/vec3.hpp>

#include <Utils/HashCombine.hpp>

#include "FieldType.hpp"

struct FieldAccess {
    FieldType fieldType = FieldType::SCALAR;
    std::string fieldName;
    int timeStepIdx = 0;
    int ensembleIdx = 0;
    size_t sizeInBytes = 0;
    bool isImageData = true; //< Only for device data: Buffer or image?
    glm::uvec3 bufferTileSize{1, 1, 1}; //< Only for device data: Buffer tile size (for better performance).

    bool operator==(const FieldAccess& other) const {
        return fieldType == other.fieldType && fieldName == other.fieldName
                && other.timeStepIdx == timeStepIdx && other.ensembleIdx == ensembleIdx
                && other.isImageData == isImageData
                && (other.isImageData || other.bufferTileSize == bufferTileSize);
    }
};

namespace std {
template<> struct hash<FieldAccess> {
    std::size_t operator()(FieldAccess const& s) const noexcept {
        std::size_t result = 0;
        hash_combine(result, int(s.fieldType));
        hash_combine(result, s.fieldName);
        hash_combine(result, s.timeStepIdx);
        hash_combine(result, s.ensembleIdx);
        return result;
    }
};
}

#endif //CORRERENDER_FIELDACCESS_HPP
