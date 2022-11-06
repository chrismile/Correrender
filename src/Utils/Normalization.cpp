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

#include "Normalization.hpp"

void normalizeVertexPositions(
        std::vector<glm::vec3>& vertexPositions, const sgl::AABB3& aabb,
        const glm::mat4* vertexTransformationMatrixPtr) {
    glm::vec3 translation = -aabb.getCenter();
    glm::vec3 scale3D = 0.5f / aabb.getDimensions();
    float scale = std::min(scale3D.x, std::min(scale3D.y, scale3D.z));

#if _OPENMP >= 200805
    #pragma omp parallel for shared(vertexPositions, translation, scale) default(none)
#endif
    for (size_t vertexIdx = 0; vertexIdx < vertexPositions.size(); vertexIdx++) {
        glm::vec3& v = vertexPositions.at(vertexIdx);
        v = (v + translation) * scale;
    }

    if (vertexTransformationMatrixPtr != nullptr) {
        glm::mat4 transformationMatrix = *vertexTransformationMatrixPtr;

#if _OPENMP >= 200805
        #pragma omp parallel for shared(vertexPositions, transformationMatrix) default(none)
#endif
        for (size_t vertexIdx = 0; vertexIdx < vertexPositions.size(); vertexIdx++) {
            glm::vec3& v = vertexPositions.at(vertexIdx);
            glm::vec4 transformedVec = transformationMatrix * glm::vec4(v.x, v.y, v.z, 1.0f);
            v = glm::vec3(transformedVec.x, transformedVec.y, transformedVec.z);
        }
    }
}
