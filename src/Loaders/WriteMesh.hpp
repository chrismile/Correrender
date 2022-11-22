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

#ifndef CORRERENDER_WRITEMESH_HPP
#define CORRERENDER_WRITEMESH_HPP

#include <string>
#include <vector>
#include <glm/vec3.hpp>

/*
 * Saves the triangle mesh in the ASCII .stl format.
 */
void saveMeshStlAscii(
        const std::string& filename, std::vector<uint32_t> triangleIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals);
void saveMeshStlAscii(
        const std::string& filename,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals);

/*
 * Saves the triangle mesh in the binary .stl format.
 */
void saveMeshStlBinary(
        const std::string& filename, std::vector<uint32_t> triangleIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals);
void saveMeshStlBinary(
        const std::string& filename,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals);

/*
 * Saves the triangle mesh in the .obj format.
 */
void saveMeshObj(
        const std::string& filename, std::vector<uint32_t> triangleIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals);
void saveMeshObj(
        const std::string& filename,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals);

#endif //CORRERENDER_WRITEMESH_HPP
