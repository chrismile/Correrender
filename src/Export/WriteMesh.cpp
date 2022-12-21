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

#include <iostream>
#include <fstream>
#include <glm/glm.hpp>

#include <Utils/File/Logfile.hpp>
#include <Utils/Events/Stream/Stream.hpp>
#include <Utils/Mesh/IndexMesh.hpp>

#include "WriteMesh.hpp"

void saveMeshStlAscii(
        const std::string& filename, std::vector<uint32_t> triangleIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        sgl::Logfile::get()->writeError(
                "Error in saveMeshStlAscii: File \"" + filename + "\" could not be opened for writing.");
        return;
    }

    std::string fileContent = "solid printmesh\n\n";

    for (size_t i = 0; i < triangleIndices.size(); i += 3) {
        // Compute the facet normal (ignore stored normal data)
        glm::vec3 v0 = vertexPositions[triangleIndices[i]];
        glm::vec3 v1 = vertexPositions[triangleIndices[i + 1]];
        glm::vec3 v2 = vertexPositions[triangleIndices[i + 2]];
        glm::vec3 dir0 = v1 - v0;
        glm::vec3 dir1 = v2 - v0;
        glm::vec3 facetNormal = glm::normalize(glm::cross(dir0, dir1));

        fileContent +=
                "facet normal " + std::to_string(facetNormal.x)
                + " " + std::to_string(facetNormal.y) + " " + std::to_string(facetNormal.z) + "\n";
        fileContent += "\touter loop\n";
        for (size_t j = 0; j < 3; j++) {
            glm::vec3 vertex = vertexPositions[triangleIndices[i + j]];
            fileContent +=
                    "\t\tvertex " + std::to_string(vertex.x) + " "
                    + std::to_string(vertex.y) + " " + std::to_string(vertex.z) + "\n";
        }
        fileContent += "\tendloop\nendfacet\n";
    }

    fileContent = fileContent + "\nendsolid printmesh\n";

    outfile << fileContent;
    outfile.close();
}

void saveMeshStlAscii(
        const std::string& filename,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        sgl::Logfile::get()->writeError(
                "Error in saveMeshStlAscii: File \"" + filename + "\" could not be opened for writing.");
        return;
    }

    std::string fileContent = "solid printmesh\n\n";

    for (size_t i = 0; i < vertexPositions.size(); i += 3) {
        // Compute the facet normal (ignore stored normal data)
        glm::vec3 v0 = vertexPositions[i];
        glm::vec3 v1 = vertexPositions[i + 1];
        glm::vec3 v2 = vertexPositions[i + 2];
        glm::vec3 dir0 = v1 - v0;
        glm::vec3 dir1 = v2 - v0;
        glm::vec3 facetNormal = glm::normalize(glm::cross(dir0, dir1));

        fileContent +=
                "facet normal " + std::to_string(facetNormal.x)
                + " " + std::to_string(facetNormal.y) + " " + std::to_string(facetNormal.z) + "\n";
        fileContent += "\touter loop\n";
        for (size_t j = 0; j < 3; j++) {
            glm::vec3 vertex = vertexPositions[i + j];
            fileContent +=
                    "\t\tvertex " + std::to_string(vertex.x) + " "
                    + std::to_string(vertex.y) + " " + std::to_string(vertex.z) + "\n";
        }
        fileContent += "\tendloop\nendfacet\n";
    }

    fileContent = fileContent + "\nendsolid printmesh\n";

    outfile << fileContent;
    outfile.close();
}


void saveMeshStlBinary(
        const std::string& filename, std::vector<uint32_t> triangleIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        sgl::Logfile::get()->writeError(
                "Error in saveMeshStlBinary: File \"" + filename + "\" could not be opened for writing.");
        return;
    }

    size_t numTriangles = triangleIndices.size() / 3;
    size_t fileSizeBytes = 80 + 4 + ((4 * 3 * 4 + 2) * numTriangles);
    sgl::BinaryWriteStream stream(fileSizeBytes);

    // Write empty header
    for (int i = 0; i < 20; i++) {
        stream.write<uint32_t>(0);
    }
    // Write number of triangles
    stream.write<uint32_t>(uint32_t(triangleIndices.size() / 3));

    // Write all facets
    for (size_t i = 0; i < triangleIndices.size(); i += 3) {
        // Compute the facet normal (ignore stored normal data)
        glm::vec3 v0 = vertexPositions[triangleIndices[i]];
        glm::vec3 v1 = vertexPositions[triangleIndices[i + 1]];
        glm::vec3 v2 = vertexPositions[triangleIndices[i + 2]];
        glm::vec3 dir0 = v1 - v0;
        glm::vec3 dir1 = v2 - v0;
        glm::vec3 facetNormal = glm::normalize(glm::cross(dir0, dir1));

        stream.write(facetNormal);
        //stream.write(v0);
        //stream.write(v1);
        //stream.write(v2);
        stream.write(v2);
        stream.write(v1);
        stream.write(v0);
        stream.write<uint16_t>(0);
    }

    outfile.write((const char*)stream.getBuffer(), std::streamsize(stream.getSize()));
    outfile.close();
}

void saveMeshStlBinary(
        const std::string& filename,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
        sgl::Logfile::get()->writeError(
                "Error in saveMeshStlBinary: File \"" + filename + "\" could not be opened for writing.");
        return;
    }

    size_t numTriangles = vertexPositions.size() / 3;
    size_t fileSizeBytes = 80 + 4 + ((4 * 3 * 4 + 2) * numTriangles);
    sgl::BinaryWriteStream stream(fileSizeBytes);

    // Write empty header
    for (int i = 0; i < 20; i++) {
        stream.write<uint32_t>(0);
    }
    // Write number of triangles
    stream.write<uint32_t>(uint32_t(vertexPositions.size() / 3));

    // Write all facets
    for (size_t i = 0; i < vertexPositions.size(); i += 3) {
        // Compute the facet normal (ignore stored normal data)
        glm::vec3 v0 = vertexPositions[i];
        glm::vec3 v1 = vertexPositions[i + 1];
        glm::vec3 v2 = vertexPositions[i + 2];
        glm::vec3 dir0 = v1 - v0;
        glm::vec3 dir1 = v2 - v0;
        glm::vec3 facetNormal = glm::normalize(glm::cross(dir0, dir1));

        stream.write(facetNormal);
        //stream.write(v0);
        //stream.write(v1);
        //stream.write(v2);
        stream.write(v2);
        stream.write(v1);
        stream.write(v0);
        stream.write<uint16_t>(0);
    }

    outfile.write((const char*)stream.getBuffer(), std::streamsize(stream.getSize()));
    outfile.close();
}


void saveMeshObj(
        const std::string& filename, std::vector<uint32_t> triangleIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals) {
    std::string fileContent = "o printmesh\ns 1\n";

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        sgl::Logfile::get()->writeError(
                "Error in saveMeshObj: File \"" + filename + "\" could not be opened for writing.");
        return;
    }

    // Output vertices and normals
    for (size_t i = 0; i < vertexPositions.size(); i++) {
        const glm::vec3& vertex = vertexPositions[i];
        fileContent +=
                "v " + std::to_string(vertex.x) + " " + std::to_string(vertex.y) + " "
                + std::to_string(vertex.z) + "\n";
        if (!vertexNormals.empty()) {
            const glm::vec3& normal = vertexNormals[i];
            fileContent +=
                    "vn " + std::to_string(normal.x) + " " + std::to_string(normal.y) + " "
                    + std::to_string(normal.z) + "\n";
        }
    }

    // Output triangle faces
    for (size_t i = 0; i < triangleIndices.size(); i += 3) {
        uint32_t i1 = triangleIndices[i] + 1;
        uint32_t i2 = triangleIndices[i+1] + 1;
        uint32_t i3 = triangleIndices[i+2] + 1;
        fileContent +=
                "f " + std::to_string(i1) + "//" + std::to_string(i1) + " " + std::to_string(i2) + "//"
                + std::to_string(i2) + " " + std::to_string(i3) + "//" + std::to_string(i3) + "\n";
    }

    outfile << fileContent;
    outfile.close();
}

void saveMeshObj(
        const std::string& filename,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec3>& vertexNormals) {
    std::vector<uint32_t> triangleIndices;
    std::vector<glm::vec3> vertexPositionsIndexed;
    std::vector<glm::vec3> vertexNormalsIndexed;
    sgl::computeSharedIndexRepresentation(
            vertexPositions, vertexNormals,
            triangleIndices, vertexPositionsIndexed, vertexNormalsIndexed);
    saveMeshObj(filename, triangleIndices, vertexPositionsIndexed, vertexNormalsIndexed);
}
