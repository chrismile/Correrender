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

#include <fstream>

#include <Utils/Events/Stream/Stream.hpp>
#include <Utils/File/Logfile.hpp>

#include "WriteTetMesh.hpp"

bool saveBinTet(
        const std::string& filePath, const std::vector<uint32_t>& cellIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec4>& vertexColors) {
#ifndef __MINGW32__
    std::ofstream file(filePath.c_str(), std::ofstream::binary);
    if (!file.is_open()) {
        sgl::Logfile::get()->writeError(
                std::string() + "Error in BinTetWriter::saveToFile: File \""
                + filePath + "\" could not be opened for writing.");
        return false;
    }
#else
    FILE* fileptr = fopen(filePath.c_str(), "wb");
    if (fileptr == NULL) {
        sgl::Logfile::get()->writeError(
                std::string() + "Error in BinTetWriter::saveToFile: File \""
                + filePath + "\" could not be opened for writing.");
        return false;
    }
#endif

    sgl::BinaryWriteStream stream;
    stream.write(1u); //< Version number.

    stream.writeArray(cellIndices);
    stream.writeArray(vertexPositions);
    stream.writeArray(vertexColors);

#ifndef __MINGW32__
    file.write((const char*)stream.getBuffer(), stream.getSize());
    file.close();
#else
    fwrite((const void*)stream.getBuffer(), stream.getSize(), 1, fileptr);
    fclose(fileptr);
#endif

    return true;
}

bool saveTxtTet(
        const std::string& filePath, const std::vector<uint32_t>& cellIndices,
        const std::vector<glm::vec3>& vertexPositions, const std::vector<glm::vec4>& vertexColors) {
    std::ofstream file(filePath.c_str(), std::ofstream::binary);
    if (!file.is_open()) {
        sgl::Logfile::get()->writeError(
                std::string() + "Error in TxtTetWriter::saveToFile: File \""
                + filePath + "\" could not be opened for writing.");
        return false;
    }

    file << "cellIndices " << (cellIndices.size() / 4) << "\n";
    for (size_t i = 0; i < cellIndices.size(); i += 4) {
        file << cellIndices.at(i) << " " << cellIndices.at(i + 1) << " "
             << cellIndices.at(i + 2) << " " << cellIndices.at(i + 3) << "\n";
    }

    file << "vertexPositions " << vertexPositions.size() << "\n";
    for (const auto& v : vertexPositions) {
        file << v.x << " " << v.y << " " << v.z << "\n";
    }

    file << "vertexColors " << vertexColors.size() << "\n";
    for (const auto& v : vertexColors) {
        file << v.x << " " << v.y << " " << v.z << " " << v.w << "\n";
    }

    file.close();
    return true;
}
