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

#include <fstream>
#include <Utils/Events/Stream/Stream.hpp>
#include "Volume/VolumeData.hpp"
#include "CvolWriter.hpp"

bool CvolWriter::writeFieldToFile(
        const std::string& filePath, VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx) {
    std::ofstream outfile(filePath, std::ios::binary);
    if (!outfile.is_open()) {
        sgl::Logfile::get()->writeError(
                "Error in CvolWriter::writeFieldToFile: File \"" + filePath + "\" could not be opened for writing.");
        return false;
    }

    CvolFileHeader header{};
    memcpy(header.magicNumber, "cvol", 4);
    header.sizeX = size_t(volumeData->getGridSizeX());
    header.sizeY = size_t(volumeData->getGridSizeY());
    header.sizeZ = size_t(volumeData->getGridSizeZ());
    header.voxelSizeX = double(volumeData->getDx());
    header.voxelSizeY = double(volumeData->getDy());
    header.voxelSizeZ = double(volumeData->getDz());
    header.fieldType = CvolDataType::FLOAT;
    outfile.write(reinterpret_cast<char*>(&header), sizeof(CvolFileHeader));

    auto fieldData = volumeData->getFieldEntryCpu(fieldType, fieldName, timestepIdx, memberIdx);
    outfile.write(reinterpret_cast<char*>(fieldData.get()), sizeof(float) * header.sizeX * header.sizeY * header.sizeZ);

    outfile.close();
    return true;
}
