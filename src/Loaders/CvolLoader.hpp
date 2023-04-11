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

#ifndef CORRERENDER_CVOLLOADER_HPP
#define CORRERENDER_CVOLLOADER_HPP

#include "VolumeLoader.hpp"

enum class CvolDataType {
    UNSIGNED_CHAR, UNSIGNED_SHORT, FLOAT
};

#pragma pack(push, 4)
struct CvolFileHeader {
    char magicNumber[4];
    size_t sizeX, sizeY, sizeZ;
    double voxelSizeX, voxelSizeY, voxelSizeZ;
    CvolDataType fieldType;
    uint64_t padding;
};
#pragma pack(pop)

class CvolLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "cvol" }; }
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;
    bool getHasFloat32Data() override { return fileHeader.fieldType == CvolDataType::FLOAT; }

private:
    std::string dataSourceFilename;
    DataSetInformation dataSetInformation;
    int xs = 0, ys = 0, zs = 0;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    CvolFileHeader fileHeader;
    size_t numBytesData = 0;
    size_t gridNumCellsTotal = 0;
    std::string scalarAttributeName;
};

#endif //CORRERENDER_CVOLLOADER_HPP
