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

#ifndef CORRERENDER_FIELDFILELOADER_HPP
#define CORRERENDER_FIELDFILELOADER_HPP

#include "VolumeLoader.hpp"

struct FieldFileHeader {
    glm::uvec3 resolution;
    uint32_t dimensions;
    uint32_t mipLevels;
    uint32_t fieldType;
};

/**
 * Loader for the .field file format. The format stores data as follows.
 *
 * Header (24 bytes):
 * uvec3 resolution;
 * uint32_t dimensions;
 * uin32_t mipLevels;
 * uint32_t fieldType;
 *
 * At the moment, only field type 0 (32-bit float) is supported.
 * After the header, vector data with an optional scalar component is stored.
 */
class FieldFileLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "field" }; }
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;
    bool getHasFloat32Data() override { return fileHeader.fieldType == 0; }

private:
    std::string dataSourceFilename;
    DataSetInformation dataSetInformation;
    int xs = 0, ys = 0, zs = 0;
    float cellStep = 0.0f;
    FieldFileHeader fileHeader;
    ptrdiff_t dataOffset = 0;
    size_t numBytesData = 0;
    size_t gridNumCellsTotal = 0;
    std::string scalarAttributeName;
};

#endif //CORRERENDER_FIELDFILELOADER_HPP
