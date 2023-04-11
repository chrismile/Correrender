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

#ifndef CORRERENDER_DATRAWFILELOADER_HPP
#define CORRERENDER_DATRAWFILELOADER_HPP

#include "VolumeLoader.hpp"

/**
 * Loader for pairs of .dat and .raw files. .dat files store metadata about the grid, while .raw files store the data.
 * .dat files contain case-insensitive pairs of keys and values separated by a colon.
 * Example for such a key-value pair: "ObjectFileName: rbc_convection_50.raw"
 *
 * Below, a list of valid key-value pairs can be found.
 * ObjectFileName: <string> (location of the .raw data file relative to the .dat file)
 * ObjectIndices: <uint-triplet> (triplet start-stop-step of indices to use if the ObjectFileName contains a format string like "%02i").
 * TaggedFileName: <string> (optional)
 * MeshFileName: <string> (optional)
 * Name: <string> (optional)
 * Resolution: <uint> <uint> <uint>
 * Format: uchar | ushort | float | float3 | float4 (only float3 and float4 supported!)
 * SliceThickness: <float> <float> <float>
 * Range: <float> <float> (optional)
 * NbrTags: <uint> (optional)
 * ObjectType: <enum-string>=TEXTURE_VOLUME_OBJECT (optional)
 * ObjectModel: RGBA (optional)
 * GridType: <enum-string>=EQUIDISTANT (optional)
 * Timestep: <float> (optional)
 */
class DatRawFileLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "dat", "raw" }; }
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;
    bool getHasFloat32Data() override { return bytesPerEntry == 4; }

private:
    std::string dataSourceFilename;
    DataSetInformation dataSetInformation;
    std::string datFilePath;
    std::vector<std::string> rawFilePaths;
    int xs = 0, ys = 0, zs = 0;
    float cellStep = 0.0f;
    int numComponents = 0;
    size_t bytesPerEntry = 0;
    std::string formatString;
    std::string scalarAttributeName;
};

#endif //CORRERENDER_DATRAWFILELOADER_HPP
