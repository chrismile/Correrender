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

#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/FileLoader.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "LoadersUtil.hpp"
#include "CvolLoader.hpp"

bool CvolLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

    uint8_t* buffer = nullptr;
    size_t length = 0;
    size_t lengthRead = 0;
    bool loaded = sgl::loadFileFromSourceRanged(
            dataSourceFilename, buffer, lengthRead, sizeof(CvolFileHeader), length, true);
    if (!loaded) {
        sgl::Logfile::get()->writeError(
                "Error in CvolLoader::load: Couldn't open file \"" + dataSourceFilename + "\".");
        return false;
    }
    if (length < sizeof(CvolFileHeader)) {
        sgl::Logfile::get()->throwError(
                "Error in CvolLoader::load: Invalid file size for file \"" + dataSourceFilename + "\".");
    }
    fileHeader = *reinterpret_cast<CvolFileHeader*>(buffer);
    delete[] buffer;
    buffer = nullptr;

    numBytesData = length - sizeof(CvolFileHeader);

    if (memcmp(fileHeader.magicNumber, "cvol", 4) != 0) {
        sgl::Logfile::get()->throwError(
                "Error in CvolLoader::load: Invalid magic number in file \"" + dataSourceFilename + "\".");
    }

    if (fileHeader.fieldType != CvolDataType::UNSIGNED_CHAR
            && fileHeader.fieldType != CvolDataType::UNSIGNED_SHORT
            && fileHeader.fieldType != CvolDataType::FLOAT) {
        sgl::Logfile::get()->throwError(
                "Error in CvolLoader::load: Unsupported field type "
                + std::to_string(int(fileHeader.fieldType)) + " for file \"" + dataSourceFilename + "\".");
    }

    gridNumCellsTotal = fileHeader.sizeX * fileHeader.sizeY * fileHeader.sizeZ;
    if ((fileHeader.fieldType == CvolDataType::UNSIGNED_CHAR && numBytesData != gridNumCellsTotal * sizeof(uint8_t))
            || (fileHeader.fieldType == CvolDataType::UNSIGNED_SHORT && numBytesData != gridNumCellsTotal * sizeof(uint16_t))
            || (fileHeader.fieldType == CvolDataType::FLOAT && numBytesData != gridNumCellsTotal * sizeof(float))) {
        sgl::Logfile::get()->throwError(
                "Error in RbcBinFileLoader::load: Invalid number of entries for file \""
                + dataSourceFilename + "\".");
    }

    xs = int(fileHeader.sizeX);
    ys = int(fileHeader.sizeY);
    zs = int(fileHeader.sizeZ);
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float maxVoxelSize = std::max(
            float(fileHeader.voxelSizeX), std::max(float(fileHeader.voxelSizeY), float(fileHeader.voxelSizeZ)));
    dx = float(fileHeader.voxelSizeX) / maxVoxelSize / maxDimension;
    dy = float(fileHeader.voxelSizeY) / maxVoxelSize / maxDimension;
    dz = float(fileHeader.voxelSizeZ) / maxVoxelSize / maxDimension;
    volumeData->setGridExtent(xs, ys, zs, dx, dy, dz);

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    fieldNameMap[FieldType::SCALAR].emplace_back("Scalar Attribute");
    volumeData->setFieldNames(fieldNameMap);

    return true;
}

bool CvolLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    uint8_t* buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(dataSourceFilename, buffer, length, true);
    if (!loaded) {
        sgl::Logfile::get()->throwError(
                "Error in CvolLoader::getFieldEntry: Couldn't open file \"" + dataSourceFilename + "\".");
    }
    if (length < sizeof(CvolFileHeader)) {
        sgl::Logfile::get()->throwError(
                "Error in CvolLoader::getFieldEntry: Invalid file size for file \"" + dataSourceFilename + "\".");
    }
    auto* bufferRaw = reinterpret_cast<uint8_t*>(buffer + sizeof(CvolFileHeader));

    size_t totalSize = size_t(xs) * size_t(ys) * size_t(zs);
    if (fileHeader.fieldType == CvolDataType::FLOAT) {
        auto* fieldEntryBuffer = new float[totalSize];
        memcpy(fieldEntryBuffer, bufferRaw, sizeof(float) * totalSize);
        fieldEntry = new HostCacheEntryType(totalSize, fieldEntryBuffer);
    } else if (fileHeader.fieldType == CvolDataType::UNSIGNED_CHAR) {
        auto* fieldEntryBuffer = new uint8_t[totalSize];
        memcpy(fieldEntryBuffer, bufferRaw, sizeof(uint8_t) * totalSize);
        fieldEntry = new HostCacheEntryType(totalSize, fieldEntryBuffer);
    } else if (fileHeader.fieldType == CvolDataType::UNSIGNED_SHORT) {
        auto* fieldEntryBuffer = new uint16_t[totalSize];
        memcpy(fieldEntryBuffer, bufferRaw, sizeof(uint16_t) * totalSize);
        fieldEntry = new HostCacheEntryType(totalSize, fieldEntryBuffer);
    }
    /*if (fileHeader.fieldType == CvolDataType::FLOAT) {
        memcpy(fieldEntryBuffer, bufferRaw, sizeof(float) * totalSize);
    } else if (fileHeader.fieldType == CvolDataType::UNSIGNED_CHAR) {
        auto* dataField = bufferRaw;
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, totalSize), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel for default(none) shared(fieldEntryBuffer, dataField, totalSize)
#endif
        for (size_t i = 0; i < totalSize; i++) {
#endif
            fieldEntryBuffer[i] = float(dataField[i]) / 255.0f;
        }
#ifdef USE_TBB
        });
#endif
    } else if (fileHeader.fieldType == CvolDataType::UNSIGNED_SHORT) {
        auto* dataField = reinterpret_cast<uint16_t*>(bufferRaw);
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, totalSize), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel for default(none) shared(fieldEntryBuffer, dataField, totalSize)
#endif
        for (size_t i = 0; i < totalSize; i++) {
#endif
            fieldEntryBuffer[i] = float(dataField[i]) / 65535.0f;
        }
#ifdef USE_TBB
        });
#endif
    }*/

    delete[] buffer;
    return true;
}
