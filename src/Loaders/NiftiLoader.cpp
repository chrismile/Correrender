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

#include <boost/algorithm/string/case_conv.hpp>

#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/FileLoader.hpp>

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "LoadersUtil.hpp"
#include "nifti/nifti1.h"
#include "NiftiLoader.hpp"

NiftiLoader::~NiftiLoader() {
    if (header) {
        delete[] header;
        header = nullptr;
    }
}

bool NiftiLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

    uint8_t* buffer = nullptr;
    size_t length = 0;
    size_t lengthRead = 0;
    bool loaded = sgl::loadFileFromSourceRanged(
            dataSourceFilename, buffer, lengthRead, sizeof(nifti_1_header), length, true);
    if (!loaded) {
        sgl::Logfile::get()->writeError(
                "Error in NiftiLoader::load: Couldn't open file \"" + dataSourceFilename + "\".");
        return false;
    }
    if (length < sizeof(nifti_1_header)) {
        sgl::Logfile::get()->throwError(
                "Error in NiftiLoader::load: Invalid file size for file \"" + dataSourceFilename + "\".");
    }
    header = reinterpret_cast<nifti_1_header*>(buffer);
    dataOffset = ptrdiff_t(header->vox_offset);

    std::string filenameRawLower = sgl::FileUtils::get()->getPureFilename(dataSourceFilename);
    boost::to_lower(filenameRawLower);

    if (header->dim[0] != 3) {
        sgl::Logfile::get()->throwError(
                "Error in NiftiLoader::load: Invalid number of dimensions for file \""
                + dataSourceFilename + "\".");
    }

    xs = int(header->dim[1]);
    ys = int(header->dim[2]);
    zs = int(header->dim[3]);
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    dx = dy = dz = cellStep;
    float sx = std::abs(header->srow_x[0]);
    float sy = std::abs(header->srow_y[1]);
    float sz = std::abs(header->srow_z[2]);
    if (!std::isnan(sx) && !std::isnan(sy) && !std::isnan(sz)) {
        dx *= sx;
        dy *= sy;
        dz *= sz;
    }
    volumeData->setGridExtent(xs, ys, zs, dx, dy, dz);

    // Make an educated guess about the type of the attribute.
    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    if (filenameRawLower.find("volume") != std::string::npos) {
        scalarAttributeName = "Density";
    } else if (filenameRawLower.find("label") != std::string::npos) {
        scalarAttributeName = "Labels";
    } else {
        scalarAttributeName = "Scalar";
    }
    fieldNameMap[FieldType::SCALAR].push_back(scalarAttributeName);
    volumeData->setFieldNames(fieldNameMap);

    return true;
}

bool NiftiLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    uint8_t* buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(dataSourceFilename, buffer, length, true);
    if (!loaded) {
        sgl::Logfile::get()->throwError(
                "Error in NiftiLoader::getFieldEntry: Couldn't open file \"" + dataSourceFilename + "\".");
    }

    ptrdiff_t imageSizeInBytes = header->bitpix / 8;
    for (short i = 0; i < header->dim[0] && i < 7; i++) {
        imageSizeInBytes *= ptrdiff_t(header->dim[i + 1]);
    }
    if (dataOffset + imageSizeInBytes > ptrdiff_t(length)) {
        sgl::Logfile::get()->throwError(
                "Error in NiftiLoader::getFieldEntry: Invalid data size for file \"" + dataSourceFilename + "\".");
    }

    ScalarDataFormat dataFormat = ScalarDataFormat::FLOAT;
    if (header->datatype == DT_FLOAT) {
        dataFormat = ScalarDataFormat::FLOAT;
    } else if (header->datatype == DT_SIGNED_SHORT) {
        dataFormat = ScalarDataFormat::SHORT;
    } else if (header->datatype == DT_UNSIGNED_CHAR) {
        dataFormat = ScalarDataFormat::BYTE;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in NiftiLoader::getFieldEntry: Invalid data type in file \"" + dataSourceFilename + "\".");
    }

    auto* scalarAttributeField = new uint8_t[imageSizeInBytes];
    memcpy(scalarAttributeField, buffer + dataOffset, imageSizeInBytes);

    // TODO: value = header->scl_slope * valueOld + header->scl_inter
    if (std::abs(header->scl_slope - 1.0f) > 1e-4) {
        auto* scalarAttributeFieldOld = scalarAttributeField;
        scalarAttributeField = new uint8_t[xs * ys * zs * sizeof(float)];
        auto* scalarAttributeFieldFloat = reinterpret_cast<float*>(scalarAttributeField);
        if (dataFormat == ScalarDataFormat::FLOAT) {
            auto* scalarAttributeFieldOldFloat = reinterpret_cast<float*>(scalarAttributeFieldOld);
            int numEntries = xs * ys * zs;
            for (int i = 0; i < numEntries; i++) {
                scalarAttributeFieldFloat[i] = scalarAttributeFieldOldFloat[i];
            }
        } else if (dataFormat == ScalarDataFormat::SHORT) {
            auto* scalarAttributeFieldOldShort = reinterpret_cast<int16_t*>(scalarAttributeFieldOld);
            int numEntries = xs * ys * zs;
            for (int i = 0; i < numEntries; i++) {
                scalarAttributeFieldFloat[i] =
                        float(scalarAttributeFieldOldShort[i]) * header->scl_slope + header->scl_inter;
            }
        } else if (dataFormat == ScalarDataFormat::BYTE) {
            auto* scalarAttributeFieldOldByte = reinterpret_cast<uint8_t*>(scalarAttributeFieldOld);
            int numEntries = xs * ys * zs;
            for (int i = 0; i < numEntries; i++) {
                scalarAttributeFieldFloat[i] =
                        float(scalarAttributeFieldOldByte[i]) * header->scl_slope + header->scl_inter;
            }
        }
        dataFormat = ScalarDataFormat::FLOAT;
        delete[] scalarAttributeFieldOld;
    }

    volumeData->addField(
            scalarAttributeField, dataFormat, FieldType::SCALAR, scalarAttributeName, timestepIdx, memberIdx);

    delete[] buffer;
    return true;
}

bool NiftiLoader::getHasFloat32Data() {
    return header->datatype == DT_FLOAT;
}
