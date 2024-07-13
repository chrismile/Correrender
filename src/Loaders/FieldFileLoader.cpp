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

#include <Math/half/half.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/FileLoader.hpp>

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "LoadersUtil.hpp"
#include "FieldFileLoader.hpp"

bool FieldFileLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

    uint8_t* buffer = nullptr;
    size_t length = 0;
    size_t lengthRead = 0;
    bool loaded = sgl::loadFileFromSourceRanged(
            dataSourceFilename, buffer, lengthRead, sizeof(FieldFileHeader), length, true);
    if (!loaded) {
        sgl::Logfile::get()->writeError(
                "Error in FieldFileLoader::load: Couldn't open file \"" + dataSourceFilename + "\".");
        return false;
    }
    if (length < sizeof(FieldFileHeader)) {
        sgl::Logfile::get()->throwError(
                "Error in FieldFileLoader::load: Invalid file size for file \"" + dataSourceFilename + "\".");
    }
    fileHeader = *reinterpret_cast<FieldFileHeader*>(buffer);
    dataOffset = sizeof(FieldFileHeader);
    delete[] buffer;
    buffer = nullptr;

    std::string filenameRawLower = sgl::FileUtils::get()->getPureFilename(dataSourceFilename);
    sgl::toLower(filenameRawLower);
    numBytesData = length - sizeof(FieldFileHeader);
    if (filenameRawLower.find("uvwp") != std::string::npos) {
        fileHeader.dimensions = 3;
        fileHeader.mipLevels = 3;
        fileHeader.fieldType = 0;
        dataOffset = sizeof(glm::uvec3);
        numBytesData = length - sizeof(glm::uvec3);
    }

    if (fileHeader.fieldType != 0 && fileHeader.fieldType != 1) {
        sgl::Logfile::get()->throwError(
                "Error in FieldFileLoader::load: Unsupported field type "
                + std::to_string(fileHeader.fieldType) + " for file \"" + dataSourceFilename + "\".");
    }
    if (fileHeader.dimensions != 3) {
        sgl::Logfile::get()->throwError(
                "Error in FieldFileLoader::load: Unsupported number of dimensions "
                + std::to_string(fileHeader.dimensions) + " for file \"" + dataSourceFilename + "\".");
    }
    //if (fileHeader.mipLevels != 1) {
    //    sgl::Logfile::get()->throwError(
    //            "Error in FieldFileLoader::load: Unsupported number of mip levels "
    //            + std::to_string(fileHeader.mipLevels) + " for file \"" + dataSourceFilename + "\".");
    //}

    gridNumCellsTotal =
            size_t(fileHeader.resolution.x) * size_t(fileHeader.resolution.y) * size_t(fileHeader.resolution.z);
    if (fileHeader.fieldType == 0) {
        if (numBytesData != gridNumCellsTotal * sizeof(glm::vec3) && numBytesData != gridNumCellsTotal * sizeof(glm::vec4)) {
            sgl::Logfile::get()->throwError(
                    "Error in RbcBinFileLoader::load: Invalid number of entries for file \""
                    + dataSourceFilename + "\".");
        }
    } else /* if (fileHeader.fieldType == 1) */ {
        if (numBytesData != gridNumCellsTotal * sizeof(uint16_t) * 3 && numBytesData != gridNumCellsTotal * sizeof(uint16_t) * 4) {
            sgl::Logfile::get()->throwError(
                    "Error in RbcBinFileLoader::load: Invalid number of entries for file \""
                    + dataSourceFilename + "\".");
        }
    }

    xs = int(fileHeader.resolution.x);
    ys = int(fileHeader.resolution.y);
    zs = int(fileHeader.resolution.z);
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    cellStep = 1.0f / maxDimension;
    volumeData->setGridExtent(xs, ys, zs, cellStep, cellStep, cellStep);

    bool hasScalarAttributeField =
            (fileHeader.fieldType == 0 && numBytesData == gridNumCellsTotal * sizeof(glm::vec4))
            || (fileHeader.fieldType == 1 && numBytesData == gridNumCellsTotal * sizeof(uint16_t) * 4);

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    fieldNameMap[FieldType::VECTOR].emplace_back("Velocity");
    fieldNameMap[FieldType::VECTOR].emplace_back("Vorticity");
    fieldNameMap[FieldType::SCALAR].emplace_back("Velocity Magnitude");
    fieldNameMap[FieldType::SCALAR].emplace_back("Vorticity Magnitude");
    fieldNameMap[FieldType::SCALAR].emplace_back("Helicity");
    if (hasScalarAttributeField) {
        // Make an educated guess about the type of the attribute.
        if (filenameRawLower.find("borromean") != std::string::npos
            || filenameRawLower.find("magnet") != std::string::npos) {
            scalarAttributeName = "Field Strength";
        } else if (filenameRawLower.find("uvwp") != std::string::npos) {
            scalarAttributeName = "Pressure";
        } else {
            scalarAttributeName = "Scalar Attribute";
        }
        fieldNameMap[FieldType::SCALAR].push_back(scalarAttributeName);
    }
    volumeData->setFieldNames(fieldNameMap);

    return true;
}

bool FieldFileLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    uint8_t* buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(dataSourceFilename, buffer, length, true);
    if (!loaded) {
        sgl::Logfile::get()->throwError(
                "Error in FieldFileLoader::getFieldEntry: Couldn't open file \"" + dataSourceFilename + "\".");
    }
    if (length < 24) {
        sgl::Logfile::get()->throwError(
                "Error in FieldFileLoader::getFieldEntry: Invalid file size for file \"" + dataSourceFilename + "\".");
    }
    auto* dataField = reinterpret_cast<float*>(buffer + dataOffset);

    int vectorFieldNumEntries = xs * ys * zs * 3;
    int scalarFieldNumEntries = xs * ys * zs;

    auto* velocityField = new float[vectorFieldNumEntries];
    auto* velocityMagnitudeField = new float[scalarFieldNumEntries];
    auto* vorticityField = new float[vectorFieldNumEntries];
    auto* vorticityMagnitudeField = new float[scalarFieldNumEntries];
    auto* helicityField = new float[scalarFieldNumEntries];
    float* scalarAttributeField = nullptr;

    if (fileHeader.fieldType == 0) {
        if (numBytesData == gridNumCellsTotal * sizeof(glm::vec3)) {
            for (int z = 0; z < zs; z++) {
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        velocityField[IDXV(x, y, z, 0)] = dataField[IDXV(x, y, z, 0)];
                        velocityField[IDXV(x, y, z, 1)] = dataField[IDXV(x, y, z, 1)];
                        velocityField[IDXV(x, y, z, 2)] = dataField[IDXV(x, y, z, 2)];
                    }
                }
            }
        } else if (numBytesData == gridNumCellsTotal * sizeof(glm::vec4)) {
            scalarAttributeField = new float[scalarFieldNumEntries];
            for (int z = 0; z < zs; z++) {
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        velocityField[IDXV(x, y, z, 0)] = dataField[IDXV4(x, y, z, 0)];
                        velocityField[IDXV(x, y, z, 1)] = dataField[IDXV4(x, y, z, 1)];
                        velocityField[IDXV(x, y, z, 2)] = dataField[IDXV4(x, y, z, 2)];
                        scalarAttributeField[IDXS(x, y, z)] = dataField[IDXV4(x, y, z, 3)];
                    }
                }
            }
        }
    } else /* if (fileHeader.fieldType == 1) */ {
        auto* dataFieldHalf = reinterpret_cast<HalfFloat*>(dataField);
        if (numBytesData == gridNumCellsTotal * sizeof(uint16_t) * 3) {
            for (int z = 0; z < zs; z++) {
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        velocityField[IDXV(x, y, z, 0)] = float(dataFieldHalf[IDXV(x, y, z, 0)]);
                        velocityField[IDXV(x, y, z, 1)] = float(dataFieldHalf[IDXV(x, y, z, 1)]);
                        velocityField[IDXV(x, y, z, 2)] = float(dataFieldHalf[IDXV(x, y, z, 2)]);
                    }
                }
            }
        } else if (numBytesData == gridNumCellsTotal * sizeof(uint16_t) * 4) {
            scalarAttributeField = new float[scalarFieldNumEntries];
            for (int z = 0; z < zs; z++) {
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        velocityField[IDXV(x, y, z, 0)] = float(dataFieldHalf[IDXV4(x, y, z, 0)]);
                        velocityField[IDXV(x, y, z, 1)] = float(dataFieldHalf[IDXV4(x, y, z, 1)]);
                        velocityField[IDXV(x, y, z, 2)] = float(dataFieldHalf[IDXV4(x, y, z, 2)]);
                        scalarAttributeField[IDXS(x, y, z)] = float(dataFieldHalf[IDXV4(x, y, z, 3)]);
                    }
                }
            }
        }
    }

    computeVectorMagnitudeField(velocityField, velocityMagnitudeField, xs, ys, zs);
    computeVorticityField(velocityField, vorticityField, xs, ys, zs, cellStep, cellStep, cellStep);
    computeVectorMagnitudeField(vorticityField, vorticityMagnitudeField, xs, ys, zs);
    computeHelicityFieldNormalized(
            velocityField, vorticityField, helicityField, xs, ys, zs,
            dataSetInformation.useNormalizedVelocity,
            dataSetInformation.useNormalizedVorticity);

    volumeData->addField(velocityField, FieldType::VECTOR, "Velocity", timestepIdx, memberIdx);
    volumeData->addField(vorticityField, FieldType::VECTOR, "Vorticity", timestepIdx, memberIdx);
    volumeData->addField(helicityField, FieldType::SCALAR, "Helicity", timestepIdx, memberIdx);
    volumeData->addField(velocityMagnitudeField, FieldType::SCALAR, "Velocity Magnitude", timestepIdx, memberIdx);
    volumeData->addField(vorticityMagnitudeField, FieldType::SCALAR, "Vorticity Magnitude", timestepIdx, memberIdx);
    if (scalarAttributeField) {
        volumeData->addField(scalarAttributeField, FieldType::SCALAR, scalarAttributeName, timestepIdx, memberIdx);
    }

    delete[] buffer;
    return true;
}
