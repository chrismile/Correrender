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

#include <glm/vec3.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/Convert.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/FileLoader.hpp>

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "LoadersUtil.hpp"
#include "DatRawFileLoader.hpp"

struct FieldFileHeader {
    glm::uvec3 resolution;
    uint32_t dimensions;
    uint32_t mipLevels;
    uint32_t fieldType;
};

bool DatRawFileLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

    if (sgl::endsWith(dataSourceFilename, ".dat")) {
        datFilePath = dataSourceFilename;
    }
    if (sgl::endsWith(dataSourceFilename, ".raw")) {
        rawFilePaths.push_back(dataSourceFilename);

        // We need to find the corresponding .dat file.
        std::string rawFileDirectory = sgl::FileUtils::get()->getPathToFile(rawFilePaths.front());
        std::vector<std::string> filesInDir = sgl::FileUtils::get()->getFilesInDirectoryVector(rawFileDirectory);
        for (const std::string& filePath : filesInDir) {
            if (sgl::endsWith(filePath, ".dat")) {
                datFilePath = filePath;
                break;
            }
        }
        if (datFilePath.empty()) {
            sgl::Logfile::get()->writeError(
                    "Error in DatRawFileLoader::load: No .dat file found for \"" + rawFilePaths.front() + "\".");
            return false;
        }
    }

    std::vector<int> timeSteps;

    // Load the .dat metadata file.
    uint8_t* bufferDat = nullptr;
    size_t lengthDat = 0;
    bool loadedDat = sgl::loadFileFromSource(datFilePath, bufferDat, lengthDat, false);
    if (!loadedDat) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Couldn't open file \"" + datFilePath + "\".");
    }
    char* fileBuffer = reinterpret_cast<char*>(bufferDat);

    std::string lineBuffer;
    std::string stringBuffer;
    std::vector<std::string> splitLineString;
    std::map<std::string, std::string> datDict;
    for (size_t charPtr = 0; charPtr < lengthDat; ) {
        lineBuffer.clear();
        while (charPtr < lengthDat) {
            char currentChar = fileBuffer[charPtr];
            if (currentChar == '\n' || currentChar == '\r') {
                charPtr++;
                break;
            }
            lineBuffer.push_back(currentChar);
            charPtr++;
        }

        if (lineBuffer.empty()) {
            continue;
        }

        splitLineString.clear();
        sgl::splitString(lineBuffer, ':', splitLineString);
        if (splitLineString.empty()) {
            continue;
        }
        if (splitLineString.size() != 2) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: Invalid entry in file \"" + datFilePath + "\".");
        }

        std::string datKey = splitLineString.at(0);
        std::string datValue = splitLineString.at(1);
        sgl::stringTrim(datKey);
        sgl::toLower(datKey);
        sgl::stringTrim(datValue);
        datDict.insert(std::make_pair(datKey, datValue));
    }

    // Next, process the metadata.
    if (rawFilePaths.empty()) {
        auto it = datDict.find("objectfilename");
        if (it == datDict.end()) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: Entry 'ObjectFileName' missing in \""
                    + datFilePath + "\".");
        }
        auto itIndices = datDict.find("objectindices");
        if (itIndices != datDict.end()) {
            std::vector<std::string> indicesSplit;
            sgl::splitStringWhitespace(itIndices->second, indicesSplit);
            if (indicesSplit.size() != 2 && indicesSplit.size() != 3) {
                sgl::Logfile::get()->throwError(
                        "Error in DatRawFileLoader::load: Entry 'ObjectIndices' in \"" + datFilePath
                        + "\" does not have three values (start, stop, step).");
            }
            int start = int(sgl::fromString<int>(indicesSplit.at(0)));
            int stop = int(sgl::fromString<int>(indicesSplit.at(1)));
            int step = indicesSplit.size() == 2 ? 1 : int(sgl::fromString<int>(indicesSplit.at(2)));

            size_t bufferSize = it->second.size() + 100;
            for (int idx = start; idx <= stop; idx += step) {
                char* rawFilePathBuffer = new char[bufferSize];
                snprintf(rawFilePathBuffer, bufferSize, it->second.c_str(), idx);
                rawFilePaths.emplace_back(rawFilePathBuffer);
                timeSteps.push_back(idx);
                delete[] rawFilePathBuffer;
            }

            if (rawFilePaths.empty()) {
                sgl::Logfile::get()->throwError(
                        "Error in DatRawFileLoader::load: ObjectIndices found in file \"" + datFilePath
                        + "\" lead to empty set of file names.");
            }
        } else {
            rawFilePaths.push_back(it->second);
        }
        bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(rawFilePaths.front());
        if (!isAbsolutePath) {
            for (std::string& rawFilePath : rawFilePaths) {
                rawFilePath = sgl::FileUtils::get()->getPathToFile(datFilePath) + rawFilePath;
            }
        }
    }

    auto itResolution = datDict.find("resolution");
    if (itResolution == datDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Entry 'Resolution' missing in \"" + datFilePath + "\".");
    }
    std::vector<std::string> resolutionSplit;
    sgl::splitStringWhitespace(itResolution->second, resolutionSplit);
    if (resolutionSplit.size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Entry 'Resolution' in \"" + datFilePath
                + "\" does not have three values.");
    }
    xs = int(sgl::fromString<int>(resolutionSplit.at(0)));
    ys = int(sgl::fromString<int>(resolutionSplit.at(1)));
    zs = int(sgl::fromString<int>(resolutionSplit.at(2)));
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    dx = dy = dz = cellStep;

    auto itSliceThickness = datDict.find("slicethickness");
    if (itSliceThickness != datDict.end()) {
        std::vector<std::string> sliceThicknessList;
        sgl::splitStringWhitespace(itSliceThickness->second, sliceThicknessList);
        if (sliceThicknessList.size() != 3) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: Inconsistent entry 'SliceThickness' in \"" + datFilePath + "\".");
        }
        auto tx = sgl::fromString<float>(sliceThicknessList.at(0));
        auto ty = sgl::fromString<float>(sliceThicknessList.at(1));
        auto tz = sgl::fromString<float>(sliceThicknessList.at(2));
        dx *= tx;
        dy *= ty;
        dz *= tz;
    }

    auto itFormat = datDict.find("format");
    if (itFormat == datDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Entry 'Format' missing in \"" + datFilePath + "\".");
    }
    formatString = sgl::toLowerCopy(itFormat->second);
    if (formatString == "float") {
        numComponents = 1;
        bytesPerEntry = 4;
    } else if (formatString == "uchar" || formatString == "byte") {
        numComponents = 1;
        bytesPerEntry = 1;
    } else if (formatString == "ushort") {
        numComponents = 1;
        bytesPerEntry = 2;
    } else if (formatString == "float3") {
        numComponents = 3;
        bytesPerEntry = 4;
    } else if (formatString == "float4") {
        numComponents = 4;
        bytesPerEntry = 4;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Unsupported format '" + formatString + "' in file \""
                + datFilePath + "\".");
    }

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;

    if (numComponents == 1 || numComponents == 4) {
        // Make an educated guess about the type of the attribute.
        std::string filenameRawLower = sgl::FileUtils::get()->getPureFilename(dataSourceFilename);
        if (filenameRawLower.find("borromean") != std::string::npos
                || filenameRawLower.find("magnet") != std::string::npos) {
            scalarAttributeName = "Field Strength";
        } else if (numComponents > 1) {
            scalarAttributeName = "Scalar Attribute";
        } else {
            scalarAttributeName = "Density";
        }
        fieldNameMap[FieldType::SCALAR].push_back(scalarAttributeName);
    }

    if (numComponents > 1) {
        fieldNameMap[FieldType::VECTOR].emplace_back("Velocity");
        fieldNameMap[FieldType::VECTOR].emplace_back("Vorticity");
        fieldNameMap[FieldType::SCALAR].emplace_back("Velocity Magnitude");
        fieldNameMap[FieldType::SCALAR].emplace_back("Vorticity Magnitude");
        fieldNameMap[FieldType::SCALAR].emplace_back("Helicity");
    }

    volumeData->setGridExtent(xs, ys, zs, dx, dy, dz);
    if (!timeSteps.empty()) {
        volumeData->setTimeSteps(timeSteps);
    }
    volumeData->setFieldNames(fieldNameMap);

    delete[] bufferDat;
    return true;
}

bool DatRawFileLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    // Finally, load the data from the .raw file.
    uint8_t* bufferRaw = nullptr;
    size_t lengthRaw = 0;
    std::string rawFilename = rawFilePaths.at(rawFilePaths.size() == 1 ? 0 : timestepIdx);
    bool loadedRaw = sgl::loadFileFromSource(rawFilename, bufferRaw, lengthRaw, true);
    if (!loadedRaw) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Couldn't open file \"" + rawFilename + "\".");
    }

    size_t numBytesData = lengthRaw;
    size_t totalSize = size_t(xs) * size_t(ys) * size_t(zs);
    if (numBytesData != totalSize * numComponents * bytesPerEntry) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Invalid number of entries for file \"" + rawFilename + "\".");
    }

    auto vectorFieldNumEntries = size_t(xs) * size_t(ys) * size_t(zs) * 3;
    auto scalarFieldNumEntries = size_t(xs) * size_t(ys) * size_t(zs);

    float* velocityField = nullptr;
    float* velocityMagnitudeField = nullptr;
    float* vorticityField = nullptr;
    float* vorticityMagnitudeField = nullptr;
    float* helicityField = nullptr;
    void* scalarAttributeField = nullptr;

    if (numComponents > 1) {
        velocityField = new float[vectorFieldNumEntries];
        velocityMagnitudeField = new float[scalarFieldNumEntries];
        vorticityField = new float[vectorFieldNumEntries];
        vorticityMagnitudeField = new float[scalarFieldNumEntries];
        helicityField = new float[scalarFieldNumEntries];
    }

    ScalarDataFormat dataFormat = ScalarDataFormat::FLOAT;
    if (formatString == "float") {
        scalarAttributeField = new float[scalarFieldNumEntries];
        memcpy(scalarAttributeField, bufferRaw, sizeof(float) * totalSize);
    } else if (formatString == "uchar" || formatString == "byte") {
        dataFormat = ScalarDataFormat::BYTE;
        scalarAttributeField = new uint8_t[scalarFieldNumEntries];
        memcpy(scalarAttributeField, bufferRaw, sizeof(uint8_t) * totalSize);
    } else if (formatString == "ushort") {
        dataFormat = ScalarDataFormat::SHORT;
        scalarAttributeField = new uint16_t[scalarFieldNumEntries];
        memcpy(scalarAttributeField, bufferRaw, sizeof(uint16_t) * totalSize);
    } else if (formatString == "float3") {
        auto* dataField = reinterpret_cast<float*>(bufferRaw);
        for (int z = 0; z < zs; z++) {
            for (int y = 0; y < ys; y++) {
                for (int x = 0; x < xs; x++) {
                    velocityField[IDXV(x, y, z, 0)] = dataField[IDXV(x, y, z, 0)];
                    velocityField[IDXV(x, y, z, 1)] = dataField[IDXV(x, y, z, 1)];
                    velocityField[IDXV(x, y, z, 2)] = dataField[IDXV(x, y, z, 2)];
                }
            }
        }
    } else if (formatString == "float4") {
        auto* scalarAttributeFieldFloat = new float[scalarFieldNumEntries];
        scalarAttributeField = scalarAttributeFieldFloat;
        auto* dataField = reinterpret_cast<float*>(bufferRaw);
        for (int z = 0; z < zs; z++) {
            for (int y = 0; y < ys; y++) {
                for (int x = 0; x < xs; x++) {
                    velocityField[IDXV(x, y, z, 0)] = dataField[IDXV4(x, y, z, 0)];
                    velocityField[IDXV(x, y, z, 1)] = dataField[IDXV4(x, y, z, 1)];
                    velocityField[IDXV(x, y, z, 2)] = dataField[IDXV4(x, y, z, 2)];
                    scalarAttributeFieldFloat[IDXS(x, y, z)] = dataField[IDXV4(x, y, z, 3)];
                }
            }
        }
    }
    /*if (formatString == "float") {
        memcpy(scalarAttributeField, bufferRaw, sizeof(float) * totalSize);
    } else if (formatString == "uchar" || formatString == "byte") {
        auto* dataField = bufferRaw;
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, totalSize), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel for default(none) shared(scalarAttributeField, dataField, totalSize)
#endif
        for (size_t i = 0; i < totalSize; i++) {
#endif
            scalarAttributeField[i] = float(dataField[i]) / 255.0f;
        }
#ifdef USE_TBB
        });
#endif
    } else if (formatString == "ushort") {
        auto* dataField = reinterpret_cast<uint16_t*>(bufferRaw);
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, totalSize), [&](auto const& r) {
            for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel for default(none) shared(scalarAttributeField, dataField, totalSize)
#endif
        for (size_t i = 0; i < totalSize; i++) {
#endif
            scalarAttributeField[i] = float(dataField[i]) / 65535.0f;
        }
#ifdef USE_TBB
        });
#endif
    } else if (formatString == "float3") {
        auto* dataField = reinterpret_cast<float*>(bufferRaw);
        for (int z = 0; z < zs; z++) {
            for (int y = 0; y < ys; y++) {
                for (int x = 0; x < xs; x++) {
                    velocityField[IDXV(x, y, z, 0)] = dataField[IDXV(x, y, z, 0)];
                    velocityField[IDXV(x, y, z, 1)] = dataField[IDXV(x, y, z, 1)];
                    velocityField[IDXV(x, y, z, 2)] = dataField[IDXV(x, y, z, 2)];
                }
            }
        }
    } else if (formatString == "float4") {
        auto* dataField = reinterpret_cast<float*>(bufferRaw);
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
    }*/

    if (scalarAttributeField) {
        volumeData->addField(
                scalarAttributeField, dataFormat, FieldType::SCALAR, scalarAttributeName, timestepIdx, memberIdx);
    }

    if (numComponents > 1) {
        computeVectorMagnitudeField(velocityField, velocityMagnitudeField, xs, ys, zs);
        computeVorticityField(velocityField, vorticityField, xs, ys, zs, dx, dy, dz);
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
    }

    delete[] bufferRaw;
    return true;
}
