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

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
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
#include "MhdRawFileLoader.hpp"

struct FieldFileHeader {
    glm::uvec3 resolution;
    uint32_t dimensions;
    uint32_t mipLevels;
    uint32_t fieldType;
};

inline void existsAndEqual(
        const std::string& mhdFilePath, const std::map<std::string, std::string>& mhdDict,
        const std::string& key, const std::string& value) {
    auto it = mhdDict.find(key);
    if (it == mhdDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry '" + key + "' missing in \"" + mhdFilePath + "\".");
    }
    if (it->second != value) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry '" + key + "' is not equal to \"" + value + "\".");
    }
}

bool MhdRawFileLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

    if (boost::ends_with(dataSourceFilename, ".mhd")) {
        mhdFilePath = dataSourceFilename;
    }
    if (boost::ends_with(dataSourceFilename, ".raw")) {
        rawFilePath = dataSourceFilename;

        // We need to find the corresponding .mhd file.
        std::string rawFileDirectory = sgl::FileUtils::get()->getPathToFile(rawFilePath);
        std::vector<std::string> filesInDir = sgl::FileUtils::get()->getFilesInDirectoryVector(rawFileDirectory);
        for (const std::string& filePath : filesInDir) {
            if (boost::ends_with(filePath, ".mhd")) {
                mhdFilePath = filePath;
                break;
            }
        }
        if (mhdFilePath.empty()) {
            sgl::Logfile::get()->writeError(
                    "Error in MhdRawFileLoader::load: No .mhd file found for \"" + rawFilePath + "\".");
            return false;
        }
    }

    std::vector<int> timeSteps;

    // Load the .mhd metadata file.
    uint8_t* bufferMhd = nullptr;
    size_t lengthMhd = 0;
    bool loadedDat = sgl::loadFileFromSource(mhdFilePath, bufferMhd, lengthMhd, false);
    if (!loadedDat) {
        sgl::Logfile::get()->throwError(
                "Error in MhdRawFileLoader::load: Couldn't open file \"" + mhdFilePath + "\".");
    }
    char* fileBuffer = reinterpret_cast<char*>(bufferMhd);

    std::string lineBuffer;
    std::string stringBuffer;
    std::vector<std::string> splitLineString;
    std::map<std::string, std::string> mhdDict;
    for (size_t charPtr = 0; charPtr < lengthMhd; ) {
        lineBuffer.clear();
        while (charPtr < lengthMhd) {
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
        sgl::splitString(lineBuffer, '=', splitLineString);
        if (splitLineString.empty()) {
            continue;
        }
        if (splitLineString.size() != 2) {
            sgl::Logfile::get()->throwError(
                    "Error in MhdRawFileLoader::load: Invalid entry in file \"" + mhdFilePath + "\".");
        }

        std::string mhdKey = splitLineString.at(0);
        std::string mhdValue = splitLineString.at(1);
        boost::trim(mhdKey);
        //boost::to_lower(mhdKey);
        boost::trim(mhdValue);
        mhdDict.insert(std::make_pair(mhdKey, mhdValue));
    }

    // Next, process the metadata.
    if (rawFilePath.empty()) {
        auto it = mhdDict.find("ElementDataFile");
        if (it == mhdDict.end()) {
            sgl::Logfile::get()->throwError(
                    "Error in MhdRawFileLoader::load: Entry 'ElementDataFile' missing in \""
                    + mhdFilePath + "\".");
        }
        rawFilePath = it->second;
        bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(rawFilePath);
        if (!isAbsolutePath) {
            rawFilePath = sgl::FileUtils::get()->getPathToFile(mhdFilePath) + rawFilePath;
        }
    }

    existsAndEqual(mhdFilePath, mhdDict, "ObjectType", "Image");
    existsAndEqual(mhdFilePath, mhdDict, "NDims", "3");
    existsAndEqual(mhdFilePath, mhdDict, "Offset", "0 0 0");
    existsAndEqual(mhdFilePath, mhdDict, "TransformMatrix", "1 0 0 0 1 0 0 0 1");
    existsAndEqual(mhdFilePath, mhdDict, "CenterOfRotation", "0 0 0");
    existsAndEqual(mhdFilePath, mhdDict, "AnatomicalOrientation", "RAI");
    existsAndEqual(mhdFilePath, mhdDict, "BinaryData", "True");
    existsAndEqual(mhdFilePath, mhdDict, "BinaryDataByteOrderMSB", "False");
    existsAndEqual(mhdFilePath, mhdDict, "InterceptSlope", "0 1");
    existsAndEqual(mhdFilePath, mhdDict, "Modality", "MET_MOD_OTHER");
    existsAndEqual(mhdFilePath, mhdDict, "SegmentationType", "UNKNOWN");

    auto itResolution = mhdDict.find("DimSize");
    if (itResolution == mhdDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in MhdRawFileLoader::load: Entry 'DimSize' missing in \"" + mhdFilePath + "\".");
    }
    std::vector<std::string> resolutionSplit;
    sgl::splitStringWhitespace(itResolution->second, resolutionSplit);
    if (resolutionSplit.size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in MhdRawFileLoader::load: Entry 'DimSize' in \"" + mhdFilePath
                + "\" does not have three values.");
    }
    xs = int(sgl::fromString<int>(resolutionSplit.at(0)));
    ys = int(sgl::fromString<int>(resolutionSplit.at(1)));
    zs = int(sgl::fromString<int>(resolutionSplit.at(2)));
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    dx = dy = dz = cellStep;

    auto itSliceThickness = mhdDict.find("ElementSpacing");
    if (itSliceThickness != mhdDict.end()) {
        std::vector<std::string> sliceThicknessList;
        sgl::splitStringWhitespace(itSliceThickness->second, sliceThicknessList);
        if (sliceThicknessList.size() != 3) {
            sgl::Logfile::get()->throwError(
                    "Error in MhdRawFileLoader::load: Inconsistent entry 'ElementSpacing' in \"" + mhdFilePath + "\".");
        }
        auto tx = sgl::fromString<float>(sliceThicknessList.at(0));
        auto ty = sgl::fromString<float>(sliceThicknessList.at(1));
        auto tz = sgl::fromString<float>(sliceThicknessList.at(2));
        dx *= tx;
        dy *= ty;
        dz *= tz;
    }

    auto itFormat = mhdDict.find("ElementType");
    if (itFormat == mhdDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in MhdRawFileLoader::load: Entry 'ElementType' missing in \"" + mhdFilePath + "\".");
    }
    formatString = itFormat->second;
    if (formatString == "MET_FLOAT") {
        numComponents = 1;
        bytesPerEntry = 4;
    } else if (formatString == "MET_UCHAR") {
        numComponents = 1;
        bytesPerEntry = 1;
    } else if (formatString == "MET_USHORT") {
        numComponents = 1;
        bytesPerEntry = 2;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in MhdRawFileLoader::load: Unsupported format '" + formatString + "' in file \""
                + mhdFilePath + "\".");
    }

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;

    if (numComponents == 1 || numComponents == 4) {
        // Make an educated guess about the type of the attribute.
        if (numComponents > 1) {
            scalarAttributeName = "Scalar Attribute";
        } else {
            scalarAttributeName = "Density";
        }
        fieldNameMap[FieldType::SCALAR].push_back(scalarAttributeName);
    }

    volumeData->setGridExtent(xs, ys, zs, dx, dy, dz);
    if (!timeSteps.empty()) {
        volumeData->setTimeSteps(timeSteps);
    }
    volumeData->setFieldNames(fieldNameMap);

    delete[] bufferMhd;
    return true;
}

bool MhdRawFileLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    // Finally, load the data from the .raw file.
    uint8_t* bufferRaw = nullptr;
    size_t lengthRaw = 0;
    std::string rawFilename = rawFilePath;
    bool loadedRaw = sgl::loadFileFromSource(rawFilename, bufferRaw, lengthRaw, true);
    if (!loadedRaw) {
        sgl::Logfile::get()->throwError(
                "Error in MhdRawFileLoader::load: Couldn't open file \"" + rawFilename + "\".");
    }

    size_t numBytesData = lengthRaw;
    size_t totalSize = size_t(xs) * size_t(ys) * size_t(zs);
    if (numBytesData != totalSize * numComponents * bytesPerEntry) {
        sgl::Logfile::get()->throwError(
                "Error in MhdRawFileLoader::load: Invalid number of entries for file \"" + rawFilename + "\".");
    }

    auto scalarFieldNumEntries = size_t(xs) * size_t(ys) * size_t(zs);
    void* scalarAttributeField = nullptr;
    ScalarDataFormat dataFormat = ScalarDataFormat::FLOAT;
    if (formatString == "MET_FLOAT") {
        scalarAttributeField = new float[scalarFieldNumEntries];
        memcpy(scalarAttributeField, bufferRaw, sizeof(float) * totalSize);
    } else if (formatString == "MET_UCHAR") {
        dataFormat = ScalarDataFormat::BYTE;
        scalarAttributeField = new uint8_t[scalarFieldNumEntries];
        memcpy(scalarAttributeField, bufferRaw, sizeof(uint8_t) * totalSize);
    } else if (formatString == "MET_USHORT") {
        dataFormat = ScalarDataFormat::SHORT;
        scalarAttributeField = new uint16_t[scalarFieldNumEntries];
        memcpy(scalarAttributeField, bufferRaw, sizeof(uint16_t) * totalSize);
    }

    if (scalarAttributeField) {
        volumeData->addField(
                scalarAttributeField, dataFormat, FieldType::SCALAR, scalarAttributeName, timestepIdx, memberIdx);
    }

    delete[] bufferRaw;
    return true;
}
