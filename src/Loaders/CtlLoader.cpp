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

#define _FILE_OFFSET_BITS 64
#define __USE_FILE_OFFSET64

#include <cstring>
#include <boost/algorithm/string/case_conv.hpp>

#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileLoader.hpp>

#include "Volume/VolumeData.hpp"
#include "LoadersUtil.hpp"
#include "CtlLoader.hpp"

CtlLoader::CtlLoader() = default;

CtlLoader::~CtlLoader() {
    if (file) {
        closeDataFile();
    }
}

bool CtlLoader::setInputFiles(
        VolumeData* volumeData, const std::string& _filePath, const DataSetInformation& _dataSetInformation) {
    dataSetInformation = _dataSetInformation;

    // Load the .dat metadata file.
    uint8_t* bufferCtl = nullptr;
    size_t lengthCtl = 0;
    bool loadedDat = sgl::loadFileFromSource(_filePath, bufferCtl, lengthCtl, false);
    if (!loadedDat) {
        sgl::Logfile::get()->throwError(
                "Error in CtlLoader::load: Couldn't open file \"" + _filePath + "\".");
    }
    char* fileBuffer = reinterpret_cast<char*>(bufferCtl);

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    int numVars = 0;
    bool isVarsMode = false;
    std::string lineBuffer;
    std::vector<std::string> splitLineString;
    for (size_t charPtr = 0; charPtr < lengthCtl; ) {
        lineBuffer.clear();
        while (charPtr < lengthCtl) {
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

        // Comments start with '*'.
        if (lineBuffer.front() == '*') {
            continue;
        }

        splitLineString.clear();
        sgl::splitStringWhitespace(lineBuffer, splitLineString);
        if (splitLineString.empty()) {
            continue;
        }

        std::string key = splitLineString.at(0);
        boost::to_lower(key);

        if ((!isVarsMode || key != "endvars")) {
            if (splitLineString.size() < 2) {
                sgl::Logfile::get()->throwError(
                        "Error in CtlLoader::load: Expected more parameters for command \"" + key + "\".");
                return false;
            }
        }

        if (isVarsMode) {
            if (key == "endvars") {
                isVarsMode = false;
                if (numVars != int(variableDescriptors.size())) {
                    sgl::Logfile::get()->throwError(
                            "Error in CtlLoader::load: Error in file \"" + _filePath
                            + "\": Mismatch in number of variables.");
                    return false;
                }
            } else {
                std::string variableId = splitLineString.at(0);
                CtlVarDesc varDesc;
                varDesc.name = variableId;
                varDesc.numLevels = sgl::fromString<int>(splitLineString.at(1));
                varDesc.numLevels = std::max(varDesc.numLevels, ptrdiff_t(1));
                variableNameMap.insert(std::make_pair(variableId, int(variableDescriptors.size())));
                variableDescriptors.push_back(varDesc);
                fieldNameMap[FieldType::SCALAR].push_back(variableId);
            }
        }
        if (key == "dset") {
            std::string dataFileName = splitLineString.at(1);
            bool isAbsolutePath;
            if (dataFileName.front() == '^') {
                dataFileName = dataFileName.substr(1);
                isAbsolutePath = false;
            } else {
                isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(dataFileName);
            }
            if (!isAbsolutePath) {
                dataFileName = sgl::FileUtils::get()->getPathToFile(_filePath) + dataFileName;
            }
            openDataFile(dataFileName);
        } else if (key == "options") {
            std::string optionName = splitLineString.at(1);
            boost::to_lower(optionName);
            if (optionName == "big_endian") {
                info.isBigEndian = true;
            } else if (optionName == "little_endian") {
                info.isBigEndian = false;
            } else if (optionName == "sequential") {
                info.isSequential = true;
                sgl::Logfile::get()->throwError(
                        "Error in CtlLoader::load: Error in file \"" + _filePath
                        + "\": Sequential format is not supported.");
                return false;
            }
        } else if (key == "undef") {
            info.fillValue = sgl::fromString<float>(splitLineString.at(1));
        } else if (key == "xdef") {
            info.xs = sgl::fromString<int>(splitLineString.at(1));
            if (!parseDef(fileBuffer, charPtr, lengthCtl, lineBuffer, splitLineString)) {
                return false;
            }
        } else if (key == "ydef") {
            info.ys = sgl::fromString<int>(splitLineString.at(1));
            if (!parseDef(fileBuffer, charPtr, lengthCtl, lineBuffer, splitLineString)) {
                return false;
            }
        } else if (key == "zdef") {
            info.zs = sgl::fromString<int>(splitLineString.at(1));
            if (!parseDef(fileBuffer, charPtr, lengthCtl, lineBuffer, splitLineString)) {
                return false;
            }
        } else if (key == "tdef") {
            info.ts = sgl::fromString<int>(splitLineString.at(1));
            if (!parseDef(fileBuffer, charPtr, lengthCtl, lineBuffer, splitLineString)) {
                return false;
            }
        } else if (key == "edef") {
            info.es = sgl::fromString<int>(splitLineString.at(1));
            if (!parseDef(fileBuffer, charPtr, lengthCtl, lineBuffer, splitLineString)) {
                return false;
            }
            //sgl::Logfile::get()->throwError(
            //        "Error in CtlLoader::load: Error in file \"" + _filePath
            //        + "\": Ensemble data support is not yet implemented.");
            //return false;
        } else if (key == "vars") {
            isVarsMode = true;
            numVars = sgl::fromString<int>(splitLineString.at(1));
        } else {
            // Unknown command; ignore.
        }
    }

    ptrdiff_t offset = 0;
    for (CtlVarDesc& varDesc : variableDescriptors) {
        varDesc.offset = offset;
        varDesc.size3d = varDesc.numLevels * info.xs * info.ys * ptrdiff_t(sizeof(float));
        offset += varDesc.size3d;
        info.sizeAllVars3d += varDesc.size3d;
    }

    if (info.ts > 1) {
        volumeData->setNumTimeSteps(info.ts);
    }
    if (info.es > 1) {
        volumeData->setEnsembleMemberCount(info.es);
    }

    bool isLatLonData = true;
    float dxCoords = 1.0f;
    float dyCoords = 1.0f;
    float dzCoords = 1.0f;
    if (!isLatLonData) {
        // Assume regular grid.
        dzCoords = info.zs > 1 ? (lev1d[info.zs - 1] - lev1d[0]) / float(info.zs - 1) : 1.0f;
        dyCoords = info.ys > 1 ? (lat1d[info.ys - 1] - lat1d[0]) / float(info.ys - 1) : 1.0f;
        dxCoords = info.xs > 1 ? (lon1d[info.xs - 1] - lon1d[0]) / float(info.xs - 1) : 1.0f;
    }
    float maxDeltaCoords = std::max(dxCoords, std::max(dyCoords, dzCoords));
    float maxDimension = float(std::max(info.xs - 1, std::max(info.ys - 1, info.zs - 1)));
    float cellStep = 1.0f / maxDimension;
    float dx = cellStep * dataSetInformation.scale[0] * dxCoords / maxDeltaCoords;
    float dy = cellStep * dataSetInformation.scale[1] * dyCoords / maxDeltaCoords;
    float dz = cellStep * dataSetInformation.scale[2] * dzCoords / maxDeltaCoords;
    volumeData->setGridExtent(int(info.xs), int(info.ys), int(info.zs), dx, dy, dz);
    volumeData->setFieldNames(fieldNameMap);

    // Copy longitude and latitude to 2D array.
    if (!lon1d || !lat1d) {
        sgl::Logfile::get()->throwError("Error in CtlLoader::setInputFiles: Lat or lon not set.");
    }
    auto* lonData = new float[info.xs * info.ys];
    auto* latData = new float[info.xs * info.ys];
    for (ptrdiff_t y = 0; y < info.ys; y++) {
        for (ptrdiff_t x = 0; x < info.xs; x++) {
            lonData[x + y * info.xs] = lon1d[x];
        }
    }
    for (ptrdiff_t y = 0; y < info.ys; y++) {
        for (ptrdiff_t x = 0; x < info.xs; x++) {
            latData[x + y * info.xs] = lat1d[y];
        }
    }
    delete[] lon1d;
    lon1d = nullptr;
    delete[] lat1d;
    lat1d = nullptr;
    delete[] lev1d;
    lev1d = nullptr;
    volumeData->setLatLonData(latData, lonData);

    return true;
}

bool CtlLoader::parseDef(
        char*& fileBuffer, size_t& charPtr, size_t& lengthCtl,
        std::string& lineBuffer, std::vector<std::string>& splitLineString) {
    if (splitLineString.size() < 3) {
        sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Invalid number of entries.");
        return false;
    }

    auto defType = splitLineString.at(0);
    boost::to_lower(defType);
    auto dimLen = ptrdiff_t(sgl::fromString<int>(splitLineString.at(1)));
    std::string dimType = splitLineString.at(2);
    boost::to_lower(dimType);

    std::vector<float> levelsArray;
    if (dimType == "linear") {
        if (splitLineString.size() != 5) {
            sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Invalid number of entries for type linear.");
            return false;
        }
        auto start = sgl::fromString<float>(splitLineString.at(3));
        auto step = sgl::fromString<float>(splitLineString.at(4));
        for (ptrdiff_t i = 0; i < dimLen; i++) {
            levelsArray.push_back(start + step * float(i));
        }
    } else if (dimType == "levels") {
        for (size_t i = 3; i < splitLineString.size(); i++) {
            levelsArray.push_back(sgl::fromString<float>(splitLineString.at(i)));
        }
        for (; charPtr < lengthCtl; ) {
            if (levelsArray.size() == size_t(dimLen)) {
                break;
            }
            if (levelsArray.size() > size_t(dimLen)) {
                sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Size mismatch.");
                return false;
            }

            lineBuffer.clear();
            while (charPtr < lengthCtl) {
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

            // Comments start with '*'.
            if (lineBuffer.front() == '*') {
                continue;
            }

            sgl::splitStringWhitespaceTyped<float>(lineBuffer, levelsArray);
        }
        if (levelsArray.size() != size_t(dimLen)) {
            sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Too few entries.");
            return false;
        }
    } else {
        sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Unknown type.");
        return false;
    }

    if (defType == "xdef") {
        if (lon1d) {
            sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Longitude array already allocated.");
            return false;
        }
        lon1d = new float[dimLen];
        for (ptrdiff_t i = 0; i < dimLen; i++) {
            lon1d[i] = levelsArray.at(i);
        }
    } else if (defType == "ydef") {
        if (lat1d) {
            sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Latitude array already allocated.");
            return false;
        }
        lat1d = new float[dimLen];
        for (ptrdiff_t i = 0; i < dimLen; i++) {
            lat1d[i] = levelsArray.at(i);
        }
    } else if (defType == "zdef") {
        if (lev1d) {
            sgl::Logfile::get()->throwError("Error in CtlLoader::parseDef: Level array already allocated.");
            return false;
        }
        lev1d = new float[dimLen];
        for (ptrdiff_t i = 0; i < dimLen; i++) {
            lev1d[i] = levelsArray.at(i);
        }
    }

    return true;
}

bool CtlLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    auto it = variableNameMap.find(fieldName);
    if (it == variableNameMap.end()) {
        sgl::Logfile::get()->throwError(
                "Error in CtlLoader::getFieldEntry: Unknown field name \"" + fieldName + "\".");
        return false;
    }

    auto& varDesc = variableDescriptors.at(it->second);
    ptrdiff_t readOffset = (memberIdx * info.ts + timestepIdx) * info.sizeAllVars3d + varDesc.offset;
    ptrdiff_t numEntries = varDesc.size3d / ptrdiff_t(sizeof(float));
    auto* data = new float[numEntries];
    loadDataFromFile(reinterpret_cast<uint8_t*>(data), readOffset, varDesc.size3d);

    if (info.isBigEndian) {
        swapEndianness(data, int(numEntries));
    }

    for (ptrdiff_t i = 0; i < numEntries; i++) {
        if (data[i] == info.fillValue) {
            data[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }

    if (varDesc.numLevels != info.zs) {
        // Correrender currently doesn't have support for different z resolutions -> try to convert to info.zs.
        if (varDesc.numLevels != 0 && varDesc.numLevels != 1) {
            sgl::Logfile::get()->throwError(
                    "Error in CtlLoader::load: Invalid number of levels for variable \"" + fieldName + "\".");
            return false;
        }
        auto* data2d = data;
        data = new float[numEntries * varDesc.numLevels];
        for (ptrdiff_t z = 0; z < varDesc.numLevels; z++) {
            memcpy(data + z * numEntries, data2d, varDesc.size3d);
        }
        delete[] data2d;
    }

    fieldEntry = new HostCacheEntryType(info.xs * info.ys * info.zs, data);
    if (dataSetInformation.useFormatCast) {
        fieldEntry->switchNativeFormat(dataSetInformation.formatTarget);
    }

    return true;
}

bool CtlLoader::openDataFile(const std::string& dataFileName) {
#if defined(__linux__) || defined(__MINGW32__) // __GNUC__? Does GCC generally work on non-POSIX systems?
    file = fopen64(dataFileName.c_str(), "rb");
#else
    file = fopen(dataFileName.c_str(), "rb");
#endif
    if (!file) {
        sgl::Logfile::get()->writeError(
                std::string() + "Error in CtlLoader::openDataFile: File \"" + dataFileName + "\" could not be opened.");
        return false;
    }

    return true;
}

void CtlLoader::closeDataFile() {
    fclose(file);
    file = nullptr;
}

void CtlLoader::loadDataFromFile(uint8_t* destBuffer, ptrdiff_t offset, ptrdiff_t size) {
#if defined(_WIN32) && !defined(__MINGW32__)
    int ret = _fseeki64(file, offset, SEEK_SET);
#else
    int ret = fseeko(file, offset, SEEK_SET);
#endif
    if (ret != 0) {
        sgl::Logfile::get()->throwError(
                std::string() + "Error in CtlLoader::loadDataFromFile: fseek return error code.");
    }
    size_t numBytesRead = fread(destBuffer, 1, size, file);
    if (numBytesRead != size_t(size)) {
        sgl::Logfile::get()->throwError(
                std::string() + "Error in CtlLoader::loadDataFromFile: Read number of bytes does not match.");
    }
}
