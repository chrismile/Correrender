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

#define _FILE_OFFSET_BITS 64
#define __USE_FILE_OFFSET64

#include <iostream>
#include <set>
#include <unordered_set>
#include <unordered_map>

#include <eccodes.h>

#include <Utils/File/Logfile.hpp>

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "LoadersUtil.hpp"
#include "GribLoader.hpp"

std::string GribLoader::getString(codes_handle* handle, const std::string& key) {
    size_t length = 0;
    codes_get_length(handle, key.c_str(), &length);
    char* value = new char[length + 1];
    int errorCode = codes_get_string(handle, key.c_str(), value, &length);
    if (errorCode != 0) {
        sgl::Logfile::get()->throwError(
                "Error in GribLoader::getString: Cannot access value for key \"" + key
                + "\". ecCodes error message: " + codes_get_error_message(errorCode));
    }
    value[length] = '\0';
    std::string valueString = value;
    delete[] value;
    return valueString;
}

double GribLoader::getDouble(codes_handle* handle, const std::string& key) {
    double value = 0.0;
    int errorCode = codes_get_double(handle, key.c_str(), &value);
    if (errorCode != 0) {
        sgl::Logfile::get()->throwError(
                "Error in GribLoader::getDouble: Cannot access value for key \"" + key
                + "\". ecCodes error message: " + codes_get_error_message(errorCode));
    }
    return value;
}

long GribLoader::getLong(codes_handle* handle, const std::string& key) {
    long value = 0;
    int errorCode = codes_get_long(handle, key.c_str(), &value);
    if (errorCode != 0) {
        sgl::Logfile::get()->throwError(
                "Error in GribLoader::getDouble: Cannot access value for key \"" + key
                + "\". ecCodes error message: " + codes_get_error_message(errorCode));
    }
    return value;
}

GribLoader::~GribLoader() {
    if (file) {
        fclose(file);
        file = nullptr;
    }
}

bool GribLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

#if defined(__linux__) || defined(__MINGW32__) // __GNUC__? Does GCC generally work on non-POSIX systems?
    file = fopen64(dataSourceFilename.c_str(), "rb");
#else
    file = fopen(dataSourceFilename.c_str(), "rb");
#endif

    if (!file) {
        sgl::Logfile::get()->writeError(
                std::string() + "Error in GribLoader::setInputFiles: File \""
                + dataSourceFilename + "\" could not be opened.");
        return false;
    }

    int errorCode = 0;

#if defined(_WIN32) && !defined(__MINGW32__)
    _fseeki64(file, 0, SEEK_END);
    auto fileSize = _ftelli64(file);
    _fseeki64(file, 0, SEEK_SET);
#else
    fseeko(file, 0, SEEK_END);
    auto fileSize = ftello(file);
    fseeko(file, 0, SEEK_SET);
#endif

    std::vector<std::string> variableNames;
    std::unordered_set<long> encounteredLevelValuesSet;

    double* lonValues = nullptr;
    double* latValues = nullptr;

    numberOfPointsGlobal = 0;
    numLonsGlobal = 0;
    numLatsGlobal = 0;

    long currDataDate = -1, currDataTime = -1;
    while(true) {
#if defined(_WIN32) && !defined(__MINGW32__)
        auto bufferSize = _ftelli64(file); // __int64
#else
        auto filePosition = ftello(file); // __off64_t
#endif

        if (filePosition == fileSize) {
            break;
        }

        codes_handle* handle = codes_handle_new_from_file(nullptr, file, PRODUCT_GRIB, &errorCode);
        if (!handle) {
            if (errorCode == 0) {
                break;
            }
            sgl::Logfile::get()->throwError(
                    "Error in GribLoader::setInputFiles: File \"" + dataSourceFilename + "\" couldn't be opened.");
        }

        long dataDate = getLong(handle, "dataDate");
        long dataTime = getLong(handle, "dataTime");
        if (dataSetInformation.hasCustomDateTime
                && (dataDate != dataSetInformation.date || dataTime != dataSetInformation.time)) {
            CODES_CHECK(codes_handle_delete(handle), nullptr);
            if (dataDate < dataSetInformation.date
                    || (dataDate == dataSetInformation.date && dataTime < dataSetInformation.time)) {
                continue;
            } else {
                break;
            }
        }

        std::string parameterShortName = getString(handle, "shortName");
        if (currDataDate != dataDate || currDataTime != dataTime) {
            timeSteps.emplace_back(dataDate, dataTime);
            currDataDate = dataDate;
            currDataTime = dataTime;
        }

        long numberOfPoints = getLong(handle, "numberOfPoints");
        long numLons = getLong(handle, "Ni");
        long numLats = getLong(handle, "Nj");

        if (numLons * numLats != numberOfPoints) {
            sgl::Logfile::get()->throwError(
                    "Error in GribLoader::setInputFiles: File \"" + dataSourceFilename + "\" has contradictory values for "
                    "numberOfPoints, Ni and Nj.");
        }

        long level = getLong(handle, "level");
        if (encounteredLevelValuesSet.find(level) == encounteredLevelValuesSet.end()) {
            encounteredLevelValuesSet.insert(level);
        }

        double lonMin = getDouble(handle, "longitudeOfFirstGridPointInDegrees");
        double lonMax = getDouble(handle, "longitudeOfLastGridPointInDegrees");
        double latMin = getDouble(handle, "latitudeOfFirstGridPointInDegrees");
        double latMax = getDouble(handle, "latitudeOfLastGridPointInDegrees");
        double lonInc = getDouble(handle, "iDirectionIncrementInDegrees");
        double latInc = getDouble(handle, "jDirectionIncrementInDegrees");
        (void)lonMax;
        (void)latMax;

        std::string typeOfLevel = getString(handle, "typeOfLevel");
        if (typeOfLevel != "isobaricInhPa") {
            sgl::Logfile::get()->throwError(
                    "Error in GribLoader::setInputFiles: The loader currently only supports the level type isobaricInhPa. "
                    "However, a different level type is used in the file \"" + dataSourceFilename + "\".");
        }

        std::string gridType = getString(handle, "gridType");
        if (gridType != "regular_ll") {
            sgl::Logfile::get()->throwError(
                    "Error in GribLoader::setInputFiles: The loader currently only supports the grid type regular_ll. "
                    "However, a different grid type is used in the file \"" + dataSourceFilename + "\".");
        }

        /*std::cout << "N: " << numberOfPoints << std::endl;
        std::cout << "Ni: " << numLons << std::endl;
        std::cout << "Nj: " << numLats << std::endl;
        std::cout << "lonMin: " << lonMin << std::endl;
        std::cout << "lonMax: " << lonMax << std::endl;
        std::cout << "latMin: " << latMin << std::endl;
        std::cout << "latMax: " << latMax << std::endl;
        std::cout << "lonInc: " << lonInc << std::endl;
        std::cout << "latInc: " << latInc << std::endl;
        std::cout << "shortName: " << parameterShortName << std::endl;
        std::cout << "level: " << level << std::endl;
        std::cout << "typeOfLevel: " << typeOfLevel << std::endl;
        std::cout << "dataDate: " << dataDate << std::endl;
        std::cout << "dataTime: " << dataTime << std::endl;
        std::cout << std::endl;*/

        // First variable data read?
        if (variableNames.empty()) {
            numberOfPointsGlobal = numberOfPoints;
            numLonsGlobal = numLons;
            numLatsGlobal = numLats;
            lonValues = new double[numLons];
            latValues = new double[numLats];
            for (int i = 0; i < numLons; i++) {
                lonValues[i] = lonMin + i * lonInc;
            }
            for (int j = 0; j < numLats; j++) {
                latValues[j] = latMin + j * latInc;
            }
        } else {
            if (numberOfPoints != numberOfPointsGlobal || numLons != numLonsGlobal || numLats != numLatsGlobal) {
                sgl::Logfile::get()->throwError(
                        "Error in GribLoader::setInputFiles: File \"" + dataSourceFilename + "\" has inconsistent values "
                        "for numberOfPoints, Ni or Nj.");
            }
        }

        if (encounteredVariableNamesMap.find(parameterShortName) == encounteredVariableNamesMap.end()) {
            // Encountered the variable for the first time.
            encounteredVariableNamesMap.insert(std::make_pair(parameterShortName, int(variableNames.size())));
            variableNames.push_back(parameterShortName);
        } else {
            // Encountered the variable already previously.
        }
        int variableIdx = encounteredVariableNamesMap.find(parameterShortName)->second;

        size_t valuesLength = 0;
        CODES_CHECK(codes_get_size(handle, "values", &valuesLength), nullptr);
        if (valuesLength != size_t(numberOfPoints)) {
            sgl::Logfile::get()->throwError(
                    "Error in GribLoader::setInputFiles: The values array size and numberOfPoints do not match in file \""
                    + dataSourceFilename + "\".");
        }

        GribTimeStep currTimeStep{};
        currTimeStep.dataDateLoad = dataDate;
        currTimeStep.dataTimeLoad = dataTime;
        currTimeStep.variableIndex = variableIdx;
        GribVarInfo& varInfo = gribVarInfos[currTimeStep];
        varInfo.handleOffsets.push_back(filePosition);
        varInfo.levelValues.push_back(level);

        CODES_CHECK(codes_handle_delete(handle), nullptr);
    }

    numLevelsGlobal = encounteredLevelValuesSet.size();
    long idx = 0;
    for (auto& level : encounteredLevelValuesSet) {
        levelToIndexMap.insert(std::make_pair(level, idx));
        idx++;
    }

    xs = int(numLonsGlobal);
    ys = int(numLatsGlobal);
    zs = int(numLevelsGlobal);
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    // TODO: Use lon, lat, level/pressure (pv).
    dx = cellStep * dataSetInformation.scale[0];
    dy = cellStep * dataSetInformation.scale[1];
    dz = cellStep * dataSetInformation.scale[2];
    numPoints = int(xs * ys * zs);
    volumeData->setGridExtent(int(xs), int(ys), int(zs), dx, dy, dz);

    //long dataDateLoad = 20161002; // 2016-10-02
    //long dataTimeLoad = 600; // 6:00 o'clock
    //long dataDateLoad = dataSetInformation.date;
    //long dataTimeLoad = dataSetInformation.time;
    std::vector<std::string> timeStepNames;
    timeStepNames.reserve(timeSteps.size());
    for (const auto& timeStep : timeSteps) {
        std::string dateString = std::to_string(timeStep.first);
        std::string timeString = std::to_string(timeStep.second);
        timeStepNames.push_back(dateString + " " + timeString);
    }
    volumeData->setTimeSteps(timeStepNames);

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    for (size_t varIdx = 0; varIdx < variableNames.size(); varIdx++) {
        fieldNameMap[FieldType::SCALAR].push_back(variableNames.at(varIdx));
    }
    volumeData->setFieldNames(fieldNameMap);

    if (lonValues) {
        delete[] lonValues;
    }
    if (latValues) {
        delete[] latValues;
    }

    return true;
}

bool GribLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    //long dataDateLoad = 20161002; // 2016-10-02
    //long dataTimeLoad = 600; // 6:00 o'clock
    //long dataDateLoad = dataSetInformation.date;
    //long dataTimeLoad = dataSetInformation.time;

    GribTimeStep gribTimeStep;
    if (dataSetInformation.hasCustomDateTime) {
        gribTimeStep.dataDateLoad = dataSetInformation.date;
        gribTimeStep.dataTimeLoad = dataSetInformation.time;
    } else {
        auto& dateTime = timeSteps.at(timestepIdx);
        gribTimeStep.dataDateLoad = dateTime.first;
        gribTimeStep.dataTimeLoad = dateTime.second;
    }
    auto varIt = encounteredVariableNamesMap.find(fieldName);
    if (varIt == encounteredVariableNamesMap.end()) {
        return false;
    }
    gribTimeStep.variableIndex = varIt->second;

    auto varInfoIt = gribVarInfos.find(gribTimeStep);
    if (varInfoIt == gribVarInfos.end()) {
        return false;
    }
    const auto& varInfo = varInfoIt->second;

    std::vector<float*> variableSliceArray;
    variableSliceArray.resize(numLevelsGlobal);
    int errorCode = 0;
    for (size_t i = 0; i < varInfo.handleOffsets.size(); i++) {
        auto handleOffset = varInfo.handleOffsets.at(i);
        auto level = varInfo.levelValues.at(i);
#if defined(_WIN32) && !defined(__MINGW32__)
        _fseeki64(file, handleOffset, SEEK_SET);
#else
        fseeko(file, handleOffset, SEEK_SET);
#endif

        codes_handle* handle = codes_handle_new_from_file(nullptr, file, PRODUCT_GRIB, &errorCode);
        if (!handle) {
            sgl::Logfile::get()->throwError(
                    "Error in GribLoader::getFieldEntry: File \"" + dataSourceFilename + "\" couldn't be opened.");
        }

        size_t valuesLength = numberOfPointsGlobal;
        auto* valuesArrayDouble = new double[valuesLength];
        auto* valuesArrayFloat = new float[valuesLength];
        CODES_CHECK(codes_get_double_array(handle, "values", valuesArrayDouble, &valuesLength), nullptr);
        for (size_t i = 0; i < valuesLength; i++) {
            valuesArrayFloat[i] = static_cast<float>(valuesArrayDouble[i]);
        }
        auto levelIndex = levelToIndexMap.find(level)->second;
        variableSliceArray.at(levelIndex) = valuesArrayFloat;
        delete[] valuesArrayDouble;

        CODES_CHECK(codes_handle_delete(handle), nullptr);
    }

    // Merge the variable slice arrays.
    size_t numEntries = numLevelsGlobal * size_t(numLatsGlobal) * size_t(numLonsGlobal);
    float* fieldEntryBuffer = new float[numEntries];
    for (size_t level = 0; level < variableSliceArray.size(); level++) {
        const float* variableSlice = variableSliceArray.at(level);
        if (variableSlice == nullptr) {
            for (long latIdx = 0; latIdx < numLatsGlobal; latIdx++) {
                for (long lonIdx = 0; lonIdx < numLonsGlobal; lonIdx++) {
                    fieldEntryBuffer[
                            (level * size_t(numLonsGlobal * numLatsGlobal) + size_t(numLonsGlobal) * latIdx + lonIdx)] =
                                    0.0f;
                }
            }
        } else {
            for (long latIdx = 0; latIdx < numLatsGlobal; latIdx++) {
                for (long lonIdx = 0; lonIdx < numLonsGlobal; lonIdx++) {
                    fieldEntryBuffer[
                            (level * size_t(numLonsGlobal * numLatsGlobal) + size_t(numLonsGlobal) * latIdx + lonIdx)] =
                            variableSlice[size_t(numLonsGlobal) * latIdx + lonIdx];
                }
            }
            delete[] variableSlice;
        }
    }
    fieldEntry = new HostCacheEntryType(numEntries, fieldEntryBuffer);

    return true;
}
