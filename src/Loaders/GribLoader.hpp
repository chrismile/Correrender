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

#ifndef CORRERENDER_GRIBLOADER_HPP
#define CORRERENDER_GRIBLOADER_HPP

#include <Utils/HashCombine.hpp>

#include "VolumeLoader.hpp"

struct grib_handle;
typedef struct grib_handle grib_handle;
typedef struct grib_handle codes_handle;

struct GribTimeStep {
    long dataDateLoad = 0;
    long dataTimeLoad = 0;
    int variableIndex = 0;

    bool operator==(const GribTimeStep& other) const {
        return dataDateLoad == other.dataDateLoad && dataTimeLoad == other.dataTimeLoad
               && other.variableIndex == variableIndex;
    }
};

namespace std {
template<> struct hash<GribTimeStep> {
    std::size_t operator()(GribTimeStep const& s) const noexcept {
        std::size_t result = 0;
        hash_combine(result, s.dataDateLoad);
        hash_combine(result, s.dataTimeLoad);
        hash_combine(result, s.variableIndex);
        return result;
    }
};
}

struct GribVarInfo {
#if defined(_WIN32) && !defined(__MINGW32__)
    std::vector<__int64> handleOffsets;
#elif defined(__MINGW32__)
    std::vector<_off64_t> handleOffsets;
#else
    std::vector<__off64_t> handleOffsets;
#endif
    //std::vector<codes_handle*> handles;
    std::vector<long> levelValues;
};

/**
 * A loader for GRIB volume data sets.
 */
class GribLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "grb", "grib" }; }
    ~GribLoader() override;
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;

private:
    std::string getString(codes_handle* handle, const std::string& key);
    long getLong(codes_handle* handle, const std::string& key);
    double getDouble(codes_handle* handle, const std::string& key);

    std::string dataSourceFilename;
    DataSetInformation dataSetInformation;
    int xs = 0, ys = 0, zs = 0;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    int numPoints = 0;
    long numberOfPointsGlobal = 0;
    long numLonsGlobal = 0;
    long numLatsGlobal = 0;
    size_t numLevelsGlobal = 0;
    FILE* file = nullptr;
    std::vector<std::pair<long, long>> timeSteps;
    std::unordered_map<GribTimeStep, GribVarInfo> gribVarInfos;
    std::unordered_map<std::string, int> encounteredVariableNamesMap;
    std::map<long, long> levelToIndexMap;
};

#endif //CORRERENDER_GRIBLOADER_HPP
