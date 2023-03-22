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

#ifndef CORRERENDER_ZARRLOADER_HPP
#define CORRERENDER_ZARRLOADER_HPP

#include <unordered_map>
#include "VolumeLoader.hpp"

namespace z5 {
class Dataset;
}

/**
 * For more details on the zarr file format see: https://zarr.readthedocs.io/en/stable/
 * Used underlying loader library: https://github.com/constantinpape/z5
 */
class ZarrLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "zarr" }; }
    ZarrLoader();
    ~ZarrLoader() override;
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;

private:
    bool getDatasetExists(const std::string& fieldName);
    size_t getDatasetIndex(const std::string& fieldName);
    std::vector<size_t> getShape(size_t datasetIdx);
    void loadFloatArray1D(size_t datasetIdx, size_t len, float*& array);
    void loadDoubleArray1D(size_t datasetIdx, size_t len, double*& array);
    void loadFloatArray3D(size_t datasetIdx, size_t zlen, size_t ylen, size_t xlen, float*& array);
    void loadFloatArray3D(
            size_t datasetIdx, size_t time, size_t zlen, size_t ylen, size_t xlen, float*& array);

    std::vector<std::unique_ptr<z5::Dataset>> datasets;
    std::unordered_map<std::string, size_t> datasetNameMap;
    int xs = 0, ys = 0, zs = 0, ts = 0;
};

#endif //CORRERENDER_ZARRLOADER_HPP
