/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022-2024, Christoph Neuhauser
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

#ifndef CORRERENDER_HDF5LOADER_HPP
#define CORRERENDER_HDF5LOADER_HPP

#include <unordered_map>
#include "VolumeLoader.hpp"

#include <hdf5.h>

/**
 * For more details on the HDF5 file format see: https://portal.hdfgroup.org/documentation/
 */
class Hdf5Loader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "hdf5" }; }
    Hdf5Loader();
    ~Hdf5Loader() override;
    bool setInputFiles(
            VolumeData *volumeData, const std::string &filePath, const DataSetInformation &dataSetInformation) override;
    bool getFieldEntry(
            VolumeData *volumeData, FieldType fieldType, const std::string &fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType *&fieldEntry) override;
    bool getHasFloat32Data() override { return true; }

private:
    bool isOpen = false;
    hid_t fileAccessPropertyList = -1, fileId = -1, volumeDataId = -1, dataSpaceId = -1, memSpaceId = -1;
    int rank = 0;
    std::vector<hsize_t> dims;
    std::string filePath;
    DataSetInformation dataSetInformation;
    std::unordered_map<std::string, int> datasetNameMap;
    int xs = 0, ys = 0, zs = 0, ts = 0, es = 0;
    bool isColorData = false; // For storing pre-shaded volume data.
};

#endif //CORRERENDER_HDF5LOADER_HPP
