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

#include <iostream>
#include <set>
#include <stdexcept>
#include <cassert>
#include <cstring>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "Hdf5Loader.hpp"

#if defined(DEBUG) || !defined(NDEBUG)
#define myassert assert
#else
#define myassert(x)                                   \
	if (!(x))                                         \
	{                                                 \
		throw std::runtime_error("assertion failed"); \
	}
#endif

Hdf5Loader::Hdf5Loader() = default;

Hdf5Loader::~Hdf5Loader() {
    if (isOpen) {
        if (memSpaceId != H5I_INVALID_HID && H5Sclose(memSpaceId) < 0) {
            sgl::Logfile::get()->throwError(
                    "Error in Hdf5Loader::~Hdf5Loader: H5Sclose(memSpaceId) failed for file \"" + filePath + "\".");
        }
        if (dataSpaceId != H5I_INVALID_HID && H5Sclose(dataSpaceId) < 0) {
            sgl::Logfile::get()->throwError(
                    "Error in Hdf5Loader::~Hdf5Loader: H5Sclose(dataSpaceId) failed for file \"" + filePath + "\".");
        }
        if (volumeDataId != H5I_INVALID_HID && H5Dclose(volumeDataId) < 0) {
            sgl::Logfile::get()->throwError(
                    "Error in Hdf5Loader::~Hdf5Loader: H5Dclose failed for file \"" + filePath + "\".");
        }
        if (fileId != H5I_INVALID_HID && H5Fclose(fileId) < 0) {
            sgl::Logfile::get()->throwError(
                    "Error in Hdf5Loader::~Hdf5Loader: H5Fclose failed for file \"" + filePath + "\".");
        }
        if (fileAccessPropertyList != H5I_INVALID_HID && H5Pclose(fileAccessPropertyList) < 0) {
            sgl::Logfile::get()->throwError(
                    "Error in Hdf5Loader::~Hdf5Loader: H5Pclose failed for file \"" + filePath + "\".");
        }
    }
}

bool Hdf5Loader::setInputFiles(
        VolumeData* volumeData, const std::string& _filePath, const DataSetInformation& _dataSetInformation) {
    filePath = _filePath;
    dataSetInformation = _dataSetInformation;

    fileAccessPropertyList = H5Pcreate(H5P_FILE_ACCESS);
    if (fileAccessPropertyList == H5I_INVALID_HID) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::load: H5Pcreate failed for file \"" + filePath + "\".");
        return false;
    }
    H5Pset_fclose_degree(fileAccessPropertyList, H5F_CLOSE_STRONG);
    fileId = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, fileAccessPropertyList);

    if (fileId == H5I_INVALID_HID) {
        if (H5Pclose(fileAccessPropertyList) < 0) {
            sgl::Logfile::get()->throwError(
                    "Error in Hdf5Loader::setInputFiles: H5Pclose failed for file \"" + filePath + "\".");
        }
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: File \"" + filePath + "\" couldn't be opened.");
        return false;
    }
    isOpen = true;

    volumeDataId = H5Dopen(fileId, "/volumes", H5P_DEFAULT);
    if (volumeDataId == H5I_INVALID_HID) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: File \"" + filePath + "\" does not contain '/volumes' data.");
        return false;
    }

    dataSpaceId = H5Dget_space(volumeDataId);
    rank = H5Sget_simple_extent_ndims(dataSpaceId);
    dims = std::vector<hsize_t>(rank);
    H5Sget_simple_extent_dims(dataSpaceId, dims.data(), nullptr);

    std::vector<hsize_t> count = dims;
    count[0] = 1;
    memSpaceId = H5Screate_simple(rank, count.data(), nullptr);
    if (memSpaceId == H5I_INVALID_HID) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: H5Screate_simple failed for file \"" + filePath + "\".");
        return false;
    }

    // TODO: Support other data than time-dependent color data.
    if (rank != 5) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: File \"" + filePath + "\" has unsupported data rank.");
        return false;
    }
    int cs = int(dims[1]);
    if (cs != 4) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: File \"" + filePath + "\" does not contain RGBA colors.");
        return false;
    }
    isColorData = true;
    ts = int(dims[0]);
    zs = int(dims[2]);
    ys = int(dims[3]);
    xs = int(dims[4]);

    if (ts > 1) {
        volumeData->setNumTimeSteps(ts);
    }

    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    float dx = cellStep * dataSetInformation.scale[0];
    float dy = cellStep * dataSetInformation.scale[1];
    float dz = cellStep * dataSetInformation.scale[2];
    volumeData->setGridExtent(int(xs), int(ys), int(zs), dx, dy, dz);

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    fieldNameMap[FieldType::COLOR].emplace_back("Color");
    volumeData->setFieldNames(fieldNameMap);

    return true;
}

bool Hdf5Loader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    size_t numEntries = size_t(xs) * size_t(ys) * size_t(zs);
    if (isColorData) {
        numEntries *= 4;
    }

    std::vector<hsize_t> start(rank, 0);
    std::vector<hsize_t> count = dims;
    start[0] = hsize_t(timestepIdx);
    count[0] = 1;
    auto status = H5Sselect_hyperslab(
            dataSpaceId, H5S_SELECT_SET, start.data(), nullptr, count.data(), nullptr);
    if (status < 0) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: H5Sselect_hyperslab failed for file \"" + filePath + "\".");
    }

    auto* fieldEntryBuffer = new float[numEntries];
    hid_t memTypeId = H5Tcopy(H5T_NATIVE_FLOAT); // H5T_IEEE_F32LE
    if (H5Dread(volumeDataId, memTypeId, memSpaceId, dataSpaceId, H5P_DEFAULT, fieldEntryBuffer) < 0) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: H5Dread failed for file \"" + filePath + "\".");
    }
    if (H5Tclose(memTypeId) < 0) {
        sgl::Logfile::get()->writeError(
                "Error in Hdf5Loader::setInputFiles: H5Tclose failed for file \"" + filePath + "\".");
    }

    float* arrayChannelLast = fieldEntryBuffer;
    fieldEntryBuffer = new float[numEntries];
    const size_t clen = 4;
    const auto zlen = size_t(xs);
    const auto ylen = size_t(ys);
    const auto xlen = size_t(zs);
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, clen), [&](auto const& r) {
            for (auto c = r.begin(); c != r.end(); c++) {
#else
#if _OPENMP >= 201107
#pragma omp parallel for shared(clen, zlen, ylen, xlen, fieldEntryBuffer, arrayChannelLast) default(none)
#endif
    for (size_t c = 0; c < clen; c++) {
#endif
        for (size_t z = 0; z < zlen; z++) {
            for (size_t y = 0; y < ylen; y++) {
                for (size_t x = 0; x < xlen; x++) {
                    size_t idxRead = ((c * zlen + z) * ylen + y) * xlen + x; // CZYX
                    size_t idxWrite = ((z * ylen + y) * xlen + x) * clen + c; // ZYXC
                    fieldEntryBuffer[idxWrite] = arrayChannelLast[idxRead];
                }
            }
        }
    }
#ifdef USE_TBB
    });
#endif
    delete[] arrayChannelLast;

    fieldEntry = new HostCacheEntryType(numEntries, fieldEntryBuffer);
    if (dataSetInformation.useFormatCast) {
        fieldEntry->switchNativeFormat(dataSetInformation.formatTarget);
    }

    return true;
}
