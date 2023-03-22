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

#include "z5/factory.hxx"
#include "z5/filesystem/handle.hxx"
#include "z5/multiarray/xtensor_access.hxx"
#include "z5/attributes.hxx"

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "ZarrLoader.hpp"

ZarrLoader::ZarrLoader() = default;

ZarrLoader::~ZarrLoader() = default;

bool ZarrLoader::getDatasetExists(const std::string& fieldName) {
    auto it = datasetNameMap.find(fieldName);
    if (it == datasetNameMap.end()) {
        sgl::Logfile::get()->throwError(
                "Error in ZarrLoader::getFieldEntry: Unknown field name \"" + fieldName + "\".");
        return false;
    }
    return true;
}

size_t ZarrLoader::getDatasetIndex(const std::string& fieldName) {
    auto it = datasetNameMap.find(fieldName);
    if (it == datasetNameMap.end()) {
        sgl::Logfile::get()->throwError(
                "Error in ZarrLoader::getFieldEntry: Unknown field name \"" + fieldName + "\".");
        return 0;
    }
    return it->second;
}

std::vector<size_t> ZarrLoader::getShape(size_t datasetIdx) {
    auto& dataset = datasets.at(datasetIdx);
    auto shape = dataset->shape();
    return shape;
}

void ZarrLoader::loadFloatArray1D(size_t datasetIdx, size_t len, float*& array) {
    array = new float[len];
    xt::xarray<float>::shape_type startp = { 0 };
    xt::xarray<float>::shape_type countp = { len };

    auto& dataset = datasets.at(datasetIdx);
    if (dataset->getDtype() == z5::types::Datatype::float32) {
        xt::xarray<float> outputArray(countp);
        z5::multiarray::readSubarray<float>(dataset, outputArray, startp.begin());
        //memcpy(array, outputArray.data(), len);
        for (size_t i = 0; i < len; i++) {
            array[i] = outputArray.at(i);
        }
    } else if (dataset->getDtype() == z5::types::Datatype::float64) {
        xt::xarray<double> outputArray(countp);
        z5::multiarray::readSubarray<double>(dataset, outputArray, startp.begin());
        for (size_t i = 0; i < len; i++) {
            array[i] = float(outputArray.at(i));
        }
    } else {
        sgl::Logfile::get()->throwError("Error in ZarrLoader::loadFloatArray1D: Unsupported floating point format.");
    }
}

void ZarrLoader::loadDoubleArray1D(size_t datasetIdx, size_t len, double*& array) {
    array = new double [len];
    xt::xarray<double>::shape_type startp = { 0 };
    xt::xarray<double>::shape_type countp = { len };

    auto& dataset = datasets.at(datasetIdx);
    xt::xarray<double> outputArray(countp);
    z5::multiarray::readSubarray<double>(dataset, outputArray, startp.begin());
    //memcpy(array, outputArray.data(), len);
    for (size_t i = 0; i < len; i++) {
        array[i] = outputArray.at(i);
    }
}

void ZarrLoader::loadFloatArray3D(size_t datasetIdx, size_t zlen, size_t ylen, size_t xlen, float*& array) {
    array = new float[zlen * ylen * xlen];
    xt::xarray<float>::shape_type startp = { 0, 0, 0 };
    xt::xarray<float>::shape_type countp = { zlen, ylen, xlen };

    auto& dataset = datasets.at(datasetIdx);
    xt::xarray<float> outputArray(countp);
    z5::multiarray::readSubarray<float>(dataset, outputArray, startp.begin());
    //memcpy(array, outputArray.data(), xlen * ylen * zlen);

    float fillValue = 0.0f;
    dataset->getFillValue(&fillValue);

    for (size_t z = 0; z < zlen; z++) {
        for (size_t y = 0; y < ylen; y++) {
            for (size_t x = 0; x < xlen; x++) {
                float val = outputArray.at(z, y, x);
                if (val == fillValue) {
                    val = std::numeric_limits<float>::quiet_NaN();
                }
                array[IDXS(x, y, z)] = val;
            }
        }
    }
}

void ZarrLoader::loadFloatArray3D(
        size_t datasetIdx, size_t time, size_t zlen, size_t ylen, size_t xlen, float*& array) {
    array = new float[zlen * ylen * xlen];
    xt::xarray<float>::shape_type startp = { time, 0, 0, 0 };
    xt::xarray<float>::shape_type countp = { 1, zlen, ylen, xlen };

    auto& dataset = datasets.at(datasetIdx);
    xt::xarray<float> outputArray(countp);
    z5::multiarray::readSubarray<float>(dataset, outputArray, startp.begin());
    //memcpy(array, outputArray.data(), xlen * ylen * zlen);

    float fillValue = 0.0f;
    dataset->getFillValue(&fillValue);

    for (size_t z = 0; z < zlen; z++) {
        for (size_t y = 0; y < ylen; y++) {
            for (size_t x = 0; x < xlen; x++) {
                float val = outputArray.at(time, z, y, x);
                if (val == fillValue) {
                    val = std::numeric_limits<float>::quiet_NaN();
                }
                array[IDXS(x, y, z)] = val;
            }
        }
    }
}

bool ZarrLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) {
    z5::filesystem::handle::File f(filePath);
    if (!f.exists()) {
        sgl::Logfile::get()->writeError(
                "Error in ZarrLoader::setInputFiles: Directory \"" + filePath + "\" couldn't be opened.");
        return false;
    }

    std::vector<std::string> keys;
    f.keys(keys);
    std::sort(keys.begin(), keys.end());
    datasets.resize(keys.size());
    for (size_t datasetIdx = 0; datasetIdx < keys.size(); datasetIdx++) {
        datasets.at(datasetIdx) = z5::openDataset(f, keys.at(datasetIdx));
        if (!datasets.at(datasetIdx)) {
            sgl::Logfile::get()->writeError(
                    "Error in ZarrLoader::setInputFiles: Data set \"" + keys.at(datasetIdx) + "\" in directory \""
                    + filePath + "\" couldn't be opened.");
            return false;
        }
        datasetNameMap.insert(std::make_pair(keys.at(datasetIdx), datasetIdx));
    }

    std::vector<size_t> maxShape;
    for (size_t datasetIdx = 0; datasetIdx < keys.size(); datasetIdx++) {
        auto& dataset = datasets.at(datasetIdx);
        auto shape = dataset->shape();
        if (shape.size() > maxShape.size()) {
            maxShape = shape;
        }
    }

    if (maxShape.size() == 4) {
        ts = int(maxShape.at(0));
        zs = int(maxShape.at(1));
        ys = int(maxShape.at(2));
        xs = int(maxShape.at(3));
    }

    float* zCoords = nullptr;
    float* yCoords = nullptr;
    float* xCoords = nullptr;
    //bool isLatLonData = false;
    if (getDatasetExists("lon") && getDatasetExists("lat") && getDatasetExists("lev")) {
        //isLatLonData = true;
        loadFloatArray1D(getDatasetIndex("lev"), zs, zCoords);
        loadFloatArray1D(getDatasetIndex("lat"), ys, yCoords);
        loadFloatArray1D(getDatasetIndex("lon"), xs, xCoords);
    }

    // TODO: Use coords also for lat-lon-pressure?
    float dxCoords = 1.0f;
    float dyCoords = 1.0f;
    float dzCoords = 1.0f;
    if (false) { // !timestepIdx
        // Assume regular grid.
        dzCoords = (zCoords[zs - 1] - zCoords[0]) / float(zs - 1);
        dyCoords = (yCoords[ys - 1] - yCoords[0]) / float(ys - 1);
        dxCoords = (xCoords[xs - 1] - xCoords[0]) / float(xs - 1);
    }
    float maxDeltaCoords = std::max(dxCoords, std::max(dyCoords, dzCoords));

    // Get the dimensions of the underyling grid.
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    float dx = cellStep * dataSetInformation.scale[0] * dxCoords / maxDeltaCoords;
    float dy = cellStep * dataSetInformation.scale[1] * dyCoords / maxDeltaCoords;
    float dz = cellStep * dataSetInformation.scale[2] * dzCoords / maxDeltaCoords;
    //volumeData->setGridExtent(xs, ys, zs, cellStep, cellStep, cellStep);
    volumeData->setGridExtent(xs, ys, zs, dx, dy, dz);
    delete[] zCoords;
    delete[] yCoords;
    delete[] xCoords;

    // Set the names of the existing fields/datasets.
    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    for (size_t datasetIdx = 0; datasetIdx < keys.size(); datasetIdx++) {
        if (getShape(datasetIdx).size() >= 3) {
            fieldNameMap[FieldType::SCALAR].push_back(keys.at(datasetIdx));
        }
    }
    volumeData->setFieldNames(fieldNameMap);

    // Read the time steps from the file.
    ts = 1;
    if (getDatasetExists("time")) {
        size_t timeDatasetIdx = getDatasetIndex("time");
        auto timeShape = getShape(timeDatasetIdx);
        if (timeShape.size() != 1) {
            sgl::Logfile::get()->throwError("ZarrLoader::setInputFiles: Invalid time shape.");
        }
        ts = int(timeShape.front());
        float* timeSteps = nullptr;
        loadFloatArray1D(timeDatasetIdx, ts, timeSteps);
        std::vector<float> timeStepsVector;
        timeStepsVector.assign(timeSteps, timeSteps + ts);
        delete[] timeSteps;
        volumeData->setTimeSteps(timeStepsVector);
    }

    return true;
}

bool ZarrLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    auto it = datasetNameMap.find(fieldName);
    if (it == datasetNameMap.end()) {
        sgl::Logfile::get()->throwError(
                "Error in ZarrLoader::getFieldEntry: Unknown field name \"" + fieldName + "\".");
        return false;
    }

    size_t datasetIdx = getDatasetIndex(fieldName);
    float* fieldEntryBuffer = nullptr;
    loadFloatArray3D(datasetIdx, timestepIdx, zs, ys, xs, fieldEntryBuffer);
    fieldEntry = new HostCacheEntryType(xs * ys * zs, fieldEntryBuffer);

    return true;
}
