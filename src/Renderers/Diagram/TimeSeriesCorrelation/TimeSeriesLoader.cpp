/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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
#include <limits>
#include <cstring>
#include <cassert>
#include <netcdf.h>

#include <Utils/File/Logfile.hpp>

#include "TimeSeriesLoader.hpp"

#if defined(DEBUG) || !defined(NDEBUG)
#define myassert assert
#else
#define myassert(x)                                   \
	if (!(x))                                         \
	{                                                 \
		std::cerr << "assertion failed" << std::endl; \
		exit(1);                                      \
	}
#endif

void TimeSeriesLoader::loadFloatArray2D(int varid, size_t ylen, size_t xlen, float*& array) {
    array = new float[ylen * xlen];
    size_t startp[] = { 0, 0 };
    size_t countp[] = { ylen, xlen };

    nc_type vartype;
    myassert(nc_inq_vartype(ncid, varid, &vartype) == NC_NOERR);
    if (vartype == NC_FLOAT) {
        myassert(nc_get_vara_float(ncid, varid, startp, countp, array) == NC_NOERR);
    } else if (vartype == NC_DOUBLE) {
        size_t len = ylen * xlen;
        auto* arrayDouble = new double[len];
        myassert(nc_get_vara_double(ncid, varid, startp, countp, arrayDouble) == NC_NOERR);
        for (size_t i = 0; i < len; i++) {
            array[i] = float(arrayDouble[i]);
        }
        delete[] arrayDouble;
    }
}

void TimeSeriesLoader::loadFloatArray3D(int varid, size_t zlen, size_t ylen, size_t xlen, float*& array) {
    array = new float[zlen * ylen * xlen];
    size_t startp[] = { 0, 0, 0 };
    size_t countp[] = { zlen, ylen, xlen };

    nc_type vartype;
    myassert(nc_inq_vartype(ncid, varid, &vartype) == NC_NOERR);
    if (vartype == NC_FLOAT) {
        myassert(nc_get_vara_float(ncid, varid, startp, countp, array) == NC_NOERR);
    } else if (vartype == NC_DOUBLE) {
        size_t len = zlen * ylen * xlen;
        auto* arrayDouble = new double[len];
        myassert(nc_get_vara_double(ncid, varid, startp, countp, arrayDouble) == NC_NOERR);
        for (size_t i = 0; i < len; i++) {
            array[i] = float(arrayDouble[i]);
        }
        delete[] arrayDouble;
    }
}

float TimeSeriesLoader::getFloatAttribute(int varid, const char* attname) {
    float attrValue = 0.0f;
    nc_type type;
    myassert(nc_inq_atttype(ncid, varid, attname, &type) == NC_NOERR);
    if (type == NC_FLOAT) {
        nc_get_att_float(ncid, varid, attname, &attrValue);
    } else if (type == NC_DOUBLE) {
        double attrValueDouble = 0.0;
        nc_get_att_double(ncid, varid, attname, &attrValueDouble);
        attrValue = float(attrValueDouble);
    } else {
        sgl::Logfile::get()->throwError("Error in NetCdfLoader::getFloatAttribute: Invalid attribute type.");
    }
    return attrValue;
}

TimeSeriesLoader::~TimeSeriesLoader() {
    close();
}

bool TimeSeriesLoader::open(const std::string& _filePath) {
    filePath = _filePath;
    int status = nc_open(filePath.c_str(), NC_NOWRITE, &ncid);
    if (status != 0) {
        sgl::Logfile::get()->writeError(
                "Error in TimeSeriesLoader::open: File \"" + filePath + "\" couldn't be opened.");
        return false;
    }
    isOpen = true;

    int ndims = 0, nvarsp = 0;
    int dimids[NC_MAX_VAR_DIMS];
    char varname[NC_MAX_NAME];
    myassert(nc_inq(ncid, &ndims, &nvarsp, nullptr, nullptr) == NC_NOERR);

    int ndimsVar = 0;
    int nattsVar = 0;
    for (int varid = 0; varid < nvarsp; varid++) {
        nc_type type = NC_FLOAT;
        nc_inq_var(ncid, varid, varname, &type, &ndimsVar, dimids, &nattsVar);
        if ((type == NC_FLOAT || type == NC_DOUBLE) && (ndimsVar == 2 || ndimsVar == 3)) {
            varidData = varid;
            nattsVarData = nattsVar;
            break;
        }
    }

    if (varidData == -1) {
        sgl::Logfile::get()->writeError(
                "Error in TimeSeriesLoader::open: File \"" + filePath + "\" does not contain a valid variable.");
        return false;
    }

    char dimNameSamples[NC_MAX_NAME + 1];
    char dimNameTime[NC_MAX_NAME + 1];
    char dimNameWindow[NC_MAX_NAME + 1];
    size_t samples64, time64, window64;
    myassert(nc_inq_dim(ncid, dimids[0], dimNameSamples, &samples64) == NC_NOERR);
    myassert(nc_inq_dim(ncid, dimids[1], dimNameTime, &time64) == NC_NOERR);
    metadata.samples = int(samples64);
    metadata.time = int(time64);
    if (strcmp(dimNameSamples, "samples") != 0) {
        sgl::Logfile::get()->writeError(
                "Error in TimeSeriesLoader::open: Expected first dimension 'samples' in file \"" + filePath + "\".");
        return false;
    }
    if (strcmp(dimNameTime, "time") != 0) {
        sgl::Logfile::get()->writeError(
                "Error in TimeSeriesLoader::open: Expected second dimension 'time' in file \"" + filePath + "\".");
        return false;
    }
    if (ndimsVar == 3) {
        myassert(nc_inq_dim(ncid, dimids[2], dimNameWindow, &window64) == NC_NOERR);
        metadata.window = int(window64);
        if (strcmp(dimNameSamples, "samples") != 0) {
            sgl::Logfile::get()->writeError(
                    "Error in TimeSeriesLoader::open: Expected third dimension 'windows' in file \"" + filePath + "\".");
            return false;
        }
    }

    return true;
}

bool TimeSeriesLoader::close() {
    if (isOpen && nc_close(ncid) != NC_NOERR) {
        sgl::Logfile::get()->throwError(
                "Error in TimeSeriesLoader::close: nc_close failed for file \"" + filePath + "\".");
        return false;
    }
    isOpen = false;
    return true;
}

const TimeSeriesMetadata& TimeSeriesLoader::getMetadata() {
    if (!isOpen) {
        sgl::Logfile::get()->throwError("Error in TimeSeriesLoader::getMetadata: No file is opened for reading.");
    }
    return metadata;
}

TimeSeriesDataPtr TimeSeriesLoader::loadData() {
    if (!isOpen) {
        sgl::Logfile::get()->throwError("Error in TimeSeriesLoader::loadData: No file is opened for reading.");
    }

    float* dataBuffer = nullptr;
    if (metadata.window < 0) {
        loadFloatArray2D(varidData, metadata.samples, metadata.time, dataBuffer);
    } else {
        loadFloatArray3D(varidData, metadata.samples, metadata.time, metadata.window, dataBuffer);
    }

    bool hasFillValue = false;
    char attname[NC_MAX_NAME];
    float fillValue = std::numeric_limits<float>::quiet_NaN();
    for (int attnum = 0; attnum < nattsVarData; attnum++) {
        nc_inq_attname(ncid, varidData, attnum, attname);
        if (strcmp(attname, "missing_value") == 0) {
            hasFillValue = true;
            fillValue = getFloatAttribute(varidData, "missing_value");
        } else if (strcmp(attname, "_FillValue") == 0) {
            hasFillValue = true;
            fillValue = getFloatAttribute(varidData, "_FillValue");
        }
    }

    if (hasFillValue) {
        int numEntries = metadata.samples * metadata.time;
        if (metadata.window > 0) {
            numEntries *= metadata.window;
        }
        for (int i = 0; i < numEntries; i++) {
            if (dataBuffer[i] == fillValue) {
                dataBuffer[i] = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    auto timeSeriesData = std::make_shared<TimeSeriesData>(metadata, dataBuffer);
    return timeSeriesData;
}
