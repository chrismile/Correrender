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

#include <iostream>
#include <set>
#include <cassert>
#include <cstring>

#include <netcdf.h>

#include <Utils/File/Logfile.hpp>

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "NetCdfLoader.hpp"

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

NetCdfLoader::NetCdfLoader() = default;

NetCdfLoader::~NetCdfLoader() {
    if (nc_close(ncid) != NC_NOERR) {
        sgl::Logfile::get()->throwError(
                "Error in NetCdfLoader::load: nc_close failed for file \"" + filePath + "\".");
    }
}

bool NetCdfLoader::getDimensionExists(const std::string& dimensionName) {
    int dimid = 0;
    int status = nc_inq_dimid(ncid, dimensionName.c_str(), &dimid);
    return status != NC_EBADDIM;
}

bool NetCdfLoader::getVariableExists(const std::string& variableName) {
    int varid;
    int status = nc_inq_varid(ncid, variableName.c_str(), &varid);
    return status != NC_ENOTVAR;
}

void NetCdfLoader::loadFloatArray1D(const char* varname, size_t len, float*& array) {
    int varid;
    myassert(nc_inq_varid(ncid, varname, &varid) == NC_NOERR);

    array = new float[len];
    size_t startp[] = { 0 };
    size_t countp[] = { len };

    nc_type vartype;
    myassert(nc_inq_vartype(ncid, varid, &vartype) == NC_NOERR);
    if (vartype == NC_FLOAT) {
        myassert(nc_get_vara_float(ncid, varid, startp, countp, array) == NC_NOERR);
    } else if (vartype == NC_DOUBLE) {
        auto* arrayDouble = new double[len];
        myassert(nc_get_vara_double(ncid, varid, startp, countp, arrayDouble) == NC_NOERR);
        for (size_t i = 0; i < len; i++) {
            array[i] = float(arrayDouble[i]);
        }
        delete[] arrayDouble;
    }
}

void NetCdfLoader::loadFloatArray1D(int varid, size_t len, float*& array) {
    array = new float[len];
    size_t startp[] = { 0 };
    size_t countp[] = { len };

    nc_type vartype;
    myassert(nc_inq_vartype(ncid, varid, &vartype) == NC_NOERR);
    if (vartype == NC_FLOAT) {
        myassert(nc_get_vara_float(ncid, varid, startp, countp, array) == NC_NOERR);
    } else if (vartype == NC_DOUBLE) {
        auto* arrayDouble = new double[len];
        myassert(nc_get_vara_double(ncid, varid, startp, countp, arrayDouble) == NC_NOERR);
        for (size_t i = 0; i < len; i++) {
            array[i] = float(arrayDouble[i]);
        }
        delete[] arrayDouble;
    }
}

void NetCdfLoader::loadFloatArray3D(int varid, size_t zlen, size_t ylen, size_t xlen, float*& array) {
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

void NetCdfLoader::loadFloatArray3D(int varid, size_t time, size_t zlen, size_t ylen, size_t xlen, float*& array) {
    array = new float[zlen * ylen * xlen];
    size_t startp[] = { time, 0, 0, 0 };
    size_t countp[] = { 1, zlen, ylen, xlen };

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

std::string NetCdfLoader::getStringAttribute(int varid, const char* attname) {
    nc_type type;
    size_t length = 0;
    myassert(nc_inq_att(ncid, varid, attname, &type, &length) == NC_NOERR);
    myassert(type == NC_CHAR);
    char* charArray = new char[length];
    nc_get_att_text(ncid, varid, attname, charArray);
    std::string attText = std::string(charArray, length);
    delete[] charArray;
    return attText;
}

bool NetCdfLoader::setInputFiles(
        VolumeData* volumeData, const std::string& _filePath, const DataSetInformation& _dataSetInformation) {
    filePath = _filePath;
    dataSetInformation = _dataSetInformation;
    int status = nc_open(filePath.c_str(), NC_NOWRITE, &ncid);
    if (status != 0) {
        sgl::Logfile::get()->writeError(
                "Error in NetCdfLoader::load: File \"" + filePath + "\" couldn't be opened.");
        return false;
    }

    // Temporary data for storing information about the variables in the data file.
    int nvarsp = 0;
    int dimids[NC_MAX_VAR_DIMS];
    char varname[NC_MAX_NAME];
    char attname[NC_MAX_NAME];
    myassert(nc_inq(ncid, nullptr, &nvarsp, nullptr, nullptr) == NC_NOERR);

    bool uLowerCaseVariableExists = getVariableExists("u");
    bool vLowerCaseVariableExists = getVariableExists("v");
    bool wLowerCaseVariableExists = getVariableExists("w");
    bool uUpperCaseVariableExists = getVariableExists("U");
    bool vUpperCaseVariableExists = getVariableExists("V");
    bool wUpperCaseVariableExists = getVariableExists("W");

    // Get the wind speed variable IDs.
    int varIdRepresentative = -1, varIdU = -1, varIdV = -1, varIdW = -1;
    if (uLowerCaseVariableExists && vLowerCaseVariableExists && wLowerCaseVariableExists) {
        myassert(nc_inq_varid(ncid, "u", &varIdU) == NC_NOERR);
        myassert(nc_inq_varid(ncid, "v", &varIdV) == NC_NOERR);
        myassert(nc_inq_varid(ncid, "w", &varIdW) == NC_NOERR);
        varIdRepresentative = varIdU;
    } else if (uUpperCaseVariableExists && vUpperCaseVariableExists && wUpperCaseVariableExists) {
        myassert(nc_inq_varid(ncid, "U", &varIdU) == NC_NOERR);
        myassert(nc_inq_varid(ncid, "V", &varIdV) == NC_NOERR);
        myassert(nc_inq_varid(ncid, "W", &varIdW) == NC_NOERR);
        varIdRepresentative = varIdU;
    } else {
        // Get the first variable with 3 or 4 dimensions.
        for (int varid = 0; varid < nvarsp; varid++) {
            nc_type type = NC_FLOAT;
            int ndims = 0;
            int natts = 0;
            nc_inq_var(ncid, varid, varname, &type, &ndims, dimids, &natts);
            if ((type == NC_FLOAT || type == NC_DOUBLE) && ndims == 4) {
                varIdRepresentative = varid;
                break;
            }
        }
        if (varIdRepresentative == -1) {
            for (int varid = 0; varid < nvarsp; varid++) {
                nc_type type = NC_FLOAT;
                int ndims = 0;
                int natts = 0;
                nc_inq_var(ncid, varid, varname, &type, &ndims, dimids, &natts);
                if ((type == NC_FLOAT || type == NC_DOUBLE) && ndims == 3) {
                    varIdRepresentative = varid;
                    break;
                }
            }
        }
        if (varIdRepresentative < 0) {
            sgl::Logfile::get()->throwError(
                    "Error in NetCdfLoader::load: Could not find u, v, w (or U, V, W) wind speeds or any other "
                    "representative variable in file \"" + filePath + "\".");
        }
    }

    float* zCoords = nullptr;
    float* yCoords = nullptr;
    float* xCoords = nullptr;

    // Get dimensions of the wind speed variables.
    // Examples: (time, level, rlat, rlon)
    // U(time, altitude, rlat, srlon), V(time, altitude, srlat, rlon), W(time, altitude, rlat, rlon), T(time, level, rlat, rlon), ...
    // U(time, level, rlat, srlon), V(time, level, srlat, rlon), W(time, level1, rlat, rlon), T(time, level, rlat, rlon), ...
    // u(zdim, ydim, xdim), v(zdim, ydim, xdim), w(zdim, ydim, xdim)
    int numDims = 0;
    myassert(nc_inq_varndims(ncid, varIdRepresentative, &numDims) == NC_NOERR);
    bool isLatLonData = false;
    if (numDims == 3) {
        // Assuming (zdim, ydim, xdim).
        int dimensionIds[3];
        char dimNameZ[NC_MAX_NAME + 1];
        char dimNameY[NC_MAX_NAME + 1];
        char dimNameX[NC_MAX_NAME + 1];
        myassert(nc_inq_vardimid(ncid, varIdRepresentative, dimensionIds) == NC_NOERR);
        size_t zs64, ys64, xs64;
        myassert(nc_inq_dim(ncid, dimensionIds[0], dimNameZ, &zs64) == NC_NOERR);
        myassert(nc_inq_dim(ncid, dimensionIds[1], dimNameY, &ys64) == NC_NOERR);
        myassert(nc_inq_dim(ncid, dimensionIds[2], dimNameX, &xs64) == NC_NOERR);
        zs = int(zs64);
        ys = int(ys64);
        xs = int(xs64);
        zCoords = new float[zs];
        yCoords = new float[ys];
        xCoords = new float[xs];
        int varid;
        int retval = nc_inq_varid(ncid, dimNameZ, &varid);
        if (retval != NC_ENOTVAR) {
            loadFloatArray1D(dimNameZ, zs, zCoords);
            loadFloatArray1D(dimNameY, ys, yCoords);
            loadFloatArray1D(dimNameX, xs, xCoords);
        } else {
            for (int i = 0; i < zs; i++) {
                zCoords[i] = float(i);
            }
            for (int i = 0; i < ys; i++) {
                yCoords[i] = float(i);
            }
            for (int i = 0; i < xs; i++) {
                xCoords[i] = float(i);
            }
        }
    } else if (numDims == 4) {
        int dimensionIdsU[4];
        int dimensionIdsV[4];
        int dimensionIdsW[4];
        char dimNameTimeOrEnsemble[NC_MAX_NAME + 1];
        char dimNameZ[NC_MAX_NAME + 1];
        char dimNameY[NC_MAX_NAME + 1];
        char dimNameX[NC_MAX_NAME + 1];
        if (varIdU >= 0 && varIdV >= 0 && varIdW >= 0) {
            myassert(nc_inq_vardimid(ncid, varIdU, dimensionIdsU) == NC_NOERR);
            myassert(nc_inq_vardimid(ncid, varIdV, dimensionIdsV) == NC_NOERR);
            myassert(nc_inq_vardimid(ncid, varIdW, dimensionIdsW) == NC_NOERR);
        } else {
            myassert(nc_inq_vardimid(ncid, varIdRepresentative, dimensionIdsU) == NC_NOERR);
            myassert(nc_inq_vardimid(ncid, varIdRepresentative, dimensionIdsV) == NC_NOERR);
            myassert(nc_inq_vardimid(ncid, varIdRepresentative, dimensionIdsW) == NC_NOERR);
        }
        // Ignoring staggered grids for now by not querying i-th dimension using i-th variable.
        size_t tes64, zs64, ys64, xs64;
        myassert(nc_inq_dim(ncid, dimensionIdsU[0], dimNameTimeOrEnsemble, &tes64) == NC_NOERR);
        myassert(nc_inq_dim(ncid, dimensionIdsU[1], dimNameZ, &zs64) == NC_NOERR);
        myassert(nc_inq_dim(ncid, dimensionIdsW[2], dimNameY, &ys64) == NC_NOERR);
        myassert(nc_inq_dim(ncid, dimensionIdsW[3], dimNameX, &xs64) == NC_NOERR);
        if (strcmp(dimNameTimeOrEnsemble, "time") == 0) {
            ts = int(tes64);
        } else if (strcmp(dimNameTimeOrEnsemble, "ensemble") == 0
                || strcmp(dimNameTimeOrEnsemble, "member") == 0
                || strcmp(dimNameTimeOrEnsemble, "members") == 0) {
            es = int(tes64);
        } else {
            sgl::Logfile::get()->writeWarning(
                    "Warning in NetCdfLoader::setInputFiles: Unknown dimension name. Assuming time.");
            ts = int(tes64);
        }
        zs = int(zs64);
        ys = int(ys64);
        xs = int(xs64);
        zCoords = new float[zs];
        yCoords = new float[ys];
        xCoords = new float[xs];
        std::string stringDimNameX = dimNameX;
        std::string stringDimNameY = dimNameY;
        //std::string stringDimNameZ = dimNameZ;
        if (stringDimNameX.find("lon") != std::string::npos || stringDimNameY.find("lon") != std::string::npos
                || stringDimNameX.find("lat") != std::string::npos || stringDimNameY.find("lat") != std::string::npos) {
            isLatLonData = true;
        }

        if (!getVariableExists(dimNameZ) && getVariableExists("vcoord")) {
            loadFloatArray1D("vcoord", zs, zCoords);
            loadFloatArray1D(dimNameY, ys, yCoords);
            loadFloatArray1D(dimNameX, xs, xCoords);
        } else {
            int varid;
            int retval = nc_inq_varid(ncid, dimNameZ, &varid);
            if (retval != NC_ENOTVAR) {
                loadFloatArray1D(dimNameZ, zs, zCoords);
                loadFloatArray1D(dimNameY, ys, yCoords);
                loadFloatArray1D(dimNameX, xs, xCoords);
            } else {
                for (int i = 0; i < zs; i++) {
                    zCoords[i] = float(i);
                }
                for (int i = 0; i < ys; i++) {
                    yCoords[i] = float(i);
                }
                for (int i = 0; i < xs; i++) {
                    xCoords[i] = float(i);
                }
                isLatLonData = false;
            }
        }
    } else {
        sgl::Logfile::get()->throwError(
                "Error in NetCdfLoader::load: Invalid number of dimensions in file \""
                + filePath + "\".");
    }

    if (ts > 1) {
        volumeData->setNumTimeSteps(ts);
    }
    if (es > 1) {
        volumeData->setEnsembleMemberCount(es);
    }

    // TODO: Use coords also for lat-lon-pressure?
    float dxCoords = 1.0f;
    float dyCoords = 1.0f;
    float dzCoords = 1.0f;
    if (!isLatLonData) {
        // Assume regular grid.
        dzCoords = (zCoords[zs - 1] - zCoords[0]) / float(zs - 1);
        dyCoords = (yCoords[ys - 1] - yCoords[0]) / float(ys - 1);
        dxCoords = (xCoords[xs - 1] - xCoords[0]) / float(xs - 1);
    }
    float maxDeltaCoords = std::max(dxCoords, std::max(dyCoords, dzCoords));

    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    float dx = cellStep * dataSetInformation.scale[0] * dxCoords / maxDeltaCoords;
    float dy = cellStep * dataSetInformation.scale[1] * dyCoords / maxDeltaCoords;
    float dz = cellStep * dataSetInformation.scale[2] * dzCoords / maxDeltaCoords;
    volumeData->setGridExtent(int(xs), int(ys), int(zs), dx, dy, dz);
    delete[] zCoords;
    delete[] yCoords;
    delete[] xCoords;

    if (!dataSetInformation.hasCustomDateTime) {
        volumeData->setNumTimeSteps(ts);
    }

    // Set the names of the existing fields/datasets.
    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    for (int varid = 0; varid < nvarsp; varid++) {
        nc_type type = NC_FLOAT;
        int ndims = 0;
        int natts = 0;
        nc_inq_var(ncid, varid, varname, &type, &ndims, dimids, &natts);
        if ((type != NC_FLOAT && type != NC_DOUBLE) || (ndims != 3 && ndims != 4)) {
            continue;
        }

        std::string variableDisplayName = varname;
        for (int attnum = 0; attnum < natts; attnum++) {
            nc_inq_attname(ncid, varid, attnum, attname);
            if (strcmp(attname, "standard_name") == 0) {
                variableDisplayName = getStringAttribute(varid, "standard_name");
            }
        }

        fieldNameMap[FieldType::SCALAR].push_back(variableDisplayName);
        datasetNameMap.insert(std::make_pair(varname, varid));
        datasetNameMap.insert(std::make_pair(variableDisplayName, varid));
    }
    volumeData->setFieldNames(fieldNameMap);

    return true;
}

bool NetCdfLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    auto it = datasetNameMap.find(fieldName);
    if (it == datasetNameMap.end()) {
        sgl::Logfile::get()->throwError(
                "Error in NetCdfLoader::getFieldEntry: Unknown field name \"" + fieldName + "\".");
        return false;
    }

    if (dataSetInformation.hasCustomDateTime) {
        timestepIdx = dataSetInformation.time;
    }

    int varid = it->second;
    int numDims = 0;
    //int dimids[NC_MAX_VAR_DIMS];
    myassert(nc_inq_varndims(ncid, varid, &numDims) == NC_NOERR);
    //myassert(nc_inq_vardimid(ncid, varid, dimids) == NC_NOERR);

    float* fieldEntryBuffer = nullptr;
    if (numDims == 3) {
        loadFloatArray3D(varid, zs, ys, xs, fieldEntryBuffer);
    } else if (numDims == 4 && ts > 1) {
        loadFloatArray3D(varid, timestepIdx, zs, ys, xs, fieldEntryBuffer);
    } else if (numDims == 4 && es > 1) {
        loadFloatArray3D(varid, memberIdx, zs, ys, xs, fieldEntryBuffer);
    }

    fieldEntry = new HostCacheEntryType(xs * ys * zs, fieldEntryBuffer);
    if (dataSetInformation.useFormatCast) {
        fieldEntry->switchNativeFormat(dataSetInformation.formatTarget);
    }

    return true;
}
