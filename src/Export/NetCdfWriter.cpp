/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2019-2022, Christoph Neuhauser
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

#include <netcdf.h>
#include <Utils/File/Logfile.hpp>

#include "Volume/VolumeData.hpp"
#include "NetCdfWriter.hpp"

void NetCdfWriter::ncPutAttributeText(int varid, const std::string &name, const std::string &value) {
    nc_put_att_text(ncid, varid, name.c_str(), value.size(), value.c_str());
}

bool NetCdfWriter::writeFieldToFile(
        const std::string& filePath, VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx) {
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    float xOrigin = 0.0f;
    float yOrigin = 0.0f;
    float zOrigin = 0.0f;
    float dx = volumeData->getDx();
    float dy = volumeData->getDy();
    float dz = volumeData->getDz();

    int status = nc_create(filePath.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid);
    if (status != 0) {
        sgl::Logfile::get()->writeError(
                "Error in NetCdfWriter::writeFieldToFile: File \"" + filePath + "\" couldn't be opened.");
        return false;
    }

    ncPutAttributeText(NC_GLOBAL, "Conventions", "CF-1.5");
    ncPutAttributeText(NC_GLOBAL, "title", "Exported scalar field");
    ncPutAttributeText(NC_GLOBAL, "history", "Correrender");
    ncPutAttributeText(NC_GLOBAL, "institution", "Technical University of Munich, Chair of Computer Graphics and Visualization");
    ncPutAttributeText(NC_GLOBAL, "source", "Scalar field exported by Correrender, a correlation field volume renderer.");
    ncPutAttributeText(NC_GLOBAL, "references", "https://github.com/chrismile/Correrender");
    ncPutAttributeText(NC_GLOBAL, "comment", "Correrender is released under the 2-clause BSD license.");

    // Create dimensions.
    int xDim, yDim, zDim;
    nc_def_dim(ncid, "x", xs, &xDim);
    nc_def_dim(ncid, "y", ys, &yDim);
    nc_def_dim(ncid, "z", zs, &zDim);

    // Define the cell center variables.
    nc_def_var(ncid, "x", NC_FLOAT, 1, &xDim, &xVar);
    nc_def_var(ncid, "y", NC_FLOAT, 1, &yDim, &yVar);
    nc_def_var(ncid, "z", NC_FLOAT, 1, &zDim, &zVar);

    ncPutAttributeText(xVar, "coordinate_type", "Cartesian X");
    ncPutAttributeText(yVar, "coordinate_type", "Cartesian Y");
    ncPutAttributeText(zVar, "coordinate_type", "Cartesian Z");
    // VTK interprets X, Y and Z as longitude, latitude and vertical.
    // Refer to VTK/IO/NetCDF/vtkNetCDFCFReader.cxx for more information.
    // ncPutAttributeText(xVar, "axis", "X");
    // ncPutAttributeText(yVar, "axis", "Y");
    // ncPutAttributeText(zVar, "axis", "Z");

    int dimsTimeIndependent3D[] = { zDim, yDim, xDim };
    nc_def_var(ncid, fieldName.c_str(), NC_FLOAT, 3, dimsTimeIndependent3D, &scalarVar);

    // Write the grid cell centers to the x, y and z variables.
    float gridPosition = xOrigin + 0.0f * dx;
    for (size_t x = 0; x < (size_t)xs; x++) {
        nc_put_var1_float(ncid, xVar, &x, &gridPosition);
        gridPosition += dx;
    }
    gridPosition = yOrigin + 0.0f * dy;
    for (size_t y = 0; y < (size_t)ys; y++) {
        nc_put_var1_float(ncid, yVar, &y, &gridPosition);
        gridPosition += dy;
    }
    gridPosition = zOrigin + 0.0f * dz;
    for (size_t z = 0; z < (size_t)zs; z++) {
        nc_put_var1_float(ncid, zVar, &z, &gridPosition);
        gridPosition += dz;
    }

    auto fieldData = volumeData->getFieldEntryCpu(fieldType, fieldName, timestepIdx, memberIdx);
    size_t start[] = { 0, 0, 0 };
    size_t count[] = { 1, 1, size_t(xs) };
    for (int z = 0; z < zs; z++) {
        start[0] = z;
        for (int y = 0; y < ys; y++) {
            start[1] = y;
            nc_put_vara_float(ncid, scalarVar, start, count, fieldData->data<float>() + y * xs + z * xs * ys);
        }
    }

    if (nc_close(ncid) != NC_NOERR) {
        sgl::Logfile::get()->writeError(
                "Error in NetCdfWriter::writeFieldToFile: nc_close failed for file \"" + filePath + "\".");
        return false;
    }

    return true;
}
