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

#ifndef CORRERENDER_TIMESERIESLOADER_HPP
#define CORRERENDER_TIMESERIESLOADER_HPP

#include "TimeSeries.hpp"

class TimeSeriesLoader {
public:
    ~TimeSeriesLoader();
    bool open(const std::string& _filePath);
    bool close();
    const TimeSeriesMetadata& getMetadata();
    TimeSeriesDataPtr loadData();

private:
    bool isOpen = false;
    int ncid = 0;
    int varidData = -1;
    int nattsVarData = -1;
    std::string filePath;
    TimeSeriesMetadata metadata{};

    /**
     * Loads a 3D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param ylen Dimension size queried by @ref getDim.
     * @param xlen Dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray2D(int varid, size_t ylen, size_t xlen, float*& array);

    /**
     * Loads a 3D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param zlen Dimension size queried by @ref getDim.
     * @param ylen Dimension size queried by @ref getDim.
     * @param xlen Dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray3D(int varid, size_t zlen, size_t ylen, size_t xlen, float*& array);

    /**
     * Queries a float attribute of a variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param attname The name of the attribute to query.
     * @return The attribute string.
     */
    float getFloatAttribute(int varid, const char* attname);
};

#endif //CORRERENDER_TIMESERIESLOADER_HPP
