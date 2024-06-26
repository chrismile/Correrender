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

#ifndef CORRERENDER_NETCDFLOADER_HPP
#define CORRERENDER_NETCDFLOADER_HPP

#include <unordered_map>
#include "VolumeLoader.hpp"

/**
 * For more details on the NetCDF file format see: https://zarr.readthedocs.io/en/stable/
 * Used underlying loader library: https://github.com/constantinpape/z5
 */
class NetCdfLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "nc" }; }
    NetCdfLoader();
    ~NetCdfLoader() override;
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;
    bool getHasFloat32Data() override { return !dataSetInformation.useFormatCast; }
    // Metadata reuse for individual time step or ensemble member files can potentially speed up loading.
    bool getSupportsMetadataReuse() override { return true; }
    bool setMetadataFrom(VolumeLoader* other) override;

private:
    bool getDimensionExists(const std::string& dimensionName);
    bool getVariableExists(const std::string& variableName);

    /**
     * Loads a 1D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varname The name of the variable, e.g. "time".
     * @param offset The read offset in the dimension.
     * @param len The dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray1D(const char* varname, size_t offset, size_t len, float*& array);

    /**
     * Loads a 1D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param offset The read offset in the dimension.
     * @param len The dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray1D(int varid, size_t offset, size_t len, float*& array);

    /**
     * Loads a 3D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param yoff Read offset in the y dimension.
     * @param xoff Read offset in the x dimension.
     * @param ylen Dimension size queried by @ref getDim.
     * @param xlen Dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray2D(int varid, size_t yoff, size_t xoff, size_t ylen, size_t xlen, float*& array);

    /**
     * Loads a 3D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param time The time step to load.
     * @param yoff Read offset in the y dimension.
     * @param xoff Read offset in the x dimension.
     * @param ylen Dimension size queried by @ref getDim.
     * @param xlen Dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray2D(int varid, size_t time, size_t yoff, size_t xoff, size_t ylen, size_t xlen, float*& array);

    /**
     * Loads a 3D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param zoff Read offset in the z dimension.
     * @param yoff Read offset in the y dimension.
     * @param xoff Read offset in the x dimension.
     * @param zlen Dimension size queried by @ref getDim.
     * @param ylen Dimension size queried by @ref getDim.
     * @param xlen Dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray3D(
            int varid, size_t zoff, size_t yoff, size_t xoff, size_t zlen, size_t ylen, size_t xlen, float*& array);

    /**
     * Loads a 3D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param time The time step to load.
     * @param zoff Read offset in the z dimension.
     * @param yoff Read offset in the y dimension.
     * @param xoff Read offset in the x dimension.
     * @param zlen Dimension size queried by @ref getDim.
     * @param ylen Dimension size queried by @ref getDim.
     * @param xlen Dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArray3D(
            int varid, size_t time, size_t zoff, size_t yoff, size_t xoff, size_t zlen, size_t ylen, size_t xlen,
            float*& array);

    /**
     * Loads a 4D floating point variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param coff Read offset in the channel dimension.
     * @param zoff Read offset in the z dimension.
     * @param yoff Read offset in the y dimension.
     * @param xoff Read offset in the x dimension.
     * @param clen Dimension size queried by @ref getDim.
     * @param zlen Dimension size queried by @ref getDim.
     * @param ylen Dimension size queried by @ref getDim.
     * @param xlen Dimension size queried by @ref getDim.
     * @param array A pointer to a float array where the variable data is to be stored. The function will automatically
     * allocate the memory. The caller needs to deallocate the allocated memory using "delete[]".
     */
    void loadFloatArrayColorCZYX(
            int varid, size_t coff, size_t zoff, size_t yoff, size_t xoff,
            size_t clen, size_t zlen, size_t ylen, size_t xlen, float*& array);

    /**
     * Queries a string attribute of a variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param attname The name of the attribute to query.
     * @return The attribute string.
     */
    std::string getStringAttribute(int varid, const char* attname);

    /**
     * Queries a float attribute of a variable.
     * @param ncid The NetCDF file ID.
     * @param varid The ID of the variable.
     * @param attname The name of the attribute to query.
     * @return The attribute string.
     */
    float getFloatAttribute(int varid, const char* attname);

    bool isOpen = false;
    int ncid = 0;
    std::string filePath;
    DataSetInformation dataSetInformation;
    std::unordered_map<std::string, int> datasetNameMap;
    int xs = 0, ys = 0, zs = 0, ts = 0, es = 0;
    int xmin = 0, ymin = 0, zmin = 0; // In case of subselection: Offset in each dimension.
    int xst = 0, yst = 0, zst = 0; // In case of subselection: True domain size.
    std::vector<float> timeDependent2dMap;
    bool reusedMetadata = false;
    bool isColorData = false; // For storing pre-shaded volume data.

    // Fill values are optional and replaced with NaN for visualization purposes.
    std::vector<bool> varHasFillValueMap;
    std::vector<float> fillValueMap;
};

#endif //CORRERENDER_NETCDFLOADER_HPP
