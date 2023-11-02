/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
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

#ifndef CORRERENDER_CTLLOADER_HPP
#define CORRERENDER_CTLLOADER_HPP

#include <vector>
#include <unordered_map>
#include <cstdint>
#include "VolumeLoader.hpp"

struct CtlVarDesc {
    std::string name;
    ptrdiff_t offset = 0; //< Offset within one time step.
    ptrdiff_t numLevels = 0;
    ptrdiff_t size3d = 0; //< Size in 3D.
};

struct CtlInfo {
    ptrdiff_t xs = 0, ys = 0, zs = 0, ts = 1, es = 1;
    ptrdiff_t sizeAllVars3d = 0; //< Order in memory: es > ts > var > zs > ys > xs
    bool isBigEndian = false;
    bool isSequential = false; //< FORTRAN sequential data with header? (not supported so far)
    float fillValue = std::numeric_limits<float>::quiet_NaN();
};

/**
 * The .ctl control file format was created for the software GrADS (http://cola.gmu.edu/grads/gadoc/gadoc.php).
 * For more details on the file format see:
 * - General description: http://cola.gmu.edu/grads/gadoc/aboutgriddeddata.html
 * - Format documentation: http://cola.gmu.edu/grads/gadoc/descriptorfile.html
 * - Examples: http://cola.gmu.edu/grads/gadoc/aboutgriddeddata.html#formats
 * - Further information: https://www.ncl.ucar.edu/Applications/grads.shtml
 */
class CtlLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "ctl" }; }
    CtlLoader();
    ~CtlLoader() override;
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;
    bool getHasFloat32Data() override { return !dataSetInformation.useFormatCast; }

private:
    DataSetInformation dataSetInformation;
    CtlInfo info;
    std::unordered_map<std::string, int> variableNameMap;
    std::vector<CtlVarDesc> variableDescriptors;
    float* lon1d = nullptr;
    float* lat1d = nullptr;
    float* lev1d = nullptr;

    bool parseDef(
            char*& fileBuffer, size_t& charPtr, size_t& lengthCtl,
            std::string& lineBuffer, std::vector<std::string>& splitLineString);

    // Depending on the OS, different underlying methods may be used for reading from the file.
    void loadDataFromFile(uint8_t* destBuffer, ptrdiff_t offset, ptrdiff_t size);
    bool openDataFile(const std::string& dataFileName);
    void closeDataFile();
    FILE* file = nullptr;
};

#endif //CORRERENDER_CTLLOADER_HPP
