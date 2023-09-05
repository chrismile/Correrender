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

#ifndef CORRERENDER_NIFTILOADER_HPP
#define CORRERENDER_NIFTILOADER_HPP

#include "VolumeLoader.hpp"

struct nifti_1_header;

/**
 * Loader for the .nii file format. See:
 * - https://github.com/NIFTI-Imaging/nifti_clib
 * - https://nifti.nimh.nih.gov/nifti-1/
 */
class NiftiLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "nii" }; }
    ~NiftiLoader();
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;
    bool getHasFloat32Data() override;

private:
    std::string dataSourceFilename;
    DataSetInformation dataSetInformation;
    int xs = 0, ys = 0, zs = 0;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    ptrdiff_t dataOffset = 0;
    std::string scalarAttributeName;
    nifti_1_header* header = nullptr;
};

#endif //CORRERENDER_NIFTILOADER_HPP
