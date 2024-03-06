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

#ifndef CORRERENDER_VOLUMELOADER_HPP
#define CORRERENDER_VOLUMELOADER_HPP

#include <vector>
#include <string>
#include "Volume/FieldType.hpp"
#include "DataSetList.hpp"

class HostCacheEntryType;
class VolumeData;

class VolumeLoader {
public:
    virtual ~VolumeLoader() = default;
    virtual bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) = 0;
    virtual bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) = 0;
    virtual bool getHasFloat32Data() { return true; }
    // Metadata reuse for individual time step or ensemble member files can potentially speed up loading.
    virtual bool getSupportsMetadataReuse() { return false; }
    virtual bool setMetadataFrom(VolumeLoader* other) { return false; }
};

#endif //CORRERENDER_VOLUMELOADER_HPP
