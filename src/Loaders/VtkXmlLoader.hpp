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

#ifndef CORRERENDER_VTKXMLLOADER_HPP
#define CORRERENDER_VTKXMLLOADER_HPP

#include "VolumeLoader.hpp"

namespace tinyxml2 {
class XMLDocument;
}

/**
 * A VTK XML file loader. Currently, only serial image data (.vti) files are supported. For more details see:
 * pp. 11 - 19, XML File Formats: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
 * https://mathema.tician.de/what-they-dont-tell-you-about-vtk-xml-binary-formats/
 * https://vtk.org/Wiki/VTK_XML_Formats
 */
class VtkXmlLoader : public VolumeLoader {
public:
    static std::vector<std::string> getSupportedExtensions() { return { "vti", "vts", "vtr" }; }
    ~VtkXmlLoader() override;
    bool setInputFiles(
            VolumeData* volumeData, const std::string& filePath, const DataSetInformation& dataSetInformation) override;
    bool getFieldEntry(
            VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
            int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) override;

private:
    std::string dataSourceFilename;
    DataSetInformation dataSetInformation;
    tinyxml2::XMLDocument* doc = nullptr;
    int xs = 0, ys = 0, zs = 0;
    int numPoints = 0;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    bool useZlib = false;
    bool isLittleEndian = false;
    const char* appendedDataEncoded = nullptr;
    ptrdiff_t startPos = 0;
    size_t velocityOffsets[3] = { 0, 0, 0 };
    std::map<std::string, size_t> variableOffsets;
    std::map<std::string, uint32_t> variableDataSize;
    size_t numHeaderBytes = 4;
    uint8_t* rawData = nullptr;
    size_t rawDataSize = 0;
};

#endif //CORRERENDER_VTKXMLLOADER_HPP
