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

#include <Math/Math.hpp>
#include <Utils/Convert.hpp>
#include <Utils/XML.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/Zlib.hpp>
#include <Utils/File/FileLoader.hpp>

#include "base64/base64.h"
#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "LoadersUtil.hpp"
#include "VtkXmlLoader.hpp"

VtkXmlLoader::~VtkXmlLoader() {
    if (doc) {
        delete doc;
        doc = nullptr;
    }
    if (rawData) {
        delete[] rawData;
        rawData = nullptr;
    }
}

ptrdiff_t findStringInString(const char* totalString, const char* subsequence, size_t length) {
    size_t subsequenceLength = strlen(subsequence);
    for (size_t i = 0; i < length; i++) {
        if (i + subsequenceLength > length) {
            return -1;
        }
        if (strncmp(totalString + i, subsequence, subsequenceLength) == 0) {
            return ptrdiff_t(i);
        }
    }
    return -1;
}

bool VtkXmlLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

    doc = new XMLDocument;
    uint8_t* bufferData = nullptr;
    size_t sizeInBytes = 0;
    bool loadedDat = sgl::loadFileFromSource(filePath, bufferData, sizeInBytes, true);
    if (!loadedDat) {
        sgl::Logfile::get()->writeError(
                "Error in VtkXmlLoader::load: Couldn't open file \"" + filePath + "\".");
        return false;
    }
    const char* fileBuffer = reinterpret_cast<const char*>(bufferData);

    // tinyxml2 can't parse raw data, so process this file section manually if necessary.
    const char* appendedRawStartString = "<AppendedData encoding=\"raw\">";
    ptrdiff_t appendedRawStartStringPos = findStringInString(fileBuffer, appendedRawStartString, sizeInBytes);
    if (appendedRawStartStringPos >= 0) {
        const char* appendedRawEndString = "</AppendedData>";
        ptrdiff_t appendedRawEndStringPos = findStringInString(fileBuffer, appendedRawEndString, sizeInBytes);
        auto offsetStart = appendedRawStartStringPos + ptrdiff_t(strlen(appendedRawStartString));
        //auto offsetEnd = appendedRawEndStringPos + ptrdiff_t(strlen(appendedRawEndString));
        std::string fileBufferClean =
                std::string(fileBuffer, fileBuffer + offsetStart)
                + std::string(fileBuffer + appendedRawEndStringPos, fileBuffer + sizeInBytes);
        while (offsetStart < appendedRawEndStringPos) {
            char c = fileBuffer[offsetStart];
            if (c == '_' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                offsetStart++;
                if (c == '_') {
                    break;
                }
            } else {
                break;
            }
        }
        rawDataSize = appendedRawEndStringPos - offsetStart;
        rawData = new uint8_t[rawDataSize];
        memcpy(rawData, bufferData + offsetStart, rawDataSize);
        if (doc->Parse(fileBufferClean.c_str(), fileBufferClean.size()) != 0) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Couldn't parse file \"" + filePath + "\".");
        }
    } else {
        if (doc->Parse(fileBuffer, sizeInBytes) != 0) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Couldn't parse file \"" + filePath + "\".");
        }
    }
    delete[] bufferData;

    /*if (doc->LoadFile(dataSourceFilename.c_str()) != 0) {
        sgl::Logfile::get()->writeError(
                std::string() + "VtkXmlLoader::load: Couldn't open file \"" + dataSourceFilename + "\"!");
        return false;
    }*/

    XMLElement* vtkFileNode = doc->FirstChildElement("VTKFile");
    const char* typeString = vtkFileNode->Attribute("type");
    if (typeString == nullptr || (strcmp(typeString, "ImageData") != 0 && strcmp(typeString, "RectilinearGrid") != 0)) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Invalid VTKFile type string in file \""
                + dataSourceFilename + "\".");
    }

    isLittleEndian = false;
    const char* byteOrderString = vtkFileNode->Attribute("byte_order");
    if (byteOrderString != nullptr) {
        if (strcmp(byteOrderString, "LittleEndian") == 0) {
            isLittleEndian = true;
        } else if (strcmp(byteOrderString, "BigEndian") == 0) {
            isLittleEndian = false;
        } else {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Invalid byte order string in file \""
                    + dataSourceFilename + "\".");
        }
    }

    const char* headerTypeString = vtkFileNode->Attribute("header_type");
    if (headerTypeString != nullptr) {
        if (strcmp(headerTypeString, "UInt32") == 0) {
            numHeaderBytes = 4;
        } else if (strcmp(headerTypeString, "UInt64") == 0) {
            numHeaderBytes = 8;
        } else {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Unsupported VTKFile header type in file \""
                    + dataSourceFilename + "\".");
        }
    }

    useZlib = false;
    const char* compressorString = vtkFileNode->Attribute("compressor");
    if (compressorString != nullptr) {
        if (strcmp(compressorString, "vtkZLibDataCompressor") == 0) {
            useZlib = true;
        } else {
            sgl::Logfile::get()->throwError(
                    std::string() + "Error in VtkXmlLoader::load: Invalid compressor \""
                    + compressorString + "\" in file \"" + dataSourceFilename + "\".");
        }
    }


    XMLElement* imageDataNode = vtkFileNode->FirstChildElement("ImageData");
    if (!imageDataNode) {
        imageDataNode = vtkFileNode->FirstChildElement("RectilinearGrid");
    }
    if (!imageDataNode) {
        sgl::Logfile::get()->throwError(
                std::string() + "Error in VtkXmlLoader::load: No ImageData or RectilinearGrid node found in file \""
                + dataSourceFilename + "\".");
    }

    const char* wholeExtentString = imageDataNode->Attribute("WholeExtent");
    if (wholeExtentString == nullptr) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Missing WholeExtent for ImageData in file \""
                + dataSourceFilename + "\".");
    }
    std::vector<std::string> wholeExtentStringArray;
    sgl::splitStringWhitespace(wholeExtentString, wholeExtentStringArray);
    if (wholeExtentStringArray.size() != 6) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: ImageData WholeExtent attribute in file \""
                + dataSourceFilename + "\" does not have 6 entries as expected.");
    }
    std::vector<int> wholeExtentArray;
    for (auto& extentString : wholeExtentStringArray) {
        wholeExtentArray.push_back(sgl::fromString<int>(extentString));
    }

    XMLElement* pieceNode = imageDataNode->FirstChildElement("Piece");
    if (pieceNode == nullptr) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: No Piece node given in file \""
                + dataSourceFilename + "\".");
    }

    std::vector<float> spacingArray;
    const char* spacingString = imageDataNode->Attribute("Spacing");
    XMLElement* coordinatesNode = pieceNode->FirstChildElement("Coordinates");
    if (spacingString != nullptr) {
        std::vector<std::string> spacingStringArray;
        sgl::splitStringWhitespace(spacingString, spacingStringArray);
        if (spacingStringArray.size() != 3) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: ImageData Spacing attribute in file \""
                    + dataSourceFilename + "\" does not have 3 entries as expected.");
        }
        for (auto& spacingStr : spacingStringArray) {
            spacingArray.push_back(sgl::fromString<float>(spacingStr));
        }
        xs = wholeExtentArray.at(1) - wholeExtentArray.at(0) + 1;
        ys = wholeExtentArray.at(3) - wholeExtentArray.at(2) + 1;
        zs = wholeExtentArray.at(5) - wholeExtentArray.at(4) + 1;
    } else if (coordinatesNode != nullptr) {
        // Is of the form: <DataArray Name="x_coordinates" NumberOfComponents="1" type="Float32" format="appended" offset="16392"/>
        spacingArray.push_back(1.0f);
        spacingArray.push_back(1.0f);
        spacingArray.push_back(1.0f);
        int pxs = wholeExtentArray.at(1) - wholeExtentArray.at(0) + 1;
        int pys = wholeExtentArray.at(3) - wholeExtentArray.at(2) + 1;
        int pzs = wholeExtentArray.at(5) - wholeExtentArray.at(4) + 1;
        if (pxs != pys || pys != pzs) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Mismatch in number of coordinates in x, y and z direction in file \""
                    + dataSourceFilename + "\".");
        }
        // TODO: Search the point coordinates and find out the directions.
        int dim = int(std::cbrt(pxs));
        xs = dim;
        ys = dim;
        zs = dim;
        numPoints = dim * dim * dim;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Missing Spacing for ImageData or Coordinates node in file \""
                + dataSourceFilename + "\".");
    }
    numPoints = xs * ys * zs;

    const char* pieceExtentString = pieceNode->Attribute("Extent");
    if (pieceExtentString == nullptr) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Piece without Extent data detected in file \""
                + dataSourceFilename + "\".");
    }
    if (strcmp(wholeExtentString, pieceExtentString) != 0) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Pieces with a different extent than the image data detected in \""
                + dataSourceFilename + "\". This is not yet supported.");
    }

    XMLElement* pointDataNode = pieceNode->FirstChildElement("PointData");
    XMLElement* cellDataNode = pieceNode->FirstChildElement("CellData");
    XMLElement* dataNode = pointDataNode ? pointDataNode : cellDataNode;
    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;

    if (!dataNode) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: No point or cell data specified in file \""
                + dataSourceFilename + "\".");
    }

    for (sgl::XMLIterator it(dataNode, sgl::XMLNameFilter("DataArray")); it.isValid(); ++it) {
        XMLElement* dataArrayNode = *it;
        const char* dataArrayTypeString = dataArrayNode->Attribute("type");
        if (dataArrayTypeString == nullptr
                || (strcmp(dataArrayTypeString, "Float32") != 0 && strcmp(dataArrayTypeString, "Float64") != 0)) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Encountered data array with type not equal to Float32 in file \""
                    + dataSourceFilename + "\". Currently, only Float32 is supported.");
        }
        uint32_t dataSize = 4;
        if (strcmp(dataArrayTypeString, "Float64") == 0) {
            dataSize = 8;
        }

        const char* dataArrayFormatString = dataArrayNode->Attribute("format");
        if (dataArrayFormatString == nullptr || strcmp(dataArrayFormatString, "appended") != 0) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Encountered data array with format not equal to appended in file \""
                    + dataSourceFilename + "\". Currently, only appended data is supported.");
        }

        const char* dataArrayOffsetString = dataArrayNode->Attribute("offset");
        if (dataArrayOffsetString == nullptr) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Missing offset for data array in file \""
                    + dataSourceFilename + "\".");
        }
        auto offset = sgl::fromString<size_t>(dataArrayOffsetString);

        const char* dataArrayNameString = dataArrayNode->Attribute("Name");
        if (dataArrayNameString == nullptr) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Expected name for data array in file \""
                    + dataSourceFilename + "\".");
        }
        variableOffsets.emplace(dataArrayNameString, offset);
        variableDataSize.emplace(dataArrayNameString, dataSize);
        fieldNameMap[FieldType::SCALAR].emplace_back(dataArrayNameString);
    }

    XMLElement* appendedDataNode = vtkFileNode->FirstChildElement("AppendedData");
    const char* encodingString = appendedDataNode->Attribute("encoding");
    if (encodingString == nullptr || (strcmp(encodingString, "base64") != 0 && strcmp(encodingString, "raw") != 0)) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Unsupported appended data encoding in file \""
                + dataSourceFilename + "\".");
    }

    if (strcmp(encodingString, "base64") == 0) {
        appendedDataEncoded = appendedDataNode->GetText();

        size_t totalStringLength = strlen(appendedDataEncoded);
        startPos = 0;
        ptrdiff_t endPos = ptrdiff_t(totalStringLength) - 1;
        while (startPos < ptrdiff_t(totalStringLength)) {
            char c = appendedDataEncoded[startPos];
            if (c == '_' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                startPos++;
                if (c == '_') {
                    break;
                }
            } else {
                break;
            }
        }
        while (endPos >= 0) {
            char c = appendedDataEncoded[endPos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                endPos--;
            } else {
                break;
            }
        }
    }

    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    float maxSpacing = std::max(spacingArray.at(0), std::max(spacingArray.at(1), spacingArray.at(2)));
    dx = cellStep * spacingArray.at(0) / maxSpacing;
    dy = cellStep * spacingArray.at(1) / maxSpacing;
    dz = cellStep * spacingArray.at(2) / maxSpacing;
    volumeData->setGridExtent(xs, ys, zs, dx, dy, dz);

    volumeData->setFieldNames(fieldNameMap);

    return true;
}

bool VtkXmlLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    auto* fieldEntryBuffer = new float[numPoints];

    auto it = variableOffsets.find(fieldName);
    uint32_t dataSize = variableDataSize.find(fieldName)->second;
    if (rawData) {
        const uint8_t* dataArray = rawData + it->second;
        size_t numBytes = 0;
        if (numHeaderBytes == 4) {
            numBytes = *reinterpret_cast<const uint32_t*>(dataArray);
            dataArray += sizeof(uint32_t);
        } else if (numHeaderBytes == 8) {
            numBytes = *reinterpret_cast<const uint64_t*>(dataArray);
            dataArray += sizeof(uint64_t);
        } else {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: raw encoding as used in file \""
                    + dataSourceFilename + "\" is currently only supported for UInt32 and UInt64 headers.");
        }
        if (size_t(numPoints) != numBytes / size_t(dataSize)) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Data size mismatch in file \""
                    + dataSourceFilename + "\".");
        }
        if (dataSize == 4) {
            memcpy(fieldEntryBuffer, dataArray, numBytes);
        } else if (dataSize == 8) {
            const auto* dataDouble = reinterpret_cast<const double*>(dataArray);
            for (int i = 0; i < numPoints; i++) {
                fieldEntryBuffer[i] = float(dataDouble[i]);
            }
        }
    } else if (useZlib) {
        if (numHeaderBytes != 4 || dataSize != 4) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: base64 encoding as used in file \""
                    + dataSourceFilename + "\" is currently only supported for 32-bit headers and data.");
        }

        const char* headerBase64String = appendedDataEncoded + startPos + it->second;

        uint32_t numBlocks = base64DecodeUint32(headerBase64String);
        size_t headerSizeBytes = sizeof(uint32_t) * size_t(numBlocks + 3);
        size_t headerSizeBase64 = (headerSizeBytes + 2) / 3 * 4;
        auto* encodedDataHeader = new char[base64GetNumBytesDecoded(int(headerSizeBase64))];
        size_t decodedHeaderDataSizeInBytes = base64DecodeSized(
                encodedDataHeader, headerBase64String, int(headerSizeBase64));
        (void)decodedHeaderDataSizeInBytes;

        auto* header = reinterpret_cast<uint32_t*>(encodedDataHeader);
        //uint32_t numBlocks = header[0];
        uint32_t uncompressedBlockSize = header[1];
        uint32_t uncompressedLastBlockPartialSize = header[2];

        size_t bufferSize;
        if (uncompressedLastBlockPartialSize == 0) {
            bufferSize = size_t(numBlocks) * size_t(uncompressedBlockSize);
        } else {
            bufferSize =
                    size_t(numBlocks - 1) * size_t(uncompressedBlockSize) +
                    size_t(uncompressedLastBlockPartialSize);
        }

        if (bufferSize != sizeof(float) * numPoints) {
            sgl::Logfile::get()->throwError(
                    "Error in VtkXmlLoader::load: Invalid uncompressed buffer size in file \""
                    + dataSourceFilename + "\".");
        }

        size_t compressedTotalSize = 0;
        for (uint32_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
            auto compressedSize = size_t(header[blockIdx + 3]);
            compressedTotalSize += compressedSize;
        }
        const char* dataBase64String = headerBase64String + headerSizeBase64;
        size_t dataSizeBase64 = (compressedTotalSize + 2) / 3 * 4;
        auto* encodedData = new char[base64GetNumBytesDecoded(int(dataSizeBase64))];
        size_t decodedDataSizeInBytes = base64DecodeSized(
                encodedData, dataBase64String, int(dataSizeBase64));
        (void)decodedDataSizeInBytes;

        auto* compressedDataReadPtr = reinterpret_cast<uint8_t*>(encodedData);
        auto* uncompressedDataWritePtr = reinterpret_cast<uint8_t*>(fieldEntryBuffer);
        for (uint32_t blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
            auto compressedSize = size_t(header[blockIdx + 3]);
            size_t uncompressedSize;
            if (blockIdx == numBlocks - 1 && uncompressedLastBlockPartialSize != 0) {
                uncompressedSize = uncompressedLastBlockPartialSize;
            } else {
                uncompressedSize = uncompressedBlockSize;
            }
            sgl::decompressZlibData(
                    compressedDataReadPtr, compressedSize,
                    uncompressedDataWritePtr, uncompressedSize);
            compressedDataReadPtr += compressedSize;
            uncompressedDataWritePtr += uncompressedSize;
        }

        delete[] encodedData;
        delete[] encodedDataHeader;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Uncompressed appended data as used in file \""
                + dataSourceFilename + "\" is currently not yet supported.");
    }

    if (!isLittleEndian) {
        sgl::Logfile::get()->throwError(
                "Error in VtkXmlLoader::load: Big endian encoding used in file \""
                + dataSourceFilename + "\" is not yet supported.");
    }
    fieldEntry = new HostCacheEntryType(xs * ys * zs, fieldEntryBuffer);

    return true;
}
