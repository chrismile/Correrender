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

#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/Convert.hpp>

#include "Volume/VolumeData.hpp"
#include "DataSet.hpp"
#include "LoadersUtil.hpp"
#include "AmiraMeshLoader.hpp"

char* AmiraMeshLoader::skipLine(char* fileBuffer, char* fileBufferEnd) {
    char c;
    while (true) {
        if (fileBuffer == fileBufferEnd) {
            sgl::Logfile::get()->throwError(
                    "Error in AmiraMeshLoader::skipLine: Reached end of string in file  \""
                    + dataSourceFilename + "\".");
        }
        c = *fileBuffer;
        if (c == '\r' || c == '\n') {
            do {
                fileBuffer++;
                c = *fileBuffer;
            } while ((c == '\r' || c == '\n') && fileBuffer != fileBufferEnd);
            break;
        }
        fileBuffer++;
    }
    return fileBuffer;
}

AmiraMeshLoader::~AmiraMeshLoader() {
    if (buffer) {
        delete[] buffer;
        buffer = nullptr;
    }
}

bool AmiraMeshLoader::setInputFiles(
        VolumeData* volumeData, const std::string& filePath, const DataSetInformation& _dataSetInformation) {
    dataSourceFilename = filePath;
    dataSetInformation = _dataSetInformation;

    buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(dataSourceFilename, buffer, length, false);
    if (!loaded) {
        sgl::Logfile::get()->writeError(
                "Error in AmiraMeshLoader::setInputFiles: Couldn't open file \"" + dataSourceFilename + "\".");
        return false;
    }
    char* fileBuffer = reinterpret_cast<char*>(buffer);

    if (!strstr(fileBuffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1")) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Missing AmiraMesh header in file \""
                + dataSourceFilename + "\".");
    }

    char* defineLatticeLine = strstr(fileBuffer, "define Lattice ");
    defineLatticeLine += strlen("define Lattice ");
    std::vector<int> latticeDimensions;
    std::string stringBuffer;
    while (true) {
        char c = *defineLatticeLine;
        if (c == ' ' || c == '\t') {
            if (stringBuffer.length() > 0) {
                latticeDimensions.push_back(sgl::fromString<int>(stringBuffer));
                stringBuffer = "";
            }
        } else if (c == '\r' || c == '\n') {
            break;
        } else {
            stringBuffer += c;
        }
        defineLatticeLine++;
    }
    if (stringBuffer.length() > 0) {
        latticeDimensions.push_back(sgl::fromString<int>(stringBuffer));
        stringBuffer = "";
    }
    if (latticeDimensions.size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Lattice definition in file \""
                + dataSourceFilename + "\" does not contain 3 entries.");
    }
    xs = latticeDimensions.at(0);
    ys = latticeDimensions.at(1);
    zs = latticeDimensions.at(2);
    numPoints = xs * ys * zs;

    char* parametersLine = strstr(fileBuffer, "Parameters {");
    if (!parametersLine) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Lattice definition in file \""
                + dataSourceFilename + "\" does not contain 3 entries.");
    }
    char* boundingBoxLine = strstr(parametersLine, "BoundingBox ");
    char* boundingBoxLineStart = boundingBoxLine + strlen("BoundingBox ");
    char* boundingBoxLineStop = strstr(boundingBoxLineStart, ",");
    if (!boundingBoxLineStop) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Misformed BoundingBox statement in file \""
                + dataSourceFilename + "\". misses a comma");
    }
    std::string boundingBoxString(boundingBoxLineStart, boundingBoxLineStop);
    std::vector<std::string> boundingBoxStringList;
    sgl::splitStringWhitespace(boundingBoxString, boundingBoxStringList);
    if (boundingBoxStringList.size() != 6) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Misformed BoundingBox statement in file \""
                + dataSourceFilename + "\" does not contain 6 entries.");
    }
    auto xmin = sgl::fromString<float>(boundingBoxStringList.at(0));
    auto xmax = sgl::fromString<float>(boundingBoxStringList.at(1));
    auto ymin = sgl::fromString<float>(boundingBoxStringList.at(2));
    auto ymax = sgl::fromString<float>(boundingBoxStringList.at(3));
    auto zmin = sgl::fromString<float>(boundingBoxStringList.at(4));
    auto zmax = sgl::fromString<float>(boundingBoxStringList.at(5));
    float bbDimX = xmax - xmin;
    float bbDimY = ymax - ymin;
    float bbDimZ = zmax - zmin;

    char* coordTypeLine = strstr(boundingBoxLine, "CoordType \"");
    char* coordTypeLineStart = coordTypeLine + strlen("CoordType \"");
    char* coordTypeLineEnd = strstr(coordTypeLineStart, "\"");
    std::string coordTypeString(coordTypeLineStart, coordTypeLineEnd);
    if (coordTypeString != "uniform") {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Unsupported coordinate type in file \""
                + dataSourceFilename + "\".");
    }

    char* latticeLine = strstr(fileBuffer, "Lattice { float[");
    if (!latticeLine) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Missing 3D lattice definition in file \""
                + dataSourceFilename + "\".");
    }
    char* latticeLineArrayStart = latticeLine + strlen("Lattice { float[");
    char* latticeLineArrayEnd = strstr(latticeLineArrayStart, "] Data");
    if (!latticeLineArrayEnd) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Misformed 3D lattice definition in file \""
                + dataSourceFilename + "\".");
    }
    std::string numDimensionsString(latticeLineArrayStart, latticeLineArrayEnd);
    if (sgl::fromString<int>(numDimensionsString) != 3) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Lattice dimension in file \""
                + dataSourceFilename + "\" is not equal to 3.");
    }

    dataSectionStart = strstr(fileBuffer, "# Data section follows");
    if (!dataSectionStart) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Missing data section in file \""
                + dataSourceFilename + "\".");
    }
    dataSectionStart = skipLine(dataSectionStart, fileBuffer + length);
    dataSectionStart = skipLine(dataSectionStart, fileBuffer + length);

    if (sizeof(float) * 3 * numPoints > size_t(fileBuffer + length - dataSectionStart)) {
        sgl::Logfile::get()->throwError(
                "Error in AmiraMeshLoader::setInputFiles: Invalid data section size in file \""
                + dataSourceFilename + "\".");
    }

    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    float maxBbDim = std::max(bbDimX, std::max(bbDimY, bbDimZ));
    dx = cellStep * bbDimX / maxBbDim;
    dy = cellStep * bbDimY / maxBbDim;
    dz = cellStep * bbDimZ / maxBbDim;
    volumeData->setGridExtent(xs, ys, zs, dx, dy, dz);

    std::unordered_map<FieldType, std::vector<std::string>> fieldNameMap;
    fieldNameMap[FieldType::VECTOR].emplace_back("Velocity");
    fieldNameMap[FieldType::VECTOR].emplace_back("Vorticity");
    fieldNameMap[FieldType::SCALAR].emplace_back("Velocity Magnitude");
    fieldNameMap[FieldType::SCALAR].emplace_back("Vorticity Magnitude");
    fieldNameMap[FieldType::SCALAR].emplace_back("Helicity");
    volumeData->setFieldNames(fieldNameMap);

    return true;
}

bool AmiraMeshLoader::getFieldEntry(
        VolumeData* volumeData, FieldType fieldType, const std::string& fieldName,
        int timestepIdx, int memberIdx, HostCacheEntryType*& fieldEntry) {
    auto* velocityField = new float[3 * numPoints];
    memcpy(velocityField, dataSectionStart, sizeof(float) * 3 * numPoints);

    auto* velocityMagnitudeField = new float[numPoints];
    auto* vorticityField = new float[numPoints * 3];
    auto* vorticityMagnitudeField = new float[numPoints];
    auto* helicityField = new float[numPoints];

    computeVectorMagnitudeField(velocityField, velocityMagnitudeField, xs, ys, zs);
    computeVorticityField(velocityField, vorticityField, xs, ys, zs, dx, dy, dz);
    computeVectorMagnitudeField(vorticityField, vorticityMagnitudeField, xs, ys, zs);
    computeHelicityFieldNormalized(
            velocityField, vorticityField, helicityField, xs, ys, zs,
            dataSetInformation.useNormalizedVelocity,
            dataSetInformation.useNormalizedVorticity);

    volumeData->addField(velocityField, FieldType::VECTOR, "Velocity", timestepIdx, memberIdx);
    volumeData->addField(vorticityField, FieldType::VECTOR, "Vorticity", timestepIdx, memberIdx);
    volumeData->addField(helicityField, FieldType::SCALAR, "Helicity", timestepIdx, memberIdx);
    volumeData->addField(velocityMagnitudeField, FieldType::SCALAR, "Velocity Magnitude", timestepIdx, memberIdx);
    volumeData->addField(vorticityMagnitudeField, FieldType::SCALAR, "Vorticity Magnitude", timestepIdx, memberIdx);

    return true;
}
