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

#include <boost/algorithm/string/predicate.hpp>
#include <json/json.h>

#include <Utils/AppSettings.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/Regex/TransformString.hpp>

#include "DataSetList.hpp"

void processDataSetNodeChildren(Json::Value& childList, DataSetInformation* dataSetInformationParent) {
    for (Json::Value& source : childList) {
        auto* dataSetInformation = new DataSetInformation;

        // Get the type information.
        std::string typeName = source.isMember("type") ? source["type"].asString() : "volume";
        if (typeName == "node") {
            dataSetInformation->type = DataSetType::NODE;
        } else if (typeName == "volume") {
            dataSetInformation->type = DataSetType::VOLUME;
        } else {
            sgl::Logfile::get()->writeError(
                    "Error in processDataSetNodeChildren: Invalid type name \"" + typeName + "\".");
            return;
        }

        dataSetInformation->name = source["name"].asString();

        if (dataSetInformation->type == DataSetType::NODE) {
            dataSetInformationParent->children.emplace_back(dataSetInformation);
            processDataSetNodeChildren(source["children"], dataSetInformation);
            continue;
        }

        Json::Value filenames;
        if (source.isMember("filenames")) {
            filenames = source["filenames"];
        } else if (source.isMember("filename")) {
            filenames = source["filename"];
        }
        const std::string volumeDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "VolumeDataSets/";
        if (filenames.isArray()) {
            for (const auto& filename : filenames) {
                std::string pathString = filename.asString();
#ifdef _WIN32
                bool isAbsolutePath =
                    (pathString.size() > 1 && pathString.at(1) == ':')
                    || boost::starts_with(pathString, "/") || boost::starts_with(pathString, "\\");
#else
                bool isAbsolutePath = boost::starts_with(pathString, "/");
#endif
                if (isAbsolutePath) {
                    dataSetInformation->filenames.push_back(pathString);
                } else {
                    dataSetInformation->filenames.push_back(volumeDataSetsDirectory + pathString);
                }
            }
        } else {
            std::string pathString = filenames.asString();
#ifdef _WIN32
            bool isAbsolutePath =
                    (pathString.size() > 1 && pathString.at(1) == ':')
                    || boost::starts_with(pathString, "/") || boost::starts_with(pathString, "\\");
#else
            bool isAbsolutePath = boost::starts_with(pathString, "/");
#endif
            if (isAbsolutePath) {
                dataSetInformation->filenames.push_back(pathString);
            } else {
                dataSetInformation->filenames.push_back(volumeDataSetsDirectory + pathString);
            }
        }

        // Optional data: Transform.
        dataSetInformation->hasCustomTransform = source.isMember("transform");
        if (dataSetInformation->hasCustomTransform) {
            glm::mat4 transformMatrix = parseTransformString(source["transform"].asString());
            dataSetInformation->transformMatrix = transformMatrix;
        }

        // Optional data: Attribute (importance criteria) display names.
        if (source.isMember("attributes")) {
            Json::Value attributes = source["attributes"];
            if (attributes.isArray()) {
                for (Json::Value::const_iterator attributesIt = attributes.begin();
                     attributesIt != attributes.end(); ++attributesIt) {
                    dataSetInformation->attributeNames.push_back(attributesIt->asString());
                }
            } else {
                dataSetInformation->attributeNames.push_back(attributes.asString());
            }
        }

        // Optional data: The scaling in y direction.
        if (source.isMember("heightscale")) {
            dataSetInformation->heightScale = source["heightscale"].asFloat();
        }

        if (source.isMember("time")) {
            auto timeElement = source["time"];
            if (timeElement.isString()) {
                std::string timeString = timeElement.asString();
                std::vector<std::string> timeStringVector;
                sgl::splitStringWhitespace(timeString, timeStringVector);
                if (timeStringVector.size() == 1) {
                    dataSetInformation->time = sgl::fromString<int>(timeStringVector.at(0));
                } else if (timeStringVector.size() == 2) {
                    std::string dateStringRaw;
                    for (char c : timeStringVector.at(0)) {
                        if (int(c) >= int('0') && int(c) <= '9') {
                            dateStringRaw += c;
                        }
                    }
                    std::string timeStringRaw;
                    for (char c : timeStringVector.at(1)) {
                        if (int(c) >= int('0') && int(c) <= '9') {
                            timeStringRaw += c;
                        }
                    }
                    dataSetInformation->date = sgl::fromString<int>(dateStringRaw);
                    dataSetInformation->time = sgl::fromString<int>(timeStringRaw);
                }
            } else {
                dataSetInformation->time = timeElement.asInt();
            }
        }
        if (source.isMember("data_date") && source.isMember("data_time")) {
            auto dataDateElement = source["data_date"];
            auto dataTimeElement = source["data_time"];
            dataSetInformation->date = dataDateElement.asInt();
            dataSetInformation->time = dataTimeElement.asInt();
        }
        if (source.isMember("scale")) {
            auto scaleElement = source["scale"];
            if (scaleElement.isDouble()) {
                dataSetInformation->scale = glm::vec3(scaleElement.asFloat());
            } else if (scaleElement.isArray()) {
                int dim = 0;
                for (const auto& scaleDimElement : scaleElement) {
                    dataSetInformation->scale[dim] = scaleDimElement.asFloat();
                    dim++;
                }
            }
        }
        if (source.isMember("axes")) {
            auto axesElement = source["axes"];
            int dim = 0;
            for (const auto& axisElement : axesElement) {
                dataSetInformation->axes[dim] = axisElement.asInt();
                dim++;
            }
        }
        if (source.isMember("subsampling_factor")) {
            dataSetInformation->subsamplingFactor = source["subsampling_factor"].asInt();
            dataSetInformation->susamplingFactorSet = true;
        }
        if (source.isMember("velocity_field_name")) {
            dataSetInformation->velocityFieldName = source["velocity_field_name"].asString();
        }

        dataSetInformationParent->children.emplace_back(dataSetInformation);
    }
}

DataSetInformationPtr loadDataSetList(const std::string& filename) {
    // Parse the passed JSON file.
    std::ifstream jsonFileStream(filename.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return {};
    }
    jsonFileStream.close();

    DataSetInformationPtr dataSetInformationRoot(new DataSetInformation);
    Json::Value& dataSetNode = root["datasets"];
    processDataSetNodeChildren(dataSetNode, dataSetInformationRoot.get());
    return dataSetInformationRoot;
}
