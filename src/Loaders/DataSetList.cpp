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
#include <boost/algorithm/string/case_conv.hpp>
#include <json/json.h>

#include <Utils/AppSettings.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/Regex/TransformString.hpp>

#include "DataSetList.hpp"

bool jsonValueToBool(const Json::Value& value) {
    if (value.isString()) {
        std::string valueString = value.asString();
        if (valueString == "true") {
            return true;
        } else if (valueString == "false") {
            return false;
        } else {
            sgl::Logfile::get()->throwError("Error in jsonValueToBool: Invalid value \"" + valueString + "\".");
            return false;
        }
    } else {
        return value.asBool();
    }
}

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

        if (source.isMember("ensemble_range") || source.isMember("time_range")) {
            std::string rangeString;
            bool isTimeStepRange = false;
            if (source.isMember("ensemble_range")) {
                rangeString = source["ensemble_range"].asString();
            } else if (source.isMember("time_range")) {
                rangeString = source["time_range"].asString();
                isTimeStepRange = true;
            }
            dataSetInformation->useTimeStepRange = true;

            std::vector<std::string> rangeVector;
            sgl::splitStringWhitespace(rangeString, rangeVector);
            if (rangeVector.size() != 2 && rangeVector.size() != 3) {
                sgl::Logfile::get()->throwError("Error in processDataSetNodeChildren: Invalid range statement.");
            }
            if (source.isMember("ensemble_range") && source.isMember("time_range")) {
                sgl::Logfile::get()->throwError(
                        "Error in processDataSetNodeChildren: An ensemble and time range cannot be provided at the "
                        "same time.");
            }
            bool isRangeExclusive = true;
            if (source.isMember("range_exclusive")) {
                isRangeExclusive = jsonValueToBool(source["range_exclusive"]);
            } else if (source.isMember("range_inclusive")) {
                isRangeExclusive = !jsonValueToBool(source["range_inclusive"]);
            }

            int start = int(sgl::fromString<int>(rangeVector.at(0)));
            int stop = int(sgl::fromString<int>(rangeVector.at(1)));
            int step = rangeVector.size() == 3 ? int(sgl::fromString<int>(rangeVector.at(2))) : 1;
            auto filenamesPattern = dataSetInformation->filenames;
            dataSetInformation->filenames.clear();
            for (const std::string& filenamePattern : filenamesPattern) {
                size_t bufferSize = filenamePattern.size() + 100;
                char* rawFilePathBuffer = new char[bufferSize];
                for (int idx = start; idx < (isRangeExclusive ? stop : stop + 1); idx += step) {
                    snprintf(rawFilePathBuffer, bufferSize, filenamePattern.c_str(), idx);
                    dataSetInformation->filenames.emplace_back(rawFilePathBuffer);
                    if (isTimeStepRange) {
                        dataSetInformation->timeSteps.push_back(idx);
                    }
                }
                delete[] rawFilePathBuffer;
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

        // Optional data: Use one separate file per attribute?
        if (source.isMember("separate_files_per_attribute")) {
            dataSetInformation->separateFilesPerAttribute = jsonValueToBool(source["separate_files_per_attribute"]);
        }

        // Optional data: The scaling in y direction.
        if (source.isMember("heightscale")) {
            dataSetInformation->heightScale = source["heightscale"].asFloat();
        }

        // Optional data: Cast scalar data to a different format?
        if (source.isMember("format_cast")) {
            dataSetInformation->useFormatCast = true;
            std::string formatString = boost::to_lower_copy(source["format_cast"].asString());
            if (formatString == "float") {
                dataSetInformation->formatTarget = ScalarDataFormat::FLOAT;
            } else if (formatString == "short" || formatString == "ushort") {
                dataSetInformation->formatTarget = ScalarDataFormat::SHORT;
            } else if (formatString == "byte" || formatString == "ubyte") {
                dataSetInformation->formatTarget = ScalarDataFormat::BYTE;
            } else if (formatString == "float16" || formatString == "half") {
                dataSetInformation->formatTarget = ScalarDataFormat::FLOAT16;
            } else {
                sgl::Logfile::get()->throwError(
                        "Error in processDataSetNodeChildren: No match for provided format string.");
            }
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
            dataSetInformation->hasCustomDateTime = true;
        }
        if (source.isMember("data_date") && source.isMember("data_time")) {
            auto dataDateElement = source["data_date"];
            auto dataTimeElement = source["data_time"];
            dataSetInformation->date = dataDateElement.asInt();
            dataSetInformation->time = dataTimeElement.asInt();
            dataSetInformation->hasCustomDateTime = true;
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
            dataSetInformation->subsamplingFactorSet = true;
        }
        if (source.isMember("velocity_field_name")) {
            dataSetInformation->velocityFieldName = source["velocity_field_name"].asString();
        }

        if (source.isMember("axes")) {
            auto axesElement = source["axes"];
            int dim = 0;
            for (const auto& axisElement : axesElement) {
                dataSetInformation->axes[dim] = axisElement.asInt();
                dim++;
            }
        }

        if (source.isMember("standard_scalar_field")) {
            auto standardScalarFieldElement = source["standard_scalar_field"];
            if (standardScalarFieldElement.isString()) {
                dataSetInformation->standardScalarFieldName = standardScalarFieldElement.asString();
            } else if (standardScalarFieldElement.isInt()) {
                dataSetInformation->standardScalarFieldIdx = standardScalarFieldElement.asInt();
            } else {
                sgl::Logfile::get()->throwError(
                        "Error in processDataSetNodeChildren: Invalid data type for 'standard_scalar_field'.");
            }
        }

        if (source.isMember("standard_time_step")) {
            auto standardTimeStepElement = source["standard_time_step"];
            if (standardTimeStepElement.isInt()) {
                dataSetInformation->standardTimeStepIdx = standardTimeStepElement.asInt();
            } else {
                sgl::Logfile::get()->throwError(
                        "Error in processDataSetNodeChildren: Invalid data type for 'standard_time_step'.");
            }
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
