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

#ifndef CORRERENDER_DATASETLIST_HPP
#define CORRERENDER_DATASETLIST_HPP

#include <vector>
#include <string>
#include <memory>

#include <Math/Geometry/MatrixUtil.hpp>
#include <Utils/SciVis/ScalarDataFormat.hpp>

enum class DataSetType {
    NONE,
    NODE, //< Hierarchical container.
    VOLUME, //< Volume data set.
};

enum class DataSetFileDimension {
    UNSPECIFIED,
    TIME,
    ENSEMBLE,
    TIME_ENSEMBLE,
    ENSEMBLE_TIME
};

struct DataSetInformation;
typedef std::shared_ptr<DataSetInformation> DataSetInformationPtr;

struct DataSetInformation {
    DataSetType type = DataSetType::NODE;
    std::string name;
    std::vector<std::string> filenames;
    std::vector<int> timeSteps;
    bool useTimeStepRange = true;

    // For type DATA_SET_TYPE_NODE.
    std::vector<DataSetInformationPtr> children;
    int sequentialIndex = 0;

    // Optional attributes.
    bool hasCustomTransform = false;
    glm::mat4 transformMatrix = sgl::matrixIdentity();
    std::vector<std::string> attributeNames; ///< Names of the associated attributes.
    bool separateFilesPerAttribute = false;
    float heightScale = 1.0f;
    bool useFormatCast = false;
    ScalarDataFormat formatTarget = ScalarDataFormat::FLOAT;
    /// Metadata reuse for individual time step or ensemble member files can potentially speed up loading.
    bool reuseMetadata = true;

    // Date can be left 0. It is used for GRIB files storing time in a date-time format.
    // E.g., "data_date": 20161002, "data_time": 600 can be used for 2016-10-02 6:00.
    // Alternatively, "time": "2016-10-02 6:00" can be used. Time steps are specified via, e.g., "time": 10.
    bool hasCustomDateTime = false;
    int date = 0;
    int time = 0;
    // Scale along each input axis.
    glm::vec3 scale = { 1.0f, 1.0f, 1.0f };
    // Can be used for transposing axes.
    glm::ivec3 axes = { 0, 1, 2 };
    // Optional downscaling for the flow field.
    int subsamplingFactor = 1;
    bool subsamplingFactorSet = false;
    // Optionally the user can restrict the domain (0,0,0) to (xs-1, ys-1, zs-1) to a smaller selection.
    bool useDomainSubselection = false;
    glm::ivec3 domainSubselectionMin{}, domainSubselectionMax{};
    // Name of the velocity field to use (if multiple are available).
    std::string velocityFieldName;
    // Whether to use normalized velocity or normalized vorticity in helicity computation.
    bool useNormalizedVelocity = false;
    bool useNormalizedVorticity = false;

    // Standard selection for time and attribute in UI.
    std::string standardScalarFieldName;
    int standardScalarFieldIdx = 0;
    int standardTimeStepIdx = 0;

    /*inline bool operator==(const DataSetInformation& rhs) const {
        return
                this->date == rhs.date && this->time == rhs.time && this->scale == rhs.scale && this->axes == rhs.axes
                && this->velocityFieldName == rhs.velocityFieldName
                && this->useNormalizedVelocity == rhs.useNormalizedVelocity
                && this->useNormalizedVorticity == rhs.useNormalizedVorticity;
    }*/
};

DataSetInformationPtr loadDataSetList(const std::string& filename, bool isFileWatchReload);

#endif //CORRERENDER_DATASETLIST_HPP
