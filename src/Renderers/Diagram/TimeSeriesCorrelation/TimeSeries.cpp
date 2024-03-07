/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#include <cassert>
#include <utility>

#include <Utils/File/Logfile.hpp>
#include <Utils/Parallel/Reduction.hpp>

#include "TimeSeries.hpp"

TimeSeriesData::TimeSeriesData(TimeSeriesMetadata metadata, float* data) : metadata(std::move(metadata)), data(data) {
}

TimeSeriesData::~TimeSeriesData() {
    if (data) {
        delete[] data;
        data = nullptr;
    }
}

float* TimeSeriesData::getWindowData(int sidx, int widx, int wlen) {
    if (metadata.window <= 0) {
        // Data is not yet subdivided into windows.
        assert(widx + wlen <= metadata.time);
        return &data[sidx * metadata.time + widx];
    } else if (metadata.window == wlen) {
        // Data is already subdivided into windows.
        assert(widx < metadata.time);
        return &data[(sidx * metadata.time + widx) * metadata.window];
    } else {
        sgl::Logfile::get()->throwError("Error in TimeSeriesData::getWindowData: Window length mismatch.");
        return nullptr;
    }
}

void TimeSeriesData::computeMinMax() {
    int numValues = metadata.samples * metadata.time;
    if (metadata.window > 0) {
        numValues *= metadata.window;
    }
    auto minMax = sgl::reduceFloatArrayMinMax(data, numValues);
    minValue = minMax.first;
    maxValue = minMax.second;
    hasComputedMinMax = true;
}

float TimeSeriesData::getMinValue() {
    if (!hasComputedMinMax) {
        computeMinMax();
    }
    return minValue;
}

float TimeSeriesData::getMaxValue() {
    if (!hasComputedMinMax) {
        computeMinMax();
    }
    return minValue;
}
