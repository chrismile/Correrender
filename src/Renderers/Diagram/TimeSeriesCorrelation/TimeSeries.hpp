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

#ifndef CORRERENDER_TIMESERIES_HPP
#define CORRERENDER_TIMESERIES_HPP

#include <vector>
#include <memory>
#include <string>

struct TimeSeriesMetadata {
    std::string filename;
    int samples = -1; ///< Number of time series.
    int time = -1; ///< Number of time steps.
    int window = -1; ///< Optional.
};

class TimeSeriesData {
public:
    TimeSeriesData(TimeSeriesMetadata metadata, float* data);
    ~TimeSeriesData();
    float* getWindowData(int sidx, int widx, int wlen);
    float getMinValue();
    float getMaxValue();

private:
    TimeSeriesMetadata metadata;
    float* data;

    void computeMinMax();
    bool hasComputedMinMax = false;
    float minValue = 0.0f, maxValue = 0.0f;
};
typedef std::shared_ptr<TimeSeriesData> TimeSeriesDataPtr;

#endif //CORRERENDER_TIMESERIES_HPP
