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

#include <random>

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/CsvWriter.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Volume/VolumeData.hpp"
#include "Renderers/Diagram/HEBChart.hpp"
#include "SamplingTest.hpp"

struct TestCase {
    SamplingMethodType samplingMethodType = SamplingMethodType::QUASIRANDOM_PLASTIC;
    int numSamples = 100;
    double elapsedTimeMicroseconds = 0.0;
    double maximumQuantile = 0.0;
    double rangeLinear = 0.0;
};

void runTestCase(
        HEBChart* chart, TestCase& testCase,
        const std::vector<std::pair<uint32_t, uint32_t>>& blockPairs,
        const std::vector<std::vector<float>>& allValuesSortedArray) {
    chart->setSamplingMethodType(testCase.samplingMethodType);
    chart->setNumSamples(testCase.numSamples);
    auto statisticsList = chart->computeCorrelationsBlockPairs(blockPairs);

    // Binary search.
    auto invN = 1.0 / double(blockPairs.size());
    for (size_t i = 0; i < blockPairs.size(); i++) {
        float searchValue = statisticsList.maximumValues.at(i);
        const auto& allValuesSorted = allValuesSortedArray.at(i);
        size_t lower = 0;
        size_t upper = allValuesSorted.size();
        size_t middle;
        while (lower < upper) {
            middle = (lower + upper) / 2;
            float middleValue = allValuesSorted[middle];
            if (middleValue < searchValue) {
                lower = middle + 1;
            } else {
                upper = middle;
            }
        }
        testCase.maximumQuantile = invN * double(upper) / double(allValuesSorted.size());
        float minVal = allValuesSorted.front();
        float maxVal = allValuesSorted.back();
        testCase.rangeLinear += invN * double((searchValue - minVal) / (maxVal - minVal));
    }
    testCase.elapsedTimeMicroseconds = invN * statisticsList.elapsedTimeMicroseconds;
}

void runSamplingTests(const std::string& dataSetPath) {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    auto* renderer = new sgl::vk::Renderer(device, 100);

    // Load the volume data set.
    VolumeDataPtr volumeData(new VolumeData(renderer));
    DataSetInformation dataSetInformation;
    dataSetInformation.filenames = { dataSetPath };
    volumeData->setInputFiles({ "/home/neuhauser/datasets/Necker/nc/necker_t5_tk_u.nc" }, dataSetInformation, nullptr);
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();
    int ts = volumeData->getTimeStepCount();
    bool isEnsembleMode = es > 1;
    int cs = isEnsembleMode ? es : ts;
    int k = std::max(sgl::iceil(3 * cs, 100), 1);
    int numBins = 80;

    // Settings.
    constexpr CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::PEARSON;
    const auto& fieldNames = volumeData->getFieldNamesBase(FieldType::SCALAR);
    constexpr int fieldIdx = 0;
    constexpr bool useGpu = true;
    //constexpr int dfx = 16;
    //constexpr int dfy = 16;
    //constexpr int dfz = 20;
    constexpr int dfx = 8;
    constexpr int dfy = 8;
    constexpr int dfz = 8;
    int numPairsToCheck = 1000;
    int numLogSteps = 3;
    std::vector<int> numSamplesArray;
    for (int l = 0; l < numLogSteps; l++) {
        auto step = int(std::pow(10, l));
        auto maxVal = int(std::pow(10, l + 1));
        for (int i = step; i < maxVal; i += step) {
            numSamplesArray.push_back(i);
        }
    }
    numSamplesArray.push_back(int(std::pow(10, numLogSteps)));

    // Create the chart.
    auto* chart = new HEBChart();
    chart->setIsHeadlessMode(true);
    chart->setVolumeData(volumeData, true);
    chart->addScalarField(fieldIdx, fieldNames.at(fieldIdx));
    chart->setIsEnsembleMode(isEnsembleMode);
    chart->setCorrelationMeasureType(correlationMeasureType);
    chart->setUseAbsoluteCorrelationMeasure(true);
    chart->setNumBins(numBins);
    chart->setKraskovNumNeighbors(k);
    chart->setDownscalingFactors(dfx, dfy, dfz);
    chart->setUseCorrelationComputationGpu(useGpu);
    auto correlationRangeTotal = chart->getCorrelationRangeTotal();
    auto cellDistanceRange = chart->getCellDistanceRangeTotal();
    chart->setCorrelationRange(correlationRangeTotal);
    chart->setCellDistanceRange(cellDistanceRange);
    int xsd = sgl::iceil(xs, dfx);
    int ysd = sgl::iceil(ys, dfy);
    int zsd = sgl::iceil(zs, dfz);

    // Numerate all block pairs.
    std::vector<std::pair<uint32_t, uint32_t>> blockPairs;
    auto numBlocks = uint32_t(xsd * ysd * zsd);
    auto numPairs = uint32_t(numBlocks * numBlocks - numBlocks) / 2;
    for (uint32_t m = 0; m < numPairs; m++) {
        uint32_t i = (1 + sgl::uisqrt(1 + 8 * m)) / 2;
        uint32_t j = m - i * (i - 1) / 2;
        blockPairs.emplace_back(i, j);
    }

    //std::random_device rd;
    //std::mt19937 generator(rd());
    std::mt19937 generator(2);
    std::shuffle(blockPairs.begin(), blockPairs.end(), generator);
    numPairsToCheck = std::min(int(numPairs), numPairsToCheck);
    blockPairs.resize(numPairsToCheck);
    std::sort(blockPairs.begin(), blockPairs.end(), [](
            const std::pair<uint32_t, uint32_t>& x, const std::pair<uint32_t, uint32_t>& y) {
        if (x.first != y.first) {
            return x.first < y.first;
        }
        return x.second < y.second;
    });

    // Compute the ground truth.
    auto startTime = std::chrono::system_clock::now();
    std::vector<std::vector<float>> allValuesSortedArray;
    allValuesSortedArray.resize(numPairsToCheck);
    for (int m = 0; m < numPairsToCheck; m++) {
        std::cout << "Computing GT " << (double(m) / double(numPairsToCheck) * 100.0) << "%..." << std::endl;
        std::vector<float>& allValues = allValuesSortedArray.at(m);
        auto [i, j] = blockPairs.at(m);
        chart->computeAllCorrelationsBlockPair(i, j, allValues);
        std::sort(allValues.begin(), allValues.end());
    }
    auto endTime = std::chrono::system_clock::now();
    auto elapsedTime = std::chrono::duration<double>(endTime - startTime);
    std::cout << "Time for GT: " << elapsedTime.count() << "s" << std::endl;

    // Add the test cases.
    std::vector<TestCase> testCases;
    for (int samplingMethodTypeIdx = int(SamplingMethodType::RANDOM_UNIFORM);
            samplingMethodTypeIdx <= int(SamplingMethodType::QUASIRANDOM_PLASTIC);
            samplingMethodTypeIdx++) {
        for (int numSamples : numSamplesArray) {
            TestCase testCase;
            testCase.samplingMethodType = SamplingMethodType(samplingMethodTypeIdx);
            testCase.numSamples = numSamples;
            testCases.push_back(testCase);
        }
    }

    // Run the tests and write the results to a file.
    sgl::CsvWriter file(
            std::string("Sampling ") + CORRELATION_MEASURE_TYPE_NAMES[int(correlationMeasureType)] + ".csv");
    file.writeRow({ "Sampling Method", "Samples", "Time", "Quantile", "Linear" });
    for (TestCase& testCase : testCases) {
        std::cout << "Test case: " << SAMPLING_METHOD_TYPE_NAMES[int(testCase.samplingMethodType)];
        std::cout << ", samples: " << std::to_string(testCase.numSamples) << std::endl;
        runTestCase(chart, testCase, blockPairs, allValuesSortedArray);
        file.writeCell(SAMPLING_METHOD_TYPE_NAMES[int(testCase.samplingMethodType)]);
        file.writeCell(std::to_string(testCase.numSamples));
        file.writeCell(std::to_string(testCase.elapsedTimeMicroseconds));
        file.writeCell(std::to_string(testCase.maximumQuantile));
        file.writeCell(std::to_string(testCase.rangeLinear));
        file.newRow();
    }
    file.close();

    delete chart;
    delete renderer;
}
