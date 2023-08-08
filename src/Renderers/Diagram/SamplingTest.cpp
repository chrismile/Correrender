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
#include <numeric>

#include <Math/Math.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/CsvWriter.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Volume/VolumeData.hpp"
#include "Renderers/Diagram/HEBChart.hpp"
#include "NLOptDefines.hpp"
#include "SamplingTest.hpp"

struct TestCase {
    std::string testCaseName;
    SamplingMethodType samplingMethodType = SamplingMethodType::QUASIRANDOM_PLASTIC;
    int numSamples = 100;
    double elapsedTimeMicroseconds = 0.0;
    double errorQuantile = 0.0;
    double errorLinear = 0.0;
    double errorAbsolute = 0.0;

    // Bayesian Optimization.
    int numInitSamples = 20;
    int numBOIterations = 60;
    nlopt::algorithm algorithm = nlopt::GN_DIRECT_L_RAND;

    // Subsampling of used field.
    bool useMeanField = false;
    int f = 1;
    std::vector<double> meanPairMaximum;
};

void runTestCase(
        HEBChart* chart, TestCase& testCase, int numRuns,
        const std::vector<std::pair<uint32_t, uint32_t>>& blockPairs,
        const std::vector<std::vector<float>>& allValuesSortedArray,
        const std::vector<float*>& downscaledFields) {
    chart->setSamplingMethodType(testCase.samplingMethodType);
    chart->setNumSamples(testCase.numSamples);
    chart->setNumInitSamples(testCase.numInitSamples);
    chart->setNumBOIterations(testCase.numBOIterations);
    chart->setNloptAlgorithm(testCase.algorithm);
    if (testCase.useMeanField && !chart->getForcedUseMeanFields()) {
        chart->setForcedUseMeanFields(testCase.f, testCase.f, testCase.f);
    } else if (!testCase.useMeanField && chart->getForcedUseMeanFields()) {
        chart->disableForcedUseMeanFields();
    }

    auto invN = 1.0 / double(blockPairs.size() * size_t(numRuns));
    for (int runIdx = 0; runIdx < numRuns; runIdx++) {
        auto statisticsList = chart->computeCorrelationsBlockPairs(blockPairs, downscaledFields, downscaledFields);

        if (!allValuesSortedArray.empty()) {
            // Binary search.
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
                testCase.errorQuantile += invN * (1.0 - double(upper) / double(allValuesSorted.size()));
                float minVal = allValuesSorted.front();
                float maxVal = allValuesSorted.back();
                testCase.errorLinear += invN * (1.0 - double((searchValue - minVal) / (maxVal - minVal)));
                testCase.errorAbsolute += invN * double(maxVal - searchValue);
            }
        } else {
            for (size_t i = 0; i < blockPairs.size(); i++) {
                float searchValue = statisticsList.maximumValues.at(i);
                testCase.errorQuantile += invN * searchValue;
                testCase.errorLinear += invN * searchValue;
                testCase.errorAbsolute += invN * searchValue;
            }
        }
        testCase.elapsedTimeMicroseconds += invN * statisticsList.elapsedTimeMicroseconds;
    }
}

void runTestCaseBlockList(
        HEBChart* chart, TestCase& testCase, int numRuns,
        const std::vector<std::pair<uint32_t, uint32_t>>& blockPairs,
        const std::vector<float*>& downscaledFields) {
    chart->setSamplingMethodType(testCase.samplingMethodType);
    chart->setNumSamples(testCase.numSamples);
    chart->setNumInitSamples(testCase.numInitSamples);
    chart->setNumBOIterations(testCase.numBOIterations);
    chart->setNloptAlgorithm(testCase.algorithm);
    if (testCase.useMeanField && !chart->getForcedUseMeanFields()) {
        chart->setForcedUseMeanFields(testCase.f, testCase.f, testCase.f);
    } else if (!testCase.useMeanField && chart->getForcedUseMeanFields()) {
        chart->disableForcedUseMeanFields();
    }

    testCase.meanPairMaximum.resize(blockPairs.size(), 0.0);
    auto invN = 1.0 / double(blockPairs.size() * size_t(numRuns));
    auto invM = 1.0 / double(size_t(numRuns));
    for (int runIdx = 0; runIdx < numRuns; runIdx++) {
        auto statisticsList = chart->computeCorrelationsBlockPairs(blockPairs, downscaledFields, downscaledFields);

        for (size_t i = 0; i < blockPairs.size(); i++) {
            float searchValue = statisticsList.maximumValues.at(i);
            testCase.meanPairMaximum.at(i) += invM * searchValue;
            testCase.errorQuantile += invN * searchValue;
            testCase.errorLinear += invN * searchValue;
            testCase.errorAbsolute += invN * searchValue;
        }
        testCase.elapsedTimeMicroseconds += invN * statisticsList.elapsedTimeMicroseconds;
    }
}

const int TEST_CASE_DATA_ERROR = 0;
const int TEST_CASE_DATA_MAX = 1;
const int TEST_CASE_SYNTH_ERROR = 2;
const int TEST_CASE_DATA_MAX_SUBSAMPLED = 3;

void runSamplingTests(const std::string& dataSetPath, int testIdx) {
    std::cout << "Starting test case #" << testIdx << " for data set '" << dataSetPath << "'." << std::endl;
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    auto* renderer = new sgl::vk::Renderer(device, 100);

    const bool modeGT =
            testIdx != TEST_CASE_DATA_MAX && testIdx != TEST_CASE_DATA_MAX_SUBSAMPLED;
    bool isSyntheticTestCase = testIdx == TEST_CASE_SYNTH_ERROR;

    // Load the volume data set.
    VolumeDataPtr volumeData(new VolumeData(renderer));
    DataSetInformation dataSetInformation;
    dataSetInformation.filenames = { dataSetPath };
    volumeData->setInputFiles({ dataSetPath }, dataSetInformation, nullptr);
    int xs = volumeData->getGridSizeX();
    int ys = volumeData->getGridSizeY();
    int zs = volumeData->getGridSizeZ();
    int es = volumeData->getEnsembleMemberCount();
    int ts = volumeData->getTimeStepCount();
    bool isEnsembleMode = es > 1;
    int cs = isEnsembleMode ? es : ts;
    int k = std::max(sgl::iceil(3 * cs, 100), 10);
    int numBins = 80;

    // Settings.
    constexpr CorrelationMeasureType correlationMeasureType = CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV;
    const auto& fieldNames = volumeData->getFieldNamesBase(FieldType::SCALAR);
    constexpr int fieldIdx = 0;
    constexpr bool useGpu = true;
    //constexpr int dfx = 10;
    //constexpr int dfy = 10;
    //constexpr int dfz = 10;
    const int dfx = testIdx == TEST_CASE_DATA_ERROR ? 8 : 32;
    const int dfy = testIdx == TEST_CASE_DATA_ERROR ? 8 : 32;
    const int dfz = testIdx == TEST_CASE_DATA_ERROR ? 5 : (isSyntheticTestCase ? 32 : 20);
    const int numRuns = 10;//testIdx == TEST_CASE_DATA_ERROR ? 10 : 100;
    int numPairsToCheck = testIdx == TEST_CASE_DATA_ERROR ? 1000 : 3828;
    int numLogSteps = 3;
    std::vector<int> numSamplesArray;
    for (int l = 0; l < numLogSteps; l++) {
        auto step = int(std::pow(10, l));
        auto maxVal = int(std::pow(10, l + 1));
        for (int i = step; i < maxVal; i += step) {
            //if (i < 5) {
            //    continue;
            //}
            numSamplesArray.push_back(i);
        }
    }
    numSamplesArray.push_back(int(std::pow(10, numLogSteps)));
    //if (numLogSteps == 2) {
    //    numSamplesArray.push_back(200);
    //}
    bool runTestsOptimizers = false;
    bool computeGroundTruth = modeGT;

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

    if (isSyntheticTestCase) {
        chart->setCorrelationRange(glm::vec2(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()));
        chart->setSyntheticTestCase(blockPairs);
    }

    // Cache the field entries so loading is not performed when starting the first technique.
    chart->createFieldCacheForTests();

    // Compute the ground truth.
    std::vector<std::vector<float>> allValuesSortedArray;
    double timeGt = 0.0;
    if (computeGroundTruth) {
        auto startTime = std::chrono::system_clock::now();
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
        timeGt = elapsedTime.count();
        std::cout << "Time for GT: " << timeGt << "s" << std::endl;
    }

    // Compute the downscaled field.
    std::vector<float*> downscaledFields;
    //if (computeMean) {
    //    downscaledFields.resize(cs);
    //    chart->computeDownscaledFieldPerfTest(downscaledFields);
    //}

    // Add the test cases.
    std::vector<TestCase> testCases;
    if (testIdx != TEST_CASE_DATA_MAX_SUBSAMPLED) {
        SamplingMethodType firstSamplingMethodType = SamplingMethodType::RANDOM_UNIFORM;
        //firstSamplingMethodType = SamplingMethodType::BAYESIAN_OPTIMIZATION; // Just for testing, remove this line.
        for (int samplingMethodTypeIdx = int(firstSamplingMethodType);
             samplingMethodTypeIdx <= int(SamplingMethodType::QUASIRANDOM_PLASTIC);
             samplingMethodTypeIdx++) {
            for (int numSamples : numSamplesArray) {
                TestCase testCase;
                testCase.samplingMethodType = SamplingMethodType(samplingMethodTypeIdx);
                testCase.numSamples = numSamples;
                testCase.testCaseName = SAMPLING_METHOD_TYPE_NAMES[int(testCase.samplingMethodType)];
                testCases.push_back(testCase);
            }
        }

        // Add Bayesian Optimization test cases.
        if (runTestsOptimizers) {
            for (size_t algoIdx = 0; algoIdx < IM_ARRAYSIZE(NLoptAlgorithmsNoGrad); algoIdx++) {
                auto algorithm = NLoptAlgorithmsNoGrad[algoIdx];
                // Seems to hang the program. One thread worker doesn't terminate.
                if (algorithm == nlopt::LN_NEWUOA_BOUND) {
                    continue;
                }
                for (int numSamples : numSamplesArray) {
                    TestCase testCase;
                    testCase.samplingMethodType = SamplingMethodType::BAYESIAN_OPTIMIZATION;
                    testCase.algorithm = algorithm;
                    testCase.numSamples = numSamples;
                    testCase.numInitSamples = std::clamp(sgl::iceil(numSamples, 2), 1, 20);
                    testCase.testCaseName = SAMPLING_METHOD_TYPE_NAMES[int(testCase.samplingMethodType)];
                    testCase.testCaseName += std::string(" (") + NLOPT_ALGORITHM_NAMES_NOGRAD[algoIdx] + ")";
                    testCases.push_back(testCase);
                }
            }
        } else {
            for (int numSamples : numSamplesArray) {
                TestCase testCase;
                testCase.samplingMethodType = SamplingMethodType::BAYESIAN_OPTIMIZATION;
                testCase.numSamples = numSamples;
                testCase.numInitSamples = std::clamp(sgl::iceil(numSamples, 2), 1, 20);
                testCase.testCaseName = SAMPLING_METHOD_TYPE_NAMES[int(testCase.samplingMethodType)];
                testCases.push_back(testCase);
            }
        }
    } else {
        for (int f = 1; f <= 4; f *= 2) {
            TestCase testCase;
            testCase.samplingMethodType = SamplingMethodType::BAYESIAN_OPTIMIZATION;
            //testCase.samplingMethodType = SamplingMethodType::QUASIRANDOM_PLASTIC;
            testCase.numSamples = 100;
            testCase.numInitSamples = std::clamp(sgl::iceil(testCase.numSamples, 2), 1, 20);
            testCase.useMeanField = true;
            testCase.f = f;
            testCase.testCaseName = SAMPLING_METHOD_TYPE_NAMES[int(testCase.samplingMethodType)];
            testCase.testCaseName += std::string(" (f=") + std::to_string(f) + ")";
            testCases.push_back(testCase);
        }
    }

    // Run the tests and write the results to a file.
    std::string csvFilename = "Sampling ";
    if (testIdx == TEST_CASE_DATA_MAX_SUBSAMPLED) {
        csvFilename += "Mean ";
        csvFilename += CORRELATION_MEASURE_TYPE_NAMES[int(correlationMeasureType)];
    } else {
        if (isSyntheticTestCase) {
            csvFilename += "Synthetic";
        } else {
            csvFilename += CORRELATION_MEASURE_TYPE_NAMES[int(correlationMeasureType)];
        }
    }
    csvFilename += " " + std::to_string(dfx) + "x" + std::to_string(dfy) + "x" + std::to_string(dfz);
    csvFilename += ".csv";
    sgl::CsvWriter file(csvFilename);

    if (testIdx == TEST_CASE_DATA_MAX_SUBSAMPLED) {
        file.writeRow({ "Block Pair Index", "Max f=1", "Max f=2", "Max f=4" });
        for (size_t testCaseIdx = 0; testCaseIdx < testCases.size(); testCaseIdx++) {
            TestCase& testCase = testCases.at(testCaseIdx);
            std::cout << "Test case: " << testCase.testCaseName << std::endl;
            runTestCaseBlockList(chart, testCase, numRuns, blockPairs, downscaledFields);
        }

        // Sort arrays from lowest to highest.
        std::vector<int> permutationArray(blockPairs.size(), 0);
        std::iota(permutationArray.begin(), permutationArray.end(), 0);
        auto& baseMeanPairMaximum = testCases.at(0).meanPairMaximum;
        std::sort(permutationArray.begin(), permutationArray.end(),
             [&baseMeanPairMaximum](const int& i, const int& j) {
                 return baseMeanPairMaximum[i] < baseMeanPairMaximum[j];
             }
        );

        for (size_t blockPairIdx = 0; blockPairIdx < blockPairs.size(); blockPairIdx++) {
            file.writeCell(std::to_string(blockPairIdx));
            for (size_t testCaseIdx = 0; testCaseIdx < testCases.size(); testCaseIdx++) {
                TestCase& testCase = testCases.at(testCaseIdx);
                file.writeCell(std::to_string(testCase.meanPairMaximum.at(permutationArray.at(blockPairIdx))));
            }
            file.newRow();
        }
    } else {
        file.writeRow({ "Sampling Method", "Samples", "Time", "Quantile", "Linear", "Absolute", "Init Samples" });
        size_t firstMeanIdx = 0;
        for (size_t testCaseIdx = 0; testCaseIdx < testCases.size(); testCaseIdx++) {
            TestCase& testCaseAtIdx = testCases.at(testCaseIdx);
            size_t testCaseIdxReal = testCaseIdx;
            if (testCaseAtIdx.samplingMethodType == SamplingMethodType::MEAN && testCaseIdx > 0) {
                testCaseIdxReal = firstMeanIdx;
            }

            TestCase& testCase = testCases.at(testCaseIdxReal);
            if (testCase.samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION && testCase.numSamples > 400) {
                continue;
            }
            std::cout << "Test case: " << testCase.testCaseName;
            std::cout << ", samples: " << std::to_string(testCaseAtIdx.numSamples) << std::endl;
            if (testCaseIdx == testCaseIdxReal) {
                runTestCase(chart, testCase, numRuns, blockPairs, allValuesSortedArray, downscaledFields);
            }
            file.writeCell(testCase.testCaseName);
            file.writeCell(std::to_string(testCaseAtIdx.numSamples));
            file.writeCell(std::to_string(testCase.elapsedTimeMicroseconds));
            file.writeCell(std::to_string(testCase.errorQuantile));
            file.writeCell(std::to_string(testCase.errorLinear));
            file.writeCell(std::to_string(testCase.errorAbsolute));
            file.writeCell(std::to_string(testCase.numInitSamples));
            //if (testCase.samplingMethodType == SamplingMethodType::BAYESIAN_OPTIMIZATION) {
            //    file.writeCell(NLOPT_ALGORITHM_NAMES_NOGRAD[int(testCase.algorithm)]);
            //} else {
            //    file.writeCell("-");
            //}
            file.newRow();
        }
    }

    file.close();

    if (computeGroundTruth) {
        std::cout << "Time for GT: " << timeGt << "s" << std::endl;
    }

    delete chart;
    delete renderer;
}
