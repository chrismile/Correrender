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

#include <cstring>
#include <utility>

#include <boost/algorithm/string/case_conv.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <Utils/Parallel/Reduction.hpp>
#endif

#include <Utils/AppSettings.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/Events/EventManager.hpp>
#include <Utils/Parallel/Reduction.hpp>
#include <Graphics/Vulkan/Render/Data.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/Widgets/TransferFunctionWindow.hpp>
#include <ImGui/imgui_custom.h>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

#include "Loaders/DataSet.hpp"
#include "Loaders/VolumeLoader.hpp"
#include "Loaders/AmiraMeshLoader.hpp"
#include "Loaders/DatRawFileLoader.hpp"
#include "Loaders/FieldFileLoader.hpp"
#ifdef USE_ECCODES
#include "Loaders/GribLoader.hpp"
#endif
#include "Loaders/NetCdfLoader.hpp"
#include "Loaders/RbcBinFileLoader.hpp"
#include "Loaders/StructuredGridVtkLoader.hpp"
#include "Loaders/VtkXmlLoader.hpp"
#ifdef USE_ZARR
#include "Loaders/ZarrLoader.hpp"
#endif
#include "Calculators/Calculator.hpp"
#include "Calculators/VelocityCalculator.hpp"
#include "Calculators/SimilarityCalculator.hpp"
#ifdef SUPPORT_PYTORCH
#include "Calculators/PyTorchSimilarityCalculator.hpp"
#endif
#include "Renderers/RenderingModes.hpp"
#include "Renderers/Renderer.hpp"
#include "VolumeData.hpp"

template <typename T>
static std::pair<std::vector<std::string>, std::function<VolumeLoader*()>> registerVolumeLoader() {
    return { T::getSupportedExtensions(), []() { return new T{}; }};
}

VolumeLoader* VolumeData::createVolumeLoaderByExtension(const std::string& fileExtension) {
    auto it = factories.find(fileExtension);
    if (it == factories.end()) {
        sgl::Logfile::get()->throwError(
                "Error in VolumeData::createVolumeLoaderByExtension: Unsupported file extension '."
                + fileExtension + "'.");
        return nullptr;
    } else {
        return it->second();
    }
}

VolumeData::VolumeData(sgl::vk::Renderer* renderer) : renderer(renderer), multiVarTransferFunctionWindow() {
    device = renderer->getDevice();
    hostFieldCache = std::make_unique<HostFieldCache>();
    deviceFieldCache = std::make_unique<DeviceFieldCache>(device);
    fieldMinMaxCache = std::make_unique<FieldMinMaxCache>();

    typeToFieldNamesMap.insert(std::make_pair(FieldType::SCALAR, std::vector<std::string>()));
    typeToFieldNamesMap.insert(std::make_pair(FieldType::VECTOR, std::vector<std::string>()));
    typeToFieldNamesMapBase.insert(std::make_pair(FieldType::SCALAR, std::vector<std::string>()));
    typeToFieldNamesMapBase.insert(std::make_pair(FieldType::VECTOR, std::vector<std::string>()));

    // Create the list of volume loaders.
    std::map<std::vector<std::string>, std::function<VolumeLoader*()>> factoriesMap = {
            registerVolumeLoader<AmiraMeshLoader>(),
            registerVolumeLoader<DatRawFileLoader>(),
            registerVolumeLoader<FieldFileLoader>(),
#ifdef USE_ECCODES
            registerVolumeLoader<GribLoader>(),
#endif
            registerVolumeLoader<NetCdfLoader>(),
            registerVolumeLoader<RbcBinFileLoader>(),
            registerVolumeLoader<StructuredGridVtkLoader>(),
            registerVolumeLoader<VtkXmlLoader>(),
#ifdef USE_ZARR
            registerVolumeLoader<ZarrLoader>(),
#endif
    };
    for (auto& factory : factoriesMap) {
        for (const std::string& extension : factory.first) {
            factories.insert(std::make_pair(extension, factory.second));
        }
    }

    sgl::vk::ImageSamplerSettings samplerSettings{};
    imageSampler = std::make_shared<sgl::vk::ImageSampler>(device, samplerSettings, 0);
}

VolumeData::~VolumeData() {
    for (VolumeLoader* volumeLoader : volumeLoaders) {
        delete volumeLoader;
    }
    volumeLoaders.clear();
}

void VolumeData::setTransposeAxes(const glm::ivec3& axes) {
    this->transposeAxes = axes;
    transpose = true;
}

void VolumeData::setGridSubsamplingFactor(int factor) {
    subsamplingFactor = factor;
}

void VolumeData::setGridExtent(int _xs, int _ys, int _zs, float _dx, float _dy, float _dz) {
    xs = _xs;
    ys = _ys;
    zs = _zs;
    dx = _dx;
    dy = _dy;
    dz = _dz;

    if (transpose) {
        int dimensions[3] = { xs, ys, zs };
        float spacing[3] = { dx, dy, dz };
        xs = dimensions[transposeAxes[0]];
        ys = dimensions[transposeAxes[1]];
        zs = dimensions[transposeAxes[2]];
        dx = spacing[transposeAxes[0]];
        dy = spacing[transposeAxes[1]];
        dz = spacing[transposeAxes[2]];
    }

    ssxs = xs;
    ssys = ys;
    sszs = zs;

    if (subsamplingFactor > 1) {
        xs /= subsamplingFactor;
        ys /= subsamplingFactor;
        zs /= subsamplingFactor;
        dx *= float(subsamplingFactor);
        dy *= float(subsamplingFactor);
        dz *= float(subsamplingFactor);
    }

    box = sgl::AABB3(
            glm::vec3(0.0f),
            glm::vec3(float(xs - 1) * dx, float(ys - 1) * dy, float(zs - 1) * dz));
    glm::vec3 dimensions = box.getDimensions();
    float maxDimension = std::max(dimensions.x, std::max(dimensions.y, dimensions.z));
    glm::vec3 normalizedDimensions = dimensions / maxDimension;
    boxRendering.min = -normalizedDimensions * 0.25f;
    boxRendering.max = normalizedDimensions * 0.25f;
}

void VolumeData::setNumTimeSteps(int _ts) {
    ts = _ts;
}

void VolumeData::setTimeSteps(const std::vector<int>& timeSteps) {
    ts = int(timeSteps.size());
}

void VolumeData::setTimeSteps(const std::vector<float>& timeSteps) {
    ts = int(timeSteps.size());
}

void VolumeData::setTimeSteps(const std::vector<double>& timeSteps) {
    ts = int(timeSteps.size());
}

void VolumeData::setTimeSteps(const std::vector<std::string>& timeSteps) {
    ts = int(timeSteps.size());
}

void VolumeData::setEnsembleMemberCount(int _es) {
    es = _es;
}

void VolumeData::setFieldNames(const std::unordered_map<FieldType, std::vector<std::string>>& fieldNamesMap) {
    typeToFieldNamesMap = fieldNamesMap;
    typeToFieldNamesMapBase = fieldNamesMap;
}

void VolumeData::addField(
        float* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    if (transpose) {
        if (transposeAxes != glm::ivec3(0, 2, 1)) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::addScalarField: At the moment, only transposing the "
                    "Y and Z axis is supported.");
        }
        if (fieldType == FieldType::SCALAR) {
            auto* scalarFieldCopy = new float[ssxs * ssys * sszs];
            memcpy(scalarFieldCopy, fieldData, sizeof(float) * ssxs * ssys * sszs);
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(fieldData, scalarFieldCopy) default(none)
#endif
            for (int z = 0; z < sszs; z++) {
#endif
                for (int y = 0; y < ssys; y++) {
                    for (int x = 0; x < ssxs; x++) {
                        int readPos = ((y)*ssxs*sszs + (z)*ssxs + (x));
                        int writePos = ((z)*ssxs*ssys + (y)*ssxs + (x));
                        fieldData[writePos] = scalarFieldCopy[readPos];
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            delete[] scalarFieldCopy;
        } else {
            auto* vectorFieldCopy = new float[3 * ssxs * ssys * sszs];
            memcpy(vectorFieldCopy, fieldData, sizeof(float) * 3 * ssxs * ssys * sszs);
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(fieldData, vectorFieldCopy) default(none)
#endif
            for (int z = 0; z < sszs; z++) {
#endif
                for (int y = 0; y < ssys; y++) {
                    for (int x = 0; x < ssxs; x++) {
                        int readPos = ((y)*ssxs*sszs*3 + (z)*ssxs*3 + (x)*3);
                        int writePos = ((z)*ssxs*ssys*3 + (y)*ssxs*3 + (x)*3);
                        fieldData[writePos] = vectorFieldCopy[readPos];
                        fieldData[writePos + 1] = vectorFieldCopy[readPos + 2];
                        fieldData[writePos + 2] = vectorFieldCopy[readPos + 1];
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            delete[] vectorFieldCopy;
        }
    }

    if (subsamplingFactor > 1) {
        if (fieldType == FieldType::SCALAR) {
            float* scalarFieldOld = fieldData;
            fieldData = new float[xs * ys * zs];
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(fieldData, scalarFieldOld) default(none)
#endif
            for (int z = 0; z < zs; z++) {
#endif
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        int readPos =
                                ((z * subsamplingFactor)*ssxs*ssys
                                 + (y * subsamplingFactor)*ssxs
                                 + (x * subsamplingFactor));
                        int writePos = ((z)*xs*ys + (y)*xs + (x));
                        fieldData[writePos] = scalarFieldOld[readPos];
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            delete[] scalarFieldOld;
        } else {
            float* vectorFieldOld = fieldData;
            fieldData = new float[3 * xs * ys * zs];
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(fieldData, vectorFieldOld) default(none)
#endif
            for (int z = 0; z < zs; z++) {
#endif
                for (int y = 0; y < ys; y++) {
                    for (int x = 0; x < xs; x++) {
                        int readPos =
                                ((z * subsamplingFactor)*ssxs*ssys*3
                                 + (y * subsamplingFactor)*ssxs*3
                                 + (x * subsamplingFactor)*3);
                        int writePos = ((z)*xs*ys*3 + (y)*xs*3 + (x)*3);
                        fieldData[writePos] = vectorFieldOld[readPos];
                        fieldData[writePos + 1] = vectorFieldOld[readPos + 1];
                        fieldData[writePos + 2] = vectorFieldOld[readPos + 2];
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            delete[] vectorFieldOld;
        }
    }

    FieldAccess access = createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);

    if (hostFieldCache->exists(access)) {
        delete[] fieldData;
        return;
    }
    hostFieldCache->push(access, HostCacheEntry(fieldData));

    /*if (fieldType == FieldType::VECTOR) {
#ifdef USE_TBB
        float maxVectorMagnitude = tbb::parallel_reduce(
                tbb::blocked_range<int>(0, zs), 0.0f,
                [&vectorField, this](tbb::blocked_range<int> const& r, float maxVectorMagnitude) {
                    for (auto z = r.begin(); z != r.end(); z++) {
#else
        float maxVectorMagnitude = 0.0f;
#if _OPENMP >= 201107
        #pragma omp parallel for shared(fieldData) reduction(max: maxVectorMagnitude) default(none)
#endif
        for (int z = 0; z < zs; z++) {
#endif
            for (int y = 0; y < ys; y++) {
                for (int x = 0; x < xs; x++) {
                    float vx = fieldData[IDXV(x, y, z, 0)];
                    float vy = fieldData[IDXV(x, y, z, 1)];
                    float vz = fieldData[IDXV(x, y, z, 2)];
                    float vectorMagnitude = std::sqrt(vx * vx + vy * vy + vz * vz);
                    maxVectorMagnitude = std::max(maxVectorMagnitude, vectorMagnitude);
                }
            }
        }
#ifdef USE_TBB
                    return maxVectorMagnitude;
                }, sgl::max_predicate());
#endif
    }*/
}

const std::vector<std::string>& VolumeData::getFieldNames(FieldType fieldType) {
    return typeToFieldNamesMap[fieldType];
}

const std::vector<std::string>& VolumeData::getFieldNamesBase(FieldType fieldType) {
    return typeToFieldNamesMapBase[fieldType];
}

bool VolumeData::getFieldExists(FieldType fieldType, const std::string& fieldName) const {
    const std::vector<std::string>& names = typeToFieldNamesMap.find(fieldType)->second;
    auto it = std::find(names.begin(), names.end(), fieldName);
    return it != names.end();
}

bool VolumeData::setInputFiles(
        const std::vector<std::string>& _filePaths, DataSetInformation _dataSetInformation,
        glm::mat4* transformationMatrixPtr) {
    filePaths = _filePaths;
    dataSetInformation = std::move(_dataSetInformation);

    if (dataSetInformation.timeSteps.empty()) {
        ts = 1;
    } else {
        setTimeSteps(dataSetInformation.timeSteps);
    }
    tsFileCount = ts;
    esFileCount = es = int(filePaths.size()) / tsFileCount;

    for (size_t i = 0; i < filePaths.size(); i++) {
        std::string filePath = filePaths.at(i);
        std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(filePath);
        VolumeLoader* volumeLoader = createVolumeLoaderByExtension(fileExtension);
        volumeLoader->setInputFiles(this, filePath, dataSetInformation);
        volumeLoaders.push_back(volumeLoader);
    }

    // Automatically add calculators for velocity, vorticity and helicity if possible.
    bool uLowerCaseVariableExists = getFieldExists(FieldType::SCALAR, "u");
    bool vLowerCaseVariableExists = getFieldExists(FieldType::SCALAR, "v");
    bool wLowerCaseVariableExists = getFieldExists(FieldType::SCALAR, "w");
    bool uUpperCaseVariableExists = getFieldExists(FieldType::SCALAR, "U");
    bool vUpperCaseVariableExists = getFieldExists(FieldType::SCALAR, "V");
    bool wUpperCaseVariableExists = getFieldExists(FieldType::SCALAR, "W");
    bool velocityVariableExists = getFieldExists(FieldType::SCALAR, "Velocity");
    bool velocityMagnitudeVariableExists = getFieldExists(FieldType::SCALAR, "Velocity Magnitude");
    bool vorticityVariableExists = getFieldExists(FieldType::SCALAR, "Vorticity");
    bool vorticityMagnitudeVariableExists = getFieldExists(FieldType::SCALAR, "Vorticity Magnitude");
    bool helicityVariableExists = getFieldExists(FieldType::SCALAR, "Helicity");
    if (!velocityVariableExists && ((uLowerCaseVariableExists && vLowerCaseVariableExists && wLowerCaseVariableExists)
            || (uUpperCaseVariableExists && vUpperCaseVariableExists && wUpperCaseVariableExists))) {
        addCalculator(std::make_shared<VelocityCalculator>(renderer));
        velocityVariableExists = true;
    }
    if (!velocityMagnitudeVariableExists && velocityVariableExists) {
        addCalculator(std::make_shared<VectorMagnitudeCalculator>(renderer, "Velocity"));
        velocityMagnitudeVariableExists = true;
    }
    if (!vorticityVariableExists && velocityVariableExists) {
        addCalculator(std::make_shared<VorticityCalculator>(renderer));
        vorticityVariableExists = true;
    }
    if (!vorticityMagnitudeVariableExists && vorticityVariableExists) {
        addCalculator(std::make_shared<VectorMagnitudeCalculator>(renderer, "Vorticity"));
        vorticityMagnitudeVariableExists = true;
    }
    if (!helicityVariableExists && velocityVariableExists && vorticityVariableExists) {
        addCalculator(std::make_shared<HelicityCalculator>(renderer));
        helicityVariableExists = true;
    }

    addCalculator(std::make_shared<PccCalculator>(renderer));
#ifdef SUPPORT_PYTORCH
    addCalculator(std::make_shared<PyTorchSimilarityCalculator>(renderer));
#endif

    const auto& scalarFieldNames = typeToFieldNamesMap[FieldType::SCALAR];
    multiVarTransferFunctionWindow.setRequestAttributeValuesCallback([this](
            int varIdx, std::shared_ptr<float[]>& values, size_t& numValues, float& minValue, float& maxValue) {
        std::string fieldName = typeToFieldNamesMap[FieldType::SCALAR].at(varIdx);
        HostCacheEntry cacheEntry = this->getFieldEntryCpu(FieldType::SCALAR, fieldName);
        values = cacheEntry;
        numValues = this->getSlice3dEntryCount();
        std::tie(minValue, maxValue) = getMinMaxScalarFieldValue(fieldName);
    });
    multiVarTransferFunctionWindow.setAttributeNames(scalarFieldNames);

    colorLegendWidgets.clear();
    colorLegendWidgets.resize(scalarFieldNames.size());
    for (size_t i = 0; i < colorLegendWidgets.size(); i++) {
        colorLegendWidgets.at(i).setPositionIndex(0, 1);
        colorLegendWidgets.at(i).setAttributeDisplayName(scalarFieldNames.at(i));
    }
    recomputeColorLegend();

    return true;
}

void VolumeData::addCalculator(const CalculatorPtr& calculator) {
    calculators.push_back(calculator);
    if (calculator->getFilterDevice() == FilterDevice::CPU) {
        calculatorsHost.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
    } else {
        calculatorsDevice.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
    }
    typeToFieldNamesMap[calculator->getOutputFieldType()].push_back(calculator->getOutputFieldName());
    calculator->setCalculatorId(calculatorId++);
    calculator->setVolumeData(this, true);
}

FieldAccess VolumeData::createFieldAccessStruct(
        FieldType fieldType, const std::string& fieldName, int& timeStepIdx, int& ensembleIdx) const {
    if (timeStepIdx < 0) {
        timeStepIdx = currentTimeStepIdx;
    }
    if (ensembleIdx < 0) {
        ensembleIdx = currentEnsembleIdx;
    }

    FieldAccess access;
    access.fieldType = fieldType;
    access.fieldName = fieldName;
    access.timeStepIdx = timeStepIdx;
    access.ensembleIdx = ensembleIdx;
    access.sizeInBytes = getSlice3dSizeInBytes(fieldType);

    return access;
}

VolumeData::HostCacheEntry VolumeData::getFieldEntryCpu(
        FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    FieldAccess access = createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);

    if (hostFieldCache->exists(access)) {
        return hostFieldCache->reaccess(access);
    }

    size_t bufferSize = getSlice3dSizeInBytes(fieldType);
    hostFieldCache->ensureSufficientMemory(bufferSize);

    float* fieldEntryBuffer = nullptr;
    auto itCalc = calculatorsHost.find(fieldName);
    if (itCalc != calculatorsHost.end()) {
        Calculator* calculator = itCalc->second.get();
        if (calculator->getOutputFieldType() != fieldType) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::getFieldEntryCpu: Mismatch between field type and calculator output for "
                    "field \"" + fieldName + "\".");
        }
        size_t numComponents = fieldType == FieldType::SCALAR ? 1 : 3;
        fieldEntryBuffer = new float[size_t(xs) * size_t(ys) * size_t(zs) * numComponents];
        calculator->calculateCpu(timeStepIdx, ensembleIdx, fieldEntryBuffer);
    } else if (calculatorsDevice.find(fieldName) != calculatorsDevice.end()) {
        auto deviceEntry = getFieldEntryDevice(fieldType, fieldName, timeStepIdx, ensembleIdx);
        size_t numComponents = fieldType == FieldType::SCALAR ? 1 : 3;
        size_t sizeInBytes = numComponents * size_t(xs) * size_t(ys) * size_t(zs) * sizeof(float);
        if (!stagingBuffer) {
            stagingBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, sizeInBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
        }
        deviceEntry->getVulkanImage()->copyToBuffer(stagingBuffer); //< synchronous copy.
        fieldEntryBuffer = new float[size_t(xs) * size_t(ys) * size_t(zs) * numComponents];
        void* data = stagingBuffer->mapMemory();
        memcpy(fieldEntryBuffer, data, sizeInBytes);
        stagingBuffer->unmapMemory();
    } else {
        VolumeLoader* volumeLoader = nullptr;
        if (tsFileCount == 1 && esFileCount == 1) {
            volumeLoader = volumeLoaders.front();
        } else if (tsFileCount > 1 && esFileCount == 1) {
            volumeLoader = volumeLoaders.at(timeStepIdx);
        } else if (tsFileCount == 1 && esFileCount > 1) {
            volumeLoader = volumeLoaders.at(ensembleIdx);
        } else if (tsFileCount > 1 && esFileCount > 1 && !dataSetInformation.useTimeStepRange) {
            volumeLoader = volumeLoaders.at(timeStepIdx * esFileCount + ensembleIdx);
        } else if (tsFileCount > 1 && esFileCount > 1 && dataSetInformation.useTimeStepRange) {
            volumeLoader = volumeLoaders.at(ensembleIdx * tsFileCount + timeStepIdx);
        } else {
            return {};
        }

        if (!volumeLoader->getFieldEntry(
                this, fieldType, fieldName, timeStepIdx, ensembleIdx, fieldEntryBuffer)) {
            sgl::Logfile::get()->throwError(
                    "Fatal error in VolumeData::getFieldEntryCpu: Volume loader failed to load the data entry.");
            return {};
        }
    }

    // Loaders may load multiple fields at once and leave fieldEntryBuffer empty.
    if (!fieldEntryBuffer) {
        if (!hostFieldCache->exists(access)) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::getFieldEntryCpu: Loader supporting loading multiple fields at once "
                    "did not load a field called \"" + fieldName + "\".");
        }
        return hostFieldCache->reaccess(access);
    } else {
        if (transpose) {
            if (transposeAxes != glm::ivec3(0, 2, 1)) {
                sgl::Logfile::get()->throwError(
                        "Error in VolumeData::addScalarField: At the moment, only transposing the "
                        "Y and Z axis is supported.");
            }
            if (fieldType == FieldType::SCALAR) {
                auto* scalarFieldCopy = new float[ssxs * ssys * sszs];
                memcpy(scalarFieldCopy, fieldEntryBuffer, sizeof(float) * ssxs * ssys * sszs);
#ifdef USE_TBB
                tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
                    for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
                #pragma omp parallel for shared(fieldEntryBuffer, scalarFieldCopy) default(none)
#endif
                for (int z = 0; z < sszs; z++) {
#endif
                    for (int y = 0; y < ssys; y++) {
                        for (int x = 0; x < ssxs; x++) {
                            int readPos = ((y)*ssxs*sszs + (z)*ssxs + (x));
                            int writePos = ((z)*ssxs*ssys + (y)*ssxs + (x));
                            fieldEntryBuffer[writePos] = scalarFieldCopy[readPos];
                        }
                    }
                }
#ifdef USE_TBB
                });
#endif
                delete[] scalarFieldCopy;
            } else {
                auto* vectorFieldCopy = new float[3 * ssxs * ssys * sszs];
                memcpy(vectorFieldCopy, fieldEntryBuffer, sizeof(float) * 3 * ssxs * ssys * sszs);
#ifdef USE_TBB
                tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
                    for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
                #pragma omp parallel for shared(fieldEntryBuffer, vectorFieldCopy) default(none)
#endif
                for (int z = 0; z < sszs; z++) {
#endif
                    for (int y = 0; y < ssys; y++) {
                        for (int x = 0; x < ssxs; x++) {
                            int readPos = ((y)*ssxs*sszs*3 + (z)*ssxs*3 + (x)*3);
                            int writePos = ((z)*ssxs*ssys*3 + (y)*ssxs*3 + (x)*3);
                            fieldEntryBuffer[writePos] = vectorFieldCopy[readPos];
                            fieldEntryBuffer[writePos + 1] = vectorFieldCopy[readPos + 2];
                            fieldEntryBuffer[writePos + 2] = vectorFieldCopy[readPos + 1];
                        }
                    }
                }
#ifdef USE_TBB
                });
#endif
                delete[] vectorFieldCopy;
            }
        }
    }

    auto buffer = std::shared_ptr<float[]>(fieldEntryBuffer);
    hostFieldCache->push(access, buffer);
    return buffer;
}

VolumeData::DeviceCacheEntry VolumeData::getFieldEntryDevice(
        FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    FieldAccess access = createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);

    if (deviceFieldCache->exists(access)) {
        return deviceFieldCache->reaccess(access);
    }

    size_t bufferSize = getSlice3dSizeInBytes(fieldType);
    deviceFieldCache->ensureSufficientMemory(bufferSize);
    auto itCalc = calculatorsDevice.find(fieldName);

#ifdef SUPPORT_CUDA_INTEROP
    bool canUseCuda = sgl::vk::getIsCudaDeviceApiFunctionTableInitialized();
#else
    bool canUseCuda = false;
#endif

    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = uint32_t(xs);
    imageSettings.height = uint32_t(ys);
    imageSettings.depth = uint32_t(zs);
    imageSettings.imageType = VK_IMAGE_TYPE_3D;
    imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    imageSettings.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
    imageSettings.exportMemory = canUseCuda;
    imageSettings.useDedicatedAllocationForExportedMemory = false;

    if (itCalc != calculatorsDevice.end()) {
        imageSettings.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    auto image = std::make_shared<sgl::vk::Image>(device, imageSettings);
    DeviceCacheEntry deviceCacheEntry = std::make_shared<DeviceCacheEntry::element_type>(image, imageSampler);

    if (itCalc != calculatorsDevice.end()) {
        Calculator* calculator = itCalc->second.get();
        if (calculator->getOutputFieldType() != fieldType) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::getFieldEntryDevice: Mismatch between field type and calculator output for "
                    "field \"" + fieldName + "\".");
        }
        calculator->calculateDevice(timeStepIdx, ensembleIdx, deviceCacheEntry);
    } else {
        auto bufferCpu = getFieldEntryCpu(fieldType, fieldName, timeStepIdx, ensembleIdx);
        if (fieldType == FieldType::SCALAR) {
            image->uploadData(access.sizeInBytes, bufferCpu.get());
        } else {
            size_t bufferEntriesCount = size_t(xs) * size_t(ys) * size_t(zs);
            float* bufferIn = bufferCpu.get();
            auto* bufferPadded = new float[bufferEntriesCount * 4];
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<size_t>(0, bufferEntriesCount), [&](auto const& r) {
                for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 200805
            #pragma omp parallel for shared(bufferEntriesCount, bufferPadded, bufferIn) default(none)
#endif
            for (size_t i = 0; i < bufferEntriesCount; i++) {
#endif
                size_t iPadded = i * 4;
                size_t iIn = i * 3;
                bufferPadded[iPadded] = bufferIn[iIn];
                bufferPadded[iPadded + 1] = bufferIn[iIn + 1];
                bufferPadded[iPadded + 2] = bufferIn[iIn + 2];
                bufferPadded[iPadded + 3] = 0.0f;
            }
#ifdef USE_TBB
            });
#endif
            image->uploadData(bufferEntriesCount * sizeof(glm::vec4), bufferPadded);
        }
        image->uploadData(access.sizeInBytes, bufferCpu.get());
    }

    deviceFieldCache->push(access, deviceCacheEntry);
    return deviceCacheEntry;
}

std::pair<float, float> VolumeData::getMinMaxScalarFieldValue(
        const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    FieldAccess access = createFieldAccessStruct(FieldType::SCALAR, fieldName, timeStepIdx, ensembleIdx);

    if (fieldMinMaxCache->exists(access)) {
        return fieldMinMaxCache->get(access);
    }

    HostCacheEntry scalarValues = getFieldEntryCpu(FieldType::SCALAR, fieldName, timeStepIdx, ensembleIdx);
    auto minMaxVal = sgl::reduceFloatArrayMinMax(scalarValues.get(), getSlice3dEntryCount());
    fieldMinMaxCache->push(access, minMaxVal);
    return minMaxVal;
}

void VolumeData::update(float dtFrame) {
    multiVarTransferFunctionWindow.update(dt);
    hostFieldCache->updateEvictionWaitList();
    deviceFieldCache->updateEvictionWaitList();
}

void VolumeData::resetDirty() {
    if (dirty) {
        const auto& scalarFieldNames = typeToFieldNamesMap[FieldType::SCALAR];
        int numScalarFields = int(scalarFieldNames.size());
        for (int varIdx = 0; varIdx < numScalarFields; varIdx++) {
            multiVarTransferFunctionWindow.setAttributeDataDirty(varIdx);
        }
    }
    dirty = false;
}

void VolumeData::setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData) {
    if (renderData->getShaderStages()->hasDescriptorBinding(0, "transferFunctionTexture")) {
        const sgl::vk::DescriptorInfo& descriptorInfo = renderData->getShaderStages()->getDescriptorInfoByName(
                0, "transferFunctionTexture");
        if (descriptorInfo.image.arrayed == 1) {
            renderData->setStaticTexture(
                    multiVarTransferFunctionWindow.getTransferFunctionMapTextureVulkan(),
                    "transferFunctionTexture");
            renderData->setStaticBuffer(
                    multiVarTransferFunctionWindow.getMinMaxSsboVulkan(), "MinMaxBuffer");
        }
    }
}

void VolumeData::getPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines) {
    preprocessorDefines.insert(std::make_pair("USE_MULTI_VAR_TRANSFER_FUNCTION", ""));
}

void VolumeData::renderGui(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.beginNode("Volume Data")) {
        propertyEditor.addText(
                "Volume Size", std::to_string(xs) + "x" + std::to_string(ys) + "x" + std::to_string(zs));
        if (ts > 1 && propertyEditor.addSliderIntEdit(
                "Time Step", &currentTimeStepIdx, 0, ts - 1) == ImGui::EditMode::INPUT_FINISHED) {
            currentTimeStepIdx = std::clamp(currentTimeStepIdx, 0, ts - 1);
            dirty = true;
            reRender = true;
        }
        if (es > 1 && propertyEditor.addSliderIntEdit(
                "Ensemble Member", &currentEnsembleIdx, 0, es - 1) == ImGui::EditMode::INPUT_FINISHED) {
            currentEnsembleIdx = std::clamp(currentEnsembleIdx, 0, es - 1);
            dirty = true;
            reRender = true;
        }
        propertyEditor.addCheckbox("Render Color Legend", &shallRenderColorLegendWidgets);
        propertyEditor.endNode();
    }
}

void VolumeData::renderGuiCalculators(sgl::PropertyEditor& propertyEditor) {
    for (const CalculatorPtr& calculator : calculators) {
        if (!calculator->getShouldRenderGui()) {
            continue;
        }
        calculator->renderGui(propertyEditor);
        if (calculator->getHasNameChanged()) {
            updateCalculatorName(calculator);
            dirty = true;
            reRender = true;
        }
        if (calculator->getIsDirty()) {
            updateCalculator(calculator);
            dirty = true;
            reRender = true;
        }
    }
}

void VolumeData::renderGuiWindowSecondary() {
    if (multiVarTransferFunctionWindow.renderGui()) {
        reRender = true;
        if (multiVarTransferFunctionWindow.getTransferFunctionMapRebuilt()) {
            onTransferFunctionMapRebuilt();
            sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
                    ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }
    }
}

void VolumeData::renderGuiOverlay(uint32_t viewIdx) {
    if (shallRenderColorLegendWidgets) {
        int numWidgetsVisible = 0;
        for (size_t i = 0; i < colorLegendWidgets.size(); i++) {
            if (getIsTransferFunctionVisible(viewIdx, uint32_t(i))) {
                numWidgetsVisible++;
            }
        }
        int positionCounter = 0;
        for (size_t i = 0; i < colorLegendWidgets.size(); i++) {
            colorLegendWidgets.at(i).setPositionIndex(positionCounter, numWidgetsVisible);
            if (getIsTransferFunctionVisible(viewIdx, uint32_t(i))) {
                positionCounter++;
                colorLegendWidgets.at(i).setAttributeMinValue(
                        multiVarTransferFunctionWindow.getSelectedRangeMin(int(i)));
                colorLegendWidgets.at(i).setAttributeMaxValue(
                        multiVarTransferFunctionWindow.getSelectedRangeMax(int(i)));
                colorLegendWidgets.at(i).renderGui();
            }
        }
    }
}

void VolumeData::acquireTf(Renderer* renderer, int varIdx) {
    multiVarTransferFunctionWindow.loadAttributeDataIfEmpty(varIdx);
    transferFunctionToRendererMap.insert(std::make_pair(varIdx, renderer));
}

void VolumeData::releaseTf(Renderer* renderer, int varIdx) {
    auto iterRange = transferFunctionToRendererMap.equal_range(varIdx);
    auto it = iterRange.first;
    while (it != iterRange.second) {
        if (it->second == renderer) {
            transferFunctionToRendererMap.erase(it);
            break;
        }
        it++;
    }
}

bool VolumeData::getIsTransferFunctionVisible(uint32_t viewIdx, uint32_t varIdx) {
    auto iterRange = transferFunctionToRendererMap.equal_range(int(varIdx));
    auto it = iterRange.first;
    while (it != iterRange.second) {
        if (it->second->isVisibleInView(viewIdx)) {
            return true;
        }
        it++;
    }
    return false;
}

void VolumeData::onTransferFunctionMapRebuilt() {
    recomputeColorLegend();
}

void VolumeData::recomputeColorLegend() {
    for (size_t i = 0; i < colorLegendWidgets.size(); i++) {
        std::vector<sgl::Color16> transferFunctionColorMap =
                multiVarTransferFunctionWindow.getTransferFunctionMap_sRGB(int(i));
        colorLegendWidgets.at(i).setTransferFunctionColorMap(transferFunctionColorMap);
    }
}

void VolumeData::setClearColor(const sgl::Color& clearColor) {
    multiVarTransferFunctionWindow.setClearColor(clearColor);
    for (auto& colorLegendWidget : colorLegendWidgets) {
        colorLegendWidget.setClearColor(clearColor);
    }
}

void VolumeData::setUseLinearRGB(bool useLinearRGB) {
    multiVarTransferFunctionWindow.setUseLinearRGB(useLinearRGB);
}

void VolumeData::updateCalculatorName(const CalculatorPtr& calculator) {
    std::string oldFieldName;
    for (auto it = calculatorsHost.begin(); it != calculatorsHost.end(); it++) {
        if (it->second == calculator) {
            calculatorsHost.erase(it);
            calculatorsHost.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
        }
    }
    for (auto it = calculatorsDevice.begin(); it != calculatorsDevice.end(); it++) {
        if (it->second == calculator) {
            calculatorsDevice.erase(it);
            calculatorsDevice.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
        }
    }
    auto& fieldNames = typeToFieldNamesMap[calculator->getOutputFieldType()];
    int fieldNameIdx = 0;
    for (auto& fieldName : fieldNames) {
        if (fieldName == oldFieldName) {
            fieldName = calculator->getOutputFieldName();
            break;
        }
        fieldNameIdx++;
    }

    if (calculator->getOutputFieldType() == FieldType::SCALAR) {
        multiVarTransferFunctionWindow.updateAttributeName(fieldNameIdx, calculator->getOutputFieldName());
        colorLegendWidgets.at(fieldNameIdx).setAttributeDisplayName(calculator->getOutputFieldName());
    }
}

void VolumeData::updateCalculator(const CalculatorPtr& calculator) {
    hostFieldCache->removeEntriesForFieldName(calculator->getOutputFieldName());
    deviceFieldCache->removeEntriesForFieldName(calculator->getOutputFieldName());
    fieldMinMaxCache->removeEntriesForFieldName(calculator->getOutputFieldName());

    if (calculator->getOutputFieldType() == FieldType::SCALAR) {
        auto& fieldNames = typeToFieldNamesMap[calculator->getOutputFieldType()];
        int fieldNameIdx = 0;
        for (auto& fieldName : fieldNames) {
            if (fieldName == calculator->getOutputFieldName()) {
                break;
            }
            fieldNameIdx++;
        }
        multiVarTransferFunctionWindow.setAttributeDataDirty(fieldNameIdx);
    }
}

void VolumeData::setFileDialogInstance(ImGuiFileDialog* _fileDialogInstance) {
    this->fileDialogInstance = _fileDialogInstance;
}
