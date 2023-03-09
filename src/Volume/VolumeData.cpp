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

#include <queue>
#include <utility>
#include <cstring>

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
#include <ImGui/imgui_custom.h>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

#include "Loaders/DataSet.hpp"
#include "Loaders/VolumeLoader.hpp"
#include "Loaders/AmiraMeshLoader.hpp"
#include "Loaders/DatRawFileLoader.hpp"
#include "Loaders/FieldFileLoader.hpp"
#include "Loaders/CvolLoader.hpp"
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

#include "Export/NetCdfWriter.hpp"
#include "Export/CvolWriter.hpp"

#include "Calculators/Calculator.hpp"
#include "Calculators/VelocityCalculator.hpp"
#include "Calculators/BinaryOperatorCalculator.hpp"
#include "Calculators/NoiseReductionCalculator.hpp"
#include "Calculators/EnsembleVarianceCalculator.hpp"
#include "Calculators/CorrelationCalculator.hpp"
#ifdef SUPPORT_PYTORCH
#include "Calculators/PyTorchSimilarityCalculator.hpp"
#endif
#ifdef SUPPORT_TINY_CUDA_NN
#include "Calculators/TinyCudaNNSimilarityCalculator.hpp"
#endif
#ifdef SUPPORT_QUICK_MLP
#include "Calculators/QuickMLPSimilarityCalculator.hpp"
#endif
#include "Renderers/RenderingModes.hpp"
#include "Renderers/Renderer.hpp"
#include "Renderers/SceneData.hpp"
#include "Widgets/ViewManager.hpp"
#include "VolumeData.hpp"

template <typename T>
static std::pair<std::vector<std::string>, std::function<VolumeLoader*()>> registerVolumeLoader() {
    return { T::getSupportedExtensions(), []() { return new T{}; }};
}

VolumeLoader* VolumeData::createVolumeLoaderByExtension(const std::string& fileExtension) {
    auto it = factoriesLoader.find(fileExtension);
    if (it == factoriesLoader.end()) {
        sgl::Logfile::get()->writeError(
                "Error in VolumeData::createVolumeLoaderByExtension: Unsupported file extension '."
                + fileExtension + "'.", true);
        return nullptr;
    } else {
        return it->second();
    }
}

template <typename T>
static std::pair<std::vector<std::string>, std::function<VolumeWriter*()>> registerVolumeWriter() {
    return { T::getSupportedExtensions(), []() { return new T{}; }};
}

VolumeWriter* VolumeData::createVolumeWriterByExtension(const std::string& fileExtension) {
    auto it = factoriesWriter.find(fileExtension);
    if (it == factoriesWriter.end()) {
        sgl::Logfile::get()->throwError(
                "Error in VolumeData::createVolumeWriterByExtension: Unsupported file extension '."
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
    std::map<std::vector<std::string>, std::function<VolumeLoader*()>> factoriesLoaderMap = {
            registerVolumeLoader<AmiraMeshLoader>(),
            registerVolumeLoader<DatRawFileLoader>(),
            registerVolumeLoader<FieldFileLoader>(),
            registerVolumeLoader<CvolLoader>(),
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
    for (auto& factory : factoriesLoaderMap) {
        for (const std::string& extension : factory.first) {
            factoriesLoader.insert(std::make_pair(extension, factory.second));
        }
    }

    // Create the list of volume writers.
    std::map<std::vector<std::string>, std::function<VolumeWriter*()>> factoriesWriterMap = {
            registerVolumeWriter<NetCdfWriter>(),
            registerVolumeWriter<CvolWriter>(),
    };
    for (auto& factory : factoriesWriterMap) {
        for (const std::string& extension : factory.first) {
            factoriesWriter.insert(std::make_pair(extension, factory.second));
        }
    }

    // Create the list of calculators.
    factoriesCalculator.emplace_back(
            "Binary Operator", [renderer]() { return new BinaryOperatorCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "Noise Reduction", [renderer]() { return new NoiseReductionCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "Ensemble Variance", [renderer]() { return new EnsembleVarianceCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "Correlation Calculator", [renderer]() { return new CorrelationCalculator(renderer); });
#ifdef SUPPORT_PYTORCH
    factoriesCalculator.emplace_back(
            "PyTorch Similarity Calculator", [renderer]() { return new PyTorchSimilarityCalculator(renderer); });
#endif
#ifdef SUPPORT_CUDA_INTEROP
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
#ifdef SUPPORT_TINY_CUDA_NN
        factoriesCalculator.emplace_back(
                "tiny-cuda-nn Similarity Calculator", [renderer]() { return new TinyCudaNNSimilarityCalculator(renderer); });
#endif
#ifdef SUPPORT_QUICK_MLP
        factoriesCalculator.emplace_back(
                "QuickMLP Similarity Calculator", [renderer]() { return new QuickMLPSimilarityCalculator(renderer); });
#endif
    }
#endif

    sgl::vk::ImageSamplerSettings samplerSettings{};
    imageSampler = std::make_shared<sgl::vk::ImageSampler>(device, samplerSettings, 0.0f);
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

bool VolumeData::getIsScalarFieldDivergent(const std::string& fieldName) const {
    if (fieldName == "Helicity") {
        return true;
    }
    return false;
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
        if (!volumeLoader) {
            return false;
        }
        volumeLoader->setInputFiles(this, filePath, dataSetInformation);
        volumeLoaders.push_back(volumeLoader);
    }

    currentTimeStepIdx = std::clamp(dataSetInformation.standardTimeStepIdx, 0, ts - 1);
    if (!dataSetInformation.standardScalarFieldName.empty()) {
        const std::vector<std::string>& scalarFieldNames = getFieldNames(FieldType::SCALAR);
        auto it = std::find(
                scalarFieldNames.begin(), scalarFieldNames.end(), dataSetInformation.standardScalarFieldName);
        if (it != scalarFieldNames.end()) {
            standardScalarFieldIdx = int(it - scalarFieldNames.begin());
        } else {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::setInputFiles: Scalar field name \""
                    + dataSetInformation.standardScalarFieldName + "\" could not be found.");
        }
    } else {
        standardScalarFieldIdx = std::clamp(
                dataSetInformation.standardScalarFieldIdx, 0, int(getFieldNames(FieldType::SCALAR).size()) - 1);
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

    if (es > 1) {
        addCalculator(std::make_shared<CorrelationCalculator>(renderer));
#ifdef SUPPORT_PYTORCH
        addCalculator(std::make_shared<PyTorchSimilarityCalculator>(renderer));
#endif
#ifdef SUPPORT_CUDA_INTEROP
        if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
#ifdef SUPPORT_TINY_CUDA_NN
            addCalculator(std::make_shared<TinyCudaNNSimilarityCalculator>(renderer));
#endif
#ifdef SUPPORT_QUICK_MLP
            addCalculator(std::make_shared<QuickMLPSimilarityCalculator>(renderer));
#endif
        }
#endif
    }

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
    calculator->initialize();
    calculator->setCalculatorId(calculatorId++);
    calculator->setViewManager(viewManager);
    calculator->setFileDialogInstance(fileDialogInstance);
    calculator->setVolumeData(this, true);
    calculators.push_back(calculator);
    if (calculator->getFilterDevice() == FilterDevice::CPU) {
        calculatorsHost.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
    } else {
        calculatorsDevice.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
    }
    typeToFieldNamesMap[calculator->getOutputFieldType()].push_back(calculator->getOutputFieldName());

    if (!colorLegendWidgets.empty() && calculator->getOutputFieldType() == FieldType::SCALAR) {
        multiVarTransferFunctionWindow.addAttributeName(calculator->getOutputFieldName());
        colorLegendWidgets.emplace_back();
        colorLegendWidgets.back().setAttributeDisplayName(calculator->getOutputFieldName());
        std::vector<sgl::Color16> transferFunctionColorMap =
                multiVarTransferFunctionWindow.getTransferFunctionMap_sRGB(int(colorLegendWidgets.size()) - 1);
        colorLegendWidgets.back().setTransferFunctionColorMap(transferFunctionColorMap);
        for (auto& entry : scalarFieldToRendererMap) {
            entry.second->dirty = true;
        }
        for (size_t viewIdx = 0; viewIdx < viewManager->getNumViews(); viewIdx++) {
            auto calculatorRenderer = calculator->getCalculatorRenderer();
            if (!calculatorRenderer) {
                continue;
            }
            calculatorRenderer->addView(uint32_t(viewIdx));
            SceneData* viewSceneData = viewManager->getViewSceneData(uint32_t(viewIdx));
            if (*viewSceneData->sceneTexture) {
                calculatorRenderer->recreateSwapchainView(
                        uint32_t(viewIdx), *viewSceneData->viewportWidth, *viewSceneData->viewportHeight);
            }
        }
    }
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
        deviceEntry->getVulkanImage()->transitionImageLayout(
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
        deviceEntry->getVulkanImage()->copyToBuffer(stagingBuffer, renderer->getVkCommandBuffer());
        renderer->syncWithCpu();
        fieldEntryBuffer = new float[size_t(xs) * size_t(ys) * size_t(zs) * numComponents];
        void* data = stagingBuffer->mapMemory();
        memcpy(fieldEntryBuffer, data, sizeInBytes);
        stagingBuffer->unmapMemory();
        deviceEntry->getVulkanImage()->transitionImageLayout(
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
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
    imageSettings.usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT;
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

    auto itHost = calculatorsHost.find(fieldName);
    if (itHost != calculatorsHost.end() && itHost->second->getHasFixedRange()) {
        return itHost->second->getFixedRange();
    }
    auto itDevice = calculatorsDevice.find(fieldName);
    if (itDevice != calculatorsDevice.end() && itDevice->second->getHasFixedRange()) {
        return itDevice->second->getFixedRange();
    }

    if (fieldMinMaxCache->exists(access)) {
        return fieldMinMaxCache->get(access);
    }

    HostCacheEntry scalarValues = getFieldEntryCpu(FieldType::SCALAR, fieldName, timeStepIdx, ensembleIdx);
    auto minMaxVal = sgl::reduceFloatArrayMinMax(scalarValues.get(), getSlice3dEntryCount());

    // Is this a divergent scalar field? If yes, the transfer function etc. should be centered at zero.
    if (getIsScalarFieldDivergent(fieldName)) {
        float maxAbs = std::max(std::abs(minMaxVal.first), std::abs(minMaxVal.second));
        minMaxVal.first = -maxAbs;
        minMaxVal.second = maxAbs;
    }
    fieldMinMaxCache->push(access, minMaxVal);

    return minMaxVal;
}

void VolumeData::update(float dtFrame) {
    for (CalculatorPtr& calculator : calculators) {
        calculator->update(dtFrame);
    }
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
    std::queue<Calculator*> dirtyCalculators;
    for (auto& calculator : calculators) {
        if (!calculator->getShouldRenderGui()) {
            continue;
        }
        calculator->renderGui(propertyEditor);
        if (calculator->getIsDirtyDontReset()) {
            dirtyCalculators.push(calculator.get());
        }
    }
    while (!dirtyCalculators.empty()) {
        Calculator* calculator = dirtyCalculators.front();
        dirtyCalculators.pop();
        auto iterRange = calculatorUseMapRefToParent.equal_range(calculator);
        auto it = iterRange.first;
        while (it != iterRange.second) {
            Calculator* calculatorRef = it->second;
            if (!calculatorRef->getIsDirtyDontReset()) {
                calculatorRef->setIsDirty();
                dirtyCalculators.push(calculatorRef);
            }
            it++;
        }
    }

    for (int calculatorIdx = 0; calculatorIdx < int(calculators.size()); calculatorIdx++) {
        CalculatorPtr calculator = calculators.at(calculatorIdx);
        if (!calculator->getShouldRenderGui()) {
            continue;
        }
        if (calculator->getShallRemoveCalculator()) {
            removeCalculator(calculator, calculatorIdx);
            calculatorIdx--;
            dirty = true;
            reRender = true;
            continue;
        }
        if (calculator->getHasNameChanged()) {
            updateCalculatorName(calculator);
            dirty = true;
            reRender = true;
        }
        if (calculator->getHasFilterDeviceChanged()) {
            updateCalculatorFilterDevice(calculator);
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

void VolumeData::renderGuiNewCalculators() {
    for (auto& factory : factoriesCalculator) {
        if (ImGui::MenuItem(factory.first.c_str())) {
            addCalculator(CalculatorPtr(factory.second()));
        }
    }
}

void VolumeData::renderViewCalculator(uint32_t viewIdx) {
    auto varIdx = uint32_t(typeToFieldNamesMapBase[FieldType::SCALAR].size());
    for (const CalculatorPtr& calculator : calculators) {
        auto calculatorRenderer = calculator->getCalculatorRenderer();
        if (calculatorRenderer && getIsScalarFieldUsedInView(viewIdx, varIdx, calculator.get())) {
            calculatorRenderer->renderViewImpl(viewIdx);
        }
        if (calculator->getOutputFieldType() == FieldType::SCALAR) {
            varIdx++;
        }
    }
}

void VolumeData::addView(uint32_t viewIdx) {
    for (const CalculatorPtr& calculator : calculators) {
        auto calculatorRenderer = calculator->getCalculatorRenderer();
        if (!calculatorRenderer) {
            continue;
        }
        calculatorRenderer->addView(viewIdx);
    }
}

void VolumeData::removeView(uint32_t viewIdx) {
    for (const CalculatorPtr& calculator : calculators) {
        auto calculatorRenderer = calculator->getCalculatorRenderer();
        if (!calculatorRenderer) {
            continue;
        }
        calculatorRenderer->removeView(viewIdx);
    }
}

void VolumeData::recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height) {
    for (const CalculatorPtr& calculator : calculators) {
        auto calculatorRenderer = calculator->getCalculatorRenderer();
        if (!calculatorRenderer) {
            continue;
        }
        calculatorRenderer->recreateSwapchainView(viewIdx, width, height);
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

void VolumeData::acquireScalarField(Renderer* renderer, int varIdx) {
    scalarFieldToRendererMap.insert(std::make_pair(varIdx, renderer));
}

void VolumeData::releaseScalarField(Renderer* renderer, int varIdx) {
    auto iterRange = scalarFieldToRendererMap.equal_range(varIdx);
    auto it = iterRange.first;
    while (it != iterRange.second) {
        if (it->second == renderer) {
            scalarFieldToRendererMap.erase(it);
            break;
        }
        it++;
    }
}

void VolumeData::acquireScalarField(Calculator* calculator, int varIdx) {
    if (varIdx >= int(typeToFieldNamesMapBase[calculator->getOutputFieldType()].size())) {
        std::string calculatorRefName = typeToFieldNamesMap[calculator->getOutputFieldType()][varIdx];
        for (CalculatorPtr& calculatorRef : calculators) {
            if (calculatorRef->getOutputFieldName() == calculatorRefName) {
                calculatorUseMapRefToParent.insert(std::make_pair(calculatorRef.get(), calculator));
                calculatorUseMapParentToRef.insert(std::make_pair(calculator, calculatorRef.get()));
                break;
            }
        }
    }
}

void VolumeData::releaseScalarField(Calculator* calculator, int varIdx) {
    if (varIdx < int(typeToFieldNamesMapBase[calculator->getOutputFieldType()].size())) {
        return;
    }

    std::string calculatorRefName = typeToFieldNamesMap[calculator->getOutputFieldType()][varIdx];
    CalculatorPtr calculatorRef;
    for (CalculatorPtr& calcIt : calculators) {
        if (calcIt->getOutputFieldName() == calculatorRefName) {
            calculatorRef = calcIt;
            break;
        }
    }

    auto iterRangeParentToRef = calculatorUseMapParentToRef.equal_range(calculator);
    auto itParentToRef = iterRangeParentToRef.first;
    while (itParentToRef != iterRangeParentToRef.second) {
        if (itParentToRef->second == calculatorRef.get()) {
            calculatorUseMapParentToRef.erase(itParentToRef);
            break;
        }
        itParentToRef++;
    }

    auto iterRange = calculatorUseMapRefToParent.equal_range(calculatorRef.get());
    auto it = iterRange.first;
    while (it != iterRange.second) {
        if (it->second == calculator) {
            calculatorUseMapRefToParent.erase(it);
            break;
        }
        it++;
    }
}

bool VolumeData::getIsScalarFieldUsedInView(uint32_t viewIdx, uint32_t varIdx, Calculator* calculator) {
    auto iterRange = scalarFieldToRendererMap.equal_range(int(varIdx));
    auto itRend = iterRange.first;
    while (itRend != iterRange.second) {
        if (itRend->second->isVisibleInView(viewIdx)) {
            return true;
        }
        itRend++;
    }

    auto fieldBaseCount = int(typeToFieldNamesMapBase[calculator->getOutputFieldType()].size());
    auto fieldCount = int(typeToFieldNamesMap[calculator->getOutputFieldType()].size());
    auto& fieldNames = typeToFieldNamesMap[calculator->getOutputFieldType()];
    if (int(varIdx) >= fieldBaseCount) {
        auto iterRangeToRef = calculatorUseMapRefToParent.equal_range(calculator);
        auto it = iterRangeToRef.first;
        while (it != iterRangeToRef.second) {
            for (int varIdxNew = fieldBaseCount; varIdxNew < fieldCount; varIdxNew++) {
                if (it->second->getOutputFieldName() == fieldNames.at(varIdxNew)) {
                    bool isScalarFieldUsed = getIsScalarFieldUsedInView(viewIdx, varIdxNew, it->second);
                    if (isScalarFieldUsed) {
                        return true;
                    }
                }
            }
            std::string calculatorRefName = typeToFieldNamesMap[calculator->getOutputFieldType()][varIdx];
            it++;
        }
    }

    return false;
}

bool VolumeData::getIsScalarFieldUsedInAnyView(uint32_t varIdx, Calculator* calculator) {
    auto iterRange = scalarFieldToRendererMap.equal_range(int(varIdx));
    auto itRend = iterRange.first;
    while (itRend != iterRange.second) {
        if (itRend->second->isVisibleInAnyView()) {
            return true;
        }
        itRend++;
    }

    auto fieldBaseCount = int(typeToFieldNamesMapBase[calculator->getOutputFieldType()].size());
    auto fieldCount = int(typeToFieldNamesMap[calculator->getOutputFieldType()].size());
    auto& fieldNames = typeToFieldNamesMap[calculator->getOutputFieldType()];
    if (int(varIdx) >= fieldBaseCount) {
        auto iterRangeToRef = calculatorUseMapRefToParent.equal_range(calculator);
        auto it = iterRangeToRef.first;
        while (it != iterRangeToRef.second) {
            for (int varIdxNew = fieldBaseCount; varIdxNew < fieldCount; varIdxNew++) {
                if (it->second->getOutputFieldName() == fieldNames.at(varIdxNew)) {
                    bool isScalarFieldUsed = getIsScalarFieldUsedInAnyView(varIdxNew, it->second);
                    if (isScalarFieldUsed) {
                        return true;
                    }
                }
            }
            std::string calculatorRefName = typeToFieldNamesMap[calculator->getOutputFieldType()][varIdx];
            it++;
        }
    }

    return false;
}

uint32_t VolumeData::getVarIdxForCalculator(Calculator* calculator) {
    recomputeColorLegend();
    auto varIdx = uint32_t(typeToFieldNamesMapBase[calculator->getOutputFieldType()].size());
    for (CalculatorPtr& calculatorIt : calculators) {
        if (calculatorIt.get() == calculator) {
            return varIdx;
        }
        if (calculatorIt->getOutputFieldType() == calculator->getOutputFieldType()) {
            varIdx++;
        }
    }
    sgl::Logfile::get()->throwError("Error in VolumeData::getVarIdxForCalculator: Encountered unknown calculator.");
    return varIdx;
}

std::vector<std::shared_ptr<ICorrelationCalculator>> VolumeData::getCorrelationCalculatorsUsed() {
    std::vector<std::shared_ptr<ICorrelationCalculator>> correlationCalculators;
    auto varIdx = uint32_t(typeToFieldNamesMapBase[FieldType::SCALAR].size());
    for (CalculatorPtr& calculator : calculators) {
        if (calculator->getComputesCorrelation() && getIsScalarFieldUsedInAnyView(varIdx, calculator.get())) {
            correlationCalculators.push_back(std::static_pointer_cast<ICorrelationCalculator>(calculator));
        }
        if (calculator->getOutputFieldType() == FieldType::SCALAR) {
            varIdx++;
        }
    }
    return correlationCalculators;
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

size_t VolumeData::getNewCalculatorUseCount(CalculatorType calculatorType) {
    return ++calculatorTypeUseCounts[calculatorType];
}

void VolumeData::updateCalculatorName(const CalculatorPtr& calculator) {
    std::string oldFieldName;
    for (auto it = calculatorsHost.begin(); it != calculatorsHost.end(); it++) {
        if (it->second == calculator) {
            oldFieldName = it->first;
            calculatorsHost.erase(it);
            calculatorsHost.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
            break;
        }
    }
    for (auto it = calculatorsDevice.begin(); it != calculatorsDevice.end(); it++) {
        if (it->second == calculator) {
            oldFieldName = it->first;
            calculatorsDevice.erase(it);
            calculatorsDevice.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
            break;
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

void VolumeData::updateCalculatorFilterDevice(const CalculatorPtr& calculator) {
    auto itHost = calculatorsHost.find(calculator->getOutputFieldName());
    auto itDevice = calculatorsDevice.find(calculator->getOutputFieldName());
    if (itHost != calculatorsHost.end()) {
        calculatorsDevice.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
        calculatorsHost.erase(itHost);
    } else if (itDevice != calculatorsDevice.end()) {
        calculatorsHost.insert(std::make_pair(calculator->getOutputFieldName(), calculator));
        calculatorsDevice.erase(itDevice);
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
        if (fieldNameIdx == int(fieldNames.size())) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::updateCalculator: Invalid field name \""
                    + calculator->getOutputFieldName() + "\".");
        }
        multiVarTransferFunctionWindow.setAttributeDataDirty(fieldNameIdx);
    }
}

void VolumeData::removeCalculator(const CalculatorPtr& calculator, int calculatorIdx) {
    for (auto it = calculatorsHost.begin(); it != calculatorsHost.end(); it++) {
        if (it->second == calculator) {
            calculatorsHost.erase(it);
            break;
        }
    }
    for (auto it = calculatorsDevice.begin(); it != calculatorsDevice.end(); it++) {
        if (it->second == calculator) {
            calculatorsDevice.erase(it);
            break;
        }
    }
    auto& fieldNames = typeToFieldNamesMap[calculator->getOutputFieldType()];
    int fieldNameIdx = 0;
    for (auto& fieldName : fieldNames) {
        if (fieldName == calculator->getOutputFieldName()) {
            fieldName = calculator->getOutputFieldName();
            break;
        }
        fieldNameIdx++;
    }
    if (fieldNameIdx == int(fieldNames.size())) {
        sgl::Logfile::get()->throwError(
                "Error in VolumeData::removeCalculator: Invalid field name \""
                + calculator->getOutputFieldName() + "\".");
    }

    calculators.erase(calculators.begin() + calculatorIdx);
    fieldNames.erase(fieldNames.begin() + fieldNameIdx);

    auto oldCalculatorUseMapRefToParent = calculatorUseMapRefToParent;
    calculatorUseMapRefToParent.clear();
    for (auto& entry : oldCalculatorUseMapRefToParent) {
        if (entry.first != calculator.get() && entry.second != calculator.get()) {
            calculatorUseMapRefToParent.insert(entry);
        }
    }

    auto oldCalculatorUseMapParentToRef = calculatorUseMapParentToRef;
    calculatorUseMapParentToRef.clear();
    for (auto& entry : oldCalculatorUseMapParentToRef) {
        if (entry.first != calculator.get() && entry.second != calculator.get()) {
            calculatorUseMapParentToRef.insert(entry);
        }
    }

    if (calculator->getOutputFieldType() == FieldType::SCALAR) {
        multiVarTransferFunctionWindow.removeAttribute(fieldNameIdx);
        colorLegendWidgets.erase(colorLegendWidgets.begin() + fieldNameIdx);

        auto oldScalarFieldToRendererMap = scalarFieldToRendererMap;
        scalarFieldToRendererMap.clear();
        for (auto& entry : oldScalarFieldToRendererMap) {
            entry.second->dirty = true;
            entry.second->onFieldRemoved(calculator->getOutputFieldType(), fieldNameIdx);
            if (entry.first < fieldNameIdx) {
                scalarFieldToRendererMap.insert(entry);
            } else if (entry.first > fieldNameIdx) {
                scalarFieldToRendererMap.insert(std::make_pair(entry.first - 1, entry.second));
            }
        }

        auto oldTransferFunctionToRendererMap = transferFunctionToRendererMap;
        transferFunctionToRendererMap.clear();
        for (auto& entry : oldTransferFunctionToRendererMap) {
            if (entry.first < fieldNameIdx) {
                transferFunctionToRendererMap.insert(entry);
            } else if (entry.first > fieldNameIdx) {
                transferFunctionToRendererMap.insert(std::make_pair(entry.first - 1, entry.second));
            }
        }

        for (auto& calculatorIt : calculators) {
            calculatorIt->onFieldRemoved(calculator->getOutputFieldType(), fieldNameIdx);
        }
    }
}

void VolumeData::setFileDialogInstance(ImGuiFileDialog* _fileDialogInstance) {
    this->fileDialogInstance = _fileDialogInstance;
}

bool VolumeData::saveFieldToFile(const std::string& filePath, FieldType fieldType, int fieldIndex) {
    std::string filenameLower = boost::to_lower_copy(filePath);
    if (fieldType != FieldType::SCALAR) {
        sgl::Logfile::get()->writeError(
                "Error in VolumeData::saveFieldToFile: Currently, only the export of scalar fields is supported.");
        return false;
    }

    auto fieldData = getFieldEntryCpu(fieldType, typeToFieldNamesMap[fieldType].at(fieldIndex));
    std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(filePath);
    VolumeWriter* volumeWriter = createVolumeWriterByExtension(fileExtension);
    volumeWriter->writeFieldToFile(
            filePath, this, fieldType, typeToFieldNamesMap[fieldType].at(fieldIndex),
            currentTimeStepIdx, currentEnsembleIdx);
    delete volumeWriter;

    return false;
}

glm::vec3 VolumeData::screenPosToRayDir(SceneData* sceneData, int globalX, int globalY) {
    sgl::CameraPtr camera = sceneData->camera;
    uint32_t viewportWidth = *sceneData->viewportWidth;
    uint32_t viewportHeight = *sceneData->viewportHeight;
    int x = globalX - *sceneData->viewportPositionX;
    int y = globalY - *sceneData->viewportPositionY;

    glm::mat4 inverseViewMatrix = glm::inverse(camera->getViewMatrix());
    float scale = std::tan(camera->getFOVy() * 0.5f);
    glm::vec2 rayDirCameraSpace;
    rayDirCameraSpace.x = (2.0f * (float(x) + 0.5f) / float(viewportWidth) - 1.0f) * camera->getAspectRatio() * scale;
    rayDirCameraSpace.y = (2.0f * (float(viewportHeight - y - 1) + 0.5f) / float(viewportHeight) - 1.0f) * scale;
    glm::vec4 rayDirectionVec4 = inverseViewMatrix * glm::vec4(rayDirCameraSpace, -1.0, 0.0);
    glm::vec3 rayDirection = normalize(glm::vec3(rayDirectionVec4.x, rayDirectionVec4.y, rayDirectionVec4.z));
    return rayDirection;
}

bool VolumeData::pickPointScreen(
        SceneData* sceneData, int globalX, int globalY, glm::vec3& firstHit, glm::vec3& lastHit) const {
    glm::vec3 rayDirection = screenPosToRayDir(sceneData, globalX, globalY);
    return pickPointWorld(sceneData->camera->getPosition(), rayDirection, firstHit, lastHit);
}

bool VolumeData::pickPointScreenAtZ(
        SceneData* sceneData, int globalX, int globalY, int z, glm::vec3& hit) const {
    glm::vec3 rayDirection = screenPosToRayDir(sceneData, globalX, globalY);
    return pickPointWorldAtZ(sceneData->camera->getPosition(), rayDirection, z, hit);
}

bool VolumeData::pickPointWorld(
        const glm::vec3& cameraPosition, const glm::vec3& rayDirection, glm::vec3& firstHit, glm::vec3& lastHit) const {
    glm::vec3 rayOrigin = cameraPosition;
    float tNear, tFar;
    if (_rayBoxIntersection(rayOrigin, rayDirection, boxRendering.min, boxRendering.max, tNear, tFar)) {
        firstHit = rayOrigin + tNear * rayDirection;
        lastHit = rayOrigin + tFar * rayDirection;
        return true;
    }
    return false;
}

bool VolumeData::pickPointWorldAtZ(
        const glm::vec3& cameraPosition, const glm::vec3& rayDirection, int z, glm::vec3& hit) const {
    glm::vec3 rayOrigin = cameraPosition;
    float tHit;
    if (_rayZPlaneIntersection(
            rayOrigin, rayDirection,
            float(z) / float(zs - 1) * (boxRendering.max.z - boxRendering.min.z) + boxRendering.min.z,
            glm::vec2(boxRendering.min.x, boxRendering.min.y),
            glm::vec2(boxRendering.max.x, boxRendering.max.y),
            tHit)) {
        hit = rayOrigin + tHit * rayDirection;
        return true;
    }
    return false;
}

/**
 * Helper function for StreamlineTracingGrid::rayBoxIntersection (see below).
 */
bool VolumeData::_rayBoxPlaneIntersection(
        float rayOriginX, float rayDirectionX, float lowerX, float upperX, float& tNear, float& tFar) {
    if (std::abs(rayDirectionX) < 0.00001f) {
        // Ray is parallel to the x planes.
        if (rayOriginX < lowerX || rayOriginX > upperX) {
            return false;
        }
    } else {
        // Not parallel to the x planes. Compute the intersection distance to the planes.
        float t0 = (lowerX - rayOriginX) / rayDirectionX;
        float t1 = (upperX - rayOriginX) / rayDirectionX;
        if (t0 > t1) {
            // Since t0 intersection with near plane.
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        if (t0 > tNear) {
            // We want the largest tNear.
            tNear = t0;
        }
        if (t1 < tFar) {
            // We want the smallest tFar.
            tFar = t1;
        }
        if (tNear > tFar) {
            // Box is missed.
            return false;
        }
        if (tFar < 0) {
            // Box is behind ray.
            return false;
        }
    }
    return true;
}

/**
 * Implementation of ray-box intersection (idea from A. Glassner et al., "An Introduction to Ray Tracing").
 * For more details see: https://www.siggraph.org//education/materials/HyperGraph/raytrace/rtinter3.htm
 */
bool VolumeData::_rayBoxIntersection(
        const glm::vec3& rayOrigin, const glm::vec3& rayDirection, const glm::vec3& lower, const glm::vec3& upper,
        float& tNear, float& tFar) {
#ifdef TRACY_PROFILE_TRACING
    ZoneScoped;
#endif

    tNear = std::numeric_limits<float>::lowest();
    tFar = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; i++) {
        if (!_rayBoxPlaneIntersection(
                rayOrigin[i], rayDirection[i], lower[i], upper[i], tNear, tFar)) {
            return false;
        }
    }

    return true;
}

bool VolumeData::_rayZPlaneIntersection(
        const glm::vec3& rayOrigin, const glm::vec3& rayDirection, float z, glm::vec2 lowerXY, glm::vec2 upperXY,
        float& tHit) {
    if (std::abs(rayDirection.z) < 1e-5f) {
        // Plane and ray are parallel.
        return false;
    } else {
        tHit = (z - rayOrigin.z) / rayDirection.z;
        glm::vec3 hitPos = rayOrigin + tHit * rayDirection;
        if (hitPos.x >= lowerXY.x && hitPos.y >= lowerXY.y && hitPos.x <= upperXY.x && hitPos.y <= upperXY.y) {
            return true;
        }
    }
    return true;
}
