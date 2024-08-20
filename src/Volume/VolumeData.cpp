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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <Utils/Parallel/Reduction.hpp>
#endif

#include <Math/half/half.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/Events/EventManager.hpp>
#include <Utils/Parallel/Reduction.hpp>
#include <Graphics/Vulkan/Render/Data.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/Widgets/NumberFormatting.hpp>
#include <ImGui/imgui_custom.h>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

#include "Loaders/DataSet.hpp"
#include "Loaders/VolumeLoader.hpp"
#include "Loaders/AmiraMeshLoader.hpp"
#include "Loaders/DatRawFileLoader.hpp"
#include "Loaders/MhdRawFileLoader.hpp"
#include "Loaders/FieldFileLoader.hpp"
#include "Loaders/CvolLoader.hpp"
#include "Loaders/NiftiLoader.hpp"
#include "Loaders/CtlLoader.hpp"
#ifdef USE_ECCODES
#include "Loaders/GribLoader.hpp"
#endif
#include "Loaders/NetCdfLoader.hpp"
#ifdef USE_HDF5
#include "Loaders/Hdf5Loader.hpp"
#endif
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
#include "Calculators/EnsembleMeanCalculator.hpp"
#include "Calculators/EnsembleSpreadCalculator.hpp"
#include "Calculators/SetPredicateCalculator.hpp"
#include "Calculators/ResidualColorCalculator.hpp"
#include "Calculators/CorrelationCalculator.hpp"
#ifdef SUPPORT_PYTORCH
#include "Calculators/PyTorchCorrelationCalculator.hpp"
#endif
#ifdef SUPPORT_TINY_CUDA_NN
#include "Calculators/TinyCudaNNCorrelationCalculator.hpp"
#endif
#ifdef SUPPORT_QUICK_MLP
#include "Calculators/QuickMLPCorrelationCalculator.hpp"
#endif
#include "Calculators/VMLPCorrelationCalculator.hpp"
#include "Calculators/DKLCalculator.hpp"
#include "Renderers/RenderingModes.hpp"
#include "Renderers/Renderer.hpp"
#include "Renderers/SceneData.hpp"
#include "Widgets/ViewManager.hpp"
#include "CopyImageToBuffer.hpp"
#include "Histogram.hpp"
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
    typeToFieldUnitsMap.insert(std::make_pair(FieldType::SCALAR, std::vector<std::string>()));
    typeToFieldUnitsMap.insert(std::make_pair(FieldType::VECTOR, std::vector<std::string>()));

    // Create the list of volume loaders.
    std::map<std::vector<std::string>, std::function<VolumeLoader*()>> factoriesLoaderMap = {
            registerVolumeLoader<AmiraMeshLoader>(),
            registerVolumeLoader<DatRawFileLoader>(),
            registerVolumeLoader<MhdRawFileLoader>(),
            registerVolumeLoader<FieldFileLoader>(),
            registerVolumeLoader<CvolLoader>(),
            registerVolumeLoader<NiftiLoader>(),
            registerVolumeLoader<CtlLoader>(),
#ifdef USE_ECCODES
            registerVolumeLoader<GribLoader>(),
#endif
            registerVolumeLoader<NetCdfLoader>(),
#ifdef USE_HDF5
            registerVolumeLoader<Hdf5Loader>(),
#endif
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
            "Ensemble Mean", [renderer]() { return new EnsembleMeanCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "Ensemble Spread", [renderer]() { return new EnsembleSpreadCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "Set Predicate", [renderer]() { return new SetPredicateCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "Residual Color Calculator", [renderer]() { return new ResidualColorCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "Correlation Calculator", [renderer]() { return new CorrelationCalculator(renderer); });
#ifdef SUPPORT_PYTORCH
    factoriesCalculator.emplace_back(
            "PyTorch Similarity Calculator", [renderer]() { return new PyTorchCorrelationCalculator(renderer); });
#endif
#ifdef SUPPORT_CUDA_INTEROP
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
#ifdef SUPPORT_TINY_CUDA_NN
        factoriesCalculator.emplace_back(
                "tiny-cuda-nn Similarity Calculator", [renderer]() { return new TinyCudaNNCorrelationCalculator(renderer); });
#endif
#ifdef SUPPORT_QUICK_MLP
        factoriesCalculator.emplace_back(
                "QuickMLP Similarity Calculator", [renderer]() { return new QuickMLPCorrelationCalculator(renderer); });
#endif
    }
#endif
    factoriesCalculator.emplace_back(
            "VMLP Similarity Calculator", [renderer]() { return new VMLPCorrelationCalculator(renderer); });
    factoriesCalculator.emplace_back(
            "KL-Divergence Calculator", [renderer]() { return new DKLCalculator(renderer); });

    createImageSampler();

    renderRestrictionUniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, 4 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    imageToBufferCopyPass = std::make_shared<ImageToBufferCopyPass>(renderer);

    minMaxBufferWritePass = std::make_shared<MinMaxBufferWritePass>(renderer);
    for (int i = 0; i < 2; i++) {
        minMaxReductionPasses[i] = std::make_shared<MinMaxDepthReductionPass>(renderer);
    }
    computeHistogramPass = std::make_shared<ComputeHistogramPass>(renderer);
    computeHistogramMaxPass = std::make_shared<ComputeHistogramMaxPass>(renderer);
    computeHistogramDividePass = std::make_shared<ComputeHistogramDividePass>(renderer);
    divergentMinMaxPass = std::make_shared<DivergentMinMaxPass>(renderer);
}

VolumeData::~VolumeData() {
    for (VolumeLoader* volumeLoader : volumeLoaders) {
        delete volumeLoader;
    }
    volumeLoaders.clear();

    if (latData) {
        delete[] latData;
    }
    if (lonData) {
        delete[] lonData;
    }
    if (heightData) {
        delete[] heightData;
    }
}

void VolumeData::createImageSampler() {
    sgl::vk::ImageSamplerSettings samplerSettings{};
    if (textureInterpolationMode == TextureInterpolationMode::NEAREST) {
        samplerSettings.minFilter = VK_FILTER_NEAREST;
        samplerSettings.magFilter = VK_FILTER_NEAREST;
    } else if (textureInterpolationMode == TextureInterpolationMode::LINEAR) {
        samplerSettings.minFilter = VK_FILTER_LINEAR;
        samplerSettings.magFilter = VK_FILTER_LINEAR;
    }
    imageSampler = std::make_shared<sgl::vk::ImageSampler>(device, samplerSettings, 0.0f);
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
    if (separateFilesPerAttribute) {
        typeToFieldNamesMap[FieldType::SCALAR] = dataSetInformation.attributeNames;
        typeToFieldNamesMapBase[FieldType::SCALAR] = dataSetInformation.attributeNames;
    } else {
        typeToFieldNamesMap = fieldNamesMap;
        typeToFieldNamesMapBase = fieldNamesMap;
    }
}

void VolumeData::setFieldUnits(const std::unordered_map<FieldType, std::vector<std::string>>& fieldUnitsMap) {
    typeToFieldUnitsMap = fieldUnitsMap;
}

template<class T>
static void addFieldGlobal(
        T* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx,
        int xs, int ys, int zs, int ssxs, int ssys, int sszs, int subsamplingFactor,
        bool transpose, const glm::ivec3& transposeAxes, VolumeData* volumeData, HostFieldCache* hostFieldCache) {
    if (transpose) {
        if (transposeAxes != glm::ivec3(0, 2, 1)) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::addScalarField: At the moment, only transposing the "
                    "Y and Z axis is supported.");
        }
        if (fieldType == FieldType::SCALAR) {
            auto* scalarFieldCopy = new T[ssxs * ssys * sszs];
            if constexpr(std::is_same<T, HalfFloat>()) {
                size_t bufferSize = ssxs * ssys * sszs;
                for (size_t i = 0; i < bufferSize; i++) {
                    scalarFieldCopy[i] = fieldData[i];
                }
            } else {
                memcpy(scalarFieldCopy, fieldData, sizeof(T) * ssxs * ssys * sszs);
            }
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(ssxs, ssys, sszs, fieldData, scalarFieldCopy) default(none)
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
            auto* vectorFieldCopy = new T[3 * ssxs * ssys * sszs];
            if constexpr(std::is_same<T, HalfFloat>()) {
                size_t bufferSize = 3 * ssxs * ssys * sszs;
                for (size_t i = 0; i < bufferSize; i++) {
                    vectorFieldCopy[i] = fieldData[i];
                }
            } else {
                memcpy(vectorFieldCopy, fieldData, sizeof(T) * 3 * ssxs * ssys * sszs);
            }
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(ssxs, ssys, sszs, fieldData, vectorFieldCopy) default(none)
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
            T* scalarFieldOld = fieldData;
            fieldData = new T[xs * ys * zs];
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(xs, ys, zs, ssxs, ssys, subsamplingFactor, fieldData, scalarFieldOld) default(none)
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
            T* vectorFieldOld = fieldData;
            fieldData = new T[3 * xs * ys * zs];
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<int>(0, zs), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(xs, ys, zs, ssxs, ssys, subsamplingFactor, fieldData, vectorFieldOld) \
            default(none)
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

    FieldAccess access = volumeData->createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);

    if (hostFieldCache->exists(access)) {
        delete[] fieldData;
        return;
    }

    if constexpr(std::is_same<T, uint8_t>()) {
        access.sizeInBytes /= 4;
    } else if constexpr(std::is_same<T, uint16_t>()) {
        access.sizeInBytes /= 2;
    } else if constexpr(std::is_same<T, HalfFloat>()) {
        access.sizeInBytes /= 2;
    }

    size_t numEntries = 0;
    if (fieldType == FieldType::SCALAR) {
        numEntries = xs * ys * zs;
    } else if (fieldType == FieldType::VECTOR) {
        numEntries = 3 * size_t(xs * ys * zs);
    } else {
        sgl::Logfile::get()->throwError("Error in VolumeData::addField: Invalid field type.");
    }
    hostFieldCache->push(access, HostCacheEntry(new HostCacheEntryType(numEntries, fieldData)));
}

void VolumeData::addField(
        float* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, ssxs, ssys, sszs, subsamplingFactor, transpose, transposeAxes, this, hostFieldCache.get());
}

void VolumeData::addField(
        uint8_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, ssxs, ssys, sszs, subsamplingFactor, transpose, transposeAxes, this, hostFieldCache.get());
}

void VolumeData::addField(
        uint16_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, ssxs, ssys, sszs, subsamplingFactor, transpose, transposeAxes, this, hostFieldCache.get());
}

void VolumeData::addField(
        HalfFloat* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    addFieldGlobal(
            fieldData, fieldType, fieldName, timeStepIdx, ensembleIdx,
            xs, ys, zs, ssxs, ssys, sszs, subsamplingFactor, transpose, transposeAxes, this, hostFieldCache.get());
}

void VolumeData::addField(
        void* fieldData, ScalarDataFormat dataFormat, FieldType fieldType,
        const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    if (separateFilesPerAttribute) {
        std::string attributeName = typeToFieldNamesMapBase[FieldType::SCALAR][currentLoaderAttributeIdx];
        if (dataFormat == ScalarDataFormat::FLOAT) {
            addField(static_cast<float*>(fieldData), fieldType, attributeName, timeStepIdx, ensembleIdx);
        } else if (dataFormat == ScalarDataFormat::BYTE) {
            addField(static_cast<uint8_t*>(fieldData), fieldType, attributeName, timeStepIdx, ensembleIdx);
        } else if (dataFormat == ScalarDataFormat::SHORT) {
            addField(static_cast<uint16_t*>(fieldData), fieldType, attributeName, timeStepIdx, ensembleIdx);
        } else if (dataFormat == ScalarDataFormat::FLOAT16) {
            addField(static_cast<HalfFloat*>(fieldData), fieldType, attributeName, timeStepIdx, ensembleIdx);
        }
    } else {
        if (dataFormat == ScalarDataFormat::FLOAT) {
            addField(static_cast<float*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
        } else if (dataFormat == ScalarDataFormat::BYTE) {
            addField(static_cast<uint8_t*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
        } else if (dataFormat == ScalarDataFormat::SHORT) {
            addField(static_cast<uint16_t*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
        } else if (dataFormat == ScalarDataFormat::FLOAT16) {
            addField(static_cast<HalfFloat*>(fieldData), fieldType, fieldName, timeStepIdx, ensembleIdx);
        }
    }
}

const std::vector<std::string>& VolumeData::getFieldNames(FieldType fieldType) {
    if (fieldType == FieldType::SCALAR_OR_COLOR) {
        auto& scalarFieldNames = typeToFieldNamesMap[FieldType::SCALAR];
        fieldNamesMapColorOrScalar = typeToFieldNamesMap[FieldType::COLOR];
        for (const std::string& name : scalarFieldNames) {
            fieldNamesMapColorOrScalar.push_back(name);
        }
        return fieldNamesMapColorOrScalar;
    }
    return typeToFieldNamesMap[fieldType];
}

const std::vector<std::string>& VolumeData::getFieldNamesBase(FieldType fieldType) {
    return typeToFieldNamesMapBase[fieldType];
}

bool VolumeData::getIsColorField(int fieldIdx) {
    return fieldIdx < int(typeToFieldNamesMap[FieldType::COLOR].size());
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

void VolumeData::setLatLonData(float* _latData, float* _lonData) {
    if (latData) {
        delete[] latData;
        latData = nullptr;
    }
    latData = _latData;

    if (lonData) {
        delete[] lonData;
        lonData = nullptr;
    }
    lonData = _lonData;
}

void VolumeData::setHeightData(float* _heightData) {
    if (heightData) {
        delete[] heightData;
        heightData = nullptr;
    }
    heightData = _heightData;
}

bool VolumeData::setInputFiles(
        const std::vector<std::string>& _filePaths, DataSetInformation _dataSetInformation,
        glm::mat4* transformationMatrixPtr) {
    filePaths = _filePaths;
    dataSetInformation = std::move(_dataSetInformation);

    if (dataSetInformation.subsamplingFactorSet) {
        subsamplingFactor = dataSetInformation.subsamplingFactor;
    }
    if (dataSetInformation.axes != glm::ivec3(0, 1, 2)) {
        setTransposeAxes(dataSetInformation.axes);
    } else {
        transpose = false;
    }
    if (dataSetInformation.separateFilesPerAttribute) {
        separateFilesPerAttribute = dataSetInformation.separateFilesPerAttribute;
    }

    if (dataSetInformation.timeSteps.empty()) {
        ts = 1;
    } else {
        setTimeSteps(dataSetInformation.timeSteps);
    }
    tsFileCount = ts;
    if (separateFilesPerAttribute) {
        esFileCount = es = 1;
    } else {
        esFileCount = es = int(filePaths.size()) / tsFileCount;
    }

    for (size_t i = 0; i < filePaths.size(); i++) {
        std::string filePath = filePaths.at(i);
        std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(filePath);
        VolumeLoader* volumeLoader = createVolumeLoaderByExtension(fileExtension);
        if (!volumeLoader) {
            return false;
        }
        if (i > 0 && dataSetInformation.reuseMetadata && volumeLoader->getSupportsMetadataReuse()) {
            if (!volumeLoader->setMetadataFrom(volumeLoaders.front())) {
                delete volumeLoader;
                return false;
            }
        }
        if (!volumeLoader->setInputFiles(this, filePath, dataSetInformation)) {
            delete volumeLoader;
            return false;
        }
        volumeLoaders.push_back(volumeLoader);
    }

    currentTimeStepIdx = std::clamp(dataSetInformation.standardTimeStepIdx, 0, std::max(ts - 1, 0));
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
        if (standardScalarFieldIdx < 0) {
            standardScalarFieldIdx = 0;
        }
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

    // TODO: Add support for other vector fields?
    /*auto& scalarMap = typeToFieldNamesMap[FieldType::SCALAR];
    for (auto& scalarMapEntry : scalarMap) {
        if (sgl::endsWith(scalarMapEntry, ".x")) {
            std::string nameRaw = scalarMapEntry.substr(0, scalarMapEntry.size() - 2);
            if (getFieldExists(FieldType::SCALAR, nameRaw + ".y")
                    && getFieldExists(FieldType::SCALAR, nameRaw + ".z")) {
                addCalculator(std::make_shared<VelocityCalculator>(renderer));
                addCalculator(std::make_shared<VectorMagnitudeCalculator>(renderer, nameRaw));
                addCalculator(std::make_shared<VorticityCalculator>(renderer));
                addCalculator(std::make_shared<VectorMagnitudeCalculator>(renderer, nameRaw));
                addCalculator(std::make_shared<HelicityCalculator>(renderer));
            }
        }
    }*/

    bool hasScalarData = false;
    {
        auto it = typeToFieldNamesMap.find(FieldType::SCALAR);
        if (it != typeToFieldNamesMap.end() && !it->second.empty()) {
            hasScalarData = true;
        }
    }
    if ((ts > 1 || es > 1) && hasScalarData && viewManager) {
        addCalculator(std::make_shared<CorrelationCalculator>(renderer));
/*#ifdef SUPPORT_PYTORCH
        addCalculator(std::make_shared<PyTorchCorrelationCalculator>(renderer));
#endif
#ifdef SUPPORT_CUDA_INTEROP
        if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
            && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
#ifdef SUPPORT_TINY_CUDA_NN
            addCalculator(std::make_shared<TinyCudaNNCorrelationCalculator>(renderer));
#endif
#ifdef SUPPORT_QUICK_MLP
            addCalculator(std::make_shared<QuickMLPCorrelationCalculator>(renderer));
#endif
        }
#endif*/
    }

    const auto& scalarFieldNames = typeToFieldNamesMap[FieldType::SCALAR];
    const auto& fieldsUnits = typeToFieldUnitsMap[FieldType::SCALAR];
    multiVarTransferFunctionWindow.setRequestAttributeValuesCallback([this](
            int varIdx, const void** values, ScalarDataFormat* fmt, size_t& numValues, float& minValue, float& maxValue) {
        std::string fieldName = typeToFieldNamesMap[FieldType::SCALAR].at(varIdx);
        if (values && fmt) {
            HostCacheEntry cacheEntry = this->getFieldEntryCpu(FieldType::SCALAR, fieldName);
            if (cacheEntry->getScalarDataFormatNative() == ScalarDataFormat::FLOAT) {
                *values = cacheEntry->data<float>();
                *fmt = ScalarDataFormat::FLOAT;
            } else if (cacheEntry->getScalarDataFormatNative() == ScalarDataFormat::BYTE) {
                *values = cacheEntry->data<uint8_t>();
                *fmt = ScalarDataFormat::BYTE;
            } else if (cacheEntry->getScalarDataFormatNative() == ScalarDataFormat::SHORT) {
                *values = cacheEntry->data<uint16_t>();
                *fmt = ScalarDataFormat::SHORT;
            } else if (cacheEntry->getScalarDataFormatNative() == ScalarDataFormat::FLOAT16) {
                // TODO: Change once support for float16 has been added to sgl.
                *values = cacheEntry->data<float>();
                *fmt = ScalarDataFormat::FLOAT;
            }
        }

        numValues = this->getSlice3dEntryCount();
        std::tie(minValue, maxValue) = getMinMaxScalarFieldValue(fieldName);
    });

    multiVarTransferFunctionWindow.setRequestHistogramCallback([this](
            int varIdx, int histSize, std::vector<float>& histogram, float& selectedRangeMin, float& selectedRangeMax,
            float& dataRangeMin, float& dataRangeMax, bool recomputeMinMax, bool isSelectedRangeFixed) {
        // If the data is resident on the GPU, calculate the histogram manually.
        std::string fieldName = typeToFieldNamesMap[FieldType::SCALAR].at(varIdx);
        if (!this->getIsGpuResidentOrGpuCalculator(FieldType::SCALAR, fieldName)
                || varIdx < int(typeToFieldNamesMapBase[FieldType::SCALAR].size())) {
            // For now, we also exclude primary fields. In theory, all non-float32 fields should be excluded,
            // but currently no calculator will output anything but float32 values.
            return false;
        }

        if (recomputeMinMax) {
            auto itHost = calculatorsHost.find(fieldName);
            if (itHost != calculatorsHost.end() && itHost->second->getHasFixedRange()) {
                std::tie(dataRangeMin, dataRangeMax) = itHost->second->getFixedRange();
                recomputeMinMax = false;
            }
            auto itDevice = calculatorsDevice.find(fieldName);
            if (itDevice != calculatorsDevice.end() && itDevice->second->getHasFixedRange()) {
                std::tie(dataRangeMin, dataRangeMax) = itDevice->second->getFixedRange();
                recomputeMinMax = false;
            }
            if (!recomputeMinMax && getIsScalarFieldDivergent(fieldName)) {
                float maxAbs = std::max(std::abs(dataRangeMin), std::abs(dataRangeMax));
                dataRangeMin = -maxAbs;
                dataRangeMax = maxAbs;
            }
            if (!recomputeMinMax && !isSelectedRangeFixed) {
                selectedRangeMin = dataRangeMin;
                selectedRangeMax = dataRangeMax;
            }
        }

        /*FieldAccess access = createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);
        access.isImageData = false;
        access.bufferTileSize = glm::uvec3(1, 1, 1);
        if (deviceFieldCache->exists(access)) {
            return deviceFieldCache->reaccess(access);
        }*/
        DeviceCacheEntry deviceEntry = this->getFieldEntryDevice(FieldType::SCALAR, fieldName);
        const size_t sizeInBytes = getSlice3dSizeInBytes(FieldType::SCALAR, ScalarDataFormat::FLOAT);
        sgl::vk::BufferPtr dataBuffer;
        if (deviceEntry->getVulkanBuffer()) {
            dataBuffer = deviceEntry->getVulkanBuffer();
        } else if (deviceEntry->getVulkanImage()) {
            if (!imageDataCacheBuffer || imageDataCacheBuffer->getSizeInBytes() != sizeInBytes) {
                imageDataCacheBuffer = std::make_shared<sgl::vk::Buffer>(
                        device, sizeInBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_GPU_ONLY);
            }
            deviceEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
            deviceEntry->getVulkanImage()->copyToBuffer(imageDataCacheBuffer, renderer->getVkCommandBuffer());
            deviceEntry->getVulkanImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
            renderer->insertBufferMemoryBarrier(
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    imageDataCacheBuffer);
            dataBuffer = imageDataCacheBuffer;
        }

        if (!minMaxValueBuffer) {
            minMaxValueBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, 2 * sizeof(float),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
        }
        auto bufferSizeTmp = sgl::uiceil(uint32_t(this->getSlice3dEntryCount()), 256);
        for (int i = 0; i < 2; i++) {
            minMaxReductionBuffers[i] = std::make_shared<sgl::vk::Buffer>(
                    device, bufferSizeTmp * sizeof(glm::vec2),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            bufferSizeTmp = sgl::uiceil(bufferSizeTmp, 512);
        }
        if (!maxHistogramValueBuffer) {
            maxHistogramValueBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, sizeof(float),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }
        if (!histogramUintBuffer || histogramUintBuffer->getSizeInBytes() < sizeof(uint32_t) * histSize) {
            histogramUintBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, sizeof(uint32_t) * histSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
            histogramFloatBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, sizeof(float) * histSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        }

        minMaxBufferWritePass->setBuffers(dataBuffer, minMaxReductionBuffers[0]);
        for (int i = 0; i < 2; i++) {
            minMaxReductionPasses[i]->setBuffers(minMaxReductionBuffers[i], minMaxReductionBuffers[(i + 1) % 2]);
        }
        computeHistogramPass->setBuffers(minMaxValueBuffer, dataBuffer, uint32_t(histSize), histogramUintBuffer);
        computeHistogramMaxPass->setBuffers(maxHistogramValueBuffer, uint32_t(histSize), histogramUintBuffer);
        computeHistogramDividePass->setBuffers(
                maxHistogramValueBuffer, uint32_t(histSize), histogramUintBuffer, histogramFloatBuffer);
        divergentMinMaxPass->setBuffers(minMaxValueBuffer);

        int idxOut = 0;
        if (recomputeMinMax) {
            uint32_t numBlocks = sgl::uiceil(uint32_t(this->getSlice3dEntryCount()), 256);
            minMaxBufferWritePass->render();
            renderer->insertBufferMemoryBarrier(
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    minMaxReductionBuffers[0]);

            int iteration = 0;
            while (numBlocks > 1) {
                renderer->insertMemoryBarrier(
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                minMaxReductionPasses[iteration]->setInputSize(numBlocks);
                minMaxReductionPasses[iteration]->render();
                numBlocks = sgl::uiceil(numBlocks, 512);
                iteration++;
            }
            idxOut = iteration % 2;
            renderer->insertBufferMemoryBarrier(
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    minMaxReductionBuffers[idxOut]);
            if (!isSelectedRangeFixed) {
                minMaxReductionBuffers[idxOut]->copyDataTo(
                        minMaxValueBuffer, 0, 0, 2 * sizeof(float), renderer->getVkCommandBuffer());
            }
            if (getIsScalarFieldDivergent(fieldName)) {
                renderer->insertBufferMemoryBarrier(
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        minMaxValueBuffer);
                divergentMinMaxPass->render();
                renderer->insertBufferMemoryBarrier(
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        minMaxValueBuffer);
            }
        }
        if (isSelectedRangeFixed || !recomputeMinMax) {
            glm::vec2 minMaxData(selectedRangeMin, selectedRangeMax);
            minMaxValueBuffer->updateData(2 * sizeof(float), &minMaxData.x, renderer->getVkCommandBuffer());
        }
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                minMaxValueBuffer);

        histogramUintBuffer->fill(0, renderer->getVkCommandBuffer());
        maxHistogramValueBuffer->fill(0, renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        computeHistogramPass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                histogramUintBuffer);
        computeHistogramMaxPass->render();
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                maxHistogramValueBuffer);
        computeHistogramDividePass->render();
        renderer->insertMemoryBarrier(
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

        size_t stagingBufferSize = histSize * sizeof(float) + 2 * sizeof(float);
        ensureStagingBufferExists(stagingBufferSize);
        histogramFloatBuffer->copyDataTo(
                stagingBuffer, 0, 0, histSize * sizeof(float), renderer->getVkCommandBuffer());
        if (recomputeMinMax) {
            minMaxReductionBuffers[idxOut]->copyDataTo(
                    stagingBuffer, 0, histSize * sizeof(float), 2 * sizeof(float), renderer->getVkCommandBuffer());
        }
        renderer->syncWithCpu();

        auto* data = reinterpret_cast<float*>(stagingBuffer->mapMemory());
        histogram.resize(histSize);
        memcpy(histogram.data(), data, sizeof(float) * histSize);
        if (recomputeMinMax) {
            data += histSize;
            dataRangeMin = data[0];
            dataRangeMax = data[1];
            if (getIsScalarFieldDivergent(fieldName)) {
                float maxAbs = std::max(std::abs(dataRangeMin), std::abs(dataRangeMax));
                dataRangeMin = -maxAbs;
                dataRangeMax = maxAbs;
            }
            if (!isSelectedRangeFixed) {
                selectedRangeMin = dataRangeMin;
                selectedRangeMax = dataRangeMax;
            }
        }
        stagingBuffer->unmapMemory();

        if (recomputeMinMax) {
            FieldAccess access = createFieldAccessStruct(
                    FieldType::SCALAR, fieldName, currentTimeStepIdx, currentEnsembleIdx);
            fieldMinMaxCache->push(access, std::make_pair(dataRangeMin, dataRangeMax));
        }

        return true;
    });

    multiVarTransferFunctionWindow.setAttributeNames(scalarFieldNames, standardScalarFieldIdx);

    colorLegendWidgets.clear();
    colorLegendWidgets.resize(scalarFieldNames.size());
    for (size_t i = 0; i < colorLegendWidgets.size(); i++) {
        colorLegendWidgets.at(i).setPositionIndex(0, 1);
        std::string attributeDisplayName = scalarFieldNames.at(i);
        std::string fieldUnits = i < fieldsUnits.size() ? fieldsUnits.at(i) : "";
        if (!fieldUnits.empty()) {
            attributeDisplayName += " [" + fieldUnits + "]";
        }
        colorLegendWidgets.at(i).setAttributeDisplayName(attributeDisplayName);
    }
    recomputeColorLegend();

    return true;
}

void VolumeData::addCalculator(const CalculatorPtr& calculator) {
    calculator->initialize();
    calculator->setCalculatorId(calculatorId++);
    if (viewManager) {
        calculator->setViewManager(viewManager);
    }
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
        colorLegendWidgets.back().setClearColor(cachedClearColor);
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
                        uint32_t(viewIdx), *viewSceneData->viewportWidthVirtual, *viewSceneData->viewportHeightVirtual);
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

bool VolumeData::getScalarFieldSupportsBufferMode(int scalarFieldIdx) {
    const auto& scalarFieldNames = getFieldNames(FieldType::SCALAR);
    const auto& scalarFieldNamesBase = getFieldNamesBase(FieldType::SCALAR);
    if (scalarFieldIdx < int(scalarFieldNames.size()) && scalarFieldIdx >= int(scalarFieldNamesBase.size())) {
        const std::string& fieldName = scalarFieldNames.at(scalarFieldIdx);
        auto itCalc = calculatorsDevice.find(fieldName);
        return itCalc == calculatorsDevice.end() || itCalc->second->getSupportsBufferOutput();
    }
    return volumeLoaders.front()->getHasFloat32Data();
}

template<class T>
static void transposeScalarField(T* fieldEntryBuffer, int ssxs, int ssys, int sszs) {
    auto* scalarFieldCopy = new T[ssxs * ssys * sszs];
    if constexpr(std::is_same<T, HalfFloat>()) {
        size_t bufferSize = ssxs * ssys * sszs;
        for (size_t i = 0; i < bufferSize; i++) {
            scalarFieldCopy[i] = fieldEntryBuffer[i];
        }
    } else {
        memcpy(scalarFieldCopy, fieldEntryBuffer, sizeof(T) * ssxs * ssys * sszs);
    }
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
        for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
    #pragma omp parallel for shared(ssxs, ssys, sszs, fieldEntryBuffer, scalarFieldCopy) default(none)
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
}

template<class T>
static void transposeVectorField(T* fieldEntryBuffer, int ssxs, int ssys, int sszs) {
    auto* vectorFieldCopy = new T[3 * ssxs * ssys * sszs];
    if constexpr(std::is_same<T, HalfFloat>()) {
        size_t bufferSize = 3 * ssxs * ssys * sszs;
        for (size_t i = 0; i < bufferSize; i++) {
            vectorFieldCopy[i] = fieldEntryBuffer[i];
        }
    } else {
        memcpy(vectorFieldCopy, fieldEntryBuffer, sizeof(T) * 3 * ssxs * ssys * sszs);
    }
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, sszs), [&](auto const& r) {
        for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
#pragma omp parallel for shared(ssxs, ssys, sszs, fieldEntryBuffer, vectorFieldCopy) default(none)
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

bool VolumeData::getIsGpuResidentOrGpuCalculator(
        FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    FieldAccess access = createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);
    auto itCalc = calculatorsDevice.find(fieldName);
    return itCalc != calculatorsDevice.end() || deviceFieldCache->exists(access);
}

void VolumeData::ensureStagingBufferExists(size_t sizeInBytes) {
    if (!stagingBuffer || stagingBuffer->getSizeInBytes() < sizeInBytes) {
        stagingBuffer = std::make_shared<sgl::vk::Buffer>(
                device, sizeInBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);
    }
}

VolumeData::HostCacheEntry VolumeData::getFieldEntryCpu(
        FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx) {
    FieldAccess access = createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);

    if (hostFieldCache->exists(access)) {
        return hostFieldCache->reaccess(access);
    }

    size_t bufferSize = getSlice3dSizeInBytes(fieldType);
    hostFieldCache->ensureSufficientMemory(bufferSize);

    HostCacheEntryType* fieldEntry = nullptr;
    auto itCalc = calculatorsHost.find(fieldName);
    if (itCalc != calculatorsHost.end()) {
        Calculator* calculator = itCalc->second.get();
        if (calculator->getOutputFieldType() != fieldType) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::getFieldEntryCpu: Mismatch between field type and calculator output for "
                    "field \"" + fieldName + "\".");
        }
        size_t numComponents = fieldType == FieldType::SCALAR ? 1 : 3;
        size_t numEntries = size_t(xs) * size_t(ys) * size_t(zs) * numComponents;
        auto* fieldEntryBuffer = new float[numEntries];
        calculator->calculateCpu(timeStepIdx, ensembleIdx, fieldEntryBuffer);
        fieldEntry = new HostCacheEntryType(numEntries, fieldEntryBuffer);
    } else if (calculatorsDevice.find(fieldName) != calculatorsDevice.end()) {
        auto deviceEntry = getFieldEntryDevice(fieldType, fieldName, timeStepIdx, ensembleIdx);
        size_t numComponents = fieldType == FieldType::SCALAR ? 1 : 3;
        size_t numEntries = numComponents * size_t(xs) * size_t(ys) * size_t(zs);
        size_t sizeInBytes = numEntries * sizeof(float);
        ensureStagingBufferExists(sizeInBytes);
        deviceEntry->getVulkanImage()->transitionImageLayout(
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
        deviceEntry->getVulkanImage()->copyToBuffer(stagingBuffer, renderer->getVkCommandBuffer());
        renderer->syncWithCpu();
        auto* fieldEntryBuffer = new float[numEntries];
        void* data = stagingBuffer->mapMemory();
        memcpy(fieldEntryBuffer, data, sizeInBytes);
        stagingBuffer->unmapMemory();
        deviceEntry->getVulkanImage()->transitionImageLayout(
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, renderer->getVkCommandBuffer());
        fieldEntry = new HostCacheEntryType(numEntries, fieldEntryBuffer);
    } else {
        VolumeLoader* volumeLoader = nullptr;
        if (separateFilesPerAttribute) {
            auto& names = typeToFieldNamesMapBase[fieldType];
            auto itName = std::find(names.begin(), names.end(), fieldName);
            currentLoaderAttributeIdx = int(itName - names.begin());
            volumeLoader = volumeLoaders.at(currentLoaderAttributeIdx);
        } else if (tsFileCount == 1 && esFileCount == 1) {
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

        if (!volumeLoader->getFieldEntry(this, fieldType, fieldName, timeStepIdx, ensembleIdx, fieldEntry)) {
            sgl::Logfile::get()->throwError(
                    "Fatal error in VolumeData::getFieldEntryCpu: Volume loader failed to load the data entry.");
            return {};
        }
    }

    // Loaders may load multiple fields at once and leave fieldEntryBuffer empty.
    if (!fieldEntry) {
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
                if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::FLOAT) {
                    transposeScalarField(fieldEntry->dataFloat, ssxs, ssys, sszs);
                } else if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::BYTE) {
                    transposeScalarField(fieldEntry->dataByte, ssxs, ssys, sszs);
                } else if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::SHORT) {
                    transposeScalarField(fieldEntry->dataShort, ssxs, ssys, sszs);
                } else if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::FLOAT16) {
                    transposeScalarField(fieldEntry->dataFloat16, ssxs, ssys, sszs);
                }
            } else {
                if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::FLOAT) {
                    transposeVectorField(fieldEntry->dataFloat, ssxs, ssys, sszs);
                } else if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::BYTE) {
                    transposeVectorField(fieldEntry->dataByte, ssxs, ssys, sszs);
                } else if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::SHORT) {
                    transposeVectorField(fieldEntry->dataShort, ssxs, ssys, sszs);
                } else if (fieldEntry->getScalarDataFormatNative() == ScalarDataFormat::FLOAT16) {
                    transposeVectorField(fieldEntry->dataFloat16, ssxs, ssys, sszs);
                }
            }
        }
    }

    auto buffer = HostCacheEntry(fieldEntry);
    hostFieldCache->push(access, buffer);
    return buffer;
}

VolumeData::DeviceCacheEntry VolumeData::allocDeviceCacheEntryImage(
        FieldType fieldType, ScalarDataFormat scalarDataFormat) {
    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = uint32_t(xs);
    imageSettings.height = uint32_t(ys);
    imageSettings.depth = uint32_t(zs);
    imageSettings.imageType = VK_IMAGE_TYPE_3D;
    if (scalarDataFormat == ScalarDataFormat::FLOAT) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    } else if (scalarDataFormat == ScalarDataFormat::BYTE) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    } else if (scalarDataFormat == ScalarDataFormat::SHORT) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R16_UNORM : VK_FORMAT_R16G16B16A16_UNORM;
    } else if (scalarDataFormat == ScalarDataFormat::FLOAT16) {
        imageSettings.format = fieldType == FieldType::SCALAR ? VK_FORMAT_R16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    imageSettings.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageSettings.usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT;
    imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;

#ifdef SUPPORT_CUDA_INTEROP
    bool canUseCuda = sgl::vk::getIsCudaDeviceApiFunctionTableInitialized();
#else
    bool canUseCuda = false;
#endif
    imageSettings.exportMemory = canUseCuda;

    /*
     * https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkMemoryGetWin32HandleInfoKHR.html
     * "If handleType is defined as an NT handle, vkGetMemoryWin32HandleKHR must be called no more than once for each valid
     * unique combination of memory and handleType"
     * TODO: On Windows the application can only use non-dedicated allocations when it has been figured out that the handle
     * is only exported once.
     */
#ifdef _WIN32
    imageSettings.useDedicatedAllocationForExportedMemory = true;
#else
    imageSettings.useDedicatedAllocationForExportedMemory = false;
#endif

    auto image = std::make_shared<sgl::vk::Image>(device, imageSettings);
    auto deviceCacheEntry = std::make_shared<DeviceCacheEntry::element_type>(image, imageSampler);
    return deviceCacheEntry;
}

VolumeData::DeviceCacheEntry VolumeData::allocDeviceCacheEntryBuffer(
        size_t& bufferSize, FieldAccess& access,
        bool tileBufferMemory, uint32_t tileSizeX, uint32_t tileSizeY, uint32_t tileSizeZ) {
#ifdef SUPPORT_CUDA_INTEROP
    bool canUseCuda = sgl::vk::getIsCudaDeviceApiFunctionTableInitialized();
#else
    bool canUseCuda = false;
#endif
    if (tileBufferMemory) {
        bufferSize =
                4 * size_t(sgl::uiceil(uint32_t(xs), tileSizeX) * tileSizeX)
                * size_t(sgl::uiceil(uint32_t(ys), tileSizeY) * tileSizeY)
                * size_t(sgl::uiceil(uint32_t(zs), tileSizeZ) * tileSizeZ);
        access.sizeInBytes = bufferSize;
    }
    VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    auto buffer = std::make_shared<sgl::vk::Buffer>(
            device, bufferSize, bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY, true, canUseCuda, false);
    auto deviceCacheEntry = std::make_shared<DeviceCacheEntry::element_type>(
            buffer, tileSizeX, tileSizeY, tileSizeZ);
    return deviceCacheEntry;
}

void VolumeData::copyCacheEntryImageToBuffer(
        VolumeData::DeviceCacheEntry& deviceCacheEntryImage, VolumeData::DeviceCacheEntry& deviceCacheEntryBuffer) {
    auto buffer = deviceCacheEntryBuffer->getVulkanBuffer();
    auto tileSize = deviceCacheEntryBuffer->getBufferTileSize();
    if (tileSize == glm::uvec3(1, 1, 1)) {
        auto image = deviceCacheEntryImage->getVulkanImage();
        image->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
        image->copyToBuffer(buffer, renderer->getVkCommandBuffer());
        renderer->insertImageMemoryBarrier(
                image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    } else {
        auto image = deviceCacheEntryImage->getVulkanImageView();
        image->transitionImageLayout(VK_IMAGE_LAYOUT_GENERAL, renderer->getVkCommandBuffer());
        imageToBufferCopyPass->setData(xs, ys, zs, image, buffer, tileSize);
        imageToBufferCopyPass->render();
    }
    renderer->syncWithCpu();
    if (tileSize != glm::uvec3(1, 1, 1)) {
        imageToBufferCopyPass->resetData();
    }
}

VolumeData::DeviceCacheEntry VolumeData::getFieldEntryDevice(
        FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx,
        bool wantsImageData, const glm::uvec3& bufferTileSize) {
    if (fieldType == FieldType::SCALAR_OR_COLOR) {
        auto& fieldNamesColor = typeToFieldNamesMap[FieldType::COLOR];
        if (std::find(fieldNamesColor.begin(), fieldNamesColor.end(), fieldName) == fieldNamesColor.end()) {
            fieldType = FieldType::SCALAR;
        } else {
            fieldType = FieldType::COLOR;
        }
    }
    FieldAccess access = createFieldAccessStruct(fieldType, fieldName, timeStepIdx, ensembleIdx);
    access.isImageData = wantsImageData;
    access.bufferTileSize = bufferTileSize;

    if (deviceFieldCache->exists(access)) {
        return deviceFieldCache->reaccess(access);
    }

    auto itCalc = calculatorsDevice.find(fieldName);
    HostCacheEntry bufferCpu;
    ScalarDataFormat scalarDataFormat = ScalarDataFormat::FLOAT;
    if (itCalc == calculatorsDevice.end()) {
        bufferCpu = getFieldEntryCpu(fieldType, fieldName, timeStepIdx, ensembleIdx);
        scalarDataFormat = bufferCpu->getScalarDataFormatNative();
        if (scalarDataFormat == ScalarDataFormat::BYTE) {
            access.sizeInBytes /= 4;
        } else if (scalarDataFormat == ScalarDataFormat::SHORT || scalarDataFormat == ScalarDataFormat::FLOAT16) {
            access.sizeInBytes /= 2;
        }
    }
    size_t bufferSize = getSlice3dSizeInBytes(fieldType, scalarDataFormat);
    deviceFieldCache->ensureSufficientMemory(bufferSize);

    bool tileBufferMemory = bufferTileSize.x > 1 || bufferTileSize.y > 1 || bufferTileSize.z > 1;
    const uint32_t tileSizeX = bufferTileSize.x;
    const uint32_t tileSizeY = bufferTileSize.y;
    const uint32_t tileSizeZ = bufferTileSize.z;

    DeviceCacheEntry deviceCacheEntry;
    if (wantsImageData) {
        deviceCacheEntry = allocDeviceCacheEntryImage(fieldType, scalarDataFormat);
    } else {
        deviceCacheEntry = allocDeviceCacheEntryBuffer(
                bufferSize, access, tileBufferMemory, tileSizeX, tileSizeY, tileSizeZ);
    }

    if (itCalc != calculatorsDevice.end()) {
        Calculator* calculator = itCalc->second.get();
        if (calculator->getOutputFieldType() != fieldType) {
            sgl::Logfile::get()->throwError(
                    "Error in VolumeData::getFieldEntryDevice: Mismatch between field type and calculator output for "
                    "field \"" + fieldName + "\".");
        }

        // All calculators so far output their data to images.
        if (!wantsImageData && !calculator->getSupportsBufferOutput()) {
            auto deviceCacheEntryImage = allocDeviceCacheEntryImage(fieldType, scalarDataFormat);
            calculator->calculateDevice(timeStepIdx, ensembleIdx, deviceCacheEntryImage);
            copyCacheEntryImageToBuffer(deviceCacheEntryImage, deviceCacheEntry);
        } else {
            calculator->calculateDevice(timeStepIdx, ensembleIdx, deviceCacheEntry);
        }
    } else if (wantsImageData) {
        auto& image = deviceCacheEntry->getVulkanImage();
        if (fieldType == FieldType::SCALAR || fieldType == FieldType::COLOR) {
            image->uploadData(access.sizeInBytes, bufferCpu->getDataNative());
        } else if (fieldType == FieldType::VECTOR) {
            size_t bufferEntriesCount = size_t(xs) * size_t(ys) * size_t(zs);
            if (scalarDataFormat == ScalarDataFormat::FLOAT) {
                const float* bufferIn = bufferCpu->data<float>();
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
                image->uploadData(bufferEntriesCount * 4 * sizeof(float), bufferPadded);
                delete[] bufferPadded;
            } else if (scalarDataFormat == ScalarDataFormat::BYTE) {
                const uint8_t* bufferIn = bufferCpu->data<uint8_t>();
                auto* bufferPadded = new uint8_t[bufferEntriesCount * 4];
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
                    bufferPadded[iPadded + 3] = 0;
                }
#ifdef USE_TBB
                });
#endif
                image->uploadData(bufferEntriesCount * 4 * sizeof(uint8_t), bufferPadded);
                delete[] bufferPadded;
            } else if (scalarDataFormat == ScalarDataFormat::SHORT) {
                const uint16_t* bufferIn = bufferCpu->data<uint16_t>();
                auto* bufferPadded = new uint16_t[bufferEntriesCount * 4];
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
                    bufferPadded[iPadded + 3] = 0;
                }
#ifdef USE_TBB
                });
#endif
                image->uploadData(bufferEntriesCount * 4 * sizeof(uint16_t), bufferPadded);
                delete[] bufferPadded;
            } else if (scalarDataFormat == ScalarDataFormat::FLOAT16) {
                const HalfFloat* bufferIn = bufferCpu->data<HalfFloat>();
                auto* bufferPadded = new HalfFloat[bufferEntriesCount * 4];
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
                    bufferPadded[iPadded + 3] = HalfFloat(0.0f);
                }
#ifdef USE_TBB
                });
#endif
                image->uploadData(bufferEntriesCount * 4 * sizeof(HalfFloat), bufferPadded);
                delete[] bufferPadded;
            }
        }
        access.sizeInBytes = image->getDeviceMemoryAllocationSize();
    } else {
        auto& buffer = deviceCacheEntry->getVulkanBuffer();
        if (tileBufferMemory) {
            size_t numEntries = bufferSize / 4;
            auto* linearBufferData = bufferCpu->getDataFloat();
            auto* tiledBufferData = new float[numEntries];
            auto tileNumVoxels = tileSizeX * tileSizeY * tileSizeZ;
            uint32_t xst = sgl::uiceil(xs, tileSizeX);
            uint32_t yst = sgl::uiceil(ys, tileSizeY);
            uint32_t zst = sgl::uiceil(zs, tileSizeZ);
            uint32_t numTilesTotal = xst * yst * zst;
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<uint32_t>(0, numTilesTotal), [&](auto const& r) {
                for (auto tileIdx = r.begin(); tileIdx != r.end(); tileIdx++) {
#else
#if _OPENMP >= 200805
            #pragma omp parallel for default(none) shared(numTilesTotal, tileNumVoxels, xst, yst, zst) \
            shared(tileSizeX, tileSizeY, tileSizeZ, linearBufferData, tiledBufferData)
#endif
            for (uint32_t tileIdx = 0; tileIdx < numTilesTotal; tileIdx++) {
#endif
                uint32_t xt = tileIdx % xst;
                uint32_t yt = (tileIdx / xst) % yst;
                uint32_t zt = tileIdx / (xst * yst);
                for (uint32_t voxelIdx = 0; voxelIdx < tileNumVoxels; voxelIdx++) {
                    uint32_t vx = voxelIdx % tileSizeX;
                    uint32_t vy = (voxelIdx / tileSizeX) % tileSizeY;
                    uint32_t vz = voxelIdx / (tileSizeX * tileSizeY);
                    uint32_t x = vx + xt * tileSizeX;
                    uint32_t y = vy + yt * tileSizeY;
                    uint32_t z = vz + zt * tileSizeZ;
                    float value = 0.0f;
                    if (x < uint32_t(xs) && y < uint32_t(ys) && z < uint32_t(zs)) {
                        value = linearBufferData[IDXS(x, y, z)];
                    }
                    tiledBufferData[tileIdx * tileNumVoxels + voxelIdx] = value;
                }
            }
#ifdef USE_TBB
            });
#endif
            buffer->uploadData(access.sizeInBytes, tiledBufferData);
            delete[] tiledBufferData;
        } else {
            buffer->uploadData(access.sizeInBytes, bufferCpu->getDataNative());
        }
        access.sizeInBytes = buffer->getDeviceMemoryAllocationSize();
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

    std::pair<float, float> minMaxVal;
    HostCacheEntry scalarValues = getFieldEntryCpu(FieldType::SCALAR, fieldName, timeStepIdx, ensembleIdx);
    if (scalarValues->getScalarDataFormatNative() == ScalarDataFormat::FLOAT) {
        minMaxVal = sgl::reduceFloatArrayMinMax(scalarValues->data<float>(), getSlice3dEntryCount());
    } else if (scalarValues->getScalarDataFormatNative() == ScalarDataFormat::BYTE) {
        minMaxVal = sgl::reduceUnormByteArrayMinMax(scalarValues->data<uint8_t>(), getSlice3dEntryCount());
    } else if (scalarValues->getScalarDataFormatNative() == ScalarDataFormat::SHORT) {
        minMaxVal = sgl::reduceUnormShortArrayMinMax(scalarValues->data<uint16_t>(), getSlice3dEntryCount());
    } else if (scalarValues->getScalarDataFormatNative() == ScalarDataFormat::FLOAT16) {
        minMaxVal = sgl::reduceHalfFloatArrayMinMax(scalarValues->data<HalfFloat>(), getSlice3dEntryCount());
    }

    // Is this a divergent scalar field? If yes, the transfer function etc. should be centered at zero.
    if (getIsScalarFieldDivergent(fieldName)) {
        float maxAbs = std::max(std::abs(minMaxVal.first), std::abs(minMaxVal.second));
        minMaxVal.first = -maxAbs;
        minMaxVal.second = maxAbs;
    }
    fieldMinMaxCache->push(access, minMaxVal);

    return minMaxVal;
}

AuxiliaryMemoryToken VolumeData::pushAuxiliaryMemoryCpu(size_t sizeInBytes) {
    return hostFieldCache->pushAuxiliaryMemory(sizeInBytes);
}

void VolumeData::popAuxiliaryMemoryCpu(AuxiliaryMemoryToken token) {
    hostFieldCache->popAuxiliaryMemory(token);
}

AuxiliaryMemoryToken VolumeData::pushAuxiliaryMemoryDevice(size_t sizeInBytes) {
    return deviceFieldCache->pushAuxiliaryMemory(sizeInBytes);
}

void VolumeData::popAuxiliaryMemoryDevice(AuxiliaryMemoryToken token) {
    deviceFieldCache->popAuxiliaryMemory(token);
}


bool VolumeData::getHasLatLonData() {
    return latData && lonData;
}

void VolumeData::getLatLonData(const float*& _latData, const float*& _lonData) {
    _latData = latData;
    _lonData = lonData;
}

void VolumeData::displayLayerInfo(sgl::PropertyEditor& propertyEditor, int zPlaneCoord) {
    zPlaneCoord = std::clamp(zPlaneCoord, 0, zs - 1);
    if (heightData) {
        propertyEditor.addText("Z Level Height", getHeightString(heightData[zPlaneCoord]));
    }
}

bool VolumeData::getHasHeightData() const {
    return heightData != nullptr;
}

float VolumeData::getHeightDataForZ(int z) const {
    z = std::clamp(z, 0, zs - 1);
    return heightData[z];
}

float VolumeData::getHeightDataForZWorld(float zWorld) const {
    // Opposite of: float(z) / float(zs > 1 ? zs - 1 : 1) * (boxRendering.max.z - boxRendering.min.z) + boxRendering.min.z
    float z = (zWorld - boxRendering.min.z) / (boxRendering.max.z - boxRendering.min.z) * float(zs - 1);
    float zFloor = std::floor(z);
    auto factor = z - zFloor;
    auto h0 = getHeightDataForZ(int(zFloor));
    auto h1 = getHeightDataForZ(int(zFloor) + 1);
    return glm::mix(h0, h1, factor);
}

std::string VolumeData::getHeightString(float height) const {
    return sgl::getNiceNumberString(height, 5) + "m";
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
        const auto& scalarFieldNamesBase = typeToFieldNamesMapBase[FieldType::SCALAR];
        const auto& scalarFieldNames = typeToFieldNamesMap[FieldType::SCALAR];
        // 2023-03-30: Make sure no data is reset for base fields to avoid recomputing on every calculator change.
        int startIdx = isFirstDirty ? int(0) : int(scalarFieldNamesBase.size());
        int numScalarFields = int(scalarFieldNames.size());
        for (auto& calculator : calculators) {
            calculator->setVolumeData(this, false);
        }
        for (int varIdx = startIdx; varIdx < numScalarFields; varIdx++) {
            if (isFirstDirty && varIdx == standardScalarFieldIdx) {
                continue;
            }
            multiVarTransferFunctionWindow.setAttributeDataDirty(varIdx);
        }
    }
    dirty = false;
    isFirstDirty = false;
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
    if (renderData->getShaderStages()->hasDescriptorBinding(0, "RenderRestrictionUniformBuffer")) {
        renderData->setStaticBuffer(
                renderRestrictionUniformBuffer, "RenderRestrictionUniformBuffer");
    }
}

void VolumeData::getPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines) {
    preprocessorDefines.insert(std::make_pair("USE_MULTI_VAR_TRANSFER_FUNCTION", ""));
    if (useRenderRestriction) {
        preprocessorDefines.insert(std::make_pair("USE_RENDER_RESTRICTION", ""));
        if (renderRestrictionDistanceMetric == DistanceMetric::EUCLIDEAN) {
            preprocessorDefines.insert(std::make_pair("RENDER_RESTRICTION_EUCLIDEAN_DISTANCE", ""));
        } else if (renderRestrictionDistanceMetric == DistanceMetric::CHEBYSHEV) {
            preprocessorDefines.insert(std::make_pair("RENDER_RESTRICTION_CHEBYSHEV_DISTANCE", ""));
        }
    }
}

void VolumeData::setBaseFieldsDirty() {
    dirty = true;
    reRender = true;
    auto numFieldsBase = int(typeToFieldNamesMapBase[FieldType::SCALAR].size());
    for (int fieldIdx = 0; fieldIdx < numFieldsBase; fieldIdx++) {
        multiVarTransferFunctionWindow.setAttributeDataDirty(fieldIdx);
    }
}

void VolumeData::setCurrentTimeStepIdx(int newTimeStepIdx) {
    if (currentTimeStepIdx != newTimeStepIdx) {
        setBaseFieldsDirty();
    }
    currentTimeStepIdx = std::clamp(newTimeStepIdx, 0, std::max(ts - 1, 0));
}

void VolumeData::setCurrentEnsembleIdx(int newEnsembleIdx) {
    if (currentEnsembleIdx != newEnsembleIdx) {
        setBaseFieldsDirty();
    }
    currentEnsembleIdx = std::clamp(newEnsembleIdx, 0, std::max(es - 1, 0));
}

void VolumeData::renderGui(sgl::PropertyEditor& propertyEditor) {
    if (propertyEditor.beginNode("Volume Data")) {
        propertyEditor.addText(
                "Volume Size", std::to_string(xs) + "x" + std::to_string(ys) + "x" + std::to_string(zs));
        if (ts > 1 && propertyEditor.addSliderIntEdit(
                "Time Step", &currentTimeStepIdx, 0, ts - 1) == ImGui::EditMode::INPUT_FINISHED) {
            currentTimeStepIdx = std::clamp(currentTimeStepIdx, 0, std::max(ts - 1, 0));
            setBaseFieldsDirty();
        }
        if (es > 1 && propertyEditor.addSliderIntEdit(
                "Ensemble Member", &currentEnsembleIdx, 0, es - 1) == ImGui::EditMode::INPUT_FINISHED) {
            currentEnsembleIdx = std::clamp(currentEnsembleIdx, 0, std::max(es - 1, 0));
            setBaseFieldsDirty();
        }
        if (propertyEditor.addCombo(
                "Texture Interpolation", (int*)&textureInterpolationMode,
                TEXTURE_INTERPOLATION_MODE_NAMES, IM_ARRAYSIZE(TEXTURE_INTERPOLATION_MODE_NAMES))) {
            createImageSampler();
            deviceFieldCache->doForEach([this](const DeviceCacheEntry& cacheEntry) {
                cacheEntry->setImageSampler(imageSampler);
            });
            dirty = true;
            reRender = true;
        }
        propertyEditor.addCheckbox("Render Color Legend", &shallRenderColorLegendWidgets);
        propertyEditor.endNode();
    }

    // Moved from VolumeData::setRenderRestriction, as command buffer is not valid in VolumeData::update.
    if (useRenderRestriction && renderRestrictionUniformBufferDirty) {
        renderRestrictionUniformBuffer->updateData(
                sizeof(glm::vec4), &renderRestriction, renderer->getVkCommandBuffer());
        renderer->insertBufferMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                renderRestrictionUniformBuffer);
        renderRestrictionUniformBufferDirty = false;
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

    std::vector<Calculator*> calculatorResetTfWindowList;
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
            calculatorResetTfWindowList.push_back(calculator.get());
            dirty = true;
            reRender = true;
        }
    }

    // The code below was moved outside updateCalculator to avoid the use of invalid cache resources in
    // setAttributeDataDirty.
    if (!calculatorResetTfWindowList.empty()) {
        /*
         * 2024-05-02: In case of a dirty calculator, we should prepare the visualization pipeline first before
         * multiVarTransferFunctionWindow.setAttributeDataDirty may trigger a recompute using invalid data.
         */
        prepareVisualizationPipelineCallback();
    }
    auto& fieldNames = typeToFieldNamesMap[FieldType::SCALAR];
    for (Calculator* calculator : calculatorResetTfWindowList) {
        if (calculator->getOutputFieldType() == FieldType::SCALAR) {
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
}

void VolumeData::renderGuiNewCalculators() {
    for (auto& factory : factoriesCalculator) {
        if (ImGui::MenuItem(factory.first.c_str())) {
            addCalculator(CalculatorPtr(factory.second()));
            dirty = true; //< Necessary for new transfer function map.
            reRender = true;
        }
    }
}

void VolumeData::renderViewCalculator(uint32_t viewIdx) {
    auto varIdx = uint32_t(typeToFieldNamesMapBase[FieldType::SCALAR].size());
    for (const CalculatorPtr& calculator : calculators) {
        auto calculatorRenderer = calculator->getCalculatorRenderer();
        if (calculatorRenderer && getIsScalarFieldUsedInView(viewIdx, varIdx, calculator.get())
                && calculator->getIsCalculatorRendererEnabled()) {
            calculatorRenderer->renderViewImpl(viewIdx);
        }
        if (calculator->getOutputFieldType() == FieldType::SCALAR) {
            varIdx++;
        }
    }
}

void VolumeData::renderViewCalculatorPostOpaque(uint32_t viewIdx) {
    auto varIdx = uint32_t(typeToFieldNamesMapBase[FieldType::SCALAR].size());
    for (const CalculatorPtr& calculator : calculators) {
        auto calculatorRenderer = calculator->getCalculatorRenderer();
        if (calculatorRenderer && getIsScalarFieldUsedInView(viewIdx, varIdx, calculator.get())
                && calculator->getIsCalculatorRendererEnabled()) {
            calculatorRenderer->renderViewPostOpaqueImpl(viewIdx);
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
    // 2023-03-30: We don't need a reload if the range is fixed (avoid recomputing on every calculator change).
    if (!multiVarTransferFunctionWindow.getIsSelectedRangeFixed(varIdx)) {
        multiVarTransferFunctionWindow.loadAttributeDataIfEmpty(varIdx);
    }
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

void VolumeData::acquireTf(Calculator* renderer, int varIdx) {
    if (!multiVarTransferFunctionWindow.getIsSelectedRangeFixed(varIdx)) {
        multiVarTransferFunctionWindow.loadAttributeDataIfEmpty(varIdx);
    }
    // Only load the TF on calling acquireTf for now.
}

void VolumeData::releaseTf(Calculator* renderer, int varIdx) {
    // Only load the TF on calling acquireTf for now.
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

const std::vector<CalculatorPtr>& VolumeData::getCalculators() {
    return calculators;
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
    for (auto& calculator : calculators) {
        if (calculator->getUseTransferFunction()) {
            auto fieldCount = int(typeToFieldNamesMap[FieldType::SCALAR].size());
            for (int varIdx = 0; varIdx < fieldCount; varIdx++) {
                if (calculator->getUsesScalarFieldIdx(varIdx)
                        && multiVarTransferFunctionWindow.getIsVariableDirty(varIdx)) {
                    calculator->setIsDirty();
                    break;
                }
            }
        }
    }
    multiVarTransferFunctionWindow.resetDirty();
}

void VolumeData::recomputeColorLegend() {
    for (size_t i = 0; i < colorLegendWidgets.size(); i++) {
        std::vector<sgl::Color16> transferFunctionColorMap =
                multiVarTransferFunctionWindow.getTransferFunctionMap_sRGB(int(i));
        colorLegendWidgets.at(i).setTransferFunctionColorMap(transferFunctionColorMap);
    }
}

void VolumeData::setClearColor(const sgl::Color& clearColor) {
    cachedClearColor = clearColor;
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

    if (renderRestrictionCalculator == calculator.get()) {
        renderRestrictionCalculator = nullptr;
        useRenderRestriction = false;
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

void VolumeData::setRenderRestriction(
        Calculator* calculator, DistanceMetric distanceMetric, const glm::vec3& position, float radius) {
    if (!renderRestrictionCalculator) {
        renderRestrictionCalculator = calculator;
        useRenderRestriction = true;
        shallReloadRendererShaders = true;
    }
    glm::vec4 uniformData = glm::vec4(position, radius);
    if (uniformData != renderRestriction) {
        renderRestriction = uniformData;
        renderRestrictionUniformBufferDirty = true;
        reRender = true;
    }
    if (distanceMetric != renderRestrictionDistanceMetric) {
        renderRestrictionDistanceMetric = distanceMetric;
        shallReloadRendererShaders = true;
    }
}

void VolumeData::resetRenderRestriction(Calculator* calculator) {
    if (renderRestrictionCalculator == calculator) {
        renderRestrictionCalculator = nullptr;
        useRenderRestriction = false;
        shallReloadRendererShaders = true;
    }
}

void VolumeData::setFileDialogInstance(ImGuiFileDialog* _fileDialogInstance) {
    this->fileDialogInstance = _fileDialogInstance;
}

bool VolumeData::saveFieldToFile(const std::string& filePath, FieldType fieldType, int fieldIndex) {
    std::string filenameLower = sgl::toLowerCopy(filePath);
    if (fieldType != FieldType::SCALAR) {
        sgl::Logfile::get()->writeError(
                "Error in VolumeData::saveFieldToFile: Currently, only the export of scalar fields is supported.");
        return false;
    }

    auto fieldData = getFieldEntryCpu(fieldType, typeToFieldNamesMap[fieldType].at(fieldIndex));
    std::string fileExtension = sgl::FileUtils::get()->getFileExtensionLower(filePath);
    VolumeWriter* volumeWriter = createVolumeWriterByExtension(fileExtension);
    bool retVal = volumeWriter->writeFieldToFile(
            filePath, this, fieldType, typeToFieldNamesMap[fieldType].at(fieldIndex),
            currentTimeStepIdx, currentEnsembleIdx);
    delete volumeWriter;

    return retVal;
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
            float(z) / float(zs > 1 ? zs - 1 : 1) * (boxRendering.max.z - boxRendering.min.z) + boxRendering.min.z,
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
