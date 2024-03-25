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

#ifndef CORRERENDER_VOLUMEDATA_HPP
#define CORRERENDER_VOLUMEDATA_HPP

#include <vector>
#include <list>
#include <map>
#include <unordered_set>
#include <string>
#include <memory>
#include <cstdint>

#include <glm/vec3.hpp>

#include <Math/Geometry/AABB3.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>
#include <ImGui/Widgets/ColorLegendWidget.hpp>

#include "Loaders/DataSetList.hpp"
#include "Cache/FieldCache.hpp"
#include "Cache/DeviceCacheEntry.hpp"
#include "Cache/HostCacheEntry.hpp"
#include "DistanceMetrics.hpp"
#include "FieldType.hpp"
#include "FieldAccess.hpp"

namespace sgl {
class PropertyEditor;
namespace vk {
class RenderData;
typedef std::shared_ptr<RenderData> RenderDataPtr;
class Renderer;
}}

namespace IGFD {
class FileDialog;
}
typedef IGFD::FileDialog ImGuiFileDialog;

struct SceneData;
class MainApp;
class ViewManager;
class VolumeLoader;
class VolumeWriter;
class Renderer;
class Calculator;
typedef std::shared_ptr<Calculator> CalculatorPtr;
enum class CalculatorType : uint32_t;
class ICorrelationCalculator;

class ImageToBufferCopyPass;
class DivergentMinMaxPass;
class MinMaxBufferWritePass;
class MinMaxDepthReductionPass;
class ComputeHistogramPass;
class ComputeHistogramMaxPass;
class ComputeHistogramDividePass;
class DivergentMinMaxPass;

class VolumeData {
    friend class MainApp;
public:
    using HostCacheEntry = std::shared_ptr<HostCacheEntryType>;
    using DeviceCacheEntry = std::shared_ptr<DeviceCacheEntryType>;

public:
    explicit VolumeData(sgl::vk::Renderer* renderer);
    ~VolumeData();

    /**
     * Called every frame
     * @param dtFrame The elapsed time in seconds since the last frame.
     */
    void update(float dtFrame);

    /**
     * Renders the entries in the property editor.
     * @return true if the gather shader needs to be reloaded.
     */
    void renderGui(sgl::PropertyEditor& propertyEditor);
    void renderGuiCalculators(sgl::PropertyEditor& propertyEditor);
    void renderGuiNewCalculators();
    void renderViewCalculator(uint32_t viewIdx);
    void renderViewCalculatorPostOpaque(uint32_t viewIdx);
    /**
     * For rendering secondary ImGui windows (e.g., for transfer function widgets).
     * @return true if the gather shader needs to be reloaded.
     */
    void renderGuiWindowSecondary();
    /**
     * For rendering secondary, overlay ImGui windows.
     * @return true if the gather shader needs to be reloaded.
     */
    void renderGuiOverlay(uint32_t viewIdx);

    /// Certain GUI widgets might need the clear color.
    virtual void setClearColor(const sgl::Color& clearColor);
    /// Whether to use linear RGB when rendering.
    virtual void setUseLinearRGB(bool useLinearRGB);
    inline sgl::MultiVarTransferFunctionWindow& getMultiVarTransferFunctionWindow() { return multiVarTransferFunctionWindow; }
    inline void setViewManager(ViewManager* _viewManager) { viewManager = _viewManager; }
    void addView(uint32_t viewIdx);
    void removeView(uint32_t viewIdx);
    void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height);

    void setTransposeAxes(const glm::ivec3& axes);
    void setGridSubsamplingFactor(int factor);
    void setGridExtent(int _xs, int _ys, int _zs, float _dx, float _dy, float _dz);
    void setNumTimeSteps(int _ts);
    void setTimeSteps(const std::vector<int>& timeSteps);
    void setTimeSteps(const std::vector<float>& timeSteps);
    void setTimeSteps(const std::vector<double>& timeSteps);
    void setTimeSteps(const std::vector<std::string>& timeSteps);
    void setEnsembleMemberCount(int _es);
    void setFieldNames(const std::unordered_map<FieldType, std::vector<std::string>>& fieldNamesMap);
    void setFieldUnits(const std::unordered_map<FieldType, std::vector<std::string>>& fieldUnitsMap);
    void addField(float* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(uint8_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(uint16_t* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(HalfFloat* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addField(
            void* fieldData, ScalarDataFormat dataFormat, FieldType fieldType,
            const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void setLatLonData(float* _latData, float* _lonData);
    void setHeightData(float* _heightData);
    void addCalculator(const CalculatorPtr& calculator);
    const std::vector<std::string>& getFieldNames(FieldType fieldType);
    const std::vector<std::string>& getFieldNamesBase(FieldType fieldType);
    [[nodiscard]] bool getFieldExists(FieldType fieldType, const std::string& fieldName) const;
    [[nodiscard]] bool getIsScalarFieldDivergent(const std::string& fieldName) const;
    [[nodiscard]] inline int getGridSizeX() const { return xs; }
    [[nodiscard]] inline int getGridSizeY() const { return ys; }
    [[nodiscard]] inline int getGridSizeZ() const { return zs; }
    [[nodiscard]] inline int getTimeStepCount() const { return ts; }
    [[nodiscard]] inline int getEnsembleMemberCount() const { return es; }
    [[nodiscard]] inline int getCurrentTimeStepIdx() const { return currentTimeStepIdx; }
    [[nodiscard]] inline int getCurrentEnsembleIdx() const { return currentEnsembleIdx; }
    void setCurrentTimeStepIdx(int newTimeStepIdx);
    void setCurrentEnsembleIdx(int newEnsembleIdx);
    [[nodiscard]] inline size_t getSlice3dSizeInBytes(FieldType fieldType) const {
        return size_t(xs) * size_t(ys) * size_t(zs) * sizeof(float) * (fieldType == FieldType::SCALAR ? 1 : 3);
    }
    [[nodiscard]] inline size_t getSlice3dSizeInBytes(FieldType fieldType, ScalarDataFormat dataFormat) const {
        size_t sizeInBytes = size_t(xs) * size_t(ys) * size_t(zs) * (fieldType == FieldType::SCALAR ? 1 : 3);
        if (dataFormat == ScalarDataFormat::FLOAT) {
            sizeInBytes *= sizeof(float);
        } else if (dataFormat == ScalarDataFormat::SHORT || dataFormat == ScalarDataFormat::FLOAT16) {
            sizeInBytes *= sizeof(uint16_t);
        }
        return sizeInBytes;
    }
    [[nodiscard]] inline size_t getSlice3dEntryCount() const { return size_t(xs) * size_t(ys) * size_t(zs); }
    [[nodiscard]] inline float getDx() const { return dx; }
    [[nodiscard]] inline float getDy() const { return dy; }
    [[nodiscard]] inline float getDz() const { return dz; }
    [[nodiscard]] inline float getDt() const { return dt; }
    [[nodiscard]] inline const sgl::AABB3& getBoundingBox() { return box; }
    [[nodiscard]] inline const sgl::AABB3& getBoundingBoxRendering() { return boxRendering; }
    inline const std::vector<std::string>& getFilePaths() { return filePaths; }
    inline const DataSetInformation& getDataSetInformation() { return dataSetInformation; }
    FieldAccess createFieldAccessStruct(
            FieldType fieldType, const std::string& fieldName, int& timeStepIdx, int& ensembleIdx) const;

    /// Only fields with 32-bit floats not coming from a calculator support buffer mode currently.
    bool getScalarFieldSupportsBufferMode(int scalarFieldIdx);

    virtual bool setInputFiles(
            const std::vector<std::string>& _filePaths, DataSetInformation _dataSetInformation,
            glm::mat4* transformationMatrixPtr);

    virtual bool saveFieldToFile(const std::string& filePath, FieldType fieldType, int fieldIndex);

    /**
     * Only has to be called by the main thread once after loading is finished.
     */
    virtual void recomputeHistogram() {}

    // Sets the global file dialog.
    void setFileDialogInstance(ImGuiFileDialog* fileDialogInstance);

    /// Returns if the visualization mapping needs to be re-generated.
    [[nodiscard]] inline bool isDirty() const { return dirty; }
    /// Called by MainApp to reset the dirty flag.
    void resetDirty();
    /// Returns if the data needs to be re-rendered, but the visualization mapping is valid.
    virtual bool needsReRender() { bool tmp = reRender; reRender = false; return tmp; }
    [[nodiscard]] inline bool getShallReloadRendererShaders() {
        bool tmp = shallReloadRendererShaders;
        shallReloadRendererShaders = false;
        return tmp;
    }

    // TODO
    /// For changing performance measurement modes.
    //virtual bool setNewState(const InternalState& newState) { return false; }
    /// For changing internal settings programmatically and not over the GUI.
    //virtual bool setNewSettings(const SettingsMap& settings) {}

    /*
     * Retrieves the 3D field data. If the time step or member index is less than 0, the value globally selected is
     * used.
     */
    HostCacheEntry getFieldEntryCpu(
            FieldType fieldType, const std::string& fieldName, int timeStepIdx = -1, int ensembleIdx = -1);
    DeviceCacheEntry getFieldEntryDevice(
            FieldType fieldType, const std::string& fieldName, int timeStepIdx = -1, int ensembleIdx = -1,
            bool wantsImageData = true, const glm::uvec3& bufferTileSize = glm::uvec3(1, 1, 1));
    std::pair<float, float> getMinMaxScalarFieldValue(
            const std::string& fieldName, int timeStepIdx = -1, int ensembleIdx = -1);

    /*
     * Returns whether the field is either resident on the GPU or will be calculated on the GPU.
     * Using this function, it can be determined whether GPU operations can be done without additional
     * host -> device memory transfers.
     */
    bool getIsGpuResidentOrGpuCalculator(
            FieldType fieldType, const std::string& fieldName, int timeStepIdx = -1, int ensembleIdx = -1);

    /*
     * Registers auxiliary CPU or device memory used by calculators in the field cache.
     */
    AuxiliaryMemoryToken pushAuxiliaryMemoryCpu(size_t sizeInBytes);
    void popAuxiliaryMemoryCpu(AuxiliaryMemoryToken token);
    AuxiliaryMemoryToken pushAuxiliaryMemoryDevice(size_t sizeInBytes);
    void popAuxiliaryMemoryDevice(AuxiliaryMemoryToken token);

    // For querying lat/lon coordinates possibly associated with the grid.
    bool getHasLatLonData();
    void getLatLonData(const float*& _latData, const float*& _lonData);

    // For querying height and pressure data for individual levels.
    /// Uses PropertyEditor to display z layer info (height in m or pressure in hPa).
    void displayLayerInfo(sgl::PropertyEditor& propertyEditor, int zPlaneCoord);
    [[nodiscard]] bool getHasHeightData() const;
    [[nodiscard]] float getHeightDataForZ(int z) const;
    [[nodiscard]] float getHeightDataForZWorld(float zWorld) const;
    [[nodiscard]] std::string getHeightString(float height) const;

    // Keep track of transfer function use in renderers to display overlays in renderer.
    void acquireTf(Renderer* renderer, int varIdx);
    void releaseTf(Renderer* renderer, int varIdx);
    void acquireTf(Calculator* renderer, int varIdx);
    void releaseTf(Calculator* renderer, int varIdx);
    void onTransferFunctionMapRebuilt();
    void acquireScalarField(Renderer* renderer, int varIdx);
    void releaseScalarField(Renderer* renderer, int varIdx);
    void acquireScalarField(Calculator* calculator, int varIdx);
    void releaseScalarField(Calculator* calculator, int varIdx);
    bool getIsScalarFieldUsedInView(uint32_t viewIdx, uint32_t varIdx, Calculator* calculator = nullptr);
    bool getIsScalarFieldUsedInAnyView(uint32_t varIdx, Calculator* calculator = nullptr);
    uint32_t getVarIdxForCalculator(Calculator* calculator);
    const std::vector<CalculatorPtr>& getCalculators();
    std::vector<std::shared_ptr<ICorrelationCalculator>> getCorrelationCalculatorsUsed();
    [[nodiscard]] inline int getStandardScalarFieldIdx() const { return standardScalarFieldIdx; }

    /// Sets data bindings used across renderers.
    virtual void setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData);
    virtual void getPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines);
    [[nodiscard]] inline const sgl::vk::ImageSamplerPtr& getImageSampler() const { return imageSampler; }

    // Tracks use count of calculator classes.
    size_t getNewCalculatorUseCount(CalculatorType calculatorType);

    // For restricting the volume rendering to certain areas.
    void setRenderRestriction(
            Calculator* calculator, DistanceMetric distanceMetric, const glm::vec3& position, float radius);
    void resetRenderRestriction(Calculator* calculator);

    /**
     * Picks a point on the simulation domain boundary mesh using screen coordinates
     * (assuming origin at upper left corner of viewport).
     * @param globalX The x position on the screen (usually the mouse position).
     * @param globalY The y position on the screen (usually the mouse position).
     * @param firstHit The first hit point on the boundary mesh (closest to the camera) is stored in this variable.
     * @param lastHit The last hit point on the boundary mesh (furthest away from the camera) is stored in this variable.
     * @return True if a point on the mesh was hit.
     */
    bool pickPointScreen(SceneData* sceneData, int globalX, int globalY, glm::vec3& firstHit, glm::vec3& lastHit) const;
    bool pickPointScreenAtZ(SceneData* sceneData, int globalX, int globalY, int z, glm::vec3& hit) const;

    /**
     * Picks a point on the simulation domain boundary mesh using screen coordinates
     * (assuming origin at upper left corner of viewport).
     * @param globalX The x position on the screen (usually the mouse position).
     * @param globalY The y position on the screen (usually the mouse position).
     * @param firstHit The first hit point on the boundary mesh (closest to the camera) is stored in this variable.
     * @param lastHit The last hit point on the boundary mesh (furthest away from the camera) is stored in this variable.
     * @return True if a point on the mesh was hit.
     */
    bool pickPointWorld(
            const glm::vec3& cameraPosition, const glm::vec3& rayDirection, glm::vec3& firstHit, glm::vec3& lastHit) const;
    bool pickPointWorldAtZ(
            const glm::vec3& cameraPosition, const glm::vec3& rayDirection, int z, glm::vec3& hit) const;

protected:
    /// Size in x, y, z, time and ensemble dimensions.
    int xs = 0, ys = 0, zs = 0, ts = 1, es = 1;
    int tsFileCount = 1, esFileCount = 1;
    /// Distance between two neighboring points in x/y/z/time direction.
    float dx = 0.0f, dy = 0.0f, dz = 0.0f, dt = 1.0f;
    /// Box encompassing all grid points.
    sgl::AABB3 box, boxRendering;
    bool transpose = false;
    glm::ivec3 transposeAxes = glm::ivec3(0, 1, 2);
    int subsamplingFactor = 1;
    int ssxs = 0, ssys = 0, sszs = 0;

    int currentTimeStepIdx = 0;
    int currentEnsembleIdx = 0;

    // Cache system.
    sgl::vk::Renderer* renderer = nullptr;
    sgl::vk::Device* device = nullptr;
    ViewManager* viewManager = nullptr;
    std::vector<std::string> filePaths;
    DataSetInformation dataSetInformation;
    std::unique_ptr<HostFieldCache> hostFieldCache;
    std::unique_ptr<DeviceFieldCache> deviceFieldCache;
    std::unique_ptr<FieldMinMaxCache> fieldMinMaxCache;
    std::unordered_map<FieldType, std::vector<std::string>> typeToFieldNamesMap;
    std::unordered_map<FieldType, std::vector<std::string>> typeToFieldNamesMapBase; ///< Without calculator output.
    std::unordered_map<FieldType, std::vector<std::string>> typeToFieldUnitsMap; ///< Without calculator output.
    sgl::vk::ImageSamplerPtr imageSampler{};

    // Utility functions.
    void copyCacheEntryImageToBuffer(
            VolumeData::DeviceCacheEntry& deviceCacheEntryImage, VolumeData::DeviceCacheEntry& deviceCacheEntryBuffer);
    VolumeData::DeviceCacheEntry allocDeviceCacheEntryImage(FieldType fieldType, ScalarDataFormat scalarDataFormat);
    VolumeData::DeviceCacheEntry allocDeviceCacheEntryBuffer(
            size_t& bufferSize, FieldAccess& access,
            bool tileBufferMemory, uint32_t tileSizeX, uint32_t tileSizeY, uint32_t tileSizeZ);
    std::shared_ptr<ImageToBufferCopyPass> imageToBufferCopyPass;

    // For calculating the histogram on the GPU.
    void ensureStagingBufferExists(size_t sizeInBytes);
    sgl::vk::BufferPtr imageDataCacheBuffer;
    sgl::vk::BufferPtr minMaxValueBuffer;
    sgl::vk::BufferPtr maxHistogramValueBuffer;
    sgl::vk::BufferPtr minMaxReductionBuffers[2];
    sgl::vk::BufferPtr histogramUintBuffer;
    sgl::vk::BufferPtr histogramFloatBuffer;
    std::shared_ptr<MinMaxBufferWritePass> minMaxBufferWritePass;
    std::shared_ptr<MinMaxDepthReductionPass> minMaxReductionPasses[2];
    std::shared_ptr<ComputeHistogramPass> computeHistogramPass;
    std::shared_ptr<ComputeHistogramMaxPass> computeHistogramMaxPass;
    std::shared_ptr<ComputeHistogramDividePass> computeHistogramDividePass;
    std::shared_ptr<DivergentMinMaxPass> divergentMinMaxPass;

    bool dirty = true; ///< Should be set to true if the representation changed.
    bool isFirstDirty = true;
    bool reRender = false;
    ImGuiFileDialog* fileDialogInstance = nullptr;
    sgl::MultiVarTransferFunctionWindow multiVarTransferFunctionWindow;
    sgl::Color cachedClearColor;

    // Color legend widgets for different attributes.
    bool getIsTransferFunctionVisible(uint32_t viewIdx, uint32_t varIdx);
    void recomputeColorLegend();
    void setBaseFieldsDirty();
    bool shallRenderColorLegendWidgets = true;
    std::vector<sgl::ColorLegendWidget> colorLegendWidgets;
    /// Keep track of transfer function use in renderers to display overlays in renderer.
    std::unordered_multimap<int, Renderer*> transferFunctionToRendererMap;

    /*
     * Render restriction that can be set by calculators.
     * TODO: This should be done per view, but currently setRenderDataBindings and getPreprocessorDefines are not yet
     * view-dependent.
     */
    bool useRenderRestriction = false;
    Calculator* renderRestrictionCalculator = nullptr;
    glm::vec4 renderRestriction{}; ///< Position + radius.
    sgl::vk::BufferPtr renderRestrictionUniformBuffer{};
    bool renderRestrictionUniformBufferDirty = true;
    DistanceMetric renderRestrictionDistanceMetric = DistanceMetric::EUCLIDEAN;
    bool shallReloadRendererShaders = false;

    /*
     * Keep track of which scalar fields are used in which view to only display auxiliary calculator renderers
     * if they are used.
     */
    std::unordered_multimap<int, Renderer*> scalarFieldToRendererMap;
    std::unordered_multimap<Calculator*, Calculator*> calculatorUseMapRefToParent;
    std::unordered_multimap<Calculator*, Calculator*> calculatorUseMapParentToRef;
    int standardScalarFieldIdx = 0;
    bool separateFilesPerAttribute = false;
    int currentLoaderAttributeIdx = 0;

    // File loaders.
    VolumeLoader* createVolumeLoaderByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<VolumeLoader*()>> factoriesLoader;
    std::vector<VolumeLoader*> volumeLoaders;

    // File writers.
    VolumeWriter* createVolumeWriterByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<VolumeWriter*()>> factoriesWriter;

    // Calculators for derived variables.
    void updateCalculatorName(const CalculatorPtr& calculator);
    void updateCalculatorFilterDevice(const CalculatorPtr& calculator);
    void updateCalculator(const CalculatorPtr& calculator);
    void removeCalculator(const CalculatorPtr& calculator, int calculatorIdx);
    std::vector<CalculatorPtr> calculators;
    std::unordered_map<std::string, CalculatorPtr> calculatorsHost;
    std::unordered_map<std::string, CalculatorPtr> calculatorsDevice;
    size_t calculatorId = 0;
    sgl::vk::BufferPtr stagingBuffer; ///< For transferring calculator output from the GPU to the CPU.
    std::vector<std::pair<std::string, std::function<Calculator*()>>> factoriesCalculator;
    std::unordered_map<CalculatorType, size_t> calculatorTypeUseCounts;

    // Associated lat/lon data (may be nullptr).
    float* latData = nullptr;
    float* lonData = nullptr;

    // Associated height (in m) or pressure (in hPa) data (may be nullptr).
    float* heightData = nullptr;

private:
    static glm::vec3 screenPosToRayDir(SceneData* sceneData, int globalX, int globalY);
    static bool _rayBoxIntersection(
            const glm::vec3& rayOrigin, const glm::vec3& rayDirection, const glm::vec3& lower, const glm::vec3& upper,
            float& tNear, float& tFar);
    static bool _rayBoxPlaneIntersection(
            float rayOriginX, float rayDirectionX, float lowerX, float upperX, float& tNear, float& tFar);
    static bool _rayZPlaneIntersection(
            const glm::vec3& rayOrigin, const glm::vec3& rayDirection, float z, glm::vec2 lowerXY, glm::vec2 upperXY,
            float& tHit);
};

typedef std::shared_ptr<VolumeData> VolumeDataPtr;

#endif //CORRERENDER_VOLUMEDATA_HPP
