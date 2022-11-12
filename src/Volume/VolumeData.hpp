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

class VolumeLoader;
class Renderer;
class Calculator;
typedef std::shared_ptr<Calculator> CalculatorPtr;

class VolumeData {
public:
    using HostCacheEntry = std::shared_ptr<float[]>;
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
    void addField(float* fieldData, FieldType fieldType, const std::string& fieldName, int timeStepIdx, int ensembleIdx);
    void addCalculator(const CalculatorPtr& calculator);
    const std::vector<std::string>& getFieldNames(FieldType fieldType);
    const std::vector<std::string>& getFieldNamesBase(FieldType fieldType);
    [[nodiscard]] bool getFieldExists(FieldType fieldType, const std::string& fieldName) const;
    [[nodiscard]] inline int getGridSizeX() const { return xs; }
    [[nodiscard]] inline int getGridSizeY() const { return ys; }
    [[nodiscard]] inline int getGridSizeZ() const { return zs; }
    [[nodiscard]] inline int getTimeStepCount() const { return ts; }
    [[nodiscard]] inline int getEnsembleMemberCount() const { return es; }
    [[nodiscard]] inline size_t getSlice3dSizeInBytes(FieldType fieldType) const {
        return size_t(xs) * size_t(ys) * size_t(zs) * sizeof(float) * (fieldType == FieldType::SCALAR ? 1 : 3);
    }
    [[nodiscard]] inline size_t getSlice3dEntryCount() const { return size_t(xs) * size_t(ys) * size_t(zs); }
    [[nodiscard]] inline float getDx() const { return dx; }
    [[nodiscard]] inline float getDy() const { return dy; }
    [[nodiscard]] inline float getDz() const { return dz; }
    [[nodiscard]] inline float getDt() const { return dt; }
    [[nodiscard]] inline const sgl::AABB3& getBoundingBox() { return box; }
    [[nodiscard]] inline const sgl::AABB3& getBoundingBoxRendering() { return boxRendering; }
    inline const std::vector<std::string>& getFilePaths() { return filePaths; }

    virtual bool setInputFiles(
            const std::vector<std::string>& _filePaths, DataSetInformation _dataSetInformation,
            glm::mat4* transformationMatrixPtr);

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
            FieldType fieldType, const std::string& fieldName, int timeStepIdx = -1, int ensembleIdx = -1);
    std::pair<float, float> getMinMaxScalarFieldValue(
            const std::string& fieldName, int timeStepIdx = -1, int ensembleIdx = -1);

    /// Keep track of transfer function use in renderers to display overlays in renderer.
    void acquireTf(Renderer* renderer, int varIdx);
    void releaseTf(Renderer* renderer, int varIdx);
    void onTransferFunctionMapRebuilt();

    /// Sets data bindings used across renderers.
    virtual void setRenderDataBindings(const sgl::vk::RenderDataPtr& renderData);
    virtual void getPreprocessorDefines(std::map<std::string, std::string>& preprocessorDefines);
    [[nodiscard]] inline const sgl::vk::ImageSamplerPtr& getImageSampler() const { return imageSampler; }

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

    /// Cache system.
    sgl::vk::Renderer* renderer = nullptr;
    sgl::vk::Device* device = nullptr;
    std::vector<std::string> filePaths;
    DataSetInformation dataSetInformation;
    std::unique_ptr<HostFieldCache> hostFieldCache;
    std::unique_ptr<DeviceFieldCache> deviceFieldCache;
    std::unique_ptr<FieldMinMaxCache> fieldMinMaxCache;
    std::unordered_map<FieldType, std::vector<std::string>> typeToFieldNamesMap;
    std::unordered_map<FieldType, std::vector<std::string>> typeToFieldNamesMapBase; ///< Without calculator output.
    sgl::vk::ImageSamplerPtr imageSampler{};

    bool dirty = true; ///< Should be set to true if the representation changed.
    bool reRender = false;
    ImGuiFileDialog* fileDialogInstance = nullptr;
    sgl::MultiVarTransferFunctionWindow multiVarTransferFunctionWindow;

    // Color legend widgets for different attributes.
    bool getIsTransferFunctionVisible(uint32_t viewIdx, uint32_t tfIdx);
    void recomputeColorLegend();
    bool shallRenderColorLegendWidgets = true;
    std::vector<sgl::ColorLegendWidget> colorLegendWidgets;
    /// Keep track of transfer function use in renderers to display overlays in renderer.
    std::unordered_multimap<int, Renderer*> transferFunctionToRendererMap;

    // File loaders.
    VolumeLoader* createVolumeLoaderByExtension(const std::string& fileExtension);
    std::map<std::string, std::function<VolumeLoader*()>> factories;
    std::vector<VolumeLoader*> volumeLoaders;

    // Calculators for derived variables.
    void updateCalculatorName(const CalculatorPtr& calculator);
    void updateCalculator(const CalculatorPtr& calculator);
    std::vector<CalculatorPtr> calculators;
    std::unordered_map<std::string, CalculatorPtr> calculatorsHost;
    std::unordered_map<std::string, CalculatorPtr> calculatorsDevice;
    size_t calculatorId = 0;

private:
    FieldAccess createFieldAccessStruct(
            FieldType fieldType, const std::string& fieldName, int& timeStepIdx, int& ensembleIdx) const;
};

typedef std::shared_ptr<VolumeData> VolumeDataPtr;

#endif //CORRERENDER_VOLUMEDATA_HPP
