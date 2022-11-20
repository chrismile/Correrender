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

#ifndef CORRERENDER_CALCULATOR_HPP
#define CORRERENDER_CALCULATOR_HPP

#include <string>

#include <Graphics/Vulkan/Buffers/Buffer.hpp>

#include "Volume/FieldType.hpp"

namespace sgl {
class PropertyEditor;
namespace vk {
class Renderer;
}
}

namespace IGFD {
class FileDialog;
}
typedef IGFD::FileDialog ImGuiFileDialog;

class ViewManager;
class VolumeData;
class Renderer;
typedef std::shared_ptr<Renderer> RendererPtr;
class DeviceCacheEntryType;
typedef std::shared_ptr<DeviceCacheEntryType> DeviceCacheEntry;

enum class FilterDevice {
    CPU, VULKAN, CUDA
};

/**
 * Derives new fields from existing fields. Called 'Calculator', e.g., in ParaView.
 */
class Calculator {
public:
    explicit Calculator(sgl::vk::Renderer* renderer) : renderer(renderer) {}
    virtual ~Calculator() = default;
    inline void setCalculatorId(size_t _calculatorId) { calculatorId = _calculatorId; }
    virtual void setViewManager(ViewManager* _viewManager) {}
    virtual void setVolumeData(VolumeData* _volumeData, bool isNewData);
    bool getIsDirty();
    bool getHasNameChanged();
    bool getHasFilterDeviceChanged();
    virtual void update(float dt) {}
    void renderGui(sgl::PropertyEditor& propertyEditor);
    virtual RendererPtr getCalculatorRenderer() { return {}; }
    [[nodiscard]] virtual bool getShouldRenderGui() const { return false; }
    void setFileDialogInstance(ImGuiFileDialog* _fileDialogInstance) { this->fileDialogInstance = _fileDialogInstance; }

    // Metadata about the filter.
    virtual FieldType getOutputFieldType() { return FieldType::SCALAR; }
    virtual std::string getOutputFieldName() = 0;
    virtual FilterDevice getFilterDevice() = 0;
    [[nodiscard]] virtual bool getHasFixedRange() const { return false; }
    [[nodiscard]] virtual std::pair<float, float> getFixedRange() const { return std::make_pair(-1.0f, 1.0f); }

    /// Writes the derived data to the output data of size VolumeData::xs*ys*zs.
    virtual void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) {}
    virtual void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) {}

protected:
    virtual void renderGuiImpl(sgl::PropertyEditor& propertyEditor) {}
    VolumeData* volumeData = nullptr;
    sgl::vk::Renderer* renderer = nullptr;
    bool dirty = false; ///< Recompute the data?
    bool hasNameChanged = false;
    bool hasFilterDeviceChanged = false;
    ImGuiFileDialog* fileDialogInstance = nullptr;
    size_t calculatorId = 0;
};

typedef std::shared_ptr<Calculator> CalculatorPtr;

#endif //CORRERENDER_CALCULATOR_HPP
