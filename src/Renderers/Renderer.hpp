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

#ifndef CORRERENDER_RENDERER_HPP
#define CORRERENDER_RENDERER_HPP

#include <memory>
#include <Utils/Events/EventManager.hpp>

#include "RenderingModes.hpp"
#include "SceneData.hpp"

namespace sgl {
class TransferFunctionWindow;
class PropertyEditor;

namespace vk {
class Buffer;
typedef std::shared_ptr<Buffer> BufferPtr;
}
}

namespace IGFD {
class FileDialog;
}
typedef IGFD::FileDialog ImGuiFileDialog;

class SettingsMap;
class ViewManager;
class VolumeData;
typedef std::shared_ptr<VolumeData> VolumeDataPtr;
enum class FieldType : uint32_t;

// How should NaN values be handled in the transfer function?
enum class NaNHandling {
    IGNORE, SHOW_AS_YELLOW
};
const char* const NAN_HANDLING_NAMES[] = {
        "Ignore", "Show as Yellow"
};
const char* const NAN_HANDLING_IDS[] = {
        "ignore", "yellow"
};

class Renderer {
    friend class VolumeData;
public:
    Renderer(std::string windowName, ViewManager* viewManager);
    [[nodiscard]] virtual RenderingMode getRenderingMode() const = 0;
    virtual void initialize();
    virtual ~Renderer();

    /// Returns if the visualization mapping needs to be re-generated.
    [[nodiscard]] inline bool isDirty() const { return dirty; }
    /// Called by MainApp to reset the dirty flag.
    inline void resetDirty() { dirty = false; }
    /// Returns if the data needs to be re-rendered, but the visualization mapping is valid.
    [[nodiscard]] virtual bool needsReRender();
    [[nodiscard]] virtual bool needsReRenderView(uint32_t viewIdx);
    [[nodiscard]] inline bool getIsRasterizer() const { return isRasterizer; }
    [[nodiscard]] virtual bool getIsOpaqueRenderer() const { return true; }
    [[nodiscard]] virtual bool getIsOverlayRenderer() const { return false; }
    [[nodiscard]] virtual bool getShallRenderWithoutData() const { return false; }
    /// Called when the camera has moved.
    virtual void onHasMoved(uint32_t viewIdx) {}
    /// If the re-rendering was triggered from an outside source, frame accumulation cannot be used.
    virtual void notifyReRenderTriggeredExternally() { internalReRender = false; }
    /// Called when the clear color changed.
    virtual void setClearColor(const sgl::Color& clearColor) {}
    /**
     * Sets whether linear RGB or sRGB should be used for rendering. Most renderers won't need to do anything special,
     * as the transfer function data is automatically updated to use the correct format.
     */
    virtual void setUseLinearRGB(bool useLinearRGB) {}
    virtual void setFileDialogInstance(ImGuiFileDialog* _fileDialogInstance) {
        fileDialogInstance = _fileDialogInstance;
    }
    virtual void setSettings(const SettingsMap& settings);
    virtual void getSettings(SettingsMap& settings);

    /// Called when the transfer function was changed.
    virtual void onTransferFunctionMapRebuilt() {}
    /// Called when a calculator demands the reloading of shaders via VolumeData.
    virtual void reloadShaders() {}

    virtual void setVolumeData(VolumeDataPtr& volumeData, bool isNewData) = 0;
    virtual void onFieldRemoved(FieldType fieldType, int fieldIdx) {}

    bool isVisibleInView(uint32_t viewIdx);
    bool isVisibleInAnyView();
    virtual void renderView(uint32_t viewIdx) final;
    virtual void renderViewPre(uint32_t viewIdx) final;
    virtual void renderViewPostOpaque(uint32_t viewIdx) final;
    virtual void addView(uint32_t viewIdx) final;
    virtual void removeView(uint32_t viewIdx) final;
    virtual void recreateSwapchainView(uint32_t viewIdx, uint32_t width, uint32_t height);
    virtual void renderGui(sgl::PropertyEditor& propertyEditor) final;

    /// Renders GUI overlays. The "dirty" and "reRender" flags might be set depending on the user's actions.
    virtual void renderGuiOverlay(uint32_t viewIdx);
    /// For rendering secondary ImGui windows (e.g., for transfer function widgets).
    virtual void renderGuiWindowSecondary() {}
    /// Updates the internal logic (called once per frame).
    virtual void update(float dt, bool isMouseGrabbed) {}
    /// Returns whether the mouse is grabbed by the renderer.
    virtual bool getHasGrabbedMouse() const { return false; }

    inline const std::string& getWindowName() { return windowName; }
    [[nodiscard]] inline ViewManager* getViewManager() const { return viewManager; }
    [[nodiscard]] inline const VolumeDataPtr & getLineData() const { return volumeData; }
    [[nodiscard]] inline bool getHasLineData() const { return volumeData.get() != nullptr; }
    [[nodiscard]] inline size_t getCreationId() const { return creationId; }
    inline void setCreationId(size_t _creationId) { creationId = _creationId; }

protected:
    virtual void renderViewImpl(uint32_t viewIdx) {}
    virtual void renderViewPreImpl(uint32_t viewIdx) {}
    virtual void renderViewPostOpaqueImpl(uint32_t viewIdx) {}
    virtual void addViewImpl(uint32_t viewIdx) {}
    virtual void removeViewImpl(uint32_t viewIdx) {}
    virtual void renderGuiImpl(sgl::PropertyEditor& propertyEditor) {}

    bool isInitialized = false;
    size_t creationId = 0;
    sgl::ListenerToken onTransferFunctionMapRebuiltListenerToken{};

    // Metadata about renderer.
    bool isRasterizer = true;
    std::string windowName;
    void updateViewComboSelection();
    std::vector<bool> viewVisibilityArray;
    std::string showInViewComboValue;
    ImGuiFileDialog* fileDialogInstance = nullptr;

    ViewManager* viewManager;
    sgl::vk::Renderer* renderer = nullptr;
    VolumeDataPtr volumeData;
    bool dirty = true;
    bool reRender = true;
    std::vector<bool> reRenderViewArray;
    bool internalReRender = true; ///< For use in renderers with frame data accumulation.

    //uint32_t selectedTimeStep = 0;
    //uint32_t selectedEnsembleMember = 0;
    //int selectedAttributeIndex = 0;
    //int selectedAttributeIndexUi = 0;
};

typedef std::shared_ptr<Renderer> RendererPtr;

#endif //CORRERENDER_RENDERER_HPP
