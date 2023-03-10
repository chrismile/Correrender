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

#ifndef CORRERENDER_VIEWMANAGER_HPP
#define CORRERENDER_VIEWMANAGER_HPP

#include <vector>
#include <cstdint>
#include <cstddef>

namespace sgl {
class Color;
}

namespace sgl { namespace vk {
class Renderer;
}}

struct SceneData;
class DataView;

class ViewManager {
public:
    ViewManager(sgl::Color* clearColor, sgl::vk::Renderer* renderer) : clearColor(clearColor), renderer(renderer) {}
    [[nodiscard]] size_t getNumViews() const { return sceneDataArray.size(); }
    SceneData* getViewSceneData(uint32_t viewIdx);
    DataView* getDataView(uint32_t viewIdx);
    void addView(DataView* dataView, SceneData* viewSceneData);
    void removeView(uint32_t viewIdx);
    [[nodiscard]] inline const sgl::Color& getClearColor() const { return *clearColor; }
    [[nodiscard]] inline sgl::vk::Renderer* getRenderer() { return renderer; }

    // For querying the state of the mouse in the views.
    [[nodiscard]] inline int getMouseHoverWindowIndex() const { return mouseHoverWindowIndex; }
    inline void setMouseHoverWindowIndex(int idx) { mouseHoverWindowIndex = idx; }

private:
    sgl::Color* clearColor;
    sgl::vk::Renderer* renderer;
    std::vector<SceneData*> sceneDataArray;
    std::vector<DataView*> dataViewArray;
    int mouseHoverWindowIndex = -1;
};
#endif //CORRERENDER_VIEWMANAGER_HPP
