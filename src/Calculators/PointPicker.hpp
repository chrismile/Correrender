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

#ifndef CORRERENDER_POINTPICKER_HPP
#define CORRERENDER_POINTPICKER_HPP

#include <functional>
#include <glm/vec3.hpp>

class ViewManager;
class VolumeData;

typedef std::function<void(const glm::vec3& refPos)> RefPosSetter;
typedef std::function<bool(int)> ViewUsedIndexQuery;

class PointPicker {
public:
    PointPicker(
            ViewManager* viewManager, bool& fixPickingZPlane,
            RefPosSetter _refPosSetter, ViewUsedIndexQuery _viewUsedIndexQuery);
    void setVolumeData(VolumeData* _volumeData, bool isNewData);
    inline void setKeyMod(int _keyMod) { keyMod = _keyMod; }
    inline void setMouseButton(int _mouseButton) { mouseButton = _mouseButton; }
    void update(float dt);

private:
    ViewManager* viewManager;
    bool& fixPickingZPlane;
    RefPosSetter refPosSetter;
    ViewUsedIndexQuery viewUsedIndexQuery;
    VolumeData* volumeData = nullptr;
    int keyMod;
    int mouseButton;

    // Focus point picking/moving information.
    void setReferencePointFromFocusPoint();
    bool hasHitInformation = false;
    glm::vec3 focusPoint{};
    glm::vec3 firstHit{}, lastHit{};
    glm::vec3 hitLookingDirection{};
};

#endif //CORRERENDER_POINTPICKER_HPP
