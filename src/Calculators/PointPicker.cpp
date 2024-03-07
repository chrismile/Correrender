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

#include <utility>
#include <glm/glm.hpp>

#include <Math/Math.hpp>
#include <Input/Mouse.hpp>
#include <Input/Keyboard.hpp>
#include <ImGui/imgui.h>

#include "Renderers/SceneData.hpp"
#include "Widgets/ViewManager.hpp"
#include "Volume/VolumeData.hpp"
#include "PointPicker.hpp"

PointPicker::PointPicker(
        ViewManager* _viewManager, bool& fixPickingZPlane, float& fixedZPlanePercentage,
        RefPosSetter _refPosSetter, ViewUsedIndexQuery _viewUsedIndexQuery)
        : viewManager(_viewManager), fixPickingZPlane(fixPickingZPlane), fixedZPlanePercentage(fixedZPlanePercentage),
          refPosSetter(std::move(_refPosSetter)), viewUsedIndexQuery(std::move(_viewUsedIndexQuery)) {
    keyMod = KMOD_CTRL;
    mouseButton = 1;
}

void PointPicker::setVolumeData(VolumeData* _volumeData, bool isNewData) {
    volumeData = _volumeData;
}

void PointPicker::onUpdatePositionFixed() {
    auto aabb = volumeData->getBoundingBoxRendering();
    auto z = aabb.min.z + (aabb.max.z - aabb.min.z) * fixedZPlanePercentage;
    focusPoint = glm::vec3(focusPoint.x, focusPoint.y, z);
    setReferencePointFromFocusPoint();
}

void PointPicker::overwriteFocusPointFromRefPoint(const glm::vec3& _refPoint) {
    glm::ivec3 maxCoord(
            volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
    sgl::AABB3 gridAabb = volumeData->getBoundingBoxRendering();
    focusPoint = _refPoint / glm::vec3(maxCoord) * (gridAabb.max - gridAabb.min) + gridAabb.min;
}

void PointPicker::update(float dt) {
    // Use mouse for selection of reference point.
    int mouseHoverWindowIndex = viewManager->getMouseHoverWindowIndex();
    if (mouseHoverWindowIndex >= 0) {
        SceneData* sceneData = viewManager->getViewSceneData(uint32_t(mouseHoverWindowIndex));
        if (viewUsedIndexQuery(mouseHoverWindowIndex)) {
            if (sgl::Keyboard->getModifier() & SDL_Keymod(keyMod)) {
                if (sgl::Mouse->buttonPressed(mouseButton) || (sgl::Mouse->isButtonDown(mouseButton)
                        && sgl::Mouse->mouseMoved())) {
                    ImVec2 mousePosGlobal = ImGui::GetMousePos();
                    //int mouseGlobalX = sgl::Mouse->getX();
                    if (fixPickingZPlane) {
                        glm::vec3 centerHit;
                        //int z = volumeData->getGridSizeZ() / 2;
                        auto z = int(std::round(float(volumeData->getGridSizeZ()) * fixedZPlanePercentage));
                        bool rayHasHitMesh = volumeData->pickPointScreenAtZ(
                                sceneData, int(mousePosGlobal.x), int(mousePosGlobal.y), z, centerHit);
                        if (rayHasHitMesh) {
                            auto aabb = volumeData->getBoundingBoxRendering();
                            focusPoint = centerHit;
                            firstHit = glm::vec3(centerHit.x, centerHit.y, aabb.max.z);
                            lastHit = glm::vec3(centerHit.x, centerHit.y, aabb.min.z);
                            hitLookingDirection = glm::vec3(0.0f, 0.0f, -glm::sign(sceneData->camera->getPosition().z));
                            hasHitInformation = true;
                            setReferencePointFromFocusPoint();
                        }
                    } else {
                        bool rayHasHitMesh = volumeData->pickPointScreen(
                                sceneData, int(mousePosGlobal.x), int(mousePosGlobal.y), firstHit, lastHit);
                        if (rayHasHitMesh) {
                            focusPoint = firstHit;
                            hitLookingDirection = glm::normalize(firstHit - sceneData->camera->getPosition());
                            hasHitInformation = true;
                            setReferencePointFromFocusPoint();
                        }
                    }
                }

                if (sgl::Mouse->getScrollWheel() > 0.1f || sgl::Mouse->getScrollWheel() < -0.1f) {
                    if (!hasHitInformation) {
                        glm::mat4 inverseViewMatrix = glm::inverse(sceneData->camera->getViewMatrix());
                        glm::vec3 lookingDirection = glm::vec3(
                                -inverseViewMatrix[2].x, -inverseViewMatrix[2].y, -inverseViewMatrix[2].z);

                        float moveAmount = sgl::Mouse->getScrollWheel() * dt * 0.5f;
                        glm::vec3 moveDirection = focusPoint - sceneData->camera->getPosition();
                        moveDirection *= float(sgl::sign(glm::dot(lookingDirection, moveDirection)));
                        if (glm::length(moveDirection) < 1e-4) {
                            moveDirection = lookingDirection;
                        }
                        moveDirection = glm::normalize(moveDirection);
                        focusPoint = focusPoint + moveAmount * moveDirection;
                    } else {
                        float moveAmount = sgl::Mouse->getScrollWheel() * dt;
                        glm::vec3 newFocusPoint = focusPoint + moveAmount * hitLookingDirection;
                        float t = glm::dot(newFocusPoint - firstHit, hitLookingDirection);
                        t = glm::clamp(t, 0.0f, glm::length(lastHit - firstHit));
                        focusPoint = firstHit + t * hitLookingDirection;
                    }
                    setReferencePointFromFocusPoint();
                }
            }
        }
    }
}

void PointPicker::setReferencePointFromFocusPoint() {
    glm::ivec3 maxCoord(
            volumeData->getGridSizeX() - 1, volumeData->getGridSizeY() - 1, volumeData->getGridSizeZ() - 1);
    sgl::AABB3 gridAabb = volumeData->getBoundingBoxRendering();
    glm::vec3 position = (focusPoint - gridAabb.min) / (gridAabb.max - gridAabb.min);
    position *= glm::vec3(maxCoord);
    glm::ivec3 referencePointNew = glm::ivec3(glm::round(position));
    referencePointNew = glm::clamp(referencePointNew, glm::ivec3(0), maxCoord);
    refPosSetter(referencePointNew);
}
