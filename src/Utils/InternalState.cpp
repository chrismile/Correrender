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

#include "InternalState.hpp"

#include "Volume/VolumeData.hpp"
#include "InternalState.hpp"

void getTestModesDvr(std::vector<InternalState>& states, InternalState state) {
    state.renderingMode = RENDERING_MODE_DIRECT_VOLUME_RENDERING;
    state.name = "Direct Volume Rendering";
    states.push_back(state);
}

std::vector<InternalState> getTestModesPaper() {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    std::vector<InternalState> states;
    std::vector<glm::ivec2> windowResolutions = {
            //glm::ivec2(1920, 1080),
            glm::ivec2(3840, 2160)
    };
    bool isIntegratedGpu =
            device->getDeviceDriverId() == VK_DRIVER_ID_MOLTENVK
            || ((device->getDeviceDriverId() == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS
                 || device->getDeviceDriverId() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
                && device->getDeviceType() == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    if (isIntegratedGpu) {
        windowResolutions = { glm::ivec2(1920, 1080) };
    } else {
        windowResolutions = { glm::ivec2(3840, 2160) };
    }
    std::vector<DataSetDescriptor> dataSetDescriptors = {
            //DataSetDescriptor("Rings"),
            DataSetDescriptor("Aneurysm"),
            //DataSetDescriptor("Convection Rolls"),
            //DataSetDescriptor("Femur (Vis2021)"),
            //DataSetDescriptor("Bearing"),
            //DataSetDescriptor("Convection Rolls"),
            //DataSetDescriptor("Tangaroa (t=200)"),
    };
    std::vector<std::string> transferFunctionNames = {
            //"Standard.xml"
    };
    InternalState state;

    for (size_t i = 0; i < dataSetDescriptors.size(); i++) {
        state.dataSetDescriptor = dataSetDescriptors.at(i);
        for (size_t j = 0; j < windowResolutions.size(); j++) {
            state.windowResolution = windowResolutions.at(j);
            if (!transferFunctionNames.empty()) {
                state.transferFunctionName = transferFunctionNames.at(i);
            }
            getTestModesDvr(states, state);
        }
    }

    bool runStatesTwoTimesForErrorMeasure = true;
    if (runStatesTwoTimesForErrorMeasure) {
        std::vector<InternalState> oldStates = states;
        states.clear();
        for (size_t i = 0; i < oldStates.size(); i++) {
            InternalState state = oldStates.at(i);
            states.push_back(state);
            state.name += "(2)";
            states.push_back(state);
        }
    }

    for (InternalState& state : states) {
        state.nameRaw = state.name;
    }

    // Append model name to state name if more than one model is loaded
    if (dataSetDescriptors.size() > 1 || windowResolutions.size() > 1) {
        for (InternalState& state : states) {
            state.name =
                    sgl::toString(state.windowResolution.x)
                    + "x" + sgl::toString(state.windowResolution.y)
                    + " " + state.dataSetDescriptor.name + " " + state.name;
        }
    }

    return states;
}

std::vector<InternalState> getTestModes() {
    return getTestModesPaper();
}
