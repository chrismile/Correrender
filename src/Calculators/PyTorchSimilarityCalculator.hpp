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

#ifndef CORRERENDER_PYTORCHSIMILARITYCALCULATOR_HPP
#define CORRERENDER_PYTORCHSIMILARITYCALCULATOR_HPP

#include "SimilarityCalculator.hpp"

enum class PyTorchDevice {
    CPU,
#ifdef SUPPORT_CUDA_INTEROP
    CUDA
#endif
};
const char* const PYTORCH_DEVICE_NAMES[] = {
        "CPU", "CUDA"
};

struct ModuleWrapper;

/**
 * Calls a PyTorch model using TorchScript to compute the ensemble similarity between a reference point and the volume.
 *
 * Information on how to save a PyTorch model to a TorchScript intermediate representation file in Python:
 * Models saved using "torch.save(model.state_dict(), 'model_name.pt')" can only be read in Python. Instead, use:
 *
 * https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
 * example_input, example_label = next(iter(dataloader))
 * script_module = torch.jit.trace(model, example_input)  # -> torch.jit.ScriptModule
 * script_module = torch.jit.script(model)  # Alternative
 * script_module.save('model_name.pt')
 *
 * Loading in Python: script_module = torch.jit.load('model_name.pt')
 *
 * https://pytorch.org/tutorials/advanced/cpp_export.html
 * "If you need to exclude some methods in your nn.Module because they use Python features that TorchScript doesn't
 * support yet, you could annotate those with @torch.jit.ignore."
 *
 * Examples of using custom operators: https://github.com/pytorch/pytorch/tree/master/test/custom_operator
 *
 * https://pytorch.org/docs/stable/jit.html
 * Mixing tracing and scripting: @torch.jit.script - "Traced functions can call script functions."
 *
 * How to add metadata? Add _extra_files=extra_files as an argument to torch.jit.save, e.g.:
 * extra_files = { 'model_info.json': '{ "metric_name": "Mutual Information (MI)", "input_format": "concat_value_position" }' }
 * torch.jit.save(script_module, 'model_name.pt', _extra_files=extra_files)
 */
class PyTorchSimilarityCalculator : public EnsembleSimilarityCalculator {
public:
    explicit PyTorchSimilarityCalculator(sgl::vk::Renderer* renderer);
    ~PyTorchSimilarityCalculator();
    std::string getOutputFieldName() override { return "Similarity Metric (Torch)"; }
    FilterDevice getFilterDevice() override { return FilterDevice::CPU; }
    void calculateCpu(int timeStepIdx, int ensembleIdx, float* buffer) override;
    void calculateDevice(int timeStepIdx, int ensembleIdx, const DeviceCacheEntry& deviceCacheEntry) override;

    bool loadModelFromFile(const std::string& modelPath);
    void setPyTorchDevice(PyTorchDevice pyTorchDeviceNew);

protected:
    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    void renderGuiImpl(sgl::PropertyEditor& propertyEditor) override;

private:
    PyTorchDevice pyTorchDevice = PyTorchDevice::CPU;
    std::shared_ptr<ModuleWrapper> wrapper;
    std::string modelFilePath;
    std::string fileDialogDirectory;

    // Data for CPU rendering.
    std::vector<sgl::vk::BufferPtr> renderImageStagingBuffers;
    sgl::vk::BufferPtr denoisedImageStagingBuffer;
    sgl::vk::FencePtr renderFinishedFence;
    sgl::vk::FencePtr denoiseFinishedFence;
    float* renderedImageData = nullptr;
    float* denoisedImageData = nullptr;

#ifdef SUPPORT_CUDA_INTEROP
    // Synchronization primitives.
    sgl::vk::BufferPtr inputDataBufferVk, outputImageBufferVk;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr inputDataBufferCu, outputImageBufferCu;
    std::vector<sgl::vk::CommandBufferPtr> postRenderCommandBuffers;
    std::vector<sgl::vk::SemaphoreVkCudaDriverApiInteropPtr> renderFinishedSemaphores;
    std::vector<sgl::vk::SemaphoreVkCudaDriverApiInteropPtr> denoiseFinishedSemaphores;
    uint64_t timelineValue = 0;
#endif
    //std::shared_ptr<FeatureCombinePass> featureCombinePass;
};

#endif //CORRERENDER_PYTORCHSIMILARITYCALCULATOR_HPP
