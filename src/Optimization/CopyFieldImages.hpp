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

#ifndef CORRERENDER_COPYFIELDIMAGES_HPP
#define CORRERENDER_COPYFIELDIMAGES_HPP

namespace sgl { namespace vk {
class Device;
class Image;
typedef std::shared_ptr<Image> ImagePtr;
class ImageView;
typedef std::shared_ptr<ImageView> ImageViewPtr;
#ifdef CUDA_ENABLED
class TextureCudaExternalMemoryVk;
typedef std::shared_ptr<TextureCudaExternalMemoryVk> TextureCudaExternalMemoryVkPtr;
#endif
}}

struct CopyFieldImageDestinationData {
    sgl::vk::ImageViewPtr* inputImageGT;
    sgl::vk::ImageViewPtr* inputImageOpt;
#ifdef CUDA_ENABLED
    sgl::vk::TextureCudaExternalMemoryVkPtr* cudaInputImageGT;
    sgl::vk::TextureCudaExternalMemoryVkPtr* cudaInputImageOpt;
#endif
};

/**
 * Copies the content of the two field images to another set of field images.
 * This is necessary, as the computations for transfer function optimization may run in a separate thread.
 */
void copyFieldImages(
        sgl::vk::Device* device, uint32_t xs, uint32_t ys, uint32_t zs,
        const sgl::vk::ImagePtr& fieldImageGT, const sgl::vk::ImagePtr& fieldImageOpt,
        CopyFieldImageDestinationData& copyFieldImageDestinationData,
        uint32_t fieldIdxGT, uint32_t fieldIdxOpt, bool isSampled, bool exportMemory);

#endif //CORRERENDER_COPYFIELDIMAGES_HPP
