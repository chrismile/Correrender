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

#ifndef CORRERENDER_VECTORBACKENDSKIA_HPP
#define CORRERENDER_VECTORBACKENDSKIA_HPP

#ifdef SUPPORT_SKIA
#include <core/SkRefCnt.h>
#endif

#include <Graphics/Vector/VectorBackend.hpp>

class SkCanvas;
class SkPaint;
class SkTypeface;
typedef uint32_t SkColor;
struct SkiaCache;

class VectorBackendSkia : public sgl::VectorBackend {
public:
    static const char* getClassID() { return "Skia"; }
    [[nodiscard]] const char* getID() const override { return getClassID(); }
    static bool checkIsSupported();

    explicit VectorBackendSkia(sgl::VectorWidget* vectorWidget);
    void initialize() override;
    void destroy() override;
    void onResize() override;
    void renderStart() override;
    void renderEnd() override;
    bool renderGuiPropertyEditor(sgl::PropertyEditor& propertyEditor) override;
    void copyVectorBackendSettingsFrom(VectorBackend* backend) override;

    // Font API.
    sk_sp<SkTypeface> createDefaultTypeface();

    SkCanvas* getCanvas();
    void initializePaint(SkPaint* paint);

private:
    VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT;
    bool usePaintAA = true;
    bool useInternalAA = false;
    SkiaCache* skiaCache = nullptr;
};

SkColor toSkColor(const sgl::Color& col);

#endif //CORRERENDER_VECTORBACKENDSKIA_HPP
