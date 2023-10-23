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

#ifndef CORRERENDER_HOSTCACHEENTRY_HPP
#define CORRERENDER_HOSTCACHEENTRY_HPP

#include <memory>

#include <Utils/SciVis/ScalarDataFormat.hpp>

class HalfFloat;
class VolumeData;

class HostCacheEntryType {
    friend class VolumeData;
public:
    explicit HostCacheEntryType(size_t numEntries, float* dataFloat)
            : scalarDataFormatNative(ScalarDataFormat::FLOAT), numEntries(numEntries), dataFloat(dataFloat) {}
    explicit HostCacheEntryType(size_t numEntries, uint8_t* dataByte)
            : scalarDataFormatNative(ScalarDataFormat::BYTE), numEntries(numEntries), dataByte(dataByte) {}
    explicit HostCacheEntryType(size_t numEntries, uint16_t* dataShort)
            : scalarDataFormatNative(ScalarDataFormat::SHORT), numEntries(numEntries), dataShort(dataShort) {}
    explicit HostCacheEntryType(size_t numEntries, HalfFloat* dataFloat16)
            : scalarDataFormatNative(ScalarDataFormat::FLOAT16), numEntries(numEntries), dataFloat16(dataFloat16) {}
    ~HostCacheEntryType();

    [[nodiscard]] ScalarDataFormat getScalarDataFormatNative() const { return scalarDataFormatNative; }
    [[nodiscard]] const void* getDataNative();
    [[nodiscard]] const float* getDataFloat();
    [[nodiscard]] const uint8_t* getDataByte();
    [[nodiscard]] const uint16_t* getDataShort();
    [[nodiscard]] const HalfFloat* getDataFloat16();

    void switchNativeFormat(ScalarDataFormat newNativeFormat);

    /*template<class T>
    [[nodiscard]] inline const T* data() {
        if constexpr (std::is_same<T, uint8_t>::value) {
            return getDataByte();
        } else if constexpr (std::is_same<T, uint16_t>::value) {
            return getDataShort();
        } else if constexpr (std::is_same<T, float>::value) {
            return getDataFloat();
        } else if constexpr (
                !std::is_same<T, uint8_t>::value
                && !std::is_same<T, uint16_t>::value
                && !std::is_same<T, float>::value) {
            static_assert(false, "TEST");
        }
        return nullptr;
    }*/

    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, float>::value, T>::type* data() {
        return getDataFloat();
    }
    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, uint8_t>::value, T>::type* data() {
        return getDataByte();
    }
    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, uint16_t>::value, T>::type* data() {
        return getDataShort();
    }
    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, HalfFloat>::value, T>::type* data() {
        return getDataFloat16();
    }

    // Access data at a certain index.
    [[nodiscard]] float getDataFloatAt(size_t idx);
    template<class T>
    [[nodiscard]] inline typename std::enable_if<std::is_same<T, float>::value, T>::type dataAt(size_t idx) {
        return getDataFloatAt(idx);
    }

private:
    ScalarDataFormat scalarDataFormatNative;
    size_t numEntries = 0;
    float* dataFloat = nullptr;
    uint8_t* dataByte = nullptr;
    uint16_t* dataShort = nullptr;
    HalfFloat* dataFloat16 = nullptr;
};

#endif //CORRERENDER_HOSTCACHEENTRY_HPP
