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

#ifndef CORRERENDER_FIELDCACHE_HPP
#define CORRERENDER_FIELDCACHE_HPP

#include <memory>

#include <Utils/File/Logfile.hpp>

#include "LRUCache.hpp"
#include "DeviceCacheEntry.hpp"
#include "Volume/FieldAccess.hpp"

namespace sgl { namespace vk {
class Device;
class Buffer;
#ifdef SUPPORT_CUDA_INTEROP
class ImageCudaExternalMemoryVk;
#endif
}}

/**
 * Cache for a 3D field within a data set containing multiple time steps and ensemble members.
 */
template<class T> class FieldCache {
    using CacheEntry = std::shared_ptr<T>;

public:
    FieldCache() = default;
    virtual ~FieldCache() = default;

    bool exists(const FieldAccess& access) {
        return cache.exists(access);
    }

    CacheEntry reaccess(const FieldAccess& access) {
        auto entry = cache.pop(access);
        cache.push(entry.first, entry.second);
        return entry.second;
    }

    virtual void ensureSufficientMemory(size_t bufferSize) {
        while (cacheSize + bufferSize > cacheSizeMax && !cache.empty()) {
            evictLast();
        }

        if (cacheSize + bufferSize > cacheSizeMax) {
            sgl::Logfile::get()->throwError(
                    "Fatal error in FieldCache::ensureSufficientMemory: "
                    "Not enough memory could be freed to store the data!");
        }
    }

    void push(const FieldAccess& access, const CacheEntry& buffer) {
        cache.push(access, buffer);
    }

    void updateEvictionWaitList() {
        auto it = evictionWaitList.begin();
        while (it != evictionWaitList.end()) {
            if (it->second.expired()) {
                it = evictionWaitList.erase(it);
                cacheSize -= it->first.sizeInBytes;
            } else {
                it++;
            }
        }
    }

protected:
    size_t cacheSize = 0, cacheSizeMax = 0;
    // What percentage of the available host/device memory should be used for caching?
    double availableMemoryFactor = 3.0 / 4.0;

    void evictLast() {
        auto lruEntry = cache.pop_last();
        typename CacheEntry::weak_type weakBufferPtr = lruEntry.second;
        if (!weakBufferPtr.expired()) {
            evictionWaitList.emplace_back(lruEntry.first, weakBufferPtr);
        } else {
            cacheSize -= lruEntry.first.sizeInBytes;
        }
    }

private:
    LRUCache<FieldAccess, CacheEntry> cache;
    std::list<std::pair<FieldAccess, typename CacheEntry::weak_type>> evictionWaitList;
};

// Host (i.e., CPU) memory cache.
class HostFieldCache : public FieldCache<float[]> {
public:
    HostFieldCache();
};

/// Vulkan + CUDA support.
class DeviceFieldCache : public FieldCache<DeviceCacheEntryType> {
public:
    explicit DeviceFieldCache(sgl::vk::Device* device);
};

#endif //CORRERENDER_FIELDCACHE_HPP
