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

#include <iostream>
#include <memory>

#include <Utils/File/Logfile.hpp>

#include "AuxiliaryMemoryToken.hpp"
#include "LRUCache.hpp"
#include "HostCacheEntry.hpp"
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

        constexpr bool failOnCacheExhaustion = false;
        if (cacheSize + bufferSize > cacheSizeMax && failOnCacheExhaustion) {
            sgl::Logfile::get()->throwError(
                    "Fatal error in FieldCache::ensureSufficientMemory: "
                    "Not enough memory could be freed to store the data!");
        }
    }

    void push(const FieldAccess& access, const CacheEntry& buffer) {
        cache.push(access, buffer);
        cacheSize += access.sizeInBytes;
    }

    void updateEvictionWaitList() {
        auto it = evictionWaitList.begin();
        while (it != evictionWaitList.end()) {
            if (it->second.expired()) {
                cacheSize -= it->first.sizeInBytes;
                it = evictionWaitList.erase(it);
            } else {
                it++;
            }
        }
    }

    void removeEntriesForFieldName(const std::string& fieldName) {
        cache.remove_if(
                [this, fieldName](std::pair<FieldAccess, CacheEntry>& entry) {
                    if (entry.first.fieldName == fieldName) {
                        typename CacheEntry::weak_type weakBufferPtr = entry.second;
                        entry.second = {};
                        if (!weakBufferPtr.expired()) {
                            evictionWaitList.emplace_back(entry.first, weakBufferPtr);
                        } else {
                            cacheSize -= entry.first.sizeInBytes;
                        }
                        return true;
                    }
                    return false;
                }
        );
    }

    // Calculators may need auxiliary memory during their lifetime, which counts to the maximum limit.
    AuxiliaryMemoryToken pushAuxiliaryMemory(size_t sizeInBytes) {
        AuxiliaryMemoryToken token = tokenCounter++;
        auxiliaryMemorySizeMap.insert(std::make_pair(token, sizeInBytes));
        auxiliaryMemorySizeTotal += sizeInBytes;
        return token;
    }
    void popAuxiliaryMemory(AuxiliaryMemoryToken token) {
        auto it = auxiliaryMemorySizeMap.find(token);
        if (it != auxiliaryMemorySizeMap.end()) {
            auxiliaryMemorySizeTotal -= it->second;
            auxiliaryMemorySizeMap.erase(it);
        } else {
            sgl::Logfile::get()->throwError("Error in FieldCache::popAuxiliaryMemory: Invalid token.");
        }
    }

protected:
    size_t cacheSize = 0, cacheSizeMax = 0;
    // What percentage of the available host/device memory should be used for caching?
    double availableMemoryFactor = 28.0 / 32.0; // 3.5/4, 7/8, 14/16, 21/24, 28/32.

    void evictLast() {
        FieldAccess access;
        typename CacheEntry::weak_type weakBufferPtr;
        {
            auto lruEntry = cache.pop_last();
            access = lruEntry.first;
            weakBufferPtr = lruEntry.second;
        }
        if (!weakBufferPtr.expired()) {
            evictionWaitList.emplace_back(access, weakBufferPtr);
        } else {
            cacheSize -= access.sizeInBytes;
        }
    }

private:
    LRUCache<FieldAccess, CacheEntry> cache;
    std::list<std::pair<FieldAccess, typename CacheEntry::weak_type>> evictionWaitList;

    // Auxiliary memory.
    AuxiliaryMemoryToken tokenCounter = 1;
    std::unordered_map<AuxiliaryMemoryToken, size_t> auxiliaryMemorySizeMap;
    size_t auxiliaryMemorySizeTotal = 0;
};

// Host (i.e., CPU) memory cache.
class HostFieldCache : public FieldCache<HostCacheEntryType> {
public:
    HostFieldCache();
};

/// Vulkan + CUDA support.
class DeviceFieldCache : public FieldCache<DeviceCacheEntryType> {
public:
    explicit DeviceFieldCache(sgl::vk::Device* device);
};

class FieldMinMaxCache {
    using CacheEntry = std::pair<float, float>;

public:
    bool exists(const FieldAccess& access);
    void push(const FieldAccess& access, const CacheEntry& entry);
    CacheEntry get(const FieldAccess& access);
    void removeEntriesForFieldName(const std::string& fieldName);

private:
    std::unordered_map<FieldAccess, CacheEntry> cache;
};

#endif //CORRERENDER_FIELDCACHE_HPP
