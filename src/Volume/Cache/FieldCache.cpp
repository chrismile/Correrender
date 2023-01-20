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

#include <Graphics/Vulkan/Utils/Device.hpp>

#include "Memory.hpp"
#include "FieldCache.hpp"

HostFieldCache::HostFieldCache() {
    size_t availableRam = getAvailableSystemMemoryBytes();
    cacheSizeMax = size_t(double(availableRam) * availableMemoryFactor);
}

DeviceFieldCache::DeviceFieldCache(sgl::vk::Device* device) {
    auto memoryHeapIndex = uint32_t(device->findMemoryHeapIndex(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT));
    size_t availableVram = device->getMemoryHeapBudgetVma(memoryHeapIndex);
    cacheSizeMax = size_t(double(availableVram) * availableMemoryFactor);
}

bool FieldMinMaxCache::exists(const FieldAccess& access) {
    return cache.find(access) != cache.end();
}

void FieldMinMaxCache::push(const FieldAccess& access, const CacheEntry& entry) {
    cache[access] = entry;
}

FieldMinMaxCache::CacheEntry FieldMinMaxCache::get(const FieldAccess& access) {
    return cache[access];
}

void FieldMinMaxCache::removeEntriesForFieldName(const std::string& fieldName) {
    auto it = cache.begin();
    while (it != cache.end()) {
        if (it->first.fieldName == fieldName) {
            it = cache.erase(it);
        } else {
            it++;
        }
    }
}
