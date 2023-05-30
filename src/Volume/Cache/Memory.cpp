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

#include "Memory.hpp"

// See: https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g
#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
size_t getUsedSystemMemoryBytes() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys - status.ullAvailPhys;
}
size_t getAvailableSystemMemoryBytes() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullAvailPhys;
}
size_t getTotalSystemMemoryBytes() {
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return status.ullTotalPhys;
}

#elif defined(__linux__)

#include <Utils/StringUtils.hpp>
#include <cstdio>
#include <unistd.h>

size_t getUsedSystemMemoryBytes() {
    size_t totalNumPages = sysconf(_SC_PHYS_PAGES);
    size_t availablePages = sysconf(_SC_AVPHYS_PAGES);
    size_t pageSizeBytes = sysconf(_SC_PAGE_SIZE);
    return (totalNumPages - availablePages) * pageSizeBytes;
}

/*
 * sysconf only provides a rough estimate of "available" pages.
 * /proc/meminfo has multiple entries (system memory usually equals RAM):
 * - MemTotal, the total size of system memory in kB.
 * - MemFree, the amount of free physical system memory in kB.
 * - MemAvailable, the amount of system memory applications still can use in kB.
 * - Cached, the amount of system memory used for caching, but most of it being freeable, in kB.
 * It is recommended to use the value of MemAvailable for estimating how much memory applications on the system can
 * still allocate, as easily freeable cached memory, among others, is contained in it.
 *
 * For more info, please refer to:
 * https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=34e431b0ae398fc54ea69ff85ec700722c9da773
 */
#ifdef USE_SYSCONF
size_t getAvailableSystemMemoryBytes() {
    size_t availablePages = sysconf(_SC_AVPHYS_PAGES);
    size_t pageSizeBytes = sysconf(_SC_PAGE_SIZE);
    return availablePages * pageSizeBytes;
}
#else
size_t getAvailableSystemMemoryBytes() {
    size_t freeMemoryBytes = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        std::string lineString(line);
        if (sgl::startsWith(lineString, "MemAvailable:")) {
            std::vector<std::string> parts;
            sgl::splitStringWhitespace(lineString, parts);
            freeMemoryBytes = sgl::fromString<size_t>(parts.at(1)) * size_t(1024);
            break;
        }
    }
    fclose(f);
    return freeMemoryBytes;
}
#endif

size_t getTotalSystemMemoryBytes() {
    size_t totalNumPages = sysconf(_SC_PHYS_PAGES);
    size_t pageSizeBytes = sysconf(_SC_PAGE_SIZE);
    return totalNumPages * pageSizeBytes;
}

#else

size_t getUsedSystemMemoryBytes() {
    return 0;
}
size_t getAvailableSystemMemoryBytes() {
    // Assume 4GiB free on other systems.
    return size_t(1024 * 1024 * 1024) * size_t(4);
}

size_t getTotalSystemMemoryBytes() {
    // Assume 4GiB free on other systems.
    return size_t(1024 * 1024 * 1024) * size_t(4);
}

#endif
