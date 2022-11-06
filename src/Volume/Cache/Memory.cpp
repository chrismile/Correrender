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
    return status.ullTotalPhys - status.ullAvailPhys;
}
#elif defined(__linux__)
#include <unistd.h>
size_t getUsedSystemMemoryBytes() {
    size_t totalNumPages = sysconf(_SC_PHYS_PAGES);
    size_t availablePages = sysconf(_SC_AVPHYS_PAGES);
    size_t pageSizeBytes = sysconf(_SC_PAGE_SIZE);
    return (totalNumPages - availablePages) * pageSizeBytes;
}
size_t getAvailableSystemMemoryBytes() {
    size_t totalNumPages = sysconf(_SC_PHYS_PAGES);
    size_t availablePages = sysconf(_SC_AVPHYS_PAGES);
    size_t pageSizeBytes = sysconf(_SC_PAGE_SIZE);
    return (totalNumPages - availablePages) * pageSizeBytes;
}
#else
size_t getUsedSystemMemoryBytes() {
    return 0;
}
size_t getAvailableSystemMemoryBytes() {
    // Assume 4GiB free on other systems.
    return size_t(1024 * 1024 * 1024) * size_t(4);
}
#endif
