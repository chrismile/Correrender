#
# BSD 2-Clause License
#
# Copyright (c) 2021, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Provides the following variables:
# Skia_FOUND, Skia_LIBRARIES, Skia_INCLUDE_DIRS, Skia_DEFINES

if (NOT DEFINED Skia_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/skia")
    set(Skia_DIR "${CMAKE_SOURCE_DIR}/third_party/skia")
endif()

if (DEFINED Skia_DIR)
    set(ADDITIONAL_PATHS_ROOT "${Skia_DIR}")
    set(ADDITIONAL_PATHS_INCLUDE "${Skia_DIR}/include")
    if (DEFINED Skia_BUILD_TYPE)
        set(ADDITIONAL_PATHS_LIBS "${Skia_DIR}/out/${Skia_BUILD_TYPE}")
    else()
        set(ADDITIONAL_PATHS_LIBS "${Skia_DIR}/out/Static" "${Skia_DIR}/out/Shared")
    endif()
endif()

find_path(Skia_ROOT
        NAMES include/core/SkSurface.h
        HINTS ${ADDITIONAL_PATHS_ROOT}
)

find_path(Skia_INCLUDE_DIR
        NAMES core/SkSurface.h
        HINTS ${ADDITIONAL_PATHS_INCLUDE}
)

find_library(Skia_LIBRARY
        NAMES skia
        HINTS ${ADDITIONAL_PATHS_LIBS}
)

if(Skia_LIBRARY AND Skia_INCLUDE_DIR)
    get_filename_component(Skia_LIBRARY_PATH "${Skia_LIBRARY}" DIRECTORY)
    set(Skia_NINJA_FILE_PATH "${Skia_LIBRARY_PATH}/obj/skia.ninja")
    if(EXISTS "${Skia_NINJA_FILE_PATH}")
        file(READ "${Skia_NINJA_FILE_PATH}" Skia_NINJA_FILE_CONTENT)
        string(REGEX MATCH "defines = ([ \-=A-Za-z0-9_]*)" _ "${Skia_NINJA_FILE_CONTENT}")
        set(Skia_DEFINES_STRING ${CMAKE_MATCH_1})
        string(REGEX MATCHALL "[^(-D)]([=A-Za-z0-9_])+" Skia_DEFINES "${Skia_DEFINES_STRING}")

        # TODO: The debug build, for whatever reason, has missing defines and dependencies not included in libskia.so.
        #get_filename_component(Skia_LIBRARY_DIR_NAME "${Skia_LIBRARY_PATH}" NAME)
        #string(TOLOWER "${Skia_LIBRARY_DIR_NAME}" Skia_LIBRARY_DIR_NAME_LOWER)
        #if(Skia_LIBRARY_DIR_NAME_LOWER MATCHES "debug")
        #    list(APPEND Skia_DEFINES "SK_DEBUG")
        #    list(APPEND Skia_DEFINES "SK_DEBUG")
        #endif()
    else()
        set(SKIA_NOT_FOUND "Failed to find Skia 'obj/skia.ninja' file.")
        if(Skia_FIND_REQUIRED)
            message(FATAL_ERROR ${SKIA_NOT_FOUND})
        else()
            if(NOT Skia_FIND_QUIETLY)
                message(STATUS ${SKIA_NOT_FOUND})
            endif()
        endif()
    endif()
endif()

mark_as_advanced(Skia_ROOT Skia_INCLUDE_DIR Skia_LIBRARY Skia_DEFINES)

if(Skia_ROOT AND Skia_INCLUDE_DIR AND Skia_LIBRARY AND Skia_LIBRARY_PATH AND Skia_DEFINES)
    set(Skia_FOUND "YES")
    message(STATUS "Found Skia: ${Skia_LIBRARY}")
    message(STATUS "Skia_DEFINES: ${Skia_DEFINES}")
else()
    set(Skia_FOUND "NO")
    set(SKIA_NOT_FOUND "Failed to find Skia.")
    if(Skia_FIND_REQUIRED)
        message(FATAL_ERROR ${SKIA_NOT_FOUND})
    else()
        if(NOT Skia_FIND_QUIETLY)
            message(STATUS ${SKIA_NOT_FOUND})
        endif()
    endif()
endif()

if(Skia_FOUND AND NOT TARGET Skia::Skia)
    set(Skia_INCLUDE_DIRS "${Skia_ROOT}" "${Skia_INCLUDE_DIR}")

    add_library(Skia::Skia UNKNOWN IMPORTED)
    set_target_properties(Skia::Skia PROPERTIES
            IMPORTED_LOCATION "${Skia_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${Skia_INCLUDE_DIRS}"
            INTERFACE_COMPILE_DEFINITIONS "${Skia_DEFINES}")

    set(Skia_LIBRARIES Skia::Skia)
endif()
