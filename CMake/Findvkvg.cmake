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
# vkvg_FOUND, vkvg_LIBRARIES, vkvg_INCLUDE_DIRS, vkvg_DEFINES

if (NOT DEFINED vkvg_DIR AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/vkvg")
    set(vkvg_DIR "${CMAKE_SOURCE_DIR}/third_party/vkvg")
endif()

if (DEFINED vkvg_DIR)
    set(ADDITIONAL_PATHS_ROOT "${vkvg_DIR}")
    set(ADDITIONAL_PATHS_INCLUDE "${vkvg_DIR}/include")
    set(ADDITIONAL_PATHS_LIBS "${vkvg_DIR}/lib")
endif()

find_path(vkvg_INCLUDE_DIR
        NAMES vkvg.h
        HINTS ${ADDITIONAL_PATHS_INCLUDE}
)

find_library(vkvg_LIBRARY
        NAMES vkvg
        HINTS ${ADDITIONAL_PATHS_LIBS}
)

mark_as_advanced(vkvg_INCLUDE_DIR vkvg_LIBRARY)

if(vkvg_INCLUDE_DIR AND vkvg_LIBRARY)
    set(vkvg_FOUND "YES")
    message(STATUS "Found vkvg: ${vkvg_LIBRARY}")
else()
    set(vkvg_FOUND "NO")
    set(vkvg_NOT_FOUND "Failed to find vkvg.")
    if(vkvg_FIND_REQUIRED)
        message(FATAL_ERROR ${vkvg_NOT_FOUND})
    else()
        if(NOT vkvg_FIND_QUIETLY)
            message(STATUS ${vkvg_NOT_FOUND})
        endif()
    endif()
endif()

if(vkvg_FOUND AND NOT TARGET vkvg::vkvg)
    set(vkvg_INCLUDE_DIRS "${vkvg_INCLUDE_DIR}")

    add_library(vkvg::vkvg UNKNOWN IMPORTED)
    set_target_properties(vkvg::vkvg PROPERTIES
            IMPORTED_LOCATION "${vkvg_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${vkvg_INCLUDE_DIRS}"
            INTERFACE_COMPILE_DEFINITIONS "${vkvg_DEFINES}")

    set(vkvg_LIBRARIES vkvg::vkvg)
endif()
