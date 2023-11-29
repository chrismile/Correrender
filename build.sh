#!/bin/bash

# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
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

set -euo pipefail

scriptpath="$( cd "$(dirname "$0")" ; pwd -P )"
projectpath="$scriptpath"
pushd $scriptpath > /dev/null

if [[ "$(uname -s)" =~ ^MSYS_NT.* ]] || [[ "$(uname -s)" =~ ^MINGW.* ]]; then
    use_msys=true
else
    use_msys=false
fi
if [[ "$(uname -s)" =~ ^Darwin.* ]]; then
    use_macos=true
else
    use_macos=false
fi
os_arch="$(uname -m)"

run_program=true
debug=false
glibcxx_debug=false
build_dir_debug=".build_debug"
build_dir_release=".build_release"
use_vcpkg=false
link_dynamic=false
custom_glslang=false
build_with_zarr_support=true
build_with_cuda_support=true
build_with_skia_support=false
skia_link_dynamically=true
build_with_vkvg_support=false
build_with_osqp_support=true
build_with_zink_support=false
if $use_msys; then
    build_with_cuda_support=false
    # VKVG support is disabled due to: https://github.com/jpbruyere/vkvg/issues/140
    build_with_vkvg_support=false
fi
# Replicability Stamp (https://www.replicabilitystamp.org/) mode for replicating a figure from the corresponding paper.
replicability=false

# Process command line arguments.
for ((i=1;i<=$#;i++));
do
    if [ ${!i} = "--do-not-run" ]; then
        run_program=false
    elif [ ${!i} = "--debug" ] || [ ${!i} = "debug" ]; then
        debug=true
    elif [ ${!i} = "--glibcxx-debug" ]; then
        glibcxx_debug=true
    elif [ ${!i} = "--vcpkg" ]; then
        use_vcpkg=true
    elif [ ${!i} = "--link-static" ]; then
        link_dynamic=false
    elif [ ${!i} = "--link-dynamic" ]; then
        link_dynamic=true
    elif [ ${!i} = "--custom-glslang" ]; then
        custom_glslang=true
    elif [ ${!i} = "--replicability" ]; then
        replicability=true
    fi
done

if [ $debug = true ]; then
    cmake_config="Debug"
    build_dir=$build_dir_debug
else
    cmake_config="Release"
    build_dir=$build_dir_release
fi
destination_dir="Shipping"
if $use_macos; then
    binaries_dest_dir="$destination_dir/Correrender.app/Contents/MacOS"
    if ! command -v brew &> /dev/null; then
        if [ ! -d "/opt/homebrew/bin" ]; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        if [ -d "/opt/homebrew/bin" ]; then
            #echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/$USER/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
fi

params_link=()
params_vcpkg=()
if [ $use_vcpkg = true ] && [ $use_macos = false ] && [ $link_dynamic = true ]; then
    params_link+=(-DVCPKG_TARGET_TRIPLET=x64-linux-dynamic)
fi
if [ $use_vcpkg = true ] && [ $use_macos = false ]; then
    params_vcpkg+=(-DUSE_STATIC_STD_LIBRARIES=On)
fi
if [ $use_vcpkg = true ]; then
    params_vcpkg+=(-DCMAKE_TOOLCHAIN_FILE="$projectpath/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake")
fi

is_installed_apt() {
    local pkg_name="$1"
    if [ "$(dpkg -l | awk '/'"$pkg_name"'/ {print }'|wc -l)" -ge 1 ]; then
        return 0
    else
        return 1
    fi
}

is_installed_pacman() {
    local pkg_name="$1"
    if pacman -Qs $pkg_name > /dev/null; then
        return 0
    else
        return 1
    fi
}

is_installed_yay() {
    local pkg_name="$1"
    if yay -Ss $pkg_name > /dev/null | grep -q 'instal'; then
        return 1
    else
        return 0
    fi
}

is_installed_yum() {
    local pkg_name="$1"
    if yum list installed "$pkg_name" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

is_installed_rpm() {
    local pkg_name="$1"
    if rpm -q "$pkg_name" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

is_installed_brew() {
    local pkg_name="$1"
    if brew list $pkg_name > /dev/null; then
        return 0
    else
        return 1
    fi
}

if $use_msys && command -v pacman &> /dev/null && [ ! -d $build_dir_debug ] && [ ! -d $build_dir_release ]; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v rsync &> /dev/null \
            || ! command -v curl &> /dev/null || ! command -v wget &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        pacman --noconfirm -S --needed make git rsync curl wget mingw64/mingw-w64-x86_64-cmake \
        mingw64/mingw-w64-x86_64-gcc mingw64/mingw-w64-x86_64-gdb
    fi

    # Dependencies of sgl and the application.
    if ! is_installed_pacman "mingw-w64-x86_64-boost" || ! is_installed_pacman "mingw-w64-x86_64-glm" \
            || ! is_installed_pacman "mingw-w64-x86_64-libarchive" || ! is_installed_pacman "mingw-w64-x86_64-tinyxml2" \
            || ! is_installed_pacman "mingw-w64-x86_64-libpng" || ! is_installed_pacman "mingw-w64-x86_64-SDL2" \
            || ! is_installed_pacman "mingw-w64-x86_64-SDL2_image" || ! is_installed_pacman "mingw-w64-x86_64-glew" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-headers" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-loader" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-validation-layers" \
            || ! is_installed_pacman "mingw-w64-x86_64-shaderc" \
            || ! is_installed_pacman "mingw-w64-x86_64-opencl-headers" \
            || ! is_installed_pacman "mingw-w64-x86_64-opencl-icd" || ! is_installed_pacman "mingw-w64-x86_64-jsoncpp" \
            || ! is_installed_pacman "mingw-w64-x86_64-nlohmann-json" || ! is_installed_pacman "mingw-w64-x86_64-blosc" \
            || ! is_installed_pacman "mingw-w64-x86_64-netcdf" || ! is_installed_pacman "mingw-w64-x86_64-eccodes" \
            || ! is_installed_pacman "mingw-w64-x86_64-eigen3" || ! is_installed_pacman "mingw-w64-x86_64-libtiff" \
            || ! is_installed_pacman "mingw-w64-x86_64-nlopt"; then
        echo "------------------------"
        echo "installing dependencies "
        echo "------------------------"
        pacman --noconfirm -S --needed mingw64/mingw-w64-x86_64-boost mingw64/mingw-w64-x86_64-glm \
        mingw64/mingw-w64-x86_64-libarchive mingw64/mingw-w64-x86_64-tinyxml2 mingw64/mingw-w64-x86_64-libpng \
        mingw64/mingw-w64-x86_64-SDL2 mingw64/mingw-w64-x86_64-SDL2_image mingw64/mingw-w64-x86_64-glew \
        mingw64/mingw-w64-x86_64-vulkan-headers mingw64/mingw-w64-x86_64-vulkan-loader \
        mingw64/mingw-w64-x86_64-vulkan-validation-layers mingw64/mingw-w64-x86_64-shaderc \
        mingw64/mingw-w64-x86_64-opencl-headers mingw64/mingw-w64-x86_64-opencl-icd mingw64/mingw-w64-x86_64-jsoncpp \
        mingw64/mingw-w64-x86_64-nlohmann-json mingw64/mingw-w64-x86_64-blosc mingw64/mingw-w64-x86_64-netcdf \
        mingw64/mingw-w64-x86_64-eccodes mingw64/mingw-w64-x86_64-eigen3 mingw64/mingw-w64-x86_64-libtiff \
        mingw64/mingw-w64-x86_64-nlopt
    fi
    if ! (is_installed_pacman "mingw-w64-x86_64-curl" || is_installed_pacman "mingw-w64-x86_64-curl-gnutls" \
            || is_installed_pacman "mingw-w64-x86_64-curl-winssl"); then
        pacman --noconfirm -S --needed mingw64/mingw-w64-x86_64-curl
    fi
elif $use_macos && command -v brew &> /dev/null && [ ! -d $build_dir_debug ] && [ ! -d $build_dir_release ]; then
    if ! is_installed_brew "git"; then
        brew install git
    fi
    if ! is_installed_brew "cmake"; then
        brew install cmake
    fi
    if ! is_installed_brew "curl"; then
        brew install curl
    fi
    if ! is_installed_brew "pkg-config"; then
        brew install pkg-config
    fi
    if ! is_installed_brew "llvm"; then
        brew install llvm
    fi
    if ! is_installed_brew "libomp"; then
        brew install libomp
    fi

    # Homebrew MoltenVK does not contain script for setting environment variables, unfortunately.
    #if ! is_installed_brew "molten-vk"; then
    #    brew install molten-vk
    #fi

    # Dependencies of sgl and the application.
    if [ $use_vcpkg = false ]; then
        if ! is_installed_brew "boost"; then
            brew install boost
        fi
        if ! is_installed_brew "glm"; then
            brew install glm
        fi
        if ! is_installed_brew "libarchive"; then
            brew install libarchive
        fi
        if ! is_installed_brew "tinyxml2"; then
            brew install tinyxml2
        fi
        if ! is_installed_brew "zlib"; then
            brew install zlib
        fi
        if ! is_installed_brew "libpng"; then
            brew install libpng
        fi
        if ! is_installed_brew "sdl2"; then
            brew install sdl2
        fi
        if ! is_installed_brew "sdl2_image"; then
            brew install sdl2_image
        fi
        if ! is_installed_brew "glew"; then
            brew install glew
        fi
        if ! is_installed_brew "opencl-headers"; then
            brew install opencl-headers
        fi
        if ! is_installed_brew "jsoncpp"; then
            brew install jsoncpp
        fi
        if ! is_installed_brew "nlohmann-json"; then
            brew install nlohmann-json
        fi
        if ! is_installed_brew "c-blosc"; then
            brew install c-blosc
        fi
        if ! is_installed_brew "netcdf"; then
            brew install netcdf
        fi
        if ! is_installed_brew "eigen"; then
            brew install eigen
        fi
        if ! is_installed_brew "libtiff"; then
            brew install libtiff
        fi
        if ! is_installed_brew "curl"; then
            brew install curl
        fi
        if ! is_installed_brew "nlopt"; then
            brew install nlopt
        fi
    fi
elif command -v apt &> /dev/null; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        sudo apt install -y cmake git curl pkg-config build-essential patchelf
    fi

    # Dependencies of sgl and the application.
    if $use_vcpkg; then
        if ! is_installed_apt "libgl-dev" || ! is_installed_apt "libxmu-dev" || ! is_installed_apt "libxi-dev" \
                || ! is_installed_apt "libx11-dev" || ! is_installed_apt "libxft-dev" \
                || ! is_installed_apt "libxext-dev" || ! is_installed_apt "libwayland-dev" \
                || ! is_installed_apt "libxkbcommon-dev" || ! is_installed_apt "libegl1-mesa-dev"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo apt install -y libgl-dev libxmu-dev libxi-dev libx11-dev libxft-dev libxext-dev libwayland-dev \
            libxkbcommon-dev libegl1-mesa-dev
        fi
    else
        if ! is_installed_apt "libboost-filesystem-dev" || ! is_installed_apt "libglm-dev" \
                || ! is_installed_apt "libarchive-dev" || ! is_installed_apt "libtinyxml2-dev" \
                || ! is_installed_apt "libpng-dev" || ! is_installed_apt "libsdl2-dev" \
                || ! is_installed_apt "libsdl2-image-dev" || ! is_installed_apt "libglew-dev" \
                || ! is_installed_apt "opencl-c-headers" || ! is_installed_apt "ocl-icd-opencl-dev" \
                || ! is_installed_apt "libjsoncpp-dev" || ! is_installed_apt "nlohmann-json3-dev" \
                || ! is_installed_apt "libblosc-dev" || ! is_installed_apt "liblz4-dev" \
                || ! is_installed_apt "libnetcdf-dev" || ! is_installed_apt "libeccodes-dev" \
                || ! is_installed_apt "libeccodes-tools" || ! is_installed_apt "libopenjp2-7-dev" \
                || ! is_installed_apt "libeigen3-dev" || ! is_installed_apt "libtiff-dev" \
                || ! is_installed_apt "libnlopt-cxx-dev"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo apt install -y libboost-filesystem-dev libglm-dev libarchive-dev libtinyxml2-dev libpng-dev libsdl2-dev \
            libsdl2-image-dev libglew-dev opencl-c-headers ocl-icd-opencl-dev libjsoncpp-dev nlohmann-json3-dev \
            libblosc-dev liblz4-dev libnetcdf-dev libeccodes-dev libeccodes-tools libopenjp2-7-dev libeigen3-dev \
            libtiff-dev libnlopt-cxx-dev
        fi
        if ! (is_installed_apt "libcurl4-openssl-dev" || is_installed_apt "libcurl4-gnutls-dev" \
                || is_installed_apt "libcurl4-nss-dev"); then
            sudo apt install -y libcurl4-openssl-dev
        fi
    fi
elif command -v pacman &> /dev/null; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        sudo pacman -S cmake git curl pkgconf base-devel patchelf
    fi

    # Dependencies of sgl and the application.
    if $use_vcpkg; then
        if ! is_installed_pacman "libgl" || ! is_installed_pacman "vulkan-devel" || ! is_installed_pacman "shaderc" \
                || ! is_installed_pacman "openssl"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo pacman -S libgl vulkan-devel shaderc openssl
        fi
    else
        if ! is_installed_pacman "boost" || ! is_installed_pacman "glm" || ! is_installed_pacman "libarchive" \
                || ! is_installed_pacman "tinyxml2" || ! is_installed_pacman "libpng" || ! is_installed_pacman "sdl2" \
                || ! is_installed_pacman "sdl2_image" || ! is_installed_pacman "glew" \
                || ! is_installed_pacman "vulkan-devel" || ! is_installed_pacman "shaderc" \
                || ! is_installed_pacman "opencl-headers" || ! is_installed_pacman "ocl-icd" \
                || ! is_installed_pacman "jsoncpp" || ! is_installed_pacman "nlohmann-json" \
                || ! is_installed_pacman "blosc" || ! is_installed_pacman "netcdf" || ! is_installed_pacman "eigen" \
                || ! is_installed_pacman "libtiff" || ! is_installed_pacman "curl" \
                || ! is_installed_pacman "nlopt"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo pacman -S boost glm libarchive tinyxml2 libpng sdl2 sdl2_image glew vulkan-devel shaderc opencl-headers \
            ocl-icd jsoncpp nlohmann-json blosc netcdf eigen libtiff curl nlopt
        fi
        if ! command -v yay &> /dev/null && ! is_installed_yay "eccodes"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            yay -S eccodes
        fi
    fi
elif command -v yum &> /dev/null; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        sudo yum install -y cmake git curl pkgconf gcc gcc-c++ patchelf
    fi

    # Dependencies of sgl and the application.
    if $use_vcpkg; then
        if ! is_installed_rpm "perl" || ! is_installed_rpm "libstdc++-devel" || ! is_installed_rpm "libstdc++-static" \
                || ! is_installed_rpm "glew-devel" || ! is_installed_rpm "libXext-devel" \
                || ! is_installed_rpm "vulkan-headers" || ! is_installed_rpm "vulkan-loader" \
                || ! is_installed_rpm "vulkan-tools" || ! is_installed_rpm "vulkan-validation-layers" \
                || ! is_installed_rpm "libshaderc-devel"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo yum install -y perl libstdc++-devel libstdc++-static glew-devel libXext-devel vulkan-headers \
            vulkan-loader vulkan-tools vulkan-validation-layers libshaderc-devel
        fi
    else
        if ! is_installed_rpm "boost-devel" || ! is_installed_rpm "glm-devel" || ! is_installed_rpm "libarchive-devel" \
                || ! is_installed_rpm "tinyxml2-devel" || ! is_installed_rpm "libpng-devel" \
                || ! is_installed_rpm "SDL2-devel" || ! is_installed_rpm "SDL2_image-devel" \
                || ! is_installed_rpm "glew-devel" || ! is_installed_rpm "vulkan-headers" \
                || ! is_installed_rpm "libshaderc-devel" || ! is_installed_rpm "opencl-headers" \
                || ! is_installed_rpm "ocl-icd" || ! is_installed_rpm "jsoncpp-devel" || ! is_installed_rpm "json-devel" \
                || ! is_installed_rpm "blosc-devel" || ! is_installed_rpm "netcdf-devel" \
                || ! is_installed_rpm "eccodes-devel" || ! is_installed_rpm "eigen3-devel" \
                || ! is_installed_rpm "libtiff-devel" || ! is_installed_rpm "libcurl-devel" \
                || ! is_installed_rpm "NLopt-devel"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo yum install -y boost-devel glm-devel libarchive-devel tinyxml2-devel libpng-devel SDL2-devel \
            SDL2_image-devel glew-devel vulkan-headers libshaderc-devel opencl-headers ocl-icd jsoncpp-devel json-devel \
            blosc-devel netcdf-devel eccodes-devel eigen3-devel libtiff-devel libcurl-devel NLopt-devel
        fi
    fi
else
    echo "Warning: Unsupported system package manager detected." >&2
fi

if ! command -v cmake &> /dev/null; then
    echo "CMake was not found, but is required to build the program."
    exit 1
fi
if ! command -v git &> /dev/null; then
    echo "git was not found, but is required to build the program."
    exit 1
fi
if ! command -v curl &> /dev/null; then
    echo "curl was not found, but is required to build the program."
    exit 1
fi
if [ $use_macos = false ] && ! command -v pkg-config &> /dev/null; then
    echo "pkg-config was not found, but is required to build the program."
    exit 1
fi

if [ ! -d "submodules/IsosurfaceCpp/src" ]; then
    echo "------------------------"
    echo "initializing submodules "
    echo "------------------------"
    git submodule init
    git submodule update
fi

[ -d "./third_party/" ] || mkdir "./third_party/"
pushd third_party > /dev/null

cmake_version=$(cmake --version | head -n 1 | awk '{print $NF}')
cmake_version_major=$(echo $cmake_version | cut -d. -f1)
cmake_version_minor=$(echo $cmake_version | cut -d. -f2)
if [[ $cmake_version_major < 3 || ($cmake_version_major == 3 && $cmake_version_minor < 18) ]]; then
    cmake_download_version="3.25.2"
    if [ ! -d "cmake-${cmake_download_version}-linux-x86_64" ]; then
        echo "------------------------"
        echo "    downloading cmake   "
        echo "------------------------"
        curl --silent --show-error --fail -OL "https://github.com/Kitware/CMake/releases/download/v${cmake_download_version}/cmake-${cmake_download_version}-linux-x86_64.tar.gz"
        tar -xf cmake-${cmake_download_version}-linux-x86_64.tar.gz -C .
    fi
    PATH="${projectpath}/third_party/cmake-${cmake_download_version}-linux-x86_64/bin:$PATH"
fi

params_sgl=()
params=()
params_run=()
params_gen=()

if [ $use_msys = true ]; then
    params_gen+=(-G "MSYS Makefiles")
    params_sgl+=(-G "MSYS Makefiles")
    params+=(-G "MSYS Makefiles")
fi

if [ $use_vcpkg = false ] && [ $use_macos = true ]; then
    params_gen+=(-DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=False)
    params_gen+=(-DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=False)
    params_gen+=(-DCMAKE_FIND_FRAMEWORK=LAST)
    params_gen+=(-DCMAKE_FIND_APPBUNDLE=NEVER)
    params_gen+=(-DCMAKE_PREFIX_PATH="$(brew --prefix)")
    params_sgl+=(-DCMAKE_INSTALL_PREFIX="../install")
    params_sgl+=(-DZLIB_ROOT="$(brew --prefix)/opt/zlib")
    params+=(-DZLIB_ROOT="$(brew --prefix)/opt/zlib")
fi

if $glibcxx_debug; then
    params_sgl+=(-DUSE_GLIBCXX_DEBUG=On)
    params+=(-DUSE_GLIBCXX_DEBUG=On)
fi

use_vulkan=false
vulkan_sdk_env_set=true
use_vulkan=true

search_for_vulkan_sdk=false
if [ $use_msys = false ] && [ -z "${VULKAN_SDK+1}" ]; then
    search_for_vulkan_sdk=true
fi

if [ $search_for_vulkan_sdk = true ]; then
    echo "------------------------"
    echo "searching for Vulkan SDK"
    echo "------------------------"

    found_vulkan=false

    if [ $use_macos = false ]; then
        if [ -d "VulkanSDK" ]; then
            VK_LAYER_PATH=""
            source "VulkanSDK/$(ls VulkanSDK)/setup-env.sh"
            pkgconfig_dir="$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig")"
            if [ -d "$pkgconfig_dir" ]; then
                export PKG_CONFIG_PATH="$pkgconfig_dir"
            fi
            found_vulkan=true
        fi

        if ! $found_vulkan && (lsb_release -a 2> /dev/null | grep -q 'Ubuntu' || lsb_release -a 2> /dev/null | grep -q 'Mint'); then
            if lsb_release -a 2> /dev/null | grep -q 'Ubuntu'; then
                distro_code_name=$(lsb_release -cs)
            else
                distro_code_name=$(cat /etc/upstream-release/lsb-release | grep "DISTRIB_CODENAME=" | sed 's/^.*=//')
            fi
            if ! compgen -G "/etc/apt/sources.list.d/lunarg-vulkan-*" > /dev/null \
                  && ! curl -s -I "https://packages.lunarg.com/vulkan/dists/${distro_code_name}/" | grep "2 404" > /dev/null; then
                echo "Setting up Vulkan SDK for $(lsb_release -ds)..."
                wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
                sudo curl --silent --show-error --fail \
                https://packages.lunarg.com/vulkan/lunarg-vulkan-${distro_code_name}.list \
                --output /etc/apt/sources.list.d/lunarg-vulkan-${distro_code_name}.list
                sudo apt update
                sudo apt install -y vulkan-sdk shaderc glslang-dev
            fi
        fi

        if [ -d "/usr/include/vulkan" ] && [ -d "/usr/include/shaderc" ]; then
            if ! grep -q VULKAN_SDK ~/.bashrc; then
                echo 'export VULKAN_SDK="/usr"' >> ~/.bashrc
            fi
            export VULKAN_SDK="/usr"
            found_vulkan=true
        fi

        if ! $found_vulkan; then
            curl --silent --show-error --fail -O https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz
            mkdir -p VulkanSDK
            tar -xf vulkan-sdk.tar.gz -C VulkanSDK
            VK_LAYER_PATH=""
            source "VulkanSDK/$(ls VulkanSDK)/setup-env.sh"

            # Fix pkgconfig file.
            shaderc_pkgconfig_file="VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig/shaderc.pc"
            if [ -f "$shaderc_pkgconfig_file" ]; then
                prefix_path=$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch")
                sed -i '3s;.*;prefix=\"'$prefix_path'\";' "$shaderc_pkgconfig_file"
                sed -i '5s;.*;libdir=${prefix}/lib;' "$shaderc_pkgconfig_file"
                export PKG_CONFIG_PATH="$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig")"
            fi
            found_vulkan=true
        fi
    else
        if [ -d "$HOME/VulkanSDK" ] && [ ! -z "$(ls -A "$HOME/VulkanSDK")" ]; then
            source "$HOME/VulkanSDK/$(ls $HOME/VulkanSDK)/setup-env.sh"
            found_vulkan=true
        else
            vulkansdk_filename=$(curl -sIkL https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg | sed -r '/filename=/!d;s/.*filename=(.*)$/\1/')
            VULKAN_SDK_VERSION=$(echo $vulkansdk_filename | sed -r 's/^.*vulkansdk-macos-(.*)\.dmg.*$/\1/')
            curl -O https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg
            sudo hdiutil attach vulkan-sdk.dmg
            # The directory was changed from '/Volumes/VulkanSDK' to, e.g., 'vulkansdk-macos-1.3.261.0'.
            vulkan_dir=$(find /Volumes -maxdepth 1 -name '[Vv]ulkan*' -not -path "/Volumes/VMware*" || true)
            sudo "${vulkan_dir}/InstallVulkan.app/Contents/MacOS/InstallVulkan" \
            --root ~/VulkanSDK/$VULKAN_SDK_VERSION --accept-licenses --default-answer --confirm-command install
            pushd ~/VulkanSDK/$VULKAN_SDK_VERSION
            sudo python3 ./install_vulkan.py || true
            popd
            sudo hdiutil unmount "${vulkan_dir}"
            source "$HOME/VulkanSDK/$(ls $HOME/VulkanSDK)/setup-env.sh"
            found_vulkan=true
        fi
    fi

    if ! $found_vulkan; then
        if [ $use_macos = false ]; then
            os_name="linux"
        else
            os_name="mac"
        fi
        echo "The environment variable VULKAN_SDK is not set but is required in the installation process."
        echo "Please refer to https://vulkan.lunarg.com/sdk/home#${os_name} for instructions on how to install the Vulkan SDK."
        exit 1
    fi
fi

if $custom_glslang; then
    if [ ! -d "./glslang" ]; then
        echo "------------------------"
        echo "  downloading glslang   "
        echo "------------------------"
        # Make sure we have no leftovers from a failed build attempt.
        if [ -d "./glslang-src" ]; then
            rm -rf "./glslang-src"
        fi
        git clone https://github.com/KhronosGroup/glslang.git glslang-src
        pushd glslang-src >/dev/null
        ./update_glslang_sources.py
        mkdir build
        pushd build >/dev/null
        cmake ${params_gen[@]+"${params_gen[@]}"} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/glslang" ..
        make -j $(nproc)
        make install
        popd >/dev/null
        popd >/dev/null
    fi
    params_sgl+=(-Dglslang_DIR="${projectpath}/third_party/glslang" -DUSE_SHADERC=Off)
fi

if [ $use_msys = false ] && [ -z "${VULKAN_SDK+1}" ]; then
    vulkan_sdk_env_set=true
fi

if [ $use_vcpkg = true ] && [ ! -d "./vcpkg" ]; then
    echo "------------------------"
    echo "    fetching vcpkg      "
    echo "------------------------"
    if $use_vulkan && [ $vulkan_sdk_env_set = false ]; then
        echo "The environment variable VULKAN_SDK is not set but is required in the installation process."
        exit 1
    fi
    git clone --depth 1 https://github.com/Microsoft/vcpkg.git
    vcpkg/bootstrap-vcpkg.sh -disableMetrics
    vcpkg/vcpkg install
fi

if [ $use_vcpkg = true ] && [ $use_macos = false ] && [ $link_dynamic = false ]; then
    params_sgl+=(-DBUILD_STATIC_LIBRARY=On)
fi

if [ ! -d "./sgl" ]; then
    echo "------------------------"
    echo "     fetching sgl       "
    echo "------------------------"
    git clone --depth 1 https://github.com/chrismile/sgl.git
fi

if [ -f "./sgl/$build_dir/CMakeCache.txt" ]; then
    if grep -q vcpkg_installed "./sgl/$build_dir/CMakeCache.txt"; then
        cache_uses_vcpkg=true
    else
        cache_uses_vcpkg=false
    fi
    if ([ $use_vcpkg = true ] && [ $cache_uses_vcpkg = false ]) || ([ $use_vcpkg = false ] && [ $cache_uses_vcpkg = true ]); then
        echo "Removing old sgl build cache..."
        if [ -d "./sgl/$build_dir_debug" ]; then
            rm -rf "./sgl/$build_dir_debug"
        fi
        if [ -d "./sgl/$build_dir_release" ]; then
            rm -rf "./sgl/$build_dir_release"
        fi
        if [ -d "./sgl/install" ]; then
            rm -rf "./sgl/install"
        fi
    fi
fi

if [ ! -d "./sgl/install" ]; then
    echo "------------------------"
    echo "     building sgl       "
    echo "------------------------"

    pushd "./sgl" >/dev/null
    mkdir -p $build_dir_debug
    mkdir -p $build_dir_release

    pushd "$build_dir_debug" >/dev/null
    cmake .. \
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_INSTALL_PREFIX="../install" \
         ${params_gen[@]+"${params_gen[@]}"} ${params_link[@]+"${params_link[@]}"} \
         ${params_vcpkg[@]+"${params_vcpkg[@]}"} ${params_sgl[@]+"${params_sgl[@]}"}
    if [ $use_vcpkg = false ] && [ $use_macos = false ]; then
        make -j $(nproc)
        make install
    fi
    popd >/dev/null

    pushd $build_dir_release >/dev/null
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="../install" \
         ${params_gen[@]+"${params_gen[@]}"} ${params_link[@]+"${params_link[@]}"} \
         ${params_vcpkg[@]+"${params_vcpkg[@]}"} ${params_sgl[@]+"${params_sgl[@]}"}
    if [ $use_vcpkg = false ] && [ $use_macos = false ]; then
        make -j $(nproc)
        make install
    fi
    popd >/dev/null

    if [ $use_macos = true ]; then
        cmake --build $build_dir_debug --parallel $(sysctl -n hw.ncpu)
        cmake --build $build_dir_debug --target install

        cmake --build $build_dir_release --parallel $(sysctl -n hw.ncpu)
        cmake --build $build_dir_release --target install
    elif [ $use_vcpkg = true ]; then
        cmake --build $build_dir_debug --parallel $(nproc)
        cmake --build $build_dir_debug --target install
        if [ $link_dynamic = true ]; then
            cp $build_dir_debug/libsgld.so install/lib/libsgld.so
        fi

        cmake --build $build_dir_release --parallel $(nproc)
        cmake --build $build_dir_release --target install
        if [ $link_dynamic = true ]; then
            cp $build_dir_release/libsgl.so install/lib/libsgl.so
        fi
    fi

    popd >/dev/null
fi

if $build_with_zarr_support; then
    if [ ! -d "./xtl" ]; then
        echo "------------------------"
        echo "    downloading xtl     "
        echo "------------------------"
        # Make sure we have no leftovers from a failed build attempt.
        if [ -d "./xtl-src" ]; then
            rm -rf "./xtl-src"
        fi
        git clone https://github.com/xtensor-stack/xtl.git xtl-src
        mkdir -p xtl-src/build
        pushd xtl-src/build >/dev/null
        cmake ${params_gen[@]+"${params_gen[@]}"} -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/xtl" ..
        make install
        popd >/dev/null
    fi
    if [ ! -d "./xtensor" ]; then
        echo "------------------------"
        echo "  downloading xtensor   "
        echo "------------------------"
        # Make sure we have no leftovers from a failed build attempt.
        if [ -d "./xtensor-src" ]; then
            rm -rf "./xtensor-src"
        fi
        git clone https://github.com/xtensor-stack/xtensor.git xtensor-src
        mkdir -p xtensor-src/build
        pushd xtensor-src/build >/dev/null
        cmake ${params_gen[@]+"${params_gen[@]}"} -Dxtl_DIR="${projectpath}/third_party/xtl/share/cmake/xtl" \
        -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/xtensor" ..
        make install
        popd >/dev/null
    fi
    if [ ! -d "./xsimd" ]; then
        echo "------------------------"
        echo "   downloading xsimd    "
        echo "------------------------"
        # Make sure we have no leftovers from a failed build attempt.
        if [ -d "./xsimd-src" ]; then
            rm -rf "./xsimd-src"
        fi
        git clone https://github.com/xtensor-stack/xsimd.git xsimd-src
        mkdir -p xsimd-src/build
        pushd xsimd-src/build >/dev/null
        cmake ${params_gen[@]+"${params_gen[@]}"} -Dxtl_DIR="${projectpath}/third_party/xtl/share/cmake/xtl" \
        -DENABLE_XTL_COMPLEX=ON \
        -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/xsimd" ..
        make install
        popd >/dev/null
    fi

    # Seems like xtensor can install its CMake config either to the share or lib folder.
    if [ -d "${projectpath}/third_party/xtensor/share/cmake/xtensor" ]; then
        xtensor_CMAKE_DIR="${projectpath}/third_party/xtensor/share/cmake/xtensor"
    else
        xtensor_CMAKE_DIR="${projectpath}/third_party/xtensor/lib/cmake/xtensor"
    fi

    if [ ! -d "./z5" ]; then
        echo "------------------------"
        echo "     downloading z5     "
        echo "------------------------"
        # Make sure we have no leftovers from a failed build attempt.
        if [ -d "./z5-src" ]; then
            rm -rf "./z5-src"
        fi
        git clone https://github.com/constantinpape/z5.git z5-src
        if [ $use_macos = true ]; then
            sed -i -e 's/SET(Boost_NO_SYSTEM_PATHS ON)/#SET(Boost_NO_SYSTEM_PATHS ON)/g' z5-src/CMakeLists.txt
        else
            sed -i '/^SET(Boost_NO_SYSTEM_PATHS ON)$/s/^/#/' z5-src/CMakeLists.txt
        fi
        if [ $use_vcpkg = true ]; then
            cat > z5-src/vcpkg.json <<EOF
{
    "\$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",
    "name": "z5",
    "version": "0.1.0",
    "dependencies": [ "boost-core", "boost-filesystem", "nlohmann-json", "blosc" ]
}
EOF
        fi
        mkdir -p z5-src/build
        pushd z5-src/build >/dev/null
        cmake ${params_gen[@]+"${params_gen[@]}"} -Dxtl_DIR="${projectpath}/third_party/xtl/share/cmake/xtl" \
        -Dxtensor_DIR="${xtensor_CMAKE_DIR}" \
        -Dxsimd_DIR="${projectpath}/third_party/xsimd/lib/cmake/xsimd" \
        -DBUILD_Z5PY=OFF -DWITH_ZLIB=ON -DWITH_LZ4=ON -DWITH_BLOSC=ON \
        -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/z5" ${params_vcpkg[@]+"${params_vcpkg[@]}"} ..
        make install
        popd >/dev/null
    fi
    params+=(-Dxtl_DIR="${projectpath}/third_party/xtl/share/cmake/xtl" \
    -Dxtensor_DIR="${xtensor_CMAKE_DIR}" \
    -Dxsimd_DIR="${projectpath}/third_party/xsimd/lib/cmake/xsimd" \
    -Dz5_DIR="${projectpath}/third_party/z5/lib/cmake/z5")
fi

if $build_with_cuda_support; then
    if [ ! -d "./tiny-cuda-nn" ]; then
        echo "------------------------"
        echo "downloading tiny-cuda-nn"
        echo "------------------------"
        git clone https://github.com/chrismile/tiny-cuda-nn.git tiny-cuda-nn --recurse-submodules
        pushd tiny-cuda-nn >/dev/null
        git checkout activations
        popd >/dev/null
    fi
    if [ ! -d "./quick-mlp" ]; then
        echo "------------------------"
        echo "  downloading QuickMLP  "
        echo "------------------------"
        git clone https://github.com/chrismile/quick-mlp.git quick-mlp --recurse-submodules
    fi
fi

if $build_with_skia_support; then
    if $skia_link_dynamically; then
        out_dir="out/Shared"
    else
        out_dir="out/Static"
    fi
    if [ ! -d "./skia/$out_dir" ]; then
        echo "------------------------"
        echo "    downloading Skia    "
        echo "------------------------"
        if [ ! -d "./skia" ]; then
            git clone https://skia.googlesource.com/skia.git
            pushd skia >/dev/null
            python3 tools/git-sync-deps
            bin/fetch-ninja
        else
            pushd skia >/dev/null
        fi
        if $skia_link_dynamically; then
            bin/gn gen out/Shared --args='is_official_build=true is_component_build=true is_debug=false skia_use_vulkan=true skia_use_system_harfbuzz=false skia_use_fontconfig=false'
            third_party/ninja/ninja -C out/Shared
            params+=(-DSkia_DIR="${projectpath}/third_party/skia" -DSkia_BUILD_TYPE=Shared)
        else
            bin/gn gen out/Static --args='is_official_build=true is_debug=false skia_use_vulkan=true skia_use_system_harfbuzz=false skia_use_fontconfig=false'
            third_party/ninja/ninja -C out/Static
            params+=(-DSkia_DIR="${projectpath}/third_party/skia" -DSkia_BUILD_TYPE=Static)
        fi
        popd >/dev/null
    fi
fi

if $build_with_vkvg_support; then
    if [ ! -d "./vkvg" ]; then
        echo "------------------------"
        echo "    downloading VKVG    "
        echo "------------------------"
        if [ -d "./vkvg-src" ]; then
            rm -rf "./vkvg-src"
        fi
        git clone --recursive https://github.com/chrismile/vkvg vkvg-src
        mkdir -p vkvg-src/build
        pushd vkvg-src/build >/dev/null
        cmake .. ${params_gen[@]+"${params_gen[@]}"} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/vkvg" \
        -DVKVG_ENABLE_VK_SCALAR_BLOCK_LAYOUT=ON -DVKVG_ENABLE_VK_TIMELINE_SEMAPHORE=ON \
        -DVKVG_USE_FONTCONFIG=OFF -DVKVG_USE_HARFBUZZ=OFF -DVKVG_BUILD_TESTS=OFF
        make -j $(nproc)
        make install
        popd >/dev/null
    fi
    params+=(-Dvkvg_DIR="${projectpath}/third_party/vkvg")
fi

if $build_with_osqp_support; then
    if [ ! -d "./osqp" ]; then
        echo "------------------------"
        echo "    downloading OSQP    "
        echo "------------------------"
        if [ -d "./osqp-src" ]; then
            rm -rf "./osqp-src"
        fi
        git clone https://github.com/osqp/osqp osqp-src
        mkdir -p osqp-src/build
        pushd osqp-src/build >/dev/null
        cmake .. ${params_gen[@]+"${params_gen[@]}"} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/osqp"
        make -j $(nproc)
        make install
        popd >/dev/null
    fi
    params+=(-Dosqp_DIR="${projectpath}/third_party/osqp/lib/cmake/osqp")
fi

if [ ! -d "${projectpath}/third_party/limbo" ]; then
    echo "------------------------"
    echo "    downloading limbo   "
    echo "------------------------"
    git clone --recursive https://github.com/chrismile/limbo.git "${projectpath}/third_party/limbo"
    pushd limbo >/dev/null
    git checkout fixes
    popd >/dev/null
fi

if $build_with_zink_support; then
    if [ ! -d "./mesa" ]; then
        # Download libdrm and Mesa.
        LIBDRM_VERSION=libdrm-2.4.115
        MESA_VERSION=mesa-23.1.3
        wget https://dri.freedesktop.org/libdrm/${LIBDRM_VERSION}.tar.xz
        wget https://archive.mesa3d.org/${MESA_VERSION}.tar.xz
        tar -xvf ${LIBDRM_VERSION}.tar.xz
        tar -xvf ${MESA_VERSION}.tar.xz

        # Install all dependencies.
        pip3 install --user meson mako
        # TODO: Add support for other operating systems.
        sudo apt-get -y build-dep mesa
        sudo apt install -y ninja libxcb-dri3-dev libxcb-present-dev libxshmfence-dev

        pushd ${LIBDRM_VERSION} >/dev/null
        meson builddir/ --prefix="${projectpath}/third_party/mesa"
        ninja -C builddir/ install
        popd >/dev/null

        pushd ${MESA_VERSION} >/dev/null
        PKG_CONFIG_PATH="${projectpath}/third_party/mesa/lib/x86_64-linux-gnu/pkgconfig" \
        meson setup builddir/ -Dprefix="${projectpath}/third_party/mesa" \
        -Dgallium-drivers=zink,swrast -Dvulkan-drivers= -Dbuildtype=release \
        -Dgallium-va=disabled -Dglx=dri -Dplatforms=x11 -Degl=enabled -Dglvnd=true
        meson install -C builddir/
        popd >/dev/null
    fi
    if [[ -z "${LD_LIBRARY_PATH+x}" ]]; then
        export LD_LIBRARY_PATH="${projectpath}/third_party/mesa/lib/x86_64-linux-gnu"
    else
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${projectpath}/third_party/mesa/lib/x86_64-linux-gnu"
    fi
    export __GLX_VENDOR_LIBRARY_NAME=mesa
    export MESA_LOADER_DRIVER_OVERRIDE=zink
    export GALLIUM_DRIVER=zink
    params+=(-DUSE_ZINK=ON)
fi

popd >/dev/null # back to project root

if [ $debug = true ]; then
    echo "------------------------"
    echo "  building in debug     "
    echo "------------------------"
else
    echo "------------------------"
    echo "  building in release   "
    echo "------------------------"
fi

if [ -f "./$build_dir/CMakeCache.txt" ]; then
    if grep -q vcpkg_installed "./$build_dir/CMakeCache.txt"; then
        cache_uses_vcpkg=true
    else
        cache_uses_vcpkg=false
    fi
    if ([ $use_vcpkg = true ] && [ $cache_uses_vcpkg = false ]) || ([ $use_vcpkg = false ] && [ $cache_uses_vcpkg = true ]); then
        echo "Removing old application build cache..."
        if [ -d "./$build_dir_debug" ]; then
            rm -rf "./$build_dir_debug"
        fi
        if [ -d "./$build_dir_release" ]; then
            rm -rf "./$build_dir_release"
        fi
        if [ -d "./$destination_dir" ]; then
            rm -rf "./$destination_dir"
        fi
    fi
fi

mkdir -p $build_dir

echo "------------------------"
echo "      generating        "
echo "------------------------"
pushd $build_dir >/dev/null
cmake .. \
    -DCMAKE_BUILD_TYPE=$cmake_config \
    -Dsgl_DIR="$projectpath/third_party/sgl/install/lib/cmake/sgl/" \
    ${params_gen[@]+"${params_gen[@]}"} ${params_link[@]+"${params_link[@]}"} \
    ${params_vcpkg[@]+"${params_vcpkg[@]}"} ${params[@]+"${params[@]}"}
popd >/dev/null

echo "------------------------"
echo "      compiling         "
echo "------------------------"
if [ $use_macos = true ]; then
    cmake --build $build_dir --parallel $(sysctl -n hw.ncpu)
elif [ $use_vcpkg = true ]; then
    cmake --build $build_dir --parallel $(nproc)
else
    pushd "$build_dir" >/dev/null
    make -j $(nproc)
    popd >/dev/null
fi

echo "------------------------"
echo "   copying new files    "
echo "------------------------"

# https://stackoverflow.com/questions/2829613/how-do-you-tell-if-a-string-contains-another-string-in-posix-sh
contains() {
    string="$1"
    substring="$2"
    if test "${string#*$substring}" != "$string"
    then
        return 0
    else
        return 1
    fi
}
startswith() {
    string="$1"
    prefix="$2"
    if test "${string#$prefix}" != "$string"
    then
        return 0
    else
        return 1
    fi
}

if $use_msys; then
    mkdir -p $destination_dir/bin

    # Copy sgl to the destination directory.
    if [ $debug = true ] ; then
        cp "./third_party/sgl/install/bin/libsgld.dll" "$destination_dir/bin"
    else
        cp "./third_party/sgl/install/bin/libsgl.dll" "$destination_dir/bin"
    fi

    # Copy the application to the destination directory.
    cp "$build_dir/Correrender.exe" "$destination_dir/bin"

    # Copy all dependencies of the application to the destination directory.
    ldd_output="$(ldd $destination_dir/bin/Correrender.exe)"
    for library in $ldd_output
    do
        if [[ $library == "$MSYSTEM_PREFIX"* ]] ;
        then
            cp "$library" "$destination_dir/bin"
        fi
        if [[ $library == libpython* ]] ;
        then
            tmp=${library#*lib}
            Python3_VERSION=${tmp%.dll}
        fi
    done
elif [ $use_macos = true ] && [ $use_vcpkg = true ]; then
    [ -d $destination_dir ] || mkdir $destination_dir
    rsync -a "$build_dir/Correrender.app/Contents/MacOS/Correrender" $destination_dir
elif [ $use_macos = true ] && [ $use_vcpkg = false ]; then
    brew_prefix="$(brew --prefix)"
    mkdir -p $destination_dir

    if [ -d "$destination_dir/Correrender.app" ]; then
        rm -rf "$destination_dir/Correrender.app"
    fi

    # Copy the application to the destination directory.
    cp -a "$build_dir/Correrender.app" "$destination_dir"

    # Copy sgl to the destination directory.
    if [ $debug = true ] ; then
        cp "./third_party/sgl/install/lib/libsgld.dylib" "$binaries_dest_dir"
    else
        cp "./third_party/sgl/install/lib/libsgl.dylib" "$binaries_dest_dir"
    fi

    # Copy all dependencies of the application and sgl to the destination directory.
    rsync -a "$VULKAN_SDK/lib/libMoltenVK.dylib" "$binaries_dest_dir"
    copy_dependencies_recursive() {
        local binary_path="$1"
        local binary_source_folder=$(dirname "$binary_path")
        local binary_name=$(basename "$binary_path")
        local binary_target_path="$binaries_dest_dir/$binary_name"
        if contains "$(file "$binary_target_path")" "dynamically linked shared library"; then
            install_name_tool -id "@executable_path/$binary_name" "$binary_target_path" &> /dev/null
        fi
        local otool_output="$(otool -L "$binary_path")"
        local otool_output=${otool_output#*$'\n'}
        while read -r line
        do
            local stringarray=($line)
            local library=${stringarray[0]}
            local library_name=$(basename "$library")
            local library_target_path="$binaries_dest_dir/$library_name"
            if ! startswith "$library" "@rpath/" \
                && ! startswith "$library" "@loader_path/" \
                && ! startswith "$library" "/System/Library/Frameworks/" \
                && ! startswith "$library" "/usr/lib/"
            then
                install_name_tool -change "$library" "@executable_path/$library_name" "$binary_target_path" &> /dev/null

                if [ ! -f "$library_target_path" ]; then
                    cp "$library" "$binaries_dest_dir"
                    copy_dependencies_recursive "$library"
                fi
            elif startswith "$library" "@rpath/"; then
                install_name_tool -change "$library" "@executable_path/$library_name" "$binary_target_path" &> /dev/null

                local rpath_grep_string="$(otool -l "$binary_target_path" | grep RPATH -A2)"
                local counter=0
                while read -r grep_rpath_line
                do
                    if [ $(( counter % 4 )) -eq 2 ]; then
                        local stringarray_grep_rpath_line=($grep_rpath_line)
                        local rpath=${stringarray_grep_rpath_line[1]}
                        if startswith "$rpath" "@loader_path"; then
                            rpath="${rpath/@loader_path/$binary_source_folder}"
                        fi
                        local library_rpath="${rpath}${library#"@rpath"}"

                        if [ -f "$library_rpath" ]; then
                            if [ ! -f "$library_target_path" ]; then
                                cp "$library_rpath" "$binaries_dest_dir"
                                copy_dependencies_recursive "$library_rpath"
                            fi
                            break
                        fi
                    fi
                    counter=$((counter + 1))
                done < <(echo "$rpath_grep_string")
            fi
        done < <(echo "$otool_output")
    }
    copy_dependencies_recursive "$build_dir/Correrender.app/Contents/MacOS/Correrender"
    if [ $debug = true ]; then
        copy_dependencies_recursive "./third_party/sgl/install/lib/libsgld.dylib"
    else
        copy_dependencies_recursive "./third_party/sgl/install/lib/libsgl.dylib"
    fi

    # Fix code signing for arm64.
    for filename in $binaries_dest_dir/*
    do
        if contains "$(file "$filename")" "arm64"; then
            codesign --force -s - "$filename" &> /dev/null
        fi
    done
${copy_dependencies_macos_post}
else
    mkdir -p $destination_dir/bin

    # Copy the application to the destination directory.
    rsync -a "$build_dir/Correrender" "$destination_dir/bin"

    # Copy all dependencies of the application to the destination directory.
    ldd_output="$(ldd $build_dir/Correrender)"

    library_blacklist=(
        "libOpenGL" "libGLdispatch" "libGL.so" "libGLX.so"
        "libwayland" "libffi." "libX" "libxcb" "libxkbcommon"
        "ld-linux" "libdl." "libutil." "libm." "libc." "libpthread." "libbsd." "librt."
    )
    if [ $use_vcpkg = true ]; then
        # We build with libstdc++.so and libgcc_s.so statically. If we were to ship them, libraries opened with dlopen will
        # use our, potentially older, versions. Then, we will get errors like "version `GLIBCXX_3.4.29' not found" when
        # the Vulkan loader attempts to load a Vulkan driver that was built with a never version of libstdc++.so.
        # I tried to solve this by using "patchelf --replace-needed" to directly link to the patch version of libstdc++.so,
        # but that made no difference whatsoever for dlopen.
        library_blacklist+=("libstdc++.so")
        library_blacklist+=("libgcc_s.so")
    fi
    for library in $ldd_output
    do
        if [[ $library != "/"* ]]; then
            continue
        fi
        is_blacklisted=false
        for blacklisted_library in ${library_blacklist[@]+"${library_blacklist[@]}"}; do
            if [[ "$library" == *"$blacklisted_library"* ]]; then
                is_blacklisted=true
                break
            fi
        done
        if [ $is_blacklisted = true ]; then
            continue
        fi
        # TODO: Add blacklist entries for pulseaudio and dependencies when not using vcpkg.
        #cp "$library" "$destination_dir/bin"
        #patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/$(basename "$library")"
        if [ $use_vcpkg = true ]; then
            cp "$library" "$destination_dir/bin"
            patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/$(basename "$library")"
        fi
    done
    patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/Correrender"
fi


# Copy the docs to the destination directory.
cp "README.md" "$destination_dir"
if [ ! -d "$destination_dir/LICENSE" ]; then
    mkdir -p "$destination_dir/LICENSE"
    cp -r "docs/license-libraries/." "$destination_dir/LICENSE/"
    cp -r "LICENSE" "$destination_dir/LICENSE/LICENSE-correrender.txt"
    cp -r "submodules/IsosurfaceCpp/LICENSE" "$destination_dir/LICENSE/graphics/LICENSE-isosurfacecpp.txt"
fi
if [ ! -d "$destination_dir/docs" ]; then
    cp -r "docs" "$destination_dir"
fi

# Create a run script.
if $use_msys; then
    printf "@echo off\npushd %%~dp0\npushd bin\nstart \"\" Correrender.exe\n" > "$destination_dir/run.bat"
elif $use_macos; then
    printf "#!/bin/sh\npushd \"\$(dirname \"\$0\")\" >/dev/null\n./Correrender.app/Contents/MacOS/Correrender\npopd\n" > "$destination_dir/run.sh"
    chmod +x "$destination_dir/run.sh"
else
    printf "#!/bin/bash\npushd \"\$(dirname \"\$0\")/bin\" >/dev/null\n./Correrender\npopd\n" > "$destination_dir/run.sh"
    chmod +x "$destination_dir/run.sh"
fi

# Replicability Stamp mode.
if $replicability; then
    mkdir -p "./Data/VolumeDataSets"
    if [ ! -f "./Data/VolumeDataSets/linear_4x4.nc" ]; then
        #echo "------------------------"
        #echo "generating synthetic data"
        #echo "------------------------"
        #pushd scripts >/dev/null
        #python3 generate_synth_box_ensembles.py
        #popd >/dev/null
        echo "------------------------"
        echo "downloading synthetic data"
        echo "------------------------"
        curl --show-error --fail \
        https://zenodo.org/records/10018860/files/linear_4x4.nc --output "./Data/VolumeDataSets/linear_4x4.nc"
    fi
    if [ ! -f "./Data/VolumeDataSets/datasets.json" ]; then
        printf "{ \"datasets\": [ { \"name\": \"linear_4x4\", \"filename\": \"linear_4x4.nc\" } ] }" >> ./Data/VolumeDataSets/datasets.json
    fi
    params_run+=(--replicability)
fi


# Run the program as the last step.
echo ""
echo "All done!"
pushd $build_dir >/dev/null

if $use_msys; then
    if [[ -z "${PATH+x}" ]]; then
        export PATH="${projectpath}/third_party/sgl/install/bin"
    elif [[ ! "${PATH}" == *"${projectpath}/third_party/sgl/install/bin"* ]]; then
        export PATH="${projectpath}/third_party/sgl/install/bin:$PATH"
    fi
elif $use_macos; then
    if [ -z "${DYLD_LIBRARY_PATH+x}" ]; then
        export DYLD_LIBRARY_PATH="${projectpath}/third_party/sgl/install/lib"
    elif contains "${DYLD_LIBRARY_PATH}" "${projectpath}/third_party/sgl/install/lib"; then
        export DYLD_LIBRARY_PATH="DYLD_LIBRARY_PATH:${projectpath}/third_party/sgl/install/lib"
    fi
else
  if [[ -z "${LD_LIBRARY_PATH+x}" ]]; then
      export LD_LIBRARY_PATH="${projectpath}/third_party/sgl/install/lib"
  elif [[ ! "${LD_LIBRARY_PATH}" == *"${projectpath}/third_party/sgl/install/lib"* ]]; then
      export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${projectpath}/third_party/sgl/install/lib"
  fi
fi

if [ $run_program = true ] && [ $use_macos = false ]; then
    ./Correrender ${params_run[@]+"${params_run[@]}"}
elif [ $run_program = true ] && [ $use_macos = true ]; then
    #open ./Correrender.app
    #open ./Correrender.app --args --perf
    ./Correrender.app/Contents/MacOS/Correrender ${params_run[@]+"${params_run[@]}"}
fi
