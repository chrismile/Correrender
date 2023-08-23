#!/bin/bash

# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Christoph Neuhauser, Felix Brendel
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

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJECTPATH="$SCRIPTPATH"
pushd $SCRIPTPATH > /dev/null

run_program=true
debug=false
glibcxx_debug=false
link_dynamic=false
params_link=()
build_dir_debug=".build_debug"
build_dir_release=".build_release"
build_with_zarr_support=true
build_with_cuda_support=true
build_with_skia_support=false
skia_link_dynamically=true
build_with_vkvg_support=false
build_with_osqp_support=true

# Process command line arguments.
custom_glslang=false
for ((i=1;i<=$#;i++));
do
    if [ ${!i} = "--do-not-run" ]; then
        run_program=false
    elif [ ${!i} = "--debug" ] || [ ${!i} = "debug" ]; then
        debug=true
    elif [ ${!i} = "--glibcxx-debug" ]; then
        glibcxx_debug=true
    elif [ ${!i} = "--custom-glslang" ]; then
        custom_glslang=true
    elif [ ${!i} = "--link-static" ]; then
        link_dynamic=false
    elif [ ${!i} = "--link-dynamic" ]; then
        link_dynamic=true
    elif [ ${!i} = "--vcpkg-triplet" ]; then
        ((i++))
        params_link+=(-DVCPKG_TARGET_TRIPLET=${!i})
        continue
    fi
done

if [ $debug = true ]; then
    cmake_config="Debug"
    build_dir=$build_dir_debug
else
    cmake_config="Release"
    build_dir=$build_dir_release
fi
if [ $link_dynamic = true ]; then
    params_link+=(-DVCPKG_TARGET_TRIPLET=x64-linux-dynamic)
fi
destination_dir="Shipping"

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

if command -v apt &> /dev/null; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        sudo apt install -y cmake git curl pkg-config build-essential patchelf
    fi

    # Dependencies of vcpkg GLEW and SDL2[x11, wayland] ports.
    if ! is_installed_apt "libxmu-dev" || ! is_installed_apt "libxi-dev" || ! is_installed_apt "libgl-dev"; then
        sudo apt install libgl-dev libxmu-dev libxi-dev libx11-dev libxft-dev libxext-dev \
        libwayland-dev libxkbcommon-dev libegl1-mesa-dev
    fi
elif command -v pacman &> /dev/null; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        sudo pacman -S cmake git curl pkgconf base-devel patchelf
    fi

    # Dependencies of vcpkg GLEW and Python ports.
    if ! is_installed_pacman "libgl" || ! is_installed_pacman "vulkan-devel" || ! is_installed_pacman "shaderc" \
            || ! is_installed_pacman "openssl"; then
        sudo pacman -S libgl vulkan-devel shaderc openssl
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

    # Dependencies of vcpkg openssl, GLEW, SDL2 and python3 ports.
    if ! is_installed_rpm "perl" || ! is_installed_rpm "libstdc++-devel" || ! is_installed_rpm "libstdc++-static" \
            || ! is_installed_rpm "glew-devel" || ! is_installed_rpm "libXext-devel" \
            || ! is_installed_rpm "vulkan-devel" || ! is_installed_rpm "libshaderc-devel"; then
        sudo yum install -y perl libstdc++-devel libstdc++-static glew-devel vulkan-headers libshaderc-devel libXext-devel
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
if ! command -v pkg-config &> /dev/null; then
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
    PATH="${PROJECTPATH}/third_party/cmake-${cmake_download_version}-linux-x86_64/bin:$PATH"
fi

os_arch="$(uname -m)"
if [[ ! -v VULKAN_SDK ]]; then
    echo "------------------------"
    echo "searching for Vulkan SDK"
    echo "------------------------"

    found_vulkan=false

    if [ -d "VulkanSDK" ]; then
        VK_LAYER_PATH=""
        source "VulkanSDK/$(ls VulkanSDK)/setup-env.sh"
        export PKG_CONFIG_PATH="$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig")"
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
        prefix_path=$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch")
        sed -i '3s;.*;prefix=\"'$prefix_path'\";' "$shaderc_pkgconfig_file"
        sed -i '5s;.*;libdir=${prefix}/lib;' "$shaderc_pkgconfig_file"
        export PKG_CONFIG_PATH="$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig")"
        found_vulkan=true
    fi

    if ! $found_vulkan; then
        echo "The environment variable VULKAN_SDK is not set but is required in the installation process."
        echo "Please refer to https://vulkan.lunarg.com/sdk/home#linux for instructions on how to install the Vulkan SDK."
        exit 1
    fi
fi

if [ ! -d "./vcpkg" ]; then
    echo "------------------------"
    echo "   fetching vcpkg       "
    echo "------------------------"
    if [[ ! -v VULKAN_SDK ]]; then
        echo "The environment variable VULKAN_SDK is not set but is required in the installation process."
        exit 1
    fi
    git clone --depth 1 https://github.com/Microsoft/vcpkg.git
    vcpkg/bootstrap-vcpkg.sh -disableMetrics
    vcpkg/vcpkg install
fi

params_sgl=()
params=()

if [ $link_dynamic = false ]; then
    params_sgl+=(-DBUILD_STATIC_LIBRARY=On)
fi

if $glibcxx_debug; then
    params_sgl+=(-DUSE_GLIBCXX_DEBUG=On)
    params+=(-DUSE_GLIBCXX_DEBUG=On)
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
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/glslang" ..
        make -j $(nproc)
        make install
        popd >/dev/null
        popd >/dev/null
    fi
    params_sgl+=(-Dglslang_DIR="${PROJECTPATH}/third_party/glslang" -DUSE_SHADERC=Off)
fi

if [ ! -d "./sgl" ]; then
    echo "------------------------"
    echo "     fetching sgl       "
    echo "------------------------"
    git clone --depth 1 https://github.com/chrismile/sgl.git
fi

if [ -f "./sgl/$build_dir/CMakeCache.txt" ]; then
    if ! grep -q vcpkg_installed "./sgl/$build_dir/CMakeCache.txt"; then
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
         -DCMAKE_TOOLCHAIN_FILE="../../vcpkg/scripts/buildsystems/vcpkg.cmake" \
         -DCMAKE_INSTALL_PREFIX="../install" \
         -DUSE_STATIC_STD_LIBRARIES=On "${params_link[@]}" "${params_sgl[@]}"
    popd >/dev/null

    pushd $build_dir_release >/dev/null
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE="../../vcpkg/scripts/buildsystems/vcpkg.cmake" \
        -DCMAKE_INSTALL_PREFIX="../install" \
        -DUSE_STATIC_STD_LIBRARIES=On "${params_link[@]}" "${params_sgl[@]}"
    popd >/dev/null

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
        cmake -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/xtl" ..
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
        cmake -Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
        -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/xtensor" ..
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
        cmake -Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
        -DENABLE_XTL_COMPLEX=ON \
        -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/xsimd" ..
        make install
        popd >/dev/null
    fi

    # Seems like xtensor can install its CMake config either to the share or lib folder.
    if [ -d "${PROJECTPATH}/third_party/xtensor/share/cmake/xtensor" ]; then
        xtensor_CMAKE_DIR="${PROJECTPATH}/third_party/xtensor/share/cmake/xtensor"
    else
        xtensor_CMAKE_DIR="${PROJECTPATH}/third_party/xtensor/lib/cmake/xtensor"
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
        sed -i '/^SET(Boost_NO_SYSTEM_PATHS ON)$/s/^/#/' z5-src/CMakeLists.txt
        cat > z5-src/vcpkg.json <<EOF
{
    "\$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",
    "name": "z5",
    "version": "0.1.0",
    "dependencies": [ "boost-core", "boost-filesystem", "nlohmann-json", "blosc" ]
}
EOF
        mkdir -p z5-src/build
        pushd z5-src/build >/dev/null
        cmake -Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
        -Dxtensor_DIR="${xtensor_CMAKE_DIR}" \
        -Dxsimd_DIR="${PROJECTPATH}/third_party/xsimd/lib/cmake/xsimd" \
        -DBUILD_Z5PY=OFF -DWITH_ZLIB=ON -DWITH_LZ4=ON -DWITH_BLOSC=ON \
        -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/z5" \
        -DCMAKE_TOOLCHAIN_FILE="$PROJECTPATH/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" ..
        make install
        popd >/dev/null
    fi
    params+=(-Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
    -Dxtensor_DIR="${xtensor_CMAKE_DIR}" \
    -Dxsimd_DIR="${PROJECTPATH}/third_party/xsimd/lib/cmake/xsimd" \
    -Dz5_DIR="${PROJECTPATH}/third_party/z5/lib/cmake/z5")
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
            params+=(-DSkia_DIR="${PROJECTPATH}/third_party/skia" -DSkia_BUILD_TYPE=Shared)
        else
            bin/gn gen out/Static --args='is_official_build=true is_debug=false skia_use_vulkan=true skia_use_system_harfbuzz=false skia_use_fontconfig=false'
            third_party/ninja/ninja -C out/Static
            params+=(-DSkia_DIR="${PROJECTPATH}/third_party/skia" -DSkia_BUILD_TYPE=Static)
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
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/vkvg" \
        -DVKVG_ENABLE_VK_SCALAR_BLOCK_LAYOUT=ON -DVKVG_ENABLE_VK_TIMELINE_SEMAPHORE=ON \
        -DVKVG_USE_FONTCONFIG=OFF -DVKVG_USE_HARFBUZZ=OFF -DVKVG_BUILD_TESTS=OFF
        make -j $(nproc)
        make install
        params+=(-Dvkvg_DIR="${PROJECTPATH}/third_party/vkvg")
        popd >/dev/null
    fi
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
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/osqp"
        make -j $(nproc)
        make install
        popd >/dev/null
    fi
    params+=(-Dosqp_DIR="${PROJECTPATH}/third_party/osqp/lib/cmake/osqp")
fi

if [ ! -d "${PROJECTPATH}/third_party/limbo" ]; then
    echo "------------------------"
    echo "    downloading limbo   "
    echo "------------------------"
    git clone --recursive https://github.com/chrismile/limbo.git "${PROJECTPATH}/third_party/limbo"
    pushd limbo >/dev/null
    git checkout fixes
    popd >/dev/null
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
    if ! grep -q vcpkg_installed "./$build_dir/CMakeCache.txt"; then
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
      -DCMAKE_TOOLCHAIN_FILE="$PROJECTPATH/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" \
      -DCMAKE_BUILD_TYPE=$cmake_config \
      -Dsgl_DIR="$PROJECTPATH/third_party/sgl/install/lib/cmake/sgl/" \
      -DUSE_STATIC_STD_LIBRARIES=On "${params_link[@]}" "${params[@]}"
popd >/dev/null

echo "------------------------"
echo "      compiling         "
echo "------------------------"
cmake --build $build_dir --parallel $(nproc)

echo "------------------------"
echo "   copying new files    "
echo "------------------------"
mkdir -p $destination_dir/bin

# Copy the application to the destination directory.
rsync -a "$build_dir/Correrender" "$destination_dir/bin"

# Copy all dependencies of the application to the destination directory.
ldd_output="$(ldd $build_dir/Correrender)"
library_blacklist=(
    "libOpenGL" "libGLdispatch" "libGL.so" "libGLX.so"
    "libwayland" "libffi." "libX" "libxcb" "libxkbcommon"
    "ld-linux" "libdl." "libutil." "libm." "libc." "libpthread." "libbsd." "librt."
    # We build with libstdc++.so and libgcc_s.so statically. If we were to ship them, libraries opened with dlopen will
    # use our, potentially older, versions. Then, we will get errors like "version `GLIBCXX_3.4.29' not found" when
    # the Vulkan loader attempts to load a Vulkan driver that was built with a never version of libstdc++.so.
    # I tried to solve this by using "patchelf --replace-needed" to directly link to the patch version of libstdc++.so,
    # but that made no difference whatsoever for dlopen.
    "libstdc++.so" "libgcc_s.so"
)
# Get name of libstdc++.so.* and path to it.
#for library in $ldd_output
#do
#  if [[ $library != "/"* ]]; then
#      continue
#  fi
#  if [[ "$(basename $library)" == "libstdc++.so"* ]]; then
#      libstdcpp_so_path="$library"
#      libstdcpp_so_filename_original="$(basename "$library")"
#      libstdcpp_so_filename_resolved="$(basename "$(readlink -f "$library")")"
#  fi
#done
for library in $ldd_output
do
    if [[ $library != "/"* ]]; then
        continue
    fi
    is_blacklisted=false
    for blacklisted_library in ${library_blacklist[@]}; do
        if [[ "$library" == *"$blacklisted_library"* ]]; then
            is_blacklisted=true
            break
        fi
    done
    if [ $is_blacklisted = true ]; then
        continue
    fi
    #if [[ "$(basename $library)" == "libstdc++.so"* ]]; then
    #    cp "$(readlink -f "$library")" "$destination_dir/bin"
    #    patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/$(basename "$(readlink -f "$library")")"
    #else
    #    cp "$library" "$destination_dir/bin"
    #    patchelf --replace-needed "$libstdcpp_so_filename_original" "$libstdcpp_so_filename_resolved" "$destination_dir/bin/$(basename "$library")"
    #    patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/$(basename "$library")"
    #fi
    cp "$library" "$destination_dir/bin"
    patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/$(basename "$library")"
done
#patchelf --replace-needed "$libstdcpp_so_filename_original" "$libstdcpp_so_filename_resolved" "$destination_dir/bin/Correrender"
patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/Correrender"

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
printf "#!/bin/bash\npushd \"\$(dirname \"\$0\")/bin\" >/dev/null\n./Correrender\npopd >/dev/null\n" > "$destination_dir/run.sh"
chmod +x "$destination_dir/run.sh"


# Run the program as the last step.
echo ""
echo "All done!"
pushd $build_dir >/dev/null

if [[ -z "${LD_LIBRARY_PATH+x}" ]]; then
    export LD_LIBRARY_PATH="${PROJECTPATH}/third_party/sgl/install/lib"
elif [[ ! "${LD_LIBRARY_PATH}" == *"${PROJECTPATH}/third_party/sgl/install/lib"* ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${PROJECTPATH}/third_party/sgl/install/lib"
fi

if [ $run_program = true ]; then
    ./Correrender
fi
