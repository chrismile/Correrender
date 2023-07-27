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

debug=false
build_dir_debug=".build_debug"
build_dir_release=".build_release"
destination_dir="Shipping"
build_with_zarr_support=true
build_with_skia_support=false
skia_link_dynamically=true
# VKVG support is disabled due to: https://github.com/jpbruyere/vkvg/issues/140
build_with_vkvg_support=false
build_with_osqp_support=true

# Process command line arguments.
custom_glslang=false
for ((i=1;i<=$#;i++));
do
    if [ ${!i} = "--custom-glslang" ]; then
        custom_glslang=true
    fi
done

is_installed_pacman() {
    local pkg_name="$1"
    if pacman -Qs $pkg_name > /dev/null; then
        return 0
    else
        return 1
    fi
}

if command -v pacman &> /dev/null && [ ! -d $build_dir_debug ] && [ ! -d $build_dir_release ]; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null \
            || ! command -v wget &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        pacman --noconfirm -S --needed make git wget mingw64/mingw-w64-x86_64-cmake \
        mingw64/mingw-w64-x86_64-gcc mingw64/mingw-w64-x86_64-gdb
    fi

    if ! is_installed_pacman "mingw-w64-x86_64-glm" \
            || ! is_installed_pacman "mingw-w64-x86_64-libpng" \
            || ! is_installed_pacman "mingw-w64-x86_64-tinyxml2" \
            || ! is_installed_pacman "mingw-w64-x86_64-boost" \
            || ! is_installed_pacman "mingw-w64-x86_64-libarchive" \
            || ! is_installed_pacman "mingw-w64-x86_64-SDL2" \
            || ! is_installed_pacman "mingw-w64-x86_64-SDL2_image" \
            || ! is_installed_pacman "mingw-w64-x86_64-glew" \
            || ! is_installed_pacman "mingw-w64-x86_64-glfw" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-headers" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-loader" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-validation-layers" \
            || ! is_installed_pacman "mingw-w64-x86_64-shaderc" \
            || ! is_installed_pacman "mingw-w64-x86_64-opencl-headers" \
            || ! is_installed_pacman "mingw-w64-x86_64-opencl-icd" \
            || ! is_installed_pacman "mingw-w64-x86_64-jsoncpp" \
            || ! is_installed_pacman "mingw-w64-x86_64-nlohmann-json" \
            || ! is_installed_pacman "mingw-w64-x86_64-netcdf" \
            || ! is_installed_pacman "mingw-w64-x86_64-eccodes" \
            || ! is_installed_pacman "mingw-w64-x86_64-blosc" \
            || ! is_installed_pacman "mingw-w64-x86_64-python" \
            || ! is_installed_pacman "mingw-w64-x86_64-eigen3" \
            || ! is_installed_pacman "mingw-w64-x86_64-nlopt" \
            || ! is_installed_pacman "mingw-w64-x86_64-libtiff"; then
        echo "------------------------"
        echo "installing dependencies "
        echo "------------------------"
        pacman --noconfirm -S --needed \
        mingw64/mingw-w64-x86_64-glm mingw64/mingw-w64-x86_64-libpng mingw64/mingw-w64-x86_64-tinyxml2 \
        mingw64/mingw-w64-x86_64-boost mingw64/mingw-w64-x86_64-libarchive \
        mingw64/mingw-w64-x86_64-SDL2 mingw64/mingw-w64-x86_64-SDL2_image mingw64/mingw-w64-x86_64-glew \
        mingw64/mingw-w64-x86_64-glfw \
        mingw64/mingw-w64-x86_64-vulkan-headers mingw64/mingw-w64-x86_64-vulkan-loader \
        mingw64/mingw-w64-x86_64-vulkan-validation-layers mingw64/mingw-w64-x86_64-shaderc \
        mingw64/mingw-w64-x86_64-opencl-headers mingw64/mingw-w64-x86_64-opencl-icd \
        mingw64/mingw-w64-x86_64-jsoncpp mingw64/mingw-w64-x86_64-nlohmann-json \
        mingw64/mingw-w64-x86_64-netcdf \
        mingw64/mingw-w64-x86_64-eccodes mingw64/mingw-w64-x86_64-blosc \
        mingw64/mingw-w64-x86_64-python mingw64/mingw-w64-x86_64-eigen3 mingw64/mingw-w64-x86_64-nlopt
    fi
    if ! (is_installed_pacman "mingw-w64-x86_64-curl" || is_installed_pacman "mingw-w64-x86_64-curl-gnutls" \
          || is_installed_pacman "mingw-w64-x86_64-curl-winssl"); then
        pacman --noconfirm -S --needed mingw64/mingw-w64-x86_64-curl
    fi
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

params_sgl=()

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
        cmake -G "MSYS Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/glslang" ..
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

if [ ! -d "./sgl/install" ]; then
    echo "------------------------"
    echo "     building sgl       "
    echo "------------------------"

    pushd "./sgl" >/dev/null
    mkdir -p .build_debug
    mkdir -p .build_release

    pushd "$build_dir_debug" >/dev/null
    cmake .. \
         -G "MSYS Makefiles" \
         -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_INSTALL_PREFIX="../install" \
         "${params_sgl[@]}"
    make -j $(nproc)
    make install
    popd >/dev/null

    pushd $build_dir_release >/dev/null
    cmake .. \
        -G "MSYS Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="../install" \
        "${params_sgl[@]}"
    make -j $(nproc)
    make install
    popd >/dev/null

    popd >/dev/null
fi

# CMake parameters for building the application.
params=()

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
        cmake -G "MSYS Makefiles" -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/xtl" ..
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
        cmake -G "MSYS Makefiles" -Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
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
        cmake -G "MSYS Makefiles" -Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
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
        mkdir -p z5-src/build
        pushd z5-src/build >/dev/null
        cmake -G "MSYS Makefiles" -Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
        -Dxtensor_DIR="${xtensor_CMAKE_DIR}" \
        -Dxsimd_DIR="${PROJECTPATH}/third_party/xsimd/lib/cmake/xsimd" \
        -DBUILD_Z5PY=OFF -DWITH_ZLIB=ON -DWITH_LZ4=ON -DWITH_BLOSC=ON \
        -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/z5" ..
        make install
        popd >/dev/null
    fi
    params+=(-Dxtl_DIR="${PROJECTPATH}/third_party/xtl/share/cmake/xtl" \
    -Dxtensor_DIR="${xtensor_CMAKE_DIR}" \
    -Dxsimd_DIR="${PROJECTPATH}/third_party/xsimd/lib/cmake/xsimd" \
    -Dz5_DIR="${PROJECTPATH}/third_party/z5/lib/cmake/z5")
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
        cmake .. -G "MSYS Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/vkvg" \
        -DVKVG_ENABLE_VK_SCALAR_BLOCK_LAYOUT=ON -DVKVG_ENABLE_VK_TIMELINE_SEMAPHORE=ON \
        -DVKVG_USE_FONTCONFIG=OFF -DVKVG_USE_HARFBUZZ=OFF -DVKVG_BUILD_TESTS=OFF
        make -j $(nproc)
        make install
        popd >/dev/null
    fi
    params+=(-Dvkvg_DIR="${PROJECTPATH}/third_party/vkvg")
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
        cmake .. -G "MSYS Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PROJECTPATH}/third_party/osqp"
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
    git clone --recursive https://github.com/resibots/limbo.git "${PROJECTPATH}/third_party/limbo"
fi

popd >/dev/null # back to project root

if [ $debug = true ] ; then
    echo "------------------------"
    echo "  building in debug     "
    echo "------------------------"

    cmake_config="Debug"
    build_dir=$build_dir_debug
else
    echo "------------------------"
    echo "  building in release   "
    echo "------------------------"

    cmake_config="Release"
    build_dir=$build_dir_release
fi
mkdir -p $build_dir
ls "$build_dir"

echo "------------------------"
echo "      generating        "
echo "------------------------"
pushd $build_dir >/dev/null
Python3_VERSION="$(find "$MSYSTEM_PREFIX/lib/" -maxdepth 1 -type d -name 'python*' -printf "%f" -quit)"
cmake .. \
    -G "MSYS Makefiles" \
    -DPython3_FIND_REGISTRY=NEVER \
    -DCMAKE_BUILD_TYPE=$cmake_config \
    -Dsgl_DIR="$PROJECTPATH/third_party/sgl/install/lib/cmake/sgl/" \
    -DPYTHONHOME="./python3" \
    -DPYTHONPATH="./python3/lib/$Python3_VERSION" \
    "${params[@]}"
Python3_VERSION=$(cat pythonversion.txt)
popd >/dev/null

echo "------------------------"
echo "      compiling         "
echo "------------------------"
pushd "$build_dir" >/dev/null
make -j $(nproc)
popd >/dev/null

echo ""


echo "------------------------"
echo "   copying new files    "
echo "------------------------"
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

# Copy libopenblas (needed by numpy) and its dependencies to the destination directory.
cp "$MSYSTEM_PREFIX/bin/libopenblas.dll" "$destination_dir/bin"
ldd_output="$(ldd "$MSYSTEM_PREFIX/bin/libopenblas.dll")"
for library in $ldd_output
do
    if [[ $library == "$MSYSTEM_PREFIX"* ]] ;
    then
        cp "$library" "$destination_dir/bin"
    fi
done

# Copy python3 to the destination directory.
if [ ! -d "$destination_dir/bin/python3" ]; then
    mkdir -p "$destination_dir/bin/python3/lib"
    cp -r "$MSYSTEM_PREFIX/lib/$Python3_VERSION" "$destination_dir/bin/python3/lib"
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
printf "@echo off\npushd %%~dp0\npushd bin\nstart \"\" Correrender.exe\n" > "$destination_dir/run.bat"


# Run the program as the last step.
echo "All done!"
pushd $build_dir >/dev/null

if [[ -z "${PATH+x}" ]]; then
    export PATH="${PROJECTPATH}/third_party/sgl/install/bin"
elif [[ ! "${PATH}" == *"${PROJECTPATH}/third_party/sgl/install/bin"* ]]; then
    export PATH="${PROJECTPATH}/third_party/sgl/install/bin:$PATH"
fi
export PYTHONHOME="/mingw64"
./Correrender

