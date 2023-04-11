:: BSD 2-Clause License
::
:: Copyright (c) 2021-2022, Christoph Neuhauser, Felix Brendel
:: All rights reserved.
::
:: Redistribution and use in source and binary forms, with or without
:: modification, are permitted provided that the following conditions are met:
::
:: 1. Redistributions of source code must retain the above copyright notice, this
::    list of conditions and the following disclaimer.
::
:: 2. Redistributions in binary form must reproduce the above copyright notice,
::    this list of conditions and the following disclaimer in the documentation
::    and/or other materials provided with the distribution.
::
:: THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
:: AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
:: IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
:: DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
:: FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
:: DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
:: SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
:: CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
:: OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
:: OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@echo off
setlocal
pushd %~dp0

set debug=false
set build_dir=".build"
set destination_dir="Shipping"
set build_with_cuda_support=true
set build_with_zarr_support=true

:: Leave empty to let cmake try to find the correct paths
set optix_install_dir=""

where cmake >NUL 2>&1 || echo cmake was not found but is required to build the program && exit /b 1

:: Creates a string with, e.g., -G "Visual Studio 17 2022".
:: Needs to be run from a Visual Studio developer PowerShell or command prompt.
if defined VCINSTALLDIR (
    set VCINSTALLDIR_ESC=%VCINSTALLDIR:\=\\%
)
if defined VCINSTALLDIR (
    set "x=%VCINSTALLDIR_ESC:Microsoft Visual Studio\\=" & set "VsPathEnd=%"
)
if defined VCINSTALLDIR (
    set cmake_generator=-G "Visual Studio %VisualStudioVersion:~0,2% %VsPathEnd:~0,4%"
)
if not defined VCINSTALLDIR (
    set cmake_generator=
)

if not exist .\submodules\IsosurfaceCpp\src (
   echo ------------------------
   echo initializing submodules
   echo ------------------------
   git submodule init   || exit /b 1
   git submodule update || exit /b 1
)

if not exist .\third_party\ mkdir .\third_party\
pushd third_party

if not exist .\vcpkg (
   echo ------------------------
   echo    fetching vcpkg
   echo ------------------------
   git clone --depth 1 https://github.com/Microsoft/vcpkg.git || exit /b 1
   call vcpkg\bootstrap-vcpkg.bat -disableMetrics             || exit /b 1
   vcpkg\vcpkg install --triplet=x64-windows                  || exit /b 1
)

if not exist .\sgl (
   echo ------------------------
   echo      fetching sgl
   echo ------------------------
   git clone --depth 1 https://github.com/chrismile/sgl.git   || exit /b 1
)

if not exist .\sgl\install (
   echo ------------------------
   echo      building sgl
   echo ------------------------
   mkdir sgl\.build 2> NUL
   pushd sgl\.build

   cmake .. %cmake_generator% ^
            -DCMAKE_TOOLCHAIN_FILE=../../vcpkg/scripts/buildsystems/vcpkg.cmake ^
            -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_CXX_FLAGS="/MP" || exit /b 1
   cmake --build . --config Debug   -- /m            || exit /b 1
   cmake --build . --config Debug   --target install || exit /b 1
   cmake --build . --config Release -- /m            || exit /b 1
   cmake --build . --config Release --target install || exit /b 1

   popd
)

set cmake_args=-DCMAKE_TOOLCHAIN_FILE="third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
               -DPYTHONHOME="./python3"                                                    ^
               -DCMAKE_CXX_FLAGS="/MP"                                                     ^
               -Dsgl_DIR="third_party/sgl/install/lib/cmake/sgl/"

set eccodes_version=2.26.0
if not exist ".\eccodes-%eccodes_version%-Source" (
    echo ------------------------
    echo   downloading ecCodes
    echo ------------------------
    curl.exe -L "https://confluence.ecmwf.int/download/attachments/45757960/eccodes-%eccodes_version%-Source.tar.gz?api=v2" --output eccodes-%eccodes_version%-Source.tar.gz
    tar -xvzf "eccodes-%eccodes_version%-Source.tar.gz"
)

:: ecCodes needs bash.exe, but it is just a dummy pointing the caller to WSL if not installed.
:: ecCodes is disabled for now due to build errors on Windows.
set use_eccodes=false
:: set bash_output=empty
:: for /f %%i in ('where bash') do set bash_location=%%i
:: IF "%bash_location%"=="" (
::     set bash_location=C:\Windows\System32\bash.exe
:: )
:: IF EXIST "%bash_location%" (
::     goto bash_exists
:: ) ELSE (
::     goto system_bash_not_exists
:: )
::
:: :: goto circumvents problems when not using EnableDelayedExpansion.
:: :bash_exists
:: set bash_output=empty
:: for /f %%i in ('"%bash_location%" --version') do set bash_output=%%i
:: IF "%bash_output%"=="" (
::     set bash_output=empty
:: )
:: :: Output is usually https://aka.ms/wslstore when WSL is not installed.
:: if not x%bash_output:wsl=%==x%bash_output% (
::     set use_eccodes=false
:: ) ELSE (
::     set use_eccodes=true
:: )
:: goto finished
::
:: :system_bash_not_exists
:: set bash_location="C:/Program Files/Git/bin/bash.exe"
:: IF EXIST "%bash_location%" (
::     set "PATH=C:\Program Files\Git\bin;%PATH%"
::     goto bash_exists
:: ) ELSE (
::     goto finished
:: )
::
:: :finished
:: echo bash_location: %bash_location%
:: echo use_eccodes: %use_eccodes%

if %use_eccodes% == true if not exist ".\eccodes-%eccodes_version%" (
    echo ------------------------
    echo    building ecCodes
    echo ------------------------
    pushd eccodes-%eccodes_version%-Source
    if not exist .\build\ mkdir .\build\
    pushd build
    cmake .. %cmake_generator% -DCMAKE_INSTALL_PREFIX=../../eccodes-%eccodes_version% -DCMAKE_CXX_FLAGS="/MP" || exit /b 1
    cmake --build . --config Debug   -- /m            || exit /b 1
    cmake --build . --config Debug   --target install || exit /b 1
    cmake --build . --config Release -- /m            || exit /b 1
    cmake --build . --config Release --target install || exit /b 1
    popd
    popd
)
set cmake_args=%cmake_args% -Deccodes_DIR="third_party/eccodes-%eccodes_version%/lib/cmake/eccodes-%eccodes_version%"

echo %cd%

if %build_with_zarr_support% == true (
    if not exist ".\xtl" (
        echo ------------------------
        echo     downloading xtl
        echo ------------------------
        :: Make sure we have no leftovers from a failed build attempt.
        if exist ".\xtl-src" (
            rmdir /s /q ".\xtl-src"
        )
        git clone https://github.com/xtensor-stack/xtl.git xtl-src
        if not exist .\xtl-src\build\ mkdir .\xtl-src\build\
        pushd "xtl-src\build"
        cmake %cmake_generator% -DCMAKE_TOOLCHAIN_FILE="%~dp0/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
        -DCMAKE_INSTALL_PREFIX="%~dp0/third_party/xtl" ..
        cmake --build . --config Release --target install || exit /b 1
        popd
    )
    if not exist ".\xtensor" (
        echo ------------------------
        echo   downloading xtensor
        echo ------------------------
        :: Make sure we have no leftovers from a failed build attempt.
        if exist ".\xtensor-src" (
            rmdir /s /q ".\xtensor-src"
        )
        git clone https://github.com/xtensor-stack/xtensor.git xtensor-src
        if not exist .\xtensor-src\build\ mkdir .\xtensor-src\build\
        pushd "xtensor-src\build"
        cmake %cmake_generator% -DCMAKE_TOOLCHAIN_FILE="%~dp0/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
        -Dxtl_DIR="%~dp0/third_party/xtl/share/cmake/xtl" ^
        -DCMAKE_INSTALL_PREFIX="%~dp0/third_party/xtensor" ..
        cmake --build . --config Release --target install || exit /b 1
        popd
    )
    if not exist ".\xsimd" (
        echo ------------------------
        echo    downloading xsimd
        echo ------------------------
        :: Make sure we have no leftovers from a failed build attempt.
        if exist ".\xsimd-src" (
            rmdir /s /q ".\xsimd-src"
        )
        git clone https://github.com/xtensor-stack/xsimd.git xsimd-src
        if not exist .\xsimd-src\build\ mkdir .\xsimd-src\build\
        pushd "xsimd-src\build"
        cmake %cmake_generator% -DCMAKE_TOOLCHAIN_FILE="%~dp0/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
        -Dxtl_DIR="%~dp0/third_party/xtl/share/cmake/xtl" ^
        -DENABLE_XTL_COMPLEX=ON ^
        -DCMAKE_INSTALL_PREFIX="%~dp0/third_party/xsimd" ..
        cmake --build . --config Release --target install || exit /b 1
        popd
    )
    if not exist ".\z5" (
        echo ------------------------
        echo      downloading z5
        echo ------------------------
        :: Make sure we have no leftovers from a failed build attempt.
        if exist ".\z5-src" (
            rmdir /s /q ".\z5-src"
        )
        git clone https://github.com/constantinpape/z5.git z5-src
        :: sed -i '/^SET(Boost_NO_SYSTEM_PATHS ON)$/s/^/#/' z5-src/CMakeLists.txt
        powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'SET\(Boost_NO_SYSTEM_PATHS ON\)', '#SET(Boost_NO_SYSTEM_PATHS ON)' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'SET\(BOOST_ROOT \"\$\{CMAKE_PREFIX_PATH\}\/Library\"\)', '#SET(BOOST_ROOT \"${CMAKE_PREFIX_PATH}/Library\")' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'SET\(BOOST_LIBRARYDIR \"\$\{CMAKE_PREFIX_PATH\}\/Library\/lib\"\)', '#SET(BOOST_LIBRARYDIR \"${CMAKE_PREFIX_PATH}/Library/lib\")' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        :: powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'find_package\(Boost 1.63.0 COMPONENTS system filesystem REQUIRED\)', 'find_package(Boost COMPONENTS system filesystem REQUIRED)' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        :: powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'find_package\(Boost 1.63.0 REQUIRED\)', 'find_package(Boost REQUIRED)' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'set\(CMAKE_MODULE_PATH \$\{CMAKE_CURRENT_SOURCE_DIR\}\/cmake\)', 'list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'set\(CMAKE_PREFIX_PATH \$\{CMAKE_PREFIX_PATH\} CACHE PATH \"\"\)', '#set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} CACHE PATH "")' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        powershell -Command "(gc z5-src/CMakeLists.txt) -replace 'if\(NOT WITHIN_TRAVIS\)', 'if(FALSE)' | Out-File -encoding ASCII z5-src/CMakeLists.txt"
        echo {> %~dp0/third_party/z5-src/vcpkg.json
        echo "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",>> %~dp0/third_party/z5-src/vcpkg.json
        echo "name": "z5",>> %~dp0/third_party/z5-src/vcpkg.json
        echo "version": "0.1.0",>> %~dp0/third_party/z5-src/vcpkg.json
        echo "dependencies": [ "boost-core", "boost-filesystem", "nlohmann-json", "blosc" ]>> %~dp0/third_party/z5-src/vcpkg.json
        echo }>> %~dp0/third_party/z5-src/vcpkg.json
        if not exist .\z5-src\build\ mkdir .\z5-src\build\
        pushd "z5-src\build"
        cmake %cmake_generator% -DCMAKE_TOOLCHAIN_FILE="%~dp0/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake" ^
        -Dxtl_DIR="%~dp0/third_party/xtl/share/cmake/xtl" ^
        -Dxtensor_DIR="%~dp0/third_party/xtensor/share/cmake/xtensor" ^
        -Dxsimd_DIR="%~dp0/third_party/xsimd/lib/cmake/xsimd" ^
        -DBUILD_Z5PY=OFF -DWITH_ZLIB=ON -DWITH_LZ4=ON -DWITH_BLOSC=ON ^
        -DCMAKE_INSTALL_PREFIX="%~dp0/third_party/z5" ..
        cmake --build . --config Release --target install || exit /b 1
        popd
    )
)
set cmake_args=%cmake_args% -Dxtl_DIR="third_party/xtl/share/cmake/xtl" ^
-Dxtensor_DIR="third_party/xtensor/share/cmake/xtensor" ^
-Dxsimd_DIR="third_party/xsimd/lib/cmake/xsimd" ^
-Dz5_DIR="third_party/z5/lib/cmake/z5"

if %build_with_cuda_support% == true (
    if not exist ".\tiny-cuda-nn" (
        echo ------------------------
        echo downloading tiny-cuda-nn
        echo ------------------------
        git clone https://github.com/chrismile/tiny-cuda-nn.git tiny-cuda-nn --recurse-submodules
        pushd "tiny-cuda-nn"
        git checkout activations
        popd
    )
    if not exist ".\quick-mlp" (
        echo ------------------------
        echo   downloading QuickMLP
        echo ------------------------
        git clone https://github.com/chrismile/quick-mlp.git quick-mlp --recurse-submodules
    )
)

if not exist ".\limbo" (
    echo ------------------------
    echo     downloading limbo
    echo ------------------------
    git clone --recursive https://github.com/resibots/limbo.git limbo
)

popd

if %debug% == true (
   echo ------------------------
   echo   building in debug
   echo ------------------------

   set cmake_config="Debug"
) else (
   echo ------------------------
   echo   building in release
   echo ------------------------

   set cmake_config="Release"
)

echo ------------------------
echo       generating
echo ------------------------
cmake %cmake_generator% %cmake_args% -S . -B %build_dir%

echo ------------------------
echo       compiling
echo ------------------------
cmake --build %build_dir% --config %cmake_config% -- /m || exit /b 1

echo ------------------------
echo    copying new files
echo ------------------------
robocopy .build\vcpkg_installed\x64-windows\tools\python3 ^
         %destination_dir%\python3 /e >NUL

if %debug% == true (
   if not exist %destination_dir%\*.pdb (
      del %destination_dir%\*.dll
   )
   robocopy %build_dir%\Debug\  %destination_dir%  >NUL
   robocopy third_party\sgl\.build\Debug %destination_dir% *.dll *.pdb >NUL
   robocopy third_party\ospray-%ospray_version%.x86_64.windows\bin %destination_dir% *.dll >NUL
) else (
   if exist %destination_dir%\*.pdb (
      del %destination_dir%\*.dll
      del %destination_dir%\*.pdb
   )
   robocopy %build_dir%\Release\ %destination_dir% >NUL
   robocopy third_party\sgl\.build\Release %destination_dir% *.dll >NUL
   robocopy third_party\ospray-%ospray_version%.x86_64.windows\bin %destination_dir% *.dll >NUL
)

echo.
echo All done!

pushd %destination_dir%
Correrender.exe
