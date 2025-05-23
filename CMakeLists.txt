cmake_minimum_required(VERSION 3.10...4.0)

project(Correrender)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
#MESSAGE(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

# not possible as sgl is built without...
#IF (UNIX AND CMAKE_BUILD_TYPE MATCHES Debug)
#    message("-- Enabling gcc stl debug checks for debug builds")
#    #add_compile_options(-D_GLIBCXX_DEBUG)
#ENDIF()

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.c src/*.hpp src/*.h Data/Shaders/*.glsl)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
include_directories(src)

file(GLOB_RECURSE SHADER_FILES Data/Shaders/*.glsl)
file(GLOB_RECURSE CUDA_MODULES Data/Shaders/*.cu)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders" PREFIX "Shaders" FILES ${SHADER_FILES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders" PREFIX "CUDA Modules" FILES ${CUDA_MODULES})
list(REMOVE_ITEM CUDA_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders/Correlation/MutualInformationKraskov.cu)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    # Avoid warning 'IMPORTED_IMPLIB not set for imported target "sgl" configuration "MinSizeRel/RelWithDebInfo".'
    set(CMAKE_CONFIGURATION_TYPES Debug Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3 /EHsc /Zc:__cplusplus")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

# Prefer GLVND by default when available.
if (${CMAKE_VERSION} VERSION_GREATER "3.11")
    cmake_policy(SET CMP0072 NEW)
endif()

if(VCPKG_TOOLCHAIN)
    if(EXISTS "${Z_VCPKG_ROOT_DIR}/triplets/${VCPKG_TARGET_TRIPLET}.cmake")
        include("${Z_VCPKG_ROOT_DIR}/triplets/${VCPKG_TARGET_TRIPLET}.cmake")
    elseif(EXISTS "${Z_VCPKG_ROOT_DIR}/triplets/community/${VCPKG_TARGET_TRIPLET}.cmake")
        include("${Z_VCPKG_ROOT_DIR}/triplets/community/${VCPKG_TARGET_TRIPLET}.cmake")
    else()
        message(FATAL_ERROR "Could not find a vcpkg triplet file for ${VCPKG_TARGET_TRIPLET}.")
    endif()
endif()

if (MSVC)
    set(DEFAULT_USE_TBB ON)
else()
    set(DEFAULT_USE_TBB OFF)
endif()

option(USE_STATIC_STD_LIBRARIES "Link with standard libraries statically (only supported on Linux for now)." OFF)
option(USE_GLIBCXX_DEBUG "Use the -D_GLIBCXX_DEBUG flag when compiling with GCC." OFF)
option(USE_TBB "Build with TBB threading support instead of using OpenMP." ${DEFAULT_USE_TBB})
option(USE_ZINK "Use the Zink OpenGL wrapper if available." OFF)
option(SUPPORT_PYTORCH "Build the PyTorch correlation calculator module." ON)
option(SUPPORT_TINY_CUDA_NN "Build the tiny-cuda-nn correlation calculator module." ON)
option(SUPPORT_QUICK_MLP "Build the QuickMLP correlation calculator module." ON)
option(SUPPORT_RENDERDOC_DEBUGGER "Build with the RenderDoc debugger." OFF)

set(DATA_PATH "${CMAKE_SOURCE_DIR}/Data" CACHE PATH "Location of folder 'Data'")
add_definitions(-DDATA_PATH=\"${DATA_PATH}\")

if ((UNIX OR MSYS OR MINGW) AND (NOT APPLE OR NOT VCPKG_TOOLCHAIN))
    find_package(PkgConfig REQUIRED)
endif()

# CUDA nvcc does not support MSYS/MinGW.
if (MSYS OR MINGW)
    set(SUPPORT_PYTORCH OFF)
    set(SUPPORT_TINY_CUDA_NN OFF)
    set(SUPPORT_QUICK_MLP OFF)
endif()


if((MSYS OR MINGW OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")) AND ${USE_GLIBCXX_DEBUG})
    add_compile_definitions(_GLIBCXX_DEBUG)
endif()


set(FPHSA_NAME_MISMATCHED TRUE)

find_package(sgl QUIET)
if(NOT sgl_FOUND)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};third_party/sgl/install/lib/cmake/sgl)
    find_package(sgl REQUIRED)
endif()
# Boost still necessary for boost::math::digamma and filesystem for dependencies.
find_package(Boost COMPONENTS system filesystem REQUIRED)
if(VCPKG_TOOLCHAIN)
    find_package(netCDF CONFIG REQUIRED)
    find_package(tinyxml2 CONFIG REQUIRED)
    find_package(jsoncpp CONFIG REQUIRED)
else()
    find_package(NetCDF REQUIRED)
    find_package(TinyXML2 REQUIRED)
    find_package(jsoncpp QUIET)
    if (NOT jsoncpp_FOUND)
        pkg_check_modules(JSONCPP jsoncpp)
    endif()
endif()

find_package(HDF5 QUIET)
if(NOT HDF5_FOUND)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Loaders/Hdf5Loader.cpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Loaders/Hdf5Loader.hpp)
    MESSAGE(STATUS "HDF5 couldn't be found. Disabling HDF5 file loading support.")
endif()

find_package(eccodes QUIET)
if(NOT eccodes_FOUND)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Loaders/GribLoader.cpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Loaders/GribLoader.hpp)
    MESSAGE(STATUS "ecCodes couldn't be found. Disabling GRIB file loading support.")
endif()

find_package(z5 QUIET)
if(NOT z5_FOUND)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Loaders/ZarrLoader.cpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Loaders/ZarrLoader.hpp)
    MESSAGE(STATUS "z5 couldn't be found. Disabling zarr file loading support.")
endif()

if (NOT ${SUPPORT_RENDERDOC_DEBUGGER})
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/RenderDocDebugger.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils/RenderDocDebugger.cpp)
endif()

get_target_property(SGL_INTERFACE_COMPILE_DEFINITIONS sgl INTERFACE_COMPILE_DEFINITIONS)
if (NOT "SUPPORT_VULKAN" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    message(FATAL_ERROR "sgl was not built with Vulkan support enabled.")
endif()

if (${SUPPORT_PYTORCH})
    find_package(Torch QUIET)
endif()
if (${SUPPORT_PYTORCH} AND NOT Torch_FOUND)
    MESSAGE(STATUS "PyTorch support was enabled, but PyTorch couldn't be found. Disabling PyTorch support.")
endif()
if (NOT ${SUPPORT_PYTORCH} OR NOT Torch_FOUND)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/PyTorchCorrelationCalculator.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/PyTorchCorrelationCalculator.cpp)
endif()

if ("SUPPORT_CUDA_INTEROP" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
        # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
        find_package(CUDA QUIET)
        if(${CUDA_FOUND})
            set(CUDA_ENABLED true)
        endif()
    else()
        # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
        find_package(CUDAToolkit QUIET)
        if(${CUDAToolkit_FOUND})
            set(CUDA_ENABLED true)
        endif()
    endif()
    if (NOT CUDA_FOUND AND NOT CUDAToolkit_FOUND)
        MESSAGE(STATUS "CUDA could not be found. Disabling tiny-cuda-nn and QuickMLP support.")
    endif()
else()
    MESSAGE(STATUS "sgl was not built with CUDA interoperability support or using unsupported host compiler. Disabling tiny-cuda-nn and QuickMLP support.")
endif()

set(CUDA_HOST_COMPILER_COMPATIBLE true)
if (CUDA_ENABLED)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        MESSAGE(STATUS "CUDA was found and is compatible with the host compiler.")
    else()
        set(CUDA_ENABLED false)
        set(CUDA_HOST_COMPILER_COMPATIBLE false)
        MESSAGE(WARNING "CUDA was found, but the compiler cannot be enabled. Please check if CUDA is compatible with the used host compiler.")
    endif()
endif()

if (CUDA_ENABLED AND ${SUPPORT_TINY_CUDA_NN})
    add_subdirectory(third_party/tiny-cuda-nn)
else()
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/TinyCudaNNCorrelationCalculator.hpp)
    list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/TinyCudaNNCorrelationCalculator.cu)
    list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Renderers/Diagram/TimeSeriesCorrelation/TimeSeriesCorrelationRenderer.cu)
endif()

if (CUDA_ENABLED AND ${SUPPORT_QUICK_MLP})
    add_subdirectory(third_party/quick-mlp)
else()
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/QuickMLPCorrelationCalculator.hpp)
    list(REMOVE_ITEM CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/QuickMLPCorrelationCalculator.cu)
endif()

if (NOT CUDA_ENABLED OR (NOT ${SUPPORT_TINY_CUDA_NN} AND NOT ${SUPPORT_QUICK_MLP}))
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/DeepLearningCudaCorrelationCalculator.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Calculators/DeepLearningCudaCorrelationCalculator.cpp)
endif()


if(VCPKG_TOOLCHAIN)
    find_package(skia CONFIG)
else()
    find_package(Skia)
endif()
if (NOT Skia_FOUND AND NOT skia_FOUND)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Renderers/Diagram/VectorBackendSkia.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Renderers/Diagram/VectorBackendSkia.cpp)
endif()

find_package(vkvg)
if (NOT vkvg_FOUND)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Renderers/Diagram/VectorBackendVkvg.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Renderers/Diagram/VectorBackendVkvg.cpp)
endif()

find_package(osqp)


if(WIN32)
    add_executable(Correrender WIN32 ${SOURCES} ${CUDA_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/CMake/winres.rc)
elseif(APPLE)
    add_executable(Correrender MACOSX_BUNDLE ${SOURCES} ${CUDA_SOURCES})
else()
    add_executable(Correrender ${SOURCES} ${CUDA_SOURCES})
endif()

if((MSYS OR MINGW OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")) AND ${USE_STATIC_STD_LIBRARIES})
    target_link_options(Correrender PRIVATE -static-libgcc -static-libstdc++)
endif()

if (VCPKG_TOOLCHAIN AND UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND ${CMAKE_VERSION} VERSION_GREATER "3.12"
        AND "SGL_STATIC_LIBRARY" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    #target_link_options(Correrender PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/CMake/symbols_linux_gcc.map")
    target_link_options(Correrender PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/CMake/symbols_linux_gcc.map")
endif()

if (APPLE)
    set(BUNDLE_PATH ${CMAKE_CURRENT_BINARY_DIR}/Correrender.app)
    set_target_properties(
            Correrender PROPERTIES
            OUTPUT_NAME "Correrender"
            MACOSX_BUNDLE_EXECUTABLE_NAME "Correrender"
            MACOSX_BUNDLE_BUNDLE_NAME "Correrender"
            FOLDER "Correrender"
            MACOSX_BUNDLE_SHORT_VERSION_STRING ""
            MACOSX_BUNDLE_LONG_VERSION_STRING ""
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/CMake/MacOSXBundleInfo.plist.in")
endif()


# Hide console window in release build on Windows/MSVC.
# See: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/recipe/VSConfigSpecificSettings
if(MSVC)
    set_target_properties(Correrender PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE")
    set_target_properties(Correrender PROPERTIES COMPILE_DEFINITIONS_DEBUG "_CONSOLE")
    set_target_properties(Correrender PROPERTIES LINK_FLAGS_RELWITHDEBINFO "/SUBSYSTEM:CONSOLE")
    set_target_properties(Correrender PROPERTIES COMPILE_DEFINITIONS_RELWITHDEBINFO "_CONSOLE")
    set_target_properties(Correrender PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS")
    set_target_properties(Correrender PROPERTIES LINK_FLAGS_MINSIZEREL "/SUBSYSTEM:WINDOWS")
endif()

if(MSVC)
    #target_compile_definitions(Correrender PRIVATE /wd4251)
    if (NOT CUDA_ENABLED)
        set_target_properties(Correrender PROPERTIES COMPILE_FLAGS /wd4251)
    endif()
    target_compile_options(Correrender PRIVATE
        $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/MP>)
endif()

if(MSVC)
    # Setting /bigobj globally will lead to problems with nvcc. Two of the errors I noticed:
    # nvcc fatal   : Don't know what to do with 'C:/bigobj'
    # nvcc fatal   : A single input file is required for a non-link phase when an outputfile is specified
    #add_compile_options(/bigobj)

    # Add /bigobj to these files, as otherwise cl.exe will complain during the debug build.
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/src/Renderers/Diagram/HEBChartCorrelation.cpp PROPERTIES COMPILE_FLAGS /bigobj)
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/src/Optimization/OLS/EigenSolver.cpp PROPERTIES COMPILE_FLAGS /bigobj)
endif()
if(MSYS OR MINGW)
    SET_SOURCE_FILES_PROPERTIES(${CMAKE_CURRENT_SOURCE_DIR}/src/Optimization/OLS/EigenSolver.cpp PROPERTIES COMPILE_FLAGS -Wa,-mbig-obj)
endif()

add_subdirectory(submodules/IsosurfaceCpp)
if("TRACY_ENABLE" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    target_compile_definitions(IsosurfaceCpp PUBLIC TRACY_ENABLE)
    target_compile_definitions(IsosurfaceCpp PUBLIC TRACY_IMPORTS)
    target_include_directories(IsosurfaceCpp PUBLIC $<TARGET_PROPERTY:sgl,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
target_link_libraries(Correrender PRIVATE IsosurfaceCpp)
target_include_directories(Correrender PRIVATE submodules)

if (NOT MSVC)
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(Correrender PRIVATE NDEBUG)
    endif()
endif()

if(MSYS OR MINGW OR (${CMAKE_GENERATOR} STREQUAL "MinGW Makefiles") OR (${CMAKE_GENERATOR} STREQUAL "MSYS Makefiles"))
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mconsole")
    else()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mwindows")
    endif()
    target_link_libraries(Correrender PUBLIC mingw32)
endif()
if ("SUPPORT_SDL2" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    if (VCPKG_TOOLCHAIN)
        target_link_libraries(Correrender PUBLIC SDL2::SDL2main)
    else()
        target_link_libraries(Correrender PUBLIC SDL2::Main)
    endif()
endif()
if ("SUPPORT_SDL3" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    if(TARGET SDL3::SDL3)
        target_link_libraries(Correrender PRIVATE SDL3::SDL3)
    elseif(TARGET SDL3::SDL3-static)
        target_link_libraries(Correrender PRIVATE SDL3::SDL3-static)
    else()
        message(FATAL_ERROR "Unknown SDL3 target.")
    endif()
endif()

target_link_libraries(Correrender PRIVATE sgl ${Boost_LIBRARIES})
target_include_directories(Correrender PRIVATE ${sgl_INCLUDES} ${Boost_INCLUDE_DIR})

if(VCPKG_TOOLCHAIN)
    target_link_libraries(Correrender PRIVATE tinyxml2::tinyxml2 netCDF::netcdf)
else()
    target_link_libraries(Correrender PRIVATE ${TINYXML2_LIBRARIES} ${NETCDF_LIBRARIES})
    target_include_directories(Correrender PRIVATE ${NETCDF_INCLUDE_DIR})
endif()
target_include_directories(Correrender PRIVATE ${sgl_INCLUDES} ${TINYXML2_INCLUDE_DIR})

if (${SUPPORT_RENDERDOC_DEBUGGER})
    target_compile_definitions(Correrender PRIVATE SUPPORT_RENDERDOC_DEBUGGER)
endif()

if(CUDA_ENABLED)
    enable_language(CUDA)
    set_target_properties(Correrender PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(Correrender PRIVATE CUDA_ENABLED)
endif()
if(CUDAToolkit_FOUND)
    # https://en.wikipedia.org/wiki/CUDA
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability
    if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
        set_property(TARGET Correrender PROPERTY CUDA_ARCHITECTURES 86 90)
    elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.2")
        set_property(TARGET Correrender PROPERTY CUDA_ARCHITECTURES 86)
    else()
        message(FATAL_ERROR "Found CUDA toolkit version ${CUDAToolkit_VERSION} is too old. CUDA 11.2 or greater is required.")
    endif()
    #if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
    #    set_property(TARGET Correrender PROPERTY CUDA_ARCHITECTURES 61 75 86 90)
    #elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.1")
    #    set_property(TARGET Correrender PROPERTY CUDA_ARCHITECTURES 61 75 86)
    #elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "10.0")
    #    set_property(TARGET Correrender PROPERTY CUDA_ARCHITECTURES 61 75)
    #else()
    #    set_property(TARGET Correrender PROPERTY CUDA_ARCHITECTURES 61)
    #endif()
    set_target_properties(Correrender PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    MESSAGE(STATUS "CUDA toolkit version: " ${CUDAToolkit_VERSION})
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(Correrender PRIVATE CUDA::cublas CUDA::cusolver)
elseif(${CUDA_FOUND})
    target_link_libraries(Correrender PRIVATE ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
endif()

if(HDF5_FOUND)
    MESSAGE(STATUS "Found HDF5 ${HDF5_VERSION}. Enabling HDF5 file loading support.")
    target_link_libraries(Correrender PRIVATE hdf5::hdf5)
    target_link_libraries(Correrender PRIVATE ${HDF5_C_LIBRARIES})
    target_include_directories(Correrender PRIVATE ${HDF5_C_INCLUDE_DIRS})
    target_compile_definitions(Correrender PRIVATE ${HDF5_C_DEFINITIONS})
    target_compile_definitions(Correrender PRIVATE USE_HDF5)
endif()

if (eccodes_FOUND)
    MESSAGE(STATUS "Found ecCodes ${eccodes_VERSION}. Enabling GRIB file loading support.")
    target_link_libraries(Correrender PRIVATE ${eccodes_LIBRARIES} eccodes)
    target_include_directories(Correrender PRIVATE ${eccodes_INCLUDE_DIR})
    target_compile_definitions(Correrender PRIVATE USE_ECCODES)
endif()

if (z5_FOUND)
    MESSAGE(STATUS "Found z5. Enabling zarr file loading support.")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    find_package(z5 REQUIRED)
    find_package(xtensor REQUIRED)
    find_package(xtl REQUIRED)
    target_link_libraries(Correrender PRIVATE z5 xtensor Threads::Threads)
    if(VCPKG_TOOLCHAIN)
        find_package(zstd CONFIG REQUIRED)
        target_link_libraries(z5 INTERFACE $<IF:$<TARGET_EXISTS:zstd::libzstd_shared>,zstd::libzstd_shared,zstd::libzstd_static>)
    endif()
    target_include_directories(Correrender PRIVATE ${z5_INCLUDE_DIRS})
    target_compile_definitions(Correrender PRIVATE USE_ZARR)
endif()

if (JSONCPP_LIBRARIES)
    target_link_libraries(Correrender PRIVATE ${JSONCPP_LIBRARIES})
    target_link_directories(Correrender PRIVATE ${JSONCPP_LIBRARY_DIRS})
    target_include_directories(Correrender PRIVATE ${JSONCPP_INCLUDE_DIRS})
endif()
if (TARGET jsoncpp_lib)
    get_target_property(jsoncpp_INCLUDES jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
    target_link_libraries(Correrender PRIVATE jsoncpp_lib)
elseif(TARGET jsoncpp_static)
    get_target_property(jsoncpp_INCLUDES jsoncpp_static INTERFACE_INCLUDE_DIRECTORIES)
    target_link_libraries(Correrender PRIVATE jsoncpp_static)
endif()
if (TARGET jsoncpp_object)
    target_link_libraries(Correrender PRIVATE jsoncpp_object)
endif()
target_include_directories(Correrender PRIVATE ${jsoncpp_INCLUDES})

if (${USE_ZINK})
    target_compile_definitions(Correrender PRIVATE USE_ZINK)
endif()

# sgl provides Vulkan headers in its interface, which are provided if Vulkan was not found during its build.
find_package(Vulkan QUIET)
if (${Vulkan_FOUND} AND NOT "USE_SGL_VULKAN_HEADERS" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS
        AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.21.0")
    target_link_libraries(Correrender PUBLIC Vulkan::Headers)
endif()

if (${SUPPORT_PYTORCH} AND Torch_FOUND)
    target_link_libraries(Correrender PRIVATE ${TORCH_LIBRARIES})
    if(CUDAToolkit_FOUND)
        target_link_libraries(Correrender PRIVATE CUDA::cupti)
    endif()
    target_include_directories(Correrender PRIVATE ${TORCH_INCLUDE_DIR})
    target_compile_definitions(Correrender PRIVATE SUPPORT_PYTORCH)
endif()

if (CUDA_ENABLED AND ${SUPPORT_TINY_CUDA_NN})
    target_link_libraries(Correrender PRIVATE qmlp::qmlp-library)
    target_compile_definitions(Correrender PRIVATE SUPPORT_TINY_CUDA_NN)
endif()

if (CUDA_ENABLED AND ${SUPPORT_QUICK_MLP})
    target_link_libraries(Correrender PRIVATE tiny-cuda-nn)
    target_compile_definitions(Correrender PRIVATE SUPPORT_QUICK_MLP)
endif()

if (Skia_FOUND OR skia_FOUND)
    if(VCPKG_TOOLCHAIN)
        target_link_libraries(Correrender PRIVATE skia::skia)
    else()
        target_link_libraries(Correrender PRIVATE Skia::Skia)
    endif()
    target_compile_definitions(Correrender PRIVATE SUPPORT_SKIA)
endif()

if (vkvg_FOUND)
    target_link_libraries(Correrender PRIVATE vkvg::vkvg)
    target_compile_definitions(Correrender PRIVATE SUPPORT_VKVG)
endif()

if (osqp_FOUND)
    target_link_libraries(Correrender PRIVATE osqp::osqpstatic) # osqp::osqp
    target_compile_definitions(Correrender PRIVATE SUPPORT_OSQP)

    # https://github.com/osqp/osqp/pull/577
    get_target_property(OSQP_INCLUDE_DIRECTORIES osqp::osqpstatic INTERFACE_INCLUDE_DIRECTORIES)
    if (EXISTS "${OSQP_INCLUDE_DIRECTORIES}/osqp_api_utils.h")
        target_compile_definitions(Correrender PRIVATE OSQP_LEGACY_API)
    endif()
endif()

if(VCPKG_TOOLCHAIN)
    find_package(CURL CONFIG REQUIRED)
else()
    find_package(CURL REQUIRED)
endif()
if (TARGET CURL::libcurl)
    target_link_libraries(Correrender PUBLIC CURL::libcurl)
else()
    target_link_libraries(Correrender PUBLIC ${CURL_LIBRARIES})
endif()
target_include_directories(Correrender PUBLIC ${CURL_INCLUDES} ${CURL_INCLUDE_DIRS})

#if(VCPKG_TOOLCHAIN)
#    find_package(TIFF CONFIG REQUIRED)
#else()
#    find_package(TIFF REQUIRED)
#endif()
find_package(TIFF REQUIRED)
target_link_libraries(Correrender PUBLIC ${TIFF_LIBRARIES})  # TIFF::TIFF, since CMake 3.5
target_include_directories(Correrender PUBLIC ${TIFF_INCLUDES} ${TIFF_INCLUDE_DIRS})

# For testing purposes.
target_compile_definitions(Correrender PRIVATE TEST_INFERENCE_SPEED)
target_compile_definitions(Correrender PRIVATE SHOW_DEBUG_OPTIONS)


if (${USE_TBB})
    find_package(TBB REQUIRED)
    target_link_libraries(Correrender PRIVATE TBB::tbb)
    target_compile_definitions(Correrender PRIVATE USE_TBB)
    if (USE_GTEST)
        target_link_libraries(LineVis_test PRIVATE TBB::tbb)
        target_compile_definitions(LineVis_test PRIVATE USE_TBB)
    endif()
else()
    # According to https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/,
    # support for LLVM OpenMP was added with Visual Studio 2019 version 16.9. According to
    # https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170,
    # both Visual Studio 2019 version 16.8 and 16.9 have _MSC_VER / MSVC_VERSION 1928, so we need to check
    # for a minimum version 1929.
    if (APPLE)
        # There are some problems with Homebrew OpenMP as of 2022-10-20.
        find_package(OpenMP QUIET)
        if (OpenMP_FOUND)
            MESSAGE(STATUS "Found OpenMP version ${OpenMP_CXX_VERSION}. Enabling OpenMP threading support.")
        else()
            MESSAGE(STATUS "OpenMP not found. Disabling OpenMP threading support.")
        endif()
    else()
        find_package(OpenMP REQUIRED)
    endif()
    if(OpenMP_FOUND)
        if(MSVC)
            option(USE_MSVC_OPENMP_LLVM "Use LLVM OpenMP with MSVC (available since Visual Studio 2019 version 16.9)." ON)
            option(USE_MSVC_OPENMP_EXPERIMENTAL "Use experimental OpenMP with MSVC (available since Visual Studio 2019 version 16.9)." ON)

            if(${MSVC_VERSION} GREATER_EQUAL 1929 AND (USE_MSVC_OPENMP_LLVM OR USE_MSVC_OPENMP_EXPERIMENTAL))
                define_property(
                        SOURCE
                        PROPERTY COMPILE_FLAGS
                        INHERITED
                        BRIEF_DOCS "brief-doc"
                        FULL_DOCS "full-doc"
                )
                if(USE_MSVC_OPENMP_LLVM)
                    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:llvm")
                    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
                    set_directory_properties(PROPERTIES COMPILE_FLAGS "/openmp:llvm")
                endif()
                if(USE_MSVC_OPENMP_EXPERIMENTAL)
                    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:experimental")
                    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
                    set_directory_properties(PROPERTIES COMPILE_FLAGS "/openmp:experimental")
                endif()
                if(USE_MSVC_OPENMP_LLVM AND USE_MSVC_OPENMP_EXPERIMENTAL)
                    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w44005")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44005")
                endif()
                set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/LineData/TrianglePayload/NodesBVHTreePayload.cpp" PROPERTIES COMPILE_FLAGS "")
                set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/CMake/winres.rc" PROPERTIES COMPILE_FLAGS "")
            endif()
        elseif(NOT TARGET OpenMP::OpenMP_CXX)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        else()
            target_link_libraries(Correrender PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()


function(compile_fatbin SRC OUTPUT)
    add_custom_command(
            OUTPUT "${OUTPUT}"
            COMMAND "${CMAKE_CUDA_COMPILER}" -fatbin -arch=all -Wno-deprecated-gpu-targets "${SRC}" -o "${OUTPUT}"
            MAIN_DEPENDENCY "${SRC}"
            VERBATIM
    )
endfunction()

if ("SUPPORT_CUDA_INTEROP" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS AND NOT MSYS AND NOT MINGW AND ${CUDA_HOST_COMPILER_COMPATIBLE})
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/Data/__cudacache__")
    MESSAGE(STATUS "Adding CUDA module compile commands.")
    foreach(CUDA_MODULE IN LISTS CUDA_MODULES)
        get_filename_component(CUDA_MODULE_NAME "${CUDA_MODULE}" NAME_WE)
        set(DEST "${CUDA_MODULE_NAME}.fatbin")
        set(OUTPUT "${CMAKE_SOURCE_DIR}/Data/__cudacache__/${DEST}")
        compile_fatbin("${CUDA_MODULE}" "${OUTPUT}")
        list(APPEND CUDA_FATBIN_FILES "${OUTPUT}")
    endforeach()
    add_custom_target(COMPILE_CUDA_MODULES ALL DEPENDS ${CUDA_FATBIN_FILES})
    add_dependencies(${CMAKE_PROJECT_NAME} COMPILE_CUDA_MODULES)
endif()

# Eigen3 includes.
if(VCPKG_TOOLCHAIN)
    find_package(Eigen3 CONFIG REQUIRED)
else()
    find_package(Eigen3 REQUIRED)
endif()
if (NOT (EIGEN3_VERSION_STRING LESS 3.3))
    target_link_libraries(Correrender PRIVATE Eigen3::Eigen)
else()
    target_include_directories(Correrender PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

# Include directories and link libraries for Limbo.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/limbo" AND Eigen3_FOUND)
    target_include_directories(Correrender PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/limbo/src")
    if (VCPKG_TOOLCHAIN)
        find_package(NLopt CONFIG REQUIRED)
        target_link_libraries(Correrender PRIVATE NLopt::nlopt)
    else()
        target_link_libraries(Correrender PRIVATE nlopt)
    endif()
    if (MSVC OR MSYS OR MINGW)
        target_link_libraries(Correrender PRIVATE ws2_32)
    endif()
    if (${USE_TBB})
        if (TBB_VERSION_MAJOR GREATER_EQUAL 2021)
            target_compile_definitions(Correrender PRIVATE USE_TBB_ONEAPI)
        endif()
    endif()
    target_compile_definitions(Correrender PRIVATE USE_LIMBO)
    MESSAGE(STATUS "Found Eigen ${EIGEN3_VERSION_STRING} and Limbo. Enabling Bayesian optimization support.")
else()
    MESSAGE(STATUS "Eigen 3 or Limbo couldn't be found. Switching off Bayesian optimization support.")
endif()
