# Correrender

Correrender is a correlation field volume renderer using the graphics API Vulkan.


## Building and running the programm

### Linux

There are two ways to build the program on Linux systems.
- Using the system package manager to install the dependencies (tested: apt on Ubuntu, pacman on Arch Linux).
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install the dependencies.

In the project root directory, two scripts `build-linux.sh` and `build-linux-vcpkg.sh` can be found. The former uses the
system package manager to install all dependencies, while the latter uses vcpkg. The build scripts will also launch the
program after successfully building it. If you wish to build the program manually, instructions can be found in the
directory `docs/compilation`.

Below, more information concerning different Linux distributions tested can be found.

#### Arch Linux

Arch Linux and its derivative Manjaro are fully supported using both build modes (package manager and vcpkg).

The Vulkan SDK, which is a dependency of this program that cannot be installed using vcpkg, will be automatically
installed using the package manager `pacman` when using the scripts.

#### Ubuntu 18.04, 20.04 & 22.04

Ubuntu 20.04 and 22.04 are fully supported.

The Vulkan SDK, which is a dependency of this program that cannot be installed using the default package sources or
vcpkg, will be automatically installed using the official Vulkan SDK PPA.

Please note that Ubuntu 18.04 is only partially supported. It ships an old version of CMake, which causes the build
process using vcpkg to fail if not updating CMake manually beforehand.

#### Other Linux Distributions

If you are using a different Linux distribution and face difficulties when building the program, please feel free to
open a [bug report](https://github.com/chrismile/Correrender/issues). In theory, the build scripts should also work on other
Linux distributions as long as the Vulkan SDK is installed manually beforehand.


### Windows

There are two ways to build the program on Windows.
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install the dependencies. The program can then be compiled using
  [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/).
- Using [MSYS2](https://www.msys2.org/) to install the dependencies and compile the program using MinGW.

In the project folder, a script called `build-windows.bat` can be found automating this build process using vcpkg and
Visual Studio. It is recommended to run the script using the `Developer PowerShell for VS 2022` (or VS 2019 depending on
your Visual Studio version). The build script will also launch the program after successfully building it.
Building the program is regularly tested on Windows 10 and 11 with Microsoft Visual Studio 2019 and 2022.

Please note that the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) needs to be installed beforehand if using
Microsoft Visual Studio for compilation.

A script `build-windows-msys2.bat` is also available to build the program using MSYS2/MinGW alternatively to using
Microsoft Visual Studio.

If you wish to build the program manually using Visual Studio and vcpkg, or using MSYS2, instructions can be found in
the directory `docs/compilation`.


### macOS

There are two ways to build the program on macOS.
- Using [Homebrew](https://brew.sh/) to install the dependencies and compile the program using LLVM/Clang (recommended).
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install the dependencies and compile the program using
  LLVM/Clang.

In the project root directory, two scripts `build-macos-vcpkg.sh` and `build-macos-brew.sh` can be found.
As macOS does not natively support Vulkan, MoltenVK, a Vulkan wrapper based on Apple's Metal API, is utilized.
Installing it via the scripts requires admin rights. MoltenVK can also be installed manually from
[the website](https://vulkan.lunarg.com/sdk/home#mac).

The program can only run with reduced feature set, as the Metal API does currently neither support geometry shaders nor
hardware-accelerated ray tracing.

Notes:
- I rented Apple hardware for a few days once for testing that running the program works on macOS.
  As I do not regularly have access to a real system running macOS, it is only tested that the program can compile in a
  CI pipeline build script on an x86_64 macOS virtual machine provided by GitHub Actions. So please note that it is not
  guaranteed that the program will continue working correctly on macOS indefinitely due to the lack of regular testing.
- To enable high DPI support, the program needs to be run from an app bundle. This happens automatically when the script
  `build-macos-brew.sh` has finished building the program. Please note that the app bundle only contains the Info.plist
  file necessary for high DPI support and is currently not yet redistributable. If you want to help with improving the
  macOS app bundle support for this project by contributing development time, please feel free to contact me.


## How to add new data sets

Under `Data/VolumeDataSets/datasets.json`, loadable volume data sets can be specified. Additionally, the user can also
open arbitrary data sets using a file explorer via "File > Open Dataset..." (or using Ctrl+O).

Below, an example for a `Data/VolumeDataSets/datasets.json` file can be found.

```json
{
    "datasets": [
        { "name" : "Karman Vortex Street", "filename": "data/vortex_street.nc" },
        { "type" : "Rotated Data", "filename": "data/other_data.dat", "transform": "rotate(270°, 1, 0, 0)" },
        { "type" : "Time Dependent Data", "filename": "data/data_timestep_%04i.dat", "time_indices": "0 2500 10" }
    ]
}
```

These files then appear with their specified name in the menu "File > Datasets". All paths must be specified relative to
the folder `Data/VolumeDataSets/` (unless they are global, like `C:/path/file.dat` or `/path/file.dat`).

Supported formats currently are:
- .nc (NetCDF format, https://www.unidata.ucar.edu/software/netcdf/).
- .zarr (Zarr format, https://zarr.readthedocs.io/en/stable/).
- .grb/.grib (GRIB format, https://weather.gc.ca/grib/what_is_GRIB_e.html).
- .vtk (structured grids in VTK format).
- .vti, .vts (structured grids in VTK XML format).
- .am (AmiraMesh format, see https://www.csc.kth.se/~weinkauf/notes/amiramesh.html).
- The custom .field, .bin, .dat and .raw file formats used in our research group.


## How to report bugs

When [reporting a bug](https://github.com/chrismile/Correrender/issues), please also attach the logfile generated by
Correrender. Below, the location of the logfile on different operating systems can be found.

- Linux: `~/.config/correrender/Logfile.html`
- Windows: `%AppData%/Correrender/Logfile.html` (i.e., `C:/Users/<USER>/AppData/Roaming/Correrender/Logfile.html`)
- macOS: `~/Library/Preferences/Correrender/Logfile.html`
