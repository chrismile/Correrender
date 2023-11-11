# Correrender

Correrender is a correlation field volume renderer using the graphics API Vulkan.


## Building and running the programm

### Linux

There are two ways to build the program on Linux systems.
- Using the system package manager to install the dependencies (tested: apt on Ubuntu, pacman on Arch Linux).
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install the dependencies.

The script `build.sh` in the project root directory can be used to build the project. If no arguments are passed, the
dependencies are installed using the system package manager. When calling the script as `./build.sh --vcpkg`, vcpkg is
used instead. The build scripts will also launch the program after successfully building it. If you wish to build the
program manually, instructions can be found in the directory `docs/compilation`.

Below, more information concerning different Linux distributions tested can be found.

#### Arch Linux

Arch Linux and its derivative Manjaro are fully supported using both build modes (package manager and vcpkg).

The Vulkan SDK, which is a dependency of this program that cannot be installed using vcpkg, will be automatically
installed using the package manager `pacman` when using the scripts.

#### Ubuntu 18.04, 20.04 & 22.04

Ubuntu 22.04 is fully supported.

The Vulkan SDK, which is a dependency of this program that cannot be installed using the default package sources or
vcpkg, will be automatically installed using the official Vulkan SDK PPA.

Please note that Ubuntu 18.04 and 20.04 ship a too old version of CMake, which causes the build process to fail.
In this case, CMake needs to be upgraded manually beforehand using the steps at https://apt.kitware.com/.

#### Other Linux Distributions

If you are using a different Linux distribution and face difficulties when building the program, please feel free to
open a [bug report](https://github.com/chrismile/Correrender/issues). In theory, the build scripts should also work on
other Linux distributions as long as the Vulkan SDK is installed manually beforehand.

#### CUDA Support

The program can use CUDA to enable optional features. If the build scripts are not able to find your CUDA installation
on Linux, add the following lines to the end of `$HOME/.profile` and log out of and then back into your user account.
`cuda-12.1` needs to be adapted depending on the CUDA version installed. On distributions other than Ubuntu and Debian,
like Arch Linux and Majaro, the library path may be different from the one below, e.g., `/usr/local/cuda-12.1/lib64`.

```sh
export CPATH=/usr/local/cuda-12.1/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH
```


### Windows

There are two ways to build the program on Windows.
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install the dependencies. The program can then be compiled using
  [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/).
- Using [MSYS2](https://www.msys2.org/) to install the dependencies and compile the program using MinGW. In this case,
  all CUDA interoperability features are disabled. Currently, the CUDA compiler nvcc only supports MSVC on Windows.

In the project folder, a script called `build-msvc.bat` can be found automating this build process using vcpkg and
Visual Studio. It is recommended to run the script using the `Developer PowerShell for VS 2022` (or VS 2019 depending on
your Visual Studio version). The build script will also launch the program after successfully building it.
Building the program is regularly tested on Windows 10 and 11 with Microsoft Visual Studio 2019 and 2022.

Please note that the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) needs to be installed beforehand if using
Microsoft Visual Studio for compilation.

The script `build.sh` in the project root directory can also be used to alternatively build the program using
MSYS2/MinGW on Windows. For this, it should be run from a MSYS2 shell.

If you wish to build the program manually using Visual Studio and vcpkg, or using MSYS2, instructions can be found in
the directory `docs/compilation`.


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


## Replicability Stamp

This repository takes part in the Graphics Replicability Stamp Initiative (GRSI).
For more information, please refer to `replicability/README.md`.
For more details on the Replicability Stamp itself, please refer to http://www.replicabilitystamp.org/.

[![](https://www.replicabilitystamp.org/logo/Reproducibility-tiny.png)](http://www.replicabilitystamp.org#https-github-com-chrismile-correrender)


## How to report bugs

When [reporting a bug](https://github.com/chrismile/Correrender/issues), please also attach the logfile generated by
Correrender. Below, the location of the logfile on different operating systems can be found.

- Linux: `~/.config/correrender/Logfile.html`
- Windows: `%AppData%/Correrender/Logfile.html` (i.e., `C:/Users/<USER>/AppData/Roaming/Correrender/Logfile.html`)
- macOS: `~/Library/Preferences/Correrender/Logfile.html`
