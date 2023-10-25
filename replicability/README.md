# Correrender

This file contains information for the TVCG Replicability Stamp. For more details on the Replicability Stamp,
please refer to http://www.replicabilitystamp.org/.

[![](https://www.replicabilitystamp.org/logo/Reproducibility-small.png)](http://www.replicabilitystamp.org#https-github-com-chrismile-correrender)

We give permission to the Replicability Stamp committee and reviewers of the Graphics Replicability Stamp Initiative
(GRSI) to review the code and advertise the review publicly after the stamp is approved.


## Submission

Adaptive Sampling of 3D Spatial Correlations for Focus+Context Visualization. \
Christoph Neuhauser, Josef Stumpfegger, Rüdiger Westermann. \
To appear in IEEE Transactions on Visualization and Computer Graphics 2023 (TVCG 2023). \
DOI: 10.1109/TVCG.2023.3326855


## Build Process

While the build process is supported both on Linux and Windows, we currently only provide a script for Linux to not
only build the application, but also reproduce a figure from the paper.

Please execute the script `./build-linux.sh --replicability` in the repository folder on the command line.
It will compile and install all requirements, download the synthetic test data set, build and execute the program
reproducing Figure 9 from the paper.
This process may take some time depending on your internet connection speed. Please note that superuser rights need to
be given to parts of the script, as it will install some dependencies using the apt (Ubuntu) or Pacman (Arch Linux)
package managers.


## Supported Operating Systems & Hardware

The build process was tested on...
- Ubuntu 20.04 and Ubuntu 22.04 with GCC
- Arch Linux (note: older versions of GCC may be required if building with CUDA support)
- Windows 10 & Windows 11 using MSVC 2022 and MSYS2

The program has been tested on the following hardware configurations:
- Ubuntu 22.04 with an NVIDIA RTX 3090
- Windows 10 with an AMD Radeon RX 6900 XT


## Troubleshooting

If an error occurs during the build process or when running the program, please open an [issue report](https://github.com/chrismile/Correrender/issues)
or contact us by e-mail. After the bug was fixed, please pull the new code and delete the directory `third_party`
before again calling `./build-linux.sh --replicability`.
