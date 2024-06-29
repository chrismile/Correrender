let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-23.11";
  pkgs = import nixpkgs { config = {}; overlays = []; };
in

pkgs.mkShell {
  packages = with pkgs; [
    cmake
    git
    curl
    pkg-config
    patchelf
    boost
    glm
    libarchive
    tinyxml-2
    libpng
    SDL2
    SDL2_image
    glew-egl
    vulkan-headers
    vulkan-loader
    vulkan-validation-layers
    shaderc
    opencl-headers
    ocl-icd
    jsoncpp
    nlohmann_json
    c-blosc
    netcdf
    hdf5
    eccodes
    eigen
    libtiff
    nlopt
  ];

  BUILD_USE_NIX = "ON";

  shellHook = ''
    echo "Run ./build.sh to build the application with Nix."
  '';
}
