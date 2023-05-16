{ compiler ? "gcc", pkgs ? import <nixpkgs> { }, mkl ? false, cuda ? false
, docs ? true }:

let

  unfree-pkgs = import <nixpkgs> { config.allowUnfree = true; };

  ctf = pkgs.callPackage ./etc/nix/ctf.nix { };

  openblas = import ./etc/nix/openblas.nix { inherit pkgs; };
  vendor = import ./etc/nix/vendor-shell.nix;

  mkl-pkg = import ./etc/nix/mkl.nix { pkgs = unfree-pkgs; };
  cuda-pkg = if cuda then (import ./cuda.nix { pkgs = unfree-pkgs; }) else { };

in pkgs.mkShell rec {

  compiler-pkg = if compiler == "gcc11" then
    pkgs.gcc11
  else if compiler == "gcc10" then
    pkgs.gcc10
  else if compiler == "gcc9" then
    pkgs.gcc9
  else if compiler == "gcc8" then
    pkgs.gcc8
  else if compiler == "gcc7" then
    pkgs.gcc7
  else if compiler == "gcc6" then
    pkgs.gcc6
  else if compiler == "gcc49" then
    pkgs.gcc49
  else if compiler == "clang13" then
    pkgs.clang_13
  else if compiler == "clang12" then
    pkgs.clang_12
  else if compiler == "clang11" then
    pkgs.clang_11
  else if compiler == "clang10" then
    pkgs.clang_10
  else if compiler == "clang9" then
    pkgs.clang_9
  else if compiler == "clang8" then
    pkgs.clang_8
  else if compiler == "clang7" then
    pkgs.clang_7
  else if compiler == "clang6" then
    pkgs.clang_6
  else if compiler == "clang5" then
    pkgs.clang_5
  else
    pkgs.gcc;

  docInputs = with pkgs; [
    emacs
    emacsPackages.ox-rst
    emacsPackages.htmlize

    python3
    python3Packages.breathe

    doxygen
    sphinx

    graphviz
  ];

  buildInputs = with pkgs;
    [

      gdb
      coreutils
      git
      vim
      clang-tools
      bear

      openmpi
      llvmPackages.openmp

      binutils
      gfortran

      ctf
      gnumake
      libtool
      autoconf
      automake
      pkg-config
    ] ++ (if mkl then mkl-pkg.buildInputs else openblas.buildInputs)
    ++ (if docs then docInputs else [ ]);

  NIX_CTF = ctf;
  CXX = "${compiler-pkg}/bin/c++";
  CC = "${compiler-pkg}/bin/cc";
  LD = "${compiler-pkg}/bin/ld";

  shellHook = ''

    ${vendor.src}

    ${vendor.cpath "${pkgs.openmpi.out}/include"}
    ${vendor.cpath "${openblas.pkg.dev}/include"}

    ${vendor.lib "${pkgs.openmpi.out}/lib"}
    ${vendor.lib "${openblas.pkg.out}/lib"}

    export OMPI_CXX=${CXX}
    export OMPI_CC=${CC}
    CXX=${CXX}
    CC=${CC}
    LD=${LD}
  '' + (if mkl then mkl-pkg.shellHook else openblas.shellHook)
    + (if cuda then cuda-pkg.shellHook else "");

}
