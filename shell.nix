{ compiler, pkgs ? import <nixpkgs> {} , with-mkl ? false }:

let

  mkl = import ./etc/nix/mkl.nix { pkgs = (import <nixpkgs> {
    config.allowUnfree = true;
  }); };

  openblas = import ./etc/nix/openblas.nix { inherit pkgs; }; 

  clang = import ./etc/nix/clang.nix { inherit pkgs; };

in

pkgs.mkShell rec {

  compiler-pkg
    = if compiler    == "gcc11" then pkgs.gcc11
    else if compiler == "gcc10" then pkgs.gcc10
    else if compiler == "gcc9" then pkgs.gcc9
    else if compiler == "gcc8" then pkgs.gcc8
    else if compiler == "gcc7" then pkgs.gcc7
    else if compiler == "gcc6" then pkgs.gcc6
    else if compiler == "gcc49" then pkgs.gcc49
    else if compiler == "clang13" then pkgs.clang_13
    else if compiler == "clang12" then pkgs.clang_12
    else if compiler == "clang11" then pkgs.clang_11
    else if compiler == "clang10" then pkgs.clang_10
    else if compiler == "clang9" then pkgs.clang_9
    else if compiler == "clang8" then pkgs.clang_8
    else if compiler == "clang7" then pkgs.clang_7
    else if compiler == "clang6" then pkgs.clang_6
    else if compiler == "clang5" then pkgs.clang_5
    else pkgs.gcc;

  buildInputs
    = with pkgs; [

        coreutils
        git

        openmpi
        llvmPackages.openmp

        binutils
        emacs
        gfortran

        gnumake
        libtool
        autoconf
        automake
        pkg-config
      ]
    ++ (if with-mkl then mkl.buildInputs else openblas.buildInputs)
    ;

  CXX = "${compiler-pkg}/bin/c++";
  CC = "${compiler-pkg}/bin/cc";
  LD = "${compiler-pkg}/bin/ld";

  shellHook
    = #(if with-mkl then mkl.shellHook else openblas.shellHook)
    ''
    export OMPI_CXX=${CXX}
    export OMPI_CC=${CC}
    CXX=${CXX}
    CC=${CC}
    LD=${LD}
    ''
    ;

}
