{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell rec {

  buildInputs = with pkgs; [

    clang
    llvmPackages.openmp

    coreutils
    git

    gcc
    blas
    openmpi
    mpi

    emacs
  ];

  /*
  openblas =  pkgs.openblas.override {
    enableStatic = true;
  };

  scalapack = import ./etc/nix/scalapack.nix {
    lib = pkgs.lib;
    stdenv = pkgs.stdenv;
    fetchFromGitHub = pkgs.fetchFromGitHub;
    cmake = pkgs.cmake;
    openssh = pkgs.openssh;
    gfortran = pkgs.gfortran;
    mpi = pkgs.mpi;
    blas = pkgs.blas;
    lapack = pkgs.lapack;
  };
  */

  shellHook = ''
    export LAPACK_PATH=${pkgs.lapack}
    export BLAS_PATH=${pkgs.blas}
    export OPENBLAS_PATH=${pkgs.openblas}
    export SCALAPACK_PATH=${pkgs.scalapack}
    export LD_LIBRARY_PATH=${pkgs.scalapack}/lib:$LD_LIBRARY_PATH
  '';

}
