{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell rec {

  buildInputs = with pkgs; [

    coreutils
    git

    gcc
    blas
    openmpi

    emacs
  ];

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

  shellHook = ''
    export LAPACK_PATH=${pkgs.lapack}
    export BLAS_PATH=${pkgs.blas}
    export SCALAPACK_PATH=${scalapack}
    export LD_LIBRARY_PATH=${scalapack}/lib:$LD_LIBRARY_PATH
  '';

}
