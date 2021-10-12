{ pkgs ? import <nixpkgs> {} , with-clang ? false }:

let
  compiler-configuration
    = if with-clang
      then (import ./etc/nix/clang.nix { inherit pkgs; }).buildInputs
      else [ pkgs.gcc ];

in

pkgs.mkShell {

  buildInputs = with pkgs; [

    coreutils
    git

    blas
    openmpi

    gnumake
    binutils
    emacs

  ] ++ compiler-configuration;

  shellHook = ''
    export LAPACK_PATH=${pkgs.lapack}
    export BLAS_PATH=${pkgs.blas}
    export OPENBLAS_PATH=${pkgs.openblas}
    export SCALAPACK_PATH=${pkgs.scalapack}
    export LD_LIBRARY_PATH=${pkgs.scalapack}/lib:$LD_LIBRARY_PATH
  '';

}
