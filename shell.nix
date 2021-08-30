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

}
