{ pkgs, ...}:

{
  buildInputs = with pkgs; [
    clang
    llvmPackages.openmp
  ];

  shellHook = ''
    export OMPI_CC=clang
    export OMPI_CXX=clang++
  '';
}

