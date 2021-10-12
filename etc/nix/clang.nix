{ pkgs, ...}:

{
  buildInputs = with pkgs; [
    clang
    llvmPackages.openmp
  ];
}

