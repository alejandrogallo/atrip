{ pkgs, ...}:

{
  buildInputs = with pkgs; [
    openblas
    scalapack
  ];

  shellHook = ''
    export SCALAPACK_PATH=${pkgs.scalapack}
    export LD_LIBRARY_PATH=${pkgs.scalapack}/lib:$LD_LIBRARY_PATH
  '';
}


