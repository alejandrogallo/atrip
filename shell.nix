{ pkgs ? import <nixpkgs> {} , with-clang ? false , with-mkl ? false }:

let

  mkl = import ./etc/nix/mkl.nix { pkgs = (import <nixpkgs> {
    config.allowUnfree = true;
  }); };

  openblas = import ./etc/nix/openblas.nix { inherit pkgs; }; 

  clang = import ./etc/nix/clang.nix { inherit pkgs; };

  compiler-configuration
    = if with-clang
      then clang.buildInputs
      else [ pkgs.gcc ];

in

pkgs.mkShell {

  buildInputs
    = with pkgs; [

        coreutils
        git

        openmpi

        gnumake
        binutils
        emacs
      ]
    ++ compiler-configuration
    ++ (if with-mkl then mkl.buildInputs else openblas.buildInputs)
    ;

  shellHook
    = (if with-clang then clang.shellHook else "")
    + (if with-mkl then mkl.shellHook else openblas.shellHook)
    ;

}
