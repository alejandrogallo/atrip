{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell rec {

  imports = [ ../shell.nix ];

  buildInputs = with pkgs; [
    emacs
    emacsPackages.ox-rst
    emacsPackages.htmlize
    python3
    python3Packages.breathe
    doxygen
    sphinx
    git
    graphviz
  ];

}
