rec {

  directory = "vendor";
  src = ''

    _add_vendor_cpath () {
      export CPATH=$CPATH:$1
      mkdir -p ${directory}/include
      ln -frs $1/* ${directory}/include/
    }

    _add_vendor_lib () {
      mkdir -p ${directory}/lib
      ln -frs $1/* ${directory}/lib/
    }

  '';

  cpath = path: ''
    _add_vendor_cpath ${path}
  '';

  lib = path: ''
    _add_vendor_lib ${path}
  '';

}
