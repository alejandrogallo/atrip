let

  versions = [
    {
      version = "3.0.0";
      sha256 = "sha256-9pm/8h2wEl2MzP55UYt3ZBzYNihyWh4e0+RWM0lqgtc=";
    }
    {
      version = "3.1.0";
      sha256 = "sha256-slwEQSTMhZwLTm6CVXT5Q5pRaDrxlQ9qzaGVH1zN8Gw=";
    }
    {
      version = "3.1.5";
      sha256 = "sha256-+/AHW0V5aF7sjVbTTU2clj5mZ4JVSFVPW/MIYQr3ITM=";
    }
    {
      version = "4.0.0";
      sha256 = "sha256-LwuKNs/rc1S0Xdo8VCXvg5PJsEEVVwthUhP6qj+XNms=";
    }
    {
      version = "4.1.5";
      sha256 = "sha256-pkCYa8JXOJ3TeYhv2uYmTIz6VryYtxzjrj372M5h2+M=";
    }
  ];

  find-version = v:
    builtins.head (builtins.filter (s: s.version == v) versions);

in openmpi-version: final: prev:

{
  openmpi = prev.openmpi.overrideAttrs (new: old:
    let version-found = find-version openmpi-version;
    in rec {

      pname = "openmpi";
      version = version-found.version;
      src = with prev.lib.versions;
        prev.fetchurl {
          url = "https://www.open-mpi.org/software/ompi/v${major version}.${
              minor version
            }/downloads/${pname}-${version}.tar.bz2";
          # sha256 = "sha256-+/AHW0V5aF7sjVbTTU2clj5mZ4JVSFVPW/MIYQr3ITM=";
          sha256 = version-found.sha256;
        };
    })

  ;
}
