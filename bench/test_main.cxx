#include <iostream>
#include <atrip.hpp>
#include <ctf.hpp>
#include <bench/hauta.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CTF::World world(argc, argv);

  const int no(hauta::option<int>(argc, argv, "--no"))
          , nv(hauta::option<int>(argc, argv, "--nv"))
          ;

  std::vector<int> symmetries(4, NS)
                 , vo({nv, no})
                 , vvoo({nv, nv, no, no})
                 , ooov({no, no, no, nv})
                 , vvvo({nv, nv, nv, no})
                 ;

  CTF::Tensor<double>
      ei(1, ooov.data(), symmetries.data(), world)
    , ea(1, vo.data(), symmetries.data(), world)
    , Tph(2, vo.data(), symmetries.data(), world)
    , Tpphh(4, vvoo.data(), symmetries.data(), world)
    , Vpphh(4, vvoo.data(), symmetries.data(), world)
    , Vhhhp(4, ooov.data(), symmetries.data(), world)
    , Vppph(4, vvvo.data(), symmetries.data(), world)
    ;

  atrip::Atrip::init();
  atrip::Atrip::run({&ei, &ea, &Tph, &Tpphh, &Vpphh, &Vhhhp, &Vppph});

  std::cout << "Hello world" << std::endl;
  MPI_Finalize();
  return 0;
}
