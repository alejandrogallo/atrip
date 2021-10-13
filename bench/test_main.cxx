#include <iostream>
#include <atrip.hpp>
#include <ctf.hpp>
#include <bench/hauta.h>

#define _print_size(what, size)                 \
  if (rank == 0) {                              \
    std::cout << #what                          \
              << " => "                         \
              << (double)size * elem_to_gb      \
              << "GB"                           \
              << std::endl;                     \
  }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CTF::World world(argc, argv);
  int rank;
  MPI_Comm_rank(world.comm, &rank);
  constexpr double elem_to_gb = 8.0 / 1024.0 / 1024.0 / 1024.0;
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

  _print_size(Vabci, no*nv*nv*nv)
  _print_size(Vabij, no*no*nv*nv)
  _print_size(Vijka, no*no*no*nv)

  ei.fill_random(-40.0, -2);
  ea.fill_random(2, 50);
  Tpphh.fill_random(0, 1);
  Tph.fill_random(0, 1);
  Vpphh.fill_random(0, 1);
  Vhhhp.fill_random(0, 1);
  Vppph.fill_random(0, 1);

  atrip::Atrip::init();
  atrip::Atrip::Input in;

  in
    // Tensors
    .with_epsilon_i(&ei)
    .with_epsilon_a(&ea)
    .with_Tai(&Tph)
    .with_Tabij(&Tpphh)
    .with_Vabij(&Vpphh)
    .with_Vijka(&Vhhhp)
    .with_Vabci(&Vppph)
    // some options
    .with_barrier(false)
    .with_iterationMod(100)
    ;

  auto out = atrip::Atrip::run(in);

  if (atrip::Atrip::rank == 0)
    std::cout << "Energy: " << out.energy << std::endl;

  MPI_Finalize();
  return 0;
}
