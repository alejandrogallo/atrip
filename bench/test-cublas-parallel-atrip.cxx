#include <mpi.h>
#include <iostream>
#include <cassert>
#include <string.h>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <map>
#include <cstdlib>
#include <unistd.h>

#include <CLI11.hpp>

#include "config.h"

#include <atrip/Acc.hpp>
#include <atrip/Chrono.hpp>

int main(int argc, char **argv) {

  using std::vector;

  MPI_Init(NULL, NULL);
  int rank, np, ngcards;
  size_t no(10), nv(no * 10), its(2);
  bool barrier = false;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  CLI::App app{"Main bench for atrip"};
  app.add_option("--no", no, "Occupied orbitals");
  app.add_option("--nv", nv, "Virtual orbitals");
  app.add_option("--its", its, "Number of iterations to be done");
  app.add_option("--barrier",
                 barrier,
                 "Call a MPI_Barrier in every iteration?");
  CLI11_PARSE(app, argc, argv);

  const size_t oo = no * no, ooo = no * oo;

  Timings timings;

  MPI_Barrier(MPI_COMM_WORLD);

  ACC_CHECK_SUCCESS("init for cuda", ACC_INIT(0));

  ACC_CHECK_SUCCESS("get ncards", ACC_DEVICE_GET_COUNT(&ngcards));

  ACC_CONTEXT ctx;
  ACC_DEVICE dev;

  char hostname[256];
  gethostname(hostname, 256);
  printf("%s with rank %d gets card %d\n", hostname, rank, rank % ngcards);

  // set contexts
  ACC_CHECK_SUCCESS("device get", ACC_DEVICE_GET(&dev, rank % ngcards));
  ACC_CHECK_SUCCESS("creating context", ACC_CONTEXT_CREATE(&ctx, 0, dev));
  ACC_CHECK_SUCCESS("setting context", ACC_CONTEXT_SET_CURRENT(ctx));
  ACC_CHECK_SUCCESS("synchronizing", ACC_DEVICE_SYNCHRONIZE());
  MPI_Barrier(MPI_COMM_WORLD);

  using host_slice_t = vector<double>;

  vector<size_t> sizes = {nv * oo, nv * no, oo, oo * no, oo * no};
  vector<ACC_DEVICE_PTR> P_phh(3), P_ph(6), H_hh(3), H_hhh(3), T_hhh(1);
  vector<vector<ACC_DEVICE_PTR> *> slices_d = {&P_phh,
                                            &P_ph,
                                            &H_hh,
                                            &H_hhh,
                                            &T_hhh};
  vector<vector<host_slice_t>> slices_h(slices_d.size());
  {
    int i = -1;
    for (auto &v : slices_d) {
      i++;
      for (auto &ptr : *v) {
        ACC_CHECK_SUCCESS("malloc",
                            ACC_MEM_ALLOC(&ptr, sizes[i] * sizeof(double)));
        slices_h[i].push_back(std::move(std::vector<double>(sizes[i])));
      }
    }
  }

  const double one = 1.0, zero = 0.0;

  printf("its: %d\n", its);
  printf("barrier: %d\n", barrier);
  printf("no: %ld\n", no);
  printf("nv: %ld\n", nv);
  printf("SIZE %f GB\n",
         (3 * nv * oo + 6 * no * nv + 3 * oo + 3 * ooo + 1 * ooo)
             * sizeof(double) / 1024.0 / 1024.0 / 1024.0);
  std::map<std::string, double> tflopss{
      {"dgemm", ooo * (no + nv) * 6.0 * 2.0 * its / 1e12},
      {"holes", ooo * no * 6.0 * 2.0 * its / 1e12},
      {"particles", ooo * nv * 6.0 * 2.0 * its / 1e12}};

  ACC_BLAS_HANDLE handle;
  ACC_CHECK_BLAS("handle create", ACC_BLAS_CREATE(&handle));
  printf("handle %ld\n", handle);

  timings["dgemm"].start();
  for (size_t i = 0; i < its; i++) {

    if (barrier) {
      MPI_Barrier(MPI_COMM_WORLD);
      timings["memcpy"].start();
      for (size_t _s = 0; _s < slices_d.size(); _s++) {
        for (size_t _b = 0; _b < slices_h[_s].size(); _b++) {
          // for (size_t _b = 0; _b < 1 ; _b++) {
          auto device = (*slices_d[_s])[_b];
          auto host = slices_h[_s][_b].data();
          ACC_MEMCPY_HOST_TO_DEV(device, host, sizes[_s]);
        }
      }
      timings["memcpy"].stop();
    }

    timings["holes"].start();
    for (size_t j = 0; j < 3; j++) {

      ACC_CHECK_BLAS(" > 'geming ...",
                            ACC_BLAS_DGEMM(handle,
                                        ACC_BLAS_OP_N,
                                        ACC_BLAS_OP_N,
                                        oo,
                                        no,
                                        no,
                                        &one,
                                        (double *)H_hhh[j],
                                        oo,
                                        (double *)H_hh[j],
                                        no,
                                        &zero,
                                        (double *)T_hhh[0],
                                        oo));

      ACC_CHECK_BLAS(" > 'geming ...",
                            ACC_BLAS_DGEMM(handle,
                                        ACC_BLAS_OP_N,
                                        ACC_BLAS_OP_T,
                                        oo,
                                        no,
                                        no,
                                        &one,
                                        (double *)H_hhh[j],
                                        oo,
                                        (double *)H_hh[j],
                                        no,
                                        &zero,
                                        (double *)T_hhh[0],
                                        oo));
    }
    timings["holes"].stop();

    timings["particles"].start();
    for (size_t j = 0; j < 6; j++) {
      ACC_CHECK_BLAS(" > 'geming ...",
                            ACC_BLAS_DGEMM(handle,
                                        ACC_BLAS_OP_T,
                                        ACC_BLAS_OP_N,
                                        oo,
                                        no,
                                        nv,
                                        &one,
                                        (double *)P_phh[j % 3],
                                        nv,
                                        (double *)P_ph[j],
                                        nv,
                                        &zero,
                                        (double *)T_hhh[0],
                                        oo));
    }
    timings["particles"].stop();

    ACC_DEVICE_SYNCHRONIZE();
  }

  timings["dgemm"].stop();

  printf("Performance: \n");
  for (auto name : {"holes", "particles", "dgemm"})
    printf("no: %ld %10s TFlops: %4.1f its: %ld np: %d/%d\n",
		    no,
		    name,
		    tflopss[name] / timings[name].count(),
		    its,
		    rank,
		    np);

  printf("Timings: \n");
  for (auto const &kv : timings)
    printf("%10s: %10f\n", kv.first.c_str(), kv.second.count());

  MPI_Finalize();
  return 0;
}
