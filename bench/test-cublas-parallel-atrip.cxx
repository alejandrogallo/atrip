#include <mpi.h>
#include <iostream>
#include <cassert>
#include <string.h>
#include <sstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <chrono>
#include <map>
#include <cstdlib>
#include <unistd.h>

#include <CLI11.hpp>

#define CUBLASAPI
#include <cublas_api.h>
#include <cublas_v2.h>

struct Timer {
  using Clock = std::chrono::high_resolution_clock;
  using Event = std::chrono::time_point<Clock>;
  std::chrono::duration<double> duration;
  Event _start;
  inline void start() noexcept { _start = Clock::now(); }
  inline void stop() noexcept { duration += Clock::now() - _start; }
  inline void clear() noexcept { duration *= 0; }
  inline double count() const noexcept { return duration.count(); }
};
using Timings = std::map<std::string, Timer>;

#define _FORMAT(_fmt, ...)                                    \
  ([&] (void) -> std::string {                                \
     int _sz = std::snprintf(nullptr, 0, _fmt, __VA_ARGS__);  \
     std::vector<char>  _out(_sz  +  1);                      \
     std::snprintf(&_out[0], _out.size(), _fmt, __VA_ARGS__); \
     return std::string(_out.data());                         \
   })()

#define _CHECK_CUDA_SUCCESS(message, ...)                               \
  do {                                                                  \
    CUresult result = __VA_ARGS__;                                      \
    printf("doing %s\n", message);                                      \
    if (result != CUDA_SUCCESS) {                                       \
      printf("\t!!CUDA_ERROR(%d): %s:%d %s\n",                          \
             result,                                                    \
             __FILE__,                                                  \
             __LINE__,                                                  \
             message);                                                  \
      return 1;                                                         \
    }                                                                   \
  } while (0)

#define _CHECK_CUBLAS_SUCCESS(message, ...)                             \
  do {                                                                  \
    cublasStatus_t result = __VA_ARGS__;                                \
    if (result != 0) {                                                  \
      printf("\t!!CUBLAS_ERROR(%d): %s:%d  %s\n",                       \
             result,                                                    \
             __FILE__,                                                  \
             __LINE__,                                                  \
             message);                                                  \
      return 1;                                                         \
    }                                                                   \
  } while (0)

int main(int  argc, char** argv) {

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
  app.add_option("--barrier", barrier, "Call a MPI_Barrier in every iteration?");
  CLI11_PARSE(app, argc, argv);

  const size_t oo = no * no, ooo = no * oo;

  Timings timings;

  MPI_Barrier(MPI_COMM_WORLD);

  _CHECK_CUDA_SUCCESS("init for cuda",
                      cuInit(0));

  _CHECK_CUDA_SUCCESS("get ncards",
                      cuDeviceGetCount(&ngcards));

  CUcontext ctx;
  CUdevice dev;

  char hostname[256];
  gethostname(hostname, 256);
  printf("%s with rank %d gets card %d\n",
         hostname,
         rank,
         rank % ngcards);

  // set contexts
  _CHECK_CUDA_SUCCESS("device get",       cuDeviceGet(&dev, rank % ngcards));
  _CHECK_CUDA_SUCCESS("creating context", cuCtxCreate(&ctx, 0, dev));
  _CHECK_CUDA_SUCCESS("setting context",  cuCtxSetCurrent(ctx));
  _CHECK_CUDA_SUCCESS("synchronizing",    cuCtxSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  using host_slice_t = vector<double>;

  vector<size_t> sizes = {nv * oo,  nv * no , oo,      oo * no,  oo * no};
  vector<CUdeviceptr>     P_phh(3), P_ph(6) , H_hh(3), H_hhh(3), T_hhh(1);
  vector<vector<CUdeviceptr>*> slices_d = {&P_phh, &P_ph , &H_hh, &H_hhh, &T_hhh};
  vector<vector<host_slice_t>> slices_h(slices_d.size());
  {
    int i = -1;
    for (auto& v: slices_d) {
      i++;
      for (auto& ptr: *v) {
        _CHECK_CUDA_SUCCESS("malloc",
                            cuMemAlloc(&ptr,
                                       sizes[i] * sizeof(double)));
        slices_h[i].push_back(std::move(std::vector<double>(sizes[i])));
      }
    }
  }

  const double one = 1.0, zero = 0.0;

  printf("its: %d\n", its);
  printf("barrier: %d\n", barrier);
  printf("no: %ld\n", no);
  printf("nv: %ld\n", nv);
  printf("SIZE %f GB\n", (3 * nv * oo
                          + 6 * no * nv
                          + 3 * oo
                          + 3 * ooo
                          + 1 * ooo
                          ) * sizeof(double) / 1024.0 / 1024.0 / 1024.0);
  std::map<std::string, double> tflopss
    {{ "dgemm",     ooo * (no + nv) * 6.0 * 2.0 * its / 1e12},
     { "holes",     ooo * no * 6.0 * 2.0 * its / 1e12},
     { "particles", ooo * nv * 6.0 * 2.0 * its / 1e12}};

  cublasHandle_t handle;
  _CHECK_CUBLAS_SUCCESS("handle create", cublasCreate(&handle));
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
          auto host   = slices_h[_s][_b].data();
          cuMemcpyHtoD(device, host, sizes[_s]);
        }
      }
      timings["memcpy"].stop();
    }


    timings["holes"].start();
    for (size_t j = 0; j < 3; j++) {

      _CHECK_CUBLAS_SUCCESS(" > 'geming ...",
                            cublasDgemm(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        oo, no, no,
                                        &one,
                                        (double*)H_hhh[j], oo,
                                        (double*)H_hh[j], no,
                                        &zero,
                                        (double*)T_hhh[0], oo));

      _CHECK_CUBLAS_SUCCESS(" > 'geming ...",
                            cublasDgemm(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_T,
                                        oo, no, no,
                                        &one,
                                        (double*)H_hhh[j], oo,
                                        (double*)H_hh[j], no,
                                        &zero,
                                        (double*)T_hhh[0], oo));


    }
    timings["holes"].stop();

    timings["particles"].start();
    for (size_t j = 0; j < 6; j++) {
      _CHECK_CUBLAS_SUCCESS(" > 'geming ...",
                            cublasDgemm(handle,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        oo, no, nv,
                                        &one,
                                        (double*)P_phh[j % 3], nv,
                                        (double*)P_ph[j], nv,
                                        &zero,
                                        (double*)T_hhh[0], oo));
    }
    timings["particles"].stop();

    cuCtxSynchronize();

  }


  timings["dgemm"].stop();



  printf("Performance: \n");
  for (auto name: {"holes", "particles", "dgemm"})
    printf("%10s TFlops: %4.1f\n",
          name,
          tflopss[name]
          / timings[name].count());

  printf("Timings: \n");
  for (auto const& kv: timings)
    printf("%10s: %10f\n", kv.first.c_str(), kv.second.count());

  MPI_Finalize();
  return 0;

}
