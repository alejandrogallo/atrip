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

#define _FORMAT(_fmt, ...)                                                     \
  ([&](void) -> std::string {                                                  \
    int _sz = std::snprintf(nullptr, 0, _fmt, __VA_ARGS__);                    \
    std::vector<char> _out(_sz + 1);                                           \
    std::snprintf(&_out[0], _out.size(), _fmt, __VA_ARGS__);                   \
    return std::string(_out.data());                                           \
  })()

#define _CHECK_CUDA_SUCCESS(message, ...)                                      \
  do {                                                                         \
    CUresult result = __VA_ARGS__;                                             \
    printf("doing %s\n", message);                                             \
    if (result != CUDA_SUCCESS) {                                              \
      printf("\t!!CUDA_ERROR(%d): %s:%d %s\n",                                 \
             result,                                                           \
             __FILE__,                                                         \
             __LINE__,                                                         \
             message);                                                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define _CHECK_CUBLAS_SUCCESS(message, ...)                                    \
  do {                                                                         \
    cublasStatus_t result = __VA_ARGS__;                                       \
    printf("CUBLAS: doing %s\n", message);                                     \
    if (result != 0) {                                                         \
      printf("\t!!CUBLAS_ERROR(%d): %s:%d  %s\n",                              \
             result,                                                           \
             __FILE__,                                                         \
             __LINE__,                                                         \
             message);                                                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);
  int rank, np, ngcards;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  Timings timings;

  MPI_Barrier(MPI_COMM_WORLD);

  _CHECK_CUDA_SUCCESS("init for cuda", cuInit(0));

  _CHECK_CUDA_SUCCESS("get ncards", cuDeviceGetCount(&ngcards));

  const size_t N = argc > 1 ? atoi(argv[1]) : 30000,
               dgemms = argc > 2 ? atoi(argv[2]) : 2,
               flops = 2 * N * N * N * dgemms;

  CUcontext ctx;
  CUdevice dev;

  char hostname[256];
  gethostname(hostname, 256);
  printf("%s with rank %d gets card %d\n", hostname, rank, rank % ngcards);

  // set contexts
  _CHECK_CUDA_SUCCESS("device get", cuDeviceGet(&dev, rank % ngcards));
  _CHECK_CUDA_SUCCESS("creating context", cuCtxCreate(&ctx, 0, dev));
  _CHECK_CUDA_SUCCESS("setting context", cuCtxSetCurrent(ctx));
  _CHECK_CUDA_SUCCESS("synchronizing", cuCtxSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  CUdeviceptr A, B, C;
  const double one = 1.0;
  printf("SIZE %f GB\n", N * N * sizeof(double) / 1024.0 / 1024.0 / 1024.0);
  _CHECK_CUDA_SUCCESS("A", cuMemAlloc(&A, N * N * sizeof(double)));
  _CHECK_CUDA_SUCCESS("B", cuMemAlloc(&B, N * N * sizeof(double)));
  _CHECK_CUDA_SUCCESS("C", cuMemAlloc(&C, N * N * sizeof(double)));

  cublasHandle_t handle;
  cublasStatus_t stat;
  _CHECK_CUBLAS_SUCCESS("handle create", cublasCreate(&handle));
  printf("handle %ld\n", handle);

  timings["dgemm"].start();
  for (size_t i = 0; i < dgemms; i++) {
    _CHECK_CUBLAS_SUCCESS(_FORMAT(" > 'geming %ld ...", i).c_str(),
                          cublasDgemm(handle,
                                      CUBLAS_OP_N,
                                      CUBLAS_OP_N,
                                      N,
                                      N,
                                      N,
                                      &one,
                                      (double *)A,
                                      N,
                                      (double *)B,
                                      N,
                                      &one,
                                      (double *)C,
                                      N));
  }

  cuCtxSynchronize();
  timings["dgemm"].stop();

  printf("dgemm Gflops: %f\n",
         flops / timings["dgemm"].count() / 1024.0 / 1024.0 / 1024.0);

  MPI_Finalize();
  return 0;
}

// Local Variables:
// compile-command: "mpic++                         \
//      -pedantic -std=c++11                        \
//      -L./cudaroot/lib64 -lcuda                   \
//      -L./cudaroot/lib64 -lcudart                 \
//      -L./cudaroot/lib64 -lcublas                 \
//      ./test-cublas-parallel.cxx -o test-cublas-parallel"
// End:
