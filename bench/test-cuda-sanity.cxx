#include <mpi.h>
#include <iostream>
#include <cassert>
#include <string.h>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>

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

int main() {
  int rank, np, ngcards;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  _CHECK_CUDA_SUCCESS("init for cuda", cuInit(0));

  _CHECK_CUDA_SUCCESS("get ncards", cuDeviceGetCount(&ngcards));

  for (size_t rank = 0; rank < ngcards; rank++) {
    CUcontext ctx;
    CUdevice dev;
    CUdevprop_st prop;
    size_t _free, total, total2;
    char *name = (char *)malloc(256);

    printf("Setting contexts\n");
    // set contexts
    _CHECK_CUDA_SUCCESS("device get", cuDeviceGet(&dev, rank));
    _CHECK_CUDA_SUCCESS("creating context", cuCtxCreate(&ctx, 0, dev));
    _CHECK_CUDA_SUCCESS("setting context", cuCtxSetCurrent(ctx));
    _CHECK_CUDA_SUCCESS("synchronizing", cuCtxSynchronize());

    _CHECK_CUDA_SUCCESS("prop get", cuDeviceGetProperties(&prop, dev));
    _CHECK_CUDA_SUCCESS("meminfo get", cuMemGetInfo(&_free, &total));
    _CHECK_CUDA_SUCCESS("name get", cuDeviceGetName(name, 256, dev));
    _CHECK_CUDA_SUCCESS("totalmem get", cuDeviceTotalMem(&total2, dev));

    printf(
        "\n"
        "CUDA CARD RANK %d\n"
        "=================\n"
        "\tname: %s\n"
        "\tShared Mem Per Block (KB): %f\n"
        "\tFree/Total mem (GB): %f/%f\n"
        "\total2 mem (GB): %f\n"
        "\n",
        dev,
        name,
        prop.sharedMemPerBlock / 1024.0,
        _free / 1024.0 / 1024.0 / 1024.0,
        total / 1024.0 / 1024.0 / 1024.0,
        total2 / 1024.0 / 1024.0 / 1024.0);

    if (_free == 0 || total == 0 || total2 == 0) return 1;

    CUdeviceptr data;
    _CHECK_CUDA_SUCCESS("memalloc 1",
                        cuMemAlloc(&data, sizeof(double) * 10000));
    _CHECK_CUDA_SUCCESS("memalloc 2",
                        cuMemAlloc(&data, sizeof(double) * 10000));
  }

  MPI_Finalize();

  return 0;
}

// Local Variables:
// compile-command: "mpic++                         \
//      -pedantic -std=c++11                        \
//      -L./cudaroot/lib64 -lcuda                   \
//      -L./cudaroot/lib64 -lcudart                 \
//      -L./cudaroot/lib64 -lcublas                 \
//      ./mem.cxx -o mem"
// End:
