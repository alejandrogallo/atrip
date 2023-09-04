#include <mpi.h>
#include <iostream>
#include <cassert>
#include <string.h>
#include <sstream>
#include <string>
#include <vector>

#include "config.h"

#include <atrip/Acc.hpp>

int main() {
  int rank, np, ngcards;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  ACC_CHECK_SUCCESS("init for cuda", ACC_INIT(0));

  ACC_CHECK_SUCCESS("get ncards", ACC_DEVICE_GET_COUNT(&ngcards));

  for (size_t rank = 0; rank < ngcards; rank++) {
    ACC_CONTEXT ctx;
    ACC_DEVICE dev;
    size_t _free, total, total2;
    char *name = (char *)malloc(256);

    printf("Setting contexts\n");
    // set contexts
    ACC_CHECK_SUCCESS("device get", ACC_DEVICE_GET(&dev, rank));
    ACC_CHECK_SUCCESS("creating context", ACC_CONTEXT_CREATE(&ctx, 0, dev));
    ACC_CHECK_SUCCESS("setting context", ACC_CONTEXT_SET_CURRENT(ctx));
    ACC_CHECK_SUCCESS("synchronizing", ACC_DEVICE_SYNCHRONIZE());

    ACC_CHECK_SUCCESS("meminfo get", ACC_MEM_GET_INFO(&_free, &total));
    ACC_CHECK_SUCCESS("name get", ACC_GET_DEVICE_NAME(name, 256, dev));
    ACC_CHECK_SUCCESS("totalmem get", ACC_DEVICE_TOTAL_MEM(&total2, dev));

    printf(
        "\n"
        "CUDA CARD RANK %d\n"
        "=================\n"
        "\tname: %s\n"
        "\tFree/Total mem (GB): %f/%f\n"
        "\total2 mem (GB): %f\n"
        "\n",
        dev,
        name,
        _free / 1024.0 / 1024.0 / 1024.0,
        total / 1024.0 / 1024.0 / 1024.0,
        total2 / 1024.0 / 1024.0 / 1024.0);

    if (_free == 0 || total == 0 || total2 == 0) return 1;

    ACC_DEVICE_PTR data;
    ACC_CHECK_SUCCESS("memalloc 1",
                        ACC_MEM_ALLOC(&data, sizeof(double) * 10000));
    ACC_CHECK_SUCCESS("memalloc 2",
                        ACC_MEM_ALLOC(&data, sizeof(double) * 10000));
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
