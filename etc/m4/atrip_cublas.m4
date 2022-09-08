

dnl Create some tests for the compilers to work with CUDA







AC_DEFUN([ATRIP_CUBLAS],
[

AC_LANG_PUSH([C++])dnl
atrip_success=no
ac_save_CXX="$CXX"
ac_save_CXXFLAGS="$CXXFLAGS"
ac_save_LDFLAGS="$LDFLAGS"


CXXFLAGS="${CXXFLAGS} $CUDA_CXXFLAGS"
LDFLAGS="${LDFLAGS} $CUDA_LDFLAGS"

AC_MSG_CHECKING([that cublas works with the CXX compiler])

AC_COMPILE_IFELSE([AC_LANG_SOURCE([_ATRIP_CUBLAS_SOURCE])],
	[
	atrip_success=yes
	AC_MSG_RESULT([yes])
	],
	[
	atrip_success=no
	AC_MSG_ERROR([Does not work!])
	])

AC_MSG_CHECKING([that the detected graphic cards have memory available])
AC_RUN_IFELSE([AC_LANG_SOURCE([_ATRIP_CUDA_MEMORY_OF_DEVICES])],
	[
	atrip_success=yes
	],
	[
	atrip_success=no
	AC_MSG_WARN([An available device reports zero memory available!])
	])

CXX="$ac_save_CXX"
CXXFLAGS="$ac_save_CXXFLAGS"
LDFLAGS="$ac_save_LDFLAGS"

AC_LANG_POP([C++])dnl
])dnl DEFUN

m4_define([_ATRIP_CUBLAS_SOURCE], [[
#include <mpi.h>
#include <mpi.h>
#include <iostream>
#include <string.h>
#include <cublas_v2.h>
#include <cuda.h>

int main() {
  MPI_Init(NULL, NULL);
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);
  cudaError_t error;

  int64_t els = 100, oo, No, Nv;
  double *PH_d, *HHP_d, *HHH_d, *F_d;
  double one(1.0);

  cuMemAlloc((CUdeviceptr*)&PH_d, els*sizeof(double));
  cuMemAlloc((CUdeviceptr*)&HHP_d, els*sizeof(double));
  cuMemAlloc((CUdeviceptr*)&HHH_d, els*sizeof(double));
  cuMemAlloc((CUdeviceptr*)&F_d, els*sizeof(double));

  stat = cublasDgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    oo, No, Nv,
                    &one,
                    HHP_d, oo, PH_d, Nv, &one, F_d, oo);
  //cudaSetDevice(rank);

  return 0;
}
]])


m4_define([_ATRIP_CUDA_MEMORY_OF_DEVICES], [[

#include <mpi.h>
#include <iostream>
#include <cassert>
#include <string.h>
#include <cuda.h>

int main() {
  MPI_Init(NULL, NULL);
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int ngcards;
  cuDeviceGetCount(&ngcards);

  for (size_t rank = 0; rank < ngcards; rank++) {
    CUcontext ctx;
    CUdevice dev;
    CUdevprop_st prop;
    size_t _free, total, total2;
    char *name = (char*)malloc(256);

    cuDeviceGet(&dev, rank);
    cuCtxCreate(&ctx, 0, dev);
    cuCtxSetCurrent(ctx);

    cuDeviceGetProperties(&prop, dev);
    cuMemGetInfo(&_free, &total);
    cuDeviceGetName(name, 256, dev);
    cuDeviceTotalMem(&total2, dev);

    printf("\n"
           "CUDA CARD RANK %d\n"
           "=================\n"
           "\tname: %s\n"
           "\tShared Mem Per Block (KB): %f\n"
           "\tFree/Total mem (GB): %f/%f\n"
           "\ttotal2 mem (GB): %f\n"
           "\n",
           dev,
           name,
           prop.sharedMemPerBlock / 1024.0,
           _free / 1024.0 / 1024.0 / 1024.0 ,
           total / 1024.0 / 1024.0 / 1024.0 ,
           total2 / 1024.0 / 1024.0 / 1024.0
           );

    if (_free == 0 || total == 0 || total2 == 0)
      return 1;

  }

  return 0;
}

]])
