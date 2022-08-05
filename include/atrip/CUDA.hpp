#pragma once

#if defined(HAVE_CUDA) && defined(__CUDACC__)
#  define __MAYBE_GLOBAL__ __global__
#  define __MAYBE_DEVICE__ __device__
#else
#  define __MAYBE_GLOBAL__
#  define __MAYBE_DEVICE__
#endif
