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

#include <bench/utils.hpp>

#include <atrip/Acc.hpp>
#include <atrip/Chrono.hpp>

Timings timings;

namespace details {

template <typename F>
F one() {
  return F(1);
}
template <>
ACC_DOUBLE_COMPLEX one() {
  return {1.0, 0};
}

template <typename F>
F zero() {
  return F(0);
}
template <>
ACC_DOUBLE_COMPLEX zero() {
  return {0.0, 0.0};
}

template <typename F>
ACC_BLAS_STATUS xgemm(ACC_BLAS_HANDLE handle,
                      const ACC_BLAS_OP transa,
                      const ACC_BLAS_OP transb,
                      const int m,
                      const int n,
                      const int k,
                      F *alpha,
                      const F *A,
                      const int lda,
                      const F *B,
                      const int ldb,
                      F *beta,
                      F *C,
                      const int ldc);

template <>
ACC_BLAS_STATUS xgemm(ACC_BLAS_HANDLE handle,
                      const ACC_BLAS_OP transa,
                      const ACC_BLAS_OP transb,
                      const int m,
                      const int n,
                      const int k,
                      double *alpha,
                      const double *A,
                      const int lda,
                      const double *B,
                      const int ldb,
                      double *beta,
                      double *C,
                      const int ldc) {
  return ACC_BLAS_DGEMM(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc);
}

template <>
ACC_BLAS_STATUS xgemm(ACC_BLAS_HANDLE handle,
                      const ACC_BLAS_OP transa,
                      const ACC_BLAS_OP transb,
                      const int m,
                      const int n,
                      const int k,
                      ACC_DOUBLE_COMPLEX *alpha,
                      const ACC_DOUBLE_COMPLEX *A,
                      const int lda,
                      const ACC_DOUBLE_COMPLEX *B,
                      const int ldb,
                      ACC_DOUBLE_COMPLEX *beta,
                      ACC_DOUBLE_COMPLEX *C,
                      const int ldc) {
  return ACC_BLAS_ZGEMM(handle,
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        A,
                        lda,
                        B,
                        ldb,
                        beta,
                        C,
                        ldc);
}

} // namespace details

template <typename F>
void run(const size_t its,
         const size_t m,
         const size_t n,
         const size_t k,
         const bool zero_p) {
  ACC_DEVICE_PTR A, B, C;
  F one = details::one<F>(),
    zero_or_one = zero_p ? details::zero<F>() : details::one<F>();
  const struct { size_t A, B, C; } sizes = {m * k, k * n, m * n};
  printf("SIZE(A) %f GB\n", sizes.A * sizeof(F) / 1024.0 / 1024.0 / 1024.0);
  printf("SIZE(B) %f GB\n", sizes.B * sizeof(F) / 1024.0 / 1024.0 / 1024.0);
  printf("SIZE(C) %f GB\n", sizes.C * sizeof(F) / 1024.0 / 1024.0 / 1024.0);
  ACC_CHECK_SUCCESS("A", ACC_MEM_ALLOC(&A, sizes.A * sizeof(F)));
  ACC_CHECK_SUCCESS("B", ACC_MEM_ALLOC(&B, sizes.B * sizeof(F)));
  ACC_CHECK_SUCCESS("C", ACC_MEM_ALLOC(&C, sizes.C * sizeof(F)));

  ACC_BLAS_HANDLE handle;
  ACC_BLAS_STATUS stat;
  ACC_CHECK_BLAS("handle create", ACC_BLAS_CREATE(&handle));
  printf("handle %p\n", handle);

  timings["dgemm"].start();
  for (size_t i = 0; i < its; i++) {
    ACC_CHECK_BLAS(_FORMAT(" > 'geming %ld ...", i).c_str(),
                   details::xgemm<F>(handle,
                                     ACC_BLAS_OP_N,
                                     ACC_BLAS_OP_N,
                                     m,
                                     n,
                                     k,
                                     &one,
                                     (F *)A,
                                     m,
                                     (F *)B,
                                     k,
                                     &zero_or_one,
                                     (F *)C,
                                     m));
  }
  ACC_CHECK_SUCCESS("synchronizing", ACC_DEVICE_SYNCHRONIZE());
  timings["dgemm"].stop();
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);
  int rank, np, ngcards;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  MPI_Barrier(MPI_COMM_WORLD);

  CLI::App app{"Simple bench for test dgemm performance C(mn) = A(mk) B(kn)"};
  size_t m, n, k, dgemms;
  bool complex_p, zero_p;
  defoption(app, "-m", m, "")->required();
  defoption(app, "-n", n, "")->required();
  defoption(app, "-k", k, "")->required();
  defoption(app, "--its", dgemms, "")->default_val(100);
  defflag(app, "--complex", complex_p, "")->default_val(false);
  defflag(app, "--zero", zero_p, "")->default_val(false);
  CLI11_PARSE(app, argc, argv);
  if (!rank)
    for (auto const &fn : input_printer) fn();

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

  if (complex_p) {
    printf("doing complex\n");
    run<ACC_DOUBLE_COMPLEX>(dgemms, m, n, k, zero_p);
  } else {
    printf("doing double\n");
    run<double>(dgemms, m, n, k, zero_p);
  }

  const size_t flops = 2 * (complex_p ? 4 : 1) * m * n * k * dgemms;

  printf("m n k: %ld %ld %ld its: %ld dgemm Tflop/s/core: %f np: %d/%d\n",
         m,
         n,
         k,
         dgemms,
         flops / timings["dgemm"].count() / 1024.0 / 1024.0 / 1024.0 / 1024.0,
         rank,
         np);

  MPI_Finalize();
  return 0;
}
