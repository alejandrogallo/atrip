// [[file:~/atrip/atrip.org::*Blas][Blas:1]]
#pragma once
namespace atrip {
  extern "C" {
    void dgemm_(
      const char *transa,
      const char *transb,
      const int *m,
      const int *n,
      const int *k,
      double *alpha,
      const double *A,
      const int *lda,
      const double *B,
      const int *ldb,
      double *beta,
      double *C,
      const int *ldc
    );
  }
}
// Blas:1 ends here
