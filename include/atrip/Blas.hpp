// Copyright 2022 Alejandro Gallo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// [[file:../../atrip.org::*Blas][Blas:1]]
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
