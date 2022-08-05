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

// [[file:~/cuda/atrip/atrip.org::*Complex%20numbers][Complex numbers:2]]
#include <atrip/Complex.hpp>
#include <atrip/CUDA.hpp>

namespace atrip {

  template <> double maybeConjugate(const double a) { return a; }
  template <> Complex maybeConjugate(const Complex a) { return std::conj(a); }

#if defined(HAVE_CUDA)

#endif


  namespace traits {
    template <typename F> bool isComplex() { return false; }
    template <> bool isComplex<double>() { return false; }
    template <> bool isComplex<Complex>() { return true; }
  namespace mpi {
    template <> MPI_Datatype datatypeOf<double>() { return MPI_DOUBLE; }
    template <> MPI_Datatype datatypeOf<Complex>() { return MPI_DOUBLE_COMPLEX; }
  }
  }

}
// Complex numbers:2 ends here
