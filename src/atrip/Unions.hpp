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

// [[file:~/cuda/atrip/atrip.org::*Unions][Unions:1]]
#pragma once

#include <algorithm>

#include <atrip/SliceUnion.hpp>
#include <atrip/CTFReader.hpp>
#include <atrip/DiskReader.hpp>

namespace atrip {

template <typename F>
class APHH : public SliceUnion<F> {
public:
  APHH(std::string const &tensor_path,
       WITH_CTF(CTF::Tensor<F> const &source_tensor, ) // generalize
       typename Slice<F>::Name name,
       size_t No,
       size_t Nv,
       size_t np,
       MPI_Comm child_world,
       MPI_Comm global_world)
      : SliceUnion<F>({Slice<F>::A, Slice<F>::B, Slice<F>::C},
                      {Nv, No, No} // size of the slices
                      ,
                      {Nv},
                      np,
                      child_world,
                      global_world,
                      name,
                      6) {
    if (tensor_path.size()) {
      this->reader = new DiskReader<APHH<F>>(tensor_path, this, No, Nv);
    } else {
#if defined(HAVE_CTF)
      this->reader = new CTFReader<APHH<F>>(&source_tensor, this, No, Nv);
#endif /* defined (HAVE_CTF) */
    }
    this->init();
  }
};

template <typename F>
class HHHA : public SliceUnion<F> {
public:
  HHHA(std::string const &tensor_path,
       WITH_CTF(CTF::Tensor<F> const &source_tensor, ) // generalize
       typename Slice<F>::Name name,
       size_t No,
       size_t Nv,
       size_t np,
       MPI_Comm child_world,
       MPI_Comm global_world)
      : SliceUnion<F>({Slice<F>::A, Slice<F>::B, Slice<F>::C},
                      {No, No, No} // size of the slices
                      ,
                      {Nv} // size of the parametrization
                      ,
                      np,
                      child_world,
                      global_world,
                      name,
                      6) {
    if (tensor_path.size()) {
      this->reader = new DiskReader<HHHA<F>>(tensor_path, this, No, Nv);
    } else {
#if defined(HAVE_CTF)
      this->reader = new CTFReader<HHHA<F>>(&source_tensor, this, No, Nv);
#endif /* defined (HAVE_CTF) */
    }
    this->init();
  }
};

template <typename F>
class ABPH : public SliceUnion<F> {
public:
  ABPH(std::string const &tensor_path,
       WITH_CTF(CTF::Tensor<F> const &source_tensor, ) // generalize
       typename Slice<F>::Name name,
       size_t No,
       size_t Nv,
       size_t np,
       MPI_Comm child_world,
       MPI_Comm global_world)
      : SliceUnion<F>({Slice<F>::AB,
                       Slice<F>::BC,
                       Slice<F>::AC,
                       Slice<F>::BA,
                       Slice<F>::CB,
                       Slice<F>::CA},
                      {Nv, No}, // size of the slices
                      {Nv, Nv}, // size of the parametrization
                      np,
                      child_world,
                      global_world,
                      name,
                      2 * 6) {
    if (tensor_path.size()) {
      this->reader = new DiskReader<ABPH<F>>(tensor_path, this, No, Nv);
    } else {
#if defined(HAVE_CTF)
      this->reader = new CTFReader<ABPH<F>>(&source_tensor, this, No, Nv);
#endif /* defined (HAVE_CTF) */
    }
    this->init();
  }
};

template <typename F>
class ABHH : public SliceUnion<F> {
public:
  ABHH(std::string const &tensor_path,
       WITH_CTF(CTF::Tensor<F> const &source_tensor, ) // generalize
       typename Slice<F>::Name name,
       size_t No,
       size_t Nv,
       size_t np,
       MPI_Comm child_world,
       MPI_Comm global_world)
      : SliceUnion<F>({Slice<F>::AB, Slice<F>::BC, Slice<F>::AC},
                      {No, No}, // size of the slices
                      {Nv, Nv}, // size of the parametrization
                      np,
                      child_world,
                      global_world,
                      name,
                      6) {
    if (tensor_path.size()) {
      this->reader = new DiskReader<ABHH<F>>(tensor_path, this, No, Nv);
    } else {
#if defined(HAVE_CTF)
      this->reader = new CTFReader<ABHH<F>>(&source_tensor, this, No, Nv);
#endif /* defined (HAVE_CTF) */
    }
    this->init();
  }
};

} // namespace atrip
// Unions:1 ends here
