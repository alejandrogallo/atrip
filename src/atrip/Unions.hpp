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
static void slice_into_vector
#if defined(ATRIP_SOURCES_IN_GPU)
    (DataPtr<F> &source,
#else
    (std::vector<F> &source,
#endif
     size_t slice_size,
     CTF::Tensor<F> &to_slice,
     std::vector<int> const low,
     std::vector<int> const up,
     CTF::Tensor<F> const &origin,
     std::vector<int> const origin_low,
     std::vector<int> const origin_up) {

  WITH_OCD
  WITH_RANK << "slicing into " << up << "," << low << " from " << origin_up
            << "," << origin_low << "\n";

#if !defined(ATRIP_DONT_SLICE) && !defined(ATRIP_DRY)
  to_slice.slice(low.data(),
                 up.data(),
                 0.0,
                 origin,
                 origin_low.data(),
                 origin_up.data(),
                 1.0);

#  if defined(ATRIP_SOURCES_IN_GPU)
  WITH_CHRONO(
      "acc:sources",
      ACC_CHECK_SUCCESS("copying sources data to device",
                        ACC_MEMCPY_HOST_TO_DEV(source,
                                               to_slice.data,
                                               sizeof(F) * slice_size));)
#  else
  memcpy(source.data(), to_slice.data, sizeof(F) * slice_size);
#  endif /* defined(ATRIP_SOURCES_IN_GPU) */

#else
  IGNORABLE(source);
  IGNORABLE(slice_size);
  IGNORABLE(to_slice);
  IGNORABLE(origin);
#  pragma message("WARNING: COMPILING WITHOUT SLICING THE TENSORS")
#endif /* !defined(ATRIP_DONT_SLICE) && !defined(ATRIP_DRY) */
}

template <typename F>
class APHH : public SliceUnion<F> {
public:
  APHH(CTF::Tensor<F> const &source_tensor,
       std::string const &tensor_path,
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
      this->reader = new CTFReader<APHH<F>>(&source_tensor, this, No, Nv);
    }
    this->init();
  }
};

template <typename F>
class HHHA : public SliceUnion<F> {
public:
  HHHA(CTF::Tensor<F> const &source_tensor,
       std::string const &tensor_path,
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
      this->reader = new CTFReader<HHHA<F>>(&source_tensor, this, No, Nv);
    }
    this->init();
  }
};

template <typename F>
class ABPH : public SliceUnion<F> {
public:
  ABPH(CTF::Tensor<F> const &source_tensor,
       std::string const &tensor_path,
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
      this->reader = new CTFReader<ABPH<F>>(&source_tensor, this, No, Nv);
    }
    this->init();
  }
};

template <typename F>
class ABHH : public SliceUnion<F> {
public:
  ABHH(CTF::Tensor<F> const &source_tensor,
       std::string const &tensor_path,
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
      this->reader = new CTFReader<ABHH<F>>(&source_tensor, this, No, Nv);
    }
    this->init();
  }
};

} // namespace atrip
// Unions:1 ends here
