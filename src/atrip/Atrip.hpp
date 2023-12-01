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

// [[file:~/cuda/atrip/atrip.org::*Header][Header:1]]
#ifndef ATRIP_HPP_
#define ATRIP_HPP_

#include <sstream>
#include <string>
#include <map>
#include "config.h"

#include <atrip/Chrono.hpp>
#include <atrip/mpi.hpp>
#include <atrip/Acc.hpp>

#include <atrip/Utils.hpp>
#include <atrip/Types.hpp>
#include <atrip/Tuples.hpp>

#include <atrip/CTF.hpp>

#define ADD_ATTRIBUTE(_type, _name, _default)                                  \
  _type _name = _default;                                                      \
  Input &with_##_name(_type i) {                                               \
    _name = i;                                                                 \
    return *this;                                                              \
  }

namespace atrip {

struct Atrip {

  static size_t rank;
  static size_t np;
  static ClusterInfo *cluster_info;
  static MPI_Comm communicator;
  static Timings chrono;
  static size_t network_send;
  static size_t local_send;
  static double bytes_sent;
  static size_t ppn;
#if defined(HAVE_ACC)
  struct CudaContext {
    ACC_BLAS_STATUS status;
    ACC_BLAS_HANDLE handle;
  };
  static CudaContext cuda;
  static struct KernelDimensions {
    struct {
      size_t blocks, threads;
    } ooo;
  } kernel_dimensions;
#endif

  static void init(MPI_Comm);

  template <typename F = double>
  struct Input {

    enum TuplesDistribution {
      NAIVE,
      GROUP_AND_SORT,
    };

    // Tensor handles coming from CTF
    ADD_ATTRIBUTE(CTF::Tensor<F> *, epsilon_i, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, epsilon_a, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, Tph, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, Tpphh, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, Vpphh, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, Vhhhp, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, Vppph, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, Jppph, nullptr);
    ADD_ATTRIBUTE(CTF::Tensor<F> *, Jhhhp, nullptr);

    // File handles coming from disk reader
    ADD_ATTRIBUTE(std::string, epsilon_i_path, "");
    ADD_ATTRIBUTE(std::string, epsilon_a_path, "");
    ADD_ATTRIBUTE(std::string, Tph_path, "");
    ADD_ATTRIBUTE(std::string, Tpphh_path, "");
    ADD_ATTRIBUTE(std::string, Vpphh_path, "");
    ADD_ATTRIBUTE(std::string, Vhhhp_path, "");
    ADD_ATTRIBUTE(std::string, Vppph_path, "");
    ADD_ATTRIBUTE(std::string, Jppph_path, "");
    ADD_ATTRIBUTE(std::string, Jhhhp_path, "");

    // Miscellaneous options
    ADD_ATTRIBUTE(bool, delete_Vppph, false)
    ADD_ATTRIBUTE(bool, rank_round_robin, false)
    ADD_ATTRIBUTE(bool, chrono, false)
    ADD_ATTRIBUTE(bool, barrier, false)
    ADD_ATTRIBUTE(bool, blocking, false)
    ADD_ATTRIBUTE(size_t, max_iterations, 0)
    ADD_ATTRIBUTE(int, iteration_mod, -1)
    ADD_ATTRIBUTE(int, percentage_mod, -1)
    ADD_ATTRIBUTE(TuplesDistribution, tuples_distribution, NAIVE)
    ADD_ATTRIBUTE(std::string, checkpoint_path, "atrip-checkpoint.yaml")
    ADD_ATTRIBUTE(bool, read_checkpoint_if_exists, true)
    ADD_ATTRIBUTE(bool, writeCheckpoint, true)
    ADD_ATTRIBUTE(float, checkpoint_at_percentage, 10)
    ADD_ATTRIBUTE(size_t, checkpoint_at_every_iteration, 0)
    ADD_ATTRIBUTE(bool, ijkabc, 0)
#if defined(HAVE_ACC)
    ADD_ATTRIBUTE(size_t, ooo_threads, 0)
    ADD_ATTRIBUTE(size_t, ooo_blocks, 0)
#endif
  };

  struct Output {
    double energy;
    double ct_energy;
  };
  template <typename F = double>
  static Output run(Input<F> const &in);
};

} // namespace atrip

#undef ADD_ATTRIBUTE
// Header:1 ends here

#endif
