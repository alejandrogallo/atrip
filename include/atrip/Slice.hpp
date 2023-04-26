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

// [[file:~/cuda/atrip/atrip.org::*Prolog][Prolog:1]]
#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <mpi.h>

#include <atrip/Tuples.hpp>
#include <atrip/Utils.hpp>
#include <atrip/CUDA.hpp>

namespace atrip {

template <typename F = double>
struct Slice {
  // Prolog:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Location][Location:1]]
  struct Location {
    size_t rank;
    size_t source;
  };
  // Location:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Type][Type:1]]
  enum Type {
    A = 10,
    B,
    C,

    // Two-parameter slices
    AB = 20,
    BC,
    AC,

    // for abci and the doubles
    CB,
    BA,
    CA,

    // The non-typed slice
    Blank = 404
  };
  // Type:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*State][State:1]]
  enum State {
    Fetch = 0,
    Dispatched = 2,
    Ready = 1,
    SelfSufficient = 911,
    Recycled = 123,
    Acceptor = 405
  };
  // State:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*The%20Info%20structure][The Info
  // structure:1]]
  struct Info {
    // which part of a,b,c the slice holds
    PartialTuple tuple;
    // The type of slice for the user to retrieve the correct one
    Type type;
    // What is the state of the slice
    State state;
    // Where the slice is to be retrieved
    Location from;
    // If the data are actually to be found in this other slice
    Type recycling;

    Info()
        : tuple{0, 0}
        , type{Blank}
        , state{Acceptor}
        , from{0, 0}
        , recycling{Blank} {}
  };

  using Ty_x_Tu = std::pair<Type, PartialTuple>;
  // The Info structure:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Name][Name:1]]
  enum Name {
    TA = 100,
    VIJKA = 101,
    VABCI = 200,
    TABIJ = 201,
    VABIJ = 202
  };
  // Name:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Database][Database:1]]
  struct LocalDatabaseElement {
    Slice<F>::Name name;
    Slice<F>::Info info;
  };
  // Database:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Database][Database:2]]
  using LocalDatabase = std::vector<LocalDatabaseElement>;
  using Database = LocalDatabase;
  // Database:2 ends here

  // [[file:~/cuda/atrip/atrip.org::*MPI%20Types][MPI Types:1]]
  struct mpi {

    static MPI_Datatype vector(size_t n, MPI_Datatype const &DT) {
      MPI_Datatype dt;
      MPI_Type_vector(n, 1, 1, DT, &dt);
      MPI_Type_commit(&dt);
      return dt;
    }

    static MPI_Datatype sliceLocation() {
      constexpr int n = 2;
      // create a sliceLocation to measure in the current architecture
      // the packing of the struct
      Slice<F>::Location measure;
      MPI_Datatype dt;
      const std::vector<int> lengths(n, 1);
      const MPI_Datatype types[n] = {usizeDt(), usizeDt()};

      static_assert(sizeof(Slice<F>::Location) == 2 * sizeof(size_t),
                    "The Location packing is wrong in your compiler");

      // measure the displacements in the struct
      size_t j = 0;
      MPI_Aint base_address, displacements[n];
      MPI_Get_address(&measure, &base_address);
      MPI_Get_address(&measure.rank, &displacements[j++]);
      MPI_Get_address(&measure.source, &displacements[j++]);
      for (size_t i = 0; i < n; i++)
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);

      MPI_Type_create_struct(n, lengths.data(), displacements, types, &dt);
      MPI_Type_commit(&dt);
      return dt;
    }

    static MPI_Datatype usizeDt() { return MPI_UINT64_T; }

    static MPI_Datatype sliceInfo() {
      constexpr int n = 5;
      MPI_Datatype dt;
      Slice<F>::Info measure;
      const std::vector<int> lengths(n, 1);
      const MPI_Datatype types[n] = {
          vector(2, usizeDt()),
          vector(sizeof(enum Type), MPI_CHAR),
          vector(sizeof(enum State), MPI_CHAR),
          sliceLocation(),
          vector(sizeof(enum Type), MPI_CHAR)
          // TODO: Why this does not work on intel mpi?
          /*, MPI_UINT64_T*/
      };

      static_assert(sizeof(enum Type) == 4, "Enum type not 4 bytes long");
      static_assert(sizeof(enum State) == 4, "Enum State not 4 bytes long");
      static_assert(sizeof(enum Name) == 4, "Enum Name not 4 bytes long");

      // create the displacements from the info measurement struct
      size_t j = 0;
      MPI_Aint base_address, displacements[n];
      MPI_Get_address(&measure, &base_address);
      MPI_Get_address(&measure.tuple[0], &displacements[j++]);
      MPI_Get_address(&measure.type, &displacements[j++]);
      MPI_Get_address(&measure.state, &displacements[j++]);
      MPI_Get_address(&measure.from, &displacements[j++]);
      MPI_Get_address(&measure.recycling, &displacements[j++]);
      for (size_t i = 0; i < n; i++)
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);

      MPI_Type_create_struct(n, lengths.data(), displacements, types, &dt);
      MPI_Type_commit(&dt);
      return dt;
    }

    static MPI_Datatype localDatabaseElement() {
      constexpr int n = 2;
      MPI_Datatype dt;
      LocalDatabaseElement measure;
      const std::vector<int> lengths(n, 1);
      const MPI_Datatype types[n] = {vector(sizeof(enum Name), MPI_CHAR),
                                     sliceInfo()};

      // measure the displacements in the struct
      size_t j = 0;
      MPI_Aint base_address, displacements[n];
      MPI_Get_address(&measure, &base_address);
      MPI_Get_address(&measure.name, &displacements[j++]);
      MPI_Get_address(&measure.info, &displacements[j++]);
      for (size_t i = 0; i < n; i++)
        displacements[i] = MPI_Aint_diff(displacements[i], base_address);

      static_assert(sizeof(LocalDatabaseElement) == sizeof(measure),
                    "Measure has bad size");

      MPI_Type_create_struct(n, lengths.data(), displacements, types, &dt);
      MPI_Type_commit(&dt);
      return vector(sizeof(LocalDatabaseElement), MPI_CHAR);
      // TODO: write tests in order to know if this works
      return dt;
    }
  };
  // MPI Types:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Static%20utilities][Static utilities:1]]
  static PartialTuple subtupleBySlice(ABCTuple abc, Type sliceType) {
    switch (sliceType) {
    case AB: return {abc[0], abc[1]};
    case BC: return {abc[1], abc[2]};
    case AC: return {abc[0], abc[2]};
    case CB: return {abc[2], abc[1]};
    case BA: return {abc[1], abc[0]};
    case CA: return {abc[2], abc[0]};
    case A: return {abc[0], 0};
    case B: return {abc[1], 0};
    case C: return {abc[2], 0};
    default: throw "Switch statement not exhaustive!";
    }
  }
  // Static utilities:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Static%20utilities][Static utilities:2]]
  static std::vector<Slice<F> *>
  hasRecycledReferencingToIt(std::vector<Slice<F>> &slices, Info const &info) {
    std::vector<Slice<F> *> result;

    for (auto &s : slices)
      if (s.info.recycling == info.type && s.info.tuple == info.tuple
          && s.info.state == Recycled)
        result.push_back(&s);

    return result;
  }
  // Static utilities:2 ends here

  // [[file:~/cuda/atrip/atrip.org::*Static%20utilities][Static utilities:3]]
  static Slice<F> &findOneByType(std::vector<Slice<F>> &slices,
                                 Slice<F>::Type type) {
    const auto sliceIt =
        std::find_if(slices.begin(), slices.end(), [&type](Slice<F> const &s) {
          return type == s.info.type;
        });
    WITH_CRAZY_DEBUG
    WITH_RANK << "\t__ looking for " << type << "\n";
    if (sliceIt == slices.end())
      throw std::domain_error("Slice one by type not found!");
    return *sliceIt;
  }
  // Static utilities:3 ends here

  // [[file:~/cuda/atrip/atrip.org::*Static%20utilities][Static utilities:4]]
  static Slice<F> &findRecycledSource(std::vector<Slice<F>> &slices,
                                      Slice<F>::Info info) {
    const auto sliceIt =
        std::find_if(slices.begin(), slices.end(), [&info](Slice<F> const &s) {
          return info.recycling == s.info.type && info.tuple == s.info.tuple
              && State::Recycled != s.info.state;
        });

    WITH_CRAZY_DEBUG
    WITH_RANK << "__slice__:find: recycling source of " << pretty_print(info)
              << "\n";
    if (sliceIt == slices.end())
      throw std::domain_error("Recycled source not found: " + pretty_print(info)
                              + " rank: " + pretty_print(Atrip::rank));
    WITH_RANK << "__slice__:find: " << pretty_print(sliceIt->info) << "\n";
    return *sliceIt;
  }
  // Static utilities:4 ends here

  // [[file:~/cuda/atrip/atrip.org::*Static%20utilities][Static utilities:5]]
  static Slice<F> &findByTypeAbc(std::vector<Slice<F>> &slices,
                                 Slice<F>::Type type,
                                 ABCTuple const &abc) {
    const auto tuple = Slice<F>::subtupleBySlice(abc, type);
    const auto sliceIt =
        std::find_if(slices.begin(),
                     slices.end(),
                     [&type, &tuple](Slice<F> const &s) {
                       return type == s.info.type && tuple == s.info.tuple;
                     });
    WITH_CRAZY_DEBUG
    WITH_RANK << "__slice__:find:" << type << " and tuple "
              << pretty_print(tuple) << "\n";
    if (sliceIt == slices.end())
      throw std::domain_error("Slice by type not found: " + pretty_print(tuple)
                              + ", " + std::to_string(type)
                              + " rank: " + std::to_string(Atrip::rank));
    return *sliceIt;
  }
  // Static utilities:5 ends here

  // [[file:~/cuda/atrip/atrip.org::*Static%20utilities][Static utilities:6]]
  static Slice<F> &findByInfo(std::vector<Slice<F>> &slices,
                              Slice<F>::Info const &info) {
    const auto sliceIt =
        std::find_if(slices.begin(), slices.end(), [&info](Slice<F> const &s) {
          // TODO: maybe implement comparison in Info struct
          return info.type == s.info.type && info.state == s.info.state
              && info.tuple == s.info.tuple
              && info.from.rank == s.info.from.rank
              && info.from.source == s.info.from.source;
        });
    WITH_CRAZY_DEBUG
    WITH_RANK << "__slice__:find:looking for " << pretty_print(info) << "\n";
    if (sliceIt == slices.end())
      throw std::domain_error("Slice by info not found: " + pretty_print(info));
    return *sliceIt;
  }
  // Static utilities:6 ends here

  // [[file:~/cuda/atrip/atrip.org::*Attributes][Attributes:1]]
  Info info;
  // Attributes:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Attributes][Attributes:2]]
  DataPtr<F> data;
#if defined(HAVE_CUDA) && !defined(ATRIP_SOURCES_IN_GPU)
  F *mpi_data;
#endif
  // Attributes:2 ends here

  // [[file:~/cuda/atrip/atrip.org::*Attributes][Attributes:3]]
  MPI_Request request;
  // Attributes:3 ends here

  // [[file:~/cuda/atrip/atrip.org::*Attributes][Attributes:4]]
  const size_t size;
  // Attributes:4 ends here

  // [[file:~/cuda/atrip/atrip.org::*Member%20functions][Member functions:1]]
  void markReady() noexcept {
    info.state = Ready;
    info.recycling = Blank;
  }
  // Member functions:1 ends here

  // [[file:~/cuda/atrip/atrip.org::*Member%20functions][Member functions:2]]
  bool isUnwrapped() const noexcept {
    return info.state == Ready || info.state == SelfSufficient;
  }
  // Member functions:2 ends here

  // [[file:~/cuda/atrip/atrip.org::*Member%20functions][Member functions:3]]
  bool isUnwrappable() const noexcept {
    return isUnwrapped() || info.state == Recycled || info.state == Dispatched;
  }

  inline bool isDirectlyFetchable() const noexcept {
    return info.state == Ready || info.state == Dispatched;
  }

  void free() noexcept {
    info.tuple = {0, 0};
    info.type = Blank;
    info.state = Acceptor;
    info.from = {0, 0};
    info.recycling = Blank;
    data = DataNullPtr;
  }

  inline bool isFree() const noexcept {
    return info.tuple == PartialTuple{0, 0} && info.type == Blank
        && info.state == Acceptor && info.from.rank == 0
        && info.from.source == 0 && info.recycling == Blank
        && data == DataNullPtr;
  }
  // Member functions:3 ends here

  // [[file:~/cuda/atrip/atrip.org::*Member%20functions][Member functions:4]]
  inline bool isRecyclable() const noexcept {
    return (info.state == Dispatched || info.state == Ready
            || info.state == Fetch)
        && hasValidDataPointer();
  }
  // Member functions:4 ends here

  // [[file:~/cuda/atrip/atrip.org::*Member%20functions][Member functions:5]]
  inline bool hasValidDataPointer() const noexcept {
    return data != DataNullPtr && info.state != Acceptor && info.type != Blank;
  }
  // Member functions:5 ends here

  // [[file:~/cuda/atrip/atrip.org::*Member%20functions][Member functions:6]]
  void unwrapAndMarkReady() {
    if (info.state == Ready) return;
    if (info.state != Dispatched)
      throw std::domain_error(
          "Can't unwrap a non-ready, non-dispatched slice!");
    markReady();
    MPI_Status status;
#ifdef HAVE_OCD
    WITH_RANK << "__slice__:mpi: waiting "
              << "\n";
#endif
    const int errorCode = MPI_Wait(&request, &status);

    // FIXME: it appears not to work to free
    // this request, investigate if this is necessary or not
    // const auto _mpi_request_free = MPI_Request_free(&request);

    // if (MPI_SUCCESS != _mpi_request_free)
    // throw "Atrip: Error freeing MPI request";

    if (errorCode != MPI_SUCCESS) throw "Atrip: Unexpected error MPI ERROR";

#if defined(HAVE_CUDA) && !defined(ATRIP_SOURCES_IN_GPU)
    // copy the retrieved mpi data to the device
    WITH_CHRONO("cuda:memcpy",
                _CHECK_CUDA_SUCCESS(
                    "copying mpi data to device",
                    cuMemcpyHtoD(data, (void *)mpi_data, sizeof(F) * size));)
    std::free(mpi_data);
#endif

#ifdef HAVE_OCD
    char errorString[MPI_MAX_ERROR_STRING];
    int errorSize;
    MPI_Error_string(errorCode, errorString, &errorSize);

    WITH_RANK << "__slice__:mpi: status "
              << "{ .source=" << status.MPI_SOURCE
              << ", .tag=" << status.MPI_TAG << ", .error=" << status.MPI_ERROR
              << ", .errCode=" << errorCode << ", .err=" << errorString << " }"
              << "\n";
#endif
  }
  // Member functions:6 ends here

  // [[file:~/cuda/atrip/atrip.org::*Epilog][Epilog:1]]
  Slice(size_t size_)
      : info({})
      , data(DataNullPtr)
#if defined(HAVE_CUDA) && !defined(ATRIP_SOURCES_IN_GPU)
      , mpi_data(nullptr)
#endif
      , size(size_) {
  }

}; // struct Slice
// Epilog:1 ends here

// [[file:~/cuda/atrip/atrip.org::*Debug][Debug:1]]
template <typename F = double>
std::ostream &operator<<(std::ostream &out,
                         typename Slice<F>::Location const &v) {
  // TODO: remove me
  out << "{.r(" << v.rank << "), .s(" << v.source << ")};";
  return out;
}

template <typename F = double>
std::ostream &operator<<(std::ostream &out, typename Slice<F>::Info const &i) {
  out << "«" << i.type << ", " << i.state << "»"
      << " ⊙ {" << i.from.rank << ", " << i.from.source << "}"
      << " ∴ {" << i.tuple[0] << ", " << i.tuple[1] << "}"
      << " ♲" << i.recycling;
  return out;
}

template <typename F>
std::string type_to_string(typename Slice<F>::Type t) {
  switch (t) {
  case Slice<F>::AB: return "AB";
  case Slice<F>::BC: return "BC";
  case Slice<F>::AC: return "AC";
  case Slice<F>::CB: return "CB";
  case Slice<F>::BA: return "BA";
  case Slice<F>::CA: return "CA";
  case Slice<F>::A: return "A";
  case Slice<F>::B: return "B";
  case Slice<F>::C: return "C";
  case Slice<F>::Blank: return "Blank";
  default: throw "Switch statement not exhaustive!";
  }
}

template <typename F>
std::string name_to_string(typename Slice<F>::Name t) {
  switch (t) {
  case Slice<F>::TA: return "TA";
  case Slice<F>::VIJKA: return "VIJKA";
  case Slice<F>::VABCI: return "VABCI";
  case Slice<F>::TABIJ: return "TABIJ";
  case Slice<F>::VABIJ: return "VABIJ";
  default: throw "Switch statement not exhaustive!";
  }
}

template <typename F>
size_t name_to_size(typename Slice<F>::Name t, size_t No, size_t Nv) {
  switch (t) {
  case Slice<F>::TA: return Nv * No * No;
  case Slice<F>::VIJKA: return No * No * No;
  case Slice<F>::VABCI: return Nv * No;
  case Slice<F>::TABIJ: return No * No;
  case Slice<F>::VABIJ: return No * No;
  default: throw "Switch statement not exhaustive!";
  }
}

template <typename F>
std::string state_to_string(typename Slice<F>::State t) {
  switch (t) {
  case Slice<F>::Fetch: return "Fetch";
  case Slice<F>::Dispatched: return "Dispatched";
  case Slice<F>::Ready: return "Ready";
  case Slice<F>::SelfSufficient: return "SelfSufficient";
  case Slice<F>::Recycled: return "Recycled";
  case Slice<F>::Acceptor: return "Acceptor";
  default: throw "Switch statement not exhaustive!";
  }
}

template <typename F = double>
std::ostream &operator<<(std::ostream &out, typename Slice<F>::State const &i) {
  out << state_to_string<F>(i);
  return out;
}

template <typename F = double>
std::ostream &operator<<(std::ostream &out, typename Slice<F>::Name const &i) {
  out << name_to_string<F>(i);
  return out;
}

template <typename F = double>
std::ostream &operator<<(std::ostream &out, typename Slice<F>::Type const &i) {
  out << type_to_string<F>(i);
  return out;
}

} // namespace atrip
// Debug:1 ends here
