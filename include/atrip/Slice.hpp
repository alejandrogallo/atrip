// [[file:../../atrip.org::*Prolog][Prolog:1]]
#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <mpi.h>

#include <atrip/Tuples.hpp>
#include <atrip/Utils.hpp>

namespace atrip {


struct Slice {

  using F = double;
// Prolog:1 ends here

// [[file:../../atrip.org::*Location][Location:1]]
struct Location { size_t rank; size_t source; };
// Location:1 ends here

// [[file:../../atrip.org::*Type][Type:1]]
enum Type
  { A = 10
  , B
  , C
  // Two-parameter slices
  , AB = 20
  , BC
  , AC
  // for abci and the doubles
  , CB
  , BA
  , CA
  // The non-typed slice
  , Blank = 404
  };
// Type:1 ends here

// [[file:../../atrip.org::*State][State:1]]
enum State {
  Fetch = 0,
  Dispatched = 2,
  Ready = 1,
  SelfSufficient = 911,
  Recycled = 123,
  Acceptor = 405
};
// State:1 ends here

// [[file:../../atrip.org::*The Info structure][The Info structure:1]]
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

  Info() : tuple{0,0}
          , type{Blank}
          , state{Acceptor}
          , from{0,0}
          , recycling{Blank}
          {}
};

using Ty_x_Tu = std::pair< Type, PartialTuple >;
// The Info structure:1 ends here

// [[file:../../atrip.org::*Name][Name:1]]
enum Name
  { TA    = 100
  , VIJKA = 101
  , VABCI = 200
  , TABIJ = 201
  , VABIJ = 202
  };
// Name:1 ends here

// [[file:../../atrip.org::*Database][Database:1]]
struct LocalDatabaseElement {
  Slice::Name name;
  Slice::Info info;
};
// Database:1 ends here

// [[file:../../atrip.org::*Database][Database:2]]
using LocalDatabase = std::vector<LocalDatabaseElement>;
using Database = LocalDatabase;
// Database:2 ends here

// [[file:../../atrip.org::*MPI Types][MPI Types:1]]
struct mpi {

  static MPI_Datatype vector(size_t n, MPI_Datatype const& DT) {
    MPI_Datatype dt;
    MPI_Type_vector(n, 1, 1, DT, &dt);
    MPI_Type_commit(&dt);
    return dt;
  }

  static MPI_Datatype sliceLocation () {
    constexpr int n = 2;
    // create a sliceLocation to measure in the current architecture
    // the packing of the struct
    Slice::Location measure;
    MPI_Datatype dt;
    const std::vector<int> lengths(n, 1);
    const MPI_Datatype types[n] = {usizeDt(), usizeDt()};

    // measure the displacements in the struct
    size_t j = 0;
    MPI_Aint displacements[n];
    MPI_Get_address(&measure.rank,   &displacements[j++]);
    MPI_Get_address(&measure.source, &displacements[j++]);
    for (size_t i = 1; i < n; i++) displacements[i] -= displacements[0];
    displacements[0] = 0;

    MPI_Type_create_struct(n, lengths.data(), displacements, types, &dt);
    MPI_Type_commit(&dt);
    return dt;
  }

  static MPI_Datatype enumDt() { return MPI_INT; }
  static MPI_Datatype usizeDt() { return MPI_UINT64_T; }

  static MPI_Datatype sliceInfo () {
    constexpr int n = 5;
    MPI_Datatype dt;
    Slice::Info measure;
    const std::vector<int> lengths(n, 1);
    const MPI_Datatype types[n]
      = { vector(2, usizeDt())
        , enumDt()
        , enumDt()
        , sliceLocation()
        , enumDt()
        };

    // create the displacements from the info measurement struct
    size_t j = 0;
    MPI_Aint displacements[n];
    MPI_Get_address(measure.tuple.data(), &displacements[j++]);
    MPI_Get_address(&measure.type,        &displacements[j++]);
    MPI_Get_address(&measure.state,       &displacements[j++]);
    MPI_Get_address(&measure.from,        &displacements[j++]);
    MPI_Get_address(&measure.recycling,   &displacements[j++]);
    for (size_t i = 1; i < n; i++) displacements[i] -= displacements[0];
    displacements[0] = 0;

    MPI_Type_create_struct(n, lengths.data(), displacements, types, &dt);
    MPI_Type_commit(&dt);
    return dt;
  }

  static MPI_Datatype localDatabaseElement () {
    constexpr int n = 2;
    MPI_Datatype dt;
    LocalDatabaseElement measure;
    const std::vector<int> lengths(n, 1);
    const MPI_Datatype types[n]
      = { enumDt()
        , sliceInfo()
        };

    // measure the displacements in the struct
    size_t j = 0;
    MPI_Aint displacements[n];
    MPI_Get_address(&measure.name, &displacements[j++]);
    MPI_Get_address(&measure.info, &displacements[j++]);
    for (size_t i = 1; i < n; i++) displacements[i] -= displacements[0];
    displacements[0] = 0;

    MPI_Type_create_struct(n, lengths.data(), displacements, types, &dt);
    MPI_Type_commit(&dt);
    return dt;
  }

};
// MPI Types:1 ends here

// [[file:../../atrip.org::*Static utilities][Static utilities:1]]
static
PartialTuple subtupleBySlice(ABCTuple abc, Type sliceType) {
  switch (sliceType) {
    case AB: return {abc[0], abc[1]};
    case BC: return {abc[1], abc[2]};
    case AC: return {abc[0], abc[2]};
    case CB: return {abc[2], abc[1]};
    case BA: return {abc[1], abc[0]};
    case CA: return {abc[2], abc[0]};
    case  A: return {abc[0], 0};
    case  B: return {abc[1], 0};
    case  C: return {abc[2], 0};
    default: throw "Switch statement not exhaustive!";
  }
}
// Static utilities:1 ends here

// [[file:../../atrip.org::*Static utilities][Static utilities:2]]
static std::vector<Slice*> hasRecycledReferencingToIt
  ( std::vector<Slice> &slices
  , Info const& info
  ) {
  std::vector<Slice*> result;

  for (auto& s: slices)
    if (  s.info.recycling == info.type
        && s.info.tuple == info.tuple
        && s.info.state == Recycled
        ) result.push_back(&s);

  return result;
}
// Static utilities:2 ends here

// [[file:../../atrip.org::*Static utilities][Static utilities:3]]
static Slice& findOneByType(std::vector<Slice> &slices, Slice::Type type) {
    const auto sliceIt
      = std::find_if(slices.begin(), slices.end(),
                      [&type](Slice const& s) {
                        return type == s.info.type;
                      });
    WITH_CRAZY_DEBUG
    WITH_RANK
      << "__slice__:find:looking for " << type << "\n";
    if (sliceIt == slices.end())
      throw std::domain_error("Slice by type not found!");
    return *sliceIt;
}
// Static utilities:3 ends here

// [[file:../../atrip.org::*Static utilities][Static utilities:4]]
static Slice&
findRecycledSource (std::vector<Slice> &slices, Slice::Info info) {
  const auto sliceIt
    = std::find_if(slices.begin(), slices.end(),
                    [&info](Slice const& s) {
                      return info.recycling == s.info.type
                          && info.tuple == s.info.tuple
                          && State::Recycled != s.info.state
                          ;
                    });

  WITH_CRAZY_DEBUG
  WITH_RANK << "__slice__:find: recycling source of "
            << pretty_print(info) << "\n";
  if (sliceIt == slices.end())
    throw std::domain_error( "Slice not found: "
                            + pretty_print(info)
                            + " rank: "
                            + pretty_print(Atrip::rank)
                            );
  WITH_RANK << "__slice__:find: " << pretty_print(sliceIt->info) << "\n";
  return *sliceIt;
}
// Static utilities:4 ends here

// [[file:../../atrip.org::*Static utilities][Static utilities:5]]
static Slice& findByTypeAbc
  ( std::vector<Slice> &slices
  , Slice::Type type
  , ABCTuple const& abc
  ) {
    const auto tuple = Slice::subtupleBySlice(abc, type);
    const auto sliceIt
      = std::find_if(slices.begin(), slices.end(),
                      [&type, &tuple](Slice const& s) {
                        return type == s.info.type
                            && tuple == s.info.tuple
                            ;
                      });
    WITH_CRAZY_DEBUG
    WITH_RANK << "__slice__:find:" << type << " and tuple "
              << pretty_print(tuple)
              << "\n";
    if (sliceIt == slices.end())
      throw std::domain_error( "Slice not found: "
                              + pretty_print(tuple)
                              + ", "
                              + pretty_print(type)
                              + " rank: "
                              + pretty_print(Atrip::rank)
                              );
    return *sliceIt;
}
// Static utilities:5 ends here

// [[file:../../atrip.org::*Static utilities][Static utilities:6]]
static Slice& findByInfo(std::vector<Slice> &slices,
                         Slice::Info const& info) {
  const auto sliceIt
    = std::find_if(slices.begin(), slices.end(),
                   [&info](Slice const& s) {
                     // TODO: maybe implement comparison in Info struct
                     return info.type == s.info.type
                       && info.state == s.info.state
                       && info.tuple == s.info.tuple
                       && info.from.rank == s.info.from.rank
                       && info.from.source == s.info.from.source
                       ;
                   });
  WITH_CRAZY_DEBUG
    WITH_RANK << "__slice__:find:looking for " << pretty_print(info) << "\n";
  if (sliceIt == slices.end())
    throw std::domain_error( "Slice by info not found: "
                             + pretty_print(info));
  return *sliceIt;
}
// Static utilities:6 ends here

// [[file:../../atrip.org::*Attributes][Attributes:1]]
Info info;
// Attributes:1 ends here

// [[file:../../atrip.org::*Attributes][Attributes:2]]
F  *data;
// Attributes:2 ends here

// [[file:../../atrip.org::*Attributes][Attributes:3]]
MPI_Request request;
// Attributes:3 ends here

// [[file:../../atrip.org::*Attributes][Attributes:4]]
const size_t size;
// Attributes:4 ends here

// [[file:../../atrip.org::*Member functions][Member functions:1]]
void markReady() noexcept {
  info.state = Ready;
  info.recycling = Blank;
}
// Member functions:1 ends here

// [[file:../../atrip.org::*Member functions][Member functions:2]]
bool isUnwrapped() const noexcept {
  return info.state == Ready
      || info.state == SelfSufficient
      ;
}
// Member functions:2 ends here

// [[file:../../atrip.org::*Member functions][Member functions:3]]
bool isUnwrappable() const noexcept {
  return isUnwrapped()
      || info.state == Recycled
      || info.state == Dispatched
      ;
}

inline bool isDirectlyFetchable() const noexcept {
  return info.state == Ready || info.state == Dispatched;
}

void free() noexcept {
  info.tuple      = {0, 0};
  info.type       = Blank;
  info.state      = Acceptor;
  info.from       = {0, 0};
  info.recycling  = Blank;
  data            = nullptr;
}

inline bool isFree() const noexcept {
  return info.tuple       == PartialTuple{0, 0}
      && info.type        == Blank
      && info.state       == Acceptor
      && info.from.rank   == 0
      && info.from.source == 0
      && info.recycling   == Blank
      && data             == nullptr
       ;
}
// Member functions:3 ends here

// [[file:../../atrip.org::*Member functions][Member functions:4]]
inline bool isRecyclable() const noexcept {
  return (  info.state == Dispatched
         || info.state == Ready
         || info.state == Fetch
         )
      && hasValidDataPointer()
      ;
}
// Member functions:4 ends here

// [[file:../../atrip.org::*Member functions][Member functions:5]]
inline bool hasValidDataPointer() const noexcept {
  return data       != nullptr
      && info.state != Acceptor
      && info.type  != Blank
      ;
}
// Member functions:5 ends here

// [[file:../../atrip.org::*Member functions][Member functions:6]]
void unwrapAndMarkReady() {
      if (info.state == Ready) return;
      if (info.state != Dispatched)
        throw
          std::domain_error("Can't unwrap a non-ready, non-dispatched slice!");
      markReady();
      MPI_Status status;
#ifdef HAVE_OCD
        WITH_RANK << "__slice__:mpi: waiting " << "\n";
#endif
      const int errorCode = MPI_Wait(&request, &status);
      if (errorCode != MPI_SUCCESS)
        throw "MPI ERROR HAPPENED....";

#ifdef HAVE_OCD
      char errorString[MPI_MAX_ERROR_STRING];
      int errorSize;
      MPI_Error_string(errorCode, errorString, &errorSize);

      WITH_RANK << "__slice__:mpi: status "
                << "{ .source="    << status.MPI_SOURCE
                << ", .tag="       << status.MPI_TAG
                << ", .error="     << status.MPI_ERROR
                << ", .errCode="   << errorCode
                << ", .err="       << errorString
                << " }"
                << "\n";
#endif
    }
// Member functions:6 ends here

// [[file:../../atrip.org::*Epilog][Epilog:1]]
Slice(size_t size_)
    : info({})
    , data(nullptr)
    , size(size_)
    {}


}; // struct Slice
// Epilog:1 ends here

// [[file:../../atrip.org::*Debug][Debug:1]]
std::ostream& operator<<(std::ostream& out, Slice::Location const& v) {
  // TODO: remove me
  out << "{.r(" << v.rank << "), .s(" << v.source << ")};";
  return out;
}

std::ostream& operator<<(std::ostream& out, Slice::Info const& i) {
  out << "«t" << i.type << ", s" << i.state << "»"
      << " ⊙ {" << i.from.rank << ", " << i.from.source << "}"
      << " ∴ {" << i.tuple[0] << ", " << i.tuple[1] << "}"
      << " ♲t" << i.recycling
      ;
  return out;
}

} // namespace atrip
// Debug:1 ends here
