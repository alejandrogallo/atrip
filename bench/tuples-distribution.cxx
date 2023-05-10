#ifdef ATRIP_DEBUG
#  undef ATRIP_DEBUG
#  define ATRIP_DEBUG 2
#endif
#define ATRIP_DONT_SLICE
#define ATRIP_DRY
#define ATRIP_ALLOCATE_ADDITIONAL_FREE_POINTERS

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "config.h"

#include <atrip/Atrip.hpp>
#include <atrip/Utils.hpp>
#include <atrip/Tuples.hpp>
#include <atrip/Unions.hpp>
#include <atrip/SliceUnion.hpp>

#include <bench/CLI11.hpp>
#include <bench/utils.hpp>

using namespace atrip;

using F = double;
using Tr = CTF::Tensor<F>;

#define INIT_DRY(name, ...)                                                    \
  do {                                                                         \
    std::vector<int64_t> lens = __VA_ARGS__;                                   \
    size_t i = 0UL;                                                            \
    name.wrld = &world;                                                        \
    name.order = lens.size();                                                  \
    name.lens = (int64_t *)malloc(sizeof(int64_t) * lens.size());              \
    name.sym = (int *)malloc(sizeof(int) * lens.size());                       \
    name.lens[i] = lens[i];                                                    \
    name.lens[i] = lens[i];                                                    \
    i++;                                                                       \
    name.lens[i] = lens[i];                                                    \
    i++;                                                                       \
    name.lens[i] = lens[i];                                                    \
    i++;                                                                       \
    i = 0;                                                                     \
    name.sym[i++] = NS;                                                        \
    name.sym[i++] = NS;                                                        \
    name.sym[i++] = NS;                                                        \
    name.sym[i++] = NS;                                                        \
  } while (0)

#define DEINIT_DRY(name)                                                       \
  do {                                                                         \
    name.order = 0;                                                            \
    name.lens = NULL;                                                          \
    name.sym = NULL;                                                           \
  } while (0)

using LocalDatabase = typename Slice<F>::LocalDatabase;
using LocalDatabaseElement = typename Slice<F>::LocalDatabaseElement;

ClusterInfo make_fake_cluster_info(size_t n_nodes, size_t ranks_per_node) {
  std::vector<RankInfo> ranks;
  std::vector<std::string> names(n_nodes);
  for (size_t i = 0; i < n_nodes; i++) {
    names[i] = "node-" + std::to_string(i);
    for (size_t r = 0; r < ranks_per_node; r++) {
      ranks.push_back({/* .name = */ names[i],
                       /* .nodeId = */ i,
                       /* .globalRank = */ i * ranks_per_node + r,
                       /* .localRank = */ r,
                       /* .ranksPerNode = */ ranks_per_node});
    }
  }
  return ClusterInfo{n_nodes, n_nodes * ranks_per_node, ranks_per_node, ranks};
}

template <typename F>
void clearUnusedSlicesForNext(SliceUnion<F> &u, ABCTuple const &abc) {
  auto const needed = u.neededSlices(abc);

  // CLEAN UP SLICES, FREE THE ONES THAT ARE NOT NEEDED ANYMORE
  for (auto &slice : u.slices) {
    // if the slice is free, then it was not used anyways
    if (slice.isFree()) continue;

    // try to find the slice in the needed slices list
    auto const found =
        std::find_if(needed.begin(),
                     needed.end(),
                     [&slice](typename Slice<F>::Ty_x_Tu const &tytu) {
                       return slice.info.tuple == tytu.second
                           && slice.info.type == tytu.first;
                     });

    // if we did not find slice in needed, then erase it
    if (found == needed.end()) {

      // We have to be careful about the data pointer,
      // for SelfSufficient, the data pointer is a source pointer
      // of the slice, so we should just wipe it.
      //
      // For Ready slices, we have to be careful if there are some
      // recycled slices depending on it.
      bool freeSlicePointer = true;

      // allow to gc unwrapped and recycled, never Fetch,
      // if we have a Fetch slice then something has gone very wrong.
      if (!slice.isUnwrapped() && slice.info.state != Slice<F>::Recycled)
        throw std::domain_error(
            "Trying to garbage collect "
            " a non-unwrapped slice! "
            + pretty_print(&slice) + pretty_print(slice.info));

      if (slice.info.state == Slice<F>::Ready) {
        auto recycled =
            Slice<F>::hasRecycledReferencingToIt(u.slices, slice.info);
        if (recycled.size()) {
          Slice<F> *newReady = recycled[0];
          newReady->markReady();
          assert(newReady->data == slice.data);
          freeSlicePointer = false;
          for (size_t i = 1; i < recycled.size(); i++) {
            auto newRecyled = recycled[i];
            newRecyled->info.recycling = newReady->info.type;
          }
        }
      }

      // if the slice is self sufficient, do not dare touching the
      // pointer, since it is a pointer to our sources in our rank.
      if (slice.info.state == Slice<F>::SelfSufficient
          || slice.info.state == Slice<F>::Recycled) {
        freeSlicePointer = false;
      }

      // make sure we get its data pointer to be used later
      // only for non-recycled, since it can be that we need
      // for next iteration the data of the slice that the recycled points
      // to
      if (freeSlicePointer) {
        u.freePointers.insert(slice.data);
      } else {
        // WITH_OCD WITH_RANK << "__gc__:not touching the free Pointer\n";
      }

      slice.free();
    } // we did not find the slice
  }
}

template <typename F>
void send(SliceUnion<F> &u,
          size_t otherRank,
          typename Slice<F>::LocalDatabaseElement const &el,
          size_t tag,
          ABCTuple abc) {
  IGNORABLE(u);
  IGNORABLE(tag);
  IGNORABLE(abc);
  bool sendData_p = false;
  auto const &info = el.info;

  if (info.state == Slice<F>::Fetch) sendData_p = true;
  if (otherRank == info.from.rank) sendData_p = false;
  if (!sendData_p) return;

#if defined(ATRIP_MPI_STAGING_BUFFERS)
  DataPtr<F> isend_buffer = u.popFreePointer();
  u.mpi_staging_buffers.insert(
      typename SliceUnion<F>::StagingBufferInfo{isend_buffer,
                                                tag,
                                                nullptr,
                                                abc});
#endif /* defined(ATRIP_MPI_STAGING_BUFFERS) */
}

LocalDatabase build_local_database(SliceUnion<F> &u, ABCTuple const &abc) {
  LocalDatabase result;

  auto const needed = u.neededSlices(abc);

  // BUILD THE DATABASE
  // we need to loop over all sliceTypes that this TensorUnion
  // is representing and find out how we will get the corresponding
  // slice for the abc we are considering right now.
  for (auto const &pair : needed) {
    auto const type = pair.first;
    auto const tuple = pair.second;
    auto const from = u.rankMap.find(abc, type);

    {
      // FIRST: look up if there is already a *Ready* slice matching what we
      // need
      auto const &it = std::find_if(u.slices.begin(),
                                    u.slices.end(),
                                    [&tuple, &type](Slice<F> const &other) {
                                      return other.info.tuple == tuple
                                          && other.info.type == type
                                          // we only want another slice when it
                                          // has already ready-to-use data
                                          && other.isUnwrappable();
                                    });
      if (it != u.slices.end()) {
        // if we find this slice, it means that we don't have to do anything
        result.push_back({u.name, it->info});
        continue;
      }
    }

    //
    // Try to find a recyling possibility ie. find a slice with the same
    // tuple and that has a valid data pointer.
    //
    auto const &recycleIt =
        std::find_if(u.slices.begin(),
                     u.slices.end(),
                     [&tuple, &type](Slice<F> const &other) {
                       return other.info.tuple == tuple
                           && other.info.type != type && other.isRecyclable();
                     });

    //
    // if we find this recylce, then we find a Blank slice
    // (which should exist by construction :THINK)
    //
    if (recycleIt != u.slices.end()) {
      auto &blank = Slice<F>::findOneByType(u.slices, Slice<F>::Blank);
      // TODO: formalize this through a method to copy information
      //       from another slice
      blank.data = recycleIt->data;
      blank.info.type = type;
      blank.info.tuple = tuple;
      blank.info.state = Slice<F>::Recycled;
      blank.info.from = from;
      blank.info.recycling = recycleIt->info.type;
      result.push_back({u.name, blank.info});
      WITH_RANK << "__db__: RECYCLING: n" << u.name << " " << pretty_print(abc)
                << " get " << pretty_print(blank.info) << " from "
                << pretty_print(recycleIt->info) << " ptr " << recycleIt->data
                << "\n";
      continue;
    }

    // in this case we have to create a new slice
    // this means that we should have a blank slice at our disposal
    // and also the freePointers should have some elements inside,
    // so we pop a data pointer from the freePointers container
    {
      auto &blank = Slice<F>::findOneByType(u.slices, Slice<F>::Blank);
      blank.info.type = type;
      blank.info.tuple = tuple;
      blank.info.from = from;

      // Handle self sufficiency
      blank.info.state =
          Atrip::rank == from.rank ? Slice<F>::SelfSufficient : Slice<F>::Fetch;
      if (blank.info.state == Slice<F>::SelfSufficient) {
        blank.data = u.sources[from.source].data();
      } else {
        blank.data = u.popFreePointer();
      }

      result.push_back({u.name, blank.info});
      continue;
    }
  }

  return result;
}

void unwrapSlice(Slice<F>::Type t, ABCTuple abc, SliceUnion<F> *u) {
  auto &slice = Slice<F>::findByTypeAbc(u->slices, t, abc);
  switch (slice.info.state) {
  case Slice<F>::Dispatched: slice.markReady(); break;
  case Slice<F>::Recycled:
    // unwrapSlice(t, abc, u);
    unwrapSlice(slice.info.recycling, abc, u);
    break;
  case Slice<F>::Fetch:
    throw std::domain_error("Can't unwrap a fetch slice!");
    break;
  case Slice<F>::Acceptor:
    throw std::domain_error("Can't unwrap an acceptor");
    break;
  case Slice<F>::Ready:
  case Slice<F>::SelfSufficient: break;
  }
}

#define PRINT_VARIABLE(v)                                                      \
  do {                                                                         \
    if (!rank) std::cout << "# " << #v << ": " << v << std::endl;              \
  } while (0)

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int no(10), nv(100), mod(1000), out_rank(0), n_nodes, ranks_per_node;
  int64_t max_iterations(-1);
  std::string tuplesDistributionString = "group";
  bool print_database, print_tuples;

  CLI::App app{"Main bench for atrip"};
  defoption(app, "-N", ranks_per_node, "Ranks per node")->required();
  defoption(app, "--nodes", n_nodes, "Number of nodes")->required();
  defoption(app, "--no", no, "Occupied orbitals")->required();
  defoption(app, "--nv", nv, "Virtual orbitals")->required();
  defoption(app, "--dist", tuplesDistributionString, "Which distribution")
      ->required();
  defoption(app, "--mod", mod, "Every each iteration there is output");
  defoption(app, "--out-rank", out_rank, "Stdout for given rank");
  defoption(app, "--max-iterations", max_iterations, "Iterations to run");
  defflag(app,
          "--db",
          print_database,
          "Wether to print the database for every "
          "iteration for rank --out-rank");
  defflag(app,
          "--print-tuples",
          print_tuples,
          "Print the tuples list for --out-rank");
  CLI11_PARSE(app, argc, argv);

  CTF::World world(argc, argv);
  auto kaun = world.comm;
  int rank, np;
  MPI_Comm_rank(kaun, &rank);
  MPI_Comm_size(kaun, &np);
  Atrip::init(world.comm);
  Atrip::cluster_info =
      new ClusterInfo(make_fake_cluster_info(n_nodes, ranks_per_node));

  if (!rank)
    if (np != n_nodes * ranks_per_node) {
      std::cout << "You have to run the application with the correct number of "
                   "mpi ranks"
                << "\nfor instance mpirun -np 100 ./tuples-distribution -N 5 "
                   "--nodes 10"
                << std::endl;
      std::exit(1);
    }

  std::cout << "\n";
  if (!rank)
    for (auto const &fn : input_printer)
      // print input parameters
      fn();

  atrip::ABCTuples tuplesList;
  atrip::TuplesDistribution *dist;
  {
    using namespace atrip;
    if (tuplesDistributionString == "naive") {
      dist = new NaiveDistribution();
      if (!rank) std::cout << "Using NAIVE distribution" << std::endl;
      tuplesList = dist->getTuples(nv, world.comm);
    } else if (tuplesDistributionString == "group") {
      dist = new group_and_sort::Distribution();
      if (!rank) std::cout << "Using group and sort distribution" << std::endl;
      tuplesList = dist->getTuples(nv, world.comm);
    } else {
      if (!rank) std::cout << "--dist should be either naive or group\n";
      exit(1);
    }
  }

  if (print_tuples) {
    for (auto const &t : tuplesList) {
      if (rank == out_rank) {
        std::cout << _FORMAT("%ld %ld %ld", t[0], t[1], t[2]) << std::endl;
      }
    }
    MPI_Barrier(kaun);
    exit(0);
  }

  // create a fake dry tensor
  Tr t_abph, t_abhh, t_tabhh, t_taphh, t_hhha;
  INIT_DRY(t_abph, {nv, nv, nv, no});
  INIT_DRY(t_abhh, {nv, nv, no, no});
  INIT_DRY(t_tabhh, {nv, nv, no, no});
  INIT_DRY(t_taphh, {nv, nv, no, no});
  INIT_DRY(t_hhha, {no, no, no, nv});

  ABPH<F> abph(t_abph, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  ABHH<F> abhh(t_abhh, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  TABHH<F> tabhh(t_tabhh, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  TAPHH<F> taphh(t_taphh, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  HHHA<F> hhha(t_hhha, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);

  std::vector<SliceUnion<F> *> unions = {&taphh, &hhha, &abph, &abhh, &tabhh};

  using Database = typename Slice<F>::Database;
  auto communicateDatabase = [&unions, np](ABCTuple const &abc,
                                           MPI_Comm const &c) -> Database {
    auto MPI_LDB_ELEMENT = Slice<F>::mpi::localDatabaseElement();

    WITH_CHRONO(
        "db:comm:ldb", // Build local database
        typename Slice<F>::LocalDatabase ldb;
        for (auto const &tensor
             : unions) {
          auto const &tensorDb = build_local_database(*tensor, abc);
          ldb.insert(ldb.end(), tensorDb.begin(), tensorDb.end());
        })

    Database db(np * ldb.size(), ldb[0]);

    WITH_CHRONO("oneshot-db:comm:allgather",
                WITH_CHRONO("db:comm:allgather",
                            MPI_Allgather(ldb.data(),
                                          /* ldb.size() * sizeof(typename
                                             Slice<F>::LocalDatabaseElement) */
                                          ldb.size(),
                                          MPI_LDB_ELEMENT,
                                          db.data(),
                                          /* ldb.size() * sizeof(typename
                                             Slice<F>::LocalDatabaseElement), */
                                          ldb.size(),
                                          MPI_LDB_ELEMENT,
                                          c);))

    WITH_CHRONO("db:comm:type:free", MPI_Type_free(&MPI_LDB_ELEMENT);)

    return db;
  };

  auto doIOPhase = [&unions, &rank, &np, mod, out_rank, no, nv, print_database](
                       Database const &db,
                       std::vector<LocalDatabaseElement> &to_send,
                       size_t iteration,
                       ABCTuple abc) {
    const size_t localDBLength = db.size() / np;

    size_t sendTag = 0, recvTag = rank * localDBLength;

    // RECIEVE PHASE ======================================================
    {
      // At this point, we have already send to everyone that fits
      auto const &begin = &db[rank * localDBLength],
                 end = begin + localDBLength;
      for (auto it = begin; it != end; ++it) {
        recvTag++;
        auto const &el = *it;
        auto &u = unionByName(unions, el.name);
        // receive method
        if (Atrip::rank != el.info.from.rank) {
          if (el.info.state == Slice<double>::Fetch) {
            auto &slice = Slice<F>::findByInfo(u.slices, el.info);
            slice.info.state = Slice<double>::Dispatched;
          }
        }
        if (print_database)
          if (rank == out_rank) {
            std::cout << _FORMAT(
                "%4s %ld %ld %d %5ld %5s %2s %14s (%ld,%ld) %ld %ld %ld",
                "RECV",
                iteration,
                el.info.from.rank,
                rank,
                recvTag,
                name_to_string<double>(el.name).c_str(),
                type_to_string<double>(el.info.type).c_str(),
                state_to_string<double>(el.info.state).c_str(),
                el.info.tuple[0],
                el.info.tuple[1],
                name_to_size<double>(el.name, no, nv),
                u.freePointers.size(),
#if defined(ATRIP_MPI_STAGING_BUFFERS)
                u.mpi_staging_buffers.size())
#else
                0UL)
#endif
                      << "\n";
          }

      } //
    }

    // SEND PHASE =========================================================
    for (size_t otherRank = 0UL; otherRank < np; otherRank++) {
      auto const &begin = &db[otherRank * localDBLength],
                 end = begin + localDBLength;
      for (auto it = begin; it != end; ++it) {
        sendTag++;
        typename Slice<F>::LocalDatabaseElement const &el = *it;
        if (el.info.from.rank != rank) continue;
        auto &u = unionByName(unions, el.name);
        if (el.info.state == Slice<F>::Fetch) { to_send.push_back(el); }
        if (rank == out_rank) {
          if (print_database)
            std::cout << _FORMAT(
                "%4s %ld %ld %ld %5ld %5s %2s %14s (%ld,%ld) %ld %ld %ld",
                "SEND",
                iteration,
                el.info.from.rank,
                otherRank,
                sendTag,
                name_to_string<double>(el.name).c_str(),
                type_to_string<double>(el.info.type).c_str(),
                state_to_string<double>(el.info.state).c_str(),
                el.info.tuple[0],
                el.info.tuple[1],
                name_to_size<double>(el.name, no, nv),
                u.freePointers.size(),
#if defined(ATRIP_MPI_STAGING_BUFFERS)
                u.mpi_staging_buffers.size())
#else
                0UL)
#endif
                      << "\n";
        }

        send(u, otherRank, el, sendTag, abc);

      } // send phase

    } // otherRank
  };

  std::vector<LocalDatabaseElement> to_send;

  MPI_Barrier(kaun);
  for (size_t it = 0; it < tuplesList.size(); it++) {

    const ABCTuple abc = dist->tupleIsFake(tuplesList[it])
                           ? tuplesList[tuplesList.size() - 1]
                           : tuplesList[it],
                   *abcNext = it == (tuplesList.size() - 1)
                                ? nullptr
                                : &tuplesList[it + 1];

    if (it == 0) {
      const auto db = communicateDatabase(abc, kaun);
      doIOPhase(db, to_send, it, abc);
      for (auto const &u : unions) {
        for (auto type : u->sliceTypes) { unwrapSlice(type, abc, u); }
      }
    }

    if (abcNext) {
      const auto db = communicateDatabase(*abcNext, kaun);
      doIOPhase(db, to_send, it + 1, *abcNext);
    }

    for (auto &u : unions) {
      for (auto type : u->sliceTypes) { unwrapSlice(type, abc, u); }
    }

    if (abcNext) {
      for (auto &u : unions) { clearUnusedSlicesForNext(*u, *abcNext); }
    }

    if (!print_database)
      if (it % mod == 0)
        std::cout << _FORMAT("%d :it %ld  %f %% ∷ %ld ∷ %f GB\n",
                             rank,
                             it,
                             100.0 * double(to_send.size())
                                 / double(tuplesList.size()),
                             to_send.size(),
                             double(to_send.size()) * sizeof(to_send[0])
                                 / 1024.0 / 1024.0 / 1024.0);

    if (max_iterations > 0 && max_iterations < it) { break; }

#if defined(ATRIP_MPI_STAGING_BUFFERS)
    // Cleanup mpi staging buffers
    for (auto &u : unions) {
      std::vector<typename SliceUnion<F>::StagingBufferInfo> to_erase;
      for (auto &i : u->mpi_staging_buffers) {
        int completed = i.abc == abc;
        if (completed) { to_erase.push_back(i); }
      }
      for (auto &i : to_erase) {
        u->freePointers.insert(i.data);
        u->mpi_staging_buffers.erase(i);
      }
    }
#endif /* defined(ATRIP_MPI_STAGING_BUFFERS) */

    MPI_Barrier(kaun);

  } /* end iterations */

  if (!rank) {
    std::cout << "=========================================================\n";
    std::cout << "FINISHING, it will segfaulten, that's ok, don't even trip"
              << std::endl;
  }
  MPI_Barrier(kaun);
  DEINIT_DRY(t_abph);
  DEINIT_DRY(t_abhh);
  DEINIT_DRY(t_tabhh);
  DEINIT_DRY(t_taphh);
  DEINIT_DRY(t_hhha);

  MPI_Finalize();
  return 0;
}
