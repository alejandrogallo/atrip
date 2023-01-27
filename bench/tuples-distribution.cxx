#include <iostream>
#define ATRIP_DEBUG 2
#include <atrip/Atrip.hpp>
#include <atrip/Tuples.hpp>
#include <atrip/Unions.hpp>
#include <bench/CLI11.hpp>
#include <bench/utils.hpp>

using namespace atrip;

using F = double;
using Tr = CTF::Tensor<F>;

#define INIT_DRY(name, ...)                                       \
  do {                                                            \
    std::vector<int64_t> lens = __VA_ARGS__;                      \
    int i = -1;                                                   \
    name.order = lens.size();                                     \
    name.lens = (int64_t*)malloc(sizeof(int64_t) * lens.size());  \
    name.sym = (int*)malloc(sizeof(int) * lens.size());           \
    name.lens[++i] = lens[i]; name.lens[++i] = lens[i];           \
    name.lens[++i] = lens[i]; name.lens[++i] = lens[i];           \
    i = 0;                                                        \
    name.sym[i++] = NS; name.sym[i++] = NS;                       \
    name.sym[i++] = NS; name.sym[i++] = NS;                       \
  } while (0)

#define DEINIT_DRY(name)                        \
  do {                                          \
    name.order = 0;                             \
    name.lens = NULL;                           \
    name.sym = NULL;                            \
  } while (0)

using LocalDatabase = typename Slice<F>::LocalDatabase;
using LocalDatabaseElement = typename Slice<F>::LocalDatabaseElement;


template <typename F>
std::string type_to_string(typename Slice<F>::Type t) {
  switch (t) {
  case Slice<F>::AB: return "AB";
  case Slice<F>::BC: return "BC";
  case Slice<F>::AC: return "AC";
  case Slice<F>::CB: return "CB";
  case Slice<F>::BA: return "BA";
  case Slice<F>::CA: return "CA";
  case  Slice<F>::A: return "A";
  case  Slice<F>::B: return "B";
  case  Slice<F>::C: return "C";
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

LocalDatabase buildLocalDatabase(SliceUnion<F> &u,
                                 ABCTuple const& abc) {
  LocalDatabase result;

  auto const needed = u.neededSlices(abc);

  // BUILD THE DATABASE
  // we need to loop over all sliceTypes that this TensorUnion
  // is representing and find out how we will get the corresponding
  // slice for the abc we are considering right now.
  for (auto const& pair: needed) {
    auto const type = pair.first;
    auto const tuple = pair.second;
    auto const from  = u.rankMap.find(abc, type);

    {
      // FIRST: look up if there is already a *Ready* slice matching what we
      // need
      auto const& it
        = std::find_if(u.slices.begin(), u.slices.end(),
                       [&tuple, &type](Slice<F> const& other) {
                         return other.info.tuple == tuple
                           && other.info.type == type
                           // we only want another slice when it
                           // has already ready-to-use data
                           && other.isUnwrappable()
                           ;
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
    auto const& recycleIt
      = std::find_if(u.slices.begin(), u.slices.end(),
                     [&tuple, &type](Slice<F> const& other) {
                       return other.info.tuple == tuple
                         && other.info.type != type
                         && other.isRecyclable()
                         ;
                     });

    //
    // if we find this recylce, then we find a Blank slice
    // (which should exist by construction :THINK)
    //
    if (recycleIt != u.slices.end()) {
      auto& blank = Slice<F>::findOneByType(u.slices, Slice<F>::Blank);
      // TODO: formalize this through a method to copy information
      //       from another slice
      blank.data = recycleIt->data;
      blank.info.type = type;
      blank.info.tuple = tuple;
      blank.info.state = Slice<F>::Recycled;
      blank.info.from = from;
      blank.info.recycling = recycleIt->info.type;
      result.push_back({u.name, blank.info});
      WITH_RANK << "__db__: RECYCLING: n" << u.name
                << " " << pretty_print(abc)
                << " get " << pretty_print(blank.info)
                << " from " << pretty_print(recycleIt->info)
                << " ptr " << recycleIt->data
                << "\n"
                ;
      continue;
    }

    // in this case we have to create a new slice
    // this means that we should have a blank slice at our disposal
    // and also the freePointers should have some elements inside,
    // so we pop a data pointer from the freePointers container
    {
      auto& blank = Slice<F>::findOneByType(u.slices, Slice<F>::Blank);
      blank.info.type = type;
      blank.info.tuple = tuple;
      blank.info.from = from;

      // Handle self sufficiency
      blank.info.state = Atrip::rank == from.rank
                        ? Slice<F>::SelfSufficient
                        : Slice<F>::Fetch
                        ;
      if (blank.info.state == Slice<F>::SelfSufficient) {
        blank.data = (F*)0xBADA55;
      } else {
        blank.data = (F*)0xA55A55;
      }

      result.push_back({u.name, blank.info});
      continue;
    }

  }

  return result;

}

void clearUnusedSlicesForNext(SliceUnion<F> &u,
                              ABCTuple const& abc) {
  auto const needed = u.neededSlices(abc);

  // CLEAN UP SLICES, FREE THE ONES THAT ARE NOT NEEDED ANYMORE
  for (auto& slice: u.slices) {
    // if the slice is free, then it was not used anyways
    if (slice.isFree()) continue;


    // try to find the slice in the needed slices list
    auto const found
      = std::find_if(needed.begin(), needed.end(),
                      [&slice] (typename Slice<F>::Ty_x_Tu const& tytu) {
                        return slice.info.tuple == tytu.second
                            && slice.info.type == tytu.first
                            ;
                      });

    // if we did not find slice in needed, then erase it
    if (found == needed.end()) {

      // allow to gc unwrapped and recycled, never Fetch,
      // if we have a Fetch slice then something has gone very wrong.
      if (!slice.isUnwrapped() && slice.info.state != Slice<F>::Recycled)
        throw
          std::domain_error(_FORMAT("Trying to garbage collect (%d, %d) "
                                    " a non-unwrapped slice! ",
                                    slice.info.type,
                                    slice.info.state));

      // it can be that our slice is ready, but it has some hanging
      // references lying around in the form of a recycled slice.
      // Of course if we need the recycled slice the next iteration
      // this would be fatal, because we would then free the pointer
      // of the slice and at some point in the future we would
      // overwrite it. Therefore, we must check if slice has some
      // references in slices and if so then
      //
      //  - we should mark those references as the original (since the data
      //    pointer should be the same)
      //
      //  - we should make sure that the data pointer of slice
      //    does not get freed.
      //
      if (slice.info.state == Slice<F>::Ready) {
        WITH_OCD WITH_RANK
          << "__gc__:" << "checking for data recycled dependencies\n";
        auto recycled
          = Slice<F>::hasRecycledReferencingToIt(u.slices, slice.info);
        if (recycled.size()) {
          Slice<F>* newReady = recycled[0];
          WITH_OCD WITH_RANK
            << "__gc__:" << "swaping recycled "
            << pretty_print(newReady->info)
            << " and "
            << pretty_print(slice.info)
            << "\n";
          newReady->markReady();

          for (size_t i = 1; i < recycled.size(); i++) {
            auto newRecyled = recycled[i];
            newRecyled->info.recycling = newReady->info.type;
            WITH_OCD WITH_RANK
              << "__gc__:" << "updating recycled "
              << pretty_print(newRecyled->info)
              << "\n";
          }

        }
      }

      slice.free();
    }  // we did not find the slice

  }
}


void unwrapSlice(Slice<F>::Type t, ABCTuple abc, SliceUnion<F> *u) {
  auto& slice = Slice<F>::findByTypeAbc(u->slices, t, abc);
  switch  (slice.info.state) {
  case Slice<F>::Dispatched:
    slice.markReady();
    break;
  case Slice<F>::Recycled:
    unwrapSlice(t, abc, u);
    break;
  }
}

#define PRINT_VARIABLE(v)                                         \
  do {                                                            \
    if (!rank) std::cout << "# " << #v << ": " << v << std::endl; \
  } while (0)

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int no(10), nv(100), mod(1000), out_rank(0);
  std::string tuplesDistributionString = "naive";

  CLI::App app{"Main bench for atrip"};
  app.add_option("--no", no, "Occupied orbitals");
  app.add_option("--nv", nv, "Virtual orbitals");
  app.add_option("--dist", tuplesDistributionString, "Which distribution");
  app.add_option("--mod", mod, "Every each iteration there is output");
  app.add_option("--out-rank", out_rank, "Stdout for given rank");
  CLI11_PARSE(app, argc, argv);

  CTF::World world(argc, argv);
  auto kaun = world.comm;
  int rank, np;
  MPI_Comm_rank(kaun, &rank);
  MPI_Comm_size(kaun, &np);
  Atrip::init(world.comm);



  atrip::ABCTuples tuplesList;
  atrip::TuplesDistribution *dist;
  {
    using namespace atrip;
    if (tuplesDistributionString == "naive") {
      dist = new NaiveDistribution();
      tuplesList = dist->getTuples(nv, world.comm);
    } else if (tuplesDistributionString == "group") {
      dist = new group_and_sort::Distribution();
      tuplesList = dist->getTuples(nv, world.comm);
    } else {
      std::cout << "--dist should be either naive or group\n";
      exit(1);
    }
  }

  double tuplesListGb
    = tuplesList.size() * sizeof(tuplesList[0])
    / 1024.0 / 1024.0 / 1024.0;

  std::cout << "\n";
  PRINT_VARIABLE(tuplesDistributionString);
  PRINT_VARIABLE(np);
  PRINT_VARIABLE(no);
  PRINT_VARIABLE(nv);
  PRINT_VARIABLE(tuplesList.size());
  PRINT_VARIABLE(tuplesListGb);

  // create a fake dry tensor
  Tr t_abph, t_abhh, t_tabhh, t_taphh, t_hhha;
  INIT_DRY(t_abph  , {nv, nv, nv, no});
  INIT_DRY(t_abhh  , {nv, nv, no, no});
  INIT_DRY(t_tabhh , {nv, nv, no, no});
  INIT_DRY(t_taphh , {nv, nv, no, no});
  INIT_DRY(t_hhha  , {no, no, no, nv});

  ABPH<F> abph(t_abph, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  ABHH<F> abhh(t_abhh, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  TABHH<F> tabhh(t_tabhh, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  TAPHH<F> taphh(t_taphh, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  HHHA<F>  hhha(t_hhha, (size_t)no, (size_t)nv, (size_t)np, kaun, kaun);
  std::vector< SliceUnion<F>* > unions = {&taphh, &hhha, &abph, &abhh, &tabhh};



  using Database = typename Slice<F>::Database;
  auto communicateDatabase
    = [ &unions
      , np
      ] (ABCTuple const& abc, MPI_Comm const& c) -> Database {

        WITH_CHRONO("db:comm:type:do",
          auto MPI_LDB_ELEMENT = Slice<F>::mpi::localDatabaseElement();
        )

        WITH_CHRONO("db:comm:ldb",
          typename Slice<F>::LocalDatabase ldb;
          for (auto const& tensor: unions) {
            auto const& tensorDb = buildLocalDatabase(*tensor, abc);
            ldb.insert(ldb.end(), tensorDb.begin(), tensorDb.end());
          }
        )

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
                                  c);
        ))

        WITH_CHRONO("db:comm:type:free", MPI_Type_free(&MPI_LDB_ELEMENT);)

        return db;
      };

  auto doIOPhase
    = [&unions, &rank, &np, mod, out_rank, no, nv] (Database const& db,
                                            std::vector<LocalDatabaseElement> &to_send,
                                            size_t iteration) {

    const size_t localDBLength = db.size() / np;

    size_t sendTag = 0
         , recvTag = rank * localDBLength
         ;

    // RECIEVE PHASE ======================================================
    {
      // At this point, we have already send to everyone that fits
      auto const& begin = &db[rank * localDBLength]
                , end   = begin + localDBLength
                ;
      for (auto it = begin; it != end; ++it) {
        recvTag++;
        auto const& el = *it;
        auto& u = unionByName(unions, el.name);
        auto& slice = Slice<F>::findByInfo(u.slices, el.info);
        slice.markReady();
        // u.receive(el.info, recvTag);
        if (rank == out_rank) {
          std::cout << _FORMAT("%4s %d %d %d %5d %5s %2s %14s (%ld,%ld) %ld",
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
                               name_to_size<double>(el.name, no, nv))
                    << "\n";
        }

      } // recv
    }

    // SEND PHASE =========================================================
    for (size_t otherRank = 0; otherRank < np; otherRank++) {
      auto const& begin = &db[otherRank * localDBLength]
                , end = begin + localDBLength
                ;
      for (auto it = begin; it != end; ++it) {
        sendTag++;
        typename Slice<F>::LocalDatabaseElement const& el = *it;
        if (el.info.from.rank != rank) continue;
        auto& u = unionByName(unions, el.name);
        if (el.info.state == Slice<F>::Fetch) {
          to_send.push_back(el);
        }
        if (rank == out_rank) {
          std::cout << _FORMAT("%4s %d %d %d %5d %5s %2s %14s (%ld,%ld) %ld",
                               "SEND",
                               iteration,
                               rank,
                               el.info.from.rank,
                               recvTag,
                               name_to_string<double>(el.name).c_str(),
                               type_to_string<double>(el.info.type).c_str(),
                               state_to_string<double>(el.info.state).c_str(),
                               el.info.tuple[0],
                               el.info.tuple[1],
                               name_to_size<double>(el.name, no, nv))
                    << "\n";
        }
        // u.send(otherRank, el, sendTag);

      } // send phase

    } // otherRank


  };

  std::vector<LocalDatabaseElement>
    to_send;

  for (size_t it = 0; it < tuplesList.size(); it++) {


    const ABCTuple abc = dist->tupleIsFake(tuplesList[it])
                       ? tuplesList[tuplesList.size() - 1]
                       : tuplesList[it]
                 ;

    if (it > 0) {
      for (auto const& u: unions) {
        clearUnusedSlicesForNext(*u, abc);
      }
    }

    const auto db = communicateDatabase(abc, kaun);
    doIOPhase(db, to_send, it);

    if (it % mod == 0)
      std::cout << _FORMAT("%ld :it %ld  %f %% ∷ %ld ∷ %f GB\n",
                           rank,
                           it,
                           100.0 * double(to_send.size()) / double(tuplesList.size()),
                           to_send.size(),
                           double(to_send.size()) * sizeof(to_send[0])
                           / 1024.0 / 1024.0 / 1024.0);


    for (auto const& u: unions) {
      for (auto type: u->sliceTypes) {
        unwrapSlice(type, abc, u);
      }
    }


  }

  std::cout << "=========================================================\n";
  std::cout << "FINISHING, it will segfaulten, that's ok, don't even trip"
            << std::endl;
  MPI_Barrier(kaun);
  DEINIT_DRY(t_abph);
  DEINIT_DRY(t_abhh);
  DEINIT_DRY(t_tabhh);
  DEINIT_DRY(t_taphh);
  DEINIT_DRY(t_hhha);

  MPI_Finalize();
  return 0;
}
