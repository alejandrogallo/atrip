// [[file:../../atrip.org::*Main][Main:1]]
#include <iomanip>

#include <atrip/Atrip.hpp>
#include <atrip/Utils.hpp>
#include <atrip/Equations.hpp>
#include <atrip/SliceUnion.hpp>
#include <atrip/Unions.hpp>

using namespace atrip;

int Atrip::rank;
int Atrip::np;

// user printing block
IterationDescriptor IterationDescription::descriptor;
void atrip::registerIterationDescriptor(IterationDescriptor d) {
  IterationDescription::descriptor = d;
}

void Atrip::init()  {
  MPI_Comm_rank(MPI_COMM_WORLD, &Atrip::rank);
  MPI_Comm_size(MPI_COMM_WORLD, &Atrip::np);
}

Atrip::Output Atrip::run(Atrip::Input const& in) {

  const int np = Atrip::np;
  const int rank = Atrip::rank;
  MPI_Comm universe = in.ei->wrld->comm;

  // Timings in seconds ================================================{{{1
  Timings chrono{};

  const size_t No = in.ei->lens[0];
  const size_t Nv = in.ea->lens[0];
  LOG(0,"Atrip") << "No: " << No << "\n";
  LOG(0,"Atrip") << "Nv: " << Nv << "\n";

  // allocate the three scratches, see piecuch
  std::vector<double> Tijk(No*No*No) // doubles only (see piecuch)
                    , Zijk(No*No*No) // singles + doubles (see piecuch)
                    // we need local copies of the following tensors on every
                    // rank
                    , epsi(No)
                    , epsa(Nv)
                    , Tai(No * Nv)
                    ;

  in.ei->read_all(epsi.data());
  in.ea->read_all(epsa.data());
  in.Tph->read_all(Tai.data());

  // COMMUNICATOR CONSTRUCTION ========================================={{{1
  //
  // Construct a new communicator living only on a single rank
  int child_size = 1
    , child_rank
    ;
  const
  int color = rank / child_size
    , crank = rank % child_size
    ;
  MPI_Comm child_comm;
  if (np == 1) {
    child_comm = universe;
  } else {
    MPI_Comm_split(universe, color, crank, &child_comm);
    MPI_Comm_rank(child_comm, &child_rank);
    MPI_Comm_size(child_comm, &child_size);
  }


  chrono["nv-slices"].start();
  // BUILD SLICES PARAMETRIZED BY NV ==================================={{{1
  LOG(0,"Atrip") << "BUILD NV-SLICES\n";
  TAPHH taphh(*in.Tpphh, (size_t)No, (size_t)Nv, (size_t)np, child_comm, universe);
  HHHA  hhha(*in.Vhhhp, (size_t)No, (size_t)Nv, (size_t)np, child_comm, universe);
  chrono["nv-slices"].stop();

  chrono["nv-nv-slices"].start();
  // BUILD SLICES PARAMETRIZED BY NV x NV =============================={{{1
  LOG(0,"Atrip") << "BUILD NV x NV-SLICES\n";
  ABPH abph(*in.Vppph, (size_t)No, (size_t)Nv, (size_t)np, child_comm, universe);
  ABHH abhh(*in.Vpphh, (size_t)No, (size_t)Nv, (size_t)np, child_comm, universe);
  TABHH tabhh(*in.Tpphh, (size_t)No, (size_t)Nv, (size_t)np, child_comm, universe);
  chrono["nv-nv-slices"].stop();

  // all tensors
  std::vector< SliceUnion* > unions = {&taphh, &hhha, &abph, &abhh, &tabhh};

  //CONSTRUCT TUPLE LIST ==============================================={{{1
  LOG(0,"Atrip") << "BUILD TUPLE LIST\n";
  const auto tuplesList = std::move(getTuplesList(Nv));
  WITH_RANK << "tupList.size() = " << tuplesList.size() << "\n";

  // GET ABC INDEX RANGE FOR RANK ======================================{{{1
  auto abcIndex = getABCRange(np, rank, tuplesList);
  size_t nIterations = abcIndex.second - abcIndex.first;

  WITH_RANK << "abcIndex = " << pretty_print(abcIndex) << "\n";
  LOG(0,"Atrip") << "#iterations: " << nIterations << "\n";

  // first abc
  const ABCTuple firstAbc = tuplesList[abcIndex.first];


  double energy(0.);

  const size_t
      iterationMod = (in.percentageMod > 0)
                  ? nIterations * in.percentageMod / 100
                  : in.iterationMod

    , iteration1Percent = nIterations * 0.01
    ;



  auto const isFakeTuple
    = [&tuplesList](size_t const i) { return i >= tuplesList.size(); };


  auto communicateDatabase
    = [ &unions
      , np
      , &chrono
      ] (ABCTuple const& abc, MPI_Comm const& c) -> Slice::Database {

        chrono["db:comm:type:do"].start();
        auto MPI_LDB_ELEMENT = Slice::mpi::localDatabaseElement();
        chrono["db:comm:type:do"].stop();

        chrono["db:comm:ldb"].start();
        Slice::LocalDatabase ldb;

        for (auto const& tensor: unions) {
          auto const& tensorDb = tensor->buildLocalDatabase(abc);
          ldb.insert(ldb.end(), tensorDb.begin(), tensorDb.end());
        }
        chrono["db:comm:ldb"].stop();

        Slice::Database db(np * ldb.size(), ldb[0]);

        chrono["oneshot-db:comm:allgather"].start();
        chrono["db:comm:allgather"].start();
        MPI_Allgather( ldb.data()
                     , ldb.size()
                     , MPI_LDB_ELEMENT
                     , db.data()
                     , ldb.size()
                     , MPI_LDB_ELEMENT
                     , c);
        chrono["db:comm:allgather"].stop();
        chrono["oneshot-db:comm:allgather"].stop();

        chrono["db:comm:type:free"].start();
        MPI_Type_free(&MPI_LDB_ELEMENT);
        chrono["db:comm:type:free"].stop();

        return db;
      };

  auto doIOPhase
    = [&unions, &rank, &np, &universe, &chrono] (Slice::Database const& db) {

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

        WITH_DBG std::cout
          << rank << ":r"
          << "♯" << recvTag << " =>"
          << " «n" << el.name
          << ", t" << el.info.type
          << ", s" << el.info.state
          << "»"
          << " ⊙ {" << rank << "⇐" << el.info.from.rank
                    << ", "
                    << el.info.from.source << "}"
          << " ∴ {" << el.info.tuple[0]
                    << ", "
                    << el.info.tuple[1]
                    << "}"
          << "\n"
          ;

        chrono["db:io:recv"].start();
        u.receive(el.info, recvTag);
        chrono["db:io:recv"].stop();

      } // recv
    }

    // SEND PHASE =========================================================
    for (size_t otherRank = 0; otherRank<np; otherRank++) {
      auto const& begin = &db[otherRank * localDBLength]
                , end = begin + localDBLength
                ;
      for (auto it = begin; it != end; ++it) {
        sendTag++;
        Slice::LocalDatabaseElement const& el = *it;

        if (el.info.from.rank != rank) continue;

        auto& u = unionByName(unions, el.name);
        WITH_DBG std::cout
          << rank << ":s"
          << "♯" << sendTag << " =>"
          << " «n" << el.name
          << ", t" << el.info.type
          << ", s" << el.info.state
          << "»"
          << " ⊙ {" << el.info.from.rank << "⇒" << otherRank
                    << ", "
                    << el.info.from.source << "}"
          << " ∴ {" << el.info.tuple[0]
                    << ", "
                    << el.info.tuple[1]
                    << "}"
          << "\n"
          ;

        chrono["db:io:send"].start();
        u.send(otherRank, el.info, sendTag);
        chrono["db:io:send"].stop();

      } // send phase

    } // otherRank


  };

#if defined(HAVE_OCD) || defined(ATRIP_PRINT_TUPLES)
  std::map<ABCTuple, double> tupleEnergies;
#endif

  const double doublesFlops
    = double(No)
    * double(No)
    * double(No)
    * (double(No) + double(Nv))
    * 2
    * 6
    / 1e9
    ;

  // START MAIN LOOP ======================================================{{{1

  Slice::Database db;

  for ( size_t i = abcIndex.first, iteration = 1
      ; i < abcIndex.second
      ; i++, iteration++
      ) {
    chrono["iterations"].start();

    // check overhead from chrono over all iterations
    chrono["start:stop"].start(); chrono["start:stop"].stop();

    // check overhead of doing a barrier at the beginning
    chrono["oneshot-mpi:barrier"].start();
    chrono["mpi:barrier"].start();
    // TODO: REMOVE
    if (in.barrier == 1)
    MPI_Barrier(universe);
    chrono["mpi:barrier"].stop();
    chrono["oneshot-mpi:barrier"].stop();

    if (iteration % iterationMod == 0 || iteration == iteration1Percent) {

      if (IterationDescription::descriptor) {
        IterationDescription::descriptor({
          iteration,
          nIterations,
          chrono["iterations"].count()
        });
      }

      LOG(0,"Atrip")
        << "iteration " << iteration
        << " [" << 100 * iteration / nIterations << "%]"
        << " (" << doublesFlops * iteration / chrono["doubles"].count()
        << "GF)"
        << " (" << doublesFlops * iteration / chrono["iterations"].count()
        << "GF)"
        << " ===========================\n";

      // PRINT TIMINGS
      if (in.chrono)
      for (auto const& pair: chrono)
        LOG(1, " ") << pair.first << " :: "
                    << pair.second.count()
                    << std::endl;

    }

    const ABCTuple abc = isFakeTuple(i)
                       ? tuplesList[tuplesList.size() - 1]
                       : tuplesList[i]
                 , *abcNext = i == (abcIndex.second - 1)
                            ? nullptr
                            : isFakeTuple(i + 1)
                            ? &tuplesList[tuplesList.size() - 1]
                            : &tuplesList[i + 1]
                 ;

    chrono["with_rank"].start();
    WITH_RANK << " :it " << iteration
              << " :abc " << pretty_print(abc)
              << " :abcN "
              << (abcNext ? pretty_print(*abcNext) : "None")
              << "\n";
    chrono["with_rank"].stop();


    // COMM FIRST DATABASE ================================================{{{1
    if (i == abcIndex.first) {
      WITH_RANK << "__first__:first database ............ \n";
      const auto __db = communicateDatabase(abc, universe);
      WITH_RANK << "__first__:first database communicated \n";
      WITH_RANK << "__first__:first database io phase \n";
      doIOPhase(__db);
      WITH_RANK << "__first__:first database io phase DONE\n";
      WITH_RANK << "__first__::::Unwrapping all slices for first database\n";
      for (auto& u: unions) u->unwrapAll(abc);
      WITH_RANK << "__first__::::Unwrapping all slices for first database DONE\n";
      MPI_Barrier(universe);
    }

    // COMM NEXT DATABASE ================================================={{{1
    if (abcNext) {
      WITH_RANK << "__comm__:" << iteration << "th communicating database\n";
      chrono["db:comm"].start();
      //const auto db = communicateDatabase(*abcNext, universe);
      db = communicateDatabase(*abcNext, universe);
      chrono["db:comm"].stop();
      chrono["db:io"].start();
      doIOPhase(db);
      chrono["db:io"].stop();
      WITH_RANK << "__comm__:" <<  iteration << "th database io phase DONE\n";
    }

    // COMPUTE DOUBLES ===================================================={{{1
    OCD_Barrier(universe);
    if (!isFakeTuple(i)) {
      WITH_RANK << iteration << "-th doubles\n";
      WITH_CHRONO(chrono["oneshot-unwrap"],
      WITH_CHRONO(chrono["unwrap"],
      WITH_CHRONO(chrono["unwrap:doubles"],
        for (auto& u: decltype(unions){&abph, &hhha, &taphh, &tabhh}) {
          u->unwrapAll(abc);
        }
      )))
      chrono["oneshot-doubles"].start();
      chrono["doubles"].start();
      doublesContribution( abc, (size_t)No, (size_t)Nv
                         // -- VABCI
                         , abph.unwrapSlice(Slice::AB, abc)
                         , abph.unwrapSlice(Slice::AC, abc)
                         , abph.unwrapSlice(Slice::BC, abc)
                         , abph.unwrapSlice(Slice::BA, abc)
                         , abph.unwrapSlice(Slice::CA, abc)
                         , abph.unwrapSlice(Slice::CB, abc)
                         // -- VHHHA
                         , hhha.unwrapSlice(Slice::A, abc)
                         , hhha.unwrapSlice(Slice::B, abc)
                         , hhha.unwrapSlice(Slice::C, abc)
                         // -- TA
                         , taphh.unwrapSlice(Slice::A, abc)
                         , taphh.unwrapSlice(Slice::B, abc)
                         , taphh.unwrapSlice(Slice::C, abc)
                         // -- TABIJ
                         , tabhh.unwrapSlice(Slice::AB, abc)
                         , tabhh.unwrapSlice(Slice::AC, abc)
                         , tabhh.unwrapSlice(Slice::BC, abc)
                         // -- TIJK
                         , Tijk.data()
                         , chrono
                         );
      WITH_RANK << iteration << "-th doubles done\n";
      chrono["doubles"].stop();
      chrono["oneshot-doubles"].stop();
    }

    // COMPUTE SINGLES =================================================== {{{1
    OCD_Barrier(universe);
    if (!isFakeTuple(i)) {
      WITH_CHRONO(chrono["oneshot-unwrap"],
      WITH_CHRONO(chrono["unwrap"],
      WITH_CHRONO(chrono["unwrap:singles"],
        abhh.unwrapAll(abc);
      )))
      chrono["reorder"].start();
      for (size_t I(0); I < Zijk.size(); I++) Zijk[I] = Tijk[I];
      chrono["reorder"].stop();
      chrono["singles"].start();
      singlesContribution( No, Nv, abc
                         , Tai.data()
                         , abhh.unwrapSlice(Slice::AB, abc)
                         , abhh.unwrapSlice(Slice::AC, abc)
                         , abhh.unwrapSlice(Slice::BC, abc)
                         , Zijk.data());
      chrono["singles"].stop();
    }


    // COMPUTE ENERGY ==================================================== {{{1
    if (!isFakeTuple(i)) {
      double tupleEnergy(0.);

      int distinct(0);
      if (abc[0] == abc[1]) distinct++;
      if (abc[1] == abc[2]) distinct--;
      const double epsabc(epsa[abc[0]] + epsa[abc[1]] + epsa[abc[2]]);

      chrono["energy"].start();
      if ( distinct == 0)
        tupleEnergy = getEnergyDistinct(epsabc, epsi, Tijk, Zijk);
      else
        tupleEnergy = getEnergySame(epsabc, epsi, Tijk, Zijk);
      chrono["energy"].stop();

#if defined(HAVE_OCD) || defined(ATRIP_PRINT_TUPLES)
      tupleEnergies[abc] = tupleEnergy;
#endif

      energy += tupleEnergy;

    }

    if (isFakeTuple(i)) {
      // fake iterations should also unwrap whatever they got
      WITH_RANK << iteration
                << "th unwrapping because of fake in "
                << i << "\n";
      for (auto& u: unions) u->unwrapAll(abc);
    }

#ifdef HAVE_OCD
    for (auto const& u: unions) {
      WITH_RANK << "__dups__:"
                << iteration
                << "-th n" << u->name << " checking duplicates\n";
      u->checkForDuplicates();
    }
#endif


    // CLEANUP UNIONS ===================================================={{{1
    OCD_Barrier(universe);
    if (abcNext) {
      chrono["gc"].start();
      WITH_RANK << "__gc__:" << iteration << "-th cleaning up.......\n";
      for (auto& u: unions) {

        u->unwrapAll(abc);
        WITH_RANK << "__gc__:n" << u->name  << " :it " << iteration
                  << " :abc " << pretty_print(abc)
                  << " :abcN " << pretty_print(*abcNext)
                  << "\n";
        for (auto const& slice: u->slices)
          WITH_RANK << "__gc__:guts:" << slice.info << "\n";
        u->clearUnusedSlicesForNext(*abcNext);

        WITH_RANK << "__gc__: checking validity\n";

#ifdef HAVE_OCD
        // check for validity of the slices
        for (auto type: u->sliceTypes) {
          auto tuple = Slice::subtupleBySlice(abc, type);
        for (auto& slice: u->slices) {
          if ( slice.info.type == type
             && slice.info.tuple == tuple
             && slice.isDirectlyFetchable()
             ) {
            if (slice.info.state == Slice::Dispatched)
              throw std::domain_error( "This slice should not be undispatched! "
                                     + pretty_print(slice.info));
          }
        }
        }
#endif


      }
      chrono["gc"].stop();
    }

      WITH_RANK << iteration << "-th cleaning up....... DONE\n";

    chrono["iterations"].stop();
    // ITERATION END ====================================================={{{1

  }
    // END OF MAIN LOOP

  MPI_Barrier(universe);

  // PRINT TUPLES ========================================================={{{1
#if defined(HAVE_OCD) || defined(ATRIP_PRINT_TUPLES)
  LOG(0,"Atrip") << "tuple energies" << "\n";
  for (size_t i = 0; i < np; i++) {
    MPI_Barrier(universe);
    for (auto const& pair: tupleEnergies) {
      if (i == rank)
        std::cout << pair.first[0]
                  << " " << pair.first[1]
                  << " " << pair.first[2]
                  << std::setprecision(15) << std::setw(23)
                  << " tupleEnergy: " << pair.second
                  << "\n"
                  ;
    }
  }
#endif

  // COMMUNICATE THE ENERGIES ============================================={{{1
  LOG(0,"Atrip") << "COMMUNICATING ENERGIES \n";
  double globalEnergy = 0;
  MPI_Reduce(&energy, &globalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, universe);

  WITH_RANK << "local energy " << energy << "\n";
  LOG(0, "Atrip") << "Energy: "
    << std::setprecision(15) << std::setw(23)
    << (- globalEnergy) << std::endl;

  // PRINT TIMINGS {{{1
  if (in.chrono)
  for (auto const& pair: chrono)
    LOG(0,"atrip:chrono") << pair.first << " "
                          << pair.second.count() << std::endl;


  LOG(0, "atrip:flops(doubles)")
    << nIterations * doublesFlops / chrono["doubles"].count() << "\n";
  LOG(0, "atrip:flops(iterations)")
    << nIterations * doublesFlops / chrono["iterations"].count() << "\n";

  // TODO: change the sign in  the getEnergy routines
  return { - globalEnergy };

}
// Main:1 ends here
