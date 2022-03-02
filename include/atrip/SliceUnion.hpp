// [[file:../../atrip.org::*The slice union][The slice union:1]]
#pragma once
#include <atrip/Debug.hpp>
#include <atrip/Slice.hpp>
#include <atrip/RankMap.hpp>

namespace atrip {

  struct SliceUnion {
    using F = double;
    using Tensor = CTF::Tensor<F>;

    virtual void
    sliceIntoBuffer(size_t iteration, Tensor &to, Tensor const& from) = 0;

    /*
     * This function should enforce an important property of a SliceUnion.
     * Namely, there can be no two Slices of the same nature.
     *
     * This means that there can be at most one slice with a given Ty_x_Tu.
     */
    void checkForDuplicates() const {
      std::vector<Slice::Ty_x_Tu> tytus;
      for (auto const& s: slices) {
        if (s.isFree()) continue;
        tytus.push_back({s.info.type, s.info.tuple});
      }

      for (auto const& tytu: tytus) {
        if (std::count(tytus.begin(), tytus.end(), tytu) > 1)
          throw "Invariance violated, more than one slice with same Ty_x_Tu";
      }

    }

    std::vector<Slice::Ty_x_Tu> neededSlices(ABCTuple const& abc) {
      std::vector<Slice::Ty_x_Tu> needed(sliceTypes.size());
      // build the needed vector
      std::transform(sliceTypes.begin(), sliceTypes.end(),
                     needed.begin(),
                     [&abc](Slice::Type const type) {
                       auto tuple = Slice::subtupleBySlice(abc, type);
                       return std::make_pair(type, tuple);
                     });
      return needed;
    }

    /* buildLocalDatabase
     *
     * It should build a database of slices so that we know what is needed
     * to fetch in the next iteration represented by the tuple 'abc'.
     *
     * 1. The algorithm works as follows, we build a database of the all
     * the slice types that we need together with their tuple.
     *
     * 2. Look in the SliceUnion if we already have this tuple,
     * if we already have it mark it (TODO)
     *
     * 3. If we don't have the tuple, look for a (state=acceptor, type=blank)
     * slice and mark this slice as type=Fetch with the corresponding type
     * and tuple.
     *
     * NOTE: The algorithm should certify that we always have enough blank
     * slices.
     *
     */
    Slice::LocalDatabase buildLocalDatabase(ABCTuple const& abc) {
      Slice::LocalDatabase result;

      auto const needed = neededSlices(abc);

      WITH_RANK << "__db__:needed:" << pretty_print(needed) << "\n";
      // BUILD THE DATABASE
      // we need to loop over all sliceTypes that this TensorUnion
      // is representing and find out how we will get the corresponding
      // slice for the abc we are considering right now.
      for (auto const& pair: needed) {
        auto const type = pair.first;
        auto const tuple = pair.second;
        auto const from  = rankMap.find(abc, type);

#ifdef HAVE_OCD
        WITH_RANK << "__db__:want:" << pretty_print(pair) << "\n";
        for (auto const& s: slices)
          WITH_RANK << "__db__:guts:ocd "
                    << s.info << " pt " << s.data
                    << "\n";
#endif

#ifdef HAVE_OCD
        WITH_RANK << "__db__: checking if exact match" << "\n";
#endif
        {
          // FIRST: look up if there is already a *Ready* slice matching what we
          // need
          auto const& it
            = std::find_if(slices.begin(), slices.end(),
                           [&tuple, &type](Slice const& other) {
                             return other.info.tuple == tuple
                                 && other.info.type == type
                                    // we only want another slice when it
                                    // has already ready-to-use data
                                 && other.isUnwrappable()
                                 ;
                           });
          if (it != slices.end()) {
            // if we find this slice, it means that we don't have to do anything
            WITH_RANK << "__db__: EXACT: found EXACT in name=" << name
                      << " for tuple " << tuple[0] << ", " << tuple[1]
                      << " ptr " << it->data
                      << "\n";
            result.push_back({name, it->info});
            continue;
          }
        }

#ifdef HAVE_OCD
        WITH_RANK << "__db__: checking if recycle" << "\n";
#endif
        // Try to find a recyling possibility ie. find a slice with the same
        // tuple and that has a valid data pointer.
        auto const& recycleIt
          = std::find_if(slices.begin(), slices.end(),
                         [&tuple, &type](Slice const& other) {
                           return other.info.tuple == tuple
                               && other.info.type != type
                               && other.isRecyclable()
                               ;
                         });

        // if we find this recylce, then we find a Blank slice
        // (which should exist by construction :THINK)
        //
        if (recycleIt != slices.end()) {
          auto& blank = Slice::findOneByType(slices, Slice::Blank);
          // TODO: formalize this through a method to copy information
          //       from another slice
          blank.data = recycleIt->data;
          blank.info.type = type;
          blank.info.tuple = tuple;
          blank.info.state = Slice::Recycled;
          blank.info.from = from;
          blank.info.recycling = recycleIt->info.type;
          result.push_back({name, blank.info});
          WITH_RANK << "__db__: RECYCLING: n" << name
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
#ifdef HAVE_OCD
        WITH_RANK << "__db__: none work, doing new" << "\n";
#endif
        {
          WITH_RANK << "__db__: NEW: finding blank in " << name
                    << " for type " << type
                    << " for tuple " << tuple[0] << ", " << tuple[1]
                    << "\n"
                    ;
          auto& blank = Slice::findOneByType(slices, Slice::Blank);
          blank.info.type = type;
          blank.info.tuple = tuple;
          blank.info.from = from;

          // Handle self sufficiency
          blank.info.state = Atrip::rank == from.rank
                           ? Slice::SelfSufficient
                           : Slice::Fetch
                           ;
          if (blank.info.state == Slice::SelfSufficient) {
            blank.data = sources[from.source].data();
          } else {
            if (freePointers.size() == 0) {
              std::stringstream stream;
              stream << "No more free pointers "
                     << "for type " << type
                     << " and name " << name
                      ;
              throw std::domain_error(stream.str());
            }
            auto dataPointer = freePointers.begin();
            freePointers.erase(dataPointer);
            blank.data = *dataPointer;
          }

          result.push_back({name, blank.info});
          continue;
        }

      }

#ifdef HAVE_OCD
      for (auto const& s: slices)
        WITH_RANK << "__db__:guts:ocd:__end__ " << s.info << "\n";
#endif


      return result;

    }

    /*
     * Garbage collect slices not needed for the next iteration.
     *
     * It will throw if it tries to gc a slice that has not been
     * previously unwrapped, as a safety mechanism.
     */
    void clearUnusedSlicesForNext(ABCTuple const& abc) {
      auto const needed = neededSlices(abc);

      // CLEAN UP SLICES, FREE THE ONES THAT ARE NOT NEEDED ANYMORE
      for (auto& slice: slices) {
        // if the slice is free, then it was not used anyways
        if (slice.isFree()) continue;


        // try to find the slice in the needed slices list
        auto const found
          = std::find_if(needed.begin(), needed.end(),
                         [&slice] (Slice::Ty_x_Tu const& tytu) {
                           return slice.info.tuple == tytu.second
                               && slice.info.type == tytu.first
                               ;
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
          if (!slice.isUnwrapped() && slice.info.state != Slice::Recycled)
            throw
              std::domain_error("Trying to garbage collect "
                                " a non-unwrapped slice! "
                                + pretty_print(&slice)
                                + pretty_print(slice.info));

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
          if (slice.info.state == Slice::Ready) {
            WITH_OCD WITH_RANK
              << "__gc__:" << "checking for data recycled dependencies\n";
            auto recycled
              = Slice::hasRecycledReferencingToIt(slices, slice.info);
            if (recycled.size()) {
              Slice* newReady = recycled[0];
              WITH_OCD WITH_RANK
                << "__gc__:" << "swaping recycled "
                << pretty_print(newReady->info)
                << " and "
                << pretty_print(slice.info)
                << "\n";
              newReady->markReady();
              assert(newReady->data == slice.data);
              freeSlicePointer = false;

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

          // if the slice is self sufficient, do not dare touching the
          // pointer, since it is a pointer to our sources in our rank.
          if (  slice.info.state == Slice::SelfSufficient
             || slice.info.state == Slice::Recycled
             ) {
            freeSlicePointer = false;
          }

          // make sure we get its data pointer to be used later
          // only for non-recycled, since it can be that we need
          // for next iteration the data of the slice that the recycled points
          // to
          if (freeSlicePointer) {
            freePointers.insert(slice.data);
            WITH_RANK << "~~~:cl(" << name << ")"
                      << " added to freePointer "
                      << pretty_print(freePointers)
                      << "\n";
          } else {
            WITH_OCD WITH_RANK << "__gc__:not touching the free Pointer\n";
          }

          // at this point, let us blank the slice
          WITH_RANK << "~~~:cl(" << name << ")"
                    << " freeing up slice "
                    << " info " << slice.info
                    << "\n";
          slice.free();
        }

      }
    }

    // CONSTRUCTOR
    SliceUnion( Tensor const& sourceTensor
              , std::vector<Slice::Type> sliceTypes_
              , std::vector<size_t> sliceLength_
              , std::vector<size_t> paramLength
              , size_t np
              , MPI_Comm child_world
              , MPI_Comm global_world
              , Slice::Name name_
              , size_t nSliceBuffers = 4
              )
              : rankMap(paramLength, np, global_world)
              , world(child_world)
              , universe(global_world)
              , sliceLength(sliceLength_)
              , sources(rankMap.nSources(),
                        std::vector<F>
                          (std::accumulate(sliceLength.begin(),
                                           sliceLength.end(),
                                           1UL, std::multiplies<size_t>())))
              , name(name_)
              , sliceTypes(sliceTypes_)
              , sliceBuffers(nSliceBuffers, sources[0])
              //, slices(2 * sliceTypes.size(), Slice{ sources[0].size() })
    { // constructor begin

      LOG(0,"Atrip") << "INIT SliceUnion: " << name << "\n";

      slices
        = std::vector<Slice>(2 * sliceTypes.size(), { sources[0].size() });
      // TODO: think exactly ^------------------- about this number

      // initialize the freePointers with the pointers to the buffers
      std::transform(sliceBuffers.begin(), sliceBuffers.end(),
                     std::inserter(freePointers, freePointers.begin()),
                     [](std::vector<F> &vec) { return vec.data(); });



      LOG(1,"Atrip") << "rankMap.nSources "
                           << rankMap.nSources() << "\n";
      LOG(1,"Atrip") << "#slices "
                           << slices.size() << "\n";
      LOG(1,"Atrip") << "#slices[0] "
                           << slices[0].size << "\n";
      LOG(1,"Atrip") << "#sources "
                           << sources.size() << "\n";
      LOG(1,"Atrip") << "#sources[0] "
                           << sources[0].size() << "\n";
      LOG(1,"Atrip") << "#freePointers "
                           << freePointers.size() << "\n";
      LOG(1,"Atrip") << "#sliceBuffers "
                           << sliceBuffers.size() << "\n";
      LOG(1,"Atrip") << "#sliceBuffers[0] "
                           << sliceBuffers[0].size() << "\n";
      LOG(1,"Atrip") << "#sliceLength "
                           << sliceLength.size() << "\n";
      LOG(1,"Atrip") << "#paramLength "
                           << paramLength.size() << "\n";
      LOG(1,"Atrip") << "GB*" << np << " "
                           << double(sources.size() + sliceBuffers.size())
                            * sources[0].size()
                            * 8 * np
                            / 1073741824.0
                           << "\n";
    } // constructor ends

    void init(Tensor const& sourceTensor) {

      CTF::World w(world);
      const int rank = Atrip::rank
              , order = sliceLength.size()
              ;
      std::vector<int> const syms(order, NS);
      std::vector<int> __sliceLength(sliceLength.begin(), sliceLength.end());
      Tensor toSliceInto(order,
                         __sliceLength.data(),
                         syms.data(),
                         w);
      LOG(1,"Atrip") << "slicing... \n";

      // setUp sources
      for (size_t it(0); it < rankMap.nSources(); ++it) {
        const size_t
          source = rankMap.isSourcePadding(rank, source) ? 0 : it;
        WITH_OCD
        WITH_RANK
          << "Init:toSliceInto it-" << it
          << " :: source " << source << "\n";
        sliceIntoBuffer(source, toSliceInto, sourceTensor);
      }

    }

    /**
     * \brief Send asynchronously only if the state is Fetch
     */
    void send( size_t otherRank
             , Slice::LocalDatabaseElement const& el
             , size_t tag) const noexcept {
      MPI_Request request;
      bool sendData_p = false;
      auto const& info = el.info;

      if (info.state == Slice::Fetch) sendData_p = true;
      // TODO: remove this because I have SelfSufficient
      if (otherRank == info.from.rank)      sendData_p = false;
      if (!sendData_p) return;

      switch (el.name) {
        case Slice::Name::TA:
        case Slice::Name::VIJKA:
          if (otherRank / 48 == Atrip::rank / 48) {
            Atrip::localSend++;
          } else {
            Atrip::networkSend++;
          }
      }

      MPI_Isend( sources[info.from.source].data()
               , sources[info.from.source].size()
               , MPI_DOUBLE /* TODO: adapt this with traits */
               , otherRank
               , tag
               , universe
               , &request
               );
      WITH_CRAZY_DEBUG
      WITH_RANK << "sent to " << otherRank << "\n";

    }

    /**
     * \brief Receive asynchronously only if the state is Fetch
     */
    void receive(Slice::Info const& info, size_t tag) noexcept {
      auto& slice = Slice::findByInfo(slices, info);

      if (Atrip::rank == info.from.rank) return;

      if (slice.info.state == Slice::Fetch) {
        // TODO: do it through the slice class
        slice.info.state = Slice::Dispatched;
        MPI_Request request;
        slice.request = request;
        MPI_Irecv( slice.data
                 , slice.size
                 , MPI_DOUBLE // TODO: Adapt this with traits
                 , info.from.rank
                 , tag
                 , universe
                 , &slice.request
                //, MPI_STATUS_IGNORE
                 );
      }
    }

    void unwrapAll(ABCTuple const& abc) {
      for (auto type: sliceTypes) unwrapSlice(type, abc);
    }

    F* unwrapSlice(Slice::Type type, ABCTuple const& abc) {
      WITH_CRAZY_DEBUG
      WITH_RANK << "__unwrap__:slice " << type << " w n "
                << name
                << " abc" << pretty_print(abc)
                << "\n";
      auto& slice = Slice::findByTypeAbc(slices, type, abc);
      WITH_RANK << "__unwrap__:info " << slice.info << "\n";
      switch  (slice.info.state) {
        case Slice::Dispatched:
          WITH_RANK << "__unwrap__:Fetch: " << &slice
                    << " info " << pretty_print(slice.info)
                    << "\n";
          slice.unwrapAndMarkReady();
          return slice.data;
          break;
        case Slice::SelfSufficient:
          WITH_RANK << "__unwrap__:SelfSufficient: " << &slice
                    << " info " << pretty_print(slice.info)
                    << "\n";
          return slice.data;
          break;
        case Slice::Ready:
          WITH_RANK << "__unwrap__:READY: UNWRAPPED ALREADY" << &slice
                    << " info " << pretty_print(slice.info)
                    << "\n";
          return slice.data;
          break;
        case Slice::Recycled:
          WITH_RANK << "__unwrap__:RECYCLED " << &slice
                    << " info " << pretty_print(slice.info)
                    << "\n";
          return unwrapSlice(slice.info.recycling, abc);
          break;
        case Slice::Fetch:
        case Slice::Acceptor:
          throw std::domain_error("Can't unwrap an acceptor or fetch slice!");
          break;
        default:
          throw std::domain_error("Unknown error unwrapping slice!");
      }
      return slice.data;
    }

    const RankMap rankMap;
    const MPI_Comm world;
    const MPI_Comm universe;
    const std::vector<size_t> sliceLength;
    std::vector< std::vector<F> > sources;
    std::vector< Slice > slices;
    Slice::Name name;
    const std::vector<Slice::Type> sliceTypes;
    std::vector< std::vector<F> > sliceBuffers;
    std::set<F*> freePointers;

  };

  SliceUnion&
  unionByName(std::vector<SliceUnion*> const& unions, Slice::Name name) {
      const auto sliceUnionIt
        = std::find_if(unions.begin(), unions.end(),
                      [&name](SliceUnion const* s) {
                        return name == s->name;
                      });
      if (sliceUnionIt == unions.end()) {
        std::stringstream stream;
        stream << "SliceUnion(" << name << ") not found!";
        throw std::domain_error(stream.str());
      }
      return **sliceUnionIt;
  }

}
// The slice union:1 ends here
