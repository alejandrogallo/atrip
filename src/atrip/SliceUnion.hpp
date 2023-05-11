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
#include <set>
#include <unordered_set>

#include <atrip/Debug.hpp>
#include <atrip/Slice.hpp>
#include <atrip/RankMap.hpp>
#include <atrip/Utils.hpp>
#include <atrip/Malloc.hpp>

#if defined(ATRIP_SOURCES_IN_GPU)
#  define SOURCES_DATA(s) (s)
#else
#  define SOURCES_DATA(s) (s).data()
#endif

namespace atrip {
// Prolog:1 ends here

// [[file:~/cuda/atrip/atrip.org::*Prolog][Prolog:2]]
template <typename F = double>
class SliceUnion {
public:
  using Tensor = CTF::Tensor<F>;

#if defined(ATRIP_MPI_STAGING_BUFFERS)
  struct StagingBufferInfo {
    DataPtr<F> data;
    size_t tag;
    MPI_Request *request;
    ABCTuple const abc;

    bool operator==(StagingBufferInfo const &o) const {
      // TODO: think about this more carefully,
      //     can two staging buffers have the same data, it should not be
      //     the case
      return o.data == data;
    }
  };

  class StagingBufferInfoHash {
  public:
    size_t operator()(StagingBufferInfo const &i) const {
      return (size_t)i.data;
    }
  };
#endif /* defined(ATRIP_MPI_STAGING_BUFFERS) */

  virtual void
  slice_into_buffer(size_t iteration, Tensor &to, Tensor const &from) = 0;

  /*
   * This function should enforce an important property of a SliceUnion.
   * Namely, there can be no two Slices of the same nature.
   *
   * This means that there can be at most one slice with a given Ty_x_Tu.
   */
  void check_for_duplicates() const {
    std::vector<typename Slice<F>::Ty_x_Tu> tytus;
    for (auto const &s : slices) {
      if (s.is_free()) continue;
      tytus.push_back({s.info.type, s.info.tuple});
    }

    for (auto const &tytu : tytus) {
      if (std::count(tytus.begin(), tytus.end(), tytu) > 1)
        throw "Invariance violated, more than one slice with same Ty_x_Tu";
    }
  }

  std::vector<typename Slice<F>::Ty_x_Tu>
  needed_slices_for_tuple(ABCTuple const &abc) {
    std::vector<typename Slice<F>::Ty_x_Tu> needed(slice_types.size());
    // build the needed vector
    std::transform(slice_types.begin(),
                   slice_types.end(),
                   needed.begin(),
                   [&abc](typename Slice<F>::Type const type) {
                     auto tuple = Slice<F>::subtuple_by_slice(abc, type);
                     return std::make_pair(type, tuple);
                   });
    return needed;
  }

  /* build_local_database
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
  typename Slice<F>::LocalDatabase build_local_database(ABCTuple const &abc) {
    typename Slice<F>::LocalDatabase result;

    auto const needed = needed_slices_for_tuple(abc);

    WITH_RANK << "__db__:needed:" << pretty_print(needed) << "\n";
    // BUILD THE DATABASE
    // we need to loop over all slice_types that this TensorUnion
    // is representing and find out how we will get the corresponding
    // slice for the abc we are considering right now.
    for (auto const &pair : needed) {
      auto const type = pair.first;
      auto const tuple = pair.second;
      auto const from = rank_map.find(abc, type);

#ifdef HAVE_OCD
      WITH_RANK << "__db__:want:" << pretty_print(pair) << "\n";
      for (auto const &s : slices)
        WITH_RANK << "__db__:guts:ocd " << s.info << " pt " << s.data << "\n";
#endif

#ifdef HAVE_OCD
      WITH_RANK << "__db__: checking if exact match"
                << "\n";
#endif
      {
        // FIRST: look up if there is already a *Ready* slice matching what we
        // need
        auto const &it = std::find_if(slices.begin(),
                                      slices.end(),
                                      [&tuple, &type](Slice<F> const &other) {
                                        return other.info.tuple == tuple
                                            && other.info.type == type
                                            // we only want another slice when
                                            // it has already ready-to-use data
                                            && other.is_unwrappable();
                                      });
        if (it != slices.end()) {
          // if we find this slice, it means that we don't have to do anything
          WITH_RANK << "__db__: EXACT: found EXACT in name=" << name
                    << " for tuple " << tuple[0] << ", " << tuple[1] << " ptr "
                    << it->data << "\n";
          result.push_back({name, it->info});
          continue;
        }
      }

#ifdef HAVE_OCD
      WITH_RANK << "__db__: checking if recycle"
                << "\n";
#endif
      // Try to find a recyling possibility ie. find a slice with the same
      // tuple and that has a valid data pointer.
      auto const &recycle_it =
          std::find_if(slices.begin(),
                       slices.end(),
                       [&tuple, &type](Slice<F> const &other) {
                         return other.info.tuple == tuple
                             && other.info.type != type
                             && other.is_recyclable();
                       });

      // if we find this recylce, then we find a Blank slice
      // (which should exist by construction :THINK)
      //
      if (recycle_it != slices.end()) {
        auto &blank = Slice<F>::find_one_by_type(slices, Slice<F>::Blank);
        // TODO: formalize this through a method to copy information
        //       from another slice
        blank.data = recycle_it->data;
        blank.info.type = type;
        blank.info.tuple = tuple;
        blank.info.state = Slice<F>::Recycled;
        blank.info.from = from;
        blank.info.recycling = recycle_it->info.type;
        result.push_back({name, blank.info});
        WITH_RANK << "__db__: RECYCLING: n" << name << " " << pretty_print(abc)
                  << " get " << pretty_print(blank.info) << " from "
                  << pretty_print(recycle_it->info) << " ptr "
                  << recycle_it->data << "\n";
        continue;
      }

      // in this case we have to create a new slice
      // this means that we should have a blank slice at our disposal
      // and also the free_pointers should have some elements inside,
      // so we pop a data pointer from the free_pointers container
#ifdef HAVE_OCD
      WITH_RANK << "__db__: none work, doing new"
                << "\n";
#endif
      {
        WITH_RANK << "__db__: NEW: finding blank in " << name << " for type "
                  << type << " for tuple " << tuple[0] << ", " << tuple[1]
                  << "\n";
        auto &blank = Slice<F>::find_one_by_type(slices, Slice<F>::Blank);
        blank.info.type = type;
        blank.info.tuple = tuple;
        blank.info.from = from;

        // Handle self sufficiency
        blank.info.state = Atrip::rank == from.rank ? Slice<F>::SelfSufficient
                                                    : Slice<F>::Fetch;
        if (blank.info.state == Slice<F>::SelfSufficient) {
#if defined(HAVE_CUDA) && !defined(ATRIP_SOURCES_IN_GPU)
          const size_t _size = sizeof(F) * slice_size;
          blank.data = pop_free_pointers();
          WITH_CHRONO("cuda:memcpy",
                      WITH_CHRONO("cuda:memcpy:self-sufficient",
                                  _CHECK_CUDA_SUCCESS(
                                      "copying mpi data to device",
                                      cuMemcpyHtoD(blank.data,
                                                   (void *)SOURCES_DATA(
                                                       sources[from.source]),
                                                   sizeof(F) * slice_size));))
#else
          blank.data = SOURCES_DATA(sources[from.source]);
#endif
        } else {
          blank.data = pop_free_pointers();
        }

        result.push_back({name, blank.info});
        continue;
      }
    }

#ifdef HAVE_OCD
    for (auto const &s : slices)
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
  void clear_unused_slices_for_next_tuple(ABCTuple const &abc) {
    auto const needed = needed_slices_for_tuple(abc);

    // CLEAN UP SLICES, FREE THE ONES THAT ARE NOT NEEDED ANYMORE
    for (auto &slice : slices) {
      // if the slice is free, then it was not used anyways
      if (slice.is_free()) continue;

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
        bool free_slice_pointer = true;

        // allow to gc unwrapped and recycled, never Fetch,
        // if we have a Fetch slice then something has gone very wrong.
        if (!slice.is_unwrapped() && slice.info.state != Slice<F>::Recycled)
          throw std::domain_error(
              "Trying to garbage collect "
              " a non-unwrapped slice! "
              + pretty_print(&slice) + pretty_print(slice.info));

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
          WITH_OCD WITH_RANK << "__gc__:"
                             << "checking for data recycled dependencies\n";
          auto recycled =
              Slice<F>::has_recycled_referencing_to_it(slices, slice.info);
          if (recycled.size()) {
            Slice<F> *new_ready = recycled[0];
            WITH_OCD WITH_RANK << "__gc__:"
                               << "swaping recycled "
                               << pretty_print(new_ready->info) << " and "
                               << pretty_print(slice.info) << "\n";
            new_ready->mark_ready();
            assert(new_ready->data == slice.data);
            free_slice_pointer = false;

            for (size_t i = 1; i < recycled.size(); i++) {
              auto new_recycled = recycled[i];
              new_recycled->info.recycling = new_ready->info.type;
              WITH_OCD WITH_RANK << "__gc__:"
                                 << "updating recycled "
                                 << pretty_print(new_recycled->info) << "\n";
            }
          }
        }

#if defined(HAVE_CUDA) && !defined(ATRIP_SOURCES_IN_GPU)
        // In cuda, SelfSufficient slices have an ad-hoc pointer
        // since it is a pointer on the device and has to be
        // brought back to the free pointer bucket of the SliceUnion.
        // Therefore, only in the Recycled case it should not be
        // put back the pointer.
        if (slice.info.state == Slice<F>::Recycled) {
          free_slice_pointer = false;
        }
#else
        // if the slice is self sufficient, do not dare touching the
        // pointer, since it is a pointer to our sources in our rank.
        if (slice.info.state == Slice<F>::SelfSufficient
            || slice.info.state == Slice<F>::Recycled) {
          free_slice_pointer = false;
        }
#endif

        // make sure we get its data pointer to be used later
        // only for non-recycled, since it can be that we need
        // for next iteration the data of the slice that the recycled points
        // to
        if (free_slice_pointer) {
          free_pointers.insert(slice.data);
          WITH_RANK << "~~~:cl(" << name << ")"
                    << " added to freePointer " << pretty_print(free_pointers)
                    << "\n";
        } else {
          WITH_OCD WITH_RANK << "__gc__:not touching the free Pointer\n";
        }

        // at this point, let us blank the slice
        WITH_RANK << "~~~:cl(" << name << ")"
                  << " freeing up slice "
                  // TODO: make this possible because of Templates
                  // TODO: there is a deduction error here
                  // << " info " << slice.info
                  << "\n";
        slice.free();
      } // we did not find the slice
    }
  }

  static size_t get_size(const std::vector<size_t> slice_length,
                         const std::vector<size_t> param_length,
                         const size_t np,
                         const MPI_Comm global_world) {
    const RankMap<F> rank_map(param_length, np);
    const size_t n_sources = rank_map.n_sources(),
                 slice_size = std::accumulate(slice_length.begin(),
                                              slice_length.end(),
                                              1UL,
                                              std::multiplies<size_t>());
    return n_sources * slice_size;
  }

  // CONSTRUCTOR
  SliceUnion(std::vector<typename Slice<F>::Type> slice_types_,
             std::vector<size_t> slice_length_,
             std::vector<size_t> param_length,
             size_t np,
             MPI_Comm child_world,
             MPI_Comm global_world,
             typename Slice<F>::Name name_,
             size_t n_slice_buffers = 4)
      : rank_map(param_length, np)
      , world(child_world)
      , universe(global_world)
      , slice_length(slice_length_)
      , slice_size(std::accumulate(slice_length.begin(),
                                   slice_length.end(),
                                   1UL,
                                   std::multiplies<size_t>()))

#if defined(ATRIP_SOURCES_IN_GPU)
      , sources(rank_map.n_sources())
#else
      , sources(rank_map.n_sources(),
#  if defined(ATRIP_DRY)
                std::vector<F>(1))
#  else
                std::vector<F>(slice_size))
#  endif /* defined(ATRIP_DRY) */
#endif   /* defined(ATRIP_SOURCES_IN_GPU) */
      , name(name_)
      , slice_types(slice_types_)
      , slice_buffers(n_slice_buffers) { // constructor begin

#if defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_CUDA)
    for (auto &ptr : sources) {
      MALLOC_DATA_POINTER("SOURCES", &ptr, sizeof(F) * slice_size);
    }
#endif

    for (auto &ptr : slice_buffers) {
      MALLOC_DATA_PTR("Slice Buffer", &ptr, sizeof(F) * slice_size);
    }

    slices = std::vector<Slice<F>>(2 * slice_types.size(), {slice_size});
    // TODO: think exactly    ^------------------- about this number

    // initialize the free_pointers with the pointers to the buffers
    std::transform(slice_buffers.begin(),
                   slice_buffers.end(),
                   std::inserter(free_pointers, free_pointers.begin()),
                   [](DataPtr<F> ptr) { return ptr; });

#if defined(HAVE_CUDA)
    LOG(1, "Atrip") << "warming communication up " << slices.size() << "\n";
    WITH_CHRONO(
        "cuda:warmup", int n_ranks = Atrip::np, request_count = 0;
        int n_sends = slice_buffers.size() * n_ranks;
        MPI_Request *requests =
            (MPI_Request *)malloc(n_sends * 2 * sizeof(MPI_Request));
        MPI_Status *statuses =
            (MPI_Status *)malloc(n_sends * 2 * sizeof(MPI_Status));
        for (int slice_id = 0; slice_id < slice_buffers.size(); slice_id++) {
          for (int rank_id = 0; rank_id < n_ranks; rank_id++) {
            MPI_Isend((void *)SOURCES_DATA(sources[0]),
                      slice_size,
                      traits::mpi::datatype_of<F>(),
                      rank_id,
                      100,
                      universe,
                      &requests[request_count++]);
            MPI_Irecv((void *)slice_buffers[slice_id],
                      slice_size,
                      traits::mpi::datatype_of<F>(),
                      rank_id,
                      100,
                      universe,
                      &requests[request_count++]);
          }
        } MPI_Waitall(n_sends * 2, requests, statuses);
        free(requests);
        free(statuses);)
#endif

    LOG(1, "Atrip") << "#slices " << slices.size() << "\n";
    WITH_RANK << "#slices[0] " << slices[0].size << "\n";
    LOG(1, "Atrip") << "#sources " << sources.size() << "\n";
    WITH_RANK << "#sources[0] " << slice_size << "\n";
    WITH_RANK << "#free_pointers " << free_pointers.size() << "\n";
    LOG(1, "Atrip") << "#slice_buffers " << slice_buffers.size() << "\n";
    LOG(1, "Atrip") << "GB*" << np << " "
                    << double(sources.size() + slice_buffers.size())
                           * slice_size * 8 * np / 1073741824.0
                    << "\n";
  } // constructor ends

  void init(Tensor const &source_tensor) {

    CTF::World w(world);
    const int rank = Atrip::rank;
#if defined(ATRIP_DRY)
    const int order = 0;
#else
    const int order = slice_length.size();
#endif /* defined(ATRIP_DRY) */
    std::vector<int> const syms(order, NS);
    std::vector<int> __slice_length(slice_length.begin(), slice_length.end());
    Tensor to_slice_into(order, __slice_length.data(), syms.data(), w);

    WITH_OCD WITH_RANK << "slicing... \n";

    // setUp sources
    size_t last_source = 0;
    for (size_t it(0); it < rank_map.n_sources(); ++it) {
      const size_t source =
          rank_map.is_source_padding(rank, last_source) ? 0 : it;
      WITH_OCD
      WITH_RANK << "Init:to_slice_into it-" << it << " :: source " << source
                << "\n";
      slice_into_buffer(source, to_slice_into, source_tensor);
      last_source = source;
    }
  }

  /*
   */
  void allocate_free_buffer() {
    DataPtr<F> new_pointer;
#if defined(HAVE_CUDA) && defined(ATRIP_SOURCES_IN_GPU)
    MALLOC_DATA_PTR("Additional free buffer",
                    &new_pointer,
                    sizeof(DataFieldType<F>) * slice_size);
#else
    MALLOC_HOST_DATA("Additional free buffer",
                     &new_pointer,
                     sizeof(DataFieldType<F>) * slice_size);
#endif
    free_pointers.insert(new_pointer);
  }

  DataPtr<F> pop_free_pointers() {
    if (free_pointers.size() == 0) {
#if defined(ATRIP_ALLOCATE_ADDITIONAL_FREE_POINTERS)
      allocate_free_buffer();
#else
      throw _FORMAT("No more free pointers for name %s",
                    name_to_string<F>(name).c_str());
#endif /* defined(ATRIP_ALLOCATE_ADDITIONAL_FREE_POINTERS) */
    }
    auto data_pointer_it = free_pointers.begin();
    auto data_pointer = *data_pointer_it;
    free_pointers.erase(data_pointer_it);
    return data_pointer;
  }

  /**
   * \brief Send asynchronously only if the state is Fetch
   */
  void send(size_t other_rank,
            typename Slice<F>::LocalDatabaseElement const &el,
            size_t tag,
            ABCTuple abc) {
    IGNORABLE(abc); // used for mpi staging only
    MPI_Request *request = (MPI_Request *)malloc(sizeof(MPI_Request));
    bool send_data_p = false;
    auto const &info = el.info;

    if (info.state == Slice<F>::Fetch) send_data_p = true;
    // TODO: remove this because I have SelfSufficient
    if (other_rank == info.from.rank) send_data_p = false;
    if (!send_data_p) return;

#if defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_CUDA)
    DataPtr<const F> source_buffer = SOURCES_DATA(sources[info.from.source]);
#else
    F *source_buffer = SOURCES_DATA(sources[info.from.source]);
#endif

    // SECTION TO START DOING THE STAGING BUFFERS
    //
    // in general it has to check whether or not the receiving buffer
    // is on a different node or on the same node.
    //
    // Only a staging buffer will be allocated if the communication
    // is to happen in an INTER-node manner.

    size_t target_node = Atrip::cluster_info->rank_infos[other_rank].node_id,
           from_node = Atrip::cluster_info->rank_infos[info.from.rank].node_id;
    const bool inter_node_communication = target_node == from_node;

    if (inter_node_communication) { goto no_mpi_staging; }

#if defined(ATRIP_MPI_STAGING_BUFFERS)
    DataPtr<F> isend_buffer = pop_free_pointers();

#  if defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_CUDA)
    WITH_CHRONO(
        "cuda:memcpy",
        WITH_CHRONO("cuda:memcpy:staging",
                    _CHECK_CUDA_SUCCESS("copying to staging buffer",
                                        cuMemcpy(isend_buffer,
                                                 source_buffer,
                                                 sizeof(F) * slice_size));))

#  elif !defined(HAVE_CUDA)
    // do cpu mpi memory staging
    WITH_CHRONO(
        "memcpy",
        WITH_CHRONO(
            "memcpy:staging",
            memcpy(isend_buffer, source_buffer, sizeof(F) * slice_size);))

#  else
#    pragma error("Not possible to do MPI_STAGING_BUFFERS with your config.h")
#  endif /* defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_CUDA) */

    goto mpi_staging_done;

#else

    // otherwise the isend_buffer will be the source buffer itself
    goto no_mpi_staging;

#endif /* defined(ATRIP_MPI_STAGING_BUFFERS) */

  no_mpi_staging:

#if defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_CUDA)
    DataPtr<F> isend_buffer = source_buffer;
#else
    F *isend_buffer = source_buffer;
#endif /* defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_CUDA) */

    goto mpi_staging_done; // do not complain that is not used, it is used when
                           // MPI_STAGING_BUFFERS is on
  mpi_staging_done:

    // We count network sends only for the largest buffers
    switch (el.name) {
    case Slice<F>::Name::TA:
      if (other_rank / Atrip::ppn == Atrip::rank / Atrip::ppn) {
        Atrip::local_send++;
      } else {
        Atrip::network_send++;
      }
    default:;
    }
    Atrip::bytes_sent += slice_size * sizeof(F);
    MPI_Isend((void *)isend_buffer,
              slice_size,
              traits::mpi::datatype_of<F>(),
              other_rank,
              tag,
              universe,
              request);
    WITH_CRAZY_DEBUG
    WITH_RANK << "sent to " << other_rank << "\n";

#if defined(ATRIP_MPI_STAGING_BUFFERS)
    if (!inter_node_communication)
      mpi_staging_buffers.insert(
          StagingBufferInfo{isend_buffer, tag, request, abc});
    else free(request);
#else
    free(request);
#endif /* defined(ATRIP_MPI_STAGING_BUFFERS) */
  }

  /**
   * \brief Receive asynchronously only if the state is Fetch
   */
  void receive(typename Slice<F>::Info const &info, size_t tag) noexcept {
    auto &slice = Slice<F>::find_by_info(slices, info);

    if (Atrip::rank == info.from.rank) return;

    if (slice.info.state == Slice<F>::Fetch) { // if-1
      // TODO: do it through the slice class
      slice.info.state = Slice<F>::Dispatched;
#if defined(HAVE_CUDA) && defined(ATRIP_SOURCES_IN_GPU)
#  if !defined(ATRIP_CUDA_AWARE_MPI)
#    error "You need CUDA aware MPI to have slices on the GPU"
#  endif
      MPI_Irecv((void *)slice.data,
#elif defined(HAVE_CUDA) && !defined(ATRIP_SOURCES_IN_GPU)
      slice.mpi_data = (F *)malloc(sizeof(F) * slice.size);
      MPI_Irecv(slice.mpi_data,
#else
      MPI_Irecv((void *)slice.data,
#endif
                slice.size,
                traits::mpi::datatype_of<F>(),
                info.from.rank,
                tag,
                universe,
                &slice.request);
    } // if-1
  }   // receive

  void unwrap_all(ABCTuple const &abc) {
    for (auto type : slice_types) unwrap_slice(type, abc);
  }

  DataPtr<F> unwrap_slice(typename Slice<F>::Type type, ABCTuple const &abc) {
    WITH_CRAZY_DEBUG
    WITH_RANK << "__unwrap__:slice " << type << " w n " << name << " abc"
              << pretty_print(abc) << "\n";
    auto &slice = Slice<F>::find_type_abc(slices, type, abc);
    // WITH_RANK << "__unwrap__:info " << slice.info << "\n";
    switch (slice.info.state) {
    case Slice<F>::Dispatched:
      WITH_RANK << "__unwrap__:Fetch: " << &slice << " info "
                << pretty_print(slice.info) << "\n";
      slice.unwrap_and_mark_ready();
      return slice.data;
      break;
    case Slice<F>::SelfSufficient:
      WITH_RANK << "__unwrap__:SelfSufficient: " << &slice << " info "
                << pretty_print(slice.info) << "\n";
      return slice.data;
      break;
    case Slice<F>::Ready:
      WITH_RANK << "__unwrap__:READY: UNWRAPPED ALREADY" << &slice << " info "
                << pretty_print(slice.info) << "\n";
      return slice.data;
      break;
    case Slice<F>::Recycled:
      WITH_RANK << "__unwrap__:RECYCLED " << &slice << " info "
                << pretty_print(slice.info) << "\n";
      return unwrap_slice(slice.info.recycling, abc);
      break;
    case Slice<F>::Fetch:
    case Slice<F>::Acceptor:
      throw std::domain_error("Can't unwrap an acceptor or fetch slice!");
      break;
    default: throw std::domain_error("Unknown error unwrapping slice!");
    }
    return slice.data;
  }

  // DESTRUCTOR

  ~SliceUnion() {
    for (auto &ptr : slice_buffers)
#if defined(HAVE_CUDA)
      cuMemFree(ptr);
#else
      std::free(ptr);
#endif
  }

  const RankMap<F> rank_map;
  const MPI_Comm world;
  const MPI_Comm universe;
  const std::vector<size_t> slice_length;
  const size_t slice_size;
#if defined(ATRIP_SOURCES_IN_GPU)
  std::vector<DataPtr<F>> sources;
#else
  std::vector<std::vector<F>> sources;
#endif
  std::vector<Slice<F>> slices;
  typename Slice<F>::Name name;
  const std::vector<typename Slice<F>::Type> slice_types;
  std::vector<DataPtr<F>> slice_buffers;
  std::set<DataPtr<F>> free_pointers;
#if defined(ATRIP_MPI_STAGING_BUFFERS)
  std::unordered_set<StagingBufferInfo, StagingBufferInfoHash>
      mpi_staging_buffers;
#endif
};

template <typename F = double>
SliceUnion<F> &union_by_name(std::vector<SliceUnion<F> *> const &unions,
                             typename Slice<F>::Name name) {
  const auto slice_union_it =
      std::find_if(unions.begin(),
                   unions.end(),
                   [&name](SliceUnion<F> const *s) { return name == s->name; });
  if (slice_union_it == unions.end()) {
    std::stringstream stream;
    stream << "SliceUnion(" << name << ") not found!";
    throw std::domain_error(stream.str());
  }
  return **slice_union_it;
}
// Prolog:2 ends here

// [[file:~/cuda/atrip/atrip.org::*Epilog][Epilog:1]]
} // namespace atrip
// Epilog:1 ends here
