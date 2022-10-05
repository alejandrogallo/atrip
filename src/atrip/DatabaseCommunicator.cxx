#include <atrip/DatabaseCommunicator.hpp>
#include <atrip/Complex.hpp>


namespace atrip {

  static
  ABCTuples get_nth_naive_tuples(size_t Nv, size_t np) {

    const size_t
      // total number of tuples for the problem
      n = Nv * (Nv + 1) * (Nv + 2) / 6 - Nv

      // all ranks should have the same number of tuples_per_rank
      , tuples_per_rank = n / np + size_t(n % np != 0)
      ;


    ABCTuples result(np);

    for (size_t a(0), g(0); a < Nv; a++)
    for (size_t b(a);       b < Nv; b++)
    for (size_t c(b);       c < Nv; c++){
      if ( a == b && b == c ) continue;
      for (size_t rank = 0; rank < np; rank++) {

        const size_t
          // start index for the global tuples list
            start = tuples_per_rank * rank

          // end index for the global tuples list
          , end = tuples_per_rank * (rank + 1)
          ;

        if ( start <= g && g < end) result[rank] = {a, b, c};

      }
      g++;
    }

    return result;

  }


  template <typename F>
  static
  typename Slice<F>::LocalDatabase
  build_local_database_fake(ABCTuple const& abc_prev,
                            ABCTuple const& abc,
                            size_t rank,
                            SliceUnion<F>* u) {

    typename Slice<F>::LocalDatabase result;

    // vector of type x tuple
    auto const needed = u->neededSlices(abc);
    auto const needed_prev = u->neededSlices(abc_prev);

    for (auto const& pair: needed) {
      auto const type = pair.first;
      auto const tuple = pair.second;
      auto const from  = u->rankMap.find(abc, type);

      // Try to find in the previously needed slices
      // one that exactly matches the tuple.
      // Not necessarily has to match the type.
      //
      // If we find it, then it means that the fake rank
      // will mark it as recycled. This covers
      // the finding of Ready slices and Recycled slices.
      {
        auto const& it
          = std::find_if(needed_prev.begin(), needed_prev.end(),
                         [&tuple, &type](typename Slice<F>::Ty_x_Tu const& o) {
                           return o.second == tuple;
                         });

        if (it != needed_prev.end()) {
          typename Slice<F>::Info info;
          info.tuple = tuple;
          info.type = type;
          info.from = from;
          info.state = Slice<F>::Recycled;
          result.push_back({u->name, info});
          continue;
        }
      }

      {
        typename Slice<F>::Info info;
        info.type = type;
        info.tuple = tuple;
        info.from = from;

        // Handle self sufficiency
        info.state = rank == from.rank
                   ? Slice<F>::SelfSufficient
                   : Slice<F>::Fetch
                   ;
        result.push_back({u->name, info});
        continue;
      }

    }

    return result;

  }



  template <typename F>
  typename Slice<F>::Database
  naiveDatabase(Unions<F> &unions,
                size_t nv,
                size_t np,
                size_t iteration,
                MPI_Comm const& c) {

    using Database = typename Slice<F>::Database;
    Database db;
    const auto tuples = get_nth_naive_tuples(nv, np);
    const auto prev_tuples = get_nth_naive_tuples(nv, np);

    for (size_t rank = 0; rank < np; rank++) {
      auto abc = tuples[rank];
      typename Slice<F>::LocalDatabase ldb;

      for (auto const& tensor: unions) {
        if (rank == Atrip::rank) {
          auto const& tensorDb = tensor->buildLocalDatabase(abc);
          ldb.insert(ldb.end(), tensorDb.begin(), tensorDb.end());
        } else {
          auto const& tensorDb
            = build_local_database_fake(prev_tuples[rank],
                                        abc,
                                        rank,
                                        tensor);
          ldb.insert(ldb.end(), tensorDb.begin(), tensorDb.end());
        }
      }

      db.insert(db.end(), ldb.begin(), ldb.end());

    }

    return db;
  }

  template 
  typename Slice<double>::Database
  naiveDatabase<double>(Unions<double> &unions,
                size_t nv,
                size_t np,
                size_t iteration,
                MPI_Comm const& c);

  template 
  typename Slice<Complex>::Database
  naiveDatabase<Complex>(Unions<Complex> &unions,
                size_t nv,
                size_t np,
                size_t iteration,
                MPI_Comm const& c);

}  // namespace atrip
