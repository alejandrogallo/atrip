// [[file:../../atrip.org::*Tuples][Tuples:1]]
#pragma once

#include <vector>
#include <array>
#include <numeric>

#include <atrip/Utils.hpp>
#include <atrip/Debug.hpp>

namespace atrip {

  using ABCTuple = std::array<size_t, 3>;
  using PartialTuple = std::array<size_t, 2>;
  using ABCTuples = std::vector<ABCTuple>;

  ABCTuples getTuplesList(size_t Nv) {
    const size_t n = Nv * (Nv + 1) * (Nv + 2) / 6 - Nv;
    ABCTuples result(n);
    size_t u(0);

    for (size_t a(0); a < Nv; a++)
    for (size_t b(a); b < Nv; b++)
    for (size_t c(b); c < Nv; c++){
      if ( a == b && b == c ) continue;
      result[u++] = {a, b, c};
    }

    return result;

  }


  std::pair<size_t, size_t>
  getABCRange(size_t np, size_t rank, ABCTuples const& tuplesList) {

    std::vector<size_t> n_tuples_per_rank(np, tuplesList.size()/np);
    const size_t
        // how many valid tuples should we still verteilen to nodes
        // since the number of tuples is not divisible by the number of nodes
        nRoundRobin = tuplesList.size() % np
        // every node must have the sanme amount of tuples in order for the
        // other nodes to receive and send somewhere, therefore
        // some nodes will get extra tuples but that are dummy tuples
      , nExtraInvalid = (np - nRoundRobin) % np
      ;

    if (nRoundRobin) for (int i = 0; i < np; i++) n_tuples_per_rank[i]++;

  #if defined(TODO)
    assert( tuplesList.size()
            ==
            ( std::accumulate(n_tuples_per_rank.begin(),
                              n_tuples_per_rank.end(),
                              0UL,
                              std::plus<size_t>())
            + nExtraInvalid
            ));
  #endif

    WITH_RANK << "nRoundRobin = " << nRoundRobin << "\n";
    WITH_RANK << "nExtraInvalid = " << nExtraInvalid << "\n";
    WITH_RANK << "ntuples = " << n_tuples_per_rank[rank] << "\n";

    auto const& it = n_tuples_per_rank.begin();

    return
      { std::accumulate(it, it + rank    , 0)
      , std::accumulate(it, it + rank + 1, 0)
      };

  }

}
// Tuples:1 ends here
