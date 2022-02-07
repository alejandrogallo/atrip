// [[file:~/cc4s/src/atrip/complex/atrip.org::*The%20rank%20mapping][The rank mapping:1]]
#pragma once

#include <vector>
#include <algorithm>

#include <atrip/Slice.hpp>

namespace atrip {

  template <typename F=double>
  struct RankMap {

    std::vector<size_t> const lengths;
    size_t const np, size;

    RankMap(std::vector<size_t> lens, size_t np_)
      : lengths(lens)
      , np(np_)
      , size(std::accumulate(lengths.begin(), lengths.end(),
                            1UL, std::multiplies<size_t>()))
    { assert(lengths.size() <= 2); }

    size_t find(typename Slice<F>::Location const& p) const noexcept {
      return p.source * np + p.rank;
    }

    size_t nSources() const noexcept {
      return size / np + size_t(size % np != 0);
    }


    bool isPaddingRank(size_t rank) const noexcept {
      return size % np == 0
          ? false
          : rank > (size % np - 1)
          ;
    }

    bool isSourcePadding(size_t rank, size_t source) const noexcept {
      return source == nSources() && isPaddingRank(rank);
    }

    typename Slice<F>::Location
    find(ABCTuple const& abc, typename Slice<F>::Type sliceType) const noexcept {
      // tuple = {11, 8} when abc = {11, 8, 9} and sliceType = AB
      const auto tuple = Slice<F>::subtupleBySlice(abc, sliceType);

      const size_t index
        = tuple[0]
        + tuple[1] * (lengths.size() > 1 ? lengths[0] : 0)
        ;

      return
        { index % np
        , index / np
        };
    }

  };

}
// The rank mapping:1 ends here
