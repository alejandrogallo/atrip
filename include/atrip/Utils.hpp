// [[file:../../atrip.org::*Utils][Utils:1]]
#pragma once
#include <sstream>
#include <string>
#include <map>
#include <chrono>

#include <ctf.hpp>

namespace atrip {


  template <typename T>
  std::string pretty_print(T&& value) {
    std::stringstream stream;
#if ATRIP_DEBUG > 1
    dbg::pretty_print(stream, std::forward<T>(value));
#endif
    return stream.str();
  }

#define WITH_CHRONO(__chrono, ...) \
  __chrono.start(); __VA_ARGS__ __chrono.stop();

  struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    using Event = std::chrono::time_point<Clock>;
    std::chrono::duration<double> duration;
    Event _start;
    inline void start() noexcept { _start = Clock::now(); }
    inline void stop() noexcept { duration += Clock::now() - _start; }
    inline void clear() noexcept { duration *= 0; }
    inline double count() const noexcept { return duration.count(); }
  };
  using Timings = std::map<std::string, Timer>;
}
// Utils:1 ends here

// [[file:../../atrip.org::*The rank mapping][The rank mapping:1]]
#pragma once

#include <vector>
#include <algorithm>

#include <atrip/Slice.hpp>

namespace atrip {
  struct RankMap {

    std::vector<size_t> const lengths;
    size_t const np, size;

    RankMap(std::vector<size_t> lens, size_t np_)
      : lengths(lens)
      , np(np_)
      , size(std::accumulate(lengths.begin(), lengths.end(),
                            1, std::multiplies<size_t>()))
    { assert(lengths.size() <= 2); }

    size_t find(Slice::Location const& p) const noexcept {
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

    Slice::Location
    find(ABCTuple const& abc, Slice::Type sliceType) const noexcept {
      // tuple = {11, 8} when abc = {11, 8, 9} and sliceType = AB
      const auto tuple = Slice::subtupleBySlice(abc, sliceType);

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
