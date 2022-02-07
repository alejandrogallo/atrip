// [[file:~/cc4s/src/atrip/complex/atrip.org::*Utils][Utils:1]]
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
