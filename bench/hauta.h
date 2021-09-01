#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include <iostream>

#define INSTANTIATE_OPTION(__type, __reader)          \
  template <> __type option(Args &as, Flag &f) {       \
    auto const v(option<std::string>(as, f));          \
    return __reader(v.c_str());                       \
  }

namespace hauta {
  typedef const std::string Flag;
  typedef std::string Arg;
  typedef const std::vector<Arg> Args;

  bool isFlagPresent(Args &a, Flag &f) {
    return std::find(a.begin(), a.end(), f) != a.end();
  }

  // option
  template<typename F> F option(Args &args, Flag &f);

  template<> std::string option(Args &a, Flag &f) {
    const auto it(std::find(a.begin(), a.end(), f));
    if (!isFlagPresent(a, f)) {
      std::cerr << "Expecting flag " << f << "\n";
      throw "";
    }
    return std::string(*(it+1));
  }

  INSTANTIATE_OPTION(size_t, std::atoi)
  INSTANTIATE_OPTION(int,    std::atoi)
  INSTANTIATE_OPTION(double, std::atof)
  INSTANTIATE_OPTION(float,  std::atof)

  template<> bool option(Args &a, Flag &f) { return isFlagPresent(a, f); }

  template<typename F> F option(Args &args, Flag &f, F const def) {
    return isFlagPresent(args, f)
         ? option<F>(args, f)
         : def
         ;
  }

  template<> bool option(Args &args, Flag &f, bool const def) {
    return isFlagPresent(args, f)
         ? !def
         : def
         ;
  }

  template <typename F>
  F option(int argc, char **argv, Flag& f, const F def) {
    return option<F>(Args {argv, argv + argc}, f, def);
  }
  template <typename F>
  F option(int argc, char **argv, Flag& f) {
    return option<F>(Args {argv, argv + argc}, f);
  }

}

#undef INSTANTIATE_OPTION
