// [[file:~/cc4s/src/atrip/complex/atrip.org::*Header][Header:1]]
#pragma once
#include <sstream>
#include <string>
#include <map>
#include <chrono>

#include <ctf.hpp>

namespace atrip {

  struct Atrip {

    static int rank;
    static int np;
    static void init();

    template <typename F=double>
    struct Input {
      CTF::Tensor<F> *ei = nullptr
                        , *ea = nullptr
                        , *Tph = nullptr
                        , *Tpphh = nullptr
                        , *Vpphh = nullptr
                        , *Vhhhp = nullptr
                        , *Vppph = nullptr
                        ;
      int maxIterations = 0, iterationMod = -1, percentageMod = -1;
      bool barrier = false;
      bool chrono = false;
      Input& with_epsilon_i(CTF::Tensor<F> * t) { ei = t; return *this; }
      Input& with_epsilon_a(CTF::Tensor<F> * t) { ea = t; return *this; }
      Input& with_Tai(CTF::Tensor<F> * t) { Tph = t; return *this; }
      Input& with_Tabij(CTF::Tensor<F> * t) { Tpphh = t; return *this; }
      Input& with_Vabij(CTF::Tensor<F> * t) { Vpphh = t; return *this; }
      Input& with_Vijka(CTF::Tensor<F> * t) { Vhhhp = t; return *this; }
      Input& with_Vabci(CTF::Tensor<F> * t) { Vppph = t; return *this; }
      Input& with_maxIterations(int i) { maxIterations = i; return *this; }
      Input& with_iterationMod(int i) { iterationMod = i; return *this; }
      Input& with_percentageMod(int i) { percentageMod = i; return *this; }
      Input& with_barrier(bool i) { barrier = i; return *this; }
      Input& with_chrono(bool i) { chrono = i; return *this; }
    };

    struct Output {
      double energy;
    };
    template <typename F=double>
    static Output run(Input<F> const& in);
  };

}
// Header:1 ends here
