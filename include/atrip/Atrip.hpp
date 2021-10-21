// [[file:~/atrip/atrip.org::*Atrip][Atrip:1]]
#pragma once
#include <sstream>
#include <string>
#include <map>

#include <ctf.hpp>

#include <atrip/Utils.hpp>

#define ADD_ATTRIBUTE(_type, _name, _default)   \
  _type _name = _default;                       \
  Input& with_ ## _name(_type i) {              \
    _name = i;                                  \
    return *this;                               \
  }

namespace atrip {

  struct Atrip {

    static int rank;
    static int np;
    static Timings chrono;
    static void init();

    struct Input {
      CTF::Tensor<double> *ei = nullptr
                        , *ea = nullptr
                        , *Tph = nullptr
                        , *Tpphh = nullptr
                        , *Vpphh = nullptr
                        , *Vhhhp = nullptr
                        , *Vppph = nullptr
                        ;
      Input& with_epsilon_i(CTF::Tensor<double> * t) { ei = t; return *this; }
      Input& with_epsilon_a(CTF::Tensor<double> * t) { ea = t; return *this; }
      Input& with_Tai(CTF::Tensor<double> * t) { Tph = t; return *this; }
      Input& with_Tabij(CTF::Tensor<double> * t) { Tpphh = t; return *this; }
      Input& with_Vabij(CTF::Tensor<double> * t) { Vpphh = t; return *this; }
      Input& with_Vijka(CTF::Tensor<double> * t) { Vhhhp = t; return *this; }
      Input& with_Vabci(CTF::Tensor<double> * t) { Vppph = t; return *this; }

      enum TuplesDistribution {
        NAIVE,
        GROUP_AND_SORT,
      };

      ADD_ATTRIBUTE(bool, chrono, false)
      ADD_ATTRIBUTE(bool, barrier, false)
      ADD_ATTRIBUTE(int, maxIterations, 0)
      ADD_ATTRIBUTE(int, iterationMod, -1)
      ADD_ATTRIBUTE(TuplesDistribution, tuplesDistribution, NAIVE)


    };

    struct Output {
      double energy;
    };
    static Output run(Input const& in);
  };

}

#undef ADD_ATTRIBUTE
// Atrip:1 ends here
