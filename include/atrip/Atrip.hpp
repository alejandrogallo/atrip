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

// [[file:../../atrip.org::*Header][Header:1]]
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
    static size_t networkSend;
    static size_t localSend;
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

      ADD_ATTRIBUTE(bool, rankRoundRobin, false)
      ADD_ATTRIBUTE(bool, chrono, false)
      ADD_ATTRIBUTE(bool, barrier, false)
      ADD_ATTRIBUTE(int, maxIterations, 0)
      ADD_ATTRIBUTE(int, iterationMod, -1)
      ADD_ATTRIBUTE(int, percentageMod, -1)
      ADD_ATTRIBUTE(TuplesDistribution, tuplesDistribution, NAIVE)


    };

    struct Output {
      double energy;
    };
    static Output run(Input const& in);
  };

}

#undef ADD_ATTRIBUTE
// Header:1 ends here
