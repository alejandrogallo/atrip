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

// [[file:../atrip.org::*Tests][Tests:1]]
#include <atrip.hpp>
#include <cassert>
// [[[[file:~/software/atrip/atrip.org::*Tests][Tests]]][]]
#include <atrip/Checkpoint.hpp>
#include <atrip/Operations.hpp>
using namespace atrip;
// ends here

#define TESTCASE(_name, ...)                                                   \
  {                                                                            \
    std::cout << "\x1b[35m-> \x1b[0m" << _name << std::endl;                   \
    __VA_ARGS__                                                                \
  }

int main() {

// [[[[file:~/software/atrip/atrip.org::*Tests][Tests]]][]]
#define _CMP_CHECK(what)                                                       \
  do {                                                                         \
    std::cout << "\t Checking " << #what << std::endl;                         \
    assert(in.what == what);                                                   \
    assert(out.what == what);                                                  \
  } while (0)

  TESTCASE(
      "Testing checkpoint reader and writers",
      const std::string out_checkpoint = "/tmp/checkpoint.yaml";
      const double energy = -1.493926352289995443;
      const size_t no = 154, nv = 1500, nranks = 48 * 10, nnodes = 10;
      const size_t iteration = 546;
      std::cout << "\twriting to " << out_checkpoint << std::endl;

      for (bool rank_round_robin
           : {true, false}) {
        atrip::Checkpoint
            out = {no, nv, nranks, nnodes, energy, iteration, rank_round_robin},
            in;

        write_checkpoint(out, out_checkpoint);
        in = read_checkpoint(out_checkpoint);

        _CMP_CHECK(no);
        _CMP_CHECK(nv);
        _CMP_CHECK(nranks);
        _CMP_CHECK(nnodes);
        _CMP_CHECK(iteration);
        _CMP_CHECK(rank_round_robin);
        _CMP_CHECK(energy);
      }

  )
#undef _CMP_CHECK

#define _CHECK(a, b)                                                           \
  do {                                                                         \
    std::cout << "\t that " << #a << " is equal to " << #b << "  (" << b       \
              << ")" << std::endl;                                             \
    assert(a == b);                                                            \
  } while (0)

  TESTCASE(
      "simple operations should work", const double a = 41.549, b = 1321.181;
      double aa = a, bb = b;
      _CHECK(atrip::acc::maybe_conjugate_scalar<double>(a), a);
      _CHECK(atrip::acc::prod<double>(a, b), a * b);
      // _CHECK(atrip::acc::real<double>(a), a);
      _CHECK(atrip::acc::sub<double>(a, b), a - b);
      _CHECK(atrip::acc::add<double>(a, b), a + b);
      atrip::acc::sum_in_place(&aa, &bb);
      _CHECK(aa, a + b);

      /*end*/)

#undef _CHECK

  // ends here

  return 0;
}
// Tests:1 ends here
