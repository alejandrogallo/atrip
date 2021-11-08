// [[file:~/atrip/atrip.org::*Prolog][Prolog:1]]
#pragma once

#include <vector>
#include <array>
#include <numeric>

// TODO: remove some
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <cassert>
#include <chrono>
#include <climits>
#include <mpi.h>

#include <atrip/Utils.hpp>
#include <atrip/Debug.hpp>

namespace atrip {
// Prolog:1 ends here

// [[file:~/atrip/atrip.org::*Tuples%20types][Tuples types:1]]
using ABCTuple = std::array<size_t, 3>;
using PartialTuple = std::array<size_t, 2>;
using ABCTuples = std::vector<ABCTuple>;

constexpr ABCTuple FAKE_TUPLE = {0, 0, 0};
// Tuples types:1 ends here

// [[file:~/atrip/atrip.org::*Distributing%20the%20tuples][Distributing the tuples:1]]
struct TuplesDistribution {
  virtual ABCTuples getTuples(size_t Nv, MPI_Comm universe) = 0;
  virtual bool tupleIsFake(ABCTuple const& t) { return t == FAKE_TUPLE; }
};
// Distributing the tuples:1 ends here

// [[file:~/atrip/atrip.org::*Node%20information][Node information:1]]
std::vector<std::string> getNodeNames(MPI_Comm comm){
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);

  std::vector<std::string> nodeList(np);
  char nodeName[MPI_MAX_PROCESSOR_NAME]
     , nodeNames[np*MPI_MAX_PROCESSOR_NAME]
     ;
  std::vector<int> nameLengths(np)
                 , off(np)
                 ;
  int nameLength;
  MPI_Get_processor_name(nodeName, &nameLength);
  MPI_Allgather(&nameLength,
                1,
                MPI_INT,
                nameLengths.data(),
                1,
                MPI_INT,
                comm);
  for (int i(1); i < np; i++)
    off[i] = off[i-1] + nameLengths[i-1];
  MPI_Allgatherv(nodeName,
                 nameLengths[rank],
                 MPI_BYTE,
                 nodeNames,
                 nameLengths.data(),
                 off.data(),
                 MPI_BYTE,
                 comm);
  for (int i(0); i < np; i++) {
    std::string const s(&nodeNames[off[i]], nameLengths[i]);
    nodeList[i] = s;
  }
  return nodeList;
}
// Node information:1 ends here

// [[file:~/atrip/atrip.org::*Node%20information][Node information:2]]
struct RankInfo {
  const std::string name;
  const size_t nodeId;
  const size_t globalRank;
  const size_t localRank;
  const size_t ranksPerNode;
};

template <typename A>
std::vector<A> unique(std::vector<A> const &xs) {
  auto result = xs;
  std::sort(result.begin(), result.end());
  auto const& last = std::unique(result.begin(), result.end());
  result.erase(last, result.end());
  return result;
}

std::vector<RankInfo>
getNodeInfos(std::vector<string> const& nodeNames) {
  std::vector<RankInfo> result;
  auto const uniqueNames = unique(nodeNames);
  auto const index = [&uniqueNames](std::string const& s) {
    auto const& it = std::find(uniqueNames.begin(), uniqueNames.end(), s);
    return std::distance(uniqueNames.begin(), it);
  };
  std::vector<size_t> localRanks(uniqueNames.size(), 0);
  size_t globalRank = 0;
  for (auto const& name: nodeNames) {
    const size_t nodeId = index(name);
    result.push_back({name,
                      nodeId,
                      globalRank++,
                      localRanks[nodeId]++,
                      std::count(nodeNames.begin(),
                                 nodeNames.end(),
                                 name)
                      });
  }
  return result;
}

struct ClusterInfo {
  const size_t nNodes, np, ranksPerNode;
  const std::vector<RankInfo> rankInfos;
};

ClusterInfo
getClusterInfo(MPI_Comm comm) {
  auto const names = getNodeNames(comm);
  auto const rankInfos = getNodeInfos(names);

  return ClusterInfo {
    unique(names).size(),
    names.size(),
    rankInfos[0].ranksPerNode,
    rankInfos
  };

}
// Node information:2 ends here

// [[file:~/atrip/atrip.org::*Naive%20list][Naive list:1]]
ABCTuples getTuplesList(size_t Nv, size_t rank, size_t np) {

  const size_t
    // total number of tuples for the problem
       n = Nv * (Nv + 1) * (Nv + 2) / 6 - Nv

    // all ranks should have the same number of tuples_per_rank
    , tuples_per_rank = n / np + size_t(n % np != 0)

    // start index for the global tuples list
    , start = tuples_per_rank * rank

    // end index for the global tuples list
    , end = tuples_per_rank * (rank + 1)
    ;

  LOG(1,"Atrip") << "tuples_per_rank = " << tuples_per_rank << "\n";
  WITH_RANK << "start, end = " << start << ", " << end << "\n";
  ABCTuples result(tuples_per_rank, FAKE_TUPLE);

  for (size_t a(0), r(0), g(0); a < Nv; a++)
  for (size_t b(a);             b < Nv; b++)
  for (size_t c(b);             c < Nv; c++){
    if ( a == b && b == c ) continue;
    if ( start <= g && g < end) result[r++] = {a, b, c};
    g++;
  }

  return result;

}
// Naive list:1 ends here

// [[file:~/atrip/atrip.org::*Naive%20list][Naive list:2]]
ABCTuples getAllTuplesList(const size_t Nv) {
  const size_t n = Nv * (Nv + 1) * (Nv + 2) / 6 - Nv;
  ABCTuples result(n);

  for (size_t a(0), u(0); a < Nv; a++)
  for (size_t b(a); b < Nv; b++)
  for (size_t c(b); c < Nv; c++){
    if ( a == b && b == c ) continue;
    result[u++] = {a, b, c};
  }

  return result;
}
// Naive list:2 ends here

// [[file:~/atrip/atrip.org::*Naive%20list][Naive list:3]]
struct NaiveDistribution : public TuplesDistribution {
  ABCTuples getTuples(size_t Nv, MPI_Comm universe) override {
    int rank, np;
    MPI_Comm_rank(universe, &rank);
    MPI_Comm_size(universe, &np);
    return getTuplesList(Nv, (size_t)rank, (size_t)np);
  }
};
// Naive list:3 ends here

// [[file:~/atrip/atrip.org::*Prolog][Prolog:1]]
namespace group_and_sort {
// Prolog:1 ends here

// [[file:~/atrip/atrip.org::*Utils][Utils:1]]
// Provides the node on which the slice-element is found
// Right now we distribute the slices in a round robin fashion
// over the different nodes (NOTE: not mpi ranks but nodes)
size_t isOnNode(size_t tuple, size_t nodes) { return tuple % nodes; }


struct Info {
  size_t nNodes;
  size_t nodeId;
};


// return the node (or all nodes) where the elements of this
// tuple are located
std::vector<size_t> getTupleNodes(ABCTuple t, size_t nNodes) {
  std::vector<size_t> result;
  ABCTuple nTuple = { isOnNode(t[0], nNodes)
                    , isOnNode(t[1], nNodes)
                    , isOnNode(t[2], nNodes)
                    };
  std::sort(nTuple.begin(), nTuple.end());
  ABCTuple::iterator it = std::unique(nTuple.begin(), nTuple.end());
  result.resize(it - nTuple.begin());
  std::copy(nTuple.begin(), it, result.begin());
  return result;
}
// Utils:1 ends here

// [[file:~/atrip/atrip.org::*Distribution][Distribution:1]]
ABCTuples specialDistribution(Info const& info, ABCTuples const& allTuples) {

  ABCTuples nodeTuples;
  size_t const nNodes(info.nNodes);

  std::map< size_t /* nodeId */, ABCTuples >
    container1d, container2d, container3d;

  // build container-n-d's
  for (auto const& t: allTuples) {
    // one which node(s) are the tuple elements located...
    // put them into the right container
    auto const _nodes = getTupleNodes(t, nNodes);
    switch (_nodes.size()) {
      case 1:
        container1d[_nodes[0]].push_back(t);
      case 2:
        container2d[ _nodes[0]
                   + nNodes * _nodes[1]
                   ].push_back(t);
      case 3:
        container3d[ _nodes[0]
                   + nNodes * _nodes[1]
                   + nNodes * nNodes * _nodes[2]
                   ].push_back(t);

    }
  }

  if (info.nodeId == 0)
    std::cout << "\tBuilding 1-d containers\n";
  // DISTRIBUTE 1-d containers
  // every tuple which is only located at one node belongs to this node
  {
    auto const& _tuplesVec = container1d[info.nodeId];
    nodeTuples.resize(_tuplesVec.size());
    std::copy(_tuplesVec.begin(), _tuplesVec.end(), nodeTuples.begin());
  }

  if (info.nodeId == 0)
    std::cout << "\tBuilding 2-d containers\n";
  // DISTRIBUTE 2-d containers
  //the tuples which are located at two nodes are half/half given to these nodes
  for (auto const& m: container2d) {

    auto const& _tuplesVec = m.second;
      const
    size_t idx = m.first % nNodes
         // remeber: m.first = idy * nNodes + idx
         , idy = m.first / nNodes
         , n_half = _tuplesVec.size() / 2
         , size = nodeTuples.size()
         ;

    size_t nextra, nbegin, nend;
    if (info.nodeId == idx) {
      nextra = n_half;
      nbegin = 0 * n_half;
      nend   = n_half;
    } else if (info.nodeId == idy) {
      nextra = _tuplesVec.size() - n_half;
      nbegin = 1 * n_half;
      nend   = _tuplesVec.size();
    } else {
      // either idx or idy is my node
      continue;
    }

    nodeTuples.resize(size + nextra);
    std::copy(_tuplesVec.begin() + nbegin,
              _tuplesVec.begin() + nend,
              nodeTuples.begin() + size);

  }

  if (info.nodeId == 0)
    std::cout << "\tBuilding 3-d containers\n";
  // DISTRIBUTE 3-d containers
  for (auto const& m: container3d){
    auto const& _tuplesVec = m.second;

      const
    size_t idx = m.first % nNodes
         , idy = (m.first / nNodes) % nNodes
         // remember: m.first = idx + idy * nNodes + idz * nNodes^2
         , idz = m.first / nNodes / nNodes
         , n_third = _tuplesVec.size() / 3
         , size = nodeTuples.size()
         ;

    size_t nextra, nbegin, nend;
    if (info.nodeId == idx) {
      nextra = n_third;
      nbegin = 0 * n_third;
      nend   = nextra;
    } else if (info.nodeId == idy) {
      nextra = n_third;
      nbegin = 1 * n_third;
      nend   = 2 * nextra;
    } else if (info.nodeId == idz) {
      nextra = _tuplesVec.size() - 2 * n_third;
      nbegin = 2 * n_third;
      nend   = _tuplesVec.size();
    } else {
      // either idx or idy or idz is my node
      continue;
    }

    nodeTuples.resize(size + nextra);
    std::copy(_tuplesVec.begin() + nbegin,
              _tuplesVec.begin() + nend,
              nodeTuples.begin() + size);

  }


  if (info.nodeId == 0) std::cout << "\tswapping tuples...\n";
  /*
   *  sort part of group-and-sort algorithm
   *  every tuple on a given node is sorted in a way that
   *  the 'home elements' are the fastest index.
   *  1:yyy 2:yyn(x) 3:yny(x) 4:ynn(x) 5:nyy 6:nyn(x) 7:nny 8:nnn
   */
  for (auto &nt: nodeTuples){
    if ( isOnNode(nt[0], nNodes) == info.nodeId ){ // 1234
      if ( isOnNode(nt[2], nNodes) != info.nodeId ){ // 24
        size_t const x(nt[0]);
        nt[0] = nt[2];         // switch first and last
        nt[2] = x;
      }
      else if ( isOnNode(nt[1], nNodes) != info.nodeId){ // 3
        size_t const x(nt[0]);
        nt[0] = nt[1];         // switch first two
        nt[1] = x;
      }
    } else {
      if ( isOnNode(nt[1], nNodes) == info.nodeId   // 56
        && isOnNode(nt[2], nNodes) != info.nodeId
        ) { // 6
        size_t const x(nt[1]);
        nt[1] = nt[2];         // switch last two
        nt[2] = x;
      }
    }
  }

  if (info.nodeId == 0) std::cout << "\tsorting list of tuples...\n";
  //now we sort the list of tuples
  std::sort(nodeTuples.begin(), nodeTuples.end());

  if (info.nodeId == 0) std::cout << "\trestoring tuples...\n";
  // we bring the tuples abc back in the order a<b<c
  for (auto &t: nodeTuples)  std::sort(t.begin(), t.end());

  return nodeTuples;

}
// Distribution:1 ends here

// [[file:~/atrip/atrip.org::*Main][Main:1]]
std::vector<ABCTuple> main(MPI_Comm universe, size_t Nv) {

  int rank, np;
  MPI_Comm_rank(universe, &rank);
  MPI_Comm_size(universe, &np);

  std::vector<ABCTuple> result;

  auto const nodeNames(getNodeNames(universe));
  size_t const nNodes = unique(nodeNames).size();
  auto const nodeInfos = getNodeInfos(nodeNames);

  // We want to construct a communicator which only contains of one
  // element per node
  bool const computeDistribution
    = nodeInfos[rank].localRank == 0;

  std::vector<ABCTuple>
    nodeTuples
      = computeDistribution
      ? specialDistribution(Info{nNodes, nodeInfos[rank].nodeId},
                            getAllTuplesList(Nv))
      : std::vector<ABCTuple>()
      ;

  LOG(1,"Atrip") << "got nodeTuples\n";

  // now we have to send the data from **one** rank on each node
  // to all others ranks of this node
    const
  int color = nodeInfos[rank].nodeId
    , key = nodeInfos[rank].localRank
    ;


  MPI_Comm INTRA_COMM;
  MPI_Comm_split(universe, color, key, &INTRA_COMM);
// Main:1 ends here

// [[file:~/atrip/atrip.org::*Main][Main:2]]
size_t const
  tuplesPerRankLocal
     = nodeTuples.size() / nodeInfos[rank].ranksPerNode
     + size_t(nodeTuples.size() % nodeInfos[rank].ranksPerNode != 0)
     ;

size_t tuplesPerRankGlobal;

MPI_Reduce(&tuplesPerRankLocal,
           &tuplesPerRankGlobal,
           1,
           MPI_UINT64_T,
           MPI_MAX,
           0,
           universe);

MPI_Bcast(&tuplesPerRankGlobal,
          1,
          MPI_UINT64_T,
          0,
          universe);

LOG(1,"Atrip") << "Tuples per rank: " << tuplesPerRankGlobal << "\n";
LOG(1,"Atrip") << "ranks per node " << nodeInfos[rank].ranksPerNode << "\n";
LOG(1,"Atrip") << "#nodes " << nNodes << "\n";
// Main:2 ends here

// [[file:~/atrip/atrip.org::*Main][Main:3]]
size_t const totalTuples
  = tuplesPerRankGlobal * nodeInfos[rank].ranksPerNode;

if (computeDistribution) {
  // pad with FAKE_TUPLEs
  nodeTuples.insert(nodeTuples.end(),
                    totalTuples - nodeTuples.size(),
                    FAKE_TUPLE);
}
// Main:3 ends here

// [[file:~/atrip/atrip.org::*Main][Main:4]]
{
  // construct mpi type for abctuple
  MPI_Datatype MPI_ABCTUPLE;
  MPI_Type_vector(nodeTuples[0].size(), 1, 1, MPI_UINT64_T, &MPI_ABCTUPLE);
  MPI_Type_commit(&MPI_ABCTUPLE);

  LOG(1,"Atrip") << "scattering tuples \n";

  result.resize(tuplesPerRankGlobal);
  MPI_Scatter(nodeTuples.data(),
              tuplesPerRankGlobal,
              MPI_ABCTUPLE,
              result.data(),
              tuplesPerRankGlobal,
              MPI_ABCTUPLE,
              0,
              INTRA_COMM);

  MPI_Type_free(&MPI_ABCTUPLE);

}
// Main:4 ends here

// [[file:~/atrip/atrip.org::*Main][Main:5]]
LOG(1,"Atrip") << "scattering tuples \n";

  return result;

}
// Main:5 ends here

// [[file:~/atrip/atrip.org::*Interface][Interface:1]]
struct Distribution : public TuplesDistribution {
  ABCTuples getTuples(size_t Nv, MPI_Comm universe) override {
    return main(universe, Nv);
  }
};
// Interface:1 ends here

// [[file:~/atrip/atrip.org::*Epilog][Epilog:1]]
} // namespace group_and_sort
// Epilog:1 ends here

// [[file:~/atrip/atrip.org::*Epilog][Epilog:1]]
}
// Epilog:1 ends here
