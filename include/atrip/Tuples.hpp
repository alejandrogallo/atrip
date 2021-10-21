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

// [[file:~/atrip/atrip.org::*Naive%20list][Naive list:1]]
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
// Naive list:1 ends here

// [[file:~/atrip/atrip.org::*Naive%20list][Naive list:2]]
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

  WITH_RANK << "nRoundRobin = " << nRoundRobin << "\n";
  WITH_RANK << "nExtraInvalid = " << nExtraInvalid << "\n";
  WITH_RANK << "ntuples = " << n_tuples_per_rank[rank] << "\n";

  auto const& it = n_tuples_per_rank.begin();

  std::pair<size_t, size_t> const
    range = { std::accumulate(it, it + rank    , 0)
            , std::accumulate(it, it + rank + 1, 0) - 1
            };

  WITH_RANK << "range = "
            << range.first << " -> " << range.second
            << std::endl;

  return range;

}
// Naive list:2 ends here

// [[file:~/atrip/atrip.org::*Naive%20list][Naive list:3]]
struct NaiveDistribution : public TuplesDistribution {
  ABCTuples getTuples(size_t Nv, MPI_Comm universe) override {
    int rank, np;
    MPI_Comm_rank(universe, &rank);
    MPI_Comm_size(universe, &np);
    auto const all = getTuplesList(Nv);
    const size_t
      tuplesPerRank
        = all.size() / np
        + size_t(all.size() % np != 0)
        ;
    //auto const range = getABCRange((size_t)np, (size_t)rank, all);

    std::pair<size_t, size_t> const
      range = { tuplesPerRank * rank
              , tuplesPerRank * (rank + 1) - 1
              };

    WITH_RANK << "range = "
              << range.first << " -> " << range.second
              << std::endl;
    std::vector<ABCTuple> result(range.second - range.first + 1, FAKE_TUPLE);
    WITH_RANK << "number of global tuples = " << all.size() << std::endl;
    WITH_RANK << "number of local tuples  = " << result.size() << std::endl;

    std::copy(range.first >= all.size()
              ? all.end()
              : all.begin() + range.first,
              // --
              range.second >= all.size()
              ? all.end()
              : all.begin() + range.first + range.second,
              // --
              result.begin());
    return result;
  }
};
// Naive list:3 ends here

// [[file:~/atrip/atrip.org::*Prolog][Prolog:1]]
namespace group_and_sort {
// Prolog:1 ends here

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

std::vector<RankInfo>
getNodeInfos(std::vector<string> const& nodeNames) {
  std::vector<RankInfo> result;
  auto uniqueNames = nodeNames;
  {
    std::sort(uniqueNames.begin(), uniqueNames.end());
    auto const& last = std::unique(uniqueNames.begin(), uniqueNames.end());
    uniqueNames.erase(last, uniqueNames.end());
  }
  const auto index = [&uniqueNames](std::string const& s) {
    auto const& it = std::find(uniqueNames.begin(), uniqueNames.end(), s);
    return std::distance(uniqueNames.begin(), it);
  };
  std::vector<size_t> localRanks(uniqueNames.size(), 0);
  size_t rank = 0;
  for (auto const& name: nodeNames) {
    const size_t nodeId = index(name);
    result.push_back({name,
                      nodeId,
                      rank++,
                      localRanks[nodeId]++,
                      std::count(nodeNames.begin(),
                                 nodeNames.end(),
                                 name)
                      });
  }
  return result;
}
// Node information:2 ends here

// [[file:~/atrip/atrip.org::*Utils][Utils:1]]
// Provides the node on which the slice-element is found
// Right now we distribute the slices in a round robin fashion
// over the different nodes (NOTE: not mpi ranks but nodes)
size_t isOnNode(size_t tuple, size_t nodes) { return tuple % nodes; }


struct Info {
  size_t nNodes;
  size_t Nv;
  size_t np;
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
std::vector<ABCTuple>
specialDistribution(Info info, std::vector<ABCTuple> const& allTuples) {

  std::vector<ABCTuple> nodeTuples;
  size_t nNodes(info.nNodes);
  size_t np(info.np);
  size_t N(allTuples.size());

  //      nodeid          tuple list
  std::map<size_t, std::vector<ABCTuple> > container1d;
  std::map<size_t, std::vector<ABCTuple> > container2d;
  std::map<size_t, std::vector<ABCTuple> > container3d;

  // build container-n-d's
  for (auto t: allTuples) {
    // one which node(s) are the tuple elements located...
    // put them into the right container
    auto nt = getTupleNodes(t, nNodes);
    if ( nt.size() == 1) container1d[nt[0]].push_back(t);
    if ( nt.size() == 2) container2d[nt[0] + nNodes*nt[1]].push_back(t);
    if ( nt.size() == 3)
      container3d[nt[0] + nNodes*nt[1] + nNodes*nNodes*nt[2]].push_back(t);
  }

  // DISTRIBUTE 1-d containers
  // every tuple which is only located at one node belongs to this node
  {
    auto const& tuplesVec = container1d[info.nodeId];
    nodeTuples.resize(tuplesVec.size());
    std::copy(tuplesVec.begin(), tuplesVec.end(), nodeTuples.begin());
  }

  // DISTRIBUTE 2-d containers
  //the tuples which are located at two nodes are half/half given to these nodes
  for (auto &m: container2d) {
    size_t idx = m.first%nNodes;
    size_t idy = m.first/nNodes;
    size_t myNode = idx;

    // either idx or idy is my node
    if (idx != info.nodeId && idy != info.nodeId) continue;
    if (idy == info.nodeId) myNode = idy;

    auto tuplesVec = m.second;
    auto n = tuplesVec.size() / 2;
    auto size = nodeTuples.size();
    if (myNode == idx) {
      nodeTuples.resize(size + n);
      std::copy(tuplesVec.begin(),
                tuplesVec.begin() + n,
                nodeTuples.begin() + size);
    } else {
      auto ny = tuplesVec.size() - n;
      nodeTuples.resize(size + ny);
      std::copy(tuplesVec.begin() + n,
                tuplesVec.end(),
                nodeTuples.begin() + size);
    }

  }

  // DISTRIBUTE 3-d containers
  // similar game for the tuples which belong to three different nodes
  for (auto m: container3d){
    auto tuplesVec = m.second;
    auto idx = m.first%nNodes;
    auto idy = (m.first/nNodes)%nNodes;
    auto idz = m.first/nNodes/nNodes;
    if (idx != info.nodeId && idy != info.nodeId && idz != info.nodeId) continue;

    size_t nx = tuplesVec.size() / 3;
    size_t n, nbegin, nend;
    if (info.nodeId == idx) {
      n = nx;
      nbegin = 0;
      nend = n;
    } else if (info.nodeId == idy) {
      n = nx;
      nbegin = n;
      nend = n + n;
    } else {
      n = tuplesVec.size() - 2 * nx;
      nbegin = 2 * nx;
      nend = 2 * nx + n;
    }

    auto size = nodeTuples.size();
    nodeTuples.resize(size + n);
    std::copy(tuplesVec.begin() + nbegin,
              tuplesVec.begin() + nend,
              nodeTuples.begin() + size);

  }


  // sort part of group-and-sort algorithm
  // every tuple on a given node is sorted in a way that
  // the 'home elements' are the fastest index.
  // 1:yyy 2:yyn(x) 3:yny(x) 4:ynn(x) 5:nyy 6:nyn(x) 7:nny 8:nnn
  size_t n = info.nodeId;
  for (auto &nt: nodeTuples){
    if ( isOnNode(nt[0], nNodes) == n ){ // 1234
      if ( isOnNode(nt[2], nNodes) != n ){ // 24
        size_t x(nt[0]); nt[0] = nt[2]; nt[2] = x; // switch first and last
      }
      else if ( isOnNode(nt[1], nNodes) != n){ // 3
        size_t x(nt[0]); nt[0] = nt[1]; nt[1] = x; // switch first two
      }
    } else {
      if ( isOnNode(nt[1], nNodes) == n   // 56
        && isOnNode(nt[2], nNodes) != n){ // 6
        size_t x(nt[1]); nt[1] = nt[2]; nt[2] = x; // switch last two
      }
    }
  }
  //now we sort the list of tuples
  std::sort(nodeTuples.begin(), nodeTuples.end());
  // we bring the tuples abc back in the order a<b<c
  for (auto &t: nodeTuples)  std::sort(t.begin(), t.end());

  return nodeTuples;

}

//determine which element has to be fetched from sources for the next iteration
std::vector<size_t> fetchElement(ABCTuple cur, ABCTuple suc){
  std::vector<size_t> result;
  ABCTuple inter;
  std::sort(cur.begin(), cur.end());
  std::sort(suc.begin(), suc.end());
  std::array<size_t,3>::iterator rit, cit, sit;
  cit = std::unique(cur.begin(), cur.end());
  sit = std::unique(suc.begin(), suc.end());
  rit = std::set_difference(suc.begin(), sit, cur.begin(), cit, inter.begin());
  result.resize(rit - inter.begin());
  std::copy(inter.begin(), rit, result.begin());
  return result;
}
// Distribution:1 ends here

// [[file:~/atrip/atrip.org::*Main][Main:1]]
std::vector<ABCTuple> main(MPI_Comm universe, size_t Nv) {

  int rank, np;
  MPI_Comm_rank(universe, &rank);
  MPI_Comm_size(universe, &np);

  std::vector<ABCTuple> result;

  const auto nodeNames(getNodeNames(universe));
  auto nodeNamesUnique(nodeNames);
  {
    const auto& last = std::unique(nodeNamesUnique.begin(),
                                   nodeNamesUnique.end());
    nodeNamesUnique.erase(last, nodeNamesUnique.end());
  }
  // we pick one rank from every node
  auto const nodeInfos = getNodeInfos(nodeNames);
  size_t const nNodes = nodeNamesUnique.size();

  // We want to construct a communicator which only contains of one
  // element per node
  bool makeDistribution
    = nodeInfos[rank].localRank == 0
    ? true
    : false
    ;

  std::vector<ABCTuple>
    nodeTuples = makeDistribution
               ? specialDistribution(Info{ nNodes
                                         , Nv
                                         , np
                                         , nodeInfos[rank].nodeId
                                         },
                                      getTuplesList(Nv))
               : std::vector<ABCTuple>()
               ;


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
const size_t
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
// Main:2 ends here

// [[file:~/atrip/atrip.org::*Main][Main:3]]
size_t const totalTuplesLocal
  = tuplesPerRankLocal
  * nodeInfos[rank].ranksPerNode;

if (makeDistribution)
  nodeTuples.insert(nodeTuples.end(),
                    totalTuplesLocal - nodeTuples.size(),
                    FAKE_TUPLE);
// Main:3 ends here

// [[file:~/atrip/atrip.org::*Main][Main:4]]
{
  std::vector<int> const
    sendCounts(nodeInfos[rank].ranksPerNode, tuplesPerRankLocal);

  std::vector<int>
    displacements(nodeInfos[rank].ranksPerNode);

  std::iota(displacements.begin(),
            displacements.end(),
            tuplesPerRankLocal);

  // important!
  result.resize(tuplesPerRankLocal);

  // construct mpi type for abctuple
  MPI_Datatype MPI_ABCTUPLE;
  MPI_Type_vector(nodeTuples[0].size(), 1, 1, MPI_UINT64_T, &MPI_ABCTUPLE);
  MPI_Type_commit(&MPI_ABCTUPLE);

  MPI_Scatterv(nodeTuples.data(),
              sendCounts.data(),
              displacements.data(),
              MPI_ABCTUPLE,
              result.data(),
              tuplesPerRankLocal,
              MPI_ABCTUPLE,
              0,
              INTRA_COMM);

  // free type
  MPI_Type_free(&MPI_ABCTUPLE);

}
// Main:4 ends here

// [[file:~/atrip/atrip.org::*Main][Main:5]]
result.insert(result.end(),
                tuplesPerRankGlobal - result.size(),
                FAKE_TUPLE);

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
