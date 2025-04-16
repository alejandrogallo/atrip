#ifndef RISKREADER_HPP_
#define RISKREADER_HPP_

#include <string>
#include <atrip/RankMap.hpp>
#include <atrip/Complex.hpp>

namespace atrip {

//WATCH OUT: still mpi comm world in our routine!
//template <typename F>
//struct Sources {
//  size_t n_sources;
//  size_t s_sources;
//  std::vector<std::vector<F>> sources;
//};

template <typename F>
Sources<F> riskReader(const std::string &file_path,
                      std::vector<size_t> tensor_dimension,
                      std::vector<size_t> slice_mapping,
                      const size_t No,
                      const size_t Nv,
                      ClusterInfo cluster_info,
                      bool rowMajor);

#if defined(HAVE_CTF)
template <typename F>
Sources<F> citfReader(CTF::Tensor<F>& tensor,
                      std::vector<size_t> tensor_dimension,
                      std::vector<size_t> slice_mapping,
                      const size_t No,
                      const size_t Nv,
                      ClusterInfo cluster_info);
#endif

template <typename F>
Sources<F> newReader(void* tensor_,
                     std::vector<size_t> tensor_dimension,
                     std::vector<size_t> slice_mapping,
                     const size_t No,
                     const size_t Nv,
                     ClusterInfo cluster_info,
                     const std::string &file_path = "",
                     bool rowMajor = false) {
  if (tensor_ == nullptr) {
    assert(!file_path.empty());
    return riskReader<F>(file_path,
                         tensor_dimension,
                         slice_mapping,
                         No,
                         Nv,
                         cluster_info,
                         rowMajor);
  }
#if defined(HAVE_CTF)
  auto *tensor = reinterpret_cast<CTF::Tensor<F>*>(tensor_);
  if (tensor == nullptr) {
    throw std::invalid_argument("Invalid tensor pointer, failed dynamic_cast");
  }
  return citfReader<F>(*tensor,
                       tensor_dimension,
                       slice_mapping,
                       No,
                       Nv,
                       cluster_info);
#endif
  if (!Atrip::rank) std::cout << "CTF not available. Abort!" << std::endl;
  assert(0);
  return Sources<F>{};
}


} // namespace atrip

#endif
