#ifndef RISKREADER_HPP_
#define RISKREADER_HPP_

#include <string>
#include <atrip/RankMap.hpp>
#include <atrip/Complex.hpp>

namespace atrip {

template <typename F>
Sources<F> riskReader(const std::string &file_path,
                      std::vector<size_t> tensor_dimension,
                      std::vector<size_t> slice_mapping,
                      const size_t No,
                      const size_t Nv,
                      bool rowMajor);

template <typename F>
Sources<F> citfReader(CTF::Tensor<F>& tensor,
                      std::vector<size_t> tensor_dimension,
                      std::vector<size_t> slice_mapping,
                      const size_t No,
                      const size_t Nv);


template <typename F>
Sources<F> newReader(void* tensor_,
                     std::vector<size_t> tensor_dimension,
                     std::vector<size_t> slice_mapping,
                     const size_t No,
                     const size_t Nv,
                     const std::string &file_path = "",
                     bool rowMajor = false) {
  if (tensor_ == nullptr) {
    assert(!file_path.empty());
    return riskReader<F>(file_path,
                         tensor_dimension,
                         slice_mapping,
                         No,
                         Nv,
                         rowMajor);
  }
//TODO preprocessor if around this cast
  auto *tensor = reinterpret_cast<CTF::Tensor<F>*>(tensor_);
  if (tensor == nullptr) {
    throw std::invalid_argument("Invalid tensor pointer, failed dynamic_cast");
  }
  return citfReader<F>(*tensor,
                       tensor_dimension,
                       slice_mapping,
                       No,
                       Nv);

}


// read file data into a vector
template <typename F>
std::vector<F> read_all(std::vector<size_t> lengths,
                        std::string const &file_path,
                        MPI_Comm comm);


} // namespace atrip

#endif
