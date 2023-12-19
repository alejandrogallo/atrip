#ifndef CTFREADER_HPP_
#define CTFREADER_HPP_

#include <mpi.h>
#include <algorithm>

#include <atrip/CTF.hpp>
#include <atrip/Reader.hpp>
#include <atrip/SliceUnion.hpp>

#define DECLARE_CTF_READER(name_)                                              \
  template <typename F>                                                        \
  class name_;                                                                 \
  template <typename F>                                                        \
  class CTFReader<name_<F>> : public CTFReaderProxy<F> {                       \
  public:                                                                      \
    using CTFReaderProxy<F>::CTFReaderProxy;                                   \
    void read(const size_t) override;                                          \
    std::string name() override { return "CTF Reader " #name_; }               \
  }

namespace atrip {

#if defined(HAVE_CTF)
template <typename F>
static void slice_into_vector
#  if defined(ATRIP_SOURCES_IN_GPU)
    (DataPtr<F> &source,
#  else
    (std::vector<F> &source,
#  endif
     size_t slice_size,
     CTF::Tensor<F> &to_slice,
     std::vector<int> const low,
     std::vector<int> const up,
     CTF::Tensor<F> const &origin,
     std::vector<int> const origin_low,
     std::vector<int> const origin_up) {

  WITH_OCD
  WITH_RANK << "slicing into " << up << "," << low << " from " << origin_up
            << "," << origin_low << "\n";

#  if !defined(ATRIP_DONT_SLICE) && !defined(ATRIP_DRY)
  to_slice.slice(low.data(),
                 up.data(),
                 0.0,
                 origin,
                 origin_low.data(),
                 origin_up.data(),
                 1.0);

#    if defined(ATRIP_SOURCES_IN_GPU)
  WITH_CHRONO(
      "acc:sources",
      ACC_CHECK_SUCCESS("copying sources data to device",
                        ACC_MEMCPY_HOST_TO_DEV(source,
                                               to_slice.data,
                                               sizeof(F) * slice_size));)
#    else
  memcpy(source.data(), to_slice.data, sizeof(F) * slice_size);
#    endif /* defined(ATRIP_SOURCES_IN_GPU) */

#  else
  IGNORABLE(source);
  IGNORABLE(slice_size);
  IGNORABLE(to_slice);
  IGNORABLE(origin);
#    pragma message("WARNING: COMPILING WITHOUT SLICING THE TENSORS")
#  endif /* !defined(ATRIP_DONT_SLICE) && !defined(ATRIP_DRY) */
}

template <typename F>
class CTFReaderProxy : public Reader {
public:
  const CTF::Tensor<F> *source_tensor;
  SliceUnion<F> *slice_union;
  const int No, Nv;
  CTF::Tensor<F> *temp_tensor = nullptr;
  CTF::World *world;
  CTFReaderProxy(const CTF::Tensor<F> *source_tensor_,
                 SliceUnion<F> *slice_union_,
                 int No_,
                 int Nv_)
      : source_tensor(source_tensor_)
      , slice_union(slice_union_)
      , No(No_)
      , Nv(Nv_) {}

  void close() override { delete temp_tensor; }
};

template <typename F>
class CTFReader;

DECLARE_CTF_READER(HHHA);
DECLARE_CTF_READER(APHH);
DECLARE_CTF_READER(ABPH);
DECLARE_CTF_READER(ABHH);

#endif /*  defined(HAVE_CTF) */

// Read a tensor file path when CTF file is not enabled in the
// program.
template <typename F>
std::vector<F> read_all(std::vector<size_t> lengths,
                        std::string const &ctf_file_path,
                        MPI_Comm comm);

} // namespace atrip

#undef DECLARE_CTF_READER
#endif
