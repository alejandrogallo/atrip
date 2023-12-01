#ifndef CTFREADER_HPP_
#define CTFREADER_HPP_

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

} // namespace atrip

#undef DECLARE_CTF_READER
#endif
