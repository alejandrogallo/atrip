#ifndef CTFREADER_HPP_
#define CTFREADER_HPP_

#include <atrip/Reader.hpp>
#include <atrip/CTF.hpp>
#include <atrip/SliceUnion.hpp>

#define DECLARE_CTF_READER(name_)                                              \
  template <typename F>                                                        \
  class name_##_CTFReader : public CTFReader<F> {                              \
  public:                                                                      \
    using CTFReader<F>::CTFReader;                                             \
    void read(const size_t slice_index) override;                              \
  }

namespace atrip {

template <typename F>
class CTFReader : public Reader {
public:
  const CTF::Tensor<F> *source_tensor;
  SliceUnion<F> *slice_union;
  const int No, Nv;
  CTFReader(const CTF::Tensor<F> *source_tensor_,
            SliceUnion<F> *slice_union_,
            int No_,
            int Nv_)
      : source_tensor(source_tensor_)
      , slice_union(slice_union_)
      , No(No_)
      , Nv(Nv_) {}

  void close() override { delete temp_tensor; }

  CTF::Tensor<F> *temp_tensor = nullptr;
  CTF::World *world;
};

DECLARE_CTF_READER(TAPHH);
DECLARE_CTF_READER(HHHA);
DECLARE_CTF_READER(ABPH);
DECLARE_CTF_READER(ABHH);

} // namespace atrip

#undef DECLARE_CTF_READER
#endif
