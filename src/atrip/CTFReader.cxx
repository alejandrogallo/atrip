#include <vector>

#include <atrip/Unions.hpp>
#include <atrip/CTFReader.hpp>

#define INSTANTIATE_READER(name_)                                              \
  template void CTFReader<name_<double>>::read(const size_t slice_index);      \
  template void CTFReader<name_<Complex>>::read(const size_t slice_index)

namespace atrip {

template <typename F>
void CTFReader<APHH<F>>::read(const size_t slice_index) {

  const int a = this->slice_union->rank_map.find(
      {static_cast<size_t>(Atrip::rank), slice_index});

  if (this->temp_tensor == nullptr) {
    const std::vector<int> voo = {this->Nv, this->No, this->No}, syms(4, NS);
    this->world = new CTF::World(this->slice_union->world);
    this->temp_tensor =
        new CTF::Tensor<F>(voo.size(), voo.data(), syms.data(), *this->world);
  }

  slice_into_vector<F>(this->slice_union->sources[slice_index],
                       this->slice_union->slice_size,
                       *this->temp_tensor,
                       std::vector<int>({0, 0, 0}),
                       std::vector<int>({this->Nv, this->No, this->No}),
                       *this->source_tensor,
                       std::vector<int>({a, 0, 0, 0}),
                       std::vector<int>({a + 1, this->Nv, this->No, this->No}));
}

INSTANTIATE_READER(APHH);

template <typename F>
void CTFReader<HHHA<F>>::read(size_t slice_index) {

  const int a = this->slice_union->rank_map.find(
      {static_cast<size_t>(Atrip::rank), slice_index});

  if (this->temp_tensor == nullptr) {
    const std::vector<int> ooo(3, this->No), syms(4, NS);
    this->world = new CTF::World(this->slice_union->world);
    this->temp_tensor =
        new CTF::Tensor<F>(ooo.size(), ooo.data(), syms.data(), *this->world);
  }

  slice_into_vector<F>(this->slice_union->sources[slice_index],
                       this->slice_union->slice_size,
                       *this->temp_tensor,
                       {0, 0, 0},
                       {this->No, this->No, this->No},
                       *this->source_tensor,
                       {0, 0, 0, a},
                       {this->No, this->No, this->No, a + 1});
}

INSTANTIATE_READER(atrip::HHHA);

template <typename F>
void CTFReader<ABPH<F>>::read(size_t slice_index) {

  const int el = this->slice_union->rank_map.find(
                {static_cast<size_t>(Atrip::rank), slice_index}),
            a = el % this->Nv, b = el / this->Nv;

  if (this->temp_tensor == nullptr) {
    const std::vector<int> vo = {this->Nv, this->No}, syms(4, NS);
    this->world = new CTF::World(this->slice_union->world);
    this->temp_tensor =
        new CTF::Tensor<F>(vo.size(), vo.data(), syms.data(), *this->world);
  }

  slice_into_vector<F>(this->slice_union->sources[slice_index],
                       this->slice_union->slice_size,
                       *this->temp_tensor,
                       {0, 0},
                       {this->Nv, this->No},
                       *this->source_tensor,
                       {a, b, 0, 0},
                       {a + 1, b + 1, this->Nv, this->No});
}

INSTANTIATE_READER(atrip::ABPH);

template <typename F>
void CTFReader<ABHH<F>>::read(size_t slice_index) {

  const int el = this->slice_union->rank_map.find(
                {static_cast<size_t>(Atrip::rank), slice_index}),
            a = el % this->Nv, b = el / this->Nv;

  if (this->temp_tensor == nullptr) {
    const std::vector<int> oo(2, this->No), syms(4, NS);
    this->world = new CTF::World(this->slice_union->world);
    this->temp_tensor =
        new CTF::Tensor<F>(oo.size(), oo.data(), syms.data(), *this->world);
  }

  slice_into_vector<F>(this->slice_union->sources[slice_index],
                       this->slice_union->slice_size,
                       *this->temp_tensor,
                       {0, 0},
                       {this->No, this->No},
                       *this->source_tensor,
                       {a, b, 0, 0},
                       {a + 1, b + 1, this->No, this->No});
}

INSTANTIATE_READER(atrip::ABHH);

} // namespace atrip
