#include <vector>

#include <atrip/Unions.hpp>
#include <atrip/CTFReader.hpp>

#define INSTANTIATE_READER(name_)                                              \
  template void CTFReader<name_<float>>::read(const size_t slice_index);       \
  template void CTFReader<name_<double>>::read(const size_t slice_index);      \
  template void CTFReader<name_<Complex>>::read(const size_t slice_index)

namespace atrip {
#if defined(HAVE_CTF)
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
#endif /* defined(HAVE_CTF) */

template <typename F>
std::vector<F> read_all(std::vector<size_t> lengths,
                        std::string const &ctf_file_path,
                        MPI_Comm comm) {
  MPI_File handle;
  MPI_Offset offset = 0;
  const size_t count = std::accumulate(lengths.begin(),
                                       lengths.end(),
                                       1UL,
                                       std::multiplies<size_t>());
  std::vector<F> buffer(count);

  LOG(0, "Atrip") << "Openning file " << ctf_file_path << "\n";
  MPI_File_open(comm,
                ctf_file_path.c_str(),
                MPI_MODE_RDONLY,
                MPI_INFO_NULL,
                &handle);

  LOG(0, "Atrip") << "Reading " << ctf_file_path << "\n";
  if (MPI_SUCCESS
      != MPI_File_read_at(handle,
                          offset,
                          buffer.data(),
                          count,
                          MPI_DOUBLE,
                          MPI_STATUS_IGNORE)) {
    throw "error reading!";
  }

  LOG(0, "Atrip") << "Closing " << ctf_file_path << "\n";
  MPI_File_close(&handle);
  return buffer;
}

template std::vector<float> read_all<float>(std::vector<size_t> lengths,
                                            std::string const &ctf_file_path,
                                            MPI_Comm comm);
template std::vector<double> read_all<double>(std::vector<size_t> lengths,
                                              std::string const &ctf_file_path,
                                              MPI_Comm comm);
template std::vector<Complex>
read_all<Complex>(std::vector<size_t> lengths,
                  std::string const &ctf_file_path,
                  MPI_Comm comm);

} // namespace atrip
