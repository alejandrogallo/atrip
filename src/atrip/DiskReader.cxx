#include <atrip/DiskReader.hpp>

#include <atrip/Unions.hpp>

#define INSTANTIATE_READER(name_)                                              \
  template void DiskReader<name_<double>>::read(const size_t slice_index);     \
  template void DiskReader<name_<Complex>>::read(const size_t slice_index)

namespace atrip {

template <typename F>
void DiskReader<APHH<F>>::read(const size_t slice_index) {

  const int /**/
      a = this->slice_union->rank_map.find(
          {static_cast<size_t>(Atrip::rank), slice_index}),
      count = this->Nv * this->No * this->No;

  const MPI_Offset offset = a * count * sizeof(F);

  if (MPI_SUCCESS
      != MPI_File_read_at(this->handle,
                          offset,
                          this->slice_union->sources[slice_index].data(),
                          count,
                          MPI_DOUBLE,
                          MPI_STATUS_IGNORE)) {
    throw "error reading!";
  }

  std::vector<F> copy(this->slice_union->sources[slice_index]);
  for (size_t i = 0; i < this->No; i++)
    for (size_t j = 0; j < this->No; j++)
      for (size_t b = 0; b < this->Nv; b++)
        this->slice_union
            ->sources[slice_index][b + i * this->Nv + j * this->Nv * this->No] =
            copy[j + i * this->No + b * this->No * this->No];
}

INSTANTIATE_READER(APHH);

template <typename F>
void DiskReader<ABPH<F>>::read(const size_t slice_index) {

  const int /**/
      el = this->slice_union->rank_map.find(
          {static_cast<size_t>(Atrip::rank), slice_index}),
      a = el % this->Nv, b = el / this->Nv, count = this->Nv * this->No;

  // Be careful, now we have
  // i + c * No + b * NoNv + a * NoNvNv
  const MPI_Offset offset = (a * this->Nv + b) * count * sizeof(F);

  if (MPI_SUCCESS
      != MPI_File_read_at(this->handle,
                          offset,
                          this->slice_union->sources[slice_index].data(),
                          count,
                          MPI_DOUBLE,
                          MPI_STATUS_IGNORE)) {
    throw "error reading!";
  }

  std::vector<F> copy(this->slice_union->sources[slice_index]);
  for (size_t i = 0; i < this->No; i++)
    for (size_t a = 0; a < this->Nv; a++)
      this->slice_union->sources[slice_index][a + i * this->Nv] =
          copy[i + a * this->No];
}

INSTANTIATE_READER(ABPH);

template <typename F>
void DiskReader<HHHA<F>>::read(const size_t slice_index) {

  const int /**/
      a = this->slice_union->rank_map.find(
          {static_cast<size_t>(Atrip::rank), slice_index}),
      count = this->No * this->No * this->No;

  const MPI_Offset offset = a * count * sizeof(F);

  if (MPI_SUCCESS
      != MPI_File_read_at(this->handle,
                          offset,
                          this->slice_union->sources[slice_index].data(),
                          count,
                          MPI_DOUBLE,
                          MPI_STATUS_IGNORE)) {
    throw "error reading!";
  }
}

INSTANTIATE_READER(HHHA);

template <typename F>
void DiskReader<ABHH<F>>::read(const size_t slice_index) {

  const int /**/
      el = this->slice_union->rank_map.find(
          {static_cast<size_t>(Atrip::rank), slice_index}),
      a = el % this->Nv, b = el / this->Nv, count = this->No * this->No;

  // Be careful, now we have
  // i + j * No + b * NoNo + a * NoNoNv
  const MPI_Offset offset = (a * this->Nv + b) * count * sizeof(F);

  if (MPI_SUCCESS
      != MPI_File_read_at(this->handle,
                          offset,
                          this->slice_union->sources[slice_index].data(),
                          count,
                          MPI_DOUBLE,
                          MPI_STATUS_IGNORE)) {
    throw "error reading!";
  }

  std::vector<F> copy(this->slice_union->sources[slice_index]);
  for (size_t i = 0; i < this->No; i++)
    for (size_t j = 0; j < this->No; j++)
      this->slice_union->sources[slice_index][i + j * this->No] =
          copy[j + i * this->No];

  // std::reverse(this->slice_union->sources[slice_index].begin(),
  // this->slice_union->sources[slice_index].end());
}

INSTANTIATE_READER(ABHH);

} // namespace atrip
