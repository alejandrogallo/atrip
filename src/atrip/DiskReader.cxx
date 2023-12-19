#include <atrip/DiskReader.hpp>

#include <atrip/Acc.hpp>
#include <atrip/Unions.hpp>
#include <atrip/Complex.hpp>

#define INSTANTIATE_READER(name_)                                              \
  template void DiskReader<name_<double>>::read(const size_t slice_index);     \
  template void DiskReader<name_<Complex>>::read(const size_t slice_index)

namespace atrip {

template <typename F>
void DiskReaderProxy<F>::read_into_buffer(
    const size_t slice_index,
    const size_t count,
    const MPI_Offset offset,
    std::function<void(std::vector<F> &, std::vector<F> &)> reorder) {

  const size_t slice_size = this->slice_union->slice_size;
#if defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_ACC)
  mpi_buffer.resize(slice_size);
  std::vector<F> &buffer = mpi_buffer;
#else
  std::vector<F> &buffer = this->slice_union->sources[slice_index];
#endif /* defined(ATRIP_SOURCES_IN_GPU) */

  if (MPI_SUCCESS
      != MPI_File_read_at(this->handle,
                          offset,
                          buffer.data(),
                          count,
                          traits::mpi::datatype_of<F>(),
                          MPI_STATUS_IGNORE)) {
    throw "error reading!";
  }

  reorder(this->reorder_buffer, buffer);

#if defined(ATRIP_SOURCES_IN_GPU) && defined(HAVE_ACC)
  WITH_CHRONO("acc:sources",
              ACC_CHECK_SUCCESS("copying sources data to device",
                                ACC_MEMCPY_HOST_TO_DEV(
                                    this->slice_union->sources[slice_index],
                                    buffer.data(),
                                    sizeof(F) * slice_size));)
#endif
}

template <typename F>
void DiskReader<APHH<F>>::read(const size_t slice_index) {
  const int /**/
      a = this->slice_union->rank_map.find(
          {static_cast<size_t>(Atrip::rank), slice_index}),
      count = this->Nv * this->No * this->No;

  const MPI_Offset offset = a * count * sizeof(F);

  this->read_into_buffer(
      slice_index,
      count,
      offset,
      [this](std::vector<F> &reorder_buffer, std::vector<F> &source_buffer) {
        reorder_buffer = source_buffer;
        for (size_t i = 0; i < this->No; i++)
          for (size_t j = 0; j < this->No; j++)
            for (size_t b = 0; b < this->Nv; b++)
              source_buffer[b + i * this->Nv + j * this->Nv * this->No] =
                  reorder_buffer[j + i * this->No + b * this->No * this->No];
      });
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

  this->read_into_buffer(
      slice_index,
      count,
      offset,
      [this](std::vector<F> &reorder_buffer, std::vector<F> &source_buffer) {
        reorder_buffer = source_buffer;
        for (size_t i = 0; i < this->No; i++)
          for (size_t a = 0; a < this->Nv; a++)
            source_buffer[a + i * this->Nv] = reorder_buffer[i + a * this->No];
      });
}

INSTANTIATE_READER(ABPH);

template <typename F>
void DiskReader<HHHA<F>>::read(const size_t slice_index) {
  const int /**/
      a = this->slice_union->rank_map.find(
          {static_cast<size_t>(Atrip::rank), slice_index}),
      count = this->No * this->No * this->No;

  const MPI_Offset offset = a * count * sizeof(F);

  this->read_into_buffer(
      slice_index,
      count,
      offset,
      [this](std::vector<F> &reorder_buffer, std::vector<F> &source_buffer) {});
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

  this->read_into_buffer(
      slice_index,
      count,
      offset,
      [this](std::vector<F> &reorder_buffer, std::vector<F> &source_buffer) {
        reorder_buffer = source_buffer;
        for (size_t i = 0; i < this->No; i++)
          for (size_t j = 0; j < this->No; j++)
            source_buffer[i + j * this->No] = reorder_buffer[j + i * this->No];
      });
}

INSTANTIATE_READER(ABHH);

} // namespace atrip
