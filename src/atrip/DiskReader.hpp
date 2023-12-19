#ifndef DISKREADER_HPP_
#define DISKREADER_HPP_

#include <string>

#include <atrip/Reader.hpp>
#include <atrip/SliceUnion.hpp>

#define DECLARE_DISK_READER(name_)                                             \
  template <typename F>                                                        \
  class name_;                                                                 \
  template <typename F>                                                        \
  class DiskReader<name_<F>> : public DiskReaderProxy<F> {                     \
    using DiskReaderProxy<F>::DiskReaderProxy;                                 \
    void read(const size_t slice_index) override;                              \
    std::string name() override { return "Disk MPI Reader: " #name_; }         \
  }

namespace atrip {

template <typename F>
class DiskReaderProxy : public Reader {
public:
  const std::string file_path;
  SliceUnion<F> *slice_union;
  const int No, Nv;
  MPI_File handle;
  std::vector<F> reorder_buffer, mpi_buffer;
  DiskReaderProxy(const std::string file_path_,
                  SliceUnion<F> *slice_union_,
                  int No_,
                  int Nv_)
      : file_path(file_path_)
      , slice_union(slice_union_)
      , No(No_)
      , Nv(Nv_) {

    MPI_File_open(MPI_COMM_WORLD,
                  file_path.c_str(),
                  MPI_MODE_RDONLY,
                  MPI_INFO_NULL,
                  &handle);
  }
  void close() override { MPI_File_close(&handle); }
  void
  read_into_buffer(const size_t slice_index,
                   const size_t count,
                   const MPI_Offset offset,
                   std::function<void(std::vector<F> &, std::vector<F> &)>);
};

template <typename F>
class DiskReader;

DECLARE_DISK_READER(APHH);
DECLARE_DISK_READER(ABPH);
DECLARE_DISK_READER(HHHA);
DECLARE_DISK_READER(ABHH);

} // namespace atrip

#undef DECLARE_DISK_READER
#endif
