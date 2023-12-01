#ifndef DISKREADER_HPP_
#define DISKREADER_HPP_

#include <string>

#include <atrip/Reader.hpp>
#include <atrip/SliceUnion.hpp>

#define DECLARE_DISK_READER(name_)                                             \
  template <typename F>                                                        \
  class name_##_DiskReader : public DiskReader<F> {                            \
    using DiskReader<F>::DiskReader;                                           \
    void read(const size_t slice_index) override;                              \
  }

namespace atrip {

template <typename F>
class DiskReader : public Reader {
public:
  const std::string file_path;
  SliceUnion<F> *slice_union;
  const int No, Nv;
  MPI_File handle;
  DiskReader(const std::string file_path_,
             SliceUnion<F> *slice_union_,
             int No_,
             int Nv_)
      : file_path(file_path_)
      , slice_union(slice_union_)
      , No(No_)
      , Nv(Nv_) {

    std::cout << "Opening MPI File" << std::endl;
    MPI_File_open(MPI_COMM_WORLD,
                  file_path.c_str(),
                  MPI_MODE_RDONLY,
                  MPI_INFO_NULL,
                  &handle);
  }
  void close() { MPI_File_close(&handle); }
};

DECLARE_DISK_READER(APHH);
DECLARE_DISK_READER(ABPH);
DECLARE_DISK_READER(HHHA);
DECLARE_DISK_READER(ABHH);

} // namespace atrip

#undef DECLARE_DISK_READER
#endif
