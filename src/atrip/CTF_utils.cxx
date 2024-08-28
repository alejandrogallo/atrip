#include <numeric>
#include <vector>

#include <atrip/Atrip.hpp>
#include <atrip/CTF_utils.hpp>
#include <atrip/Complex.hpp>

#if defined(HAVE_CTF)
namespace atrip {

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
#endif /* defined(HAVE_CTF) */
