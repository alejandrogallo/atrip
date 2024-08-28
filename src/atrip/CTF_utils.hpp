#ifndef CTF_UTILS_HPP_
#define CTF_UTILS_HPP_
#if defined(HAVE_CTF)

#  include <mpi.h>
#  include <vector>

namespace atrip {

// Read a tensor file path when CTF file is not enabled in the
// program.
template <typename F>
std::vector<F> read_all(std::vector<size_t> lengths,
                        std::string const &ctf_file_path,
                        MPI_Comm comm);

} // namespace atrip

#endif /*  defined(HAVE_CTF) */
#endif
