#ifndef CTF_MEMORY_READER_HPP_
#define CTF_MEMORY_READER_HPP_
#if defined(HAVE_CTF)

#  include <mpi.h>
#  include <algorithm>

#  include <atrip/CTF.hpp>
#  include <atrip/CTF_disk_reader.hpp>
#  include <atrip/Reader.hpp>
#  include <atrip/SliceUnion.hpp>

#  define DECLARE_CTF_MEMORY_READER(name_)                                     \
    template <typename F>                                                      \
    class name_;                                                               \
    template <typename F>                                                      \
    class CTF_memory_reader<name_<F>> : public CTF_disk_reader_proxy<F> {      \
    public:                                                                    \
      using CTF_disk_reader_proxy<F>::CTF_disk_reader_proxy;                   \
      void read(const size_t) override;                                        \
      void close(){};                                                          \
      std::string name() override { return "CTF Reader " #name_; }             \
    }

namespace atrip {

template <typename F>
class CTF_memory_reader;

DECLARE_CTF_MEMORY_READER(HHHA);
DECLARE_CTF_MEMORY_READER(APHH);
DECLARE_CTF_MEMORY_READER(ABPH);
DECLARE_CTF_MEMORY_READER(ABHH);

#  undef DECLARE_CTF_READER

} // namespace atrip

#endif /*  defined(HAVE_CTF) */
#endif
