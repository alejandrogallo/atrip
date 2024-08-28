#include <vector>

#include <atrip/Utils.hpp>
#include <atrip/Unions.hpp>
#include <atrip/CTF_memory_reader.hpp>

#define INSTANTIATE_READER(name_)                                              \
  template void CTF_memory_reader<name_<float>>::read(                         \
      const size_t slice_index);                                               \
  template void CTF_memory_reader<name_<double>>::read(                        \
      const size_t slice_index);                                               \
  template void CTF_memory_reader<name_<Complex>>::read(                       \
      const size_t slice_index)

#define IMPLEMENT_READER(name_, slice_index)                                   \
  INSTANTIATE_READER(name_);                                                   \
  template <typename F>                                                        \
  void CTF_memory_reader<name_<F>>::read(slice_index)

#if defined(HAVE_CTF)
namespace atrip {

IMPLEMENT_READER(HHHA, size_t slice_index) { return; }

IMPLEMENT_READER(APHH, size_t slice_index) { return; }

IMPLEMENT_READER(ABPH, size_t slice_index) { return; }

IMPLEMENT_READER(ABHH, size_t slice_index) { return; }

} // namespace atrip
#endif /* defined(HAVE_CTF) */
