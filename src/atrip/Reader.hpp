#ifndef ATRIP_READER_HPP_
#define ATRIP_READER_HPP_

#include <cstdlib>
#include <string>

namespace atrip {

class Reader {
public:
  virtual void read(const size_t slice_index) = 0;
  virtual void close() = 0;
  virtual std::string name() = 0;
};

enum ReaderKind {
  DISK = 1914,
#if defined(HAVE_CTF)
  CTF_DISK = 1939,
  CTF_MEMORY = 1955,
#endif /* defined(HAVE_CTF) */
};

} // namespace atrip

#endif
