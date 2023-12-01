#ifndef ATRIP_READER_HPP_
#define ATRIP_READER_HPP_

#include <cstdlib>
#include <string>

namespace atrip {

enum ReaderType {
  CTF,
  DISK
};

class Reader {
public:
  virtual void read(const size_t slice_index) = 0;
  virtual void close() = 0;
  virtual std::string name() = 0;
};

} // namespace atrip

#endif
