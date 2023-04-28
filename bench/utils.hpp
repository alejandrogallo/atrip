#ifndef UTILS_HPP_
#define UTILS_HPP_
#include <vector>
#include <string>

#ifndef _FORMAT
#  define _FORMAT(_fmt, ...)                                                   \
    ([&](void) -> std::string {                                                \
      int _sz = std::snprintf(nullptr, 0, _fmt, __VA_ARGS__);                  \
      std::vector<char> _out(_sz + 1);                                         \
      std::snprintf(&_out[0], _out.size(), _fmt, __VA_ARGS__);                 \
      return std::string(_out.data());                                         \
    })()
#endif

#endif
