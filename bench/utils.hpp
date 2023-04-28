#ifndef UTILS_HPP_
#define UTILS_HPP_
#include <string>
#include <vector>
#include <functional>

#ifndef _FORMAT
#  define _FORMAT(_fmt, ...)                                                   \
    ([&](void) -> std::string {                                                \
      int _sz = std::snprintf(nullptr, 0, _fmt, __VA_ARGS__);                  \
      std::vector<char> _out(_sz + 1);                                         \
      std::snprintf(&_out[0], _out.size(), _fmt, __VA_ARGS__);                 \
      return std::string(_out.data());                                         \
    })()
#endif

// Printer for options
std::vector<std::function<void(void)>> input_printer;
#define _register_printer(flag, variable)                                      \
  input_printer.push_back([&variable]() {                                      \
    std::cout << "Input " << flag << " " << variable << std::endl;             \
  })

#define defoption(app, flag, variable, description)                            \
  _register_printer(flag, variable);                                           \
  app.add_option(flag, variable, description)

#define defflag(app, flag, variable, description)                              \
  _register_printer(flag, variable);                                           \
  app.add_flag(flag, variable, description)

#endif
