AM_LDFLAGS = -fopenmp
AM_CPPFLAGS = -I$(top_srcdir)/include/

bin_PROGRAMS = test_main
test_main_SOURCES = test_main.cxx

test_main_LDADD = \
-lopenblas        \
-lscalapack       \
@LIBCTF_LD_LIBRARY_PATH@/libctf.a \
$(top_builddir)/src/libatrip.a

