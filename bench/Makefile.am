AM_LDFLAGS = -fopenmp

bin_PROGRAMS = test_main
test_main_SOURCES = test_main.cxx

test_main_LDADD = \
-lopenblas        \
-lscalapack       \
@LIBCTF_LD_LIBRARY_PATH@/libctf.a \
@abs_top_builddir@/src/libatrip.a

#LDFLAGS += @ATRIP_LDFLAGS@
