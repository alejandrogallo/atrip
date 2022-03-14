include $(top_srcdir)/atrip.mk

AM_CPPFLAGS = -I$(top_srcdir)/include/ $(CTF_CPPFLAGS)
AM_LDFLAGS = -lgomp

bin_PROGRAMS = test_main
test_main_SOURCES = test_main.cxx


test_main_LDADD  = \
	-lopenblas       \
	-lscalapack      \
	$(top_builddir)/src/libatrip.a

if WITH_BUILD_CTF
test_main_LDADD += $(CTF_BUILD_PATH)/lib/libctf.a
else
test_main_LDADD += @LIBCTF_LD_LIBRARY_PATH@/libctf.a
endif

