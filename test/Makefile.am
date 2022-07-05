include $(top_srcdir)/atrip.mk

AM_CPPFLAGS = -I$(top_srcdir)/include/ $(CTF_CPPFLAGS)
AM_LDFLAGS =   @LAPACK_LIBS@ @BLAS_LIBS@

bin_PROGRAMS = main
main_SOURCES = main.cxx


main_LDADD  = \
	$(top_builddir)/src/libatrip.a

if WITH_BUILD_CTF
main_LDADD += $(CTF_BUILD_PATH)/lib/libctf.a
else
main_LDADD += @LIBCTF_LD_LIBRARY_PATH@/libctf.a
endif

if WITH_CUDA
main_CXXFLAGS = $(CUDA_CXXFLAGS)
main_LDADD += $(CUDA_LDFLAGS)
endif
