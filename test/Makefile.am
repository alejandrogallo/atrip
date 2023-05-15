
AM_CPPFLAGS = -I$(top_srcdir)/src/ -I@LIBCTF_CPATH@
AM_LDFLAGS =   @LAPACK_LIBS@ @BLAS_LIBS@

bin_PROGRAMS = main
main_SOURCES = main.cxx


main_LDADD  = \
	$(top_builddir)/src/libatrip.a

main_LDADD += @LIBCTF_PATH@/lib/libctf.a

if WITH_CUDA
main_CXXFLAGS = $(CUDA_CXXFLAGS)
main_LDADD += $(CUDA_LDFLAGS)
endif
