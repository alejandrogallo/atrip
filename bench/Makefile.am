AUTOMAKE_OPTIONS = subdir-objects
include $(top_srcdir)/atrip.mk

AM_CPPFLAGS = -I$(top_srcdir)/include/ $(CTF_CPPFLAGS)
AM_LDFLAGS =   @LAPACK_LIBS@ @BLAS_LIBS@

bin_PROGRAMS = test_main
test_main_SOURCES = test_main.cxx


test_main_LDADD  = \
	$(top_builddir)/src/libatrip.a

if WITH_BUILD_CTF
test_main_LDADD += $(CTF_BUILD_PATH)/lib/libctf.a
else
test_main_LDADD += @LIBCTF_LD_LIBRARY_PATH@/libctf.a
endif

if WITH_CUDA
test_main_CXXFLAGS = $(CUDA_CXXFLAGS)
test_main_LDADD += $(CUDA_LDFLAGS)

AM_CXXFLAGS = $(CUDA_CXXFLAGS)
AM_LDFLAGS += $(CUDA_LDFLAGS)

bin_PROGRAMS += test-cublas-parallel-atrip
test_cublas_parallel_atrip_SOURCES = test-cublas-parallel-atrip.cxx

bin_PROGRAMS += test-cuda-sanity
test_cuda_sanity_SOURCES = test-cuda-sanity.cxx
endif
