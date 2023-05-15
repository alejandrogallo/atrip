AUTOMAKE_OPTIONS = subdir-objects
AM_CPPFLAGS =
AM_CXXFLAGS =
AM_LDFLAGS =

ATRIP_CTF = @LIBCTF_PATH@/lib/libctf.a
ATRIP_LIB = $(top_builddir)/src/libatrip.a

BENCH_CXXFLAGS = -I$(top_srcdir)/src/ \
                 -I$(top_srcdir) \
                  -fmax-errors=1 \
                  -Wextra \
                  -I@LIBCTF_PATH@/include
BENCHES_LDADD = $(ATRIP_LIB) $(ATRIP_CTF) @LAPACK_LIBS@ @BLAS_LIBS@

AM_CXXFLAGS += $(BENCH_CXXFLAGS)

bin_PROGRAMS =


##
## main entry point and bench
##
bin_PROGRAMS += atrip
atrip_SOURCES = main.cxx
atrip_LDADD = $(BENCHES_LDADD)


if !WITH_CUDA
##
## tuples distribution
##
bin_PROGRAMS += tuples-distribution
tuples_distribution_LDADD  = $(BENCHES_LDADD)
tuples_distribution_SOURCES = tuples-distribution.cxx
endif



################################################################################
## CUDA BENCHES
################################################################################

if WITH_CUDA
AM_CPPFLAGS += $(CUDA_CXXFLAGS)
BENCHES_LDADD += $(CUDA_LDFLAGS)

AM_CXXFLAGS += $(CUDA_CXXFLAGS)
AM_LDFLAGS += $(CUDA_LDFLAGS)

bin_PROGRAMS += test-cublas-parallel-atrip
test_cublas_parallel_atrip_SOURCES = test-cublas-parallel-atrip.cxx

bin_PROGRAMS += test-cuda-sanity
test_cuda_sanity_SOURCES = test-cuda-sanity.cxx

bin_PROGRAMS += test-cublas-parallel
test_cublas_parallel_SOURCES = test-cublas-parallel.cxx
endif
