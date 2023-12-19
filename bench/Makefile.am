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

bin_PROGRAMS += disk-io
disk_io_SOURCES = disk-io.cxx
disk_io_LDADD = $(BENCHES_LDADD)


if !WITH_CUDA
##
## tuples distribution
##
if WITH_CTF
bin_PROGRAMS += tuples-distribution
tuples_distribution_LDADD  = $(BENCHES_LDADD)
tuples_distribution_SOURCES = tuples-distribution.cxx
endif
endif



################################################################################
## CUDA BENCHES
################################################################################
if WITH_HIP

atrip_LDADD += -L/pfs/lustrep2/projappl/project_462000125/samantao-public/omnitools/rocm-5.2.x/omnitrace/lib -lomnitrace-user


bin_PROGRAMS += test-cuda-sanity
test_cuda_sanity_SOURCES = test-cuda-sanity.cxx
bin_PROGRAMS += test-cublas-parallel
test_cublas_parallel_SOURCES = test-cublas-parallel.cxx

bin_PROGRAMS += test-cublas-parallel-atrip
test_cublas_parallel_atrip_SOURCES = test-cublas-parallel-atrip.cxx
endif

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
