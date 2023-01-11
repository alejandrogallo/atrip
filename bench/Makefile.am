AUTOMAKE_OPTIONS = subdir-objects
include $(top_srcdir)/atrip.mk

AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir) $(CTF_CPPFLAGS)
AM_LDFLAGS =   @LAPACK_LIBS@ @BLAS_LIBS@

if WITH_BUILD_CTF
ATRIP_CTF = $(CTF_BUILD_PATH)/lib/libctf.a
else
ATRIP_CTF = @LIBCTF_LD_LIBRARY_PATH@/libctf.a
endif
ATRIP_LIB = $(top_builddir)/src/libatrip.a $(ATRIP_CTF)

bin_PROGRAMS =
BENCHES_LDADD = $(ATRIP_LIB) $(ATRIP_CTF)


##
## main entry point and bench
##
bin_PROGRAMS += atrip
atrip_SOURCES = main.cxx
atrip_CPPFLAGS = $(AM_CPPFLAGS)
atrip_LDADD = $(BENCHES_LDADD)

atrip: main.cxx
	$(NVCXX) -cuda     \
		-x cu -I../      \
		$(MPILIBS)       \
		-I$(srcdir)/     \
		$(AM_CPPFLAGS)   \
		$(DEFS)          \
		$(BENCHES_LDADD) \
		$(AM_LDFLAGS)    \
		$< -o $@

endif

if !WITH_CUDA
##
## tuples distribution
##
bin_PROGRAMS += tuples-distribution
tuples_distribution_LDADD  = $(BENCHES_LDADD)
tuples_distribution_SOURCES = tuples-distribution.cxx
endif


if WITH_CUDA
AM_CPPFLAGS += $(CUDA_CXXFLAGS)
BENCHES_LDADD += $(CUDA_LDFLAGS)

AM_CXXFLAGS = $(CUDA_CXXFLAGS)
AM_LDFLAGS += $(CUDA_LDFLAGS)

bin_PROGRAMS += test-cublas-parallel-atrip
test_cublas_parallel_atrip_SOURCES = test-cublas-parallel-atrip.cxx

bin_PROGRAMS += test-cuda-sanity
test_cuda_sanity_SOURCES = test-cuda-sanity.cxx

bin_PROGRAMS += test-cublas-parallel
test_cublas_parallel_SOURCES = test-cublas-parallel.cxx
endif
