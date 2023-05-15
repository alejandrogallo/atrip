AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -fmax-errors=1 -I@LIBCTF_CPATH@
AM_CPPFLAGS = -I@LIBCTF_CPATH@

lib_LIBRARIES = libatrip.a

libatrip_a_CPPFLAGS =
libatrip_a_SOURCES = ./atrip/Blas.cxx ./atrip/Tuples.cxx ./atrip/RankMap.cxx ./atrip/Slice.cxx
NVCC_FILES = ./atrip/Equations.cxx ./atrip/Complex.cxx ./atrip/Atrip.cxx  ./atrip/DatabaseCommunicator.cxx ./atrip/SliceUnion.cxx

if WITH_CUDA
NVCC_OBJS = $(patsubst %.cxx,%.nvcc.o,$(NVCC_FILES))
libatrip_a_CPPFLAGS += $(CUDA_CXXFLAGS)
libatrip_a_DEPENDENCIES = $(NVCC_OBJS)
libatrip_a_LIBADD = $(NVCC_OBJS)
%.nvcc.o: %.cxx
	$(NVCC) -c -x cu -ccbin="${MPICXX}" -I../ $(CPPFLAGS) -I$(LIBCTF_CPATH) $(DEFS) $(libatrip_a_CPPFLAGS) $< -o $@

#./atrip/Equations.o: ./atrip/Equations.cxx
#	$(NVCC) -c -I../ $(CPPFLAGS) $(libatrip_a_CPPFLAGS) $< -o $@
else
libatrip_a_SOURCES += $(NVCC_FILES)
endif

