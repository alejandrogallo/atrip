AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = -fmax-errors=1 -I@LIBCTF_CPATH@
AM_CPPFLAGS = -I@LIBCTF_CPATH@

lib_LIBRARIES = libatrip.a

NO_ACC_FILES = ./atrip/Blas.cxx ./atrip/Tuples.cxx ./atrip/RankMap.cxx ./atrip/Slice.cxx

libatrip_a_CPPFLAGS =
if !WITH_HIP
libatrip_a_SOURCES = $(NO_ACC_FILES)
endif

ACC_FILES = ./atrip/Equations.cxx ./atrip/Complex.cxx ./atrip/Atrip.cxx  ./atrip/DatabaseCommunicator.cxx ./atrip/SliceUnion.cxx

if WITH_CUDA

NVCC_OBJS = $(patsubst %.cxx,%.nvcc.o,$(ACC_FILES))
libatrip_a_CPPFLAGS += $(CUDA_CXXFLAGS)
libatrip_a_DEPENDENCIES = $(NVCC_OBJS)
libatrip_a_LIBADD = $(NVCC_OBJS)
%.nvcc.o: %.cxx
	$(NVCC) -c -x cu -ccbin="${MPICXX}" -I../ $(CPPFLAGS) -I$(LIBCTF_CPATH) $(DEFS) $(libatrip_a_CPPFLAGS) $< -o $@

endif

if WITH_HIP

ACC_FILES += $(NO_ACC_FILES)
HIPCC_OBJS = $(patsubst %.cxx,%.hipcc.o,$(ACC_FILES))
libatrip_a_CPPFLAGS += $(HIP_CXXFLAGS)
libatrip_a_DEPENDENCIES = $(HIPCC_OBJS)
libatrip_a_LIBADD = $(HIPCC_OBJS)
%.hipcc.o: %.cxx
	$(HIPCC) -x hip -craype-verbose -c -I../../../src -I../ $(CPPFLAGS) -I$(LIBCTF_CPATH) $(DEFS) $(libatrip_a_CPPFLAGS) $< -o $@

endif

if !WITH_HIP
if !WITH_CUDA
libatrip_a_SOURCES += $(ACC_FILES)
endif
endif
