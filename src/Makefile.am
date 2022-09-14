AUTOMAKE_OPTIONS = subdir-objects
include $(top_srcdir)/atrip.mk

AM_CXXFLAGS = $(CTF_CPPFLAGS) -fmax-errors=1
AM_CPPFLAGS = $(CTF_CPPFLAGS)

lib_LIBRARIES = libatrip.a

libatrip_a_CPPFLAGS = -I$(top_srcdir)/include/
libatrip_a_SOURCES = ./atrip/Blas.cxx
NVCC_FILES = ./atrip/Equations.cxx ./atrip/Complex.cxx ./atrip/Atrip.cxx

if WITH_CUDA
NVCC_OBJS = $(patsubst %.cxx,%.nvcc.o,$(NVCC_FILES))
libatrip_a_CPPFLAGS += $(CUDA_CXXFLAGS)
libatrip_a_DEPENDENCIES = $(NVCC_OBJS)
libatrip_a_LIBADD = $(NVCC_OBJS)
%.nvcc.o: %.cxx
	$(NVCC) -c -x cu -ccbin="${MPICXX}" -I../ $(CPPFLAGS) $(CTF_CPPFLAGS) $(DEFS) $(libatrip_a_CPPFLAGS) $< -o $@

#./atrip/Equations.o: ./atrip/Equations.cxx
#	$(NVCC) -c -I../ $(CPPFLAGS) $(libatrip_a_CPPFLAGS) $< -o $@
else
libatrip_a_SOURCES += $(NVCC_FILES)
endif

