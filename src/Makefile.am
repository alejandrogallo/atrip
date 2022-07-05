AUTOMAKE_OPTIONS = subdir-objects
include $(top_srcdir)/atrip.mk

AM_CXXFLAGS = $(CTF_CPPFLAGS) -fmax-errors=1

lib_LIBRARIES = libatrip.a

libatrip_a_CPPFLAGS = -I$(top_srcdir)/include/
libatrip_a_SOURCES = ./atrip/Atrip.cxx \
                     ./atrip/Blas.cxx \
                     ./atrip/Complex.cxx


if WITH_CUDA
libatrip_a_CPPFLAGS += $(CUDA_CXXFLAGS)
libatrip_a_DEPENDENCIES = ./atrip/Equations.o
libatrip_a_LIBADD = ./atrip/Equations.o
./atrip/Equations.o: ./atrip/Equations.cxx
	$(NVCC) -c -I../ $(CPPFLAGS) $(libatrip_a_CPPFLAGS) $< -o $@
else
libatrip_a_SOURCES += ./atrip/Equations.cxx
endif

