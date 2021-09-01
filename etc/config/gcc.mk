include etc/make/ctf.mk

CXX = mpic++

CXXFLAGS += -I$(ATRIP_ROOT)/include

CXXFLAGS += -I$(CTF_INCLUDE_PATH)
LDFLAGS += -Wl,-Bstatic -L$(CTF_BUILD_PATH)/lib -lctf
LDFLAGS += -fopenmp -L/usr/lib -lscalapack -L/opt/OpenBLAS/lib -lopenblas
LDFLAGS += -Wl,-Bdynamic
