include etc/make/ctf.mk

CXX = mpic++

CXXFLAGS += -I$(ATRIP_ROOT)/include

CXXFLAGS += -I$(CTF_INCLUDE_PATH)
CXXFLAGS += -fPIC

LDFLAGS += -fopenmp
LDFLAGS += -Wl,-Bstatic
LDFLAGS += -L$(CTF_BUILD_PATH)/lib -lctf
LDFLAGS += -L$(SCALAPACK_PATH)/lib -lscalapack
LDFLAGS += -Wl,-Bdynamic

bench: CXXFLAGS := $(filter-out -fPIC,$(CXXFLAGS))
bench: LDFLAGS += -Wl,-Bstatic
bench: LDFLAGS += -Llib/ -latrip
bench: LDFLAGS += -L$(OPENBLAS_PATH)/lib -lopenblas
bench: LDFLAGS += -Wl,-Bdynamic
