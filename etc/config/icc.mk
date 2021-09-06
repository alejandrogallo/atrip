include etc/make/ctf.mk

CXX = mpic++

bench: static

CTF_CONFIG_FLAGS = CXX=$(CXX) \
                   CXXFLAGS="-O3" \
                   LIBS="-lmkl" \
                   --no-dynamic

CXXFLAGS += -I$(ATRIP_ROOT)/include

CXXFLAGS += -I$(CTF_INCLUDE_PATH)
CXXFLAGS += -fPIC
CXXFLAGS += -O3

MKL_LIB = -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
LDFLAGS += -qopenmp -mkl
LDFLAGS += -lpthread -std=c++11
LDFLAGS += $(MKL_LIB)
LDFLAGS += -L$(CTF_BUILD_PATH)/lib -lctf

bench: CXXFLAGS := $(filter-out -fPIC,$(CXXFLAGS))
bench: LDFLAGS += -Llib/ -latrip
