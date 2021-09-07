include etc/make/ctf.mk

CXX = mpic++

bench: lib

CXXFLAGS += -I$(CTF_INCLUDE_PATH)
CXXFLAGS += -fPIC
CXXFLAGS += -std=c++11
CXXFLAGS += -pedantic -Wall

LDFLAGS += -fopenmp
LDFLAGS += -Wl,-Bstatic
LDFLAGS += -L$(CTF_BUILD_PATH)/lib -lctf
LDFLAGS += -Wl,-Bdynamic
LDFLAGS += -L$(SCALAPACK_PATH)/lib -lscalapack

bench: CXXFLAGS := $(filter-out -fPIC,$(CXXFLAGS))
bench: LDFLAGS += -Wl,-Bstatic
bench: LDFLAGS += -L$(dir $(ATRIP_STATIC_LIBRARY)) -latrip
bench: LDFLAGS += -Wl,-Bdynamic
bench: LDFLAGS += -L$(OPENBLAS_PATH)/lib -lopenblas
