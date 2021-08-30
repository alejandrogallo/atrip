include etc/ctf.mk

CXX = mpic++

CXXFLAGS += -I$(ATRIP_ROOT)/include

CXXFLAGS += -I$(CTF_INCLUDE_PATH)
LDFLAGS += -L$(CTF_BUILD_PATH) -lctf
