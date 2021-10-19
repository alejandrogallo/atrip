ATRIP_ROOT := $(abspath $(PWD))
CXXFLAGS += -I$(ATRIP_ROOT)/include

CTF_COMMIT         = 968f8f9eb6aab1d6b67d2fcc1a70c9fc3b98adfa
CTF_GIT_REPOSITORY = https://github.com/cc4s/ctf
CTF_BUILD_PATH     = $(ATRIP_ROOT)/extern/build/$(CONFIG)/ctf/$(CTF_COMMIT)
CTF_SRC_PATH       = $(ATRIP_ROOT)/extern/src/$(CONFIG)/ctf/$(CTF_COMMIT)
CXXFLAGS          += -I$(CTF_BUILD_PATH)/include
