
if WITH_BUILD_CTF
CTF_COMMIT         = 968f8f9eb6aab1d6b67d2fcc1a70c9fc3b98adfa
CTF_GIT_REPOSITORY = https://github.com/cc4s/ctf
CTF_BUILD_PATH     = $(top_builddir)/extern/build/ctf/$(CTF_COMMIT)
CTF_SRC_PATH       = $(top_builddir)/extern/src/ctf/$(CTF_COMMIT)
CTF_CPPFLAGS       = -I$(CTF_BUILD_PATH)/include
include $(top_srcdir)/etc/make/ctf.mk
endif
