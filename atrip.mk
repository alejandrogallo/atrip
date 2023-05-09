
if WITH_BUILD_CTF
CTF_COMMIT         = 53ae5daad851bf3b198ebe1fa761c13b12291116
CTF_GIT_REPOSITORY = https://github.com/cc4s/ctf
CTF_BUILD_PATH     = $(top_builddir)/extern/build/ctf/$(CTF_COMMIT)
CTF_SRC_PATH       = $(top_builddir)/extern/src/ctf/$(CTF_COMMIT)
CTF_CPPFLAGS       = -I$(CTF_BUILD_PATH)/include
include $(top_srcdir)/etc/make/ctf.mk
endif
