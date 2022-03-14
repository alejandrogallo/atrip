CTF_CONFIG_FLAGS =
CTF_STATIC_LIB = $(CTF_BUILD_PATH)/lib/libctf.a
CTF_SHARED_LIB = $(CTF_BUILD_PATH)/lib/libctf.so
CTF_GIT_REPOSITORY ?= https://github.com/cyclops-community/ctf

$(CTF_SRC_PATH)/configure:
	mkdir -p $(@D)
	git clone $(CTF_GIT_REPOSITORY) $(@D)
	cd $(@D) && git checkout $(CTF_COMMIT)

$(CTF_BUILD_PATH)/Makefile: $(CTF_SRC_PATH)/configure
	mkdir -p $(CTF_BUILD_PATH)
	cd $(CTF_BUILD_PATH) && $(abspath $(CTF_SRC_PATH))/configure $(CTF_CONFIG_FLAGS)

$(CTF_STATIC_LIB): $(CTF_BUILD_PATH)/Makefile
	$(info Compiling $@)
	cd $(CTF_BUILD_PATH) && $(MAKE)

.PHONY: ctf ctf-clean
ctf: $(CTF_STATIC_LIB)

ctf-clean:
	rm -rf $(CTF_BUILD_PATH)
