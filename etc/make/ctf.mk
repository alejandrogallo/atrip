CTF_CONFIG_FLAGS = \
	CXX="$(CXX) -lmpi" \
	LIB_PATH="$(LDFLAGS)" \
	LIBS="-lmpi" \
	CXXFLAGS="$(CXXFLAGS)" \
	--no-dynamic

CTF_STATIC_LIB = $(CTF_BUILD_PATH)/lib/libctf.a
CTF_SHARED_LIB = $(CTF_BUILD_PATH)/lib/libctf.so
CTF_GIT_REPOSITORY ?= https://github.com/cyclops-community/ctf

$(CTF_SRC_PATH)/configure:
	mkdir -p $(@D)
	git clone $(CTF_GIT_REPOSITORY) $(@D)
	cd $(@D) && git checkout $(CTF_COMMIT)

# Here make sure that ctf does not builld with CUDA support
# since it is broken anyways
# 
# Also we patch the file kernel.h because it mostl
# doesn't work when we try to include ctf in a CUDACC
# compiler code.
$(CTF_BUILD_PATH)/Makefile: $(CTF_SRC_PATH)/configure
	mkdir -p $(CTF_BUILD_PATH)
	cd $(CTF_BUILD_PATH) && \
		$(abspath $(CTF_SRC_PATH))/configure NVCC="" $(CTF_CONFIG_FLAGS)
	sed -i s/CUDACC/ATRIP_NOT_CUDACC/g ${CTF_SRC_PATH}/src/interface/kernel.h


$(CTF_STATIC_LIB): $(CTF_BUILD_PATH)/Makefile
	$(info Compiling $@)
	cd $(CTF_BUILD_PATH) && $(MAKE)

.PHONY: ctf ctf-clean
ctf: $(CTF_STATIC_LIB)

ctf-clean:
	rm -rf $(CTF_BUILD_PATH)
