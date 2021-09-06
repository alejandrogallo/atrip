ATRIP_ROOT := $(PWD)
CONFIG ?= gcc
SOURCES_FILE := Sources.mk

include $(SOURCES_FILE)
include ./etc/make/emacs.mk
include ./etc/config/$(CONFIG).mk
include ./bench/config.mk

MAIN = README.org
OBJ_FILES = $(patsubst %.cxx,%.o,$(filter-out %.hpp,$(SOURCES)))
DEP_FILES = $(patsubst %.o,%.d,$(OBJ_FILES))
SHARED_LIBRARY = lib/libatrip.so
STATIC_LIBRARY = lib/libatrip.a

lib: ctf
lib: $(SHARED_LIBRARY) $(STATIC_LIBRARY)

include $(DEP_FILES)

$(SHARED_LIBRARY): $(OBJ_FILES)
	mkdir -p $(@D)
	$(CXX) -shared $< $(CXXFLAGS) $(LDFLAGS) -o $@

$(STATIC_LIBRARY): $(OBJ_FILES)
	mkdir -p $(@D)
	$(AR) rcs $@ $<

$(SOURCES_FILE): $(MAIN) config.el
	echo -n "SOURCES = " > $@
	$(EMACS) --eval '(atrip-print-sources)' >> $@

print:
	$(info $(filter-out %.hpp,$(SOURCES)))

$(SOURCES): $(MAIN)
	$(call tangle,$<)

tangle: $(SOURCES)

clean-emacs:
	-rm -v $(SOURCES)

clean:
	-rm -v $(OBJ_FILES) $(DEP_FILES)

clean-all: bench-clean clean

bench: $(BENCH_TARGETS)

.PHONY: clean tangle bench

%: %.o
	$(CXX) $< $(CXXFLAGS) $(LDFLAGS) -o $@

%.o: %.cxx
	$(CXX) -c $< $(CXXFLAGS) -o $@

%.d: %.cxx
	$(CXX) -M $< $(CXXFLAGS) -o $@
