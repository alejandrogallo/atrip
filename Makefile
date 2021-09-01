ATRIP_ROOT := $(PWD)
CONFIG ?= gcc
SOURCES_FILE := Sources.mk

include $(SOURCES_FILE)
include ./etc/make/emacs.mk
include ./etc/config/$(CONFIG).mk
include ./bench/config.mk

MAIN = README.org

$(SOURCES_FILE): $(MAIN)
	echo -n "SOURCES = " > $@
	$(EMACS) --eval '(atrip-print-sources)' >> $@

$(SOURCES): $(MAIN)
	$(call tangle,$<)

tangle: $(SOURCES)

clean:
	-rm -v $(SOURCES)

clean-all: bench-clean clean

bench: $(BENCH_TARGETS)

.PHONY: clean tangle bench

%: %.o
	$(CXX) $< $(CXXFLAGS) $(LDFLAGS) -o $@

%.o: %.cxx
	$(CXX) -c $< $(CXXFLAGS) -o $@

